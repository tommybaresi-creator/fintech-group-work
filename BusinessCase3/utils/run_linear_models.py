"""
run_linear_models.py
====================
Entry point for the linear model portfolio replication pipeline.

Runs five estimators on a shared rolling walk-forward infrastructure:

    OLS        – unconstrained baseline
    Ridge      – L2 shrinkage
    LASSO      – L1 sparse
    ElasticNet – L1 + L2
    WOLS       – exponentially weighted OLS

All hyperparameters (alpha, l1_ratio, decay factor λ) are selected
via walk-forward cross-validation on the **training set only**.
The test set is touched exclusively for final reporting.

Output
------
Saves ``data/picklefiles/linear_results.pkl`` with key:
    ``best_results``  – {model_name: result_dict}  (all 5 models)
    ``metrics_df``    – pd.DataFrame (metrics per model)
    ``alphas``        – selected hyperparameters per model

Saves diagnostic plots to ``outputs/``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from utils.models import (
    WOLSModel,
    build_elasticnet,
    build_lasso,
    build_ols,
    build_ridge,
    build_wols,
    select_alpha_elasticnet,
    select_alpha_lasso,
    select_alpha_ridge,
    select_lambda_wols,
)
from utils.risk import gross_exposure, lasso_selection_frequency
from utils.rolling_engine import run_rolling

logger = logging.getLogger(__name__)

PICKLE_DIR  = Path("data/picklefiles")
OUTPUTS_DIR = Path("outputs")
ANNUAL_FACTOR = 52


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(stem: str) -> None:
    path = OUTPUTS_DIR / f"{stem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def _plot_weight_dynamics(
    results: Dict[str, Dict],
    save_prefix: str,
) -> None:
    """Plot portfolio weight time series for each model."""
    for name, res in results.items():
        wh = res["weights_history"]
        fig, ax = plt.subplots(figsize=(14, 5))
        for col in wh.columns:
            ax.plot(wh[col], label=col, linewidth=1.2, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Portfolio Weights Over Time — {name}", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (actual, return space)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        _save_fig(f"{save_prefix}_weights_{name.lower().replace(' ', '_')}")
        plt.close(fig)


def _plot_gross_exposure(
    results: Dict[str, Dict],
    save_prefix: str,
) -> None:
    """Plot gross exposure (Σ|w|) over time for all models."""
    fig, ax = plt.subplots(figsize=(14, 5))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    for (name, res), color in zip(results.items(), palette):
        ge = res["weights_history"].abs().sum(axis=1)
        ax.plot(ge, label=name, color=color, linewidth=1.4)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.9, label="100%")
    ax.axhline(2.0, color="red",  linestyle="--", linewidth=0.9, label="200% UCITS")
    ax.set_title("Gross Exposure Over Time — All Linear Models", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Gross Exposure (Σ|w|)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save_fig(f"{save_prefix}_gross_exposure")
    plt.close(fig)


def _plot_lasso_selection(
    results: Dict[str, Dict],
    save_prefix: str,
) -> None:
    """Bar chart of LASSO selection frequency if the model is present."""
    if "LASSO" not in results:
        return
    freq = lasso_selection_frequency(results["LASSO"]["weights_history"]) * 100
    freq = freq.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    freq.plot(kind="barh", color="steelblue", ax=ax)
    ax.axvline(50, color="red", linestyle="--", linewidth=0.9, label="50 %")
    ax.set_title("LASSO — Futures Selection Frequency (test period)", fontsize=13)
    ax.set_xlabel("Frequency (%)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_fig(f"{save_prefix}_lasso_selection_freq")
    plt.close(fig)


def _plot_active_returns(
    results: Dict[str, Dict],
    save_prefix: str,
) -> None:
    """
    Cumulative active return (replica − target) for all models.

    The ideal replication line is the horizontal zero axis.
    Any systematic drift (positive or negative) indicates the model
    is running a directional bet rather than tracking.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    for (name, res), color in zip(results.items(), palette):
        r, t = res["replica_returns"], res["target_returns"]
        common = r.index.intersection(t.index)
        active = (r.loc[common] - t.loc[common]).cumsum()
        ax.plot(active, label=name, color=color, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative Active Return (replica − target)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative active return")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save_fig(f"{save_prefix}_active_returns")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_linear_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_end_date: str = "2018-12-31",
    window: int = 52,
    models_to_run: Optional[List[str]] = None,
    save_prefix: str = "linear",
) -> Dict[str, Any]:
    """
    Full linear model pipeline: tune → roll → evaluate → save.

    Parameters
    ----------
    X : pd.DataFrame, shape (T, n_futures)
        Futures return feature matrix.
    y : pd.Series, length T
        Monster index weekly returns (target).
    train_end_date : str
        ISO date string for the training / test split.
        All observations up to and including this date are training;
        subsequent observations are test.  Default ``'2018-12-31'``.
    window : int
        Rolling regression window in weeks.  Default 52.
    models_to_run : list of str, optional
        Subset of models to run.  Defaults to all five:
        ``['OLS', 'Ridge', 'LASSO', 'ElasticNet', 'WOLS']``.
    save_prefix : str
        Filename prefix for all saved figures.  Default ``'linear'``.

    Returns
    -------
    dict
        ``best_results``
            dict: model_name → result_dict.  Each result_dict contains
            ``replica_returns``, ``target_returns``, ``weights_history``.
        ``metrics_df``
            pd.DataFrame with tracking metrics per model.
        ``alphas``
            dict: model_name → selected hyperparameter(s).

    Notes
    -----
    The rolling loop starts at the first observation after the training
    period ends *minus* one window, so training-period in-sample
    predictions are also available.  The ``best_results`` dict covers
    the **full** series (train + test); downstream evaluation should
    restrict to the test period as needed.

    For the final comparison with the NN, restrict results to the test
    period before calling ``run_evaluation``:

    .. code-block:: python

        test_results = {
            name: {
                k: v.loc[v.index > '2018-12-31'] if isinstance(v, (pd.Series, pd.DataFrame)) else v
                for k, v in res.items()
            }
            for name, res in linear_data['best_results'].items()
        }
    """
    _ensure_dirs()
    sns.set_theme(style="whitegrid")

    logger.info("=" * 60)
    logger.info("LINEAR MODELS — START")
    logger.info("=" * 60)

    if models_to_run is None:
        models_to_run = ["OLS", "Ridge", "LASSO", "ElasticNet", "WOLS"]

    # ── Align indices ────────────────────────────────────────────────────────
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    feat = X.columns.tolist()

    # ── Train / test split ───────────────────────────────────────────────────
    cutoff = pd.Timestamp(train_end_date)
    train_mask = X.index <= cutoff

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    n_train = train_mask.sum()
    n_test  = (~train_mask).sum()
    logger.info(
        "Train: %d obs (%s → %s) | Test: %d obs (%s → %s) | window=%d",
        n_train,
        X.index[0].date(),
        cutoff.date(),
        n_test,
        X.index[~train_mask][0].date() if n_test else "N/A",
        X.index[-1].date(),
        window,
    )

    # ── Fit scaler on training data ONLY ────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    logger.info("Scaler fit on %d training observations", n_train)

    X_train_sc = scaler.transform(X_train.values)
    X_train_raw = X_train.values

    # ── Hyperparameter selection (training data only) ────────────────────────
    logger.info("-" * 50)
    logger.info("Hyperparameter selection (training data only) …")
    alphas: Dict[str, Any] = {}

    if "Ridge" in models_to_run:
        alphas["Ridge"] = select_alpha_ridge(X_train_sc, y_train)

    if "LASSO" in models_to_run:
        alphas["LASSO"] = select_alpha_lasso(X_train_sc, y_train)

    if "ElasticNet" in models_to_run:
        alpha_en, l1_en = select_alpha_elasticnet(X_train_sc, y_train)
        alphas["ElasticNet"] = {"alpha": alpha_en, "l1_ratio": l1_en}

    if "WOLS" in models_to_run:
        alphas["WOLS"] = select_lambda_wols(
            X_train_sc, y_train, scaler.scale_, X_train_raw, window
        )

    logger.info("Selected hyperparameters: %s", alphas)

    # ── Build model factory functions ────────────────────────────────────────
    def _make_fn(model_type: str):
        """Return a function (X_win, Y_win) → fitted model."""
        if model_type == "OLS":
            def fn(Xw, Yw):
                m = build_ols()
                m.fit(Xw, Yw)
                return m
        elif model_type == "Ridge":
            def fn(Xw, Yw):
                m = build_ridge(alphas["Ridge"])
                m.fit(Xw, Yw)
                return m
        elif model_type == "LASSO":
            def fn(Xw, Yw):
                m = build_lasso(alphas["LASSO"])
                m.fit(Xw, Yw)
                return m
        elif model_type == "ElasticNet":
            def fn(Xw, Yw):
                m = build_elasticnet(
                    alphas["ElasticNet"]["alpha"],
                    alphas["ElasticNet"]["l1_ratio"],
                )
                m.fit(Xw, Yw)
                return m
        elif model_type == "WOLS":
            def fn(Xw, Yw):
                m = build_wols(alphas["WOLS"])
                m.fit(Xw, Yw)
                return m
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return fn

    # ── Rolling loop — full series (train + test) ─────────────────────────
    best_results: Dict[str, Dict] = {}

    for name in models_to_run:
        logger.info("-" * 50)
        logger.info("Running %s …", name)

        res = run_rolling(
            X_raw          = X,
            Y              = y,
            scaler         = scaler,
            window         = window,
            build_model_fn = _make_fn(name),
            feature_names  = feat,
        )
        best_results[name] = res

    # ── Restrict to test period for metrics ──────────────────────────────────
    test_start = X.index[train_mask][-1] + pd.Timedelta(days=1)
    test_results = {
        name: {
            "replica_returns":  res["replica_returns"].loc[test_start:],
            "target_returns":   res["target_returns"].loc[test_start:],
            "weights_history":  res["weights_history"].loc[test_start:],
        }
        for name, res in best_results.items()
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    try:
        from utils.evaluation import build_metrics_table
    except ImportError:
        from evaluation import build_metrics_table

    metrics_df = build_metrics_table(test_results)

    logger.info("=" * 60)
    logger.info("RESULTS (test period)")
    logger.info("=" * 60)
    logger.info(
        "\n%s",
        metrics_df[[
            "ann_ret_replica", "ann_ret_target",
            "tracking_error", "information_ratio",
            "correlation", "sharpe_replica",
        ]].round(4).to_string()
    )

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    _plot_weight_dynamics(test_results, save_prefix)
    _plot_gross_exposure(test_results, save_prefix)
    _plot_lasso_selection(test_results, save_prefix)
    _plot_active_returns(test_results, save_prefix)

    # ── Save pickle ───────────────────────────────────────────────────────────
    payload = {
        "best_results":  test_results,   # test period only (matches NN format)
        "full_results":  best_results,   # full series including training
        "metrics_df":    metrics_df,
        "alphas":        alphas,
        "window":        window,
        "train_end_date": train_end_date,
        "scaler":        scaler,
        "feature_names": feat,
    }
    pkl_path = PICKLE_DIR / "linear_results.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("LINEAR MODELS — DONE")
    logger.info("=" * 60)
    return payload


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import setup_logging
    setup_logging()

    pkl_data = Path("data/picklefiles/data_loader.pkl")
    if not pkl_data.exists():
        logger.error("data_loader.pkl not found — run run_data_loader() first.")
        sys.exit(1)

    with open(pkl_data, "rb") as fh:
        data = pickle.load(fh)

    run_linear_models(data["X"], data["y"])