"""
Performance evaluation and comparison utilities for the Portfolio Replication project.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

PICKLE_DIR = Path("data/picklefiles")
OUTPUTS_DIR = Path("outputs")
ANNUAL_FACTOR = 52  # weekly data → annualise by ×52 (returns) or ×√52 (vol)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(stem: str) -> None:
    path = OUTPUTS_DIR / f"{stem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


def _align(replica: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return both series restricted to their common index."""
    idx = replica.index.intersection(target.index)
    return replica.loc[idx], target.loc[idx]


# ── Individual metrics ────────────────────────────────────────────────────────


def annualized_return(returns: pd.Series, freq: int = ANNUAL_FACTOR) -> float:
    """
    Compute the annualised mean return.

    Parameters
    ----------
    returns : pd.Series
        Periodic return series.
    freq : int
        Periods per year (52 for weekly).

    Returns
    -------
    float
        Annualised mean return.
    """
    return float(returns.mean() * freq)


def annualized_volatility(returns: pd.Series, freq: int = ANNUAL_FACTOR) -> float:
    """
    Compute annualised volatility (standard deviation).

    Parameters
    ----------
    returns : pd.Series
    freq : int

    Returns
    -------
    float
    """
    return float(returns.std() * np.sqrt(freq))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    freq: int = ANNUAL_FACTOR,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
    risk_free : float
        Annualised risk-free rate. Default 0.0.
    freq : int

    Returns
    -------
    float
        Sharpe ratio, or ``nan`` if volatility is zero.
    """
    vol = annualized_volatility(returns, freq)
    if vol == 0.0:
        return float("nan")
    return (annualized_return(returns, freq) - risk_free) / vol


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute the maximum drawdown from a return series.

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.45 = 45 % loss).
    """
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    return float(abs(dd.min()))


def tracking_error(
    replica: pd.Series,
    target: pd.Series,
    freq: int = ANNUAL_FACTOR,
) -> float:
    """
    Compute annualised Tracking Error Volatility (TEV).

    TEV = std(replica_return − target_return) × √freq

    Parameters
    ----------
    replica : pd.Series
    target : pd.Series
    freq : int

    Returns
    -------
    float
    """
    r, t = _align(replica, target)
    return float((r - t).std() * np.sqrt(freq))


def information_ratio(
    replica: pd.Series,
    target: pd.Series,
    freq: int = ANNUAL_FACTOR,
) -> float:
    """
    Compute the Information Ratio.

    IR = annualised active return / tracking error

    Parameters
    ----------
    replica : pd.Series
    target : pd.Series
    freq : int

    Returns
    -------
    float
        Information ratio, or ``nan`` if tracking error is zero.
    """
    r, t = _align(replica, target)
    active = annualized_return(r - t, freq)
    te = tracking_error(r, t, freq)
    if te == 0.0:
        return float("nan")
    return active / te


def calculate_var(returns, confidence=0.01, horizon=4, method='conservative'):
    """
    Calculate Value at Risk (VaR) using non-Gaussian methods.

    Parameters:
    returns (array-like): Historical returns (single-period)
    confidence (float): Confidence level (e.g., 0.01 for 1% VaR)
    horizon (int): Time horizon in periods (weeks)
    method (str): 'historical', 'cornish_fisher', or 'conservative' (max of both)

    Returns:
    float: VaR as a positive number (loss)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        # Fallback: not enough data, use simple std-based estimate
        sigma = np.std(returns)
        return sigma * np.sqrt(horizon) * 2.33  # around 1% Gaussian as fallback

    # --- Historical VaR ---
    # Scale single-period returns to the desired horizon
    # Using overlapping blocks if possible, otherwise sqrt scaling on the quantile
    if len(returns) >= horizon:
        # Compute rolling multi-period returns for the given horizon
        cumulative = np.array([
            np.prod(1 + returns[j:j+horizon]) - 1
            for j in range(len(returns) - horizon + 1)
        ])
        var_historical = -np.percentile(cumulative, confidence * 100)
    else:
        # Not enough data for horizon-scaled returns, use sqrt-time scaling
        var_historical = -np.percentile(returns, confidence * 100) * np.sqrt(horizon)

    if method == 'historical':
        return max(var_historical, 0.0)

    # --- Cornish-Fisher VaR ---
    sigma = np.std(returns)
    s = stats.skew(returns) if len(returns) > 2 else 0.0  # skewness
    k = stats.kurtosis(returns, fisher=True) if len(returns) > 3 else 0.0  # excess kurtosis

    z = stats.norm.ppf(confidence)  # negative value for left tail

    # Cornish-Fisher expansion: adjusts z for skewness and kurtosis
    z_cf = (z
            + (z**2 - 1) * s / 6
            + (z**3 - 3*z) * k / 24
            - (2*z**3 - 5*z) * s**2 / 36)

    var_cf = -z_cf * sigma * np.sqrt(horizon)

    if method == 'cornish_fisher':
        return max(var_cf, 0.0)

    # --- Conservative: take the maximum of both ---
    return max(var_historical, var_cf, 0.0)


def compute_metrics(
    replica: pd.Series,
    target: pd.Series,
    model_name: str = "Model",
) -> Dict:
    """
    Compute the full set of performance and replication quality metrics.

    Parameters
    ----------
    replica : pd.Series
        Replica portfolio weekly returns.
    target : pd.Series
        Target index weekly returns.
    model_name : str
        Label used in the output dictionary.

    Returns
    -------
    dict
        Keys: ``model``, ``ann_ret_replica``, ``ann_ret_target``,
        ``ann_vol_replica``, ``ann_vol_target``, ``sharpe_replica``,
        ``sharpe_target``, ``tracking_error``, ``information_ratio``,
        ``correlation``, ``max_dd_replica``, ``max_dd_target``,
        ``n_obs``.
    """
    r, t = _align(replica, target)
    return {
        "model": model_name,
        "ann_ret_replica": annualized_return(r),
        "ann_ret_target": annualized_return(t),
        "ann_vol_replica": annualized_volatility(r),
        "ann_vol_target": annualized_volatility(t),
        "sharpe_replica": sharpe_ratio(r),
        "sharpe_target": sharpe_ratio(t),
        "tracking_error": tracking_error(r, t),
        "information_ratio": information_ratio(r, t),
        "correlation": float(r.corr(t)),
        "max_dd_replica": max_drawdown(r),
        "max_dd_target": max_drawdown(t),
        "n_obs": len(r),
    }


def build_metrics_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a summary metrics DataFrame for all models.

    Parameters
    ----------
    results : dict
        Keys are model names; values are dicts containing at minimum
        ``replica_returns`` (pd.Series) and ``target_returns`` (pd.Series).

    Returns
    -------
    pd.DataFrame
        One row per model with all metrics from :func:`compute_metrics`.
    """
    rows = [
        compute_metrics(v["replica_returns"], v["target_returns"], model_name=k)
        for k, v in results.items()
    ]
    return pd.DataFrame(rows).set_index("model")


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_cumulative_returns(
    results: Dict[str, Dict],
    save_name: str = "eval_01_cumulative_returns",
) -> None:
    """
    Overlay cumulative returns for all models and the target.

    Parameters
    ----------
    results : dict
        Keys are model names; values must contain ``replica_returns``
        and ``target_returns``.
    save_name : str
        Output filename stem (no extension).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Target plotted once (use first entry)
    target = next(iter(results.values()))["target_returns"]
    ax.plot(
        (1 + target).cumprod(),
        label="Target",
        color="black",
        linewidth=2.5,
        linestyle="--",
    )

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    for (name, res), color in zip(results.items(), palette):
        ax.plot(
            (1 + res["replica_returns"]).cumprod(),
            label=name,
            color=color,
            linewidth=1.8,
        )

    ax.set_title("Cumulative Returns — All Models vs Target", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (start = 1)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_tracking_metrics(
    metrics_df: pd.DataFrame,
    save_name: str = "eval_02_tracking_metrics",
) -> None:
    """
    Side-by-side bar charts for Tracking Error and Information Ratio.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`build_metrics_table`.
    save_name : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Tracking error
    ax = axes[0]
    ax.bar(
        metrics_df.index,
        metrics_df["tracking_error"] * 100,
        color="steelblue",
        alpha=0.85,
    )
    ax.set_title("Tracking Error — annualised (%)", fontsize=13)
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # Information ratio
    ax = axes[1]
    colors = [
        "seagreen" if v >= 0 else "tomato" for v in metrics_df["information_ratio"]
    ]
    ax.bar(metrics_df.index, metrics_df["information_ratio"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Information Ratio", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_drawdowns(
    results: Dict[str, Dict],
    save_name: str = "eval_03_drawdowns",
) -> None:
    """
    Overlay drawdown series for all models and the target.

    Parameters
    ----------
    results : dict
    save_name : str
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    target = next(iter(results.values()))["target_returns"]
    cum_t = (1 + target).cumprod()
    ax.fill_between(
        cum_t.index,
        -(1 - cum_t / cum_t.cummax()),
        0,
        alpha=0.15,
        color="black",
        label="Target",
    )
    ax.plot(-(1 - cum_t / cum_t.cummax()), color="black", linewidth=2, linestyle="--")

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    for (name, res), color in zip(results.items(), palette):
        cum = (1 + res["replica_returns"]).cumprod()
        ax.plot(-(1 - cum / cum.cummax()), label=name, color=color, linewidth=1.5)

    ax.set_title("Drawdowns — All Models vs Target", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_scatter_returns(
    results: Dict[str, Dict],
    save_name: str = "eval_04_scatter_returns",
) -> None:
    """
    Scatter plots of weekly replica vs target returns for each model.

    Parameters
    ----------
    results : dict
    save_name : str
    """
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (name, res) in enumerate(results.items()):
        ax = axes_flat[idx]
        r, t = _align(res["replica_returns"], res["target_returns"])
        lim = max(r.abs().max(), t.abs().max()) * 1.1
        ax.scatter(t, r, alpha=0.4, s=14, color="steelblue")
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{name}\ncorr = {r.corr(t):.3f}", fontsize=11)
        ax.set_xlabel("Target return")
        ax.set_ylabel("Replica return")
        ax.grid(alpha=0.3)

    for idx in range(len(results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Weekly Returns: Replica vs Target (out-of-sample)", fontsize=14)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_rolling_correlation(
    results: Dict[str, Dict],
    window: int = 26,
    save_name: str = "eval_05_rolling_correlation",
) -> None:
    """
    Rolling correlation between replica and target for all models.

    Parameters
    ----------
    results : dict
    window : int
        Rolling window in weeks (default 26 ≈ 6 months).
    save_name : str
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    for (name, res), color in zip(results.items(), palette):
        r, t = _align(res["replica_returns"], res["target_returns"])
        roll_corr = r.rolling(window).corr(t)
        ax.plot(roll_corr, label=name, color=color, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color="black", linewidth=0.4, linestyle=":")
    ax.set_title(f"Rolling {window}-Week Correlation (Replica vs Target)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    save_name: str = "eval_06_metrics_heatmap",
) -> None:
    """
    Heatmap of key metrics normalised for easy visual comparison.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`build_metrics_table`.
    save_name : str
    """
    cols = [
        "ann_ret_replica",
        "ann_vol_replica",
        "sharpe_replica",
        "tracking_error",
        "information_ratio",
        "correlation",
        "max_dd_replica",
    ]
    display_names = {
        "ann_ret_replica": "Ann. Return",
        "ann_vol_replica": "Ann. Volatility",
        "sharpe_replica": "Sharpe",
        "tracking_error": "Tracking Error",
        "information_ratio": "Info. Ratio",
        "correlation": "Correlation",
        "max_dd_replica": "Max Drawdown",
    }
    subset = metrics_df[cols].rename(columns=display_names)

    # Z-score normalise each column for visual comparison
    normalised = (subset - subset.mean()) / (subset.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(4, len(metrics_df) * 0.8 + 2)))
    sns.heatmap(
        normalised,
        annot=subset.round(3),
        fmt="g",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Z-score (normalised across models)"},
    )
    ax.set_title("Model Comparison — Key Metrics (colour = z-score)", fontsize=13)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────


def run_evaluation(
    results: Dict[str, Dict],
    save_prefix: str = "eval",
) -> pd.DataFrame:
    """
    Full evaluation pipeline: compute all metrics and generate comparison plots.

    Saves ``data/picklefiles/evaluation.pkl`` and six ``.png`` figures
    to ``outputs/``.

    Parameters
    ----------
    results : dict
        Mapping of model name → result dict. Each result dict must contain:

        ``replica_returns``
            pd.Series of weekly replica portfolio returns.
        ``target_returns``
            pd.Series of weekly target index returns.

    save_prefix : str
        Prefix prepended to all output filenames.

    Returns
    -------
    pd.DataFrame
        Metrics table with one row per model and columns for all metrics.
    """
    _ensure_dirs()
    logger.info("=" * 60)
    logger.info("EVALUATION — START  (%d models)", len(results))
    logger.info("=" * 60)

    sns.set_theme(style="whitegrid")

    # Build metrics table
    metrics_df = build_metrics_table(results)

    # Log summary
    summary_cols = [
        "tracking_error",
        "information_ratio",
        "correlation",
        "sharpe_replica",
    ]
    logger.info("\n%s", metrics_df[summary_cols].round(4).to_string())

    # Generate all plots
    plot_cumulative_returns(results, save_name=f"{save_prefix}_01_cumulative_returns")
    plot_tracking_metrics(metrics_df, save_name=f"{save_prefix}_02_tracking_metrics")
    plot_drawdowns(results, save_name=f"{save_prefix}_03_drawdowns")
    plot_scatter_returns(results, save_name=f"{save_prefix}_04_scatter_returns")
    plot_rolling_correlation(results, save_name=f"{save_prefix}_05_rolling_correlation")
    plot_metrics_heatmap(metrics_df, save_name=f"{save_prefix}_06_metrics_heatmap")

    # Save pickle
    payload = {"metrics": metrics_df, "results": results}
    pkl_path = PICKLE_DIR / "evaluation.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("EVALUATION — DONE")
    logger.info("=" * 60)
    return metrics_df


def run_evaluation_with_costs(
    results: Dict[str, Dict],
    cost_bps: float = 5.0,
    save_prefix: str = "eval",
) -> pd.DataFrame:
    """
    Evaluation pipeline that appends net-of-cost metrics alongside gross.

    Runs the full :func:`run_evaluation` pipeline, then adds net-of-cost
    columns (``net_ann_ret``, ``net_sharpe``, ``net_tracking_error``,
    ``net_information_ratio``) for models that supply ``weights_history``.

    Parameters
    ----------
    results : dict
        Same format as :func:`run_evaluation`.  Models that include
        ``weights_history`` receive net-of-cost columns; others get NaN.
    cost_bps : float
        One-way transaction cost in basis points. Default 5.
    save_prefix : str
        Filename prefix for all saved figures.

    Returns
    -------
    pd.DataFrame
        Combined gross + net metrics table.
    """
    try:
        from utils.transaction_costs import apply_transaction_costs
    except ImportError:
        from transaction_costs import apply_transaction_costs

    # Run standard gross evaluation first
    metrics = run_evaluation(results, save_prefix=save_prefix)

    # Append net columns
    net_rows: Dict[str, Dict] = {}
    for name, res in results.items():
        if "weights_history" not in res:
            logger.warning("'%s' has no weights_history — net metrics = NaN", name)
            net_rows[name] = {
                "net_ann_ret": float("nan"),
                "net_sharpe": float("nan"),
                "net_te": float("nan"),
                "net_ir": float("nan"),
            }
            continue

        net = apply_transaction_costs(
            res["replica_returns"], res["weights_history"], cost_bps=cost_bps
        )
        tgt = res["target_returns"]
        common = net.index.intersection(tgt.index)
        net, tgt_a = net.loc[common], tgt.loc[common]

        active = net - tgt_a
        vol = float(net.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
        ann_r = float(net.mean() * ANNUAL_FACTOR)
        te = float(active.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
        ir = (float(active.mean() * ANNUAL_FACTOR) / te) if te > 0 else float("nan")

        net_rows[name] = {
            "net_ann_ret": ann_r,
            "net_sharpe": (ann_r / vol) if vol > 0 else float("nan"),
            "net_te": te,
            "net_ir": ir,
        }
        logger.info(
            "%-30s | net IR=%.4f | net TE=%.4f | cost=%.0f bps",
            name,
            ir,
            te,
            cost_bps,
        )

    net_df = pd.DataFrame(net_rows).T
    net_df.index.name = "model"
    combined = metrics.join(net_df)

    # Save updated pickle
    pkl_path = PICKLE_DIR / "evaluation_with_costs.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"metrics": combined, "results": results, "cost_bps": cost_bps}, fh)
    logger.info("Pickle saved → %s", pkl_path)

    return combined


def compute_regime_metrics(
    results: Dict[str, Dict],
    regime_splits: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute core metrics separately for each market regime.

    Default splits:
        pre_covid:  test start → 2020-02-21
        post_covid: 2020-02-24 → test end

    Returns a MultiIndex DataFrame: (model, regime) × metrics.
    """
    if regime_splits is None:
        regime_splits = {
            "pre_covid":  ("1900-01-01", "2020-02-21"),
            "post_covid": ("2020-02-24", "2099-12-31"),
        }

    rows = []
    for name, res in results.items():
        rep = res["replica_returns"]
        tgt = res["target_returns"]

        for regime, (start, end) in regime_splits.items():
            r = rep.loc[start:end]
            t = tgt.loc[start:end]

            if len(r) < 10:
                continue

            m = compute_metrics(r, t, model_name=name)
            m["regime"] = regime
            rows.append(m)

    df = pd.DataFrame(rows)
    df = df.set_index(["model", "regime"])
    return df


def plot_regime_analysis(
    regime_df: pd.DataFrame,
    save_prefix: str = "regime",
) -> None:
    """
    Generate two regime comparison figures and save them.

    Figure 1 — Four-panel bar chart: TE, IR, Correlation, Sharpe for
    pre-COVID vs post-COVID side by side per model.

    Figure 2 — Two-panel delta chart: ΔTE and ΔIR (post − pre) per model,
    highlighting which models degrade most under regime change.

    Parameters
    ----------
    regime_df : pd.DataFrame
        MultiIndex (model, regime) × metrics, as returned by
        :func:`compute_regime_metrics`.
    save_prefix : str
        Filename prefix for the two saved figures.
    """
    _ensure_dirs()
    models = regime_df.index.get_level_values("model").unique()
    x = np.arange(len(models))
    width = 0.35

    metrics_to_plot = [
        ("tracking_error",    "Tracking Error (annualised)",  False),
        ("information_ratio", "Information Ratio",             True),
        ("correlation",       "Correlation with target",       False),
        ("sharpe_replica",    "Sharpe Ratio",                  True),
    ]

    # ── Figure 1: pre vs post side-by-side ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Regime Analysis — Pre vs Post COVID", fontsize=15, fontweight="bold")

    for ax, (metric, label, add_hline) in zip(axes.flatten(), metrics_to_plot):
        pre  = regime_df.xs("pre_covid",  level="regime")[metric].reindex(models)
        post = regime_df.xs("post_covid", level="regime")[metric].reindex(models)

        ax.bar(x - width / 2, pre,  width, label="Pre-COVID",  color="steelblue", alpha=0.85)
        ax.bar(x + width / 2, post, width, label="Post-COVID", color="coral",     alpha=0.85)

        if add_hline:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_fig(f"{save_prefix}_01_pre_post")
    plt.close(fig)

    # ── Figure 2: delta (post − pre) ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        "Regime Shift Impact — Post minus Pre COVID", fontsize=13, fontweight="bold"
    )

    for ax, (metric, label) in zip(axes2, [
        ("tracking_error",    "ΔTracking Error (post − pre)"),
        ("information_ratio", "ΔInformation Ratio (post − pre)"),
    ]):
        pre   = regime_df.xs("pre_covid",  level="regime")[metric].reindex(models)
        post  = regime_df.xs("post_covid", level="regime")[metric].reindex(models)
        delta = post - pre

        colors = ["seagreen" if v >= 0 else "tomato" for v in delta]
        ax.bar(x, delta, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_fig(f"{save_prefix}_02_delta")
    plt.close(fig2)


def run_regime_analysis(
    results: Dict[str, Dict],
    regime_splits: Optional[Dict] = None,
    save_prefix: str = "regime",
) -> pd.DataFrame:
    """
    Full regime analysis pipeline: compute per-regime metrics, generate plots,
    save pickle.

    Parameters
    ----------
    results : dict
        Same format as :func:`run_evaluation`.
    regime_splits : dict, optional
        Keys are regime labels; values are ``(start, end)`` ISO date tuples.
        Defaults to pre/post COVID split at 2020-02-21 / 2020-02-24.
    save_prefix : str
        Filename prefix for all saved figures.  Default ``'regime'``.

    Returns
    -------
    pd.DataFrame
        MultiIndex (model, regime) × metrics DataFrame.
    """
    _ensure_dirs()
    sns.set_theme(style="whitegrid")

    logger.info("=" * 60)
    logger.info("REGIME ANALYSIS — START  (%d models)", len(results))
    logger.info("=" * 60)

    regime_df = compute_regime_metrics(results, regime_splits)

    for metric, label in [
        ("tracking_error",    "Tracking Error"),
        ("information_ratio", "Information Ratio"),
        ("correlation",       "Correlation"),
        ("sharpe_replica",    "Sharpe Ratio"),
    ]:
        pivot = regime_df[metric].unstack(level="regime").round(4)
        logger.info("── %s ──\n%s", label, pivot.to_string())

    plot_regime_analysis(regime_df, save_prefix=save_prefix)

    pkl_path = PICKLE_DIR / "regime_analysis.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"regime_df": regime_df, "results": results}, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("REGIME ANALYSIS — DONE")
    logger.info("=" * 60)
    return regime_df


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke-test: load existing pickle and re-run evaluation."""
    import sys

    sys.path.insert(0, ".")
    from utils import setup_logging

    setup_logging()

    pkl_path = Path("data/picklefiles/evaluation.pkl")
    if not pkl_path.exists():
        logger.error(
            "No evaluation pickle found at %s. Run an experiment first.", pkl_path
        )
        sys.exit(1)

    with open(pkl_path, "rb") as fh:
        saved = pickle.load(fh)
    run_evaluation(saved["results"])
