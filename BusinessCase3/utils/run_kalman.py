"""
run_kalman.py
=============
Kalman filter portfolio weight estimator.

State-space model
-----------------
The portfolio weight vector w_t ∈ ℝⁿ follows a random walk
(state transition equation) and is observed noisily via futures returns
(measurement equation):

    State:       w_t  = w_{t-1}  + η_t,    η_t ~ N(0, Q)     [Rw]
    Observation: y_t  = X_t ᵀ w_t  + ε_t,  ε_t ~ N(0, R)     [R1]

where
    n     = number of futures (state dimension)
    X_t   = (n,) futures return vector at time t
    y_t   = scalar target index return at time t

The state transition matrix F = I (identity) encodes the assumption that
weights evolve continuously without drift.  Q controls how quickly the
filter allows weights to change; R controls how much noise there is in
the target return given the current weights.

Prediction (prior)
    ŵ_{t|t-1}  = ŵ_{t-1|t-1}                  (random walk)
    P_{t|t-1}  = P_{t-1|t-1} + Q

Update (posterior)
    H_t        = X_t ᵀ                         (1×n observation matrix)
    ν_t        = y_t  - H_t  ŵ_{t|t-1}        (innovation, scalar)
    S_t        = H_t P_{t|t-1} H_t ᵀ + R       (innovation variance, scalar)
    K_t        = P_{t|t-1} H_t ᵀ / S_t        (n×1 Kalman gain)
    ŵ_{t|t}    = ŵ_{t|t-1} + K_t ν_t
    P_{t|t}    = (I - K_t H_t) P_{t-1|t-1} (I - K_t H_t)ᵀ + K_t R K_t ᵀ

The Joseph form of the covariance update (last line) is used instead of
the simpler P = (I - KH)P to preserve positive semi-definiteness
numerically, which matters over long rolling periods.

One-step-ahead prediction
    Replica return at t: ŷ_{t|t-1} = X_t ᵀ ŵ_{t|t-1}
    This uses the PRIOR weight (before assimilating y_t), ensuring no
    look-ahead: the weight held from t-1 to t was decided without
    knowing y_t.

Hyperparameter estimation (Q, R)
    Q = q·I and R are scalar multiples of the identity / a scalar.
    Estimated from the training period:

        R  = var(OLS residuals on training data)
           = var(y_train - X_train @ β_OLS)

        q  = argmin_{q} mean squared one-step innovation on training
             data (grid search, range [R/1000, R/1]).  Minimising mean
             squared innovation is equivalent to maximising the
             likelihood of the Kalman filter predictions on training
             data, and is a standard approach.

    This is entirely contained within the training set.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.risk import gross_exposure as compute_gross_exp

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


# ── Q/R estimation ────────────────────────────────────────────────────────────

def _estimate_R(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """
    Estimate observation noise variance R from OLS residuals on training data.

    R = var(y_train - X_train @ β_OLS)

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    y_train : np.ndarray, shape (n_train,)

    Returns
    -------
    float
        Estimated R (positive scalar).
    """
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_train, y_train)
    residuals = y_train - ols.predict(X_train)
    R = float(np.var(residuals, ddof=1))
    logger.info("Estimated R (observation noise) = %.6f", R)
    return R


def _select_q_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    R: float,
    q_grid: Optional[List[float]] = None,
) -> float:
    """
    Select process noise scalar q via grid search on training data.

    For each candidate q, run the Kalman filter over the training period
    and compute the mean squared one-step-ahead innovation.  The q that
    minimises this is the maximum-likelihood estimate (training data only).

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    R : float
        Observation noise variance.
    q_grid : list of float, optional
        Candidate q values.  Defaults to a log-uniform grid from R/1000 to R.

    Returns
    -------
    float
        Best q.
    """
    if q_grid is None:
        q_grid = list(np.logspace(np.log10(R / 1000), np.log10(R), 12))

    n, n_feat = X_train.shape
    best_q    = q_grid[0]
    best_msi  = float("inf")   # mean squared innovation

    for q in q_grid:
        Q = q * np.eye(n_feat)

        # Diffuse initialisation
        w  = np.zeros(n_feat)
        P  = np.eye(n_feat) * R * 100

        msi = 0.0
        for t in range(n):
            # Predict
            # (random walk: w_prior = w_post, P_prior = P_post + Q)
            P_prior = P + Q

            # Innovation
            H   = X_train[t]                              # (n_feat,)
            nu  = y_train[t] - float(H @ w)               # scalar
            S   = float(H @ P_prior @ H) + R               # scalar
            msi += nu ** 2 / max(S, 1e-12)

            # Update
            K = P_prior @ H / max(S, 1e-12)               # (n_feat,)
            w = w + K * nu
            I_KH = np.eye(n_feat) - np.outer(K, H)
            # Joseph form: P = (I-KH) P (I-KH)^T + K R K^T
            P = I_KH @ P_prior @ I_KH.T + np.outer(K, K) * R

        msi /= n
        if msi < best_msi:
            best_msi = msi
            best_q   = q

    logger.info(
        "Kalman Q grid search: best q = %.2e  (mean squared innovation = %.6f)",
        best_q, best_msi,
    )
    return best_q


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanWeightFilter:
    """
    Sequential Kalman filter for time-varying portfolio weights.

    Parameters
    ----------
    n_features : int
        Dimension of the state (= number of futures).
    Q : np.ndarray, shape (n_features, n_features)
        Process noise covariance.
    R : float
        Observation noise variance.
    w0 : np.ndarray, shape (n_features,), optional
        Initial state estimate.  Defaults to zeros.
    P0 : np.ndarray, shape (n_features, n_features), optional
        Initial state covariance.  Defaults to diffuse prior R·100·I.
    """

    def __init__(
        self,
        n_features: int,
        Q: np.ndarray,
        R: float,
        w0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        self.n = n_features
        self.Q = Q.copy()
        self.R = float(R)

        self.w = np.zeros(n_features) if w0 is None else w0.copy()
        self.P = np.eye(n_features) * R * 100 if P0 is None else P0.copy()

    def step(self, X_t: np.ndarray, y_t: float) -> Tuple[float, np.ndarray]:
        """
        One Kalman filter step: predict, generate forecast, then update.

        The forecast uses the **prior** (predicted) weight, ensuring
        strict one-step-ahead prediction with no look-ahead.

        Parameters
        ----------
        X_t : np.ndarray, shape (n_features,)
            Futures returns at time t.
        y_t : float
            Target return at time t (used for the update step only).

        Returns
        -------
        tuple[float, np.ndarray]
            ``(y_hat, w_posterior)``
            ``y_hat`` — one-step-ahead forecast (prior prediction).
            ``w_posterior`` — posterior weight after assimilating y_t.
        """
        # ── Predict ──────────────────────────────────────────────────────────
        # F = I → w_prior = w_post, P_prior = P_post + Q
        P_prior = self.P + self.Q

        # One-step-ahead forecast (uses prior, no look-ahead)
        y_hat = float(X_t @ self.w)

        # ── Update ───────────────────────────────────────────────────────────
        H   = X_t
        nu  = y_t - float(H @ self.w)          # innovation (scalar)
        S   = float(H @ P_prior @ H) + self.R  # innovation variance (scalar)
        K   = P_prior @ H / max(S, 1e-12)      # Kalman gain (n,)

        w_post = self.w + K * nu
        I_KH   = np.eye(self.n) - np.outer(K, H)
        # Joseph form for numerical stability
        P_post = I_KH @ P_prior @ I_KH.T + np.outer(K, K) * self.R

        # Update state
        self.w = w_post
        self.P = P_post

        return y_hat, w_post.copy()


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_kalman_diagnostics(
    result: Dict,
    save_prefix: str,
) -> None:
    """Three-panel diagnostic: weights, gross exposure, active returns."""
    wh   = result["weights_history"]
    rep  = result["replica_returns"]
    tgt  = result["target_returns"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)

    # Panel 1 — weights over time
    ax = axes[0]
    for col in wh.columns:
        ax.plot(wh[col], label=col, linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Kalman Filter — Portfolio Weights Over Time", fontsize=12)
    ax.set_ylabel("Weight")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2 — gross exposure
    ax = axes[1]
    ge = wh.abs().sum(axis=1)
    ax.plot(ge, color="purple", linewidth=1.4)
    ax.axhline(1.0, color="grey", linestyle=":",  linewidth=0.8, label="100%")
    ax.axhline(2.0, color="red",  linestyle="--", linewidth=0.8, label="200% UCITS")
    ax.set_title("Gross Exposure (Σ|w|)", fontsize=12)
    ax.set_ylabel("Gross Exposure")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3 — cumulative active return
    ax = axes[2]
    common = rep.index.intersection(tgt.index)
    active = (rep.loc[common] - tgt.loc[common]).cumsum()
    ax.plot(active, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(active.index, active, 0, alpha=0.15, color="steelblue")
    ax.set_title("Cumulative Active Return (replica − target)", fontsize=12)
    ax.set_ylabel("Cumulative active return")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=30)

    plt.suptitle("Kalman Filter — Diagnostics", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(f"{save_prefix}_diagnostics")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_kalman(
    X: pd.DataFrame,
    y: pd.Series,
    train_end_date: str = "2018-12-31",
    save_prefix: str = "kalman",
) -> Dict[str, Any]:
    """
    Full Kalman filter pipeline: estimate Q/R → filter → evaluate → save.

    Q and R are estimated from the training data only.  The filter is
    initialised at the beginning of the full time series using a diffuse
    prior and run sequentially through both train and test periods.

    Parameters
    ----------
    X : pd.DataFrame, shape (T, n_futures)
        Futures return feature matrix.
    y : pd.Series, length T
        Monster index weekly returns (target).
    train_end_date : str
        ISO date string separating train from test.
        Default ``'2018-12-31'``.
    save_prefix : str
        Filename prefix for all saved figures.

    Returns
    -------
    dict
        ``best_result``
            Result dict for the test period:
            ``replica_returns``, ``target_returns``, ``weights_history``.
        ``full_result``
            Same keys, covering the full series.
        ``params``
            dict: ``{'R': ..., 'q': ..., 'Q': ...}``.
    """
    _ensure_dirs()
    sns.set_theme(style="whitegrid")

    logger.info("=" * 60)
    logger.info("KALMAN FILTER — START")
    logger.info("=" * 60)

    # ── Align ────────────────────────────────────────────────────────────────
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    feat = X.columns.tolist()
    n_feat = len(feat)

    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)
    dates = y.index

    # ── Train / test split ───────────────────────────────────────────────────
    cutoff = pd.Timestamp(train_end_date)
    train_mask = X.index <= cutoff
    n_train = int(train_mask.sum())

    X_train = X_arr[:n_train]
    y_train = y_arr[:n_train]

    logger.info(
        "Train: %d obs (%s → %s) | Test: %d obs",
        n_train,
        dates[0].date(),
        cutoff.date(),
        len(X_arr) - n_train,
    )

    # ── Estimate R and q from training data ──────────────────────────────────
    logger.info("Estimating R …")
    R = _estimate_R(X_train, y_train)

    logger.info("Selecting q via grid search on training data …")
    q = _select_q_grid(X_train, y_train, R)

    Q = q * np.eye(n_feat)
    logger.info("Parameters: R = %.2e | q = %.2e", R, q)

    # ── Initialise filter ────────────────────────────────────────────────────
    # Warm-start the state with an OLS estimate from the training data
    # (better than zeros for numerical stability at t=0)
    ols_init = LinearRegression(fit_intercept=False)
    ols_init.fit(X_train, y_train)
    w0 = ols_init.coef_.copy()
    P0 = np.eye(n_feat) * R  # tight prior around OLS estimate

    kf = KalmanWeightFilter(n_feat, Q, R, w0=w0, P0=P0)
    logger.info("Filter initialised with OLS weights (warm start)")

    # ── Sequential filter — full series ──────────────────────────────────────
    replica_list:  List[float]      = []
    weights_list:  List[np.ndarray] = []

    for t in range(len(X_arr)):
        y_hat, w_post = kf.step(X_arr[t], y_arr[t])
        replica_list.append(y_hat)
        weights_list.append(w_post)

    replica_all = pd.Series(replica_list, index=dates, name="replica")
    target_all  = y.rename("target")
    wh_all      = pd.DataFrame(weights_list, index=dates, columns=feat)

    full_result = {
        "replica_returns":  replica_all,
        "target_returns":   target_all,
        "weights_history":  wh_all,
    }

    # ── Restrict to test period ───────────────────────────────────────────────
    test_start = dates[train_mask][-1] + pd.Timedelta(days=1)
    best_result = {
        "replica_returns":  replica_all.loc[test_start:],
        "target_returns":   target_all.loc[test_start:],
        "weights_history":  wh_all.loc[test_start:],
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    try:
        from utils.evaluation import compute_metrics
    except ImportError:
        from evaluation import compute_metrics

    m = compute_metrics(
        best_result["replica_returns"],
        best_result["target_returns"],
        model_name="Kalman",
    )
    logger.info(
        "Test period — TE=%.4f | IR=%.4f | Corr=%.4f | Sharpe=%.4f",
        m["tracking_error"], m["information_ratio"],
        m["correlation"],    m["sharpe_replica"],
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_kalman_diagnostics(best_result, save_prefix)

    # ── Save pickle ───────────────────────────────────────────────────────────
    payload = {
        "best_result":  best_result,
        "full_result":  full_result,
        "params":       {"R": R, "q": q, "Q": Q},
        "metrics":      m,
        "train_end_date": train_end_date,
        "feature_names": feat,
    }
    pkl_path = PICKLE_DIR / "kalman_results.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("KALMAN FILTER — DONE")
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

    run_kalman(data["X"], data["y"])