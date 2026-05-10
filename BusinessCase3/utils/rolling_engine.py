"""
rolling_engine.py
=================
Walk-forward rolling regression engine shared by all linear models.

Design decisions
----------------
One-step-ahead consistency
    At every step j the model is fit on the window [j-W, j) and
    predicts the return **at j** using X_raw[j] (contemporaneous
    features).  The same logic applies throughout both train and test
    phases — there is no off-by-one asymmetry between periods.

    This reflects the practical portfolio construction workflow: on day
    t the return at t is realised; the portfolio held over [t-1, t] was
    decided using weights from the model fit on [t-1-W, t-1].

Actual portfolio weights
    The regression is always fit on standardised X (StandardScaler fit on
    training data only).  For OLS and WOLS, Y is kept raw and β lives in
    (Y_raw / X_norm) space, so actual weights are:

        w_i = β_i / σ_{X,i}

    For regularised models (Ridge, LASSO, ElasticNet), Y is also
    standardised (via an optional ``scaler_Y`` argument) so that the
    penalty α is on a comparable scale regardless of Y magnitude.  β
    then lives in (Y_norm / X_norm) space and actual weights are:

        w_i = β_i · σ_Y / σ_{X,i}

    Both conversions produce weights in return space (units of Y_raw /
    X_raw), which is what ``apply_transaction_costs`` and the downstream
    evaluation expect. 

Gross replica returns
    Stored returns are *gross* of transaction costs.  Costs are applied
    downstream via ``utils.transaction_costs.apply_transaction_costs``,
    which uses the ``weights_history`` DataFrame.

Scaler ownership
    The scaler is fit externally (in ``run_linear_models``) on training
    data only and passed in.  The engine never refits or modifies it.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.evaluation import calculate_var

logger = logging.getLogger(__name__)

ANNUAL_FACTOR: int = 52


# ── Core rolling loop ─────────────────────────────────────────────────────────

def run_rolling(
    X_raw: pd.DataFrame,
    Y: pd.Series,
    scaler: StandardScaler,
    window: int,
    build_model_fn: Callable,
    feature_names: Optional[List[str]] = None,
    scaler_Y: Optional[StandardScaler] = None,
) -> Dict:
    """
    Walk-forward rolling regression with one-step-ahead prediction.

    Parameters
    ----------
    X_raw : pd.DataFrame, shape (T, n_features)
        Raw (unscaled) feature matrix covering the full period
        (train + test concatenated, aligned to Y).
    Y : pd.Series, length T
        Target return series.
    scaler : sklearn.preprocessing.StandardScaler
        StandardScaler already fit on the training portion of X only.
        Applied uniformly to the full X here.
    window : int
        Rolling window length W in periods.
    build_model_fn : callable
        Signature: ``fn(X_sc_window, Y_window) -> fitted estimator``.
        Called at each step; must return an object with a ``coef_``
        attribute (ndarray, shape (n_features,)).
    feature_names : list of str, optional
        Column names for the returned ``weights_history`` DataFrame.
        Defaults to ``X_raw.columns``.
    scaler_Y : StandardScaler, optional
        If provided (for regularised models — Ridge, LASSO, ElasticNet),
        the target window is standardised before fitting and the actual
        portfolio weights are recovered as:

            w_i = β_i · σ_Y / σ_{X,i}

        where σ_Y = ``scaler_Y.scale_[0]``. Where Y is normalised
        for penalised estimators so that the regularisation strength α
        is comparable across different Y magnitudes.  When ``None`` (OLS
        and WOLS), Y is used raw and weights are ``β / σ_X`` as before.

    Returns
    -------
    dict
        ``replica_returns``
            pd.Series of gross weekly replica returns (one per step).
        ``target_returns``
            pd.Series of aligned target returns.
        ``weights_history``
            pd.DataFrame of actual portfolio weights in return space,
            shape (T - window, n_features), indexed by date.

    Notes
    -----
    For LASSO, some β_i = 0 exactly.  The corresponding w_i = 0 as
    well (division by σ does not change sparsity).

    VaR (UCITS 1M 99%) is evaluated using the Cornish-Fisher +
    historical conservative estimator (``calculate_var``) applied to the
    accumulated realised replica return history.  Weights are scaled down
    proportionally when VaR > 20%.  At least 52 realised returns are
    required before scaling is activated.

    The first ``window`` observations have no prediction and are
    excluded from all returned series.
    """
    if feature_names is None:
        feature_names = list(X_raw.columns)

    X_arr = X_raw.values.astype(float)
    X_sc  = scaler.transform(X_arr)          # (T, F) — scaler already fit
    y_arr = Y.values.astype(float)
    dates = Y.index
    n, n_feat = X_arr.shape

    # Pre-compute σ_Y once (used only when scaler_Y is provided)
    sigma_Y: float = float(scaler_Y.scale_[0]) if scaler_Y is not None else 1.0

    replica_list:  List[float]      = []
    weights_list:  List[np.ndarray] = []
    date_list:     List             = []
    var_bw_list:   List[float]      = []   # backward-looking VaR (realised returns)
    var_fw_list:   List[float]      = []   # forward-looking VaR (scenario P&L)
    scale_list:    List[float]      = []

    for j in range(window, n):
        X_win = X_sc[j - window : j]    # (W, F) — standardised
        Y_win = y_arr[j - window : j]   # (W,)

        # ── Fit model ────────────────────────────────────────────────────────
        if scaler_Y is not None:
            # Penalised models: normalise Y so α is on a comparable scale
            Y_win_fit = scaler_Y.transform(Y_win.reshape(-1, 1)).ravel()
            model = build_model_fn(X_win, Y_win_fit)
            # Unscale: β is in (Y_norm / X_norm) space → multiply by σ_Y / σ_X
            actual_w = model.coef_ * sigma_Y / scaler.scale_
        else:
            model = build_model_fn(X_win, Y_win)
            # OLS / WOLS: β in (Y_raw / X_norm) space → divide by σ_X
            actual_w = model.coef_ / scaler.scale_

        # ── VaR scaling (Cornish-Fisher + historical conservative) ───────────
        # Both approaches activate after ≥ 52 realised replica returns (≈ 1 year).
        #
        # Backward-looking: VaR of the last 52 realised portfolio returns.
        #   Pro: reflects actual historical P&L.
        #   Con: slow to react — high leverage in calm markets produces
        #        low realised returns → low VaR → no scaling.
        #
        # Forward-looking: fix today's weights w_t and apply them to the last
        #   52 factor return rows → scenario P&L distribution.
        #   Pro: reacts immediately to large current positions, even in calm markets.
        #   Con: assumes a static portfolio over the lookback window.
        #
        # Scaling uses max(var_bw, var_fw) — the more conservative of the two.
        scale  = 1.0
        var_bw = float("nan")
        var_fw = float("nan")

        if len(replica_list) >= 52:
            # Backward-looking
            recent = np.array(replica_list[-52:])
            var_bw = calculate_var(recent, confidence=0.01, horizon=4, method='conservative')

            # Forward-looking: current weights applied to past 52 factor returns
            factor_window = X_arr[j - 52 : j]
            scenario_pnl  = factor_window @ actual_w
            var_fw = calculate_var(scenario_pnl, confidence=0.01, horizon=4, method='conservative')

            # Conservative: take the larger of the two
            var_eff = max(var_bw, var_fw)
            if var_eff > 0.20:
                scale    = 0.20 / var_eff
                actual_w = actual_w * scale
                logger.debug(
                    "VaR breach at %s | VaR_bw=%.4f | VaR_fw=%.4f | scale=%.4f",
                    dates[j].date(), var_bw, var_fw, scale,
                )

        var_bw_list.append(var_bw)
        var_fw_list.append(var_fw)
        scale_list.append(scale)

        # ── Gross replica return: X_raw[j] @ w ──────────────────────────────
        pred_gross = float(X_arr[j] @ actual_w)

        replica_list.append(pred_gross)
        weights_list.append(actual_w.copy())
        date_list.append(dates[j])

    replica  = pd.Series(replica_list, index=date_list, name="replica")
    target   = Y.loc[date_list]
    wh       = pd.DataFrame(weights_list, index=date_list, columns=feature_names)
    var_bw_s = pd.Series(var_bw_list, index=date_list, name="var_backward")
    var_fw_s = pd.Series(var_fw_list, index=date_list, name="var_forward")
    scale_s  = pd.Series(scale_list,  index=date_list, name="scale")

    # ── Summary statistics ────────────────────────────────────────────────────
    active       = replica.values - target.values
    te           = float(active.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
    ir           = (float(active.mean() * ANNUAL_FACTOR) / te) if te > 0 else float("nan")
    ge_mean      = float(wh.abs().sum(axis=1).mean())
    n_scaled     = int((scale_s < 1.0).sum())
    var_bw_clean = var_bw_s.dropna()
    var_fw_clean = var_fw_s.dropna()
    var_bw_mean  = float(var_bw_clean.mean()) if len(var_bw_clean) else float("nan")
    var_fw_mean  = float(var_fw_clean.mean()) if len(var_fw_clean) else float("nan")
    logger.info(
        "Rolling done | n=%d | TE=%.4f | IR=%.4f | mean GE=%.4f"
        " | mean VaR_bw=%.4f | mean VaR_fw=%.4f | VaR-scaled steps=%d/%d",
        len(replica), te, ir, ge_mean,
        var_bw_mean, var_fw_mean, n_scaled, len(var_bw_clean),
    )

    return {
        "replica_returns":      replica,
        "target_returns":       target,
        "weights_history":      wh,
        "var_backward_history": var_bw_s,
        "var_forward_history":  var_fw_s,
        "scale_history":        scale_s,
    }