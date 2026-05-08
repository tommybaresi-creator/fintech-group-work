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
    The regression is fit on standardised X (StandardScaler fit on
    training data only).  Coefficients β live in standardised space.
    The actual portfolio weight on futures i is:

        w_i = β_i / σ_{X,i}

    where σ_{X,i} = ``scaler.scale_[i]``.  This conversion is
    applied at every step; only actual weights are stored.

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

from utils.risk import scale_to_var_limit

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

    Returns
    -------
    dict
        ``replica_returns``
            pd.Series of gross weekly replica returns (one per step).
        ``target_returns``
            pd.Series of aligned target returns.
        ``weights_history``
            pd.DataFrame of actual portfolio weights (w = β / σ_X),
            shape (T - window, n_features), indexed by date.

    Notes
    -----
    For LASSO, some β_i = 0 exactly.  The corresponding w_i = 0 as
    well (division by σ_X_i does not change sparsity).

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

    replica_list: List[float]     = []
    weights_list: List[np.ndarray] = []
    date_list:    List           = []

    for j in range(window, n):
        X_win = X_sc[j - window : j]    # (W, F) — standardised
        Y_win = y_arr[j - window : j]   # (W,)

        model = build_model_fn(X_win, Y_win)

        # Actual portfolio weights: β / σ_X
        actual_w = model.coef_ / scaler.scale_

        if j >= window + 52:
            factor_window = X_arr[j - 52 : j]      # (52, n_features) — scenario returns
            actual_w, _, _ = scale_to_var_limit(actual_w, factor_window)

        # Gross replica return: X_raw[j] @ w  (exact, no mean-offset approximation)
        pred_gross = float(X_arr[j] @ actual_w)

        replica_list.append(pred_gross)
        weights_list.append(actual_w.copy())
        date_list.append(dates[j])

    replica = pd.Series(replica_list, index=date_list, name="replica")
    target  = Y.loc[date_list]
    wh      = pd.DataFrame(weights_list, index=date_list, columns=feature_names)

    # Log summary statistics
    active  = replica.values - target.values
    te      = float(active.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
    ir      = (float(active.mean() * ANNUAL_FACTOR) / te) if te > 0 else float("nan")
    ge_mean = float(wh.abs().sum(axis=1).mean())
    logger.info(
        "Rolling done | n=%d | TE=%.4f | IR=%.4f | mean gross exp=%.4f",
        len(replica), te, ir, ge_mean,
    )

    return {
        "replica_returns":  replica,
        "target_returns":   target,
        "weights_history":  wh,
    }