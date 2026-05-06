"""
risk.py
=======
UCITS risk constraint utilities shared across all model families.

VaR methodology
---------------
The UCITS III Absolute VaR approach requires:

    1-month VaR at 99% confidence ≤ 20% of NAV

We implement **historical-simulation VaR**, which is the regulatory
standard and avoids the normal-distribution tail underestimation of
the parametric approach.

The computation is *forward-looking*: it projects the **current**
weight vector onto the distribution of **recent factor returns** to
obtain a scenario P&L distribution, then reads off the empirical
1st percentile.  Square-root-of-time scaling extrapolates weekly
scenarios to the 1-month horizon.

This is in contrast to the incorrect backward-looking approach where
VaR is estimated from the replica's own historical return series
(circular, and not UCITS-compliant because it confounds current
position risk with past positioning).

Commitment approach
-------------------
UCITS also permits a simpler alternative: gross leverage
(= Σ|w_i|) ≤ 200%.  Both constraints are provided here;
users may apply whichever is appropriate for their context.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── UCITS limits ──────────────────────────────────────────────────────────────
UCITS_VAR_LIMIT: float    = 0.20   # 1M VaR(99%) ≤ 20%
UCITS_LEVERAGE_LIMIT: float = 2.00  # gross leverage ≤ 200%
VAR_CONFIDENCE: float     = 0.99
VAR_HORIZON_WEEKS: int    = 4       # ≈ 1 calendar month


# ── VaR (historical simulation) ───────────────────────────────────────────────

def compute_var_hs(
    weights: np.ndarray,
    factor_returns: np.ndarray,
    confidence: float = VAR_CONFIDENCE,
    horizon_weeks: int = VAR_HORIZON_WEEKS,
) -> float:
    """
    Historical-simulation VaR for a given weight vector.

    Projects ``weights`` onto each row of ``factor_returns`` to obtain
    a scenario P&L series, then extracts the empirical lower quantile
    and scales to the required horizon.

    Parameters
    ----------
    weights : np.ndarray, shape (n_features,)
        Current portfolio weight vector (actual return-space weights,
        i.e.  coef_ / σ_X  for regression-based models).
    factor_returns : np.ndarray, shape (lookback, n_features)
        Recent weekly futures return matrix used to build the
        scenario distribution.  Typically the last 52 observations.
    confidence : float
        VaR confidence level (e.g. 0.99 for 99 %).
    horizon_weeks : int
        Investment horizon in weeks.  4 ≈ 1 calendar month.

    Returns
    -------
    float
        VaR as a **positive** number representing potential loss.
        Returns 0.0 if the distribution implies no loss at this
        confidence level.

    Notes
    -----
    Square-root-of-time scaling  √T  is standard for historical
    simulation; it is exact under i.i.d. weekly returns and is an
    accepted UCITS approximation for short horizons.
    """
    scenario_pnl = factor_returns @ weights          # (lookback,) scenario P&Ls
    weekly_var   = -np.percentile(scenario_pnl, (1.0 - confidence) * 100)
    monthly_var  = weekly_var * np.sqrt(horizon_weeks)
    return float(max(monthly_var, 0.0))


def scale_to_var_limit(
    weights: np.ndarray,
    factor_returns: np.ndarray,
    max_var: float = UCITS_VAR_LIMIT,
    confidence: float = VAR_CONFIDENCE,
    horizon_weeks: int = VAR_HORIZON_WEEKS,
) -> Tuple[np.ndarray, float, float]:
    """
    Scale weights down if the implied VaR exceeds the UCITS limit.

    Under historical simulation, portfolio VaR scales linearly with
    the weight vector (VaR(λw) = λ·VaR(w) for λ > 0), so a simple
    proportional rescaling enforces the constraint exactly.

    Parameters
    ----------
    weights : np.ndarray
        Raw weight vector.
    factor_returns : np.ndarray
        Recent factor return matrix for VaR estimation.
    max_var : float
        VaR upper bound.  Default 0.20 (UCITS Absolute VaR limit).
    confidence : float
    horizon_weeks : int

    Returns
    -------
    tuple[np.ndarray, float, float]
        ``(scaled_weights, scale_factor, var_before_scaling)``
        ``scale_factor = 1.0`` if no scaling was needed.
    """
    var = compute_var_hs(weights, factor_returns, confidence, horizon_weeks)
    if var <= max_var or var <= 0.0:
        return weights.copy(), 1.0, var

    scale = max_var / var
    logger.debug(
        "VaR %.4f > limit %.4f → scaling weights by %.4f",
        var, max_var, scale,
    )
    return weights * scale, scale, var


# ── Gross exposure ────────────────────────────────────────────────────────────

def gross_exposure(weights: np.ndarray) -> float:
    """
    Compute portfolio gross exposure (commitment approach).

    Gross exposure = Σ_i |w_i|.  Under the UCITS commitment approach
    this must not exceed 200 % of NAV.

    Parameters
    ----------
    weights : np.ndarray

    Returns
    -------
    float
    """
    return float(np.abs(weights).sum())


def scale_to_leverage_limit(
    weights: np.ndarray,
    max_leverage: float = UCITS_LEVERAGE_LIMIT,
) -> Tuple[np.ndarray, float]:
    """
    Scale weights proportionally so gross exposure ≤ ``max_leverage``.

    Parameters
    ----------
    weights : np.ndarray
    max_leverage : float
        Maximum gross exposure.  Default 2.0 (200 %).

    Returns
    -------
    tuple[np.ndarray, float]
        ``(scaled_weights, scale_factor)``.
        ``scale_factor = 1.0`` if no scaling was needed.
    """
    ge = gross_exposure(weights)
    if ge <= max_leverage or ge <= 0.0:
        return weights.copy(), 1.0
    scale = max_leverage / ge
    return weights * scale, scale


# ── LASSO selection frequency ─────────────────────────────────────────────────

def lasso_selection_frequency(weights_history: pd.DataFrame) -> pd.Series:
    """
    Compute the fraction of periods in which each feature has a non-zero weight.

    Useful for interpreting which futures consistently drive the replica
    under LASSO.

    Parameters
    ----------
    weights_history : pd.DataFrame
        Weight history (T × n_features).  Zero entries represent
        LASSO-zeroed positions.

    Returns
    -------
    pd.Series
        Selection frequency in [0, 1], indexed by feature name.
    """
    return (weights_history != 0.0).mean().rename("selection_frequency")