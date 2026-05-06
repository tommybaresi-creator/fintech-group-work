"""
models.py
=========
Sklearn model factories and CV-based hyperparameter selection
for the rolling-window portfolio replication pipeline.

All CV is performed on the **training set only** via ``TimeSeriesSplit``.
The test set is never touched during tuning.

Scaling convention
------------------
Only X is standardised (``StandardScaler`` fit on training data).
Y is kept in raw return space throughout so that:

* regression coefficients β are in units of (Y_return / X_std)
* actual portfolio weights are recovered as  w_i = β_i / σ_{X,i}
* TC and gross-exposure calculations are in natural return space

The penalty term for Ridge/LASSO/EN is therefore applied to
coefficients that are already on a comparable scale (unit-variance X),
so the selected alpha is directly interpretable as a regularisation
strength relative to the return-scale of Y.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ANNUAL_FACTOR: int = 52  # weekly data

# ── Default grids ─────────────────────────────────────────────────────────────
# Grids are intentionally wide; CV will pick within them.
# With unscaled Y (weekly returns ~0.5 % std) and unit-variance X,
# meaningful alpha for Ridge/LASSO lives in [1e-6, 1e-1].
_ALPHAS_RIDGE = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]
_ALPHAS_LASSO = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
_ALPHAS_EN    = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
_L1_RATIOS_EN = [0.1, 0.3, 0.5, 0.7, 0.9]
_LAMBDAS_WOLS = [0.85, 0.90, 0.94, 0.96, 0.97, 0.98, 0.99]


# ── Exponential decay weights ─────────────────────────────────────────────────

def exp_weights(window: int, lam: float) -> np.ndarray:
    """
    Compute exponential decay sample weights for a rolling window.

    Observation at lag k (k=0 is the most recent) receives weight λ^k,
    so recent data dominates as λ → 1.  Weights are **not** normalised
    (sklearn handles sample-weight normalisation internally).

    Parameters
    ----------
    window : int
        Number of observations in the rolling window.
    lam : float
        Decay factor in (0, 1].  λ=1 recovers uniform weights (plain OLS).

    Returns
    -------
    np.ndarray, shape (window,)
        Weights from oldest (index 0) to newest (index window-1).
    """
    idx = np.arange(window)
    return lam ** (window - 1 - idx)


# ── Hyperparameter selection ──────────────────────────────────────────────────

def select_alpha_ridge(
    X_train_sc: np.ndarray,
    Y_train: pd.Series,
    alphas: Optional[List[float]] = None,
    n_splits: int = 5,
) -> float:
    """
    Select Ridge regularisation strength via walk-forward CV on training data.

    Parameters
    ----------
    X_train_sc : np.ndarray, shape (n_train, n_features)
        Standardised training features (fit on training set only).
    Y_train : pd.Series, length n_train
        Training target returns (raw, unscaled).
    alphas : list of float, optional
        Grid to search.  Defaults to ``_ALPHAS_RIDGE``.
    n_splits : int
        Number of ``TimeSeriesSplit`` folds.

    Returns
    -------
    float
        Best alpha selected by MSE on held-out folds.
    """
    if alphas is None:
        alphas = _ALPHAS_RIDGE
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = RidgeCV(alphas=alphas, cv=tscv, fit_intercept=False)
    cv.fit(X_train_sc, Y_train.values)
    logger.info("Ridge CV best alpha = %.2e", cv.alpha_)
    return float(cv.alpha_)


def select_alpha_lasso(
    X_train_sc: np.ndarray,
    Y_train: pd.Series,
    alphas: Optional[List[float]] = None,
    n_splits: int = 5,
) -> float:
    """
    Select LASSO regularisation strength via walk-forward CV on training data.

    Parameters
    ----------
    X_train_sc : np.ndarray
    Y_train : pd.Series
    alphas : list of float, optional
    n_splits : int

    Returns
    -------
    float
        Best alpha.
    """
    if alphas is None:
        alphas = _ALPHAS_LASSO
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = LassoCV(alphas=alphas, cv=tscv, fit_intercept=False, max_iter=10_000)
    cv.fit(X_train_sc, Y_train.values)
    logger.info("LASSO CV best alpha = %.2e", cv.alpha_)
    return float(cv.alpha_)


def select_alpha_elasticnet(
    X_train_sc: np.ndarray,
    Y_train: pd.Series,
    alphas: Optional[List[float]] = None,
    l1_ratios: Optional[List[float]] = None,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """
    Select ElasticNet (alpha, l1_ratio) via walk-forward CV on training data.

    Parameters
    ----------
    X_train_sc : np.ndarray
    Y_train : pd.Series
    alphas : list of float, optional
    l1_ratios : list of float, optional
    n_splits : int

    Returns
    -------
    tuple[float, float]
        ``(best_alpha, best_l1_ratio)``.
    """
    if alphas is None:
        alphas = _ALPHAS_EN
    if l1_ratios is None:
        l1_ratios = _L1_RATIOS_EN
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=tscv,
        fit_intercept=False,
        max_iter=10_000,
    )
    cv.fit(X_train_sc, Y_train.values)
    logger.info(
        "ElasticNet CV best alpha = %.2e | l1_ratio = %.2f",
        cv.alpha_,
        cv.l1_ratio_,
    )
    return float(cv.alpha_), float(cv.l1_ratio_)


def select_lambda_wols(
    X_train_sc: np.ndarray,
    Y_train: pd.Series,
    scaler_scale: np.ndarray,
    X_train_raw: np.ndarray,
    window: int,
    lambdas: Optional[List[float]] = None,
    n_splits: int = 3,
) -> float:
    """
    Select WOLS exponential decay factor via walk-forward CV minimising TE.

    **Criterion: Tracking Error, not IR.**
    Minimising TE is the correct objective for replication — we want the
    replica to closely shadow the target, not to generate excess returns
    above it.  Selecting on IR would optimise for consistent
    outperformance, which is a different (and wrong) goal.

    For each (fold, lambda) pair:
    1. Roll over the validation observations using a window ending just
       before each.
    2. Predict each validation return as  X_raw[j] @ (coef_ / σ_X).
    3. Score via annualised TE against the validation targets.

    Parameters
    ----------
    X_train_sc : np.ndarray, shape (n_train, n_features)
        Standardised training features.
    Y_train : pd.Series, length n_train
    scaler_scale : np.ndarray, shape (n_features,)
        ``scaler.scale_`` — standard deviations used in X standardisation.
    X_train_raw : np.ndarray, shape (n_train, n_features)
        Raw (unscaled) training features, used for actual return computation.
    window : int
    lambdas : list of float, optional
    n_splits : int

    Returns
    -------
    float
        Best lambda minimising mean out-of-fold TE.
    """
    if lambdas is None:
        lambdas = _LAMBDAS_WOLS

    tscv = TimeSeriesSplit(n_splits=n_splits)
    te_scores: Dict[float, List[float]] = {l: [] for l in lambdas}

    for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X_train_sc)):
        X_sc_fold = np.vstack([X_train_sc[tr_idx], X_train_sc[vl_idx]])
        X_raw_fold = np.vstack([X_train_raw[tr_idx], X_train_raw[vl_idx]])
        Y_fold = pd.concat([
            Y_train.iloc[tr_idx],
            Y_train.iloc[vl_idx],
        ]).reset_index(drop=True)
        n_tr = len(tr_idx)

        for lam in lambdas:
            preds: List[float] = []
            targets: List[float] = []

            for j in range(max(window, n_tr), n_tr + len(vl_idx)):
                X_win = X_sc_fold[j - window : j]
                Y_win = Y_fold.iloc[j - window : j]
                ew = exp_weights(window, lam)

                m = LinearRegression(fit_intercept=False)
                m.fit(X_win, Y_win.values, sample_weight=ew)

                # Actual portfolio return (return space, not scaled space)
                actual_w = m.coef_ / scaler_scale
                pred = float(X_raw_fold[j] @ actual_w)

                preds.append(pred)
                targets.append(float(Y_fold.iloc[j]))

            if len(preds) > 1:
                active = np.array(preds) - np.array(targets)
                te = float(active.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
                te_scores[lam].append(te)

    mean_te = {l: float(np.mean(v)) for l, v in te_scores.items() if v}
    if not mean_te:
        logger.warning("WOLS CV produced no scores — defaulting lambda=0.97")
        return 0.97

    best_lam = min(mean_te, key=mean_te.get)
    logger.info(
        "WOLS CV scores (lambda → mean TE): %s",
        " | ".join(f"{l:.2f}→{v:.4f}" for l, v in sorted(mean_te.items())),
    )
    logger.info("WOLS CV best lambda = %.2f  (TE = %.4f)", best_lam, mean_te[best_lam])
    return best_lam


# ── Model builder functions ───────────────────────────────────────────────────

def build_ols() -> LinearRegression:
    """Return an unconstrained OLS estimator (no intercept)."""
    return LinearRegression(fit_intercept=False)


def build_ridge(alpha: float) -> Ridge:
    """
    Return a Ridge estimator with the given regularisation strength.

    Parameters
    ----------
    alpha : float
        L2 penalty coefficient.  Should be selected via
        :func:`select_alpha_ridge`.
    """
    return Ridge(alpha=alpha, fit_intercept=False)


def build_lasso(alpha: float) -> Lasso:
    """
    Return a LASSO estimator with the given regularisation strength.

    Parameters
    ----------
    alpha : float
        L1 penalty coefficient.  Should be selected via
        :func:`select_alpha_lasso`.
    """
    return Lasso(alpha=alpha, fit_intercept=False, max_iter=10_000)


def build_elasticnet(alpha: float, l1_ratio: float) -> ElasticNet:
    """
    Return an ElasticNet estimator.

    Parameters
    ----------
    alpha : float
        Overall penalty strength.
    l1_ratio : float
        Mix between L1 (1.0) and L2 (0.0).
    """
    return ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10_000
    )


def build_wols(lam: float) -> "WOLSModel":
    """
    Return a WOLS wrapper with the given decay factor.

    Parameters
    ----------
    lam : float
        Exponential decay factor λ ∈ (0, 1].  λ=1 → uniform weights.
    """
    return WOLSModel(lam=lam)


# ── WOLS wrapper ──────────────────────────────────────────────────────────────

class WOLSModel:
    """
    Thin wrapper around ``LinearRegression`` that applies exponential
    sample weights at fit time.

    Exposes the same ``fit`` / ``predict`` / ``coef_`` interface as
    sklearn estimators so it integrates cleanly with
    :func:`~utils.rolling_engine.run_rolling`.

    Parameters
    ----------
    lam : float
        Decay factor λ ∈ (0, 1].
    """

    def __init__(self, lam: float) -> None:
        self.lam = lam
        self._inner = LinearRegression(fit_intercept=False)
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WOLSModel":
        """
        Fit with exponential sample weights.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        w = exp_weights(len(y), self.lam)
        self._inner.fit(X, y, sample_weight=w)
        self.coef_ = self._inner.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        return self._inner.predict(X)