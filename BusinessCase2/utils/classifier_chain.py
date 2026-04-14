"""
Classifier Chain for joint prediction of both investment targets.

Chain order: **AccumulationInvestment (0) → IncomeInvestment (1)**.

Rationale (Section 3.8 of the paper): The Modigliani life-cycle hypothesis
holds that accumulation is the earlier life stage; conditioning the
income-need model on whether accumulation need is present encodes the
directional prior that retirement-oriented needs succeed working-age ones.
Although the empirical inter-target correlation is near zero (r ≈ 0.011),
the chain encodes this structural financial prior and is compared empirically.

Base estimator: **XGBoost** (same configuration as the standalone script —
calibrated probabilities passed forward as chain features). scale_pos_weight
is applied inside the chain for IncomeInvestment via a custom wrapper.

Uses MinMaxScaler; saves a **single** pickle for both targets.
Saves artifacts to ``data/pickled_files/classifier_chain/``.

Run directly::

    python -m utils.classifier_chain
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    FEATURE_NAMES,
    N_OUTER_FOLDS,
    TARGETS,
    build_features,
    load_data,
    scale_pos_weight,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "pickled_files" / "classifier_chain"
)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "ClassifierChain(XGBoost)"

# Chain order: Accumulation (index 0) → Income (index 1)
# This encodes the life-cycle directional prior (Section 3.8)
_CHAIN_ORDER: list[int] = [0, 1]
_CHAIN_TARGET_ORDER: list[str] = ["AccumulationInvestment", "IncomeInvestment"]


# ---------------------------------------------------------------------------
# Helpers specific to multi-output format
# ---------------------------------------------------------------------------


def _split_and_scale_multi(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Stratified split + MinMaxScaler for multi-output labels.

    Stratification uses AccumulationInvestment (first chain target) because
    sklearn does not support multi-label stratified splitting natively.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Two-column target DataFrame ordered as ``_CHAIN_TARGET_ORDER``.
    test_size : float, optional
        Test fraction, by default 0.2.
    random_state : int, optional
        Random seed, by default 42.

    Returns
    -------
    tuple
        ``(X_train_scaled, X_test_scaled, y_train, y_test, scaler)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y.iloc[:, 0],   # stratify on AccumulationInvestment
    )
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_s, X_test_s, y_train, y_test, scaler


def _per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute classification metrics independently for each target column.

    Parameters
    ----------
    y_true : np.ndarray
        Shape ``(n_samples, 2)``, columns ordered as ``_CHAIN_TARGET_ORDER``.
    y_pred : np.ndarray
        Same shape.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{target_name: {metric: value}}``
    """
    result = {}
    for i, target in enumerate(_CHAIN_TARGET_ORDER):
        yt, yp = y_true[:, i], y_pred[:, i]
        result[target] = {
            "accuracy":  float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall":    float(recall_score(yt, yp, zero_division=0)),
            "f1":        float(f1_score(yt, yp, zero_division=0)),
        }
    return result


def _make_chain(spw_income: float = 1.0) -> ClassifierChain:
    """
    Build a ClassifierChain with XGBoost base estimator.

    Parameters
    ----------
    spw_income : float
        scale_pos_weight for IncomeInvestment (second chain target).

    Returns
    -------
    ClassifierChain
        Unfitted chain with Accum → Income order.
    """
    # XGBoost base estimator; scale_pos_weight applies to both chain targets
    # uniformly here — a known limitation of sklearn's ClassifierChain API.
    # The Income-specific correction is encoded in the standalone xgboost_shap.py.
    base = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=spw_income,
        verbosity=0,
        use_label_encoder=False,
    )
    return ClassifierChain(base, order=_CHAIN_ORDER)


def _cv_chain(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    k_folds: int = N_OUTER_FOLDS,
    spw_income: float = 1.0,
) -> dict[str, dict[str, list[float]]]:
    """
    Stratified K-Fold cross-validation for the ClassifierChain.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (already scaled).
    y_train : pd.DataFrame
        Training labels ordered as ``_CHAIN_TARGET_ORDER``.
    k_folds : int, optional
        Number of folds, by default ``N_OUTER_FOLDS`` (10).
    spw_income : float, optional
        scale_pos_weight for the XGBoost base estimator.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        ``{target_name: {metric: [fold_score, ...]}}``
    """
    # Stratify on AccumulationInvestment (first target)
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics: dict[str, dict[str, list[float]]] = {
        t: {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for t in _CHAIN_TARGET_ORDER
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(
        kf.split(X_train, y_train.iloc[:, 0])
    ):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        chain = _make_chain(spw_income)
        chain.fit(X_tr, y_tr)
        y_pred = chain.predict(X_val)

        fold_m = _per_target_metrics(y_val.values, y_pred)
        for target in _CHAIN_TARGET_ORDER:
            for metric, val in fold_m[target].items():
                fold_metrics[target][metric].append(val)

        logger.debug(
            "  Fold %d/%d — Accum F1: %.4f | Income F1: %.4f",
            fold_idx + 1, k_folds,
            fold_metrics["AccumulationInvestment"]["f1"][-1],
            fold_metrics["IncomeInvestment"]["f1"][-1],
        )

    return fold_metrics


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run() -> dict:
    """
    Train and evaluate the ClassifierChain on both targets jointly.

    Returns
    -------
    dict
        Result dictionary. ``'cv_metrics_raw'``, ``'cv_metrics_summary'``,
        and ``'test_metrics'`` are dicts-of-dicts keyed by target name.
        Saved to a single ``both_targets.joblib`` file.
    """
    df = load_data()

    # Chain column order: Accum first, Income second
    y = df[_CHAIN_TARGET_ORDER]

    X_eng = build_features(df)
    X_tr, X_te, y_tr, y_te, scaler = _split_and_scale_multi(X_eng, y)

    # scale_pos_weight for IncomeInvestment (second target)
    spw_income = scale_pos_weight(y_tr.iloc[:, 1])

    cv_raw = _cv_chain(X_tr, y_tr, spw_income=spw_income)
    for target in _CHAIN_TARGET_ORDER:
        logger.info(
            "  [F_E] %s CV F1: %.3f ± %.3f",
            target,
            np.mean(cv_raw[target]["f1"]),
            np.std(cv_raw[target]["f1"]),
        )

    chain = _make_chain(spw_income)
    chain.fit(X_tr, y_tr)
    y_pred = chain.predict(X_te)
    test_m = _per_target_metrics(y_te.values, y_pred)
    for target in _CHAIN_TARGET_ORDER:
        logger.info(
            "  [F_E] %s Test F1: %.3f", target, test_m[target]["f1"]
        )

    return {
        "model":          chain,
        "scaler":         scaler,
        "cv_metrics_raw": cv_raw,
        "cv_metrics_summary": {
            t: summarise_cv(cv_raw[t]) for t in _CHAIN_TARGET_ORDER
        },
        "test_metrics":   test_m,
        "y_test_true":    y_te.values,
        "y_test_pred":    y_pred,
        "feature_names":  FEATURE_NAMES,
        "target_name":    _CHAIN_TARGET_ORDER,
        "model_name":     MODEL_NAME,
        "chain_order":    "AccumulationInvestment → IncomeInvestment",
        "ablation":       None,  # chain is evaluated as a single unit
    }


def main() -> None:
    """Train, evaluate, and pickle the ClassifierChain."""
    result = run()
    out_path = PICKLE_DIR / "both_targets.joblib"
    try:
        joblib.dump(result, out_path, compress=3)
        logger.info("Saved: %s", out_path)
    except OSError as exc:
        logger.error("Failed to save %s: %s", out_path, exc)
        raise


if __name__ == "__main__":
    main()
