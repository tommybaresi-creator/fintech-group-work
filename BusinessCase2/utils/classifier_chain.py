"""
Classifier Chain for joint prediction of both investment targets.

Chain order: AccumulationInvestment (0) → IncomeInvestment (1).

Rationale: The Modigliani life-cycle hypothesis holds that
accumulation is the earlier life stage; conditioning the income-need model
on whether accumulation need is present encodes the directional prior that
retirement-oriented needs succeed working-age ones. Although the empirical
inter-target correlation is near zero (r ≈ 0.011), the chain encodes this
structural prior and is compared empirically.

CV leakage fix: MinMaxScaler is refitted inside each fold of _cv_chain,
not once on the full training set before CV.

Known limitation: sklearn ClassifierChain applies the same base estimator
to all chain targets. scale_pos_weight cannot be set per-target.
For IncomeInvestment (38% positive) this under-corrects imbalance.
The standalone xgboost_shap.py applies the correct per-target correction.

Saves a single pickle for both targets.
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
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "classifier_chain"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "ClassifierChain(XGBoost)"

_CHAIN_ORDER: list        = [0, 1]
_CHAIN_TARGET_ORDER: list = ["AccumulationInvestment", "IncomeInvestment"]


def _make_chain() -> ClassifierChain:
    """
    Build a ClassifierChain with XGBoost base estimator.

    scale_pos_weight is NOT set here because ClassifierChain cannot apply
    different weights per target. This is a known sklearn limitation.
    """
    base = XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=1.0,  # cannot be per-target inside ClassifierChain
        verbosity=0, use_label_encoder=False,
    )
    return ClassifierChain(base, order=_CHAIN_ORDER)


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def _cv_chain(
    X_train_raw: pd.DataFrame,
    y_train: pd.DataFrame,
    k_folds: int = N_OUTER_FOLDS,
) -> dict:
    """
    Stratified K-Fold CV for the ClassifierChain.

    MinMaxScaler is refitted inside each fold on that fold's training data —
    the correct leakage-free pattern. X_train_raw must be UNSCALED.
    Stratification is on AccumulationInvestment (first chain target).
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {t: {"accuracy": [], "precision": [], "recall": [], "f1": []} for t in _CHAIN_TARGET_ORDER}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train.iloc[:, 0])):
        X_tr_raw  = X_train_raw.iloc[tr_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_tr      = y_train.iloc[tr_idx]
        y_val     = y_train.iloc[val_idx]

        # Refit scaler on this fold's training data only
        fold_scaler = MinMaxScaler()
        X_tr  = pd.DataFrame(fold_scaler.fit_transform(X_tr_raw),  columns=X_tr_raw.columns,  index=X_tr_raw.index)
        X_val = pd.DataFrame(fold_scaler.transform(X_val_raw),     columns=X_val_raw.columns, index=X_val_raw.index)

        chain = _make_chain()
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


def run() -> dict:
    df = load_data()
    y  = df[_CHAIN_TARGET_ORDER]
    X_eng = build_features(df)

    # Split using stratification on AccumulationInvestment (first target)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y.iloc[:, 0]
    )

    # CV: scaler refits inside each fold
    cv_raw = _cv_chain(X_tr_raw, y_tr)
    for target in _CHAIN_TARGET_ORDER:
        logger.info(
            "  [F_E] %s CV F1: %.3f ± %.3f",
            target, np.mean(cv_raw[target]["f1"]), np.std(cv_raw[target]["f1"]),
        )

    # Final model: fit scaler on full training set
    final_scaler = MinMaxScaler()
    X_tr = pd.DataFrame(final_scaler.fit_transform(X_tr_raw), columns=X_tr_raw.columns, index=X_tr_raw.index)
    X_te = pd.DataFrame(final_scaler.transform(X_te_raw),     columns=X_te_raw.columns, index=X_te_raw.index)

    chain = _make_chain()
    chain.fit(X_tr, y_tr)
    y_pred = chain.predict(X_te)
    test_m = _per_target_metrics(y_te.values, y_pred)
    for target in _CHAIN_TARGET_ORDER:
        logger.info("  [F_E] %s Test F1: %.3f", target, test_m[target]["f1"])

    return {
        "model":              chain,
        "scaler":             final_scaler,
        "cv_metrics_raw":     cv_raw,
        "cv_metrics_summary": {t: summarise_cv(cv_raw[t]) for t in _CHAIN_TARGET_ORDER},
        "test_metrics":       test_m,
        "y_test_true":        y_te.values,
        "y_test_pred":        y_pred,
        "feature_names":      FEATURE_NAMES,
        "target_name":        _CHAIN_TARGET_ORDER,
        "model_name":         MODEL_NAME,
        "chain_order":        "AccumulationInvestment → IncomeInvestment",
        "ablation":           None,
    }


def main() -> None:
    result = run()
    out_path = PICKLE_DIR / "both_targets.joblib"
    joblib.dump(result, out_path, compress=3)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()