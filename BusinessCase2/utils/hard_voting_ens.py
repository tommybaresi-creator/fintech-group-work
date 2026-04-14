"""
Hard Voting Ensemble for both investment targets.

Composition: Logistic Regression + XGBoost (calibrated) + SVM (RBF).
Identical component set to soft_voting_ens.py; differs only in aggregation
(majority vote vs probability averaging).

Hard voting discards probability information before aggregation and is
generally inferior to soft voting for well-calibrated models. Included
for empirical comparison only.

Same Pipeline encapsulation as soft_voting_ens.py — leakage-free CV.

Saves artifacts to data/pickled_files/hard_voting_ens/.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_INNER_FOLDS,
    TARGETS,
    build_baseline_features,
    build_features,
    compute_brier_score,
    compute_cv_metrics,
    compute_test_metrics,
    load_data,
    make_result_dict,
    scale_pos_weight,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "hard_voting_ens"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "HardVotingEnsemble(LR+XGB+SVM)"


def _make_model(spw: float = 1.0, class_weight=None) -> VotingClassifier:
    """Hard-voting ensemble with per-estimator Pipelines. Leakage-free CV."""
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear",
            max_iter=1000, random_state=42,
            class_weight=class_weight,
        )),
    ])

    xgb_pipe = Pipeline([
        ("clf", CalibratedClassifierCV(
            XGBClassifier(
                random_state=42, eval_metric="logloss",
                scale_pos_weight=spw, verbosity=0, use_label_encoder=False,
            ),
            method="isotonic", cv=N_INNER_FOLDS,
        )),
    ])

    svm_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(
            kernel="rbf", probability=True,
            random_state=42, class_weight=class_weight,
        )),
    ])

    return VotingClassifier(
        estimators=[("lr", lr_pipe), ("xgb", xgb_pipe), ("svm", svm_pipe)],
        voting="hard",
    )


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y   = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    cw  = "balanced"          if target_col == "IncomeInvestment" else None

    # ---- F_E -----------------------------------------------------------------
    X_eng = build_features(df)
    X_tr, X_te, y_tr, y_te = split_data(X_eng, y)

    cv_raw = compute_cv_metrics(_make_model(spw, cw), X_tr, y_tr)
    logger.info("  [F_E] CV  F1: %.3f ± %.3f", np.mean(cv_raw["f1"]), np.std(cv_raw["f1"]))

    model = _make_model(spw, cw)
    model.fit(X_tr, y_tr)
    test_m = compute_test_metrics(model, X_te, y_te)
    logger.info("  [F_E] Test F1 (thr=0.5): %.3f", test_m["f1"])

    # Hard voting: predict_proba available via probability=True on SVM
    try:
        brier = compute_brier_score(model, X_te, y_te)
        logger.info("  [F_E] Brier: %.4f", brier)
    except ValueError:
        brier = None
        logger.warning("  Hard voting Brier score unavailable")

    try:
        thr_info = select_threshold_pr_curve(model, X_te, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    # ---- F_B (ablation) ------------------------------------------------------
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_data(X_base, y)
    cv_raw_b = compute_cv_metrics(_make_model(spw, cw), X_tr_b, y_tr_b)
    model_b  = _make_model(spw, cw)
    model_b.fit(X_tr_b, y_tr_b)
    test_m_b = compute_test_metrics(model_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1: %.3f  (ΔF_E−F_B = %+.3f)", test_m_b["f1"], test_m["f1"] - test_m_b["f1"])

    return make_result_dict(
        model=model,
        scaler=None,
        cv_metrics_raw=cv_raw,
        test_metrics=test_m,
        y_test_true=y_te.values,
        y_test_pred=model.predict(X_te),
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw,   "cv_metrics_summary": summarise_cv(cv_raw),   "test_metrics": test_m},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        brier_score=brier,
        threshold_info=thr_info,
    )


def main() -> None:
    df = load_data()
    for target in TARGETS:
        result = run_for_target(df, target)
        out_path = PICKLE_DIR / f"{target.lower()}.joblib"
        joblib.dump(result, out_path, compress=3)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()