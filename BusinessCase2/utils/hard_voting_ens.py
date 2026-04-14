"""
Hard Voting Ensemble: LR + XGBoost (calibrated) + SVM (RBF).

Included for comparison only — hard voting discards probability information
before aggregation and is generally inferior to soft voting for well-calibrated
models. Hyperparameter tuning applied. Output as .pkl.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
    get_propensity_scores,
    load_data,
    make_result_dict,
    save_result,
    scale_pos_weight,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "HardVotingEnsemble(LR+XGB+SVM)"
FOLDER: str = "hard_voting_ens"

COMBINED_GRID = {
    "lr__clf__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "lr__clf__penalty": ["l1", "l2"],
    "svm__clf__C":      [0.1, 0.5, 1.0, 5.0, 10.0],
    "svm__clf__gamma":  ["scale", "auto", 0.01, 0.1],
}


def _make_ensemble(spw=1.0, class_weight=None) -> VotingClassifier:
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear",
            max_iter=1000, random_state=42, class_weight=class_weight,
        )),
    ])
    xgb_pipe = Pipeline([
        ("clf", CalibratedClassifierCV(
            XGBClassifier(
                scale_pos_weight=spw, random_state=42, eval_metric="logloss",
                verbosity=0, use_label_encoder=False,
            ),
            method="isotonic", cv=N_INNER_FOLDS,
        )),
    ])
    svm_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight=class_weight)),
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

    X_eng = build_features(df)
    X_tr, X_te, y_tr, y_te = split_data(X_eng, y)

    logger.info("  Tuning ensemble (inner %d-fold CV)...", N_INNER_FOLDS)
    best_ensemble, best_params, best_cv_score = tune_hyperparameters(
        _make_ensemble(spw, cw), COMBINED_GRID, X_tr, y_tr, n_iter=20
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params, best_cv_score)

    cv_raw = compute_cv_metrics(best_ensemble, X_tr, y_tr)
    logger.info("  [F_E] CV F1: %.3f ± %.3f", np.mean(cv_raw["f1"]), np.std(cv_raw["f1"]))

    best_ensemble.fit(X_tr, y_tr)
    test_m = compute_test_metrics(best_ensemble, X_te, y_te)
    logger.info("  [F_E] Test F1=%.3f  Precision=%.3f", test_m["f1"], test_m["precision"])

    try:
        y_proba = get_propensity_scores(best_ensemble, X_te)
        brier   = compute_brier_score(best_ensemble, X_te, y_te)
    except ValueError:
        y_proba = None
        brier   = None

    try:
        thr_info = select_threshold_pr_curve(best_ensemble, X_te, y_te)
    except (ValueError, AttributeError) as exc:
        logger.warning("  Threshold optimisation failed: %s", exc)
        thr_info = None

    # F_B ablation
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_data(X_base, y)
    ens_b = _make_ensemble(spw, cw)
    cv_raw_b = compute_cv_metrics(ens_b, X_tr_b, y_tr_b)
    ens_b.fit(X_tr_b, y_tr_b)
    test_m_b = compute_test_metrics(ens_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)", test_m_b["f1"], test_m["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=best_ensemble, scaler=None,
        cv_metrics_raw=cv_raw, test_metrics=test_m,
        y_test_true=y_te.values, y_test_pred=best_ensemble.predict(X_te),
        y_test_proba=y_proba,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw,   "cv_metrics_summary": summarise_cv(cv_raw),   "test_metrics": test_m},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        brier_score=brier, threshold_info=thr_info,
    )
    path = save_result(result, FOLDER, target_col)
    logger.info("  Saved: %s", path)
    return result


def main():
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()