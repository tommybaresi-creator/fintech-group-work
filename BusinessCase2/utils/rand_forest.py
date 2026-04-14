"""
Random Forest for both investment targets.

Fixes applied:
- Hyperparameter tuning via RandomizedSearchCV
- Both uncalibrated AND calibrated test metrics stored (diagnose calibration effect)
- y_test_proba stored
- Output saved as .pkl
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    build_features,
    calibrate_model,
    compute_brier_score,
    compute_cv_metrics,
    compute_test_metrics,
    get_propensity_scores,
    load_data,
    make_result_dict,
    save_result,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "RandomForest"
FOLDER: str = "rand_forest"

RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.5],
}


def _make_model(**kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # F_E
    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr, y_te = split_data(X_eng, y)

    # Hyperparameter tuning
    logger.info("  Hyperparameter tuning (F_E)...")
    best_model_e, best_params_e, best_cv_score = tune_hyperparameters(
        _make_model(), RF_PARAM_GRID, X_tr_e, y_tr, n_iter=30
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params_e, best_cv_score)

    # Outer CV (uncalibrated — calibration is post-hoc)
    cv_raw_e = compute_cv_metrics(best_model_e, X_tr_e, y_tr)
    logger.info("  [F_E] CV F1: %.3f ± %.3f", np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"]))

    # Fit uncalibrated for importances + pre-cal metrics
    base_e = _make_model(**{k.replace("clf__", ""): v for k, v in best_params_e.items()
                             if not k.startswith("scaler")})
    base_e.fit(X_tr_e, y_tr)
    test_m_uncal  = compute_test_metrics(base_e, X_te_e, y_te)
    brier_pre     = compute_brier_score(base_e, X_te_e, y_te)
    feat_imp      = base_e.feature_importances_
    logger.info("  [F_E] Uncalibrated Test F1=%.3f  Brier=%.4f", test_m_uncal["f1"], brier_pre)

    # Calibrate (pass UNFITTED model — cv=5 refits internally)
    cal_model_e  = calibrate_model(_make_model(**{k.replace("clf__", ""): v for k, v in best_params_e.items()
                                                   if not k.startswith("scaler")}), X_tr_e, y_tr)
    test_m_e     = compute_test_metrics(cal_model_e, X_te_e, y_te)
    y_proba_e    = get_propensity_scores(cal_model_e, X_te_e)
    brier_post   = compute_brier_score(cal_model_e, X_te_e, y_te)
    logger.info("  [F_E] Calibrated  Test F1=%.3f  Brier=%.4f  (Δ cal=%+.4f)",
                test_m_e["f1"], brier_post, brier_post - brier_pre)

    try:
        thr_info = select_threshold_pr_curve(cal_model_e, X_te_e, y_te)
    except ValueError as exc:
        logger.warning("  Threshold optimisation failed: %s", exc)
        thr_info = None

    # F_B ablation
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_data(X_base, y)
    best_model_b, _, _ = tune_hyperparameters(_make_model(), RF_PARAM_GRID, X_tr_b, y_tr_b, n_iter=30)
    cv_raw_b   = compute_cv_metrics(best_model_b, X_tr_b, y_tr_b)
    cal_b      = calibrate_model(_make_model(), X_tr_b, y_tr_b)
    test_m_b   = compute_test_metrics(cal_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=cal_model_e, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=cal_model_e.predict(X_te_e),
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_e,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        feature_importances=feat_imp,
        test_metrics_uncalibrated=test_m_uncal,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        threshold_info=thr_info,
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