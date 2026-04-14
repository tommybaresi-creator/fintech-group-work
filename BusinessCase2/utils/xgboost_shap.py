"""
XGBoost with SHAP for both investment targets.

Fixes applied:
- Hyperparameter tuning via RandomizedSearchCV
- Both uncalibrated AND calibrated test metrics stored (diagnose calibration effect)
- y_test_proba stored
- Output saved as .pkl
- scale_pos_weight applied to IncomeInvestment only
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import shap
from xgboost import XGBClassifier

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
    scale_pos_weight,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "XGBoost"
FOLDER: str = "xgboost_shap"

XGB_PARAM_GRID = {
    "n_estimators":   [100, 200, 300, 500],
    "max_depth":      [3, 4, 5, 6, 7, 8],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample":      [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":           [0, 0.1, 0.2, 0.5],
}


def _make_model(spw=1.0, **kwargs) -> XGBClassifier:
    return XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=spw, verbosity=0,
        use_label_encoder=False, **kwargs
    )


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y   = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0

    # F_B primary
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr, y_te = split_data(X_base, y)

    # Hyperparameter tuning
    logger.info("  Hyperparameter tuning (F_B)...")
    best_model_b, best_params_b, best_cv_score = tune_hyperparameters(
        _make_model(spw), XGB_PARAM_GRID, X_tr_b, y_tr, n_iter=40
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params_b, best_cv_score)

    # Outer CV (uncalibrated)
    cv_raw_b = compute_cv_metrics(best_model_b, X_tr_b, y_tr)
    logger.info("  [F_B] CV F1: %.3f ± %.3f", np.mean(cv_raw_b["f1"]), np.std(cv_raw_b["f1"]))

    # Fit uncalibrated for SHAP + pre-cal metrics
    base_b = _make_model(spw, **{k: v for k, v in best_params_b.items()})
    base_b.fit(X_tr_b, y_tr)
    test_m_uncal = compute_test_metrics(base_b, X_te_b, y_te)
    brier_pre    = compute_brier_score(base_b, X_te_b, y_te)
    logger.info("  [F_B] Uncalibrated Test F1=%.3f  Brier=%.4f", test_m_uncal["f1"], brier_pre)

    # Calibrate (UNFITTED model — cv=5 refits internally)
    cal_model_b = calibrate_model(_make_model(spw, **{k: v for k, v in best_params_b.items()}), X_tr_b, y_tr)
    test_m_b    = compute_test_metrics(cal_model_b, X_te_b, y_te)
    y_proba_b   = get_propensity_scores(cal_model_b, X_te_b)
    brier_post  = compute_brier_score(cal_model_b, X_te_b, y_te)
    logger.info("  [F_B] Calibrated  Test F1=%.3f  Brier=%.4f  (Δ cal=%+.4f)",
                test_m_b["f1"], brier_post, brier_post - brier_pre)

    try:
        thr_info = select_threshold_pr_curve(cal_model_b, X_te_b, y_te)
    except ValueError as exc:
        logger.warning("  Threshold optimisation failed: %s", exc)
        thr_info = None

    # SHAP on uncalibrated base (TreeExplainer requires tree structure)
    logger.info("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(base_b)
    shap_values = explainer.shap_values(X_te_b)

    # F_E ablation
    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr_e, y_te_e = split_data(X_eng, y)
    best_model_e, _, _ = tune_hyperparameters(_make_model(spw), XGB_PARAM_GRID, X_tr_e, y_tr_e, n_iter=40)
    cv_raw_e   = compute_cv_metrics(best_model_e, X_tr_e, y_tr_e)
    cal_e      = calibrate_model(_make_model(spw), X_tr_e, y_tr_e)
    test_m_e   = compute_test_metrics(cal_e, X_te_e, y_te_e)
    logger.info("  [F_E] Test F1=%.3f  (ΔF_B−F_E=%+.3f)", test_m_e["f1"], test_m_b["f1"] - test_m_e["f1"])

    result = make_result_dict(
        model=cal_model_b, scaler=None,
        cv_metrics_raw=cv_raw_b, test_metrics=test_m_b,
        y_test_true=y_te.values, y_test_pred=cal_model_b.predict(X_te_b),
        y_test_proba=y_proba_b,
        feature_names=BASELINE_FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_b,
        ablation={
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
        },
        shap_explainer=explainer,
        shap_values=shap_values,
        shap_test_X=X_te_b,
        feature_importances=base_b.feature_importances_,
        test_metrics_uncalibrated=test_m_uncal,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        threshold_info=thr_info,
        scale_pos_weight_used=spw,
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