"""
XGBoost with SHAP for both investment targets.


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
    compute_test_metrics,
    evaluate_at_threshold,
    get_propensity_scores,
    load_data,
    make_result_dict,
    nested_cv_with_tuning,
    no_skill_brier,
    save_result,
    scale_pos_weight,
    select_threshold_on_val,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "XGBoost"
FOLDER: str = "xgboost_shap"

XGB_PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.1, 0.2, 0.5],
}


def _make_model(spw=1.0, **kwargs) -> XGBClassifier:
    # N2 fix: use_label_encoder removed
    return XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=spw, verbosity=0,
        **kwargs
    )


def _make_model_factory(spw=1.0):
    """Returns a factory callable for nested_cv_with_tuning."""
    def factory():
        return _make_model(spw)
    return factory


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y   = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    baseline = no_skill_brier(y)
    logger.info("  No-skill Brier baseline: %.4f", baseline)

    # ---- F_B (primary for XGBoost — trees learn interactions natively) -----
    X_base = build_baseline_features(df)

    # C1 fix: carve out a validation set from training data for threshold selection
    X_tr_b_full, X_te_b, y_tr_full, y_te = split_data(X_base, y)
    X_tr_b, X_val_b, y_tr_b, y_val_b = split_data(X_tr_b_full, y_tr_full, test_size=0.2)

    # C4 fix: TRUE nested CV — HP tuning runs inside each outer fold
    logger.info("  Nested CV (F_B) — tuning inside each outer fold...")
    nested_b = nested_cv_with_tuning(
        _make_model_factory(spw), XGB_PARAM_GRID,
        X_tr_b_full, y_tr_full, n_iter=40
    )
    cv_raw_b = nested_b["cv_metrics_raw"]
    logger.info("  [F_B] Nested CV F1: %.3f ± %.3f",
                np.mean(cv_raw_b["f1"]), np.std(cv_raw_b["f1"], ddof=1))

    # Final model: tune once on full X_tr_b for deployment
    logger.info("  Tuning final F_B model...")
    best_model_b, best_params_b, _ = tune_hyperparameters(
        _make_model(spw), XGB_PARAM_GRID, X_tr_b_full, y_tr_full, n_iter=40
    )

    # Uncalibrated for SHAP + pre-cal metrics
    base_b = _make_model(spw, **{k: v for k, v in best_params_b.items()})
    base_b.fit(X_tr_b_full, y_tr_full)
    brier_pre = compute_brier_score(base_b, X_te_b, y_te)

    # Calibrate (unfitted model — cv=5 refits internally)
    cal_model_b = calibrate_model(_make_model(spw, **{k: v for k, v in best_params_b.items()}), X_tr_b_full, y_tr_full)
    test_m_b    = compute_test_metrics(cal_model_b, X_te_b, y_te)
    y_proba_b   = get_propensity_scores(cal_model_b, X_te_b)
    brier_post  = compute_brier_score(cal_model_b, X_te_b, y_te)

    logger.info("  [F_B] Brier: %.4f → %.4f (pre→post)  baseline=%.4f",
                brier_pre, brier_post, baseline)
    logger.info("  [F_B] Test F1=%.3f  Precision=%.3f", test_m_b["f1"], test_m_b["precision"])

    # C1 fix: threshold selected on validation set
    try:
        thr_info = select_threshold_on_val(cal_model_b, X_val_b, y_val_b)
        thr_test_m = evaluate_at_threshold(cal_model_b, X_te_b, y_te, thr_info["threshold"])
        logger.info("  [F_B] Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
                    thr_info["threshold"], thr_test_m["precision"],
                    thr_test_m["recall"], thr_test_m["f1"])
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    # SHAP on uncalibrated base (TreeExplainer requires tree structure)
    logger.info("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(base_b)
    shap_values = explainer.shap_values(X_te_b)

    # ---- F_E (ablation) — C2 fix: tuned independently ----------------------
    X_eng = build_features(df)
    X_tr_e_full, X_te_e, y_tr_e_full, y_te_e = split_data(X_eng, y)

    logger.info("  Nested CV (F_E ablation) — tuning inside each outer fold...")
    nested_e = nested_cv_with_tuning(
        _make_model_factory(spw), XGB_PARAM_GRID,
        X_tr_e_full, y_tr_e_full, n_iter=40
    )
    cv_raw_e = nested_e["cv_metrics_raw"]

    best_model_e, best_params_e, _ = tune_hyperparameters(
        _make_model(spw), XGB_PARAM_GRID, X_tr_e_full, y_tr_e_full, n_iter=40
    )
    cal_e   = calibrate_model(_make_model(spw, **{k: v for k, v in best_params_e.items()}), X_tr_e_full, y_tr_e_full)
    test_m_e = compute_test_metrics(cal_e, X_te_e, y_te_e)
    logger.info("  [F_E] Nested CV F1: %.3f ± %.3f  Test F1=%.3f  (ΔF_B−F_E=%+.3f)",
                np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1),
                test_m_e["f1"], test_m_b["f1"] - test_m_e["f1"])

    result = make_result_dict(
        model=cal_model_b, scaler=None,
        cv_metrics_raw=cv_raw_b, test_metrics=test_m_b,
        y_test_true=y_te.values, y_test_pred=cal_model_b.predict(X_te_b),
        y_test_proba=y_proba_b,
        feature_names=BASELINE_FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_b,
        threshold_metrics=thr_test_m,
        ablation={
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
        },
        shap_explainer=explainer,
        shap_values=shap_values,
        shap_test_X=X_te_b,
        feature_importances=base_b.feature_importances_,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        no_skill_brier=baseline,
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