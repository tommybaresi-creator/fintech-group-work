"""
XGBoost classifier with SHAP feature attribution for IncomeInvestment and
AccumulationInvestment.

Theory
------
XGBoost is a gradient-boosted decision tree model that builds an additive
ensemble sequentially. Each new tree is trained to fit the negative gradient
(pseudo-residuals) of the loss function with respect to the current ensemble
prediction. This corresponds to a second-order Taylor approximation of the
objective function, enabling efficient and accurate optimisation.

Regularisation is applied through multiple mechanisms:
- lambda (L2 regularisation on leaf weights), which penalises large leaf
  values and improves numerical stability.
- gamma (minimum loss reduction required for a split), which prevents
  unnecessary tree growth.
- learning rate (eta), which scales the contribution of each tree. Smaller
  values improve generalisation but require more boosting rounds.

Class imbalance is handled using scale_pos_weight, which rescales gradients
for the positive class by the ratio n_neg / n_pos. This effectively adjusts the
loss landscape to compensate for skewed class distributions.

SHAP (SHapley Additive exPlanations) provides a theoretically grounded method
for feature attribution based on cooperative game theory. The Shapley value of
a feature represents its average marginal contribution across all possible
feature subsets, satisfying properties such as efficiency, symmetry, dummy,
and linearity.

TreeSHAP computes exact Shapley values for tree ensembles in polynomial time by
exploiting the structure of decision paths. This enables both global
(aggregate feature importance) and local (individual prediction explanation)
interpretability.

Implementation
--------------
This script applies XGBoost to both feature sets:
- F_B (baseline): primary representation, as tree-based models naturally learn
  interactions without engineered features.
- F_E (engineered): ablation set for comparison.

Hyperparameters are tuned independently using true nested cross-validation with
a 10-fold outer loop and a 3-fold inner RandomizedSearchCV loop (40 iterations
per fold), ensuring unbiased performance estimation.

SHAP values are computed on the uncalibrated XGBoost model because TreeSHAP
requires direct access to the native tree structure, which is not available
inside CalibratedClassifierCV wrappers.

Final probabilities and propensity scores are derived from the isotonic-
calibrated model to ensure well-calibrated outputs.

All results, including metrics, SHAP values, predictions, and model artifacts,
are serialized to:

    data/pickled_files/xgboost_shap/
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

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


def _make_model(spw: float = 1.0, **kwargs: Any) -> XGBClassifier:
    """
    Construct an XGBClassifier with a fixed random seed, logloss eval metric,
    silenced verbosity, and the supplied ``scale_pos_weight``.

    ``use_label_encoder`` has been removed (deprecated in XGBoost 1.6+).
    Additional keyword arguments are forwarded to the constructor, enabling the
    same factory to serve both default and tuned configurations.

    Parameters
    ----------
    spw : float
        ``scale_pos_weight`` — set to n_neg / n_pos for IncomeInvestment, 1.0
        for AccumulationInvestment.
    **kwargs
        Additional XGBClassifier constructor arguments.

    Returns
    -------
    XGBClassifier
    """
    return XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=spw, verbosity=0,
        **kwargs,
    )


def _make_model_factory(spw: float = 1.0) -> Callable[[], XGBClassifier]:
    """
    Return a zero-argument factory callable producing fresh XGBClassifier instances.

    Required by :func:`nested_cv_with_tuning`, which calls ``factory()`` once
    per outer fold to ensure each fold starts from an unfitted estimator.

    Parameters
    ----------
    spw : float
        ``scale_pos_weight`` forwarded to :func:`_make_model`.

    Returns
    -------
    Callable[[], XGBClassifier]
    """
    def factory() -> XGBClassifier:
        return _make_model(spw)
    return factory


def run_for_target(df: Any, target_col: str) -> dict:
    """
    Run the full XGBoost training, calibration, SHAP computation, CV, and
    serialisation workflow for a single target column.

    Steps:
    1. F_B (primary) — true nested CV, final model tuning, uncalibrated model
       for SHAP and pre-calibration Brier, calibrated model for metrics.
    2. MiFID II threshold selection on the validation split.
    3. Compute SHAP values via ``TreeExplainer`` on the uncalibrated base model
       (explains pre-calibration log-odds).
    4. F_E (ablation) — independently nested CV and final model.
    5. Assemble and save the result dict.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataset from :func:`load_data`.
    target_col : str
        One of the TARGETS constants.

    Returns
    -------
    dict
        Standardised result dict (see :func:`make_result_dict`).
    """
    logger.info("Target: %s", target_col)
    y   = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    baseline = no_skill_brier(y)
    logger.info("  No-skill Brier baseline: %.4f", baseline)

    # ---- F_B (primary for XGBoost — trees learn interactions natively) -----
    X_base = build_baseline_features(df)

    # carve out a validation set from training data for threshold selection
    X_tr_b_full, X_te_b, y_tr_full, y_te = split_data(X_base, y)
    X_tr_b, X_val_b, y_tr_b, y_val_b = split_data(X_tr_b_full, y_tr_full, test_size=0.2)

    # TRUE nested CV — HP tuning runs inside each outer fold
    logger.info("  Nested CV (F_B) — tuning inside each outer fold...")
    nested_b = nested_cv_with_tuning(
        _make_model_factory(spw), XGB_PARAM_GRID,
        X_tr_b_full, y_tr_full, n_iter=40,
    )
    cv_raw_b = nested_b["cv_metrics_raw"]
    logger.info(
        "  [F_B] Nested CV F1: %.3f ± %.3f",
        np.mean(cv_raw_b["f1"]), np.std(cv_raw_b["f1"], ddof=1),
    )

    # Final model: tune once on full X_tr_b for deployment
    logger.info("  Tuning final F_B model...")
    best_model_b, best_params_b, _ = tune_hyperparameters(
        _make_model(spw), XGB_PARAM_GRID, X_tr_b_full, y_tr_full, n_iter=40,
    )

    # Uncalibrated for SHAP + pre-cal metrics
    base_b = _make_model(spw, **{k: v for k, v in best_params_b.items()})
    base_b.fit(X_tr_b_full, y_tr_full)
    brier_pre = compute_brier_score(base_b, X_te_b, y_te)

    # Calibrate (unfitted model — cv=5 refits internally)
    cal_model_b = calibrate_model(
        _make_model(spw, **{k: v for k, v in best_params_b.items()}),
        X_tr_b_full, y_tr_full,
    )
    test_m_b    = compute_test_metrics(cal_model_b, X_te_b, y_te)
    y_proba_b   = get_propensity_scores(cal_model_b, X_te_b)
    brier_post  = compute_brier_score(cal_model_b, X_te_b, y_te)

    logger.info(
        "  [F_B] Brier: %.4f → %.4f (pre→post)  baseline=%.4f",
        brier_pre, brier_post, baseline,
    )
    logger.info("  [F_B] Test F1=%.3f  Precision=%.3f", test_m_b["f1"], test_m_b["precision"])

    # threshold selected on validation set
    try:
        thr_info   = select_threshold_on_val(cal_model_b, X_val_b, y_val_b)
        thr_test_m = evaluate_at_threshold(cal_model_b, X_te_b, y_te, thr_info["threshold"])
        logger.info(
            "  [F_B] Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
            thr_info["threshold"], thr_test_m["precision"],
            thr_test_m["recall"], thr_test_m["f1"],
        )
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    # SHAP on uncalibrated base (TreeExplainer requires tree structure)
    logger.info("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(base_b)
    shap_values = explainer.shap_values(X_te_b)

    # ---- F_E (ablation) — tuned independently ----------------------
    X_eng = build_features(df)
    X_tr_e_full, X_te_e, y_tr_e_full, y_te_e = split_data(X_eng, y)

    logger.info("  Nested CV (F_E ablation) — tuning inside each outer fold...")
    nested_e = nested_cv_with_tuning(
        _make_model_factory(spw), XGB_PARAM_GRID,
        X_tr_e_full, y_tr_e_full, n_iter=40,
    )
    cv_raw_e = nested_e["cv_metrics_raw"]

    best_model_e, best_params_e, _ = tune_hyperparameters(
        _make_model(spw), XGB_PARAM_GRID, X_tr_e_full, y_tr_e_full, n_iter=40,
    )
    cal_e    = calibrate_model(
        _make_model(spw, **{k: v for k, v in best_params_e.items()}),
        X_tr_e_full, y_tr_e_full,
    )
    test_m_e = compute_test_metrics(cal_e, X_te_e, y_te_e)
    logger.info(
        "  [F_E] Nested CV F1: %.3f ± %.3f  Test F1=%.3f  (ΔF_B−F_E=%+.3f)",
        np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1),
        test_m_e["f1"], test_m_b["f1"] - test_m_e["f1"],
    )

    result = make_result_dict(
        model=cal_model_b, scaler=None,
        cv_metrics_raw=cv_raw_b, test_metrics=test_m_b,
        y_test_true=y_te.values, y_test_pred=cal_model_b.predict(X_te_b),
        y_test_proba=y_proba_b,
        feature_names=BASELINE_FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_b,
        threshold_metrics=thr_test_m,
        ablation={
            "baseline": {
                "cv_metrics_raw":     cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics":       test_m_b,
            },
            "engineered": {
                "cv_metrics_raw":     cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics":       test_m_e,
            },
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


def main() -> None:
    """Entry point: load the raw dataset and train on every target in TARGETS."""
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()
