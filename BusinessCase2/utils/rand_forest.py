"""
Random Forest classifier for IncomeInvestment and AccumulationInvestment.

Theory
------
A Random Forest is a bagging ensemble composed of B decision trees, each
trained on an independently drawn bootstrap sample of the training data
(sampling with replacement).

At each split, only a random subset of sqrt(p) features is considered as
candidates, which decorrelates individual trees and reduces variance in the
ensemble.

The final prediction is obtained by averaging the per-tree posterior
probabilities (soft voting). This reduces variance compared to a single
decision tree, with variance decreasing as the number of trees increases and
as inter-tree correlation decreases.

Tree depth and minimum sample constraints control the bias-variance trade-off:
shallower trees increase bias but reduce variance, while deeper trees reduce
bias at the cost of higher variance.

Raw probability outputs from decision trees are typically poorly calibrated,
especially under class imbalance. To address this, isotonic regression
(cross-validated with cv=5) is applied post-hoc to produce calibrated
probabilities suitable for downstream decision-making in the MiFID II
recommendation setting.

Implementation
--------------
This script primarily uses the baseline feature set F_B (7 features) as Random
Forests naturally capture non-linear interactions. Including engineered
interaction features from F_E may introduce redundancy and dilute feature
importance interpretability due to correlated inputs.

The engineered feature set F_E is evaluated separately as an ablation study.

Two model versions are trained:
- An uncalibrated Random Forest used to extract Gini-based feature importances
- A calibrated version wrapped with CalibratedClassifierCV used for all
  predictive performance metrics and probability estimates

Because calibration modifies the probability space, feature importances are
only extracted from the uncalibrated model.

Hyperparameters are tuned using true nested cross-validation:
a 10-fold outer loop combined with a 3-fold inner RandomizedSearchCV loop
(30 iterations per fold). This ensures unbiased generalisation estimates.

All results, including metrics, predictions, probabilities, and model
artifacts, are serialised to:

    data/pickled_files/rand_forest/
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

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
    compute_test_metrics,
    evaluate_at_threshold,
    get_propensity_scores,
    load_data,
    make_result_dict,
    nested_cv_with_tuning,
    no_skill_brier,
    save_result,
    select_threshold_on_val,
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


def _make_model(**kwargs: Any) -> RandomForestClassifier:
    """
    Instantiate a RandomForestClassifier with a fixed random seed and full
    parallel execution.

    Any additional keyword arguments are forwarded directly to the constructor,
    allowing the same factory to serve both default and tuned configurations.

    Parameters
    ----------
    **kwargs
        Additional constructor arguments (e.g., n_estimators, max_depth).

    Returns
    -------
    RandomForestClassifier
    """
    return RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)


def _make_model_factory(**kwargs: Any) -> Callable[[], RandomForestClassifier]:
    """
    Return a zero-argument factory callable producing fresh RandomForest instances.

    Required by :func:`nested_cv_with_tuning`, which calls ``factory()`` once
    per outer fold to guarantee independence between folds.

    Parameters
    ----------
    **kwargs
        Constructor arguments forwarded to :func:`_make_model`.

    Returns
    -------
    Callable[[], RandomForestClassifier]
    """
    def factory() -> RandomForestClassifier:
        return _make_model(**kwargs)
    return factory


def run_for_target(df: Any, target_col: str) -> dict:
    """
    Run the full Random Forest training, calibration, CV, and serialisation
    workflow for a single target column.

    Steps:
    1. F_B (primary) — true nested CV, final model tuning, uncalibrated model
       for feature importances, calibrated model for metrics and propensity.
    2. MiFID II threshold selection on the validation split.
    3. F_E (ablation) — independently nested CV and final model.
    4. Assemble and save the result dict.

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
    y = df[target_col]
    baseline = no_skill_brier(y)
    logger.info("  No-skill Brier baseline: %.4f", baseline)

    # ---- F_B (primary) ----------------------------------------------------
    X_base = build_baseline_features(df)

    # validation split for threshold selection
    X_tr_b_full, X_te_b, y_tr_full, y_te = split_data(X_base, y)
    X_tr_b, X_val_b, y_tr_b, y_val_b = split_data(X_tr_b_full, y_tr_full, test_size=0.2)

    # true nested CV
    logger.info("  Nested CV (F_B primary)...")
    nested_b = nested_cv_with_tuning(
        _make_model_factory(), RF_PARAM_GRID,
        X_tr_b_full, y_tr_full, n_iter=30,
    )
    cv_raw_b = nested_b["cv_metrics_raw"]
    logger.info(
        "  [F_B] Nested CV F1: %.3f ± %.3f",
        np.mean(cv_raw_b["f1"]), np.std(cv_raw_b["f1"], ddof=1),
    )

    # Final model: tune on full training set
    best_model_b, best_params_b, _ = tune_hyperparameters(
        _make_model(), RF_PARAM_GRID, X_tr_b_full, y_tr_full, n_iter=30,
    )

    # Uncalibrated for importances + pre-cal Brier
    base_b = _make_model(**{k: v for k, v in best_params_b.items()})
    base_b.fit(X_tr_b_full, y_tr_full)
    feat_imp  = base_b.feature_importances_
    brier_pre = compute_brier_score(base_b, X_te_b, y_te)

    # Calibrate (unfitted model)
    cal_model_b = calibrate_model(
        _make_model(**{k: v for k, v in best_params_b.items()}),
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

    # threshold on validation set
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

    # ---- F_E (ablation) — tuned independently ----------------------
    X_eng = build_features(df)
    X_tr_e_full, X_te_e, y_tr_e_full, y_te_e = split_data(X_eng, y)

    logger.info("  Nested CV (F_E ablation)...")
    nested_e = nested_cv_with_tuning(
        _make_model_factory(), RF_PARAM_GRID,
        X_tr_e_full, y_tr_e_full, n_iter=30,
    )
    cv_raw_e = nested_e["cv_metrics_raw"]

    best_model_e, best_params_e, _ = tune_hyperparameters(
        _make_model(), RF_PARAM_GRID, X_tr_e_full, y_tr_e_full, n_iter=30,
    )
    cal_e    = calibrate_model(
        _make_model(**{k: v for k, v in best_params_e.items()}),
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
        feature_names=BASELINE_FEATURE_NAMES,   # F_B is primary
        target_name=target_col,
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
        feature_importances=feat_imp,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        no_skill_brier=baseline,
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
