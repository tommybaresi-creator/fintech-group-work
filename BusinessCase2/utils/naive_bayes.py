"""
Gaussian Naive Bayes for both investment targets.

No scaling (likelihood ratio invariant to linear rescaling).
F_B only (Gaussian independence assumption already violated by base features).
Hyperparameter tuning: var_smoothing grid search.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
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

MODEL_NAME: str = "GaussianNB"
FOLDER: str = "naive_bayes"

GNB_PARAM_GRID = {
    "var_smoothing": np.logspace(-12, -1, 30),
}


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]

    X_base = build_baseline_features(df)
    X_tr, X_te, y_tr, y_te = split_data(X_base, y)

    # Hyperparameter tuning
    logger.info("  Hyperparameter tuning (var_smoothing)...")
    best_model, best_params, best_cv_score = tune_hyperparameters(
        GaussianNB(), GNB_PARAM_GRID, X_tr, y_tr, n_iter=20
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params, best_cv_score)

    cv_raw = compute_cv_metrics(best_model, X_tr, y_tr)
    logger.info("  [F_B] CV F1: %.3f ± %.3f", np.mean(cv_raw["f1"]), np.std(cv_raw["f1"]))

    best_model.fit(X_tr, y_tr)
    test_m  = compute_test_metrics(best_model, X_te, y_te)
    y_proba = get_propensity_scores(best_model, X_te)
    brier   = compute_brier_score(best_model, X_te, y_te)
    logger.info("  [F_B] Test F1=%.3f  Precision=%.3f  Brier=%.4f",
                test_m["f1"], test_m["precision"], brier)

    try:
        thr_info = select_threshold_pr_curve(best_model, X_te, y_te)
    except ValueError as exc:
        logger.warning("  Threshold optimisation failed: %s", exc)
        thr_info = None

    result = make_result_dict(
        model=best_model, scaler=None,
        cv_metrics_raw=cv_raw, test_metrics=test_m,
        y_test_true=y_te.values, y_test_pred=best_model.predict(X_te),
        y_test_proba=y_proba,
        feature_names=BASELINE_FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params,
        ablation=None, threshold_info=thr_info, brier_score=brier,
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