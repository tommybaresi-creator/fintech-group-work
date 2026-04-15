"""
Gaussian Naive Bayes for both investment targets.


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
    compute_test_metrics,
    evaluate_at_threshold,
    get_propensity_scores,
    load_data,
    make_result_dict,
    no_skill_brier,
    save_result,
    select_threshold_on_val,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "GaussianNB"
FOLDER: str = "naive_bayes"

# M4 note: GNB assumes continuous Gaussian features. F_B contains Gender
# (binary, {0,1}) and FamilyMembers (discrete integer). These violate the
# distributional assumption. The consequence is inflated variance estimates
# (hence the large optimal var_smoothing for AccumulationInvestment) and
# unreliable probability outputs (Brier near no-skill baseline). This is a
# known limitation retained for interpretable baseline comparison.

GNB_PARAM_GRID = {
    "var_smoothing": np.logspace(-12, -1, 30),
}


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]
    baseline = no_skill_brier(y)
    logger.info("  No-skill Brier baseline: %.4f", baseline)

    X_base = build_baseline_features(df)

    # C1 fix: validation split for threshold
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_base, y)
    X_tr, X_val, y_tr, y_val = split_data(X_tr_full, y_tr_full, test_size=0.2)

    best_model, best_params, _ = tune_hyperparameters(
        GaussianNB(), GNB_PARAM_GRID, X_tr_full, y_tr_full, n_iter=20
    )
    logger.info("  Best var_smoothing=%.2e  (inner CV F1 via tuning)", best_params["var_smoothing"])

    # Note: GNB is simple enough that nested CV adds minimal value here.
    # We run standard 10-fold CV with the tuned model.
    from utils.preprocessing import compute_cv_metrics
    cv_raw = compute_cv_metrics(best_model, X_tr_full, y_tr_full)
    logger.info("  [F_B] CV F1: %.3f ± %.3f",
                np.mean(cv_raw["f1"]), np.std(cv_raw["f1"], ddof=1))

    best_model.fit(X_tr_full, y_tr_full)
    test_m  = compute_test_metrics(best_model, X_te, y_te)
    y_proba = get_propensity_scores(best_model, X_te)
    brier   = compute_brier_score(best_model, X_te, y_te)
    logger.info("  [F_B] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
                test_m["f1"], test_m["precision"], brier, baseline)

    try:
        thr_info   = select_threshold_on_val(best_model, X_val, y_val)
        thr_test_m = evaluate_at_threshold(best_model, X_te, y_te, thr_info["threshold"])
        logger.info("  Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
                    thr_info["threshold"], thr_test_m["precision"],
                    thr_test_m["recall"], thr_test_m["f1"])
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    result = make_result_dict(
        model=best_model, scaler=None,
        cv_metrics_raw=cv_raw, test_metrics=test_m,
        y_test_true=y_te.values, y_test_pred=best_model.predict(X_te),
        y_test_proba=y_proba,
        feature_names=BASELINE_FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params,
        threshold_metrics=thr_test_m,
        ablation=None,
        brier_score=brier,
        no_skill_brier=baseline,
        gnb_discrete_feature_warning=(
            "F_B contains Gender (binary) and FamilyMembers (discrete integer). "
            "GNB's Gaussian assumption is violated for these features, contributing "
            "to poor calibration. Retained as interpretable baseline."
        ),
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