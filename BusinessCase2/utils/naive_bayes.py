"""
Gaussian Naive Bayes classifier for IncomeInvestment and
AccumulationInvestment.

Theory
------
Naive Bayes models the joint probability of the class label and the feature
vector as:

    P(y, x) = P(y) * ∏_i P(x_i | y)

by assuming conditional independence among all features given the class label.
The Gaussian variant further assumes that each feature conditioned on the class
follows a normal distribution:

    P(x_i | y) ~ N(μ_{iy}, σ²_{iy})

where μ_{iy} and σ²_{iy} are the class-specific mean and variance estimated
from the training data. Classification is performed by selecting the class
with the highest posterior probability P(y | x) via Bayes' theorem.

Additive variance smoothing, controlled by the `var_smoothing` hyperparameter,
adds a small constant to the per-class variances. This prevents numerical
instabilities caused by near-zero variance features and acts as a form of
regularisation.

The model serves as an interpretable probabilistic baseline. The baseline
feature set F_B includes variables such as Gender (binary {0, 1}) and
FamilyMembers (discrete integer), which formally violate the Gaussian
assumption. This mismatch can degrade probability calibration and predictive
performance; however, it is intentionally retained for comparison purposes.

Implementation
--------------
This script operates exclusively on the baseline feature set F_B, reflecting
the known distributional limitations of Gaussian Naive Bayes on mixed-type
features. The `var_smoothing` hyperparameter is optimised over a log-spaced
grid of 30 values using RandomizedSearchCV.

Model performance is then evaluated using stratified 10-fold cross-validation
with the tuned hyperparameter. Because Gaussian Naive Bayes has only a single
hyperparameter and is computationally inexpensive, nested cross-validation is
not employed, as it would provide minimal additional benefit compared to a
simpler flat cross-validation approach.

For regulatory alignment, a MiFID II precision-constrained decision threshold
is selected on a held-out validation split, provided that the
validation-set precision satisfies the required constraint.

All results, including cross-validation metrics, test performance, predictions,
probabilities, selected hyperparameters, and threshold-based evaluations, are
serialised to:

    data/pickled_files/naive_bayes/
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

# GNB assumes continuous Gaussian features.  F_B contains Gender
# (binary, {0, 1}) and FamilyMembers (discrete integer).  These violate the
# distributional assumption.  The consequence is inflated variance estimates
# (hence the large optimal var_smoothing for AccumulationInvestment) and
# unreliable probability outputs (Brier near no-skill baseline).  This is a
# known limitation retained for interpretable baseline comparison.

GNB_PARAM_GRID = {
    "var_smoothing": np.logspace(-12, -1, 30),
}


def run_for_target(df: object, target_col: str) -> dict:
    """
    Run the full Gaussian NB training, CV, evaluation, and serialisation
    workflow for a single target column.

    Operates on F_B only.  Tunes ``var_smoothing`` via RandomizedSearchCV,
    then runs standard 10-fold CV with the tuned model, evaluates at the
    default 0.5 threshold and at the MiFID II precision-constrained threshold
    (selected on the validation split), and saves the result dict to disk.

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

    X_base = build_baseline_features(df)

    # validation split for threshold
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_base, y)
    X_tr, X_val, y_tr, y_val = split_data(X_tr_full, y_tr_full, test_size=0.2)

    best_model, best_params, _ = tune_hyperparameters(
        GaussianNB(), GNB_PARAM_GRID, X_tr_full, y_tr_full, n_iter=20,
    )
    logger.info("  Best var_smoothing=%.2e  (inner CV F1 via tuning)", best_params["var_smoothing"])

    # Note: GNB is simple enough that nested CV adds minimal value here.
    # We run standard 10-fold CV with the tuned model.
    cv_raw = compute_cv_metrics(best_model, X_tr_full, y_tr_full)
    logger.info(
        "  [F_B] CV F1: %.3f ± %.3f",
        np.mean(cv_raw["f1"]), np.std(cv_raw["f1"], ddof=1),
    )

    best_model.fit(X_tr_full, y_tr_full)
    test_m  = compute_test_metrics(best_model, X_te, y_te)
    y_proba = get_propensity_scores(best_model, X_te)
    brier   = compute_brier_score(best_model, X_te, y_te)
    logger.info(
        "  [F_B] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
        test_m["f1"], test_m["precision"], brier, baseline,
    )

    try:
        thr_info   = select_threshold_on_val(best_model, X_val, y_val)
        thr_test_m = evaluate_at_threshold(best_model, X_te, y_te, thr_info["threshold"])
        logger.info(
            "  Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
            thr_info["threshold"], thr_test_m["precision"],
            thr_test_m["recall"], thr_test_m["f1"],
        )
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


def main() -> None:
    """Entry point: load the raw dataset and train on every target in TARGETS."""
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()
