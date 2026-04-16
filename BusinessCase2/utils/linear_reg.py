"""
Logistic Regression Classifier for IncomeInvestment and AccumulationInvestment.

Theory
------
Logistic Regression is a probabilistic linear classifier that models the
log-odds of the positive class as a linear combination of the input features:

    log(p / (1 - p)) = w^T x + b

where `p` is the probability of the positive class, `w` is the vector of model
coefficients, and `b` is the intercept. The parameters are estimated by
minimising the negative log-likelihood, also known as the cross-entropy loss.

Regularisation is applied to prevent overfitting and improve generalisation.
Two forms are considered:

- L1 (Lasso) regularisation: Encourages sparsity in the coefficient vector,
  effectively performing feature selection by shrinking some coefficients
  exactly to zero.
- L2 (Ridge) regularisation: Penalises large coefficients by shrinking them
  uniformly towards zero, improving numerical stability and reducing variance.

The strength of regularisation is controlled by the inverse parameter `C`.
Smaller values of `C` correspond to stronger regularisation.

Because Logistic Regression is sensitive to the scale of the input features,
all predictors are standardised to zero mean and unit variance using a
StandardScaler within each cross-validation fold to avoid data leakage.

For the imbalanced IncomeInvestment target (approximately 38% positive
instances), the option `class_weight='balanced'` is employed. This re-weights
the loss function inversely proportional to class frequencies, mitigating bias
towards the majority class and improving recall of the minority class.

Implementation
--------------
This script implements a Pipeline consisting of StandardScaler followed by
LogisticRegression for predicting both IncomeInvestment and
AccumulationInvestment.

Two feature configurations are evaluated:

- F_E (Engineered Features): The primary feature set containing domain-
  specific engineered predictors.
- F_B (Baseline Features): A reduced set used for ablation analysis to
  quantify the added value of feature engineering.

Hyperparameters (C, penalty, and max_iter) are optimised using
RandomizedSearchCV within a true nested cross-validation framework
(10 outer folds and 3 inner folds). This design ensures that fold-level F1
scores remain unbiased and suitable for downstream statistical comparisons,
such as the Wilcoxon signed-rank test.

After hyperparameter tuning, the final model is refitted on the full training
partition and evaluated on a held-out test set. Performance is assessed both:

1. At the default probability threshold (0.5).
2. At a MiFID II precision-constrained threshold, selected on a validation
   split to satisfy regulatory requirements while maximising recall.

Both F_E and F_B configurations are tuned and evaluated independently, ensuring a fair comparison between the engineered and baseline
feature sets.

All results—including cross-validation metrics, test performance,
predictions, probabilities, selected hyperparameters, and threshold-based
evaluations—are serialised to:

    data/pickled_files/linear_reg/

These outputs are designed for seamless integration with the
show_results.py module for reporting and analysis.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    build_features,
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

MODEL_NAME: str = "LogisticRegression"
FOLDER: str = "linear_reg"

LR_PARAM_GRID = {
    "clf__C":        [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "clf__penalty":  ["l1", "l2"],
    "clf__max_iter": [500, 1000, 2000],
}


def _make_pipeline(
    penalty: str = "l1",
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[str] = None,
) -> Pipeline:
    """
    Construct a StandardScaler + LogisticRegression Pipeline.

    The solver is determined at construction time from the penalty argument:
    ``liblinear`` for L1 (required by sklearn), ``lbfgs`` for L2.

    Parameters
    ----------
    penalty : str
        Regularisation type — 'l1' or 'l2'.
    C : float
        Inverse regularisation strength.
    max_iter : int
        Maximum number of solver iterations.
    class_weight : str or None
        Pass 'balanced' to re-weight for class imbalance, or None.

    Returns
    -------
    Pipeline
        Unfitted scaler + classifier pipeline.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty, solver=solver, C=C,
            max_iter=max_iter, random_state=42,
            class_weight=class_weight,
        )),
    ])


def _make_pipeline_factory(class_weight: Optional[str] = None) -> Callable[[], Pipeline]:
    """
    Return a zero-argument factory callable that produces a fresh Pipeline.

    Required by :func:`nested_cv_with_tuning`, which calls ``factory()`` once
    per outer fold to ensure each fold starts from an unfitted estimator.

    Parameters
    ----------
    class_weight : str or None
        Forwarded to :func:`_make_pipeline`.

    Returns
    -------
    Callable[[], Pipeline]
    """
    def factory() -> Pipeline:
        return _make_pipeline(class_weight=class_weight)
    return factory


def run_for_target(df: Any, target_col: str) -> dict:
    """
    Run the full training, evaluation, and serialisation workflow for one target.

    Steps:
    1. Build F_E and perform a true nested CV to get unbiased fold-level F1.
    2. Tune a final model on the full F_E training set and evaluate on the test set.
    3. Select a MiFID II compliance threshold on the validation split.
    4. Independently repeat steps 1–2 for the F_B ablation.
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
    y  = df[target_col]
    cw = "balanced" if target_col == "IncomeInvestment" else None
    baseline = no_skill_brier(y)

    # F_E
    X_eng = build_features(df)
    X_tr_e_full, X_te_e, y_tr_full, y_te = split_data(X_eng, y)
    X_tr_e, X_val_e, y_tr_e, y_val_e = split_data(X_tr_e_full, y_tr_full, test_size=0.2)

    # nested CV
    logger.info("  Nested CV (F_E)...")
    nested_e = nested_cv_with_tuning(
        _make_pipeline_factory(cw), LR_PARAM_GRID,
        X_tr_e_full, y_tr_full, n_iter=20,
    )
    cv_raw_e = nested_e["cv_metrics_raw"]
    logger.info(
        "  [F_E] Nested CV F1: %.3f ± %.3f",
        np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1),
    )

    best_pipe_e, best_params_e, _ = tune_hyperparameters(
        _make_pipeline(class_weight=cw), LR_PARAM_GRID, X_tr_e_full, y_tr_full, n_iter=20,
    )
    best_pipe_e.fit(X_tr_e_full, y_tr_full)
    test_m_e  = compute_test_metrics(best_pipe_e, X_te_e, y_te)
    y_proba_e = get_propensity_scores(best_pipe_e, X_te_e)
    brier     = compute_brier_score(best_pipe_e, X_te_e, y_te)
    logger.info(
        "  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
        test_m_e["f1"], test_m_e["precision"], brier, baseline,
    )

    try:
        thr_info   = select_threshold_on_val(best_pipe_e, X_val_e, y_val_e)
        thr_test_m = evaluate_at_threshold(best_pipe_e, X_te_e, y_te, thr_info["threshold"])
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b, y_tr_b_full, y_te_b = split_data(X_base, y)

    logger.info("  Nested CV (F_B ablation)...")
    nested_b = nested_cv_with_tuning(
        _make_pipeline_factory(cw), LR_PARAM_GRID,
        X_tr_b_full, y_tr_b_full, n_iter=20,
    )
    cv_raw_b = nested_b["cv_metrics_raw"]
    best_pipe_b, _, _ = tune_hyperparameters(
        _make_pipeline(class_weight=cw), LR_PARAM_GRID, X_tr_b_full, y_tr_b_full, n_iter=20,
    )
    best_pipe_b.fit(X_tr_b_full, y_tr_b_full)
    test_m_b = compute_test_metrics(best_pipe_b, X_te_b, y_te_b)
    logger.info(
        "  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)",
        test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"],
    )

    result = make_result_dict(
        model=best_pipe_e, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=best_pipe_e.predict(X_te_e),
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_e,
        threshold_metrics=thr_test_m,
        ablation={
            "engineered": {
                "cv_metrics_raw":     cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics":       test_m_e,
            },
            "baseline": {
                "cv_metrics_raw":     cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics":       test_m_b,
            },
        },
        brier_score=brier,
        no_skill_brier=baseline,
        class_weight=cw,
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
