"""
Soft Voting Ensemble classifier for IncomeInvestment and AccumulationInvestment.

Theory
------
A soft voting ensemble combines the posterior class probabilities produced by
each base classifier and assigns the class with the highest average probability.

Unlike hard voting, which relies solely on discrete class labels, soft voting
preserves probabilistic information and is therefore more robust to individual
model overconfidence. It also reduces variance by averaging predictions across
models with different inductive biases.

The variance reduction effect arises from model diversity: each base learner
makes different types of errors, and averaging tends to cancel uncorrelated
noise while preserving shared signal. This typically improves generalisation
without increasing bias.

The ensemble consists of three complementary models:
- Logistic Regression: linear decision boundary, strong performance on
  well-scaled and approximately linear feature spaces.
- XGBoost: gradient-boosted decision trees that capture non-linear interactions
  in tabular data.
- Support Vector Machine (RBF kernel): margin-based classifier that performs
  well when classes are separable in a non-linear feature space.

Diversity is further encouraged through heterogeneous preprocessing:
StandardScaler is used for Logistic Regression, while MinMaxScaler is used for
SVM, ensuring that each model operates in a representation best suited to its
inductive bias.

Implementation
--------------
This script trains a three-member soft voting ensemble (LR + XGB + SVM) using
both engineered features F_E (primary) and baseline features F_B (ablation).

Hyperparameters for Logistic Regression and SVM are tuned using
RandomizedSearchCV. XGBoost is internally wrapped with CalibratedClassifierCV
to ensure stable probability outputs.

The Logistic Regression component resolves solver–penalty compatibility
dynamically via a helper function (solver_for_penalty), preventing invalid
configurations during hyperparameter search.

Both feature sets are tuned independently to ensure a fair comparison
between engineered and baseline representations.

All results are serialized to:

    data/pickled_files/soft_voting_ens/
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

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

MODEL_NAME: str = "SoftVotingEnsemble(LR+XGB+SVM)"
FOLDER: str = "soft_voting_ens"


def solver_for_penalty(penalty: str) -> str:
    """
    Return the correct sklearn solver for a given LR penalty.

    ``liblinear`` is used for L1 regularisation; ``lbfgs`` for L2.
    Baking ``solver='liblinear'`` at construction time makes it inconsistent
    with a tuned ``penalty='l2'``.  This function ensures the solver is always
    derived from the penalty, regardless of which value the grid search selects.
    """
    return "liblinear" if penalty == "l1" else "lbfgs"


# Grid: LR penalty is not searched to avoid solver mismatch during search;
# solver is resolved at construction via solver_for_penalty().
# After tuning, the final ensemble is rebuilt with the correct solver
# via _rebuild_from_params().
LR_SVM_GRID = {
    "lr__clf__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "svm__clf__C":      [0.1, 0.5, 1.0, 5.0, 10.0],
    "svm__clf__gamma":  ["scale", "auto", 0.01, 0.1],
}


def _make_ensemble(
    spw: float = 1.0,
    class_weight: Optional[str] = None,
    lr_penalty: str = "l1",
    lr_C: float = 1.0,
    svm_C: float = 1.0,
    svm_gamma: Any = "scale",
) -> VotingClassifier:
    """
    Construct a three-member soft ``VotingClassifier`` (LR + XGB + SVM).

    Each member is wrapped in its own ``Pipeline`` with an appropriate scaler.
    XGB is additionally wrapped in ``CalibratedClassifierCV`` for probability
    calibration.  Class-imbalance parameters (``spw`` and ``class_weight``) are
    forwarded to XGB and LR / SVM respectively.

    solver is resolved at construction from ``lr_penalty`` via
    ``solver_for_penalty()``, ensuring LR penalty and solver are always
    consistent.
    ``estimator=`` used (not the deprecated ``base_estimator=``).
    ``use_label_encoder`` removed from XGBClassifier.
    ``eval_metric`` removed (no ``eval_set`` passed — dead config).

    Parameters
    ----------
    spw : float
        ``scale_pos_weight`` for XGBClassifier.
    class_weight : str or None
        Class weighting for LR and SVM (``'balanced'`` or ``None``).
    lr_penalty : str
        LR regularisation penalty (``'l1'`` or ``'l2'``).
    lr_C : float
        LR inverse regularisation strength.
    svm_C : float
        SVM regularisation parameter.
    svm_gamma : str or float
        SVM kernel coefficient.

    Returns
    -------
    VotingClassifier
    """
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=lr_penalty,
            solver=solver_for_penalty(lr_penalty),
            C=lr_C,
            max_iter=1000,
            random_state=42,
            class_weight=class_weight,
        )),
    ])
    xgb_pipe = Pipeline([
        ("clf", CalibratedClassifierCV(
            estimator=XGBClassifier(
                scale_pos_weight=spw,
                random_state=42,
                verbosity=0,
            ),
            method="isotonic",
            cv=N_INNER_FOLDS,
        )),
    ])
    svm_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=svm_C,
            gamma=svm_gamma,
            probability=True,
            random_state=42,
            class_weight=class_weight,
        )),
    ])
    return VotingClassifier(
        estimators=[("lr", lr_pipe), ("xgb", xgb_pipe), ("svm", svm_pipe)],
        voting="soft",
    )


def _make_search_ensemble(
    spw: float = 1.0,
    class_weight: Optional[str] = None,
) -> VotingClassifier:
    """Return a default ensemble suitable for ``RandomizedSearchCV``."""
    return _make_ensemble(spw=spw, class_weight=class_weight)


def _rebuild_from_params(
    spw: float,
    class_weight: Optional[str],
    best_params: dict,
) -> VotingClassifier:
    """
    Reconstruct the ensemble with tuned hyperparameters.

    Called after ``tune_hyperparameters`` to produce the final fitted model
    with the correct solver derived from the tuned LR parameters.

    Parameters
    ----------
    spw : float
        ``scale_pos_weight`` for XGBClassifier.
    class_weight : str or None
        Class weighting passed through to LR and SVM.
    best_params : dict
        Best hyperparameters from ``tune_hyperparameters``.

    Returns
    -------
    VotingClassifier
        Unfitted ensemble ready for ``fit``.
    """
    return _make_ensemble(
        spw=spw,
        class_weight=class_weight,
        lr_C=best_params.get("lr__clf__C", 1.0),
        svm_C=best_params.get("svm__clf__C", 1.0),
        svm_gamma=best_params.get("svm__clf__gamma", "scale"),
    )


def _make_factory(
    spw: float,
    cw: Optional[str],
):
    """Return a zero-argument factory for ``nested_cv_with_tuning``."""
    def factory() -> VotingClassifier:
        return _make_search_ensemble(spw, cw)
    return factory


def run_for_target(df: Any, target_col: str) -> dict:
    """
    Run the full soft-voting ensemble training, CV, threshold selection, and
    serialisation workflow for a single target column.

    Steps:
    1. F_E (primary) — tune LR + SVM hyperparameters via inner CV, compute
       10-fold outer CV metrics (nested, unbiased), fit final
       model on ``X_tr_full``, evaluate at default and MiFID II thresholds
       (threshold selected on ``X_val``, not ``X_te``).
    2. F_B (ablation) — independently tune and evaluate.
    3. Assemble and save the result dict.

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
    cw  = "balanced"          if target_col == "IncomeInvestment" else None
    baseline = no_skill_brier(y)

    X_eng = build_features(df)
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_eng, y)

    # X_val is carved from X_tr_full for threshold selection only.
    # The final model is trained on all of X_tr_full; the threshold is selected
    # by evaluating that same final model on X_val — no distribution mismatch.
    _, X_val, _, y_val = split_data(X_tr_full, y_tr_full, test_size=0.2)

    # F_E: nested CV (HP tuning inside each outer fold)
    logger.info("  Nested CV (F_E)...")
    cv_raw_e = nested_cv_with_tuning(
        _make_factory(spw, cw), LR_SVM_GRID, X_tr_full, y_tr_full, n_iter=20,
    )["cv_metrics_raw"]
    logger.info(
        "  [F_E] CV F1: %.3f ± %.3f",
        np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1),
    )

    # Tune once on full X_tr_full for the final deployment model
    _, best_params_e, best_cv_e = tune_hyperparameters(
        _make_search_ensemble(spw, cw), LR_SVM_GRID, X_tr_full, y_tr_full, n_iter=20,
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params_e, best_cv_e)

    # Rebuild with correct solver from tuned params, then fit on full X_tr_full
    final_ens_e = _rebuild_from_params(spw, cw, best_params_e)
    final_ens_e.fit(X_tr_full, y_tr_full)

    test_m_e  = compute_test_metrics(final_ens_e, X_te, y_te)
    y_proba_e = get_propensity_scores(final_ens_e, X_te)
    brier_e   = compute_brier_score(final_ens_e, X_te, y_te)
    logger.info(
        "  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
        test_m_e["f1"], test_m_e["precision"], brier_e, baseline,
    )

    # threshold selected on X_val against the FINAL model
    thr_info, thr_test_m = None, None
    try:
        thr_info   = select_threshold_on_val(final_ens_e, X_val, y_val)
        thr_test_m = evaluate_at_threshold(final_ens_e, X_te, y_te, thr_info["threshold"])
        logger.info(
            "  Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
            thr_info["threshold"], thr_test_m["precision"],
            thr_test_m["recall"], thr_test_m["f1"],
        )
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)

    # F_B ablation — independently tuned
    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b, y_tr_b_full, y_te_b = split_data(X_base, y)

    logger.info("  Nested CV (F_B ablation)...")
    cv_raw_b = nested_cv_with_tuning(
        _make_factory(spw, cw), LR_SVM_GRID, X_tr_b_full, y_tr_b_full, n_iter=20,
    )["cv_metrics_raw"]

    _, best_params_b, _ = tune_hyperparameters(
        _make_search_ensemble(spw, cw), LR_SVM_GRID, X_tr_b_full, y_tr_b_full, n_iter=20,
    )
    final_ens_b = _rebuild_from_params(spw, cw, best_params_b)
    final_ens_b.fit(X_tr_b_full, y_tr_b_full)
    test_m_b = compute_test_metrics(final_ens_b, X_te_b, y_te_b)
    logger.info(
        "  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)",
        test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"],
    )

    result = make_result_dict(
        model=final_ens_e, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=final_ens_e.predict(X_te),
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
        brier_score=brier_e,
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