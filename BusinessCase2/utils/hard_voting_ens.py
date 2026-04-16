"""
Hard Voting Ensemble Classifier for Investment Propensity Prediction.

Theory
------
Hard voting (majority voting) assigns the class predicted by the majority of
member classifiers, ignoring individual probability estimates. This approach
is robust to poorly calibrated probabilities because the final decision depends
only on the predicted labels. However, it sacrifices the variance-reduction
benefits of probability averaging available in soft voting. For binary
classification with three classifiers, at least two must agree to determine
the final prediction. Under class imbalance, hard voting may bias predictions
toward the majority class due to the absence of threshold adjustment.

This ensemble combines Logistic Regression (LR), Extreme Gradient Boosting
(XGB), and Support Vector Machines (SVM), enabling a controlled comparison
with its soft-voting counterpart to isolate the impact of vote aggregation.

Implementation
--------------
This script trains a `VotingClassifier` with `voting="hard"` on two feature
sets: engineered features (F_E) and baseline features (F_B) for ablation
analysis. Logistic Regression and SVM hyperparameters are tuned using
cross-validation. XGBoost is calibrated to enable probability-based metrics
such as the Brier score and MiFID II threshold optimization. Results are
serialized to `data/pickled_files/hard_voting_ens/`.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocessing import (  
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_INNER_FOLDS,
    TARGETS,
    build_baseline_features,
    build_features,
    compute_brier_score,
    compute_cv_metrics,
    compute_test_metrics,
    evaluate_at_threshold,
    get_propensity_scores,
    load_data,
    make_result_dict,
    no_skill_brier,
    save_result,
    scale_pos_weight,
    select_threshold_on_val,
    split_data,
    summarise_cv,
    tune_hyperparameters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME: str = "HardVotingEnsemble(LR+XGB+SVM)"
FOLDER: str = "hard_voting_ens"

COMBINED_GRID: Dict[str, list] = {
    "lr__clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "lr__clf__penalty": ["l1", "l2"],
    "svm__clf__C": [0.1, 0.5, 1.0, 5.0, 10.0],
    "svm__clf__gamma": ["scale", "auto", 0.01, 0.1],
}

class DynamicSolverLR(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression wrapper with dynamic solver selection.

    The solver is chosen at fit time based on the penalty parameter:
    - ``liblinear`` for L1 regularization.
    - ``lbfgs`` for L2 regularization.

    This ensures compatibility with hyperparameter tuning via
    ``RandomizedSearchCV``.
    """

    def __init__(
        self,
        penalty: str = "l1",
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def _make_lr(self) -> LogisticRegression:
        """Instantiate the underlying Logistic Regression model."""
        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        return LogisticRegression(
            penalty=self.penalty,
            solver=solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> "DynamicSolverLR":
        """Fit the internal Logistic Regression model."""
        self.lr_ = self._make_lr()
        self.lr_.fit(X, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, "lr_")
        return self.lr_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, "lr_")
        return self.lr_.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return estimator parameters for compatibility with sklearn."""
        return {
            "penalty": self.penalty,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "DynamicSolverLR":
        """Set estimator parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def _make_ensemble(
    spw: float = 1.0,
    class_weight: Optional[str] = None,
) -> VotingClassifier:
    """
    Construct the hard voting ensemble (LR + XGB + SVM).

    Parameters
    ----------
    spw : float, default=1.0
        Scale positive weight for XGBoost to address class imbalance.
    class_weight : str or None, default=None
        Class weighting for Logistic Regression and SVM.

    Returns
    -------
    VotingClassifier
        Configured hard voting ensemble.
    """
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", DynamicSolverLR(class_weight=class_weight)),
        ]
    )

    xgb_pipe = Pipeline(
        steps=[
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=XGBClassifier(
                        scale_pos_weight=spw,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    ),
                    method="isotonic",
                    cv=N_INNER_FOLDS,
                ),
            )
        ]
    )

    svm_pipe = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    return VotingClassifier(
        estimators=[
            ("lr", lr_pipe),
            ("xgb", xgb_pipe),
            ("svm", svm_pipe),
        ],
        voting="hard",
    )

def run_for_target(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """
    Execute the full training and evaluation pipeline for a given target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_col : str
        Target variable name.

    Returns
    -------
    Dict[str, Any]
        Standardized result dictionary.
    """
    logger.info("Target: %s", target_col)

    y = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    cw = "balanced" if target_col == "IncomeInvestment" else None
    baseline_brier = no_skill_brier(y)

    # ---------------------- Engineered Features (F_E) ---------------------- #
    X_eng = build_features(df)
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_eng, y)
    X_tr, X_val, y_tr, y_val = split_data(
        X_tr_full, y_tr_full, test_size=0.2
    )

    best_ens_e, best_params_e, best_cv_e = tune_hyperparameters(
        _make_ensemble(spw, cw),
        COMBINED_GRID,
        X_tr_full,
        y_tr_full,
        n_iter=20,
    )

    logger.info(
        "Best params: %s (inner CV F1=%.3f)",
        best_params_e,
        best_cv_e,
    )

    cv_raw_e = compute_cv_metrics(best_ens_e, X_tr_full, y_tr_full)
    best_ens_e.fit(X_tr_full, y_tr_full)
    test_metrics_e = compute_test_metrics(best_ens_e, X_te, y_te)

    # Probability-based metrics
    try:
        y_proba = get_propensity_scores(best_ens_e, X_te)
        brier_score = compute_brier_score(best_ens_e, X_te, y_te)
    except ValueError:
        y_proba = None
        brier_score = None

    # Threshold optimization
    try:
        thr_info = select_threshold_on_val(best_ens_e, X_val, y_val)
        thr_test_metrics = evaluate_at_threshold(
            best_ens_e, X_te, y_te, thr_info["threshold"]
        )
    except (ValueError, AttributeError):
        thr_info = None
        thr_test_metrics = None

    # ---------------------- Baseline Features (F_B) ------------------------- #
    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b, y_tr_b_full, y_te_b = split_data(X_base, y)

    best_ens_b, _, _ = tune_hyperparameters(
        _make_ensemble(spw, cw),
        COMBINED_GRID,
        X_tr_b_full,
        y_tr_b_full,
        n_iter=20,
    )

    cv_raw_b = compute_cv_metrics(best_ens_b, X_tr_b_full, y_tr_b_full)
    best_ens_b.fit(X_tr_b_full, y_tr_b_full)
    test_metrics_b = compute_test_metrics(best_ens_b, X_te_b, y_te_b)

    # ---------------------- Result Serialization ---------------------------- #
    result = make_result_dict(
        model=best_ens_e,
        scaler=None,
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_metrics_e,
        y_test_true=y_te.values,
        y_test_pred=best_ens_e.predict(X_te),
        y_test_proba=y_proba,
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        best_params=best_params_e,
        threshold_metrics=thr_test_metrics,
        ablation={
            "engineered": {
                "cv_metrics_raw": cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics": test_metrics_e,
            },
            "baseline": {
                "cv_metrics_raw": cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics": test_metrics_b,
            },
        },
        brier_score=brier_score,
        no_skill_brier=baseline_brier,
    )

    path = save_result(result, FOLDER, target_col)
    logger.info("Saved results to: %s", path)

    return result

def main() -> None:
    """Load the dataset and run the ensemble for all targets."""
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()