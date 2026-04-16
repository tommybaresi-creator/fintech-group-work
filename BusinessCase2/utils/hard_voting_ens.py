"""
Hard Voting Ensemble Classifier for Investment Propensity Prediction.

... (docstring unchanged) ...
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

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

warnings.filterwarnings("ignore")

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


class ClassifierPipeline(ClassifierMixin, Pipeline):
    """
    Pipeline subclass that is unambiguously recognised as a classifier by
    sklearn's ``is_classifier()`` check.

    ``VotingClassifier._validate_estimators()`` calls ``is_classifier(est)``,
    which reads the *class-level* ``_estimator_type`` attribute via MRO.
    Setting ``_estimator_type`` on a plain Pipeline *instance* is silently
    ignored by that check.  Mixing in ``ClassifierMixin`` at the class level
    ensures ``ClassifierPipeline._estimator_type == 'classifier'`` for every
    instance without any further monkey-patching.
    """
    pass


class DynamicSolverLR(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression wrapper with dynamic solver selection.

    Chooses ``liblinear`` for L1 regularisation and ``lbfgs`` for L2,
    since ``lbfgs`` does not support L1 and ``liblinear`` is single-threaded
    but more robust for small datasets.
    """

    _estimator_type = "classifier"

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
        self.lr_ = self._make_lr()
        self.lr_.fit(X, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "lr_")
        return self.lr_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "lr_")
        return self.lr_.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "penalty": self.penalty,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "DynamicSolverLR":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _make_ensemble(
    spw: float = 1.0,
    class_weight: Optional[str] = None,
) -> VotingClassifier:
    """
    Construct the hard voting ensemble (LR + XGB + SVM).

    Both LR and SVM sub-estimators are wrapped in ``ClassifierPipeline``
    (a ``ClassifierMixin`` + ``Pipeline`` subclass) so that
    ``VotingClassifier._validate_estimators()`` recognises them as classifiers
    via the class-level ``_estimator_type`` attribute.  Plain ``Pipeline``
    instances fail this check even when ``_estimator_type`` is patched at the
    instance level.
    """
    lr_pipe = ClassifierPipeline([
        ("scaler", StandardScaler()),
        ("clf", DynamicSolverLR(class_weight=class_weight)),
    ])

    xgb_clf = CalibratedClassifierCV(
        estimator=XGBClassifier(
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        method="isotonic",
        cv=N_INNER_FOLDS,
    )

    svm_pipe = ClassifierPipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight=class_weight,
        ))
    ])

    return VotingClassifier(
        estimators=[
            ("lr", lr_pipe),
            ("xgb", xgb_clf),
            ("svm", svm_pipe),
        ],
        voting="hard",
    )


def run_for_target(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    logger.info("Target: %s", target_col)

    y = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    baseline_brier = no_skill_brier(y)

    X_eng = build_features(df)
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_eng, y)
    X_tr, X_val, y_tr, y_val = split_data(X_tr_full, y_tr_full, test_size=0.2)

    best_ens_e, best_params_e, best_cv_e = tune_hyperparameters(
        _make_ensemble(spw, None),
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

    try:
        y_proba = get_propensity_scores(best_ens_e, X_te)
        brier_score = compute_brier_score(best_ens_e, X_te, y_te)
    except ValueError:
        y_proba = None
        brier_score = None

    try:
        thr_info = select_threshold_on_val(best_ens_e, X_val, y_val)
        thr_test_metrics = evaluate_at_threshold(
            best_ens_e, X_te, y_te, thr_info["threshold"]
        )
    except (ValueError, AttributeError):
        thr_info = None
        thr_test_metrics = None

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
        ablation={},
        brier_score=brier_score,
        no_skill_brier=baseline_brier,
    )

    path = save_result(result, FOLDER, target_col)
    logger.info("Saved results to: %s", path)

    return result


def main() -> None:
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()