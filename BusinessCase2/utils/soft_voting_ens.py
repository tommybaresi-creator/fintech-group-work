"""
Soft Voting Ensemble for both investment targets.


"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
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

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "SoftVotingEnsemble(LR+XGB+SVM)"
FOLDER: str = "soft_voting_ens"


class DynamicSolverLR(BaseEstimator, ClassifierMixin):
    """
    D3 fix: Logistic Regression wrapper that resolves the solver at fit time
    based on the penalty parameter. When a tuner sets penalty='l2', the solver
    automatically switches to 'lbfgs'. Previous code baked solver='liblinear'
    at construction time, making it inconsistent with tuned penalty='l2'.
    """

    def __init__(self, penalty="l1", C=1.0, max_iter=1000,
                 class_weight=None, random_state=42):
        self.penalty      = penalty
        self.C            = C
        self.max_iter     = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def _make_lr(self):
        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        return LogisticRegression(
            penalty=self.penalty, solver=solver, C=self.C,
            max_iter=self.max_iter, class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        self.lr_ = self._make_lr()
        self.lr_.fit(X, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.lr_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.lr_.predict_proba(X)

    def get_params(self, deep=True):
        return {
            "penalty": self.penalty, "C": self.C,
            "max_iter": self.max_iter, "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# Grid: only LR and SVM tuned. XGB handled internally by CalibratedClassifierCV.
LR_SVM_GRID = {
    "lr__clf__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "lr__clf__penalty": ["l1", "l2"],
    "svm__clf__C":      [0.1, 0.5, 1.0, 5.0, 10.0],
    "svm__clf__gamma":  ["scale", "auto", 0.01, 0.1],
}


def _make_ensemble(spw=1.0, class_weight=None) -> VotingClassifier:
    """
    Soft voting with per-estimator Pipelines.
    D3 fix: DynamicSolverLR resolves solver at fit time.
    D2 fix: estimator= (not base_estimator=), XGB_GRID removed.
    N2 fix: use_label_encoder removed.
    """
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DynamicSolverLR(class_weight=class_weight)),
    ])
    xgb_pipe = Pipeline([
        ("clf", CalibratedClassifierCV(
            estimator=XGBClassifier(
                scale_pos_weight=spw, random_state=42,
                eval_metric="logloss", verbosity=0,
            ),
            method="isotonic", cv=N_INNER_FOLDS,
        )),
    ])
    svm_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(
            kernel="rbf", probability=True,
            random_state=42, class_weight=class_weight,
        )),
    ])
    return VotingClassifier(
        estimators=[("lr", lr_pipe), ("xgb", xgb_pipe), ("svm", svm_pipe)],
        voting="soft",
    )


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y   = df[target_col]
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0
    cw  = "balanced"          if target_col == "IncomeInvestment" else None
    baseline = no_skill_brier(y)

    X_eng = build_features(df)
    X_tr_full, X_te, y_tr_full, y_te = split_data(X_eng, y)
    X_tr, X_val, y_tr, y_val = split_data(X_tr_full, y_tr_full, test_size=0.2)

    # F_E: tune LR+SVM components
    logger.info("  Tuning F_E ensemble (inner %d-fold)...", N_INNER_FOLDS)
    best_ens_e, best_params_e, best_cv_e = tune_hyperparameters(
        _make_ensemble(spw, cw), LR_SVM_GRID, X_tr_full, y_tr_full, n_iter=20
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params_e, best_cv_e)

    cv_raw_e = compute_cv_metrics(best_ens_e, X_tr_full, y_tr_full)
    logger.info("  [F_E] CV F1: %.3f ± %.3f",
                np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1))

    best_ens_e.fit(X_tr_full, y_tr_full)
    test_m_e  = compute_test_metrics(best_ens_e, X_te, y_te)
    y_proba_e = get_propensity_scores(best_ens_e, X_te)
    brier_e   = compute_brier_score(best_ens_e, X_te, y_te)
    logger.info("  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
                test_m_e["f1"], test_m_e["precision"], brier_e, baseline)

    try:
        thr_info   = select_threshold_on_val(best_ens_e, X_val, y_val)
        thr_test_m = evaluate_at_threshold(best_ens_e, X_te, y_te, thr_info["threshold"])
        logger.info("  Val threshold=%.3f → Test P=%.3f R=%.3f F1=%.3f",
                    thr_info["threshold"], thr_test_m["precision"],
                    thr_test_m["recall"], thr_test_m["f1"])
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    # F_B ablation — C2 fix: independently tuned
    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b, y_tr_b_full, y_te_b = split_data(X_base, y)
    X_tr_b, X_val_b, y_tr_b, y_val_b = split_data(X_tr_b_full, y_tr_b_full, test_size=0.2)

    logger.info("  Tuning F_B ablation ensemble...")
    best_ens_b, best_params_b, _ = tune_hyperparameters(
        _make_ensemble(spw, cw), LR_SVM_GRID, X_tr_b_full, y_tr_b_full, n_iter=20
    )
    cv_raw_b = compute_cv_metrics(best_ens_b, X_tr_b_full, y_tr_b_full)
    best_ens_b.fit(X_tr_b_full, y_tr_b_full)
    test_m_b = compute_test_metrics(best_ens_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)",
                test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=best_ens_e, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=best_ens_e.predict(X_te),
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_e,
        threshold_metrics=thr_test_m,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        brier_score=brier_e,
        no_skill_brier=baseline,
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