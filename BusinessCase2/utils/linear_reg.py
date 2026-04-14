"""
Logistic Regression for both investment targets.

Fixes applied:
- StandardScaler inside Pipeline (CV leakage eliminated)
- Hyperparameter tuning via RandomizedSearchCV (inner 3-fold)
- y_test_proba stored in result dict
- Output saved as .pkl not .joblib
- Warnings suppressed
"""

import logging
import sys
import warnings
from pathlib import Path

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

MODEL_NAME: str = "LogisticRegression"
FOLDER: str = "linear_reg"

LR_PARAM_GRID = {
    "clf__C":         [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "clf__penalty":   ["l1", "l2"],
    "clf__max_iter":  [500, 1000, 2000],
}


def _make_pipeline(penalty="l1", C=1.0, max_iter=1000, class_weight=None) -> Pipeline:
    """Pipeline encapsulates scaler — refits per fold in CV automatically."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty, solver=solver, C=C,
            max_iter=max_iter, random_state=42,
            class_weight=class_weight,
        )),
    ])


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y  = df[target_col]
    cw = "balanced" if target_col == "IncomeInvestment" else None

    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr, y_te = split_data(X_eng, y)

    # Hyperparameter tuning (inner 3-fold CV on unscaled data via Pipeline)
    logger.info("  Hyperparameter tuning (F_E)...")
    base_pipe = _make_pipeline(class_weight=cw)
    best_pipe, best_params, best_cv_score = tune_hyperparameters(
        base_pipe, LR_PARAM_GRID, X_tr_e, y_tr, n_iter=20
    )
    logger.info("  Best params: %s  (inner CV F1=%.3f)", best_params, best_cv_score)

    # Outer CV with best configuration
    cv_raw_e = compute_cv_metrics(best_pipe, X_tr_e, y_tr)
    logger.info("  [F_E] CV F1: %.3f ± %.3f", np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"]))

    # Final model
    best_pipe.fit(X_tr_e, y_tr)
    test_m_e   = compute_test_metrics(best_pipe, X_te_e, y_te)
    y_proba_e  = get_propensity_scores(best_pipe, X_te_e)
    brier      = compute_brier_score(best_pipe, X_te_e, y_te)
    logger.info("  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f",
                test_m_e["f1"], test_m_e["precision"], brier)

    try:
        thr_info = select_threshold_pr_curve(best_pipe, X_te_e, y_te)
    except ValueError as exc:
        logger.warning("  Threshold optimisation failed: %s", exc)
        thr_info = None

    # Ablation: F_B
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_data(X_base, y)
    base_pipe_b = _make_pipeline(class_weight=cw)
    best_pipe_b, _, _ = tune_hyperparameters(base_pipe_b, LR_PARAM_GRID, X_tr_b, y_tr_b, n_iter=20)
    cv_raw_b  = compute_cv_metrics(best_pipe_b, X_tr_b, y_tr_b)
    best_pipe_b.fit(X_tr_b, y_tr_b)
    test_m_b  = compute_test_metrics(best_pipe_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=best_pipe, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=best_pipe.predict(X_te_e),
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        threshold_info=thr_info, brier_score=brier, class_weight=cw,
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