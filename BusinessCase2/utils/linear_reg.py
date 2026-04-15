"""
Logistic Regression for both investment targets.


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


def _make_pipeline(penalty="l1", C=1.0, max_iter=1000, class_weight=None) -> Pipeline:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty, solver=solver, C=C,
            max_iter=max_iter, random_state=42,
            class_weight=class_weight,
        )),
    ])


def _make_pipeline_factory(class_weight=None):
    def factory():
        return _make_pipeline(class_weight=class_weight)
    return factory


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y  = df[target_col]
    cw = "balanced" if target_col == "IncomeInvestment" else None
    baseline = no_skill_brier(y)

    # F_E
    X_eng = build_features(df)
    X_tr_e_full, X_te_e, y_tr_full, y_te = split_data(X_eng, y)
    X_tr_e, X_val_e, y_tr_e, y_val_e = split_data(X_tr_e_full, y_tr_full, test_size=0.2)

    # C4: nested CV
    logger.info("  Nested CV (F_E)...")
    nested_e = nested_cv_with_tuning(
        _make_pipeline_factory(cw), LR_PARAM_GRID,
        X_tr_e_full, y_tr_full, n_iter=20
    )
    cv_raw_e = nested_e["cv_metrics_raw"]
    logger.info("  [F_E] Nested CV F1: %.3f ± %.3f",
                np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"], ddof=1))

    best_pipe_e, best_params_e, _ = tune_hyperparameters(
        _make_pipeline(class_weight=cw), LR_PARAM_GRID, X_tr_e_full, y_tr_full, n_iter=20
    )
    best_pipe_e.fit(X_tr_e_full, y_tr_full)
    test_m_e  = compute_test_metrics(best_pipe_e, X_te_e, y_te)
    y_proba_e = get_propensity_scores(best_pipe_e, X_te_e)
    brier     = compute_brier_score(best_pipe_e, X_te_e, y_te)
    logger.info("  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
                test_m_e["f1"], test_m_e["precision"], brier, baseline)

    try:
        thr_info   = select_threshold_on_val(best_pipe_e, X_val_e, y_val_e)
        thr_test_m = evaluate_at_threshold(best_pipe_e, X_te_e, y_te, thr_info["threshold"])
    except ValueError as exc:
        logger.warning("  Threshold selection failed: %s", exc)
        thr_info, thr_test_m = None, None

    # F_B ablation — C2 fix: tuned independently
    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b, y_tr_b_full, y_te_b = split_data(X_base, y)

    logger.info("  Nested CV (F_B ablation)...")
    nested_b = nested_cv_with_tuning(
        _make_pipeline_factory(cw), LR_PARAM_GRID,
        X_tr_b_full, y_tr_b_full, n_iter=20
    )
    cv_raw_b = nested_b["cv_metrics_raw"]
    best_pipe_b, _, _ = tune_hyperparameters(
        _make_pipeline(class_weight=cw), LR_PARAM_GRID, X_tr_b_full, y_tr_b_full, n_iter=20
    )
    best_pipe_b.fit(X_tr_b_full, y_tr_b_full)
    test_m_b = compute_test_metrics(best_pipe_b, X_te_b, y_te_b)
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=best_pipe_e, scaler=None,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=best_pipe_e.predict(X_te_e),
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME, best_params=best_params_e,
        threshold_metrics=thr_test_m,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        brier_score=brier,
        no_skill_brier=baseline,
        class_weight=cw,
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