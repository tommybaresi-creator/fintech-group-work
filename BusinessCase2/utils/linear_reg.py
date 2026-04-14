"""
Logistic Regression classifier for both investment targets.

Key design decisions:
- StandardScaler inside a Pipeline — scaler is refitted on each CV fold's
  training data automatically, preventing leakage.
- L1 regularisation (solver='liblinear') on F_E: automatically zeros out
  one of the collinear pair (FinancialEducation, RiskPropensity, r=0.68).
- class_weight='balanced' applied to IncomeInvestment only (38% positive).
  AccumulationInvestment is 51% balanced; correcting it degrades performance.
- Ablation: F_E with L1 (primary) vs F_B with L1 (baseline).

Saves artifacts to data/pickled_files/linear_reg/.
"""

import logging
import sys
from pathlib import Path

import joblib
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
    load_data,
    make_result_dict,
    select_threshold_pr_curve,
    split_and_standardize,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "linear_reg"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "LogisticRegression"


def _make_pipeline(penalty: str = "l1", class_weight=None) -> Pipeline:
    """
    LR wrapped in a Pipeline with StandardScaler.

    The Pipeline ensures the scaler is refitted on each fold's training data
    when passed to compute_cv_metrics — the correct leakage-free pattern.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            random_state=42,
            class_weight=class_weight,
        )),
    ])


def run_for_target(df, target_col: str) -> dict:
    """
    Train and evaluate Logistic Regression for one binary target.

    CV uses the Pipeline directly on unscaled data — scaler refits per fold.
    Final model uses split_and_standardize for the test evaluation.
    """
    logger.info("Target: %s", target_col)
    y  = df[target_col]
    cw = "balanced" if target_col == "IncomeInvestment" else None

    # ---- F_E -----------------------------------------------------------------
    X_eng = build_features(df)

    # CV: pass unscaled X_tr + Pipeline (scaler refits inside each fold)
    X_tr_e, X_te_e, y_tr, y_te = split_data(X_eng, y)
    cv_raw_e = compute_cv_metrics(_make_pipeline("l1", cw), X_tr_e, y_tr)
    logger.info("  [F_E, L1] CV  F1: %.3f ± %.3f", np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"]))

    # Final model: fit Pipeline on full train, evaluate on test
    pipe_e = _make_pipeline("l1", cw)
    pipe_e.fit(X_tr_e, y_tr)
    test_m_e = compute_test_metrics(pipe_e, X_te_e, y_te)
    logger.info("  [F_E, L1] Test F1 (thr=0.5): %.3f", test_m_e["f1"])

    try:
        thr_info = select_threshold_pr_curve(pipe_e, X_te_e, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    brier = compute_brier_score(pipe_e, X_te_e, y_te)
    logger.info("  [F_E, L1] Brier: %.4f", brier)

    # ---- F_B (ablation) ------------------------------------------------------
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_data(X_base, y)
    cv_raw_b = compute_cv_metrics(_make_pipeline("l1", cw), X_tr_b, y_tr_b)
    pipe_b = _make_pipeline("l1", cw)
    pipe_b.fit(X_tr_b, y_tr_b)
    test_m_b = compute_test_metrics(pipe_b, X_te_b, y_te_b)
    logger.info("  [F_B, L1] Test F1: %.3f  (ΔF_E−F_B = %+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    return make_result_dict(
        model=pipe_e,
        scaler=None,           # scaler is inside the Pipeline
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_m_e,
        y_test_true=y_te.values,
        y_test_pred=pipe_e.predict(X_te_e),
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        threshold_info=thr_info,
        brier_score=brier,
        class_weight=cw,
    )


def main() -> None:
    df = load_data()
    for target in TARGETS:
        result = run_for_target(df, target)
        out_path = PICKLE_DIR / f"{target.lower()}.joblib"
        joblib.dump(result, out_path, compress=3)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()