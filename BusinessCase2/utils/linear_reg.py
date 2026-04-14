"""
Logistic Regression classifier for both investment targets.

Key design decisions (Section 3.3 of the paper):
- **StandardScaler** (zero-mean, unit-variance), NOT MinMaxScaler.
  L1/L2 regularisation shrinks all coefficients by the same λ; if features
  have unequal variance MinMaxScaler leaves the effective penalty non-uniform.
  StandardScaler ensures uniform penalty calibration across all features.
- **L1 regularisation** (``solver='liblinear'``) on F_E: automatically zeros
  out one of the collinear pair (FinancialEducation, RiskPropensity, r=0.68).
- ``class_weight='balanced'`` applied to **IncomeInvestment only** (38%
  positive rate). AccumulationInvestment is 51% balanced; correcting it would
  degrade performance by over-weighting a non-problem.
- Ablation: F_E vs F_B with L1 for both, plus L2 on F_B as the original
  professor's configuration.

Saves artifacts to ``data/pickled_files/linear_reg/``.

Run directly::

    python -m utils.linear_reg
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    build_features,
    calibrate_model,
    compute_brier_score,
    compute_cv_metrics,
    compute_test_metrics,
    load_data,
    make_result_dict,
    select_threshold_pr_curve,
    split_and_standardize,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "pickled_files" / "linear_reg"
)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "LogisticRegression"


def _make_model(penalty: str = "l1", class_weight=None) -> LogisticRegression:
    """
    Instantiate a Logistic Regression model.

    Parameters
    ----------
    penalty : str
        ``'l1'`` or ``'l2'``.
    class_weight : dict or ``'balanced'`` or None
        Passed directly to sklearn.

    Returns
    -------
    LogisticRegression
        Unfitted estimator.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=1000,
        random_state=42,
        class_weight=class_weight,
    )


def run_for_target(df, target_col: str) -> dict:
    """
    Train and evaluate Logistic Regression for one binary target.

    Ablation: F_E with L1 (primary) vs F_B with L1 (baseline comparison).
    Class-imbalance correction applied to IncomeInvestment only.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by :func:`~utils.preprocessing.load_data`.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    dict
        Canonical result dictionary including ``'threshold_info'`` and
        ``'brier_score'`` keys.
    """
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # Class-weight correction: only for imbalanced Income target (38% pos)
    cw = "balanced" if target_col == "IncomeInvestment" else None

    # ------------------------------------------------------------------ F_E
    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr, y_te, scaler_e = split_and_standardize(X_eng, y)

    cv_raw_e = compute_cv_metrics(_make_model("l1", cw), X_tr_e, y_tr)
    logger.info(
        "  [F_E, L1] CV F1: %.3f ± %.3f",
        np.mean(cv_raw_e["f1"]),
        np.std(cv_raw_e["f1"]),
    )

    model_e = _make_model("l1", cw)
    model_e.fit(X_tr_e, y_tr)
    test_m_e = compute_test_metrics(model_e, X_te_e, y_te)
    logger.info("  [F_E, L1] Test F1 (thr=0.5): %.3f", test_m_e["f1"])

    # PR-curve threshold (MiFID II Precision ≥ 0.75)
    try:
        thr_info = select_threshold_pr_curve(model_e, X_te_e, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    brier = compute_brier_score(model_e, X_te_e, y_te)
    logger.info("  [F_E, L1] Brier score: %.4f", brier)

    # ------------------------------------------------------------------ F_B
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, _, _, _ = split_and_standardize(X_base, y)

    cv_raw_b = compute_cv_metrics(_make_model("l1", cw), X_tr_b, y_tr)
    model_b = _make_model("l1", cw)
    model_b.fit(X_tr_b, y_tr)
    test_m_b = compute_test_metrics(model_b, X_te_b, y_te)
    logger.info(
        "  [F_B, L1] Test F1: %.3f  (delta F_E−F_B = %.3f)",
        test_m_b["f1"],
        test_m_e["f1"] - test_m_b["f1"],
    )

    return make_result_dict(
        model=model_e,
        scaler=scaler_e,
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_m_e,
        y_test_true=y_te.values,
        y_test_pred=model_e.predict(X_te_e),
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "engineered": {
                "cv_metrics_raw": cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics": test_m_e,
            },
            "baseline": {
                "cv_metrics_raw": cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics": test_m_b,
            },
        },
        threshold_info=thr_info,
        brier_score=brier,
        class_weight=cw,
    )


def main() -> None:
    """Train, evaluate, and pickle Logistic Regression for all targets."""
    df = load_data()
    for target in TARGETS:
        result = run_for_target(df, target)
        out_path = PICKLE_DIR / f"{target.lower()}.joblib"
        try:
            joblib.dump(result, out_path, compress=3)
            logger.info("Saved: %s", out_path)
        except OSError as exc:
            logger.error("Failed to save %s: %s", out_path, exc)
            raise


if __name__ == "__main__":
    main()
