"""
Random Forest classifier for both investment targets.

Key design decisions (Section 3.3 of the paper):
- **No scaling.** A threshold split x_j > t is invariant to any strictly
  monotonic transformation; linear scaling is a special case.
- **F_E** (engineered feature set). RF cannot be harmed by pre-computed
  interaction terms — it will simply learn equivalent splits anyway.
- **Post-hoc isotonic calibration** via ``CalibratedClassifierCV``. RF
  probability estimates are biased toward 0.5 (averaging over many trees
  compresses probabilities). Isotonic regression corrects this without
  assuming a sigmoidal error form (Section 3.6).
- **Gini feature importances** stored for interpretability.

Saves artifacts to ``data/pickled_files/rand_forest/``.

Run directly::

    python -m utils.rand_forest
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    split_data,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "pickled_files" / "rand_forest"
)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "RandomForest"


def _make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def run_for_target(df, target_col: str) -> dict:
    """
    Train, calibrate, and evaluate Random Forest for one binary target.

    Ablation: F_E (primary) vs F_B (baseline comparison).
    Post-hoc isotonic calibration applied to the final model.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by :func:`~utils.preprocessing.load_data`.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    dict
        Canonical result dictionary with ``'feature_importances'``,
        ``'brier_score_pre_cal'``, ``'brier_score'``, and
        ``'threshold_info'`` keys.
    """
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # ------------------------------------------------------------------ F_E
    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr, y_te = split_data(X_eng, y)

    cv_raw_e = compute_cv_metrics(_make_model(), X_tr_e, y_tr)
    logger.info(
        "  [F_E] CV F1: %.3f ± %.3f",
        np.mean(cv_raw_e["f1"]),
        np.std(cv_raw_e["f1"]),
    )

    model_e = _make_model()
    model_e.fit(X_tr_e, y_tr)

    brier_pre = compute_brier_score(model_e, X_te_e, y_te)
    logger.info("  [F_E] Brier pre-calibration:  %.4f", brier_pre)

    cal_model_e = calibrate_model(model_e, X_tr_e, y_tr)
    test_m_e = compute_test_metrics(cal_model_e, X_te_e, y_te)
    brier_post = compute_brier_score(cal_model_e, X_te_e, y_te)
    logger.info("  [F_E] Test F1 (thr=0.5): %.3f", test_m_e["f1"])
    logger.info(
        "  [F_E] Brier post-calibration: %.4f  (Δ=%.4f)",
        brier_post,
        brier_post - brier_pre,
    )

    try:
        thr_info = select_threshold_pr_curve(cal_model_e, X_te_e, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    # ------------------------------------------------------------------ F_B
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, _, _ = split_data(X_base, y)

    cv_raw_b = compute_cv_metrics(_make_model(), X_tr_b, y_tr)
    model_b = _make_model()
    model_b.fit(X_tr_b, y_tr)
    cal_model_b = calibrate_model(model_b, X_tr_b, y_tr)
    test_m_b = compute_test_metrics(cal_model_b, X_te_b, y_te)
    logger.info(
        "  [F_B] Test F1: %.3f  (delta F_E−F_B = %.3f)",
        test_m_b["f1"],
        test_m_e["f1"] - test_m_b["f1"],
    )

    return make_result_dict(
        model=cal_model_e,
        scaler=None,
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_m_e,
        y_test_true=y_te.values,
        y_test_pred=cal_model_e.predict(X_te_e),
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
        feature_importances=model_e.feature_importances_,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        threshold_info=thr_info,
    )


def main() -> None:
    """Train, calibrate, evaluate, and pickle Random Forest for all targets."""
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
