"""
XGBoost classifier with SHAP explainability for both investment targets.

Key design decisions (Section 3.3 of the paper):
- **F_B** (professor's baseline) used as the primary feature set. Ablation
  shows that XGBoost learns all relevant interactions natively through
  sequential splits; adding pre-computed interaction terms (F_E) introduces
  correlated features that dilute SHAP importance without lifting F1.
- **scale_pos_weight = neg/pos** applied to IncomeInvestment only (38%
  positive rate). Not applied to AccumulationInvestment (51%, balanced).
- **Post-hoc isotonic calibration.** XGBoost probabilities can be
  overconfident for IncomeInvestment at high predicted probabilities;
  isotonic regression corrects this nonparametrically (Section 3.6).
- **SHAP TreeExplainer** values pre-computed and pickled so that
  ``bestmodel_*.ipynb`` and ``rec_sys.ipynb`` render explanations without
  retraining.

Saves artifacts to ``data/pickled_files/xgboost_shap/``.

Run directly::

    python -m utils.xgboost_shap
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import shap
from xgboost import XGBClassifier

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
    scale_pos_weight,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "pickled_files" / "xgboost_shap"
)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "XGBoost"


def _make_model(spw: float = 1.0) -> XGBClassifier:
    """
    Instantiate an XGBClassifier.

    Parameters
    ----------
    spw : float
        ``scale_pos_weight`` value. Pass ``neg/pos`` ratio for imbalanced
        targets; leave at 1.0 for balanced ones.

    Returns
    -------
    XGBClassifier
        Unfitted estimator.
    """
    return XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=spw,
        verbosity=0,
        use_label_encoder=False,
    )


def run_for_target(df, target_col: str) -> dict:
    """
    Train, calibrate, and explain XGBoost for one binary target.

    Primary feature set: F_B. Ablation comparison against F_E.
    scale_pos_weight applied for IncomeInvestment only.
    Post-hoc isotonic calibration before threshold optimisation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by :func:`~utils.preprocessing.load_data`.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    dict
        Canonical result dictionary with additional keys:
        ``'shap_explainer'``, ``'shap_values'``, ``'shap_test_X'``,
        ``'feature_importances'``, ``'brier_score_pre_cal'``,
        ``'brier_score'``, ``'threshold_info'``.
    """
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # scale_pos_weight: only for imbalanced Income target
    spw = scale_pos_weight(y) if target_col == "IncomeInvestment" else 1.0

    # ------------------------------------------------------------------ F_B (primary)
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, y_tr, y_te = split_data(X_base, y)

    cv_raw_b = compute_cv_metrics(_make_model(spw), X_tr_b, y_tr)
    logger.info(
        "  [F_B] CV F1: %.3f ± %.3f",
        np.mean(cv_raw_b["f1"]),
        np.std(cv_raw_b["f1"]),
    )

    model_b = _make_model(spw)
    model_b.fit(X_tr_b, y_tr)

    brier_pre = compute_brier_score(model_b, X_te_b, y_te)
    logger.info("  [F_B] Brier pre-calibration:  %.4f", brier_pre)

    cal_model_b = calibrate_model(model_b, X_tr_b, y_tr)
    test_m_b = compute_test_metrics(cal_model_b, X_te_b, y_te)
    brier_post = compute_brier_score(cal_model_b, X_te_b, y_te)
    logger.info("  [F_B] Test F1 (thr=0.5): %.3f", test_m_b["f1"])
    logger.info(
        "  [F_B] Brier post-calibration: %.4f  (Δ=%.4f)",
        brier_post,
        brier_post - brier_pre,
    )

    try:
        thr_info = select_threshold_pr_curve(cal_model_b, X_te_b, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    # SHAP on the uncalibrated model (TreeExplainer requires the base tree model)
    logger.info("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model_b)
    shap_values = explainer.shap_values(X_te_b)

    # ------------------------------------------------------------------ F_E (ablation)
    X_eng = build_features(df)
    X_tr_e, X_te_e, _, _ = split_data(X_eng, y)

    cv_raw_e = compute_cv_metrics(_make_model(spw), X_tr_e, y_tr)
    model_e = _make_model(spw)
    model_e.fit(X_tr_e, y_tr)
    cal_model_e = calibrate_model(model_e, X_tr_e, y_tr)
    test_m_e = compute_test_metrics(cal_model_e, X_te_e, y_te)
    logger.info(
        "  [F_E] Test F1: %.3f  (delta F_B−F_E = %.3f)",
        test_m_e["f1"],
        test_m_b["f1"] - test_m_e["f1"],
    )

    return make_result_dict(
        model=cal_model_b,
        scaler=None,
        cv_metrics_raw=cv_raw_b,
        test_metrics=test_m_b,
        y_test_true=y_te.values,
        y_test_pred=cal_model_b.predict(X_te_b),
        feature_names=BASELINE_FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "baseline": {
                "cv_metrics_raw": cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics": test_m_b,
            },
            "engineered": {
                "cv_metrics_raw": cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics": test_m_e,
            },
        },
        shap_explainer=explainer,
        shap_values=shap_values,
        shap_test_X=X_te_b,
        feature_importances=model_b.feature_importances_,
        brier_score_pre_cal=brier_pre,
        brier_score=brier_post,
        threshold_info=thr_info,
        scale_pos_weight_used=spw,
    )


def main() -> None:
    """Train, calibrate, explain, and pickle XGBoost for all targets."""
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
