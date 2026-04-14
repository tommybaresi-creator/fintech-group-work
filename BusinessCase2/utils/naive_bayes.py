"""
Gaussian Naive Bayes for both investment targets.

Key design decisions (Section 3.3):
- No scaling. GNB's likelihood ratio is invariant to linear rescaling.
- F_B only. The Gaussian independence assumption is already violated by
  base features; adding F_E interactions compounds the violation without
  benefit.
- No class_weight: GNB estimates class priors from training data via
  Bayesian inference — the statistically correct imbalance handling.

Saves artifacts to data/pickled_files/naive_bayes/.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    compute_brier_score,
    compute_cv_metrics,
    compute_test_metrics,
    load_data,
    make_result_dict,
    select_threshold_pr_curve,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "naive_bayes"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "GaussianNB"


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]

    X_base = build_baseline_features(df)
    X_tr, X_te, y_tr, y_te = split_data(X_base, y)

    cv_raw = compute_cv_metrics(GaussianNB(), X_tr, y_tr)
    logger.info("  [F_B] CV  F1: %.3f ± %.3f", np.mean(cv_raw["f1"]), np.std(cv_raw["f1"]))

    model = GaussianNB()
    model.fit(X_tr, y_tr)
    test_m = compute_test_metrics(model, X_te, y_te)
    logger.info("  [F_B] Test F1 (thr=0.5): %.3f", test_m["f1"])

    try:
        thr_info = select_threshold_pr_curve(model, X_te, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    brier = compute_brier_score(model, X_te, y_te)
    logger.info("  [F_B] Brier: %.4f", brier)

    return make_result_dict(
        model=model,
        scaler=None,
        cv_metrics_raw=cv_raw,
        test_metrics=test_m,
        y_test_true=y_te.values,
        y_test_pred=model.predict(X_te),
        feature_names=BASELINE_FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation=None,
        threshold_info=thr_info,
        brier_score=brier,
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