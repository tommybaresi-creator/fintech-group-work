"""
Gaussian Naive Bayes classifier for both investment targets.

Key design decisions (Section 3.3 of the paper):
- **No scaling.** GNB estimates per-class parameters (μ_kj, σ²_kj) for each
  feature independently. The likelihood ratio depends only on within-class
  distributions; rescaling changes μ and σ proportionally, leaving the
  ratio invariant. Scaling is therefore strictly unnecessary.
- **F_B only.** NB is tested on the professor's baseline feature set only.
  The Gaussian independence assumption is already violated by the base
  features; adding constructed interactions (F_E) compounds this violation
  without the benefit that motivates them for Logistic Regression.
- No class-weight correction: GNB uses Bayesian priors for imbalance.
  The prior is set via ``priors=None`` (sklearn default), which estimates
  class priors from the training data — the statistically correct approach.

Saves artifacts to ``data/pickled_files/naive_bayes/``.

Run directly::

    python -m utils.naive_bayes
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

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "pickled_files" / "naive_bayes"
)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "GaussianNB"


def _make_model() -> GaussianNB:
    return GaussianNB()


def run_for_target(df, target_col: str) -> dict:
    """
    Train and evaluate Gaussian Naive Bayes for one binary target.

    Uses F_B only with no scaling. 10-fold stratified CV for outer
    performance estimation.

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

    X_base = build_baseline_features(df)
    X_tr, X_te, y_tr, y_te = split_data(X_base, y)

    cv_raw = compute_cv_metrics(_make_model(), X_tr, y_tr)
    logger.info(
        "  [F_B] CV F1: %.3f ± %.3f",
        np.mean(cv_raw["f1"]),
        np.std(cv_raw["f1"]),
    )

    model = _make_model()
    model.fit(X_tr, y_tr)
    test_m = compute_test_metrics(model, X_te, y_te)
    logger.info("  [F_B] Test F1 (thr=0.5): %.3f", test_m["f1"])

    try:
        thr_info = select_threshold_pr_curve(model, X_te, y_te)
    except ValueError as exc:
        logger.warning("Threshold optimisation failed: %s", exc)
        thr_info = None

    brier = compute_brier_score(model, X_te, y_te)
    logger.info("  [F_B] Brier score: %.4f", brier)

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
        ablation=None,   # NB tested on F_B only; no ablation comparison
        threshold_info=thr_info,
        brier_score=brier,
    )


def main() -> None:
    """Train, evaluate, and pickle Gaussian Naive Bayes for all targets."""
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
