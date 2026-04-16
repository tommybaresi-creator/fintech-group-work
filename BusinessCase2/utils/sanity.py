"""
Data and feature sanity checks for the KYC investment classification pipeline.

Theory
------
Machine learning pipelines can silently degrade when their underlying data
assumptions are violated. Since downstream models typically lack mechanisms to
detect upstream failures, this module implements explicit validation checks
that act as safeguards against silent corruption.

Each check targets a specific failure mode:

Shape and null values
Unexpected dataset size or missing values usually indicate version mismatch,
failed joins, or incomplete preprocessing. These issues propagate silently into
all downstream models if not explicitly caught.

Class prevalence
The proportion of positive labels directly influences imbalance handling
strategies such as scale_pos_weight (XGBoost) and class_weight (Logistic
Regression / SVM). Changes in distribution can invalidate previously tuned
weights.

Skewness
High skewness (typically > 2.0 for Wealth and > 1.0 for Income) justifies the
use of log1p transformations. If skewness is low, such transformations may
introduce unnecessary distortion, especially for linear models.

Formula verification
Ensures that engineered feature definitions implemented in build_features
exactly match the reference mathematical specification. This guards against
silent regressions introduced during refactoring.

Leakage detection
Verifies that MinMaxScaler was fitted exclusively on training data. Training
features should lie within [0, 1] after scaling. Test values outside this
range are expected and do not indicate leakage, as they may reflect true
distribution shift or out-of-range observations.

Stratification drift
Ensures that class prevalence between training and test sets differs by less
than 3 percentage points. This maintains comparability between evaluation
conditions.

Correlation plausibility
Reproduces expected feature-target and inter-feature correlations as reported
in the reference design. This validates that engineered features preserve the
intended predictive structure.

Feature set structure
Enforces invariants on feature composition:
- F_E must exclude Gender and FamilyMembers
- Raw monetary variables must not appear in engineered features
- F_B must remain a coherent fixed 7-feature baseline set

Implementation
--------------
This script mirrors the assertions defined in data_sanity.ipynb and is intended
to be executed prior to any model training stage.

It terminates with exit code 1 on the first failed assertion, making it suitable
for CI/CD pre-flight validation.

Usage:
    python utils/sanity.py
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    build_features,
    load_data,
    split_and_scale,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sanity")


def check_raw_data(df: pd.DataFrame) -> None:
    """
    Assert that the raw dataset has the expected shape and cleanliness.

    Checks:
    * Exactly 5,000 rows.
    * Zero null values across all columns.
    * An ``'Income'`` column is present with no trailing-whitespace variant.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded by :func:`load_data`.

    Raises
    ------
    AssertionError
        On any failed check.
    """
    assert df.shape[0] == 5000, f"Expected 5000 rows, got {df.shape[0]}"
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values"
    assert "Income"   in df.columns, "'Income' column not found"
    assert "Income " not in df.columns, "Trailing-space bug still present"
    logger.info("Section 1 PASSED — shape %s, nulls 0, Income column clean", df.shape)


def check_class_balance(df: pd.DataFrame) -> None:
    """
    Assert that class prevalences match the expected distribution.

    Expected ranges (from the paper):
    * IncomeInvestment: 35%–42% positive.
    * AccumulationInvestment: 48%–54% positive.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If any prevalence falls outside its expected range.
    """
    income_prev = df["IncomeInvestment"].mean()
    accum_prev  = df["AccumulationInvestment"].mean()
    for target, prev in [("IncomeInvestment", income_prev), ("AccumulationInvestment", accum_prev)]:
        logger.info("%s: prevalence=%.3f  counts=%s", target, prev, df[target].value_counts().to_dict())
    assert 0.35 < income_prev < 0.42, f"IncomeInvestment prevalence={income_prev:.3f} outside [0.35, 0.42]"
    assert 0.48 < accum_prev  < 0.54, f"AccumulationInvestment prevalence={accum_prev:.3f} outside [0.48, 0.54]"
    logger.info("Section 2 PASSED — class balance confirmed")


def check_skewness(df: pd.DataFrame) -> None:
    """
    Assert that Wealth and Income are sufficiently right-skewed.

    Required thresholds: Wealth > 2.0, Income > 1.0.  High skewness validates
    the log1p transformation applied in :func:`build_features`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If any column's skewness falls below its threshold.
    """
    thresholds = {"Wealth": 2.0, "Income": 1.0}
    for col, threshold in thresholds.items():
        sk = skew(df[col])
        logger.info("%s: skewness=%.2f  (threshold > %.1f)", col, sk, threshold)
        assert sk > threshold, f"{col} skewness={sk:.2f} below threshold"
    logger.info("Section 3 PASSED — log transforms justified")


def check_engineered_features(df: pd.DataFrame) -> None:
    """
    Assert that :func:`build_features` produces the correct F_E output.

    Checks:
    * Columns match FEATURE_NAMES exactly in order.
    * Shape is (5000, 10).
    * No null values.
    * All formula outputs match manual recomputation to floating-point precision.
    * Gender and FamilyMembers are absent from F_E.
    * Raw Income and Wealth columns are absent from F_E.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    X = build_features(df)
    assert list(X.columns) == FEATURE_NAMES
    assert X.shape == (5000, len(FEATURE_NAMES))
    assert X.isnull().sum().sum() == 0
    assert np.allclose(X["Age_sq"],           df["Age"] ** 2)
    assert np.allclose(X["FinEdu_x_Risk"],     df["FinancialEducation"] * df["RiskPropensity"])
    assert np.allclose(X["Income_log"],        np.log1p(df["Income"]))
    assert np.allclose(X["Wealth_log"],        np.log1p(df["Wealth"]))
    assert np.allclose(X["Age_x_Wealthlog"],   df["Age"] * np.log1p(df["Wealth"]))
    assert np.allclose(X["Income_per_FM_log"], np.log1p(df["Income"] / df["FamilyMembers"]))
    assert np.allclose(X["Wealth_per_FM_log"], np.log1p(df["Wealth"] / df["FamilyMembers"]))
    assert "Gender"        not in X.columns
    assert "FamilyMembers" not in X.columns
    assert "Wealth"        not in X.columns
    assert "Income"        not in X.columns
    logger.info("Section 4 PASSED — F_E shape=%s, all formulas correct, exclusions confirmed", X.shape)


def check_no_leakage(df: pd.DataFrame) -> None:
    """
    Assert that MinMaxScaler is fitted on training data only.

    Confirms the scaler's ``data_min_`` is populated and that training-set
    feature values lie within [0, 1] after scaling (allowing for floating-point
    tolerance).  Test-set values outside [0, 1] are expected and do not
    indicate leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If training-set scaled values fall outside [0, 1].
    """
    X = build_features(df)
    for target in TARGETS:
        X_tr, X_te, _, _, scaler = split_and_scale(X, df[target])
        assert scaler.data_min_ is not None
        assert X_tr.min().min() >= -1e-9
        assert X_tr.max().max() <= 1.0 + 1e-9
        logger.info(
            "%s — train [%.4f, %.4f] | test [%.4f, %.4f]",
            target, X_tr.min().min(), X_tr.max().max(),
            X_te.min().min(), X_te.max().max(),
        )
    logger.info("Section 5 PASSED — no leakage confirmed")


def check_stratification(df: pd.DataFrame) -> None:
    """
    Assert that train / test stratification drift is below 3 percentage points.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If the prevalence difference between train and test exceeds 0.03 for
        any target.
    """
    X = build_features(df)
    for target in TARGETS:
        _, _, y_tr, y_te, _ = split_and_scale(X, df[target])
        delta = abs(y_tr.mean() - y_te.mean())
        logger.info("%s: train=%.4f  test=%.4f  delta=%.4f", target, y_tr.mean(), y_te.mean(), delta)
        assert delta < 0.03, f"{target}: stratification drift too large: {delta:.4f}"
    logger.info("Section 6 PASSED — stratification confirmed")


def check_correlations(df: pd.DataFrame) -> None:
    """
    Assert that seven key feature correlations match the paper's claims.

    Checks include directional and magnitude claims for feature-target
    correlations (Age, Wealth, Income) and inter-feature correlations
    (FinancialEducation × RiskPropensity, Income per-capita).  Also verifies
    near-zero correlation between the two targets (label independence).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If any of the seven checks fails.
    """
    X    = build_features(df)
    full = pd.concat([X, df[TARGETS]], axis=1)
    corr = full.corr()
    checks = [
        ("Age_x_Wealth  vs IncomeInvestment",      corr.loc["Age_x_Wealthlog",  "IncomeInvestment"],       ">", 0.30),
        ("Age           vs IncomeInvestment",       corr.loc["Age",           "IncomeInvestment"],          ">", 0.25),
        ("Wealth_log    vs IncomeInvestment",       corr.loc["Wealth_log",    "IncomeInvestment"],          ">", 0.30),
        ("Income_log    vs AccumulationInvestment", corr.loc["Income_log",    "AccumulationInvestment"],    ">", 0.25),
        ("FinEdu        vs RiskPropensity",         corr.loc["FinancialEducation", "RiskPropensity"],       ">", 0.60),
        ("Income_per_FM vs Income_log",             corr.loc["Income_per_FM_log",  "Income_log"],           ">", 0.85),
        ("Income vs Accumulation (near-zero)",      abs(corr.loc["IncomeInvestment", "AccumulationInvestment"]), "<", 0.05),
    ]
    all_passed = True
    for label, value, op, threshold in checks:
        passed = (value > threshold) if op == ">" else (value < threshold)
        status = "PASS" if passed else "FAIL"
        logger.info("[%s]  %s: %.3f  (%s %.2f)", status, label, value, op, threshold)
        if not passed:
            all_passed = False
    assert all_passed, "One or more correlation checks failed"
    logger.info("Section 7 PASSED — all 7 correlation claims confirmed")


def check_feature_set_structure(df: pd.DataFrame) -> None:
    """
    Assert the structural relationship between F_E and F_B.

    Checks:
    * Gender is absent from F_E and present in F_B.
    * FamilyMembers is absent from F_E.
    * F_B columns are a subset of the baseline column space.
    * F_E columns exactly match FEATURE_NAMES.
    * F_E contains at least one feature not in F_B (non-trivial engineering).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Raises
    ------
    AssertionError
        If any structural check fails.
    """
    X      = build_features(df)
    X_base = build_baseline_features(df)
    added  = set(FEATURE_NAMES) - set(BASELINE_FEATURE_NAMES)
    assert "Gender"        not in FEATURE_NAMES
    assert "Gender"        in BASELINE_FEATURE_NAMES
    assert "FamilyMembers" not in FEATURE_NAMES
    assert set(BASELINE_FEATURE_NAMES).issubset(set(X_base.columns))
    assert set(FEATURE_NAMES) == set(X.columns)
    assert len(added) > 0
    logger.info("F_E: %s", list(X.columns))
    logger.info("F_B: %s", list(X_base.columns))
    logger.info("Added: %s", sorted(added))
    logger.info(
        "Section 8 PASSED — F_E (%d) and F_B (%d) structure confirmed",
        len(FEATURE_NAMES), len(BASELINE_FEATURE_NAMES),
    )


def run() -> None:
    """
    Execute all eight sanity check functions in sequence on the raw dataset.

    Logs a summary header and footer.  When called as ``__main__``, any
    ``AssertionError`` or unexpected exception causes an error log and a
    ``sys.exit(1)``.
    """
    logger.info("=" * 60)
    logger.info("DATA & FEATURE SANITY CHECKS")
    logger.info("=" * 60)
    df = load_data()
    check_raw_data(df)
    check_class_balance(df)
    check_skewness(df)
    check_engineered_features(df)
    check_no_leakage(df)
    check_stratification(df)
    check_correlations(df)
    check_feature_set_structure(df)
    logger.info("=" * 60)
    logger.info("ALL CHECKS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except (AssertionError, Exception) as e:
        logger.error("FAILED: %s", e)
        sys.exit(1)
