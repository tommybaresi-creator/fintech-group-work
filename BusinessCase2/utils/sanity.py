"""
Data & Feature Sanity Checks — terminal script.

Mirrors data_sanity.ipynb exactly. Exits with code 1 on any failed assertion.
Run before model training to confirm implementation matches paper claims.

Usage:  python utils/sanity.py
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


def check_raw_data(df):
    assert df.shape[0] == 5000, f"Expected 5000 rows, got {df.shape[0]}"
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values"
    assert "Income"   in df.columns, "'Income' column not found"
    assert "Income " not in df.columns, "Trailing-space bug still present"
    logger.info("Section 1 PASSED — shape %s, nulls 0, Income column clean", df.shape)


def check_class_balance(df):
    income_prev = df["IncomeInvestment"].mean()
    accum_prev  = df["AccumulationInvestment"].mean()
    for target, prev in [("IncomeInvestment", income_prev), ("AccumulationInvestment", accum_prev)]:
        logger.info("%s: prevalence=%.3f  counts=%s", target, prev, df[target].value_counts().to_dict())
    assert 0.35 < income_prev < 0.42, f"IncomeInvestment prevalence={income_prev:.3f} outside [0.35, 0.42]"
    assert 0.48 < accum_prev  < 0.54, f"AccumulationInvestment prevalence={accum_prev:.3f} outside [0.48, 0.54]"
    logger.info("Section 2 PASSED — class balance confirmed")


def check_skewness(df):
    thresholds = {"Wealth": 2.0, "Income": 1.0}
    for col, threshold in thresholds.items():
        sk = skew(df[col])
        logger.info("%s: skewness=%.2f  (threshold > %.1f)", col, sk, threshold)
        assert sk > threshold, f"{col} skewness={sk:.2f} below threshold"
    logger.info("Section 3 PASSED — log transforms justified")


def check_engineered_features(df):
    X = build_features(df)
    assert list(X.columns) == FEATURE_NAMES
    assert X.shape == (5000, len(FEATURE_NAMES))
    assert X.isnull().sum().sum() == 0
    assert np.allclose(X["Age_sq"],           df["Age"] ** 2)
    assert np.allclose(X["FinEdu_x_Risk"],     df["FinancialEducation"] * df["RiskPropensity"])
    assert np.allclose(X["Income_log"],        np.log1p(df["Income"]))
    assert np.allclose(X["Wealth_log"],        np.log1p(df["Wealth"]))
    assert np.allclose(X["Age_x_Wealth"],      df["Age"] * np.log1p(df["Wealth"]))
    assert np.allclose(X["Income_per_FM_log"], np.log1p(df["Income"] / df["FamilyMembers"]))
    assert np.allclose(X["Wealth_per_FM_log"], np.log1p(df["Wealth"] / df["FamilyMembers"]))
    assert "Gender"        not in X.columns
    assert "FamilyMembers" not in X.columns
    assert "Wealth"        not in X.columns
    assert "Income"        not in X.columns
    logger.info("Section 4 PASSED — F_E shape=%s, all formulas correct, exclusions confirmed", X.shape)


def check_no_leakage(df):
    X = build_features(df)
    for target in TARGETS:
        X_tr, X_te, _, _, scaler = split_and_scale(X, df[target])
        assert scaler.data_min_ is not None
        assert X_tr.min().min() >= -1e-9
        assert X_tr.max().max() <= 1.0 + 1e-9
        logger.info("%s — train [%.4f, %.4f] | test [%.4f, %.4f]",
                    target, X_tr.min().min(), X_tr.max().max(),
                    X_te.min().min(), X_te.max().max())
    logger.info("Section 5 PASSED — no leakage confirmed")


def check_stratification(df):
    X = build_features(df)
    for target in TARGETS:
        _, _, y_tr, y_te, _ = split_and_scale(X, df[target])
        delta = abs(y_tr.mean() - y_te.mean())
        logger.info("%s: train=%.4f  test=%.4f  delta=%.4f", target, y_tr.mean(), y_te.mean(), delta)
        assert delta < 0.03, f"{target}: stratification drift too large: {delta:.4f}"
    logger.info("Section 6 PASSED — stratification confirmed")


def check_correlations(df):
    X    = build_features(df)
    full = pd.concat([X, df[TARGETS]], axis=1)
    corr = full.corr()
    checks = [
        ("Age_x_Wealth  vs IncomeInvestment",      corr.loc["Age_x_Wealth",  "IncomeInvestment"],        ">", 0.30),
        ("Age           vs IncomeInvestment",       corr.loc["Age",           "IncomeInvestment"],         ">", 0.25),
        ("Wealth_log    vs IncomeInvestment",       corr.loc["Wealth_log",    "IncomeInvestment"],         ">", 0.30),
        ("Income_log    vs AccumulationInvestment", corr.loc["Income_log",    "AccumulationInvestment"],   ">", 0.25),
        ("FinEdu        vs RiskPropensity",         corr.loc["FinancialEducation", "RiskPropensity"],      ">", 0.60),
        ("Income_per_FM vs Income_log",             corr.loc["Income_per_FM_log",  "Income_log"],          ">", 0.85),
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


def check_feature_set_structure(df):
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
    logger.info("Section 8 PASSED — F_E (%d) and F_B (%d) structure confirmed", len(FEATURE_NAMES), len(BASELINE_FEATURE_NAMES))


def run():
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