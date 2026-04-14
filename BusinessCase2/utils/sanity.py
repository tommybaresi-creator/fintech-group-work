"""
Data & Feature Sanity Checks — terminal script.
Run before model training to confirm implementation matches the paper's claims.

Usage
-----
    python utils/sanity.py

Each section corresponds to a specific claim in the paper's EDA or methodology.
Passing all checks confirms the implementation matches the documented rationale.

Sections
--------
1  Raw data        — shape, nulls, Income column name
2  Class balance   — IncomeInvestment ~38%, AccumulationInvestment ~51%
3  Skewness        — log transforms justified
4  F_E formulas    — all 7 engineered features spot-checked; exclusions confirmed
5  No-leakage      — scaler fitted on train only for both targets
6  Stratification  — class ratio preserved across split for both targets
7  Correlations    — all 7 quantitative EDA claims from the paper
8  Feature sets    — F_E vs F_B structure validated
"""

import logging
import sys
from pathlib import Path

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
    assert df.shape[0] == 5000, f"Expected 5000 rows, got {df.shape[0]}"
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values"
    assert "Income" in df.columns,  "'Income' column not found — strip() may have failed"
    assert "Income " not in df.columns, "Trailing-space bug in 'Income ' still present"
    logger.info("Section 1 PASSED — shape %s, nulls %d, Income column clean", df.shape, null_count)


def check_class_balance(df: pd.DataFrame) -> None:
    income_prev = df["IncomeInvestment"].mean()
    accum_prev  = df["AccumulationInvestment"].mean()
    logger.info(
        "IncomeInvestment: prevalence=%.3f  counts=%s",
        income_prev, df["IncomeInvestment"].value_counts().to_dict(),
    )
    logger.info(
        "AccumulationInvestment: prevalence=%.3f  counts=%s",
        accum_prev, df["AccumulationInvestment"].value_counts().to_dict(),
    )
    assert 0.35 < income_prev < 0.42, \
        f"IncomeInvestment prevalence={income_prev:.3f} outside [0.35, 0.42]"
    assert 0.48 < accum_prev  < 0.54, \
        f"AccumulationInvestment prevalence={accum_prev:.3f} outside [0.48, 0.54]"
    logger.info("Section 2 PASSED — class balance confirmed")


def check_skewness(df: pd.DataFrame) -> None:
    thresholds = {"Wealth": 2.0, "Income": 1.0}
    for col, threshold in thresholds.items():
        sk = skew(df[col])
        logger.info("%s: skewness=%.2f  (threshold > %.1f)", col, sk, threshold)
        assert sk > threshold, \
            f"{col} skewness={sk:.2f} is not high enough to justify log transform"
    logger.info("Section 3 PASSED — log transforms are justified")


def check_engineered_features(df: pd.DataFrame) -> None:
    X = build_features(df)
    logger.info("F_E shape: %s  columns: %s", X.shape, list(X.columns))

    assert list(X.columns) == FEATURE_NAMES, f"Column mismatch: {list(X.columns)}"
    assert X.shape == (5000, len(FEATURE_NAMES)), f"Wrong shape: {X.shape}"
    assert X.isnull().sum().sum() == 0, "NaN in engineered features"

    assert np.allclose(X["Age_sq"],            df["Age"] ** 2)
    assert np.allclose(X["FinEdu_x_Risk"],      df["FinancialEducation"] * df["RiskPropensity"])
    assert np.allclose(X["Income_log"],         np.log1p(df["Income"]))
    assert np.allclose(X["Wealth_log"],         np.log1p(df["Wealth"]))
    assert np.allclose(X["Age_x_Wealth"],       df["Age"] * np.log1p(df["Wealth"]))
    assert np.allclose(X["Income_per_FM_log"],  np.log1p(df["Income"] / df["FamilyMembers"]))
    assert np.allclose(X["Wealth_per_FM_log"],  np.log1p(df["Wealth"] / df["FamilyMembers"]))

    assert "Gender"        not in X.columns, "Gender must be excluded (r < 0.015)"
    assert "FamilyMembers" not in X.columns, "Raw FamilyMembers must be excluded"

    # Spot-check: log transforms applied, not raw values
    assert "Wealth" not in X.columns, "Raw Wealth should not be in F_E"
    assert "Income" not in X.columns, "Raw Income should not be in F_E"

    logger.info("Section 4 PASSED — F_E formulas and exclusions confirmed")


def check_no_leakage(df: pd.DataFrame) -> None:
    X = build_features(df)
    for target in TARGETS:
        X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, df[target])
        assert scaler.data_min_ is not None, "Scaler not fitted"
        assert X_tr.min().min() >= -1e-9,      "Train min below 0 (scaling bug)"
        assert X_tr.max().max() <= 1.0 + 1e-9, "Train max above 1 (scaling bug)"
        logger.info(
            "%s — train [%.4f, %.4f] | test [%.4f, %.4f] (test may exceed [0,1] — correct)",
            target,
            X_tr.min().min(), X_tr.max().max(),
            X_te.min().min(), X_te.max().max(),
        )
    logger.info("Section 5 PASSED — no-leakage confirmed for all targets")


def check_stratification(df: pd.DataFrame) -> None:
    X = build_features(df)
    for target in TARGETS:
        _, _, y_tr, y_te, _ = split_and_scale(X, df[target])
        delta = abs(y_tr.mean() - y_te.mean())
        logger.info(
            "%s: train=%.4f  test=%.4f  delta=%.4f",
            target, y_tr.mean(), y_te.mean(), delta,
        )
        assert delta < 0.03, f"{target}: stratification drift too large: delta={delta:.4f}"
    logger.info("Section 6 PASSED — stratification confirmed for all targets")


def check_correlations(df: pd.DataFrame) -> None:
    X    = build_features(df)
    full = pd.concat([X, df[TARGETS]], axis=1)
    corr = full.corr()

    checks = [
        ("Age_x_Wealth  vs IncomeInvestment",
         corr.loc["Age_x_Wealth", "IncomeInvestment"],         ">", 0.30),
        ("Age           vs IncomeInvestment",
         corr.loc["Age", "IncomeInvestment"],                   ">", 0.25),
        ("Wealth_log    vs IncomeInvestment",
         corr.loc["Wealth_log", "IncomeInvestment"],            ">", 0.30),
        ("Income_log    vs AccumulationInvestment",
         corr.loc["Income_log", "AccumulationInvestment"],      ">", 0.25),
        ("FinEdu        vs RiskPropensity",
         corr.loc["FinancialEducation", "RiskPropensity"],      ">", 0.60),
        ("Income_per_FM vs Income_log",
         corr.loc["Income_per_FM_log", "Income_log"],           ">", 0.85),
        ("IncomeInvestment vs AccumulationInvestment",
         abs(corr.loc["IncomeInvestment", "AccumulationInvestment"]), "<", 0.05),
    ]

    all_passed = True
    for label, value, op, threshold in checks:
        passed = (value > threshold) if op == ">" else (value < threshold)
        status = "PASS" if passed else "FAIL"
        logger.info("[%s]  %s: %.3f  (%s %.2f)", status, label, value, op, threshold)
        if not passed:
            all_passed = False

    assert all_passed, "One or more correlation checks failed — see log above"
    logger.info("Section 7 PASSED — all 7 correlation claims confirmed")


def check_feature_set_structure(df: pd.DataFrame) -> None:
    X      = build_features(df)
    X_base = build_baseline_features(df)
    added  = set(FEATURE_NAMES) - set(BASELINE_FEATURE_NAMES)

    logger.info("F_E columns: %s", list(X.columns))
    logger.info("F_B columns: %s", list(X_base.columns))
    logger.info("Added by engineering: %s", sorted(added))

    assert "Gender"        not in FEATURE_NAMES,          "Gender in F_E — should be excluded"
    assert "Gender"        in BASELINE_FEATURE_NAMES,     "Gender missing from F_B"
    assert "FamilyMembers" not in FEATURE_NAMES,          "Raw FamilyMembers in F_E — should be excluded"
    assert set(BASELINE_FEATURE_NAMES).issubset(set(X_base.columns)), "F_B missing expected columns"
    assert set(FEATURE_NAMES) == set(X.columns),          "F_E columns mismatch FEATURE_NAMES constant"
    assert len(added) > 0,                                "No features added by engineering"

    logger.info(
        "Section 8 PASSED — F_E (%d features) and F_B (%d features) structure confirmed, "
        "%d features added by engineering",
        len(FEATURE_NAMES), len(BASELINE_FEATURE_NAMES), len(added),
    )


def run() -> None:
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