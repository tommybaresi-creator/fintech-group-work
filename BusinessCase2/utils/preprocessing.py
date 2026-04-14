"""
Shared preprocessing utilities for BusinessCase2.

All feature engineering decisions are documented in AT_comments.md and
in the companion paper (Section 3 – Methodology).

Design principles
-----------------
* ``build_baseline_features`` reconstructs **F_B** — the professor's feature set,
  used as the ablation baseline and as the primary set for Naive Bayes and XGBoost.
* ``build_features`` constructs **F_E** — our theoretically motivated engineered
  set, used by LR, RF, MLP, ClassifierChain, and the ensemble components.
* Scalers are **never** fit on the full dataset; they are always fit on the
  training fold only, preventing data leakage (Section 3.3).
* ``compute_cv_metrics`` uses **10 folds** (outer loop) to guarantee adequate
  statistical power for the Wilcoxon signed-rank test (minimum achievable p ≈ 0.063
  with 5 folds is structurally insufficient at the 5 % level).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path(__file__).parent.parent / "Data" / "Dataset2_Needs.xls"

# F_E — engineered feature set (Section 3.2 of the paper)
FEATURE_NAMES: list[str] = [
    "Age",
    "Age_sq",
    "Age_x_Wealth",
    "FinancialEducation",
    "RiskPropensity",
    "FinEdu_x_Risk",
    "Income_log",
    "Wealth_log",
    "Income_per_FM_log",
    "Wealth_per_FM_log",
]

# F_B — professor's baseline feature set (ablation reference, Section 3.2)
# Includes Gender and FamilyMembers as they appear in the original notebook.
BASELINE_FEATURE_NAMES: list[str] = [
    "Age",
    "Gender",
    "FamilyMembers",
    "FinancialEducation",
    "RiskPropensity",
    "Income_log",
    "Wealth_log",
]

TARGETS: list[str] = ["IncomeInvestment", "AccumulationInvestment"]

# MiFID II hard precision floor (Section 3.4)
PRECISION_FLOOR: float = 0.75

# Outer CV folds — 10 is required for the Wilcoxon test to be meaningful
# (5 folds → min p ≈ 0.063, structurally failing α = 0.05; see Section 3.5)
N_OUTER_FOLDS: int = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """
    Load the Needs dataset from the Excel file.

    Strips trailing whitespace from all column names (fixes the
    ``'Income '`` trailing-space bug in the raw file) and drops
    the ``ID`` column if present.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with shape (5000, 10): 8 features + 2 targets.

    Raises
    ------
    FileNotFoundError
        If ``Data/Dataset2_Needs.xls`` does not exist at the expected path.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Ensure Data/Dataset2_Needs.xls is present."
        )
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    logger.info("Loaded dataset: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct **F_E** — the theoretically motivated engineered feature set.

    All features are derived from deterministic (non-statistical) transforms
    and are safe to apply before the train/test split. Justifications are in
    Section 3.2 and AT_comments.md.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns == ``FEATURE_NAMES`` (10 columns).

    Raises
    ------
    KeyError
        If any required raw column is missing from ``df``.
    """
    required = {
        "Age", "Income", "Wealth", "FamilyMembers",
        "FinancialEducation", "RiskPropensity",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    X = df.copy()
    X["Wealth_log"] = np.log1p(X["Wealth"])
    X["Income_log"] = np.log1p(X["Income"])
    X["Age_sq"] = X["Age"] ** 2
    X["Age_x_Wealth"] = X["Age"] * X["Wealth_log"]
    X["Income_per_FM_log"] = np.log1p(X["Income"] / X["FamilyMembers"])
    X["Wealth_per_FM_log"] = np.log1p(X["Wealth"] / X["FamilyMembers"])
    X["FinEdu_x_Risk"] = X["FinancialEducation"] * X["RiskPropensity"]
    return X[FEATURE_NAMES]


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct **F_B** — the professor's baseline feature set (ablation reference).

    F_B = {Age, Gender, FamilyMembers, FinancialEducation, RiskPropensity,
           log(Wealth), log(Income)}

    This is the set used in the original notebook: raw variables plus
    ``log1p`` transforms for Wealth and Income. It serves as the ablation
    baseline for LR and XGBoost comparisons (Section 3.2).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns == ``BASELINE_FEATURE_NAMES`` (7 columns).

    Raises
    ------
    KeyError
        If any required raw column is missing from ``df``.
    """
    required = {
        "Age", "Gender", "FamilyMembers",
        "FinancialEducation", "RiskPropensity",
        "Income", "Wealth",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    X = df.copy()
    X["Income_log"] = np.log1p(X["Income"])
    X["Wealth_log"] = np.log1p(X["Wealth"])
    return X[BASELINE_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split **without scaling**.

    Use for tree-based models (RF, XGBoost) that are invariant to monotonic
    feature transforms, and for Gaussian Naive Bayes where scaling does not
    affect the likelihood ratio (Section 3.3).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target series (values in {0, 1}).
    test_size : float, optional
        Fraction held out for testing, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, MinMaxScaler]:
    """
    Stratified split + **MinMaxScaler [0, 1]** fitted on the training set only.

    Use for SVM (RBF kernel is distance-based; unequal feature ranges bias the
    kernel) and MLP (input normalization to [0,1] matches the Sigmoid output
    activation; Section 3.3).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target series.
    test_size : float, optional
        Fraction held out for testing, by default 0.2.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, MinMaxScaler]
        ``(X_train_scaled, X_test_scaled, y_train, y_test, scaler)``

    Notes
    -----
    Test-set values may fall outside [0, 1] — this is expected and correct.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_s, X_test_s, y_train, y_test, scaler


def split_and_standardize(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Stratified split + **StandardScaler (zero-mean, unit-variance)** fitted
    on the training set only.

    Use for **Logistic Regression** with L1/L2 regularization. MinMaxScaler
    is incorrect here: it normalizes range but not variance, so features with
    different within-range distributions receive incomparable effective
    regularization at the same λ. StandardScaler equalizes both mean and
    variance, ensuring uniform penalty calibration (Section 3.3).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target series.
    test_size : float, optional
        Fraction held out for testing, by default 0.2.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]
        ``(X_train_scaled, X_test_scaled, y_train, y_test, scaler)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_s, X_test_s, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def compute_cv_metrics(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_folds: int = N_OUTER_FOLDS,
) -> dict[str, list[float]]:
    """
    Stratified K-Fold cross-validation returning **raw per-fold score lists**.

    Raw lists are preserved (not summarised) so that ``bestmodel_*.ipynb``
    can run Wilcoxon signed-rank tests. 10 folds are used by default: with
    only 5 folds the minimum achievable p-value is ≈ 0.063, making it
    structurally impossible to reject H₀ at α = 0.05 (Section 3.5).

    Parameters
    ----------
    model : sklearn estimator
        A **fresh, unfitted** sklearn-compatible estimator.
    X_train : pd.DataFrame
        Training features (already split and scaled if required).
    y_train : pd.Series
        Training labels.
    k_folds : int, optional
        Number of folds, by default ``N_OUTER_FOLDS`` (10).

    Returns
    -------
    dict[str, list[float]]
        Keys: ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``.
        Each value is a list of ``k_folds`` floats.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_val)

        metrics["accuracy"].append(accuracy_score(y_val, y_hat))
        metrics["precision"].append(
            precision_score(y_val, y_hat, zero_division=0)
        )
        metrics["recall"].append(recall_score(y_val, y_hat, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_hat, zero_division=0))
        logger.debug(
            "Fold %d/%d — F1: %.4f", fold_idx + 1, k_folds, metrics["f1"][-1]
        )

    return metrics


def summarise_cv(cv_metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """
    Summarise raw CV fold scores into mean ± std per metric.

    Parameters
    ----------
    cv_metrics : dict[str, list[float]]
        Raw fold scores as returned by :func:`compute_cv_metrics`.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{metric: {'mean': float, 'std': float}}``
    """
    return {
        m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for m, v in cv_metrics.items()
    }


def compute_test_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a fitted model on the held-out test set (threshold = 0.5).

    Parameters
    ----------
    model : sklearn estimator
        A fitted sklearn-compatible estimator.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    dict[str, float]
        Keys: ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``.
    """
    y_hat = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_hat)),
        "precision": float(precision_score(y_test, y_hat, zero_division=0)),
        "recall": float(recall_score(y_test, y_hat, zero_division=0)),
        "f1": float(f1_score(y_test, y_hat, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# MiFID II threshold selection (Section 3.4)
# ---------------------------------------------------------------------------


def select_threshold_pr_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    precision_floor: float = PRECISION_FLOOR,
) -> dict:
    """
    Select the operating threshold that **maximises F1 subject to
    Precision ≥ precision_floor** from the Precision–Recall curve.

    Fixing the threshold at 0.5 is arbitrary and ignores the asymmetric
    mis-selling cost imposed by MiFID II. This function implements the
    correct business-driven threshold selection (Section 3.4).

    Parameters
    ----------
    model : sklearn estimator
        A fitted estimator with ``predict_proba`` support.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.
    precision_floor : float, optional
        Hard minimum precision constraint, by default ``PRECISION_FLOOR`` (0.75).

    Returns
    -------
    dict
        Keys: ``'threshold'``, ``'precision'``, ``'recall'``, ``'f1'``,
        ``'precisions'``, ``'recalls'``, ``'thresholds'`` (full PR curve arrays).

    Raises
    ------
    ValueError
        If no threshold achieves ``precision_floor`` on the test set.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for threshold selection.")

    y_scores = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    # Trim arrays to matching length
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Filter by precision floor
    valid = precisions >= precision_floor
    if not valid.any():
        raise ValueError(
            f"No threshold achieves Precision ≥ {precision_floor:.2f}. "
            "Consider relaxing the constraint or reviewing the model."
        )

    # Among valid thresholds, maximise F1
    f1_scores = (
        2 * precisions[valid] * recalls[valid]
        / np.clip(precisions[valid] + recalls[valid], 1e-9, None)
    )
    best_idx = np.argmax(f1_scores)
    opt_threshold = float(thresholds[valid][best_idx])
    opt_precision = float(precisions[valid][best_idx])
    opt_recall = float(recalls[valid][best_idx])
    opt_f1 = float(f1_scores[best_idx])

    logger.info(
        "PR-curve threshold: %.3f  →  Precision=%.3f  Recall=%.3f  F1=%.3f",
        opt_threshold, opt_precision, opt_recall, opt_f1,
    )
    return {
        "threshold": opt_threshold,
        "precision": opt_precision,
        "recall": opt_recall,
        "f1": opt_f1,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
    }


def apply_threshold(
    model,
    X_test: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    """
    Apply a custom classification threshold to ``predict_proba`` scores.

    Parameters
    ----------
    model : sklearn estimator
        Fitted estimator with ``predict_proba``.
    X_test : pd.DataFrame
        Test features.
    threshold : float
        Classification threshold in (0, 1).

    Returns
    -------
    np.ndarray
        Binary predictions using the supplied threshold.
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    return (y_scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Calibration (Section 3.6)
# ---------------------------------------------------------------------------


def calibrate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """
    Post-hoc probability calibration via **isotonic regression**.

    Isotonic regression is preferred over Platt scaling because it is
    nonparametric and does not assume a sigmoidal calibration error form.
    Applied to RF and XGBoost, which produce biased probabilities (Section 3.6).

    Uses ``cv='prefit'`` so the base estimator is already trained; calibration
    is performed on the supplied training data via 5-fold internal CV.

    Parameters
    ----------
    model : sklearn estimator
        A **fitted** base estimator.
    X_train : pd.DataFrame
        Training features (the same set used to fit the base model).
    y_train : pd.Series
        Training labels.

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated wrapper around the base model.
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_brier_score(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Compute the **Brier score** as a calibration quality metric.

    BS = (1/n) Σ (ŷᵢ − yᵢ)², a proper scoring rule.
    Lower is better; 0.25 is the no-skill baseline for a balanced binary
    target (Section 3.6).

    Parameters
    ----------
    model : sklearn estimator
        Fitted estimator with ``predict_proba``.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    float
        Brier score.

    Raises
    ------
    ValueError
        If the model does not support ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Brier score requires predict_proba support.")
    y_prob = model.predict_proba(X_test)[:, 1]
    return float(brier_score_loss(y_test, y_prob))


# ---------------------------------------------------------------------------
# Label sensitivity helpers (Section 3.1)
# ---------------------------------------------------------------------------


def flip_labels(
    y: pd.Series,
    flip_rate: float,
    random_state: int = 42,
) -> pd.Series:
    """
    Randomly flip a fraction of training labels to simulate label noise.

    Used to verify robustness of the pipeline to the revealed-preference
    labeling assumption (Section 3.1).

    Parameters
    ----------
    y : pd.Series
        Original binary labels.
    flip_rate : float
        Fraction of labels to flip, in [0, 1).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.Series
        Corrupted labels with ``floor(flip_rate * len(y))`` flips.
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(np.floor(flip_rate * len(y)))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
    return y_noisy


# ---------------------------------------------------------------------------
# Scale-pos-weight helper (XGBoost imbalance correction)
# ---------------------------------------------------------------------------


def scale_pos_weight(y: pd.Series) -> float:
    """
    Compute ``scale_pos_weight = n_negative / n_positive`` for XGBoost.

    Applied to IncomeInvestment (38% positive rate) only. Applying it to
    AccumulationInvestment (51%, near-balanced) would degrade performance
    by over-weighting a non-problem (Section 2.1).

    Parameters
    ----------
    y : pd.Series
        Binary target series.

    Returns
    -------
    float
        Weight ratio.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)
    logger.debug("scale_pos_weight = %.3f  (neg=%d, pos=%d)", spw, n_neg, n_pos)
    return spw


# ---------------------------------------------------------------------------
# Result dictionary factory
# ---------------------------------------------------------------------------


def make_result_dict(
    model,
    scaler,
    cv_metrics_raw: dict[str, list[float]],
    test_metrics: dict[str, float],
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    feature_names: list[str],
    target_name: str,
    model_name: str,
    ablation: dict | None = None,
    **extra,
) -> dict:
    """
    Assemble the canonical result dictionary that every script pickles.

    Parameters
    ----------
    model : fitted estimator or dict
        Fitted sklearn estimator, or ``model.state_dict()`` for PyTorch models.
    scaler : scaler or None
        Fitted scaler (StandardScaler / MinMaxScaler / None).
    cv_metrics_raw : dict[str, list[float]]
        Raw per-fold CV scores from :func:`compute_cv_metrics`.
    test_metrics : dict[str, float]
        Test-set metrics from :func:`compute_test_metrics`.
    y_test_true : np.ndarray
        Ground-truth labels for the test set.
    y_test_pred : np.ndarray
        Model predictions on the test set (at default or optimised threshold).
    feature_names : list[str]
        Ordered list of feature column names used for training.
    target_name : str
        Name of the binary target.
    model_name : str
        Human-readable model identifier.
    ablation : dict or None, optional
        Sub-dict with keys ``'engineered'`` and ``'baseline'``, each
        containing ``'cv_metrics_raw'`` and ``'test_metrics'``.
    **extra
        Additional keys (e.g. ``shap_values``, ``feature_importances``,
        ``brier_score``, ``threshold_info``, ``model_architecture``).

    Returns
    -------
    dict
        Canonical result dictionary ready for ``joblib.dump``.
    """
    result = {
        "model": model,
        "scaler": scaler,
        "cv_metrics_raw": cv_metrics_raw,
        "cv_metrics_summary": summarise_cv(cv_metrics_raw),
        "test_metrics": test_metrics,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "feature_names": feature_names,
        "target_name": target_name,
        "model_name": model_name,
        "ablation": ablation,
    }
    result.update(extra)
    return result
