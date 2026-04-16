"""
Shared preprocessing utilities for the KYC (Know Your Customer) investment
classification pipeline.

Theory
------
The pipeline targets two binary labels — IncomeInvestment and
AccumulationInvestment — on a 5,000-client retail banking dataset.

Raw monetary features (Income, Wealth) are strongly right-skewed. Log1p
transformations are applied to compress long tails and improve approximate
symmetry, which benefits distance-based models and models assuming linear
relationships in feature space.

Data splitting is performed using stratified sampling to preserve class
distribution across training and test sets. Hyperparameter tuning uses true
nested cross-validation: a RandomizedSearchCV inner loop (3-fold) is executed
independently within each of the 10 outer StratifiedKFold folds. This ensures
that no validation fold influences hyperparameter selection, producing
unbiased fold-level F1 estimates suitable for statistical comparisons such as
the Wilcoxon signed-rank test.

Predicted probabilities are optionally calibrated using isotonic regression
(cross-validated with cv=5), improving reliability of propensity scores.

For MiFID II compliance, investment recommendations must achieve precision
greater than or equal to 0.75. The decision threshold is selected on a held-out
validation split and never on the test set, preventing optimistic bias in
compliance reporting.

Implementation
--------------
This module provides grouped utilities for the full machine learning pipeline:

Data loading
- load_data: reads the Excel dataset and cleans column names.

Feature engineering
- build_features: constructs F_E (engineered feature set including age
  interactions, log-transformations, and per-capita features)
- build_baseline_features: constructs F_B (baseline feature set)

Feature store
- save_feature_store / load_feature_store: persist and retrieve precomputed
  feature matrices

Data splitting and scaling
- split_data: stratified split without scaling
- split_and_scale: MinMax scaling
- split_and_standardize: StandardScaler normalization

Cross-validation
- compute_cv_metrics: standard k-fold evaluation using model cloning
- nested_cv_with_tuning: full nested cross-validation with hyperparameter tuning

Threshold selection
- select_threshold_on_val: selects highest-F1 threshold satisfying precision
  constraint (>= 0.75)
- apply_threshold / evaluate_at_threshold: apply decision threshold to test set

Calibration
- calibrate_model: wraps estimator in isotonic CalibratedClassifierCV
- compute_brier_score: evaluates probabilistic calibration quality

Imbalance handling
- scale_pos_weight: computes class imbalance ratio for XGBoost
- flip_labels: utility for label-noise robustness experiments

Propensity scoring
- get_propensity_scores: extracts predicted probabilities
- segment_by_confidence: groups predictions by confidence level

Result management
- make_result_dict: constructs standardized evaluation dictionary
- save_result / load_result: serialize and retrieve model artifacts
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path(__file__).parent.parent / "Data" / "Dataset2_Needs.xls"
FEATURE_STORE: Path = Path(__file__).parent.parent / "data" / "feature_store"
PICKLE_ROOT: Path = Path(__file__).parent.parent / "data" / "pickled_files"

FEATURE_NAMES: list = [
    "Age", "Age_sq", "Age_x_Wealthlog",           
    "FinancialEducation", "RiskPropensity", "FinEdu_x_Risk",
    "Income_log", "Wealth_log",
    "Income_per_FM_log", "Wealth_per_FM_log",
]

BASELINE_FEATURE_NAMES: list = [
    "Age", "Gender", "FamilyMembers",
    "FinancialEducation", "RiskPropensity",
    "Income_log", "Wealth_log",
]

TARGETS: list = ["IncomeInvestment", "AccumulationInvestment"]

PRECISION_FLOOR: float = 0.75
N_OUTER_FOLDS: int = 10
N_INNER_FOLDS: int = 3

CONFIDENCE_HIGH: float = 0.75
CONFIDENCE_LOW: float  = 0.40


def no_skill_brier(y: pd.Series) -> float:
    """
    Compute the no-skill Brier baseline as p * (1 - p) where p is class
    prevalence.

    The baseline is not hardcoded to 0.25 but derived from the actual
    label distribution.  For IncomeInvestment (38% positive) this yields
    0.38 * 0.62 = 0.2356.
    """
    p = float(y.mean())
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """
    Load the raw Excel dataset, strip column-name whitespace, and drop the ID
    column if present.

    Returns
    -------
    pd.DataFrame
        Raw 5,000-row dataset with clean column names.

    Raises
    ------
    FileNotFoundError
        If the Excel file is not found at the expected path.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the engineered feature set F_E (10 features).

    The features are: Age, Age², Age × log1p(Wealth), FinancialEducation,
    RiskPropensity, FinancialEducation × RiskPropensity, log1p(Income),
    log1p(Wealth), log1p(Income / FamilyMembers), log1p(Wealth / FamilyMembers).

    Note: Age_x_Wealthlog = Age * log1p(Wealth).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing at minimum: Age, Income, Wealth, FamilyMembers,
        FinancialEducation, RiskPropensity.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the FEATURE_NAMES columns.

    Raises
    ------
    KeyError
        If any required source columns are absent from df.
    """
    required = {"Age", "Income", "Wealth", "FamilyMembers", "FinancialEducation", "RiskPropensity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    X = df.copy()
    X["Wealth_log"]           = np.log1p(X["Wealth"])
    X["Income_log"]           = np.log1p(X["Income"])
    X["Age_sq"]               = X["Age"] ** 2
    X["Age_x_Wealthlog"]      = X["Age"] * X["Wealth_log"]   # Age * log1p(Wealth)
    X["Income_per_FM_log"]    = np.log1p(X["Income"] / X["FamilyMembers"])
    X["Wealth_per_FM_log"]    = np.log1p(X["Wealth"] / X["FamilyMembers"])
    X["FinEdu_x_Risk"]        = X["FinancialEducation"] * X["RiskPropensity"]
    return X[FEATURE_NAMES]


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the baseline feature set F_B (7 features).

    Includes the 7 features from the professor's specification: Age, Gender,
    FamilyMembers, FinancialEducation, RiskPropensity, log1p(Income),
    log1p(Wealth).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing at minimum: Age, Gender, FamilyMembers,
        FinancialEducation, RiskPropensity, Income, Wealth.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the BASELINE_FEATURE_NAMES columns.

    Raises
    ------
    KeyError
        If any required source columns are absent from df.
    """
    required = {"Age", "Gender", "FamilyMembers", "FinancialEducation", "RiskPropensity", "Income", "Wealth"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    X = df.copy()
    X["Income_log"] = np.log1p(X["Income"])
    X["Wealth_log"] = np.log1p(X["Wealth"])
    return X[BASELINE_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------


def save_feature_store(df: Optional[pd.DataFrame] = None) -> None:
    """
    Build F_E, F_B, and targets from the raw dataframe and pickle them to the
    feature store directory.

    If ``df`` is None, the raw dataset is loaded automatically via
    :func:`load_data`.  The feature store directory is created if it does not
    already exist.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded raw dataset.  Loaded from disk when omitted.
    """
    if df is None:
        df = load_data()
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    X_e = build_features(df)
    X_b = build_baseline_features(df)
    y   = df[TARGETS]
    for name, obj in [("F_E", X_e), ("F_B", X_b), ("targets", y)]:
        with open(FEATURE_STORE / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)


def load_feature_store() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deserialise and return the pre-computed feature sets as a three-element
    tuple (F_E, F_B, targets).

    Returns
    -------
    tuple of pd.DataFrame
        (F_E, F_B, targets) in that order.

    Raises
    ------
    FileNotFoundError
        If :func:`save_feature_store` has not been run first.
    """
    paths = [FEATURE_STORE / f"{n}.pkl" for n in ("F_E", "F_B", "targets")]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run save_feature_store() first.")
    result = []
    for p in paths:
        with open(p, "rb") as f:
            result.append(pickle.load(f))
    return tuple(result)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split WITHOUT scaling."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, MinMaxScaler]:
    """Stratified split + MinMaxScaler fitted on training data only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_s, X_te_s, y_train, y_test, scaler


def split_and_standardize(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Stratified split + StandardScaler fitted on training data only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_s, X_te_s, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def compute_cv_metrics(model: Any, X_train: pd.DataFrame, y_train: pd.Series, k_folds: int = N_OUTER_FOLDS) -> dict:
    """
    Stratified K-Fold cross-validation with model cloning on every fold.

    Pass UNSCALED data when using a Pipeline — the Pipeline refits the scaler
    independently on each fold's training split.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Unfitted model to clone and evaluate on each fold.
    X_train : pd.DataFrame
        Full training feature matrix (not scaled when using a Pipeline).
    y_train : pd.Series
        Full training labels.
    k_folds : int
        Number of outer CV folds (default: N_OUTER_FOLDS = 10).

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1 — each mapping to a list of
        k_folds float scores.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        y_hat = fold_model.predict(X_val)

        metrics["accuracy"].append(accuracy_score(y_val, y_hat))
        metrics["precision"].append(precision_score(y_val, y_hat, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_hat, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_hat, zero_division=0))

    return metrics


def summarise_cv(cv_metrics: dict) -> dict:
    """
    Summarise per-fold CV metrics into mean and standard deviation.

    Uses ddof=1 (sample standard deviation over k fold observations)
    rather than population std.  For 10 folds this changes std by a factor of
    sqrt(10/9) ≈ 1.054 — a small but statistically correct adjustment.

    Parameters
    ----------
    cv_metrics : dict
        Output of :func:`compute_cv_metrics` or :func:`nested_cv_with_tuning`.

    Returns
    -------
    dict
        Keys match cv_metrics; values are dicts with 'mean' and 'std'.
    """
    return {
        m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
        for m, v in cv_metrics.items()
    }


def compute_test_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted model on the test set at the default 0.5 threshold.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1 (all floats).
    """
    y_hat = model.predict(X_test)
    return {
        "accuracy":  float(accuracy_score(y_test, y_hat)),
        "precision": float(precision_score(y_test, y_hat, zero_division=0)),
        "recall":    float(recall_score(y_test, y_hat, zero_division=0)),
        "f1":        float(f1_score(y_test, y_hat, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_hyperparameters(
    model: Any,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = N_INNER_FOLDS,
    scoring: str = "f1",
    error_score: str = "raise",
) -> Tuple[Any, dict, float]:
    """
    Run RandomizedSearchCV as the inner CV loop for hyperparameter tuning.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Unfitted estimator whose hyperparameters will be searched.
    param_grid : dict
        Hyperparameter search space passed to RandomizedSearchCV.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    n_iter : int
        Number of random parameter settings to sample.
    cv : int
        Number of inner CV folds.
    scoring : str
        Metric to optimise (default: 'f1').

    Returns
    -------
    tuple
        (best_estimator, best_params, best_score)
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def nested_cv_with_tuning(
    model_factory: Any,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    k_folds: int = N_OUTER_FOLDS,
    inner_cv: int = N_INNER_FOLDS,
) -> dict:
    """
    True nested cross-validation with inner hyperparameter tuning.

    Hyperparameter search runs inside each outer fold independently, operating
    only on that fold's training data.  The resulting fold-level CV F1 scores
    are unbiased — no outer-fold observation has been seen during HP selection
    — making them valid for downstream Wilcoxon model-comparison tests.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable that returns a fresh unfitted estimator or
        Pipeline.  Called once per outer fold to ensure independence.
    param_grid : dict
        Hyperparameter search space.
    X_train : pd.DataFrame
        Full training features (unscaled if using a Pipeline).
    y_train : pd.Series
        Full training labels.
    n_iter : int
        Random search iterations per outer fold.
    k_folds : int
        Number of outer CV folds (default: N_OUTER_FOLDS = 10).
    inner_cv : int
        Number of inner CV folds for HP tuning (default: N_INNER_FOLDS = 3).

    Returns
    -------
    dict
        Keys:
        - ``cv_metrics_raw``: dict mapping metric name to list of fold scores.
        - ``best_params_per_fold``: list of best-param dicts, one per fold.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    best_params_per_fold = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        # HP tuning inside this fold — val_idx never seen
        search = RandomizedSearchCV(
            estimator=model_factory(),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42),
            scoring="f1",
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best_params_per_fold.append(search.best_params_)

        y_hat = search.best_estimator_.predict(X_val)
        metrics["accuracy"].append(accuracy_score(y_val, y_hat))
        metrics["precision"].append(precision_score(y_val, y_hat, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_hat, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_hat, zero_division=0))

        logger.debug(
            "Fold %d/%d — F1=%.4f  best_params=%s",
            fold_idx + 1, k_folds, metrics["f1"][-1], search.best_params_,
        )

    return {"cv_metrics_raw": metrics, "best_params_per_fold": best_params_per_fold}


# ---------------------------------------------------------------------------
# MiFID II threshold selection 
# ---------------------------------------------------------------------------


def select_threshold_on_val(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    precision_floor: float = PRECISION_FLOOR,
) -> dict:
    """
    Select the MiFID II compliance threshold on a VALIDATION set.

    The threshold is chosen on ``(X_val, y_val)``, never on the final test set,
    to prevent optimistic bias.  The returned threshold is then applied to the
    held-out test set via :func:`evaluate_at_threshold` for unbiased reporting.

    Typical workflow in each model script::

        X_tr, X_te, y_tr, y_te = split_data(X, y)           # 80 / 20
        X_tr2, X_val, y_tr2, y_val = split_data(X_tr, y_tr, test_size=0.2)
        model.fit(X_tr2, y_tr2)
        thr_info = select_threshold_on_val(model, X_val, y_val)
        # report metrics on X_te at thr_info['threshold'] — unbiased

    Parameters
    ----------
    model : fitted sklearn estimator
        Must implement ``predict_proba``.
    X_val : pd.DataFrame
        Validation features (not used during training).
    y_val : pd.Series
        Validation labels.
    precision_floor : float
        Minimum required precision (default: PRECISION_FLOOR = 0.75).

    Returns
    -------
    dict
        Keys: threshold, val_precision, val_recall, val_f1.

    Raises
    ------
    ValueError
        If no threshold achieves the required precision on the validation set.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    y_scores = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]
    valid = precisions >= precision_floor
    if not valid.any():
        raise ValueError(
            f"No threshold achieves Precision >= {precision_floor:.2f} on validation set. "
            f"Max = {precisions.max():.3f}."
        )
    f1s  = 2 * precisions[valid] * recalls[valid] / np.clip(precisions[valid] + recalls[valid], 1e-9, None)
    best = np.argmax(f1s)
    return {
        "threshold":     float(thresholds[valid][best]),
        "val_precision": float(precisions[valid][best]),
        "val_recall":    float(recalls[valid][best]),
        "val_f1":        float(f1s[best]),
    }


def apply_threshold(model: Any, X_test: pd.DataFrame, threshold: float) -> np.ndarray:
    """
    Apply a pre-fixed probability threshold to the test set.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must implement ``predict_proba``.
    X_test : pd.DataFrame
    threshold : float

    Returns
    -------
    np.ndarray
        Binary integer predictions (0 or 1).
    """
    return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)


def evaluate_at_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """
    Compute unbiased test-set metrics at a pre-fixed threshold.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series
    threshold : float
        Threshold fixed from the validation set; never optimised on the test set.

    Returns
    -------
    dict
        Keys: threshold, precision, recall, f1.
    """
    y_pred = apply_threshold(model, X_test, threshold)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }


def get_propensity_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return calibrated propensity scores (P(positive class)) for the given data.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must implement ``predict_proba``.
    X : pd.DataFrame

    Returns
    -------
    np.ndarray
        Shape (n_samples,) of floats in [0, 1].

    Raises
    ------
    ValueError
        If the model does not implement ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    return model.predict_proba(X)[:, 1]


def segment_by_confidence(propensity: np.ndarray) -> dict:
    """
    Segment clients into three confidence tiers based on propensity scores.

    * High (> CONFIDENCE_HIGH = 0.75): automate recommendation.
    * Uncertain: route to human advisor.
    * Low (< CONFIDENCE_LOW = 0.40): no action.

    Parameters
    ----------
    propensity : np.ndarray
        Propensity scores from :func:`get_propensity_scores`.

    Returns
    -------
    dict
        Keys: 'high', 'unsure', 'low' — each with sub-keys 'mask', 'count',
        'indices'.
    """
    high   = propensity > CONFIDENCE_HIGH
    low    = propensity < CONFIDENCE_LOW
    unsure = ~high & ~low
    return {
        "high":   {"mask": high,   "count": int(high.sum()),   "indices": np.where(high)[0]},
        "unsure": {"mask": unsure, "count": int(unsure.sum()), "indices": np.where(unsure)[0]},
        "low":    {"mask": low,    "count": int(low.sum()),    "indices": np.where(low)[0]},
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """
    Apply post-hoc isotonic calibration using a 5-fold internal CV.

    ``cv='prefit'`` is deliberately avoided — it calibrates on the same data
    used to train the base model, inflating quality estimates.  Instead, cv=5
    refits the base model internally on each calibration fold.  Pass an
    UNFITTED model.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted base estimator.
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated wrapper.
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_brier_score(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Compute the Brier score (mean squared probability error) on the test set.

    Lower is better; the no-skill baseline equals p * (1 - p) where p is
    class prevalence (see :func:`no_skill_brier`).

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the model does not implement ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Brier score requires predict_proba.")
    return float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))


# ---------------------------------------------------------------------------
# Label sensitivity
# ---------------------------------------------------------------------------


def flip_labels(y: pd.Series, flip_rate: float, random_state: int = 42) -> pd.Series:
    """
    Randomly flip a fraction of binary labels to simulate label noise.

    Used for label-noise sensitivity analysis to assess how robustly each model
    handles annotation errors.

    Parameters
    ----------
    y : pd.Series
        Original binary labels.
    flip_rate : float
        Fraction of labels to flip, in [0, 1].
    random_state : int
        Seed for the random number generator (default: 42).

    Returns
    -------
    pd.Series
        Copy of y with floor(flip_rate * len(y)) labels flipped.
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(np.floor(flip_rate * len(y)))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
    return y_noisy


# ---------------------------------------------------------------------------
# Imbalance correction
# ---------------------------------------------------------------------------


def scale_pos_weight(y: pd.Series) -> float:
    """
    Compute the XGBoost ``scale_pos_weight`` as n_neg / n_pos.

    Should be applied to IncomeInvestment only (38% positive); the
    AccumulationInvestment target is approximately balanced.

    Parameters
    ----------
    y : pd.Series
        Binary target labels.

    Returns
    -------
    float
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


# ---------------------------------------------------------------------------
# Result factory
# ---------------------------------------------------------------------------


def make_result_dict(
    model: Any,
    scaler: Any,
    cv_metrics_raw: dict,
    test_metrics: dict,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    feature_names: list,
    target_name: str,
    model_name: str,
    y_test_proba: Optional[np.ndarray] = None,
    ablation: Optional[dict] = None,
    best_params: Optional[dict] = None,
    threshold_metrics: Optional[dict] = None,
    **extra: Any,
) -> dict:
    """
    Assemble a standardised result dictionary from a fitted model run.

    Stores the model object, scaler, raw per-fold CV metrics, their summary
    statistics, test-set metrics at the default 0.5 threshold, optional
    MiFID II threshold metrics, per-observation predictions and probabilities,
    feature names, ablation sub-results, and any extra keyword arguments.

    The ``threshold_metrics`` entry holds the output of
    :func:`evaluate_at_threshold` evaluated at the validation-selected threshold
    , as distinct from the default-threshold ``test_metrics``.

    Parameters
    ----------
    model : fitted estimator
    scaler : fitted scaler or None
        None when scaling is handled inside a Pipeline.
    cv_metrics_raw : dict
        Per-fold metric lists from CV.
    test_metrics : dict
        Metrics at the default 0.5 threshold.
    y_test_true : np.ndarray
        Ground-truth test labels.
    y_test_pred : np.ndarray
        Predicted labels at default threshold.
    feature_names : list
        Names of the features used.
    target_name : str
    model_name : str
    y_test_proba : np.ndarray, optional
        Predicted probabilities for the positive class.
    ablation : dict, optional
        Nested result dicts for alternative feature sets.
    best_params : dict, optional
        Best hyperparameters from tuning.
    threshold_metrics : dict, optional
        Metrics at the MiFID II compliance threshold.
    **extra
        Additional model-specific fields (e.g., shap_values, feature_importances).

    Returns
    -------
    dict
    """
    result = {
        "model":              model,
        "scaler":             scaler,
        "cv_metrics_raw":     cv_metrics_raw,
        "cv_metrics_summary": summarise_cv(cv_metrics_raw),
        "test_metrics":       test_metrics,
        "threshold_metrics":  threshold_metrics,
        "y_test_true":        y_test_true,
        "y_test_pred":        y_test_pred,
        "y_test_proba":       y_test_proba,
        "feature_names":      feature_names,
        "target_name":        target_name,
        "model_name":         model_name,
        "ablation":           ablation,
        "best_params":        best_params,
    }
    result.update(extra)
    return result


def save_result(result: dict, folder: str, target_name: str) -> Path:
    """
    Pickle a result dict to ``PICKLE_ROOT / folder / {target_name.lower()}.pkl``.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    result : dict
        Output of :func:`make_result_dict`.
    folder : str
        Subdirectory name under PICKLE_ROOT (e.g., 'linear_reg').
    target_name : str
        Target column name; used as the file stem.

    Returns
    -------
    Path
        Resolved path of the written pickle file.
    """
    out_dir = PICKLE_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_name.lower()}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path


def load_result(folder: str, target_name: str) -> dict:
    """
    Load and return a previously pickled result dict.

    Parameters
    ----------
    folder : str
        Subdirectory name under PICKLE_ROOT.
    target_name : str
        Target column name used as the file stem when saving.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If the expected pickle file does not exist.
    """
    path = PICKLE_ROOT / folder / f"{target_name.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No pickle at {path}.")
    with open(path, "rb") as f:
        return pickle.load(f)
