"""
Shared preprocessing utilities for BusinessCase2.

All feature engineering decisions are documented in AT_comments.md and
in the companion paper (Section 3 – Methodology).

Design principles
-----------------
* build_baseline_features → F_B (professor's set, ablation baseline)
* build_features          → F_E (theoretically motivated engineered set)
* Scalers fitted on training fold only — never on full dataset (Section 3.3)
* compute_cv_metrics clones the model every fold (prevents state accumulation)
* 10 outer folds for Wilcoxon test statistical power (Section 3.5)
* calibrate_model uses cv=5, NOT cv='prefit' (Section 3.6)
* Warnings suppressed globally — clean output only
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Suppress all warnings globally for clean output
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
    "Age", "Age_sq", "Age_x_Wealth",
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

# Confidence segmentation thresholds (Section 3 — recommendation pipeline)
CONFIDENCE_HIGH: float = 0.75
CONFIDENCE_LOW: float  = 0.40


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load Needs dataset, strip column whitespace, drop ID."""
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
    """Construct F_E — 10 theoretically motivated features. Section 3.2."""
    required = {"Age", "Income", "Wealth", "FamilyMembers", "FinancialEducation", "RiskPropensity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    X = df.copy()
    X["Wealth_log"]        = np.log1p(X["Wealth"])
    X["Income_log"]        = np.log1p(X["Income"])
    X["Age_sq"]            = X["Age"] ** 2
    X["Age_x_Wealth"]      = X["Age"] * X["Wealth_log"]
    X["Income_per_FM_log"] = np.log1p(X["Income"] / X["FamilyMembers"])
    X["Wealth_per_FM_log"] = np.log1p(X["Wealth"] / X["FamilyMembers"])
    X["FinEdu_x_Risk"]     = X["FinancialEducation"] * X["RiskPropensity"]
    return X[FEATURE_NAMES]


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct F_B — professor's 7-feature baseline. Section 3.2."""
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
    """Build F_E, F_B, targets and persist to data/feature_store/."""
    if df is None:
        df = load_data()
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    X_e = build_features(df)
    X_b = build_baseline_features(df)
    y   = df[TARGETS]
    for name, obj in [("F_E", X_e), ("F_B", X_b), ("targets", y)]:
        with open(FEATURE_STORE / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    logger.info("Feature store saved to %s", FEATURE_STORE)


def load_feature_store() -> tuple:
    """Load (X_engineered, X_baseline, targets_df) from feature store."""
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


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split WITHOUT scaling. Use for trees and Naive Bayes."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Stratified split + MinMaxScaler [0,1] on train only. Use for SVM and MLP."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_s, X_te_s, y_train, y_test, scaler


def split_and_standardize(X, y, test_size=0.2, random_state=42):
    """Stratified split + StandardScaler on train only. Use for Logistic Regression."""
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


def compute_cv_metrics(model, X_train, y_train, k_folds=N_OUTER_FOLDS) -> dict:
    """
    Stratified K-Fold CV with model cloning every fold.

    Model is cloned on every fold — prevents state accumulation across folds.
    If a Pipeline is passed, the scaler inside refits per fold automatically.
    Pass UNSCALED data when using a Pipeline.

    Returns raw per-fold lists for Wilcoxon tests.
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
    """Summarise raw fold scores into {metric: {mean, std}}."""
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in cv_metrics.items()}


def compute_test_metrics(model, X_test, y_test) -> dict:
    """Evaluate fitted model on test set at threshold=0.5."""
    y_hat = model.predict(X_test)
    return {
        "accuracy":  float(accuracy_score(y_test, y_hat)),
        "precision": float(precision_score(y_test, y_hat, zero_division=0)),
        "recall":    float(recall_score(y_test, y_hat, zero_division=0)),
        "f1":        float(f1_score(y_test, y_hat, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# MiFID II threshold selection
# ---------------------------------------------------------------------------


def select_threshold_pr_curve(model, X_test, y_test, precision_floor=PRECISION_FLOOR) -> dict:
    """
    Select threshold maximising F1 subject to Precision >= precision_floor.
    Fixing threshold at 0.5 ignores the MiFID II mis-selling cost. Section 3.4.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    y_scores = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]
    valid = precisions >= precision_floor
    if not valid.any():
        raise ValueError(
            f"No threshold achieves Precision >= {precision_floor:.2f}. "
            f"Max = {precisions.max():.3f}."
        )
    f1s  = 2 * precisions[valid] * recalls[valid] / np.clip(precisions[valid] + recalls[valid], 1e-9, None)
    best = np.argmax(f1s)
    return {
        "threshold": float(thresholds[valid][best]),
        "precision": float(precisions[valid][best]),
        "recall":    float(recalls[valid][best]),
        "f1":        float(f1s[best]),
        "precisions":  precisions,
        "recalls":     recalls,
        "thresholds":  thresholds,
    }


def apply_threshold(model, X_test, threshold: float) -> np.ndarray:
    """Apply custom threshold to predict_proba scores."""
    return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)


def get_propensity_scores(model, X) -> np.ndarray:
    """
    Extract calibrated propensity scores for ALL clients.
    Used by recommendation engine for ranking and confidence segmentation.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    return model.predict_proba(X)[:, 1]


def segment_by_confidence(propensity: np.ndarray) -> dict:
    """
    Segment clients into high/uncertain/low confidence groups.
    High (>0.75): automate recommendation
    Uncertain (0.40-0.75): route to human advisor
    Low (<0.40): no action
    Returns counts and indices for each segment.
    """
    high    = propensity > CONFIDENCE_HIGH
    low     = propensity < CONFIDENCE_LOW
    unsure  = ~high & ~low
    return {
        "high":    {"mask": high,   "count": int(high.sum()),   "indices": np.where(high)[0]},
        "unsure":  {"mask": unsure, "count": int(unsure.sum()), "indices": np.where(unsure)[0]},
        "low":     {"mask": low,    "count": int(low.sum()),    "indices": np.where(low)[0]},
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(model, X_train, y_train):
    """
    Post-hoc calibration via isotonic regression, cv=5.

    Uses cv=5 NOT cv='prefit'. cv='prefit' calibrates on the same data
    used to train the base model — calibration leakage. With cv=5 the base
    estimator is refitted internally across 5 folds. Section 3.6.

    Pass UNFITTED model — cv=5 fits it internally.
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_brier_score(model, X_test, y_test) -> float:
    """Brier score: lower=better, 0.25=no-skill baseline. Section 3.6."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Brier score requires predict_proba.")
    return float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))


# ---------------------------------------------------------------------------
# Label sensitivity
# ---------------------------------------------------------------------------


def flip_labels(y: pd.Series, flip_rate: float, random_state: int = 42) -> pd.Series:
    """Randomly flip flip_rate fraction of labels to simulate label noise."""
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
    Compute scale_pos_weight = n_neg/n_pos for XGBoost.
    Apply to IncomeInvestment only (38% positive). Section 2.1.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_hyperparameters(model, param_grid: dict, X_train, y_train,
                          n_iter: int = 20, cv: int = N_INNER_FOLDS,
                          scoring: str = "f1") -> tuple:
    """
    RandomizedSearchCV for hyperparameter tuning (inner CV loop).

    Parameters
    ----------
    model      : unfitted estimator or Pipeline
    param_grid : dict of parameter distributions
    X_train    : training features (unscaled if Pipeline)
    y_train    : training labels
    n_iter     : number of random configurations to try
    cv         : inner CV folds (default N_INNER_FOLDS=3)
    scoring    : optimisation metric (default 'f1')

    Returns
    -------
    (best_estimator, best_params, best_score)
    """
    from sklearn.model_selection import RandomizedSearchCV
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


# ---------------------------------------------------------------------------
# Result factory
# ---------------------------------------------------------------------------


def make_result_dict(
    model, scaler, cv_metrics_raw, test_metrics,
    y_test_true, y_test_pred, feature_names,
    target_name, model_name,
    y_test_proba=None,
    ablation=None,
    best_params=None,
    **extra,
) -> dict:
    """
    Canonical result dict pickled by every model script.

    y_test_proba : np.ndarray, calibrated propensity scores predict_proba[:,1]
                   Used by recommendation engine for ranking and segmentation.
    best_params  : dict, hyperparameter tuning results (None if no tuning).
    """
    result = {
        "model":              model,
        "scaler":             scaler,
        "cv_metrics_raw":     cv_metrics_raw,
        "cv_metrics_summary": summarise_cv(cv_metrics_raw),
        "test_metrics":       test_metrics,
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
    """Save result dict as pickle to data/pickled_files/<folder>/."""
    out_dir = PICKLE_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_name.lower()}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path


def load_result(folder: str, target_name: str) -> dict:
    """Load result dict from data/pickled_files/<folder>/."""
    path = PICKLE_ROOT / folder / f"{target_name.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No pickle at {path}. Run model script first.")
    with open(path, "rb") as f:
        return pickle.load(f)