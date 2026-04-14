"""
Shared preprocessing utilities for BusinessCase2.

Design principles
-----------------
* ``build_baseline_features`` reconstructs **F_B** — the professor's feature set,
  used as the ablation baseline and as the primary set for Naive Bayes and XGBoost.
* ``build_features`` constructs **F_E** — our theoretically motivated engineered
  set, used by LR, RF, MLP, ClassifierChain, and the ensemble components.
* Scalers are **never** fit on the full dataset; they are always fit on the
  training fold only, preventing data leakage.
* ``compute_cv_metrics`` **clones** the model on every fold to prevent state
  accumulation across folds — critical for ensemble models and calibrated wrappers.
* ``compute_cv_metrics`` uses **10 folds** (outer loop) to guarantee adequate
  statistical power for the Wilcoxon signed-rank test (minimum achievable p ≈ 0.063
  with 5 folds is structurally insufficient at the 5% level).
* ``calibrate_model`` uses ``cv=5`` (NOT ``cv='prefit'``) to avoid fitting the
  isotonic calibrator on the same data used to train the base model.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path(__file__).parent.parent / "Data" / "Dataset2_Needs.xls"
FEATURE_STORE: Path = Path(__file__).parent.parent / "data" / "feature_store"

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """
    Load the Needs dataset. Strips column whitespace and drops ID.

    Returns
    -------
    pd.DataFrame  shape (5000, 9)

    Raises
    ------
    FileNotFoundError
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    logger.info("Loaded: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct F_E — the theoretically motivated engineered feature set.

    All transforms are deterministic and safe to apply before splitting.

    Returns
    -------
    pd.DataFrame  columns == FEATURE_NAMES  (10 features)
    """
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
    """
    Construct F_B — the professor's baseline feature set (ablation reference).

    F_B = {Age, Gender, FamilyMembers, FinancialEducation, RiskPropensity,
           log(Wealth), log(Income)}

    Returns
    -------
    pd.DataFrame  columns == BASELINE_FEATURE_NAMES  (7 features)
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
    Build F_E, F_B, and targets and persist to data/feature_store/.

    Called once before model training. All model scripts load from here.
    Raw construction involves no statistical fitting — safe to apply on
    the full dataset before splitting.
    """
    if df is None:
        df = load_data()

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)

    X_e = build_features(df)
    X_b = build_baseline_features(df)
    y   = df[TARGETS]

    for name, obj in [("F_E", X_e), ("F_B", X_b), ("targets", y)]:
        path = FEATURE_STORE / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved %s → %s  shape=%s", name, path, obj.shape)

    logger.info("Feature store complete — 3 files written to %s", FEATURE_STORE)


def load_feature_store() -> tuple:
    """
    Load F_E, F_B, and targets from data/feature_store/.

    Returns
    -------
    tuple  (X_engineered, X_baseline, targets_df)
    """
    paths = [FEATURE_STORE / f"{n}.pkl" for n in ("F_E", "F_B", "targets")]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run save_feature_store() first.")
    result = []
    for p in paths:
        with open(p, "rb") as f:
            result.append(pickle.load(f))
    logger.info("Feature store loaded from %s", FEATURE_STORE)
    return tuple(result)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified split WITHOUT scaling.

    Use for tree models (invariant to monotonic transforms) and Naive Bayes
    (likelihood ratio invariant to linear rescaling).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Stratified split + MinMaxScaler [0,1] fitted on training set only.

    Use for SVM and MLP. Test values may exceed [0,1] — expected and correct.

    WARNING: Do NOT pass the scaled output to compute_cv_metrics — the scaler
    has already seen all training observations, causing CV leakage. Use a
    Pipeline instead (see soft_voting_ens.py for the correct pattern).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_s, X_te_s, y_train, y_test, scaler


def split_and_standardize(X, y, test_size=0.2, random_state=42):
    """
    Stratified split + StandardScaler fitted on training set only.

    Use for Logistic Regression. MinMaxScaler is incorrect for L1/L2
    regularization: it normalizes range but not variance, producing
    incomparable effective regularization at the same λ. Section 3.3.

    WARNING: same CV leakage caveat as split_and_scale. Use Pipeline for CV.
    """
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


def compute_cv_metrics(model, X_train, y_train, k_folds=N_OUTER_FOLDS):
    """
    Stratified K-Fold CV returning raw per-fold score lists.

    The model is CLONED on every fold to prevent state accumulation across
    folds. This is critical for ensemble models, calibrated wrappers, and
    any estimator where fit() does not fully reinitialize internal state.

    If a Pipeline is passed, the scaler inside it is refitted on each fold's
    training data automatically — the correct leakage-free pattern.

    Parameters
    ----------
    model : fresh, unfitted sklearn estimator or Pipeline
    X_train : unscaled DataFrame if using a Pipeline; pre-scaled otherwise
    y_train : binary Series
    k_folds : int, default N_OUTER_FOLDS (10)

    Returns
    -------
    dict  {metric: [fold_score, ...]}  keys: accuracy, precision, recall, f1
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
        logger.debug("Fold %d/%d — F1: %.4f", fold_idx + 1, k_folds, metrics["f1"][-1])

    return metrics


def summarise_cv(cv_metrics: dict) -> dict:
    """Summarise raw fold scores into {metric: {mean, std}}."""
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in cv_metrics.items()}


def compute_test_metrics(model, X_test, y_test) -> dict:
    """Evaluate fitted model on the held-out test set at threshold=0.5."""
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
    Select the threshold maximising F1 subject to Precision >= precision_floor.

    Fixing threshold at 0.5 ignores the asymmetric mis-selling cost from
    MiFID II. This implements the correct business-driven selection.

    Raises
    ------
    ValueError  if no threshold achieves the precision floor.
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
            f"Max achievable = {precisions.max():.3f}."
        )

    f1_scores = 2 * precisions[valid] * recalls[valid] / np.clip(precisions[valid] + recalls[valid], 1e-9, None)
    best      = np.argmax(f1_scores)

    result = {
        "threshold": float(thresholds[valid][best]),
        "precision": float(precisions[valid][best]),
        "recall":    float(recalls[valid][best]),
        "f1":        float(f1_scores[best]),
        "precisions":  precisions,
        "recalls":     recalls,
        "thresholds":  thresholds,
    }
    logger.info(
        "PR threshold: %.3f → P=%.3f R=%.3f F1=%.3f",
        result["threshold"], result["precision"], result["recall"], result["f1"],
    )
    return result


def apply_threshold(model, X_test, threshold: float) -> np.ndarray:
    """Apply a custom threshold to predict_proba scores."""
    return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(model, X_train, y_train):
    """
    Post-hoc calibration via isotonic regression with cv=5.

    Uses cv=5, NOT cv='prefit'. cv='prefit' fits the calibrator on the same
    data used to train the base model, inflating calibration quality (leakage).
    With cv=5 the base estimator is refitted internally across 5 folds.

    Isotonic regression is preferred over Platt scaling: it is nonparametric
    and does not assume a sigmoidal calibration error form

    Parameters
    ----------
    model : UNFITTED base estimator (cv=5 refits it internally)
    X_train, y_train : full training set
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_brier_score(model, X_test, y_test) -> float:
    """
    Brier score: BS = (1/n) sum((p_hat - y)^2). Lower is better.
    0.25 = no-skill baseline for balanced binary target.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Brier score requires predict_proba.")
    return float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))


# ---------------------------------------------------------------------------
# Label sensitivity (Section 3.1)
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
    Compute scale_pos_weight = n_neg / n_pos for XGBoost.

    Apply to IncomeInvestment only (38% positive). Caller is responsible
    for not applying this to the balanced AccumulationInvestment target.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)
    logger.debug("scale_pos_weight=%.3f (neg=%d, pos=%d)", spw, n_neg, n_pos)
    return spw


# ---------------------------------------------------------------------------
# Result factory
# ---------------------------------------------------------------------------


def make_result_dict(
    model, scaler, cv_metrics_raw, test_metrics,
    y_test_true, y_test_pred, feature_names,
    target_name, model_name, ablation=None, **extra,
) -> dict:
    """
    Assemble the canonical result dict pickled by every model script.

    Keys shared across all models: model, scaler, cv_metrics_raw,
    cv_metrics_summary, test_metrics, y_test_true, y_test_pred,
    feature_names, target_name, model_name, ablation.
    Additional keys passed via **extra (shap_values, brier_score, etc.).
    """
    result = {
        "model":              model,
        "scaler":             scaler,
        "cv_metrics_raw":     cv_metrics_raw,
        "cv_metrics_summary": summarise_cv(cv_metrics_raw),
        "test_metrics":       test_metrics,
        "y_test_true":        y_test_true,
        "y_test_pred":        y_test_pred,
        "feature_names":      feature_names,
        "target_name":        target_name,
        "model_name":         model_name,
        "ablation":           ablation,
    }
    result.update(extra)
    return result