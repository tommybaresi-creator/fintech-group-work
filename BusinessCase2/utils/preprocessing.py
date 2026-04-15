"""
Shared preprocessing utilities for BusinessCase2.

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
    "Age", "Age_sq", "Age_x_Wealthlog",           # renamed D1 fix
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
    No-skill Brier baseline = p * (1-p) where p = prevalence.
    M1 fix: not hardcoded 0.25. For IncomeInvestment (38% pos): 0.38*0.62 = 0.2356.
    """
    p = float(y.mean())
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
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
    Construct F_E. Note: Age_x_Wealthlog = Age * log1p(Wealth).
    D1 fix: renamed from Age_x_Wealth to Age_x_Wealthlog to match the formula.
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
    """Construct F_B — professor's 7-feature baseline."""
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
    if df is None:
        df = load_data()
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    X_e = build_features(df)
    X_b = build_baseline_features(df)
    y   = df[TARGETS]
    for name, obj in [("F_E", X_e), ("F_B", X_b), ("targets", y)]:
        with open(FEATURE_STORE / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)


def load_feature_store() -> tuple:
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
    """Stratified split WITHOUT scaling."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Stratified split + MinMaxScaler on train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_s, X_te_s, y_train, y_test, scaler


def split_and_standardize(X, y, test_size=0.2, random_state=42):
    """Stratified split + StandardScaler on train only."""
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
    Pass UNSCALED data when using a Pipeline — the Pipeline refits scaler per fold.
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
    N5 fix: ddof=1 (sample std of k observations, not population std).
    For 10 folds this changes std by factor sqrt(10/9) ≈ 1.054 — small but correct.
    """
    return {
        m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
        for m, v in cv_metrics.items()
    }


def compute_test_metrics(model, X_test, y_test) -> dict:
    """Evaluate at threshold=0.5."""
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


def tune_hyperparameters(model, param_grid: dict, X_train, y_train,
                          n_iter: int = 20, cv: int = N_INNER_FOLDS,
                          scoring: str = "f1") -> tuple:
    """RandomizedSearchCV inner CV loop."""
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


def nested_cv_with_tuning(model_factory, param_grid: dict, X_train, y_train,
                           n_iter: int = 20, k_folds: int = N_OUTER_FOLDS,
                           inner_cv: int = N_INNER_FOLDS) -> dict:
    """
    C4 fix: TRUE nested CV.
    HP tuning runs inside each outer fold independently on that fold's training data.
    The CV F1 scores used in Wilcoxon tests are unbiased — no outer fold data
    has been seen during HP selection.

    Parameters
    ----------
    model_factory : callable
        Returns a fresh unfitted estimator or Pipeline. Called once per outer fold.
    param_grid : dict
        Hyperparameter search space.
    X_train, y_train : training data (full, unscaled if using Pipeline)
    n_iter : random search iterations per outer fold
    k_folds : outer CV folds (default 10)
    inner_cv : inner CV folds for HP tuning (default 3)

    Returns
    -------
    dict with keys:
        cv_metrics_raw : {metric: [fold_score]}
        best_params_per_fold : [dict]  — HP selected in each outer fold
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

        logger.debug("Fold %d/%d — F1=%.4f  best_params=%s",
                     fold_idx + 1, k_folds, metrics["f1"][-1], search.best_params_)

    return {"cv_metrics_raw": metrics, "best_params_per_fold": best_params_per_fold}


# ---------------------------------------------------------------------------
# MiFID II threshold selection — C1 fix
# ---------------------------------------------------------------------------


def select_threshold_on_val(model, X_val, y_val,
                             precision_floor: float = PRECISION_FLOOR) -> dict:
    """
    C1 fix: threshold is selected on a VALIDATION set (X_val, y_val),
    NOT on the final test set. The returned threshold is then applied to
    the held-out test set for unbiased reporting.

    Workflow in each model script:
        X_tr, X_te, y_tr, y_te = split_data(X, y)          # 80/20
        X_tr2, X_val, y_tr2, y_val = split_data(X_tr, y_tr, test_size=0.2)
        model.fit(X_tr2, y_tr2)
        thr_info = select_threshold_on_val(model, X_val, y_val)
        # report metrics on X_te at thr_info['threshold'] — unbiased
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
        "threshold": float(thresholds[valid][best]),
        "val_precision": float(precisions[valid][best]),
        "val_recall":    float(recalls[valid][best]),
        "val_f1":        float(f1s[best]),
    }


def apply_threshold(model, X_test, threshold: float) -> np.ndarray:
    """Apply pre-fixed threshold to test set. Returns binary predictions."""
    return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)


def evaluate_at_threshold(model, X_test, y_test, threshold: float) -> dict:
    """Unbiased test metrics at a pre-fixed threshold."""
    y_pred = apply_threshold(model, X_test, threshold)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }


def get_propensity_scores(model, X) -> np.ndarray:
    """Calibrated propensity scores for recommendation engine."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    return model.predict_proba(X)[:, 1]


def segment_by_confidence(propensity: np.ndarray) -> dict:
    """Segment clients: high / uncertain / low propensity."""
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


def calibrate_model(model, X_train, y_train):
    """
    Post-hoc isotonic calibration with cv=5.
    cv='prefit' is deliberately avoided — it calibrates on the same data
    used to train the base model, inflating quality estimates.
    Pass UNFITTED model — cv=5 refits it internally.
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_brier_score(model, X_test, y_test) -> float:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Brier score requires predict_proba.")
    return float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))


# ---------------------------------------------------------------------------
# Label sensitivity
# ---------------------------------------------------------------------------


def flip_labels(y: pd.Series, flip_rate: float, random_state: int = 42) -> pd.Series:
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
    """scale_pos_weight = n_neg/n_pos. Apply to IncomeInvestment only."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


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
    threshold_metrics=None,
    **extra,
) -> dict:
    """
    threshold_metrics: dict from evaluate_at_threshold() — unbiased test metrics
                       at the pre-fixed validation-selected threshold (C1 fix).
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
    out_dir = PICKLE_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_name.lower()}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path


def load_result(folder: str, target_name: str) -> dict:
    path = PICKLE_ROOT / folder / f"{target_name.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No pickle at {path}.")
    with open(path, "rb") as f:
        return pickle.load(f)