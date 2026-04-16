"""
Multi-Layer Perceptron (PyTorch) classifier for IncomeInvestment and
AccumulationInvestment.

Theory
------
A feedforward neural network applies successive learned affine transformations
interleaved with non-linear activation functions to model complex and
potentially non-linear decision boundaries. Each hidden layer follows the
pattern: Linear → BatchNorm1d → ReLU → Dropout(0.2).

Batch normalisation re-centres and re-scales activations to zero mean and unit
variance within each mini-batch, stabilising gradient flow and enabling the use
of higher learning rates. Dropout randomly zeroes 20% of activations during
training, acting as an implicit ensemble regulariser that reduces co-adaptation
between neurons and mitigates overfitting.

Training minimises binary cross-entropy with logits (BCEWithLogitsLoss), which
numerically combines the sigmoid activation and binary cross-entropy into a
single, more stable computation.

Class imbalance in the IncomeInvestment target (approximately 38% positive
instances) is addressed using the pos_weight argument, which up-weights the
positive-class gradient contributions by the ratio n_neg / n_pos. This weight
is recomputed from each fold's training labels during cross-validation to
prevent information leakage from the validation fold.

The Adam optimiser is employed for efficient stochastic gradient-based
optimisation, together with a ReduceLROnPlateau scheduler that halves the
learning rate when the training loss plateaus, enabling finer convergence.
All features are scaled to the [0, 1] interval using a MinMaxScaler fitted
independently on each fold's training data to avoid data leakage.

Implementation
--------------
This script evaluates two neural network architectures with hidden layer sizes
16-8-1 and 32-16-8-1 using stratified 10-fold cross-validation on the engineered
feature set F_E. The architecture achieving the highest mean cross-validated
F1 score is selected.

A final model with the selected architecture is then trained on the full F_E
training partition. An F_B ablation study evaluates the same architecture on
the baseline feature set, enabling a controlled comparison of the impact of
feature engineering.

Results are serialised to:

    data/pickled_files/mlp/

The saved outputs include both the live model instance, enabling probability
predictions through a scikit-learn–compatible wrapper, and the PyTorch
state_dict for portability and reproducibility.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_OUTER_FOLDS,
    PRECISION_FLOOR,
    TARGETS,
    build_baseline_features,
    build_features,
    load_data,
    make_result_dict,
    no_skill_brier,
    save_result,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "MLP"
FOLDER: str = "mlp"

ARCHITECTURES = {
    "16-8-1":    [16, 8],
    "32-16-8-1": [32, 16, 8],
}


class InvestmentDataset(Dataset):
    """
    PyTorch Dataset wrapping a feature DataFrame and a binary target Series.

    Converts both to ``FloatTensor`` at construction time for efficient batched
    retrieval by a ``DataLoader``.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Store features and labels as FloatTensors.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        y : pd.Series
            Binary target labels of shape (n_samples,).
        """
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """Return the (features, label) FloatTensor pair at position idx."""
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """
    Fully connected neural network with configurable hidden layer sizes.

    Each hidden layer applies: Linear → BatchNorm1d → ReLU → Dropout(0.2).
    The output layer is a single linear neuron producing a raw logit (no
    sigmoid), compatible with ``BCEWithLogitsLoss`` during training and
    ``torch.sigmoid`` at inference.
    """

    def __init__(self, input_size: int, hidden_sizes: list) -> None:
        """
        Build the sequential layer stack.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_sizes : list of int
            Number of neurons in each hidden layer, from first to last.
        """
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)

    def predict_proba(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """
        sklearn-compatible ``predict_proba``, returning shape (n_samples, 2).

        Parameters
        ----------
        X_scaled : pd.DataFrame
            Already-scaled feature matrix.

        Returns
        -------
        np.ndarray
            Columns [P(negative), P(positive)].
        """
        self.eval()
        with torch.no_grad():
            logits = self(torch.FloatTensor(X_scaled.values))
            p1 = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """
        Return hard binary predictions by thresholding ``predict_proba`` at 0.5.

        Parameters
        ----------
        X_scaled : pd.DataFrame
            Already-scaled feature matrix.

        Returns
        -------
        np.ndarray
            Integer array of 0 / 1 predictions.
        """
        return (self.predict_proba(X_scaled)[:, 1] >= 0.5).astype(int)


def _train(model: MLP, loader: DataLoader, pos_weight: float = None, epochs: int = 100) -> None:
    """
    Train the MLP in-place for the specified number of epochs.

    Uses ``BCEWithLogitsLoss``, Adam optimiser (lr=0.001), and a
    ``ReduceLROnPlateau`` scheduler that halves the learning rate when the
    per-epoch training loss stagnates for 10 consecutive epochs.

    Parameters
    ----------
    model : MLP
        Unfitted (or partially fitted) network to train in-place.
    loader : DataLoader
        Mini-batch loader for the training set.
    pos_weight : float or None
        Positive-class loss weight — n_neg / n_pos for IncomeInvestment, None
        for AccumulationInvestment.
    epochs : int
        Number of full passes over the training data.
    """
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]) if pos_weight is not None else None
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(loader))


def _evaluate(model: MLP, loader: DataLoader) -> dict:
    """
    Evaluate the MLP on a DataLoader in inference mode.

    Parameters
    ----------
    model : MLP
        Fitted network.
    loader : DataLoader
        Mini-batch loader for the evaluation set.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1 (floats), proba (np.ndarray of
        positive-class probabilities), true (np.ndarray of ground-truth labels).
    """
    model.eval()
    preds, probs, true = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            scores = torch.sigmoid(model(X_batch))
            probs.extend(scores.numpy().tolist())
            preds.extend((scores > 0.5).float().numpy().tolist())
            true.extend(y_batch.numpy().tolist())
    return {
        "accuracy":  float(accuracy_score(true, preds)),
        "precision": float(precision_score(true, preds, zero_division=0)),
        "recall":    float(recall_score(true, preds, zero_division=0)),
        "f1":        float(f1_score(true, preds, zero_division=0)),
        "proba":     np.array(probs),
        "true":      np.array(true),
    }


def _cv_mlp_arch(
    X_raw: pd.DataFrame,
    y: pd.Series,
    hidden_sizes: list,
    target_col: str,
    k_folds: int = N_OUTER_FOLDS,
    epochs: int = 100,
) -> dict:
    """
    Run stratified K-fold CV for a single MLP architecture.

    MinMaxScaler is fitted exclusively on each fold's training split to prevent
    leakage.  For IncomeInvestment, ``pos_weight`` is recomputed from that
    fold's training labels to prevent leakage from the validation fold.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Unscaled feature matrix.
    y : pd.Series
        Binary target labels.
    hidden_sizes : list of int
        Hidden layer widths defining the architecture.
    target_col : str
        Target column name used to decide whether to apply ``pos_weight``.
    k_folds : int
        Number of CV folds.
    epochs : int
        Training epochs per fold.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1 — each a list of k_folds floats.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for tr_idx, val_idx in kf.split(X_raw, y):
        X_tr_raw  = X_raw.iloc[tr_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_tr      = y.iloc[tr_idx]
        y_val     = y.iloc[val_idx]

        fold_scaler = MinMaxScaler()
        X_tr  = pd.DataFrame(
            fold_scaler.fit_transform(X_tr_raw),
            columns=X_tr_raw.columns, index=X_tr_raw.index,
        )
        X_val = pd.DataFrame(
            fold_scaler.transform(X_val_raw),
            columns=X_val_raw.columns, index=X_val_raw.index,
        )

        # pos_weight from THIS fold's labels
        fold_pw = None
        if target_col == "IncomeInvestment":
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            fold_pw = n_neg / max(n_pos, 1)

        tr_loader  = DataLoader(InvestmentDataset(X_tr,  y_tr),  batch_size=32, shuffle=True)
        val_loader = DataLoader(InvestmentDataset(X_val, y_val), batch_size=32)

        net = MLP(input_size=X_tr.shape[1], hidden_sizes=hidden_sizes)
        _train(net, tr_loader, pos_weight=fold_pw, epochs=epochs)
        fold_m = _evaluate(net, val_loader)
        for k in ["accuracy", "precision", "recall", "f1"]:
            metrics[k].append(fold_m[k])

    return metrics


def run_for_target(df: object, target_col: str) -> dict:
    """
    Run the full MLP architecture search, final model training, threshold
    selection, and serialisation workflow for a single target column.

    Steps:
    1. Evaluate all architectures in ARCHITECTURES via 10-fold CV on F_E;
       select the one with the highest mean CV F1.
    2. Train the final model on the full F_E training partition.
    3. Select the MiFID II compliance threshold on the validation split.
    4. Train an F_B ablation with the same selected architecture.
    5. Assemble and save the result dict, including both the live model instance
       and the state_dict.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataset from :func:`load_data`.
    target_col : str
        One of the TARGETS constants.

    Returns
    -------
    dict
        Standardised result dict (see :func:`make_result_dict`).
    """
    logger.info("Target: %s", target_col)
    y = df[target_col]
    baseline = no_skill_brier(y)

    global_pw = None
    if target_col == "IncomeInvestment":
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        global_pw = n_neg / max(n_pos, 1)
        logger.info("  Global pos_weight=%.3f (final model only; per-fold in CV)", global_pw)

    X_eng = build_features(df)
    X_tr_full, X_te_raw, y_tr_full, y_te = split_data(X_eng, y)
    X_tr_raw, X_val_raw, y_tr_split, y_val_split = split_data(X_tr_full, y_tr_full, test_size=0.2)

    # architecture search result IS the outer CV
    logger.info("  Architecture search (%d-fold CV per architecture)...", N_OUTER_FOLDS)
    best_arch_name = None
    best_f1 = -1
    best_cv_raw = None

    for arch_name, hidden_sizes in ARCHITECTURES.items():
        cv_m = _cv_mlp_arch(X_tr_full, y_tr_full, hidden_sizes, target_col, epochs=80)
        mean_f1 = np.mean(cv_m["f1"])
        logger.info("    %s: CV F1=%.3f ± %.3f", arch_name, mean_f1, np.std(cv_m["f1"], ddof=1))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_arch_name = arch_name
            best_cv_raw = cv_m

    best_hidden = ARCHITECTURES[best_arch_name]
    logger.info("  Best: %s  (CV F1=%.3f)", best_arch_name, best_f1)

    # Final model
    final_scaler = MinMaxScaler()
    X_tr_e  = pd.DataFrame(
        final_scaler.fit_transform(X_tr_full),
        columns=X_tr_full.columns, index=X_tr_full.index,
    )
    X_te_e  = pd.DataFrame(
        final_scaler.transform(X_te_raw),
        columns=X_te_raw.columns, index=X_te_raw.index,
    )
    X_val_e = pd.DataFrame(
        final_scaler.transform(X_val_raw),
        columns=X_val_raw.columns, index=X_val_raw.index,
    )

    net_e = MLP(input_size=X_tr_e.shape[1], hidden_sizes=best_hidden)
    _train(
        net_e,
        DataLoader(InvestmentDataset(X_tr_e, y_tr_full), batch_size=32, shuffle=True),
        pos_weight=global_pw, epochs=100,
    )
    test_eval  = _evaluate(net_e, DataLoader(InvestmentDataset(X_te_e, y_te), batch_size=32))
    test_m_e   = {k: test_eval[k] for k in ["accuracy", "precision", "recall", "f1"]}
    y_proba_e  = test_eval["proba"]
    brier      = float(brier_score_loss(y_te.values, y_proba_e))
    y_pred_e   = (y_proba_e >= 0.5).astype(int)
    logger.info(
        "  [F_E] Test F1=%.3f  Precision=%.3f  Brier=%.4f (baseline=%.4f)",
        test_m_e["f1"], test_m_e["precision"], brier, baseline,
    )

    # threshold from validation set
    thr_test_m = None
    try:
        val_eval  = _evaluate(net_e, DataLoader(InvestmentDataset(X_val_e, y_val_split), batch_size=32))
        val_proba = val_eval["proba"]
        val_true  = val_eval["true"]
        precisions, recalls, thresholds = precision_recall_curve(val_true, val_proba)
        precisions, recalls = precisions[:-1], recalls[:-1]
        valid = precisions >= PRECISION_FLOOR
        if valid.any():
            f1s  = 2 * precisions[valid] * recalls[valid] / np.clip(precisions[valid] + recalls[valid], 1e-9, None)
            thr  = float(thresholds[valid][np.argmax(f1s)])
            y_thr = (y_proba_e >= thr).astype(int)
            thr_test_m = {
                "threshold": thr,
                "precision": float(precision_score(y_te.values, y_thr, zero_division=0)),
                "recall":    float(recall_score(y_te.values, y_thr, zero_division=0)),
                "f1":        float(f1_score(y_te.values, y_thr, zero_division=0)),
            }
            logger.info(
                "  Val thr=%.3f → Test P=%.3f R=%.3f F1=%.3f",
                thr, thr_test_m["precision"], thr_test_m["recall"], thr_test_m["f1"],
            )
    except Exception as exc:
        logger.warning("  Threshold selection failed: %s", exc)

    # F_B ablation
    X_base = build_baseline_features(df)
    X_tr_b_full, X_te_b_raw, y_tr_b_full, y_te_b = split_data(X_base, y)
    cv_raw_b = _cv_mlp_arch(X_tr_b_full, y_tr_b_full, best_hidden, target_col, epochs=80)
    final_scaler_b = MinMaxScaler()
    X_tr_b = pd.DataFrame(
        final_scaler_b.fit_transform(X_tr_b_full),
        columns=X_tr_b_full.columns, index=X_tr_b_full.index,
    )
    X_te_b = pd.DataFrame(
        final_scaler_b.transform(X_te_b_raw),
        columns=X_te_b_raw.columns, index=X_te_b_raw.index,
    )
    net_b = MLP(input_size=X_tr_b.shape[1], hidden_sizes=best_hidden)
    _train(
        net_b,
        DataLoader(InvestmentDataset(X_tr_b, y_tr_b_full), batch_size=32, shuffle=True),
        pos_weight=global_pw, epochs=100,
    )
    test_b   = _evaluate(net_b, DataLoader(InvestmentDataset(X_te_b, y_te_b), batch_size=32))
    test_m_b = {k: test_b[k] for k in ["accuracy", "precision", "recall", "f1"]}
    logger.info(
        "  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)",
        test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"],
    )

    # store BOTH live model instance (for predict_proba) AND state_dict
    result = make_result_dict(
        model=net_e,                      # live instance — predict_proba works
        scaler=final_scaler,
        cv_metrics_raw=best_cv_raw, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=y_pred_e,
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME,
        best_params={
            "architecture":      best_arch_name,
            "hidden_sizes":      best_hidden,
            "pos_weight_global": global_pw,
            "pos_weight_note":   "per-fold pos_weight recomputed inside CV folds",
        },
        threshold_metrics=thr_test_m,
        ablation={
            "engineered": {
                "cv_metrics_raw":     best_cv_raw,
                "cv_metrics_summary": summarise_cv(best_cv_raw),
                "test_metrics":       test_m_e,
            },
            "baseline": {
                "cv_metrics_raw":     cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics":       test_m_b,
            },
        },
        model_state_dict=net_e.state_dict(),  # also stored for portability
        model_architecture=f"input→{'→'.join(str(h) for h in best_hidden)}→1",
        brier_score=brier,
        no_skill_brier=baseline,
    )
    path = save_result(result, FOLDER, target_col)
    logger.info("  Saved: %s", path)
    return result


def main() -> None:
    """Entry point: load the raw dataset and train on every target in TARGETS."""
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()
