"""
Multi-Layer Perceptron (PyTorch) classifier for both investment targets.

Architecture: input → 16 → 8 → 1
Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout(0.2)
Output: Sigmoid (binary classification)

The 64→32→16→1 architecture from preliminary experiments contains ~3,000
parameters for 10 input features and ~4,000 training observations — it is
overparameterized for this dataset and consistently outperformed by tree-based
methods (Section 3.7 of the paper). The compact 16→8→1 architecture (≈200
parameters) is statistically appropriate.

Training: BCELoss | Adam(lr=0.001) | ReduceLROnPlateau(patience=10) | 100 epochs
Scaling: MinMaxScaler [0,1] — matches the Sigmoid output activation (Section 3.3).
The model's ``state_dict`` (not the Module) is stored in the pickle so
``bestmodel_*.ipynb`` can load metrics without importing PyTorch.

Saves artifacts to ``data/pickled_files/mlp/``.

Run directly::

    python -m utils.mlp
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGETS,
    build_baseline_features,
    build_features,
    compute_brier_score,
    load_data,
    make_result_dict,
    split_and_scale,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "mlp"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME: str = "MLP"
_ARCHITECTURE: str = "input→16(BN+ReLU+Drop0.2)→8(BN+ReLU+Drop0.2)→1(Sigmoid)"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class InvestmentDataset(Dataset):
    """
    PyTorch Dataset wrapping a feature DataFrame and a target Series.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (already scaled).
    y : pd.Series
        Binary target series.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Compact feedforward network for binary classification.

    Architecture: input → 16 → 8 → 1
    Deliberately compact: ~200 parameters for 10 input features and
    ~4,000 training observations. The previously used 64→32→16→1 network
    (~3,000 parameters) is overparameterized for this dataset (Section 3.7).

    Parameters
    ----------
    input_size : int
        Number of input features.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_size)``.

        Returns
        -------
        torch.Tensor
            Scalar output in (0, 1) per sample, shape ``(batch_size,)``.
        """
        return self.layers(x).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train(
    model: MLP,
    train_loader: DataLoader,
    epochs: int = 100,
) -> None:
    """
    Train the MLP in-place with BCELoss, Adam, and ReduceLROnPlateau.

    ReduceLROnPlateau halves the learning rate when the training loss stops
    improving for ``patience=10`` epochs, preventing oscillation near minima.

    Parameters
    ----------
    model : MLP
        Uninitialised (freshly instantiated) MLP.
    train_loader : DataLoader
        Training data loader.
    epochs : int, optional
        Maximum number of training epochs, by default 100.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            logger.debug(
                "  Epoch %d/%d — loss: %.4f  lr: %.6f",
                epoch + 1,
                epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )


def _evaluate(model: MLP, loader: DataLoader) -> dict[str, float]:
    """
    Evaluate a trained MLP on a DataLoader.

    Parameters
    ----------
    model : MLP
        Fitted MLP.
    loader : DataLoader
        Data loader.

    Returns
    -------
    dict[str, float]
        Keys: ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``.
    """
    model.eval()
    all_preds: list[float] = []
    all_true: list[float] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = (model(X_batch) > 0.5).float()
            all_preds.extend(preds.numpy().tolist())
            all_true.extend(y_batch.numpy().tolist())

    return {
        "accuracy":  float(accuracy_score(all_true, all_preds)),
        "precision": float(precision_score(all_true, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_true, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_true, all_preds, zero_division=0)),
    }


def _cv_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_folds: int = 10,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, list[float]]:
    """
    Stratified K-Fold cross-validation for the MLP.

    Each fold trains a fresh network from scratch for unbiased variance
    estimation across 10 folds.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (scaled).
    y_train : pd.Series
        Training labels.
    k_folds : int, optional
        Number of folds, by default 10.
    epochs : int, optional
        Training epochs per fold, by default 100.
    batch_size : int, optional
        Mini-batch size, by default 32.

    Returns
    -------
    dict[str, list[float]]
        Raw per-fold scores.
    """
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [], "f1": []
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        tr_loader = DataLoader(
            InvestmentDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            InvestmentDataset(X_val, y_val), batch_size=batch_size
        )

        net = MLP(input_size=X_tr.shape[1])
        _train(net, tr_loader, epochs=epochs)
        fold_m = _evaluate(net, val_loader)

        for key in metrics:
            metrics[key].append(fold_m[key])
        logger.debug(
            "  Fold %d/%d — F1: %.4f", fold_idx + 1, k_folds, fold_m["f1"]
        )

    return metrics


# ---------------------------------------------------------------------------
# Per-target runner
# ---------------------------------------------------------------------------


def run_for_target(df, target_col: str) -> dict:
    """
    Train and evaluate the MLP for one binary target.

    Architecture: input→16→8→1 with BatchNorm+ReLU+Dropout(0.2) per layer.
    10-fold stratified CV. Final model trained on full training set.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by :func:`~utils.preprocessing.load_data`.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    dict
        Canonical result dictionary. ``'model'`` holds a ``state_dict``
        dict (not the Module) for pickle portability.
    """
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # ------------------------------------------------------------------ F_E
    X_eng = build_features(df)
    X_tr_e, X_te_e, y_tr, y_te, scaler_e = split_and_scale(X_eng, y)

    logger.info("  Running 10-fold CV on F_E...")
    cv_raw_e = _cv_mlp(X_tr_e, y_tr)
    logger.info(
        "  [F_E] CV F1: %.3f ± %.3f",
        np.mean(cv_raw_e["f1"]),
        np.std(cv_raw_e["f1"]),
    )

    tr_loader = DataLoader(
        InvestmentDataset(X_tr_e, y_tr), batch_size=32, shuffle=True
    )
    te_loader = DataLoader(InvestmentDataset(X_te_e, y_te), batch_size=32)

    net_e = MLP(input_size=X_tr_e.shape[1])
    _train(net_e, tr_loader)
    test_m_e = _evaluate(net_e, te_loader)
    logger.info("  [F_E] Test F1: %.3f", test_m_e["f1"])

    y_test_pred: list[float] = []
    net_e.eval()
    with torch.no_grad():
        for X_batch, _ in te_loader:
            y_test_pred.extend((net_e(X_batch) > 0.5).float().numpy().tolist())

    # ------------------------------------------------------------------ F_B (ablation)
    X_base = build_baseline_features(df)
    X_tr_b, X_te_b, _, _, _ = split_and_scale(X_base, y)

    logger.info("  Running 10-fold CV on F_B (ablation)...")
    cv_raw_b = _cv_mlp(X_tr_b, y_tr)

    tr_loader_b = DataLoader(
        InvestmentDataset(X_tr_b, y_tr), batch_size=32, shuffle=True
    )
    te_loader_b = DataLoader(InvestmentDataset(X_te_b, y_te), batch_size=32)
    net_b = MLP(input_size=X_tr_b.shape[1])
    _train(net_b, tr_loader_b)
    test_m_b = _evaluate(net_b, te_loader_b)
    logger.info(
        "  [F_B] Test F1: %.3f  (delta F_E−F_B = %.3f)",
        test_m_b["f1"],
        test_m_e["f1"] - test_m_b["f1"],
    )

    return make_result_dict(
        model=net_e.state_dict(),
        scaler=scaler_e,
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_m_e,
        y_test_true=y_te.values,
        y_test_pred=np.array(y_test_pred),
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "engineered": {
                "cv_metrics_raw": cv_raw_e,
                "cv_metrics_summary": summarise_cv(cv_raw_e),
                "test_metrics": test_m_e,
            },
            "baseline": {
                "cv_metrics_raw": cv_raw_b,
                "cv_metrics_summary": summarise_cv(cv_raw_b),
                "test_metrics": test_m_b,
            },
        },
        model_architecture=_ARCHITECTURE,
    )


def main() -> None:
    """Train, evaluate, and pickle the MLP for all targets."""
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
