"""
Multi-Layer Perceptron (PyTorch) for both investment targets.

Architecture: input → 16 → 8 → 1
Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout(0.2)
Output: Sigmoid

CV leakage fix: MinMaxScaler is refitted inside each fold of _cv_mlp,
not once on the full training set before CV.

Scaling: MinMaxScaler [0,1] — matches Sigmoid output activation.
The model's state_dict is stored in the pickle for portability.

Saves artifacts to data/pickled_files/mlp/.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_OUTER_FOLDS,
    TARGETS,
    build_baseline_features,
    build_features,
    load_data,
    make_result_dict,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PICKLE_DIR: Path = Path(__file__).parent.parent / "data" / "pickled_files" / "mlp"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME:    str = "MLP"
_ARCHITECTURE: str = "input→16(BN+ReLU+Drop0.2)→8(BN+ReLU+Drop0.2)→1(Sigmoid)"


class InvestmentDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """
    Compact 16→8→1 network. ~200 parameters for 10 input features.
    The 64→32→16→1 used in preliminary experiments (~3,000 parameters)
    is overparameterized for ~4,000 training observations (Section 3.7).
    """
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8),          nn.BatchNorm1d(8),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(8, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


def _train(model: MLP, loader: DataLoader, epochs: int = 100) -> None:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Removed verbose=False to fix TypeError
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
        if (epoch + 1) % 20 == 0:
            logger.debug("  Epoch %d/%d — loss: %.4f", epoch + 1, epochs, epoch_loss / len(loader))


def _evaluate(model: MLP, loader: DataLoader) -> dict:
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.extend((model(X_batch) > 0.5).float().numpy().tolist())
            true.extend(y_batch.numpy().tolist())
    return {
        "accuracy":  float(accuracy_score(true, preds)),
        "precision": float(precision_score(true, preds, zero_division=0)),
        "recall":    float(recall_score(true, preds, zero_division=0)),
        "f1":        float(f1_score(true, preds, zero_division=0)),
    }


def _cv_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_folds: int = N_OUTER_FOLDS,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict:
    """
    Stratified K-Fold CV for the MLP.

    The MinMaxScaler is refitted on each fold's training data only —
    the correct leakage-free pattern. X_train must be UNSCALED.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        # Refit scaler on this fold's training data only
        fold_scaler = MinMaxScaler()
        X_tr  = pd.DataFrame(fold_scaler.fit_transform(X_tr_raw),  columns=X_tr_raw.columns,  index=X_tr_raw.index)
        X_val = pd.DataFrame(fold_scaler.transform(X_val_raw),     columns=X_val_raw.columns, index=X_val_raw.index)

        tr_loader  = DataLoader(InvestmentDataset(X_tr,  y_tr),  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(InvestmentDataset(X_val, y_val), batch_size=batch_size)

        net = MLP(input_size=X_tr.shape[1])
        _train(net, tr_loader, epochs=epochs)
        fold_m = _evaluate(net, val_loader)

        for key in metrics:
            metrics[key].append(fold_m[key])
        logger.debug("  Fold %d/%d — F1: %.4f", fold_idx + 1, k_folds, fold_m["f1"])

    return metrics


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # ---- F_E -----------------------------------------------------------------
    X_eng = build_features(df)
    # split_data returns unscaled — CV will scale inside each fold
    X_tr_e_raw, X_te_e_raw, y_tr, y_te = split_data(X_eng, y)

    logger.info("  Running %d-fold CV on F_E...", N_OUTER_FOLDS)
    cv_raw_e = _cv_mlp(X_tr_e_raw, y_tr)
    logger.info("  [F_E] CV  F1: %.3f ± %.3f", np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"]))

    # Final model: fit scaler on full train, evaluate on test
    final_scaler_e = MinMaxScaler()
    X_tr_e = pd.DataFrame(final_scaler_e.fit_transform(X_tr_e_raw), columns=X_tr_e_raw.columns, index=X_tr_e_raw.index)
    X_te_e = pd.DataFrame(final_scaler_e.transform(X_te_e_raw),     columns=X_te_e_raw.columns, index=X_te_e_raw.index)

    tr_loader = DataLoader(InvestmentDataset(X_tr_e, y_tr), batch_size=32, shuffle=True)
    te_loader = DataLoader(InvestmentDataset(X_te_e, y_te), batch_size=32)

    net_e = MLP(input_size=X_tr_e.shape[1])
    _train(net_e, tr_loader)
    test_m_e = _evaluate(net_e, te_loader)
    logger.info("  [F_E] Test F1: %.3f", test_m_e["f1"])

    y_test_pred = []
    net_e.eval()
    with torch.no_grad():
        for X_batch, _ in te_loader:
            y_test_pred.extend((net_e(X_batch) > 0.5).float().numpy().tolist())

    # ---- F_B (ablation) ------------------------------------------------------
    X_base = build_baseline_features(df)
    X_tr_b_raw, X_te_b_raw, y_tr_b, y_te_b = split_data(X_base, y)

    logger.info("  Running %d-fold CV on F_B (ablation)...", N_OUTER_FOLDS)
    cv_raw_b = _cv_mlp(X_tr_b_raw, y_tr_b)

    final_scaler_b = MinMaxScaler()
    X_tr_b = pd.DataFrame(final_scaler_b.fit_transform(X_tr_b_raw), columns=X_tr_b_raw.columns, index=X_tr_b_raw.index)
    X_te_b = pd.DataFrame(final_scaler_b.transform(X_te_b_raw),     columns=X_te_b_raw.columns, index=X_te_b_raw.index)

    tr_loader_b = DataLoader(InvestmentDataset(X_tr_b, y_tr_b), batch_size=32, shuffle=True)
    te_loader_b = DataLoader(InvestmentDataset(X_te_b, y_te_b), batch_size=32)
    net_b = MLP(input_size=X_tr_b.shape[1])
    _train(net_b, tr_loader_b)
    test_m_b = _evaluate(net_b, te_loader_b)
    logger.info("  [F_B] Test F1: %.3f  (ΔF_E−F_B = %+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    return make_result_dict(
        model=net_e.state_dict(),
        scaler=final_scaler_e,
        cv_metrics_raw=cv_raw_e,
        test_metrics=test_m_e,
        y_test_true=y_te.values,
        y_test_pred=np.array(y_test_pred),
        feature_names=FEATURE_NAMES,
        target_name=target_col,
        model_name=MODEL_NAME,
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        model_architecture=_ARCHITECTURE,
    )


def main() -> None:
    df = load_data()
    for target in TARGETS:
        result = run_for_target(df, target)
        out_path = PICKLE_DIR / f"{target.lower()}.joblib"
        joblib.dump(result, out_path, compress=3)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()