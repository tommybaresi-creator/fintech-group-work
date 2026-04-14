"""
Multi-Layer Perceptron (PyTorch) for both investment targets.

Fixes applied:
- BCEWithLogitsLoss + pos_weight for IncomeInvestment imbalance correction
- MinMaxScaler refits inside each CV fold (no leakage)
- y_test_proba (sigmoid scores) stored in result dict
- Architecture tuning: tests 16→8→1 and 32→16→8→1
- Output saved as .pkl
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
    save_result,
    split_data,
    summarise_cv,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME: str = "MLP"
FOLDER: str = "mlp"

# Architectures to test
ARCHITECTURES = {
    "16-8-1":    [16, 8],
    "32-16-8-1": [32, 16, 8],
}


class InvestmentDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Configurable MLP. hidden_sizes: list of hidden layer widths."""
    def __init__(self, input_size: int, hidden_sizes: list):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        # No Sigmoid here — using BCEWithLogitsLoss (numerically stable)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)

    def predict_proba_numpy(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """Return sigmoid scores as numpy array — used for propensity scores."""
        self.eval()
        with torch.no_grad():
            logits = self(torch.FloatTensor(X_scaled.values))
            return torch.sigmoid(logits).numpy()


def _train(model: MLP, loader: DataLoader, pos_weight=None, epochs: int = 100):
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
    model.eval()
    preds, probs, true = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            scores = torch.sigmoid(logits)
            probs.extend(scores.numpy().tolist())
            preds.extend((scores > 0.5).float().numpy().tolist())
            true.extend(y_batch.numpy().tolist())
    return {
        "accuracy":  float(accuracy_score(true, preds)),
        "precision": float(precision_score(true, preds, zero_division=0)),
        "recall":    float(recall_score(true, preds, zero_division=0)),
        "f1":        float(f1_score(true, preds, zero_division=0)),
        "proba":     np.array(probs),
    }


def _cv_mlp(X_train_raw: pd.DataFrame, y_train: pd.Series,
             hidden_sizes: list, pos_weight=None,
             k_folds: int = N_OUTER_FOLDS, epochs: int = 100) -> dict:
    """
    CV with scaler refitting inside each fold — no leakage.
    X_train_raw must be UNSCALED.
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train)):
        X_tr_raw  = X_train_raw.iloc[tr_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_tr      = y_train.iloc[tr_idx]
        y_val     = y_train.iloc[val_idx]

        # Refit scaler on this fold's training data
        fold_scaler = MinMaxScaler()
        X_tr  = pd.DataFrame(fold_scaler.fit_transform(X_tr_raw),  columns=X_tr_raw.columns,  index=X_tr_raw.index)
        X_val = pd.DataFrame(fold_scaler.transform(X_val_raw),     columns=X_val_raw.columns, index=X_val_raw.index)

        tr_loader  = DataLoader(InvestmentDataset(X_tr,  y_tr),  batch_size=32, shuffle=True)
        val_loader = DataLoader(InvestmentDataset(X_val, y_val), batch_size=32)

        net = MLP(input_size=X_tr.shape[1], hidden_sizes=hidden_sizes)
        _train(net, tr_loader, pos_weight=pos_weight, epochs=epochs)
        fold_m = _evaluate(net, val_loader)

        for key in ["accuracy", "precision", "recall", "f1"]:
            metrics[key].append(fold_m[key])
        logger.debug("  Fold %d/%d — F1: %.4f", fold_idx + 1, k_folds, fold_m["f1"])

    return metrics


def run_for_target(df, target_col: str) -> dict:
    logger.info("Target: %s", target_col)
    y = df[target_col]

    # Imbalance correction for Income target only
    pos_weight = None
    if target_col == "IncomeInvestment":
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_weight = n_neg / n_pos
        logger.info("  Applying pos_weight=%.3f for class imbalance", pos_weight)

    # Architecture search
    X_eng = build_features(df)
    X_tr_raw, X_te_raw, y_tr, y_te = split_data(X_eng, y)

    logger.info("  Architecture search...")
    best_arch_name = None
    best_arch_f1   = -1
    best_arch_cv   = None

    for arch_name, hidden_sizes in ARCHITECTURES.items():
        cv_m = _cv_mlp(X_tr_raw, y_tr, hidden_sizes=hidden_sizes, pos_weight=pos_weight, epochs=80)
        mean_f1 = np.mean(cv_m["f1"])
        logger.info("    Architecture %s: CV F1=%.3f ± %.3f", arch_name, mean_f1, np.std(cv_m["f1"]))
        if mean_f1 > best_arch_f1:
            best_arch_f1   = mean_f1
            best_arch_name = arch_name
            best_arch_cv   = cv_m

    best_hidden_sizes = ARCHITECTURES[best_arch_name]
    logger.info("  Best architecture: %s  (CV F1=%.3f)", best_arch_name, best_arch_f1)

    # Full CV on best architecture
    cv_raw_e = _cv_mlp(X_tr_raw, y_tr, hidden_sizes=best_hidden_sizes, pos_weight=pos_weight)
    logger.info("  [F_E] CV F1: %.3f ± %.3f", np.mean(cv_raw_e["f1"]), np.std(cv_raw_e["f1"]))

    # Final model: fit scaler on full training set
    final_scaler = MinMaxScaler()
    X_tr_e = pd.DataFrame(final_scaler.fit_transform(X_tr_raw), columns=X_tr_raw.columns, index=X_tr_raw.index)
    X_te_e = pd.DataFrame(final_scaler.transform(X_te_raw),     columns=X_te_raw.columns, index=X_te_raw.index)

    tr_loader = DataLoader(InvestmentDataset(X_tr_e, y_tr), batch_size=32, shuffle=True)
    te_loader = DataLoader(InvestmentDataset(X_te_e, y_te), batch_size=32)

    net_e = MLP(input_size=X_tr_e.shape[1], hidden_sizes=best_hidden_sizes)
    _train(net_e, tr_loader, pos_weight=pos_weight, epochs=100)
    test_eval_e = _evaluate(net_e, te_loader)
    test_m_e    = {k: test_eval_e[k] for k in ["accuracy", "precision", "recall", "f1"]}
    y_proba_e   = test_eval_e["proba"]
    logger.info("  [F_E] Test F1=%.3f  Precision=%.3f", test_m_e["f1"], test_m_e["precision"])

    y_test_pred = (y_proba_e > 0.5).astype(int)

    # F_B ablation
    X_base = build_baseline_features(df)
    X_tr_b_raw, X_te_b_raw, y_tr_b, y_te_b = split_data(X_base, y)
    cv_raw_b = _cv_mlp(X_tr_b_raw, y_tr_b, hidden_sizes=best_hidden_sizes, pos_weight=pos_weight)
    final_scaler_b = MinMaxScaler()
    X_tr_b = pd.DataFrame(final_scaler_b.fit_transform(X_tr_b_raw), columns=X_tr_b_raw.columns, index=X_tr_b_raw.index)
    X_te_b = pd.DataFrame(final_scaler_b.transform(X_te_b_raw),     columns=X_te_b_raw.columns, index=X_te_b_raw.index)
    tr_loader_b = DataLoader(InvestmentDataset(X_tr_b, y_tr_b), batch_size=32, shuffle=True)
    te_loader_b = DataLoader(InvestmentDataset(X_te_b, y_te_b), batch_size=32)
    net_b = MLP(input_size=X_tr_b.shape[1], hidden_sizes=best_hidden_sizes)
    _train(net_b, tr_loader_b, pos_weight=pos_weight, epochs=100)
    test_eval_b = _evaluate(net_b, te_loader_b)
    test_m_b    = {k: test_eval_b[k] for k in ["accuracy", "precision", "recall", "f1"]}
    logger.info("  [F_B] Test F1=%.3f  (ΔF_E−F_B=%+.3f)", test_m_b["f1"], test_m_e["f1"] - test_m_b["f1"])

    result = make_result_dict(
        model=net_e.state_dict(), scaler=final_scaler,
        cv_metrics_raw=cv_raw_e, test_metrics=test_m_e,
        y_test_true=y_te.values, y_test_pred=y_test_pred,
        y_test_proba=y_proba_e,
        feature_names=FEATURE_NAMES, target_name=target_col,
        model_name=MODEL_NAME,
        best_params={"architecture": best_arch_name, "hidden_sizes": best_hidden_sizes,
                     "pos_weight": pos_weight},
        ablation={
            "engineered": {"cv_metrics_raw": cv_raw_e, "cv_metrics_summary": summarise_cv(cv_raw_e), "test_metrics": test_m_e},
            "baseline":   {"cv_metrics_raw": cv_raw_b, "cv_metrics_summary": summarise_cv(cv_raw_b), "test_metrics": test_m_b},
        },
        model_architecture=f"input→{'→'.join(str(h) for h in best_hidden_sizes)}→1",
    )
    path = save_result(result, FOLDER, target_col)
    logger.info("  Saved: %s", path)
    return result


def main():
    df = load_data()
    for target in TARGETS:
        run_for_target(df, target)


if __name__ == "__main__":
    main()