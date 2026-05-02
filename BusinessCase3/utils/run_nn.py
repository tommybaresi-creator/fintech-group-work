"""
Neural-network weight generator for Portfolio Replication.

Architecture: a WeightGenerator network (MLP or LSTM) that takes a window
of past futures returns as input and outputs portfolio weights directly.
Training optimises tracking error (MSE on replica vs target returns) with
an optional L1 exposure penalty. VaR scaling is applied post-inference to
enforce the UCITS 1M VaR(99%) ≤ 20% constraint.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

PICKLE_DIR = Path("data/picklefiles")
OUTPUTS_DIR = Path("outputs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(stem: str) -> None:
    path = OUTPUTS_DIR / f"{stem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


def _compute_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon_weeks: int = 4,
) -> float:
    """
    Parametric (normal) VaR scaled to a multi-week horizon.

    Parameters
    ----------
    returns : np.ndarray
        Recent weekly returns (e.g. last 52 observations).
    confidence : float
        Confidence level (e.g. 0.99 for 99 %).
    horizon_weeks : int
        Horizon in weeks (4 ≈ 1 month).

    Returns
    -------
    float
        VaR as a positive number.
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    var = -(mu * horizon_weeks + sigma * np.sqrt(horizon_weeks) * z)
    return float(max(var, 0.0))


# ── Model ─────────────────────────────────────────────────────────────────────

# Base class resolves to nn.Module when torch is available, object otherwise.
# Methods that use torch will raise ImportError at call time if torch is absent.
_BaseModule = nn.Module if TORCH_AVAILABLE else object


class WeightGenerator(_BaseModule):  # type: ignore[misc]
    """
    Portfolio weight generator network (MLP or LSTM).

    Takes a sliding window of futures returns as input and outputs a
    raw (unconstrained) weight vector. VaR scaling is applied outside
    the network.

    Parameters
    ----------
    input_dim : int
        Number of input features (= number of futures).
    window : int
        Lookback window length (time steps).
    hidden_dims : list of int
        Hidden layer sizes. For LSTM, ``hidden_dims[0]`` is the hidden
        state size and ``len(hidden_dims)`` is the number of layers.
    output_dim : int
        Number of output weights (= number of futures).
    mode : str
        ``'mlp'`` — flatten window then dense layers.
        ``'lstm'`` — LSTM encoder + linear head.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        window: int,
        hidden_dims: List[int],
        output_dim: int,
        mode: str = "mlp",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.window = window
        self.input_dim = input_dim

        if mode == "mlp":
            flat = input_dim * window
            layers: List[nn.Module] = []
            prev = flat
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        elif mode == "lstm":
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dims[0],
                num_layers=len(hidden_dims),
                batch_first=True,
                dropout=dropout if len(hidden_dims) > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dims[-1], output_dim)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'mlp' or 'lstm'.")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, window, n_futures)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_futures)`` — unnormalised weights.
        """
        if self.mode == "mlp":
            return self.net(x.reshape(x.size(0), -1))
        else:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


# ── Data preparation ──────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, window: int) -> np.ndarray:
    """
    Build sliding-window sequences from a 2-D array.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, n_features)``.
    window : int
        Lookback window length.

    Returns
    -------
    np.ndarray
        Shape ``(T - window, window, n_features)``.
    """
    return np.stack([X[i - window:i] for i in range(window, len(X))])


# ── Loss ──────────────────────────────────────────────────────────────────────

def _portfolio_loss(
    weights: "torch.Tensor",
    target_returns: "torch.Tensor",
    factor_returns: "torch.Tensor",
    l1_penalty: float = 0.0,
) -> "torch.Tensor":
    """
    Tracking-error MSE loss with optional L1 gross-exposure penalty.

    Parameters
    ----------
    weights : torch.Tensor
        Shape ``(batch, n_futures)``.
    target_returns : torch.Tensor
        Shape ``(batch,)``.
    factor_returns : torch.Tensor
        Shape ``(batch, n_futures)`` — futures returns at prediction date.
    l1_penalty : float
        Coefficient on the mean absolute weight (encourages sparsity).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    replica = (weights * factor_returns).sum(dim=1)
    mse = ((replica - target_returns) ** 2).mean()
    l1 = weights.abs().mean() * l1_penalty
    return mse + l1


# ── Training ──────────────────────────────────────────────────────────────────

def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
) -> Tuple["WeightGenerator", List[float], List[float]]:
    """
    Train a WeightGenerator on a fixed train / val split with early stopping.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature array, shape ``(T_train, n_futures)``.
    y_train : np.ndarray
        Training target array, shape ``(T_train,)``.
    X_val : np.ndarray
        Validation feature array, shape ``(T_val, n_futures)``.
    y_val : np.ndarray
        Validation target array, shape ``(T_val,)``.
    config : dict
        Hyperparameter dictionary. Expected keys:

        ``window``
            Lookback window (int).
        ``mode``
            ``'mlp'`` or ``'lstm'`` (str).
        ``hidden_dims``
            Hidden layer sizes (list of int).
        ``dropout``
            Dropout probability (float).
        ``lr``
            Adam learning rate (float).
        ``epochs``
            Maximum training epochs (int).
        ``batch_size``
            Mini-batch size (int).
        ``l1_penalty``
            L1 exposure penalty coefficient (float).
        ``patience``
            Early-stopping patience in epochs (int).

    Returns
    -------
    tuple
        ``(model, train_loss_history, val_loss_history)``

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    window     = config["window"]
    n_futures  = X_train.shape[1]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build sequences
    Xs_tr = _make_sequences(X_train, window)   # (T-W, W, F)
    ys_tr = y_train[window:]
    fs_tr = X_train[window:]                    # factor returns at prediction step

    Xs_vl = _make_sequences(X_val, window)
    ys_vl = y_val[window:]
    fs_vl = X_val[window:]

    def _t(a: np.ndarray) -> "torch.Tensor":
        return torch.FloatTensor(a).to(device)

    dataset = TensorDataset(_t(Xs_tr), _t(ys_tr), _t(fs_tr))
    loader  = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        drop_last=False,
    )
    Xv_t, yv_t, fv_t = _t(Xs_vl), _t(ys_vl), _t(fs_vl)

    model = WeightGenerator(
        input_dim=n_futures,
        window=window,
        hidden_dims=config.get("hidden_dims", [64, 32]),
        output_dim=n_futures,
        mode=config.get("mode", "mlp"),
        dropout=config.get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    l1_pen     = config.get("l1_penalty", 0.0)
    patience   = config.get("patience", 30)
    epochs     = config.get("epochs", 300)
    best_val   = float("inf")
    best_state: Optional[Dict] = None
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        # ── Training step ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb, fb in loader:
            optimizer.zero_grad()
            loss = _portfolio_loss(model(xb), yb, fb, l1_pen)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xs_tr)

        # ── Validation step ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = _portfolio_loss(
                model(Xv_t), yv_t, fv_t, l1_pen
            ).item()

        scheduler.step(val_loss)
        train_hist.append(epoch_loss)
        val_hist.append(val_loss)

        # ── Early stopping ───────────────────────────────────────────────────
        if val_loss < best_val - 1e-8:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "Epoch %4d/%d | train=%.6f | val=%.6f | patience %d/%d",
                epoch, epochs, epoch_loss, val_loss, no_improve, patience,
            )

        if no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Best model restored (val_loss=%.6f)", best_val)

    return model, train_hist, val_hist


# ── Inference ─────────────────────────────────────────────────────────────────

def evaluate_nn(
    model: "WeightGenerator",
    X: np.ndarray,
    y: pd.Series,
    config: Dict,
    feature_names: List[str],
    max_var_threshold: float = 0.20,
    var_lookback: int = 52,
) -> Dict:
    """
    Run walk-forward inference with VaR scaling on the full dataset.

    For each time step ``t ≥ window``:
    1. Feed ``X[t-window : t]`` into the model to get raw weights.
    2. If ≥ ``var_lookback`` replica returns are available, compute VaR
       and scale weights down if VaR > ``max_var_threshold``.
    3. Record replica return for period ``t``.

    Parameters
    ----------
    model : WeightGenerator
        Trained model.
    X : np.ndarray
        Full feature array, shape ``(T, n_futures)``.
    y : pd.Series
        Full target return series (length ``T``); provides dates.
    config : dict
        Must contain ``'window'``.
    feature_names : list of str
        Column names for the weights DataFrame.
    max_var_threshold : float
        UCITS 1M VaR(99 %) limit. Default 0.20.
    var_lookback : int
        Number of recent returns used to estimate VaR. Default 52.

    Returns
    -------
    dict
        Keys: ``replica_returns`` (pd.Series), ``target_returns`` (pd.Series),
        ``weights_history`` (pd.DataFrame), ``gross_exposures`` (list),
        ``var_values`` (list), ``scaling_factors`` (list).

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    window = config["window"]
    device = next(model.parameters()).device
    model.eval()

    dates        = y.index.to_numpy()
    replica_list: List[float] = []
    target_list:  List[float] = []
    weights_list: List[np.ndarray] = []
    gross_exp:    List[float] = []
    var_vals:     List[float] = []
    scale_facts:  List[float] = []

    with torch.no_grad():
        for t in range(window, len(X)):
            x_win = X[t - window: t]                              # (W, F)
            x_ten = torch.FloatTensor(x_win).unsqueeze(0).to(device)  # (1, W, F)
            w     = model(x_ten).squeeze(0).cpu().numpy()         # (F,)

            # VaR scaling
            scale = 1.0
            if len(replica_list) >= var_lookback:
                recent = np.array(replica_list[-var_lookback:])
                var    = _compute_var(recent, confidence=0.99, horizon_weeks=4)
                if var > max_var_threshold:
                    scale = max_var_threshold / var
                    w     = w * scale
                    var   = var * scale         # recompute after scaling
            else:
                var = float("nan")

            var_vals.append(var)
            scale_facts.append(scale)
            gross_exp.append(float(np.abs(w).sum()))
            weights_list.append(w.copy())

            replica_ret = float(np.dot(X[t], w))
            target_ret  = float(y.iloc[t])
            replica_list.append(replica_ret)
            target_list.append(target_ret)

    out_dates = dates[window:]
    weights_df = pd.DataFrame(weights_list, index=out_dates, columns=feature_names)

    return {
        "replica_returns": pd.Series(replica_list, index=out_dates, name="Replica"),
        "target_returns":  pd.Series(target_list,  index=out_dates, name="Target"),
        "weights_history": weights_df,
        "gross_exposures": pd.Series(gross_exp,   index=out_dates, name="gross_exposure"),
        "var_values":      pd.Series(var_vals,    index=out_dates, name="var"),
        "scaling_factors": pd.Series(scale_facts, index=out_dates, name="scale"),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    label: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train", linewidth=1.5)
    ax.plot(val_losses,   label="Val",   linewidth=1.5)
    ax.set_title(f"Training Curves — {label}", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(stem)
    plt.close(fig)


def _plot_weights(
    weights_df: pd.DataFrame,
    label: str,
    stem: str,
    top_n: int = 8,
) -> None:
    top_cols = weights_df.abs().mean().sort_values(ascending=False).head(top_n).index
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in top_cols:
        ax.plot(weights_df[col], label=col, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {top_n} Portfolio Weights Over Time — {label}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(stem)
    plt.close(fig)


def _plot_gross_var(
    result: Dict,
    label: str,
    stem: str,
    max_var_threshold: float = 0.20,
) -> None:
    dates = result["replica_returns"].index
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(dates, result["gross_exposures"], color="purple", linewidth=1.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="100% (no leverage)")
    ax.axhline(2.0, color="red",  linestyle="--", linewidth=0.8, label="200% (UCITS limit)")
    ax.set_title(f"Gross Exposure Over Time — {label}", fontsize=12)
    ax.set_ylabel("Gross Exposure")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    var_clean = pd.Series(result["var_values"], index=dates).dropna()
    ax.plot(var_clean, color="orange", linewidth=1.3)
    ax.axhline(
        max_var_threshold, color="red", linestyle="--",
        linewidth=1.0, label=f"VaR limit ({max_var_threshold:.0%})",
    )
    ax.set_title(f"1M VaR(99%) Over Time — {label}", fontsize=12)
    ax.set_ylabel("VaR")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    _save_fig(stem)
    plt.close(fig)


# ── Default configurations ────────────────────────────────────────────────────

DEFAULT_CONFIGS: List[Dict] = [
    # MLP — short window, no L1
    dict(mode="mlp",  window=26, hidden_dims=[64, 32],      dropout=0.2, lr=1e-3,
         epochs=300,  batch_size=32, l1_penalty=0.0,        patience=30),
    # MLP — long window, light L1
    dict(mode="mlp",  window=52, hidden_dims=[64, 32],      dropout=0.2, lr=1e-3,
         epochs=300,  batch_size=32, l1_penalty=1e-3,       patience=30),
    # MLP — deep, short window, L1
    dict(mode="mlp",  window=26, hidden_dims=[128, 64, 32], dropout=0.3, lr=5e-4,
         epochs=300,  batch_size=32, l1_penalty=1e-3,       patience=30),
    # LSTM — long window
    dict(mode="lstm", window=52, hidden_dims=[64],          dropout=0.2, lr=1e-3,
         epochs=300,  batch_size=32, l1_penalty=0.0,        patience=30),
]


# ── Main entry point ──────────────────────────────────────────────────────────

def run_nn(
    X: pd.DataFrame,
    y: pd.Series,
    configs: Optional[List[Dict]] = None,
    train_frac:         float = 0.60,
    val_frac:           float = 0.15,
    max_var_threshold:  float = 0.20,
) -> Dict:
    """
    Full NN experiment: train each config, evaluate on test set, compare.

    Data split:

    * **Train**: first ``train_frac`` of observations — model parameters.
    * **Val**: next ``val_frac`` — early stopping only (no look-ahead).
    * **Test**: remainder — reported out-of-sample performance.

    Saves ``data/picklefiles/nn_results.pkl`` and diagnostic plots to
    ``outputs/``.

    Parameters
    ----------
    X : pd.DataFrame
        Futures return feature matrix, shape ``(T, n_futures)``.
    y : pd.Series
        Monster index returns (target), length ``T``.
    configs : list of dict, optional
        Hyperparameter configurations. Defaults to ``DEFAULT_CONFIGS``.
    train_frac : float
        Fraction of data for training (default 0.60).
    val_frac : float
        Fraction of data for validation (default 0.15).
    max_var_threshold : float
        UCITS 1M VaR(99%) constraint (default 0.20).

    Returns
    -------
    dict
        Keys:

        ``best_result``
            Result dict for the best config (by information ratio).
        ``all_results``
            List of result dicts for every config.
        ``metrics_df``
            pd.DataFrame with one row per config.
        ``best_config``
            Hyperparameter dict of the winning config.
        ``feature_names``
            List of futures column names.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    RuntimeError
        If all configurations fail.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    _ensure_dirs()
    logger.info("=" * 60)
    logger.info("NN EXPERIMENT — START")
    logger.info("=" * 60)

    if configs is None:
        configs = DEFAULT_CONFIGS

    # ── Align and convert ────────────────────────────────────────────────────
    common = X.index.intersection(y.index)
    X_df   = X.loc[common]
    y_s    = y.loc[common]
    feat   = X_df.columns.tolist()
    T      = len(X_df)

    X_arr  = X_df.values.astype(np.float32)
    y_arr  = y_s.values.astype(np.float32)

    # ── Split indices ────────────────────────────────────────────────────────
    t_train = int(T * train_frac)
    t_val   = int(T * (train_frac + val_frac))
    logger.info(
        "Data split: train=%d | val=%d | test=%d (total=%d)",
        t_train, t_val - t_train, T - t_val, T,
    )

    X_tr, y_tr = X_arr[:t_train],        y_arr[:t_train]
    X_vl, y_vl = X_arr[t_train:t_val],   y_arr[t_train:t_val]
    test_start  = y_s.index[t_val]

    sns.set_theme(style="whitegrid")
    all_results: List[Dict]  = []
    metrics_rows: List[Dict] = []

    for i, cfg in enumerate(configs):
        label = (
            f"{cfg['mode'].upper()}"
            f"_w{cfg['window']}"
            f"_h{'x'.join(str(h) for h in cfg['hidden_dims'])}"
            f"_l1{cfg['l1_penalty']}"
        )
        logger.info("-" * 50)
        logger.info("Config %d/%d: %s", i + 1, len(configs), label)

        try:
            model, tr_hist, vl_hist = train_nn(X_tr, y_tr, X_vl, y_vl, cfg)

            # Training curves
            _plot_training_curves(
                tr_hist, vl_hist, label,
                stem=f"nn_cfg{i+1:02d}_training_curves",
            )

            # Full walk-forward inference
            res = evaluate_nn(
                model, X_arr, y_s, cfg, feat, max_var_threshold
            )

            # Restrict to test period for fair comparison
            res["replica_returns"] = res["replica_returns"].loc[test_start:]
            res["target_returns"]  = res["target_returns"].loc[test_start:]
            res["weights_history"] = res["weights_history"].loc[test_start:]
            res["gross_exposures"] = res["gross_exposures"].loc[test_start:]
            res["var_values"]      = res["var_values"].loc[test_start:]
            res["scaling_factors"] = res["scaling_factors"].loc[test_start:]
            res["config"]       = cfg
            res["model_name"]   = label
            res["model"]        = model
            res["train_losses"] = tr_hist
            res["val_losses"]   = vl_hist

            all_results.append(res)

            # Weights and risk plots
            _plot_weights(
                res["weights_history"], label,
                stem=f"nn_cfg{i+1:02d}_weights",
            )
            _plot_gross_var(
                res, label,
                stem=f"nn_cfg{i+1:02d}_gross_var",
                max_var_threshold=max_var_threshold,
            )

            # Metrics
            try:
                from utils.evaluation import compute_metrics
            except ImportError:
                from evaluation import compute_metrics

            metrics = compute_metrics(
                res["replica_returns"], res["target_returns"], label
            )
            metrics_rows.append(metrics)

            logger.info(
                "  TE=%.4f | IR=%.3f | Corr=%.3f",
                metrics["tracking_error"],
                metrics["information_ratio"],
                metrics["correlation"],
            )

        except Exception as exc:
            logger.error("Config %d failed: %s", i + 1, exc, exc_info=True)

    if not all_results:
        raise RuntimeError("All NN configurations failed — check logs above.")

    metrics_df = pd.DataFrame(metrics_rows)
    if "model" in metrics_df.columns:
        metrics_df = metrics_df.set_index("model")

    # ── Best config by information ratio ────────────────────────────────────
    best_idx    = metrics_df["information_ratio"].idxmax()
    best_pos    = list(metrics_df.index).index(best_idx)
    best_result = all_results[best_pos]
    best_config = configs[best_pos]

    logger.info("=" * 60)
    logger.info("Best config: %s", best_idx)
    logger.info(
        "  IR=%.4f | TE=%.4f | Corr=%.4f",
        metrics_df.loc[best_idx, "information_ratio"],
        metrics_df.loc[best_idx, "tracking_error"],
        metrics_df.loc[best_idx, "correlation"],
    )

    # ── Save pickle ──────────────────────────────────────────────────────────
    payload = {
        "best_result":   best_result,
        "all_results":   all_results,
        "metrics_df":    metrics_df,
        "best_config":   best_config,
        "feature_names": feat,
    }
    pkl_path = PICKLE_DIR / "nn_results.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Pickle saved → %s", pkl_path)
    logger.info("NN EXPERIMENT — DONE")
    logger.info("=" * 60)

    return payload


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import setup_logging
    setup_logging()

    pkl_data = Path("data/picklefiles/data_loader.pkl")
    if not pkl_data.exists():
        logger.error("data_loader.pkl not found — run run_data_loader() first.")
        sys.exit(1)

    with open(pkl_data, "rb") as fh:
        data = pickle.load(fh)

    run_nn(data["X"], data["y"])