"""
Denoising Autoencoder collaborative filter for the investment product
recommendation pipeline.

Theory
------
A standard autoencoder with linear activations and a bottleneck of width k
is algebraically equivalent to truncated SVD at rank k (Baldi & Hornik, 1989).
Adding non-linear activations (ReLU) strictly generalises SVD: the encoder can
warp and fold the latent space, capturing non-linear co-purchase patterns that
SVD's orthogonal linear decomposition cannot represent.

Architecture
------------
    Input r_c ∈ {0,1}^m
      [Dropout(0.3) — training only, denoising]
      → Linear(m, 8) → BatchNorm(8) → ReLU
      → Linear(8, k) → ReLU              ← bottleneck
      → Linear(k, 8) → BatchNorm(8) → ReLU
      → Linear(8, m)                     ← raw logits (sigmoid at inference)

The BatchNorm layers follow the same rationale as in utils/mlp.py: they
stabilise gradient flow when the input is binary-valued and the network is
shallow.  The output layer produces logits, not probabilities, which is
numerically more stable for BCEWithLogitsLoss during training.

Loss function
-------------
Standard BCE on a sparse binary matrix is dominated by the ~92% zero entries:
a trivially zero-predicting network achieves near-zero loss.  The positive-weight
correction restores gradient balance:

    pos_weight = n_zeros / n_ones  ≈ 11

    L = Σ_{c,p} [w * R[c,p] * BCE + (1 − R[c,p]) * BCE]

This mirrors the scale_pos_weight argument in XGBoost already used for the
same class-imbalance reason in the classification stage.

Denoising
---------
During training, Dropout(0.3) corrupts the input before encoding.  The
reconstruction loss is computed against the ORIGINAL (uncorrupted) R row.
This forces the encoder to learn inter-product co-occurrence patterns rather
than copying the sparse input — the denoising autoencoder objective
(Vincent et al., 2008).  At inference, dropout is disabled (model.eval()).

Implementation
--------------
- InteractionAutoencoder : nn.Module with configurable bottleneck size k.
- train_autoencoder      : fit with early stopping on validation loss.
- score_all_ae           : full (n_clients, n_products) probability matrix.
- select_k_ae            : held-out AUC curve, overlay-able with SVD results.
- recommend_ae           : MiFID II-constrained top-N recommendations.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from utils.products import ACCUMULATION, INCOME

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class InteractionAutoencoder(nn.Module):
    """
    Denoising Autoencoder for client-product interaction reconstruction.

    Input dropout corrupts the interaction vector during training, forcing
    the encoder to learn inter-product co-occurrence patterns rather than
    copying the sparse input.  The output is raw logits; apply sigmoid
    externally at inference time.

    Parameters
    ----------
    n_products : int
        Number of products (input/output dimension).
    k : int
        Bottleneck dimension (latent space size).
    input_dropout : float
        Dropout probability applied to the input during training only.
    """

    def __init__(
        self,
        n_products: int = 11,
        k: int = 4,
        input_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout)
        self.encoder = nn.Sequential(
            nn.Linear(n_products, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, k),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, n_products),  # logits — sigmoid applied externally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits; input dropout active only in training mode."""
        if self.training:
            x = self.input_drop(x)
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_autoencoder(
    R: np.ndarray,
    k: int,
    epochs: int = 200,
    batch_size: int = 32,
    val_frac: float = 0.2,
    random_state: int = 42,
    es_patience: int = 20,
    verbose: bool = True,
) -> InteractionAutoencoder:
    """
    Train an InteractionAutoencoder with early stopping on validation loss.

    The model is trained to reconstruct each client's interaction row.
    Input dropout during training implements the denoising objective:
    the loss is computed against the ORIGINAL row, not the corrupted input.

    Parameters
    ----------
    R : np.ndarray
        Binary interaction matrix (n_clients, n_products).
    k : int
        Bottleneck dimension.
    epochs : int
        Maximum training epochs (default 200; early stopping may stop sooner).
    batch_size : int
        Mini-batch size (default 32, matching mlp.py).
    val_frac : float
        Fraction of client rows held out for validation and early stopping.
    random_state : int
        Random seed for reproducibility.
    es_patience : int
        Early stopping: halt after this many epochs without improvement.
    verbose : bool
        If True, print validation loss every 50 epochs.

    Returns
    -------
    InteractionAutoencoder
        Best model state restored from the epoch with lowest validation loss.
    """
    rng = np.random.default_rng(random_state)
    n = R.shape[0]
    n_val = max(batch_size, int(np.floor(val_frac * n)))

    val_rows = rng.choice(n, size=n_val, replace=False)
    tr_rows  = np.setdiff1d(np.arange(n), val_rows)

    R_t = torch.FloatTensor(R.astype(float))
    tr_loader  = DataLoader(
        TensorDataset(R_t[tr_rows]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(R_t[val_rows]), batch_size=batch_size)

    # pos_weight corrects for ~92% zero-entry dominance in the gradient
    pos_weight = torch.tensor((R == 0).sum() / max((R == 1).sum(), 1), dtype=torch.float32)

    model     = InteractionAutoencoder(n_products=R.shape[1], k=k)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for (x_batch,) in tr_loader:
            optimizer.zero_grad()
            # Reconstruction target is the ORIGINAL row, not the noisy input.
            # model.forward applies dropout internally when model.training=True.
            loss = criterion(model(x_batch), x_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                val_loss += criterion(model(x_batch), x_batch).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k_: v.clone() for k_, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= es_patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  (best val_loss={best_val_loss:.4f})")
                break

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_all_ae(model: InteractionAutoencoder, R: np.ndarray) -> np.ndarray:
    """
    Compute the full (n_clients, n_products) reconstruction probability matrix.

    Scores are sigmoid(logit) ∈ (0, 1) and used for ranking only.  A high
    score r̂_cp means the autoencoder predicts that client c is likely to
    purchase product p, given their observed purchase history as context.

    Parameters
    ----------
    model : InteractionAutoencoder
        Trained model (returned by :func:`train_autoencoder`).
    R : np.ndarray
        Full interaction matrix (n_clients, n_products).  Used as input context.

    Returns
    -------
    np.ndarray
        Probability matrix of shape (n_clients, n_products), dtype float32.
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(R.astype(float)))
        return torch.sigmoid(logits).numpy()


# ---------------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------------


def select_k_ae(
    R: np.ndarray,
    k_range: range = range(1, 7),
    val_frac: float = 0.2,
    random_state: int = 42,
    selection_epochs: int = 60,
    batch_size: int = 32,
    plot: bool = True,
    svd_auc_scores: Optional[Dict[int, float]] = None,
) -> Dict:
    """
    Select the optimal autoencoder bottleneck size via held-out AUC.

    Uses the same held-out evaluation protocol as SVD's select_k: 20% of
    known positive interactions are masked from the input; AUC is computed
    on those held-out pairs versus all zeros.

    Parameters
    ----------
    R : np.ndarray
        Binary interaction matrix (n_clients, n_products).
    k_range : range
        Candidate bottleneck sizes to evaluate.
    val_frac : float
        Fraction of positive entries to hold out for AUC evaluation.
    random_state : int
        Random seed.
    selection_epochs : int
        Training epochs per candidate k (fewer than final training for speed).
    batch_size : int
        Mini-batch size.
    plot : bool
        If True, plot AUC vs k, optionally overlaid with SVD curve.
    svd_auc_scores : dict or None
        If provided, overlay the SVD AUC curve from :func:`utils.svd_rec.select_k`
        on the same axes for direct comparison.

    Returns
    -------
    dict
        Keys: k_star (int), auc_scores (dict k→auc), k_range (list).
    """
    rng = np.random.default_rng(random_state)

    pos_rows, pos_cols = np.where(R == 1)
    n_pos = len(pos_rows)
    n_val = max(1, int(np.floor(val_frac * n_pos)))

    val_idx  = rng.choice(n_pos, size=n_val, replace=False)
    val_rows = pos_rows[val_idx]
    val_cols = pos_cols[val_idx]

    R_train = R.astype(float).copy()
    R_train[val_rows, val_cols] = 0.0

    neg_rows, neg_cols = np.where(R_train == 0)
    eval_rows   = np.concatenate([val_rows,   neg_rows])
    eval_cols   = np.concatenate([val_cols,   neg_cols])
    eval_labels = np.concatenate([np.ones(n_val), np.zeros(len(neg_rows))])

    auc_scores: Dict[int, float] = {}

    for k in k_range:
        model = train_autoencoder(
            R_train, k=k, epochs=selection_epochs, batch_size=batch_size,
            val_frac=0.2, random_state=random_state, es_patience=10, verbose=False,
        )
        R_hat = score_all_ae(model, R_train)
        scores = R_hat[eval_rows, eval_cols]
        auc_scores[k] = float(roc_auc_score(eval_labels, scores))
        print(f"  k={k:2d}  AUC={auc_scores[k]:.4f}")

    k_star = max(auc_scores, key=auc_scores.get)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks  = list(auc_scores.keys())
        auc = list(auc_scores.values())
        ax.plot(ks, auc, marker="s", linewidth=2, color="#DD8452", label="Autoencoder")
        if svd_auc_scores is not None:
            svd_ks  = [k for k in svd_auc_scores if k in range(min(ks), max(ks) + 1)]
            svd_auc = [svd_auc_scores[k] for k in svd_ks]
            ax.plot(svd_ks, svd_auc, marker="o", linewidth=2, color="#4C72B0",
                    linestyle="--", label="SVD")
        ax.axvline(k_star, color="red", linestyle=":", label=f"AE k* = {k_star}")
        ax.set_xlabel("Bottleneck size k")
        ax.set_ylabel("AUC (held-out positives vs all zeros)")
        ax.set_title("Rank selection — SVD vs Autoencoder")
        ax.legend()
        plt.tight_layout()
        plt.show()

    print(f"AE k* = {k_star}  (held-out AUC = {auc_scores[k_star]:.4f})")
    return {"k_star": k_star, "auc_scores": auc_scores, "k_range": list(auc_scores.keys())}


# ---------------------------------------------------------------------------
# Constrained recommendation
# ---------------------------------------------------------------------------


def recommend_ae(
    client_indices: List[int],
    R_hat_ae: np.ndarray,
    clients_df: pd.DataFrame,
    products_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Return MiFID II-compliant Autoencoder recommendations for a list of clients.

    Eligible set and constraint logic are identical to recommend_svd:
      1. Need type filter : type(p) must match a confirmed need of the client.
      2. Risk cap         : SRI_p <= RiskPropensity_c  (MiFID II hard constraint).

    Parameters
    ----------
    client_indices : list of int
        Row indices into R_hat_ae (and clients_df).
    R_hat_ae : np.ndarray
        Probability matrix from :func:`score_all_ae` (n_clients, n_products).
    clients_df : pd.DataFrame
        Must contain columns: RiskPropensity, need_income, need_accum.
    products_df : pd.DataFrame
        Product catalogue from utils.products.get_products().
    top_n : int
        Maximum recommendations per need type per client.

    Returns
    -------
    pd.DataFrame
        Columns: client_idx, need_type, rank, product_id, SRI, ae_score, status.
    """
    need_map = {
        "need_income": INCOME,
        "need_accum":  ACCUMULATION,
    }

    rows = []
    for c in client_indices:
        client_row = clients_df.loc[c, ["RiskPropensity"] + list(need_map.keys())]
        risk = client_row["RiskPropensity"]

        for need_col, prod_type in need_map.items():
            if not client_row[need_col]:
                continue

            mask     = (products_df["type"] == prod_type) & (products_df["SRI"] <= risk)
            eligible = products_df[mask]

            if eligible.empty:
                rows.append({
                    "client_idx": c,
                    "need_type":  prod_type,
                    "rank":       None,
                    "product_id": None,
                    "SRI":        None,
                    "ae_score":   None,
                    "status":     "no_compliant_product",
                })
                continue

            eligible = eligible.assign(
                ae_score=R_hat_ae[c, eligible.index]
            ).sort_values("ae_score", ascending=False).head(top_n)

            for rank, row in enumerate(eligible.itertuples(index=False), start=1):
                rows.append({
                    "client_idx": c,
                    "need_type":  prod_type,
                    "rank":       rank,
                    "product_id": row.product_id,
                    "SRI":        row.SRI,
                    "ae_score":   round(row.ae_score, 6),
                    "status":     "ok",
                })

    return pd.DataFrame(rows)
