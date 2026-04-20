"""
SVD collaborative filter for the investment product recommendation pipeline.

Theory
------
Truncated Singular Value Decomposition decomposes the client-product
interaction matrix R ∈ {0,1}^(n×m) as:

    R ≈ U_k Σ_k V_k^T

where:
  U_k ∈ ℝ^(n×k)  — client embeddings in latent investment-style space
  Σ_k ∈ ℝ^(k×k)  — diagonal matrix of k largest singular values
  V_k^T ∈ ℝ^(k×m) — product embeddings in the same latent space

By the Eckart-Young theorem, this is the best rank-k approximation of R
in the Frobenius norm among all rank-k matrices.  The latent dimensions
capture dominant co-purchase patterns (e.g., "conservative income seeker",
"aggressive accumulator"), corresponding to the principal directions of
variation in client purchase behaviour.

The predicted score for an unseen client-product pair is:

    r̂_cp = u_c · v_p   (inner product after absorbing Σ into U or V)

Scores are used for ranking within the MiFID II-compliant eligible set
(need type match + SRI_p ≤ RiskPropensity_c); they are not calibrated
probabilities and should never be thresholded as such.

Rank selection
--------------
k is selected via held-out AUC: 20% of known interactions (1-entries only)
are masked, the SVD is fitted on the remaining entries, and AUC is computed
on the held-out set.  Only 1-entries are masked because R=0 means "not yet
purchased", not "disliked" — evaluating on random zeros would inflate AUC
trivially.

Limitation
----------
Standard SVD minimises ||R - R̂||_F treating all zero entries equally.
With ~92% zeros, the optimisation is dominated by zero-entry reconstruction,
pulling scores for unobserved pairs toward zero (zero-inflation bias).
This is why the hard risk-cap filter is essential: SVD scores are used for
relative ranking within the compliant set, not as absolute thresholds.

Implementation
--------------
- select_k: held-out AUC curve over k ∈ {1, …, k_max}, returns optimal k.
- fit_svd: truncated SVD via scipy.sparse.linalg.svds (singular values in
  descending order).
- score_all_svd: full (n_clients, n_products) score matrix.
- recommend_svd: constrained top-N recommendations for a list of clients.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score

from utils.products import ACCUMULATION, INCOME


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_svd_embeddings(
    U: np.ndarray,
    clients_df: pd.DataFrame,
    propensity_col: str = 'p_hat_income',
    figsize: tuple = (7, 5),
) -> None:
    """
    Scatter plot of client embeddings in the first two SVD latent dimensions.

    Parameters
    ----------
    U : np.ndarray
        Client embedding matrix (n_clients, k), output of fit_svd().
    clients_df : pd.DataFrame
        Must contain the column named by propensity_col.
    propensity_col : str
        Column used to colour the scatter points (default 'p_hat_income').
    figsize : tuple
        Figure size passed to plt.subplots().
    """
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        U[:, 0], U[:, 1],
        c=clients_df[propensity_col],
        cmap='RdYlGn', s=3, alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax, label=propensity_col)
    ax.set_xlabel('Latent dimension 1')
    ax.set_ylabel('Latent dimension 2')
    ax.set_title(f'Client embeddings (U) — coloured by {propensity_col}')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------------


def select_k(
    R: np.ndarray,
    k_range: range = range(1, 11),
    val_frac: float = 0.2,
    random_state: int = 42,
    plot: bool = True,
) -> Dict:
    """
    Select the optimal SVD rank via held-out AUC.

    Only known positive interactions (R=1) are held out.  Evaluating on
    randomly sampled zero entries would not measure collaborative filtering
    quality — it would measure the trivial ability to score zeros low.

    Parameters
    ----------
    R : np.ndarray
        Binary interaction matrix of shape (n_clients, n_products).
    k_range : range
        Candidate ranks to evaluate.  Must satisfy max(k_range) < min(n, m).
    val_frac : float
        Fraction of positive entries to hold out for validation (default 0.2).
    random_state : int
        Random seed for reproducibility.
    plot : bool
        If True, plot AUC vs k and mark the optimal rank.

    Returns
    -------
    dict
        Keys: k_star (int), auc_scores (dict k→auc), k_range (list).
    """
    rng = np.random.default_rng(random_state)

    # Indices of all positive interactions
    pos_rows, pos_cols = np.where(R == 1)
    n_pos = len(pos_rows)
    n_val = max(1, int(np.floor(val_frac * n_pos)))

    val_idx = rng.choice(n_pos, size=n_val, replace=False)
    val_rows, val_cols = pos_rows[val_idx], pos_cols[val_idx]

    # Build training matrix with held-out entries zeroed
    R_train = R.astype(float).copy()
    R_train[val_rows, val_cols] = 0.0

    # All zero entries (including newly masked ones) serve as negatives
    neg_rows, neg_cols = np.where(R_train == 0)

    # Labels: 1 for held-out positives, 0 for all zeros
    eval_rows   = np.concatenate([val_rows,   neg_rows])
    eval_cols   = np.concatenate([val_cols,   neg_cols])
    eval_labels = np.concatenate([np.ones(n_val), np.zeros(len(neg_rows))])

    auc_scores: Dict[int, float] = {}
    k_max = min(R.shape) - 1  # svds requires k < min(n, m)

    for k in k_range:
        if k > k_max:
            break
        U, sigma, Vt = svds(R_train, k=k)
        # Compute scores only at the evaluation indices — avoids materialising
        # the full (n_clients, n_products) R_hat and the k×k diag temporary.
        # U_s[i] · Vt.T[j]  ==  (U Σ Vt)[i, j]
        U_s = U * sigma                                         # (n_clients, k)
        scores = (U_s[eval_rows] * Vt.T[eval_cols]).sum(axis=1)
        auc_scores[k] = float(roc_auc_score(eval_labels, scores))

    k_star = max(auc_scores, key=auc_scores.get)

    if plot:
        ks  = list(auc_scores.keys())
        auc = list(auc_scores.values())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, auc, marker="o", linewidth=2, label="Held-out AUC")
        ax.axvline(k_star, color="red", linestyle="--", label=f"k* = {k_star}")
        ax.set_xlabel("Rank k")
        ax.set_ylabel("AUC (held-out positives vs all zeros)")
        ax.set_title("SVD rank selection — held-out AUC")
        ax.legend()
        plt.tight_layout()
        plt.show()

    print(f"k* = {k_star}  (held-out AUC = {auc_scores[k_star]:.4f})")
    return {"k_star": k_star, "auc_scores": auc_scores, "k_range": list(auc_scores.keys())}


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def fit_svd(R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the truncated SVD of R, returning factors in descending order.

    scipy.sparse.linalg.svds returns singular values in ascending order;
    this function flips them to the conventional descending order so that
    the first latent dimension always captures the most variance.

    Parameters
    ----------
    R : np.ndarray
        Binary interaction matrix of shape (n_clients, n_products).
    k : int
        Number of latent dimensions.  Must satisfy k < min(n_clients, n_products).

    Returns
    -------
    tuple (U, sigma, Vt)
        U     : (n_clients, k)  — client embeddings
        sigma : (k,)            — singular values, descending
        Vt    : (k, n_products) — product embeddings (transposed)
    """
    U, sigma, Vt = svds(R.astype(float), k=k)
    # svds returns ascending order — flip to descending
    order = np.argsort(sigma)[::-1]
    return U[:, order], sigma[order], Vt[order, :]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_all_svd(
    U: np.ndarray,
    sigma: np.ndarray,
    Vt: np.ndarray,
) -> np.ndarray:
    """
    Compute the full (n_clients, n_products) score matrix R̂ = U Σ V^T.

    Scores are real-valued and used for ranking only.  They are not
    calibrated probabilities; do not threshold them.

    Parameters
    ----------
    U : np.ndarray
        Client embedding matrix (n_clients, k).
    sigma : np.ndarray
        Singular values (k,), descending.
    Vt : np.ndarray
        Product embedding matrix transposed (k, n_products).

    Returns
    -------
    np.ndarray
        Score matrix of shape (n_clients, n_products).
    """
    return U @ np.diag(sigma) @ Vt


# ---------------------------------------------------------------------------
# Constrained recommendation
# ---------------------------------------------------------------------------


def recommend_svd(
    client_indices: List[int],
    R_hat: np.ndarray,
    clients_df: pd.DataFrame,
    products_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Return MiFID II-compliant SVD recommendations for a list of clients.

    For each client, the eligible product set is determined by:
      1. Need type filter  : type(p) must match a confirmed need of the client.
      2. Risk cap          : SRI_p <= RiskPropensity_c  (MiFID II hard constraint).

    Products in the eligible set are ranked by descending SVD score.
    If no product passes both filters, the client appears in the output with
    product_id = None and a note in the 'status' column.

    Parameters
    ----------
    client_indices : list of int
        Row indices into R_hat (and clients_df).
    R_hat : np.ndarray
        Score matrix of shape (n_clients, n_products) from :func:`score_all_svd`.
    clients_df : pd.DataFrame
        Must contain columns: RiskPropensity, need_income, need_accum.
        Index must align with rows of R_hat.
    products_df : pd.DataFrame
        Product catalogue from utils.products.get_products().
        Index must align with columns of R_hat.
    top_n : int
        Maximum number of recommendations per need type per client.

    Returns
    -------
    pd.DataFrame
        Columns: client_idx, need_type, rank, product_id, SRI, svd_score, status.
        Sorted by client_idx, need_type, rank.
    """
    need_map = {
        "need_income": INCOME,
        "need_accum":  ACCUMULATION,
    }

    rows = []
    for c in client_indices:
        # Read both needed values in a single .loc call (one label lookup).
        client_row = clients_df.loc[c, ["RiskPropensity"] + list(need_map.keys())]
        risk = client_row["RiskPropensity"]

        for need_col, prod_type in need_map.items():
            if not client_row[need_col]:
                continue

            # Eligible: correct type AND SRI <= client risk
            mask = (products_df["type"] == prod_type) & (products_df["SRI"] <= risk)
            eligible = products_df[mask]

            if eligible.empty:
                rows.append({
                    "client_idx": c,
                    "need_type":  prod_type,
                    "rank":       None,
                    "product_id": None,
                    "SRI":        None,
                    "svd_score":  None,
                    "status":     "no_compliant_product",
                })
                continue

            # .assign() returns a new DataFrame without a defensive copy of the
            # underlying data blocks, avoiding the SettingWithCopyWarning.
            eligible = eligible.assign(
                svd_score=R_hat[c, eligible.index]
            ).sort_values("svd_score", ascending=False).head(top_n)

            # itertuples is an order of magnitude faster than iterrows because
            # it yields namedtuples instead of constructing a Series per row.
            for rank, row in enumerate(eligible.itertuples(index=False), start=1):
                rows.append({
                    "client_idx": c,
                    "need_type":  prod_type,
                    "rank":       rank,
                    "product_id": row.product_id,
                    "SRI":        row.SRI,
                    "svd_score":  round(row.svd_score, 6),
                    "status":     "ok",
                })

    return pd.DataFrame(rows)
