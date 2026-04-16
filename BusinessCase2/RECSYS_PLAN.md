# Recommendation System — Implementation Plan

## Overview

Three new utility modules + notebook cells extending `rec_sys.ipynb`.
SVD and Autoencoder are independent of each other but both depend on Tasks 1–2.

```
Task 1: products.py          (product catalogue + interaction matrix)
Task 2: propensity bridge    (Stage 1 output → clients_df)
Task 3: svd_rec.py           (SVD collaborative filter)
Task 4: autoencoder_rec.py   (Autoencoder collaborative filter)
Task 5: comparison           (evaluation metrics in notebook)
```

---

## Task 1 — Product Catalogue and Interaction Matrix

**File:** `utils/products.py`

### 1.1 Define product catalogue

Create a DataFrame of 11 products:

| Field | Values |
|---|---|
| `product_id` | P01 … P11 |
| `type` | `Accumulation` (6 products) or `Income` (5 products) |
| `SRI` | float in [0.12, 0.88] |

**SRI grids:**
- Accumulation: `{0.15, 0.28, 0.42, 0.56, 0.72, 0.88}` — wider range because `AccumulationInvestment` has 51% positive rate, implying a more diverse client base that needs wider product coverage.
- Income: `{0.12, 0.22, 0.35, 0.50, 0.65}` — skewed conservative because income seekers are empirically older and wealthier (Modigliani lifecycle hypothesis), which correlates with lower `RiskPropensity`.

### 1.2 Build interaction matrix

**Function:** `build_interaction_matrix(df, products_df) -> np.ndarray (5000, 11)`

**Rule (revealed preference):**

```
R[c, p] = 1  iff  type(p) == NeedType(c)
               and p = argmin_{p': type(p') = NeedType(c)} |SRI_p' − RiskPropensity_c|
```

For each need type where the client's label is 1, assign the interaction to the single product whose SRI is closest to the client's `RiskPropensity`. This is the utility-maximizing product under mean-variance preferences — the one a compliant advisor would have sold.

Each client has at most 2 interactions (one per need type). Expected matrix density ≈ 8%.

### 1.3 Sanity checks

- `R.shape == (5000, 11)`
- `R.sum(axis=1).max() <= 2` (each client bought at most 2 products)
- `R.sum(axis=0)` bar chart — product popularity should roughly track the `RiskPropensity` distribution of the client base
- Total interactions ≈ 4,450 (38% × 5000 income + 51% × 5000 accum)

---

## Task 2 — Propensity Score Bridge

**Location:** new cells in `rec_sys.ipynb`, after existing SHAP cells

### 2.1 Full-dataset inference

Apply the best pickled model per target to the full 5,000-client feature matrix `X = build_features(df)`.

Output: `clients_df` with columns:

| Column | Type | Description |
|---|---|---|
| `RiskPropensity` | float | from raw dataset |
| `p_hat_income` | float [0,1] | calibrated propensity for IncomeInvestment |
| `p_hat_accum` | float [0,1] | calibrated propensity for AccumulationInvestment |
| `need_income` | bool | `p_hat_income >= threshold_income` |
| `need_accum` | bool | `p_hat_accum >= threshold_accum` |

Thresholds come from `threshold_info` in each pickle (PR-curve optimal, precision ≥ 0.75).

### 2.2 Why calibrated probabilities matter here

The propensity scores feed directly into the final priority ranking. An uncalibrated score of 0.9 that actually corresponds to a 0.6 true positive rate would over-serve clients who don't genuinely have the need. Isotonic calibration (already applied in all pickles) ensures the scores are reliable probability estimates, verified by Brier score < 0.19 for both targets.

---

## Task 3 — SVD Collaborative Filter

**File:** `utils/svd_rec.py`

### 3.1 Rank selection

**Function:** `select_k(R, k_range=range(1, 11), val_frac=0.2, random_state=42) -> dict`

1. Randomly mask 20% of the 1-entries in R (held-out set). Only 1-entries are masked because 0 means "not yet purchased", not "disliked" — evaluating on randomly sampled 0-entries would give a trivially high AUC.
2. For each k in `k_range`:
   - Fit `scipy.sparse.linalg.svds(R_train, k=k)`
   - Compute `R_hat = U @ diag(sigma) @ Vt`
   - Compute AUC on held-out (client, product) pairs
3. Return `{k: auc}` dict and plot AUC vs k.

Expected result: AUC plateaus after k ≈ 2–4. The scree plot of singular values will show the same elbow — the first 2–4 dimensions correspond to the dominant financial archetypes (lifecycle stage, risk tolerance, wealth level).

### 3.2 Decomposition

**Function:** `fit_svd(R, k) -> (U, sigma, Vt)`

```python
U, sigma, Vt = svds(R, k=k)   # U: (5000,k), sigma: (k,), Vt: (k,11)
```

`svds` returns the k largest singular values. `U` rows are client embeddings in latent space; `Vt` columns are product embeddings. The inner product `u_c · v_p` (with singular values absorbed) measures alignment between the client's latent preferences and the product's latent characteristics.

### 3.3 Scoring

**Function:** `score_all_svd(U, sigma, Vt) -> np.ndarray (5000, 11)`

```python
R_hat = U @ np.diag(sigma) @ Vt
```

Scores are in ℝ — not probabilities. They are only used for **ranking within the eligible set**, so no calibration is needed. Do not apply sigmoid: the ordinal ranking is what matters, not the absolute value.

### 3.4 Constrained recommendation

**Function:** `recommend_svd(client_idx, R_hat, clients_df, products_df, top_n=3) -> DataFrame`

For each client:
1. Determine eligible product set: `type(p) == NeedType(c)` AND `SRI_p <= RiskPropensity_c`
2. Sort eligible products by `R_hat[c, p]` descending
3. Return top-N

If no product passes the risk cap, return empty (document this client as "no compliant recommendation available"). Do not relax the constraint.

---

## Task 4 — Autoencoder Collaborative Filter

**File:** `utils/autoencoder_rec.py`

### 4.1 Architecture

```
Input (11)
  → Dense(8) → BatchNorm → ReLU
  → Dense(k)             → ReLU    ← bottleneck
  → Dense(8) → BatchNorm → ReLU
  → Dense(11)            → Sigmoid
```

**Class:** `InteractionAutoencoder(nn.Module)`, k as constructor parameter.

Follows existing MLP conventions in `utils/mlp.py`: PyTorch, Adam(lr=0.001), ReduceLROnPlateau, BCELoss.

`Sigmoid` output maps each reconstructed score to [0,1], interpretable as purchase probability per product. This also allows direct comparison with propensity scores downstream.

### 4.2 Loss function

Standard BCE is dominated by 0-entries (92% of the matrix). Correct with positive-weight:

```python
pos_weight = (R == 0).sum() / (R == 1).sum()  # ≈ 11
criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

This is the identical principle as `scale_pos_weight` in the existing XGBoost — restores gradient balance between the minority class (purchases) and majority class (non-purchases).

### 4.3 Input dropout (denoising)

During training, apply Dropout(p=0.3) to the input vector before the encoder. The loss is computed against the **original** (unmasked) R row.

Without this, the network can trivially copy the few 1-entries directly to output. Input dropout forces the encoder to reconstruct masked interactions from context, learning inter-product co-occurrence patterns — the collaborative signal. This is the denoising autoencoder (Vincent et al., 2008).

At inference, dropout is disabled (`model.eval()`).

### 4.4 Training loop

**Function:** `train_autoencoder(R, k, epochs=200, batch_size=32, val_frac=0.2) -> model`

- Split R rows into train/val (80/20)
- DataLoader with `batch_size=32` (same as existing MLP)
- ReduceLROnPlateau on validation loss (patience=10)
- Early stopping (patience=20) to prevent overfitting on the small matrix
- Save best model state_dict by validation loss

### 4.5 Rank selection

**Function:** `select_k_ae(R, k_range=range(1, 11)) -> dict`

Same held-out AUC procedure as SVD Task 3.1. Plot both SVD and AE AUC-vs-k curves on the same axes for direct comparison.

### 4.6 Scoring

**Function:** `score_all_ae(model, R) -> np.ndarray (5000, 11)`

```python
model.eval()
with torch.no_grad():
    R_hat_ae = model(torch.tensor(R, dtype=torch.float32)).numpy()
```

### 4.7 Constrained recommendation

Identical to SVD Task 3.4. Same eligible set logic, same fallback policy.

---

## Task 5 — Comparison and Evaluation

**Location:** final cells in `rec_sys.ipynb`

### 5.1 Per-model metrics

Compute for SVD and AE separately across all 5,000 clients:

| Metric | Formula | Business meaning |
|---|---|---|
| **Coverage rate** | `clients with ≥1 recommendation / 5000` | What fraction of the client base the system can serve |
| **Suitability pass rate** | `recommendations with SRI_p ≤ RiskPropensity_c / total recommendations` | MiFID II compliance rate |
| **Need alignment rate** | `recommendations matching confirmed need type / total recommendations` | Advisor accuracy |
| **Avg propensity of served clients** | `mean p̂` for clients who received a recommendation | Whether the system prioritises high-confidence needs |

### 5.2 Model comparison

| Metric | Business meaning |
|---|---|
| **Recommendation overlap** (Jaccard of top-1 per client) | High overlap → SVD and AE are redundant, pick simpler. Low overlap → AE captures signal SVD cannot → hybrid justified. |
| **Coverage delta** (AE coverage − SVD coverage) | Whether non-linearity expands the set of clients that can be served |
| **AUC on held-out interactions** (from rank selection) | Intrinsic quality of latent factor recovery |

### 5.3 Final recommendation table (sample)

Display a sample of 10 clients showing:
`client_id | need_type | p_hat | SVD_top1 (SRI) | AE_top1 (SRI) | agree`

---

## File summary

| File | New / Modified | Contents |
|---|---|---|
| `utils/products.py` | New | Product catalogue, `build_interaction_matrix` |
| `utils/svd_rec.py` | New | `select_k`, `fit_svd`, `score_all_svd`, `recommend_svd` |
| `utils/autoencoder_rec.py` | New | `InteractionAutoencoder`, `train_autoencoder`, `select_k_ae`, `score_all_ae`, `recommend_ae` |
| `rec_sys.ipynb` | Modified | Tasks 2–5 implemented as new cells |

---

## Dependency graph

```
products.py  ──┬──→  svd_rec.py       ──┐
               │                         ├──→  rec_sys.ipynb (comparison)
               └──→  autoencoder_rec.py ──┘
                          ↑
               (both consume R from products.py)

rec_sys.ipynb (existing: propensity loading + SHAP)
               ↓
           Task 2: clients_df
               ↓
           Tasks 3 + 4: SVD and AE recommendations
               ↓
           Task 5: comparison
```