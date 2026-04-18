# Recommendation System — SVD and Autoencoder
## Analysis and Business Interpretation

---

## 1. Overview

The recommendation pipeline operates in two sequential stages. First, a binary classifier
identifies which clients have a confirmed investment need (income or accumulation). Second,
a collaborative filtering model scores each client–product pair and selects the
highest-ranked compliant product within the MiFID II eligibility constraints.

Two collaborative filtering architectures are evaluated in parallel:

- **Truncated SVD** — a linear matrix factorisation baseline grounded in the
  Eckart–Young theorem (best rank-*k* approximation in Frobenius norm).
- **Denoising Autoencoder** — a non-linear generalisation of SVD with ReLU
  activations and input dropout, trained to reconstruct masked interaction vectors.

Both models operate on the same client–product interaction matrix **R ∈ {0,1}^(5000×11)**,
built via the *revealed preference* principle: the product assigned to each client is
the one whose Synthetic Risk Indicator (SRI) is closest to the client's `RiskPropensity`,
which is the utility-maximising choice under mean-variance preferences.

---

## 2. Need Identification (Stage 1)

The best-performing model for both targets is XGBoost with SHAP regularisation.

| Target | F1 | Precision | Recall |
|---|---|---|---|
| IncomeInvestment | 0.641 | **0.821** | 0.526 |
| AccumulationInvestment | 0.792 | **0.835** | 0.752 |

Applied to the full 5,000-client population:

| Segment | Clients | Share |
|---|---|---|
| Income need confirmed | 1,889 | 37.8% |
| Accumulation need confirmed | 2,543 | 50.9% |
| At least one confirmed need | **3,465** | **69.3%** |

### Interpretation

The threshold was selected to enforce **precision ≥ 0.75**, the MiFID II floor for
automated suitability assessments. This deliberately accepts low recall — particularly
for income (52.6%) — in exchange for high confidence in each flagged case.

The asymmetry between targets is meaningful. Accumulation investors are better identified
(F1 = 0.79) because their profile is more homogeneous: younger, wealth-accumulating
clients. Income investors are harder to classify (F1 = 0.64) because the signal is
noisier — older clients with heterogeneous wealth levels, not all of whom are actively
seeking income products at the time of observation.

The low income recall (~53%) implies approximately **950 genuine income investors** are
not reached by the automated system. This is a known trade-off: conservative thresholds
protect the bank from mis-selling but leave real needs unmet. These clients represent a
natural target for advisor-led outreach.

---

## 3. Product Catalogue and Interaction Matrix (Stage 2)

Eleven products are defined across two need types, with SRI spanning [0.12, 0.88]:

- **6 Accumulation products** (SRI: 0.15, 0.28, 0.42, 0.56, 0.72, 0.88) — wider range
  reflecting the diverse accumulation client base (51% positive rate).
- **5 Income products** (SRI: 0.12, 0.22, 0.35, 0.50, 0.65) — conservative bias
  consistent with the Modigliani lifecycle hypothesis: income seekers are empirically
  older and wealthier, correlating with lower risk propensity.

The interaction matrix has **4,484 observed interactions** at a density of **8.15%**.
Each client holds at most two interactions (one per need type). The sparsity is structural,
not incidental — it reflects the fact that clients are assigned to a single product per
need based on their risk profile, not a broad portfolio.

---

## 4. SVD Collaborative Filter (Stage 3)

### Methodology

Truncated SVD decomposes **R ≈ U_k Σ_k V_k^T**, where:
- **U_k ∈ ℝ^(5000×k)** — client embeddings in latent preference space
- **V_k^T ∈ ℝ^(k×11)** — product embeddings in the same space
- The score r̂_cp = u_c · v_p measures alignment between a client's latent profile
  and a product's latent characteristics

Rank *k* is selected by holding out 20% of known positive interactions (R=1 entries only)
and measuring held-out AUC. Only positives are held out because R=0 means "not yet
purchased", not "disliked" — evaluating on zeros would inflate AUC trivially.

**Optimal rank: k\* = 5, held-out AUC = 0.6267**

### Constrained Recommendation

For each client with a confirmed need, the eligible product set is filtered by:
1. **Need type**: product type must match the confirmed need.
2. **Risk cap**: SRI_p ≤ RiskPropensity_c — the MiFID II hard constraint, never relaxed.

Products in the eligible set are ranked by descending SVD score.

### Results

| Metric | Value |
|---|---|
| Eligible clients | 3,465 |
| Clients served (≥1 recommendation) | **3,303 (95.3%)** |
| Clients with no compliant product | 185 (4.7%) |
| Suitability pass rate | **100.0%** |

### Known Limitation

SVD minimises ‖R − R̂‖_F treating all zero entries equally. With 92% zeros, the
optimisation is dominated by zero-entry reconstruction, pulling predicted scores for
unobserved pairs toward zero — a *zero-inflation bias*. In practice, this causes SVD
to systematically favour the most conservative products (lowest SRI), regardless of the
client's actual risk tolerance.

---

## 5. Denoising Autoencoder (Stage 4)

### Architecture

```
Input (11) → Dropout(0.3) [training only]
  → Linear(11,8) → BatchNorm(8) → ReLU
  → Linear(8,k)  → ReLU                  ← bottleneck
  → Linear(k,8)  → BatchNorm(8) → ReLU
  → Linear(8,11)                          ← logits → sigmoid at inference
```

The **denoising objective** (Vincent et al., 2008): input dropout randomly corrupts
the interaction vector during training; the loss is computed against the **original**
uncorrupted row. This forces the encoder to learn inter-product co-occurrence patterns
rather than copying the sparse input. At inference, dropout is disabled.

**Class imbalance correction**: BCEWithLogitsLoss with
`pos_weight = n_zeros / n_ones ≈ 11`, restoring gradient balance between the minority
class (purchases) and majority class (non-purchases). This is identical in principle to
the `scale_pos_weight` used in the XGBoost classifier at Stage 1.

**Training**: Adam (lr=0.001), ReduceLROnPlateau (factor=0.5, patience=10),
early stopping (patience=20). Training stopped at epoch 199 with
best validation loss = **0.1765**.

### Rank Selection

**Optimal bottleneck: k\* = 6, held-out AUC = 0.8360**

The AUC curve for the autoencoder consistently lies above the SVD curve across all
tested bottleneck sizes. This confirms the Baldi & Hornik (1989) result: a linear
autoencoder converges to PCA/SVD; adding non-linear activations strictly generalises
the representational capacity.

### Score Discrimination

| Entry type | Mean AE score |
|---|---|
| Zero entries (not purchased) | 0.1400 |
| One entries (purchased) | **0.9858** |

This near-perfect separation confirms the AE has learned to distinguish genuine
purchase patterns from absent ones — a quality not achievable with linear SVD on
this matrix density.

### Results

| Metric | Value |
|---|---|
| Eligible clients | 3,465 |
| Clients served (≥1 recommendation) | **3,303 (95.3%)** |
| Clients with no compliant product | 185 (4.7%) |
| Suitability pass rate | **100.0%** |

---

## 6. Model Comparison (Stage 5)

### Quantitative Summary

| Metric | SVD | Autoencoder |
|---|---|---|
| Held-out AUC | 0.6267 | **0.8360** |
| Coverage rate (of 5,000) | 66.1% | 66.1% |
| Suitability pass rate | 100.0% | 100.0% |
| Avg p̂_income of served clients | 0.704 | 0.704 |
| Bottleneck k\* | 5 | 6 |
| **AUC delta (AE − SVD)** | | **+0.2093** |
| **Recommendation overlap (Jaccard)** | | **0.482** |
| **Coverage delta (AE − SVD)** | | **+0.0%** |

### Key Findings

**1. AUC gap (+0.21): the AE captures what SVD cannot**

A 0.21 AUC lift is substantial. SVD at 0.63 is only marginally above random on this
sparse matrix — it recovers the dominant global patterns (income vs accumulation) but
struggles with within-type variation. The AE at 0.84 demonstrates that non-linear
co-occurrence patterns carry significant predictive signal that the linear decomposition
discards.

**2. Coverage is identical and model-independent**

Both models serve exactly 3,303 clients. Coverage is not a function of the scoring
model — it is determined entirely by the MiFID II risk cap. Clients with RiskPropensity
below the minimum product SRI (0.12) cannot be served by either model.

**3. Jaccard = 0.48: the models disagree on ~half of recommendations**

When they disagree, the pattern is consistent: the AE recommends a higher-SRI product
than SVD. Example disagreements from the sample output:

| Client | Need | SVD | AE |
|---|---|---|---|
| 3 | Income | P10 (SRI=0.50) | **P11 (SRI=0.65)** |
| 5 | Accumulation | P01 (SRI=0.15) | **P03 (SRI=0.42)** |
| 6 | Accumulation | P01 (SRI=0.15) | **P02 (SRI=0.28)** |
| 9 | Income | P08 (SRI=0.22) | **P09 (SRI=0.35)** |

SVD's conservative drift is not a reflection of client preferences — it is an artefact
of zero-inflation bias. The AE's pos_weight correction and non-linear encoder recover
the client's actual risk profile more accurately, as confirmed by the superior AUC.

---

## 7. Business Interpretation

### What the numbers mean operationally

**Regulatory risk: zero.** Both models achieve 100% suitability pass rate. The hard
MiFID II constraint (SRI ≤ RiskPropensity) is enforced at the recommendation layer, not
at the model layer, so it cannot be violated regardless of what either model scores.

**Revenue impact of model choice.**
SVD's conservative bias has a direct cost: it routes clients to lower-SRI products when
their profile supports higher ones. Higher-SRI products carry higher expected returns for
clients and typically higher margins for the bank. With ~48% of recommendations differing
between models, deploying SVD instead of AE means systematically leaving this value on
the table for roughly 1,600 clients.

**The 185 clients with no compliant product.**
These clients have a confirmed investment need but no existing product fits within their
risk tolerance. This is a product gap, not a model failure. Adding a single ultra-
conservative product with SRI < 0.12 (e.g. a capital-protected money-market instrument)
would immediately unlock this segment, adding up to 185 new recommendation opportunities
without any changes to the model or compliance framework.

**The 1,535 clients with no confirmed need.**
These clients were not flagged by the classifier. A subset of them are genuine non-
investors; another subset are borderline cases below the precision threshold. Rather
than treating this group as unreachable, they represent a natural lead list for
advisor-initiated outreach — particularly given that the income classifier missed
approximately 950 genuine investors.

**Explainability trade-off.**
SVD has a natural advantage for regulatory documentation: the latent dimensions have
interpretable financial analogues (lifecycle stage, risk archetype), and the score
r̂_cp = u_c · v_p is a simple inner product. The autoencoder is harder to explain
to auditors, but the constraint logic — need type filter and SRI cap — is fully
transparent and documentable regardless of how the underlying score is produced.

### Recommended deployment

1. **Deploy the autoencoder for primary recommendations.** The +0.21 AUC gap and
   near-perfect score separation (0.986 vs 0.140) demonstrate materially superior
   product matching. The conservative drift of SVD has a concrete revenue cost.

2. **Use SVD as a consistency signal.** When SVD and AE agree (≈50% of cases), the
   recommendation is robust and can be delivered with high confidence. When they
   disagree, flag the case for human advisor review — these are precisely the clients
   where judgment adds value and automated confidence is lower.

3. **Address the product gap.** Commission one ultra-conservative product (SRI < 0.12)
   to serve the 185 currently unreachable clients.

4. **Recover the income miss.** Run a secondary pass over the 1,535 unserved clients
   using a relaxed precision threshold (e.g. 0.65 instead of 0.75), with recommendations
   routed to advisors rather than delivered automatically. This preserves MiFID II
   compliance while recovering a portion of the missed income need population.
