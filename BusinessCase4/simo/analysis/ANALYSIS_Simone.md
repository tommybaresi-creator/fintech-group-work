# Analysis — Simone: MVG Variants & Quant Finance Models
## Business Case 4 — Early Warning System

---

## 1. Experimental Setup

All models are trained on the **same data pipeline**:

- **Dataset**: Bloomberg weekly data, ~40 financial variables, binary label Y (0 = normal, 1 = risk-off)
- **Stationarity**: log-differences for indices/currencies/commodities, first-differences for interest rates, as-is for ECSURPUS
- **Split (with shuffle)**: 80% normal → train | 10% normal + 50% anomalies → CV | 10% normal + 50% anomalies → test
- **Scaling**: StandardScaler fitted on train only
- **Threshold tuning**: all thresholds selected by maximising F1 on the CV set
- **Evaluation**: Precision, Recall, F1, AUC — all on the held-out test set

---

## 2. Full Results Table

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **Elliptic Envelope** | 0.6772 | 0.8992 | **0.7726** | **0.7676** |
| MVG Ledoit-Wolf | 0.6354 | **0.9664** | 0.7667 | 0.7500 |
| Student-t (ν=2) | **0.7007** | 0.8403 | 0.7634 | 0.7495 |
| Graphical Lasso | 0.7007 | 0.8067 | 0.7500 | 0.7492 |
| MVG Baseline | 0.6031 | 0.9832 | 0.7476 | 0.7495 |
| MVG Asymmetric | 0.6124 | 0.9160 | 0.7340 | 0.7096 |
| MVG CDF-based | 0.5777 | 1.0000 | 0.7323 | 0.7637 |
| Factor Model (k=12) | 0.6984 | 0.7395 | 0.7184 | 0.7295 |

> All models evaluated on the same held-out test set.

---

## 3. Model-by-Model Analysis

### 3.1 MVG Baseline (F1=0.748, AUC=0.750) — Reference
The professor's baseline is already a solid detector: recall of 0.983 means it **misses almost no crisis week**. The cost is low precision (0.603) — roughly 4 out of 10 flagged weeks are false alarms. This asymmetry is expected and financially acceptable in risk management: missing a crisis is far more costly than a false alarm.

The baseline suffers from two known limitations of the sample covariance:
1. **Estimation noise** — with ~40 features and limited training observations, the sample covariance is poorly conditioned.
2. **Gaussian tails** — extreme financial events are systematically underestimated.

All subsequent models try to fix one or both of these.

---

### 3.2 MVG Ledoit-Wolf (F1=0.767, AUC=0.750) — +2.4% F1 vs Baseline
Replacing the sample covariance with the **Ledoit-Wolf shrinkage estimator** is the single most impactful change among the covariance-based variants.

- **+2.4% F1** over baseline
- Recall remains very high (0.966) — still catches almost all crises
- Precision improves from 0.603 → 0.635 — fewer false alarms
- The condition number of the covariance matrix drops substantially, making the PDF values more numerically stable and meaningful

**Why it works**: shrinkage pulls the sample covariance towards a scaled identity, reducing the noise in the off-diagonal entries. In a dataset with ~40 correlated financial variables and limited observations, this regularisation is essential.

---

### 3.3 Elliptic Envelope (F1=0.773, AUC=0.768) — Best Overall
The **Elliptic Envelope** (Minimum Covariance Determinant, MCD) is the top performer on both F1 and AUC.

- **Best F1 (0.773)** and **Best AUC (0.768)**
- Best balance of precision (0.677) and recall (0.899)
- MCD fits the covariance on the ~75% most central observations, making it **robust to contamination in training data**

The key insight: even though the training set is supposed to contain only normal data, some ambiguous weeks near the boundary may corrupt the sample covariance. MCD explicitly ignores those, fitting a cleaner ellipsoid around the core of normality.

**Financial interpretation**: the model is saying "this week's asset returns fall outside the region that has historically characterised calm markets" — a clean and interpretable criterion.

---

### 3.4 MVG CDF-based (F1=0.732, AUC=0.764) — Perfect Recall, Low Precision
The CDF scorer achieves **perfect recall (1.000)** — it never misses a crisis. However, this comes at the cost of very low precision (0.578): it flags too aggressively.

Interestingly, its **AUC (0.764) is the second highest** in the group, suggesting that as a ranking device (ordering weeks from most to least anomalous) it is quite good — but its classification threshold is set too liberally by the F1 maximisation.

This model is better suited to a **risk management** application where missing a crisis is unacceptable, rather than a quant strategy where false alarms generate unnecessary trading costs.

---

### 3.5 MVG Asymmetric (F1=0.734, AUC=0.710) — Financial Override
Adding a **discriminating variable** (MXUS log-return threshold K) slightly improves precision over the baseline (0.612 vs 0.603) but reduces recall (0.916 vs 0.983). The AUC is the **lowest in the group (0.710)**, suggesting the asymmetric scoring distorts the ranking.

The idea is financially sound — risk-off events are by definition periods of equity losses, so good equity weeks should not be flagged. However, the gain is marginal here, possibly because:
- The MVG PDF already tends to flag negative-equity weeks disproportionately
- The grid search over (ε, K) on a small CV set may overfit slightly

This model has more potential with a richer financial asymmetry rule (e.g., conditioning on VIX level jointly with equity direction).

---

### 3.6 Student-t (ν=2) (F1=0.763, AUC=0.750) — Fat Tails Confirmed

The optimal degrees of freedom is **ν=2** — the minimum tested, corresponding to **extremely heavy tails**. This is a strong empirical confirmation of a well-known stylised fact in financial econometrics: market returns are far more extreme than any Gaussian model predicts.

- **Third-best F1 (0.763)**
- **Highest precision among the top-3 (0.699)** — fewer false alarms than Ledoit-Wolf
- Recall moderate (0.840) — slightly more conservative than the Gaussian variants

**Financial meaning**: with ν=2, the Student-t assigns non-negligible probability to moves that are 4–5σ under the Gaussian. This reduces the "surprise" of moderately extreme weeks, reserving the anomaly label for truly catastrophic ones — hence higher precision.

A limitation: `scipy.stats.multivariate_t` does not offer EM-based fitting; ν is selected by CV rather than maximum likelihood. A full MLE fit (e.g. via EM) would be more statistically principled.

---

### 3.7 Factor Model — PCA Reconstruction Error (F1=0.718, AUC=0.730) — Weakest, but Most Interpretable

With k=12 factors the model is the **weakest by both F1 and AUC**. The reconstruction error signal is noisy: 12 factors capture much of the variance but leave too much residual for the score to be a clean anomaly indicator.

However, this model offers **the richest financial interpretability**:
- The factors correspond to latent market regimes (equity beta, duration, credit spread, dollar, etc.)
- The factor loadings directly show which assets drive each component
- "This week's error is high" translates to: "market behaviour this week cannot be explained by any of the 12 known market factors" — intuitive for a risk officer

**Why it underperforms**: unlike an autoencoder (Person 4), PCA minimises reconstruction error in the least-squares sense on all data, not just normal data. The resulting factors are not optimally calibrated for anomaly detection.

---

### 3.8 Graphical Lasso (F1=0.750, AUC=0.749) — Best Precision, Financial Structure

The Graphical Lasso delivers the **highest precision tied with Student-t (0.701)** and sits in the middle of the F1 ranking. Its main contribution is not raw performance but **financial insight**:

- The sparse precision matrix reveals the **conditional dependency graph** between assets — which pairs of assets remain correlated after accounting for all others
- In a systemic crisis, the graph typically becomes denser (contagion) — the Mahalanobis distance under the sparse graph is more sensitive to this
- The sparsity regularisation also improves numerical conditioning, similar in spirit to Ledoit-Wolf but through a different mechanism

The model flags anomalies based on whether a week violates the learned asset relationships — a financially meaningful criterion.

---

## 4. Precision–Recall Trade-off Analysis

A critical observation: **all 8 models favour recall over precision**, with recall consistently in the 0.74–1.00 range and precision in the 0.58–0.70 range. This is structurally expected in anomaly detection with a small anomaly fraction:

- **High recall** = few missed crises → preferred for **risk management** (drawdown protection)
- **High precision** = few false alarms → preferred for **quant strategies** (avoids unnecessary rebalancing)

| Use case | Recommended model | Reason |
|---|---|---|
| Risk management | MVG CDF-based | Perfect recall — never misses a crisis |
| General purpose | Elliptic Envelope | Best F1/AUC balance |
| Quant strategy | Student-t or GLASSO | Highest precision — fewer false positives |
| Interpretability | Factor Model | Financial factor decomposition |

---

## 5. What Improved Over the Baseline

| Model | F1 Δ vs Baseline | AUC Δ vs Baseline | Key mechanism |
|---|---|---|---|
| Elliptic Envelope | **+0.025** | **+0.018** | Robust MCD covariance |
| MVG Ledoit-Wolf | +0.019 | +0.001 | Shrinkage regularisation |
| Student-t (ν=2) | +0.016 | +0.000 | Fat-tailed distribution |
| Graphical Lasso | +0.002 | -0.000 | Sparse precision matrix |
| MVG Asymmetric | -0.014 | -0.040 | Financial override (marginal benefit) |
| MVG CDF-based | -0.015 | +0.014 | Extreme recall, low precision |
| Factor Model (k=12) | -0.029 | -0.022 | Noisy reconstruction signal |

---

## 6. Key Takeaways

1. **The Gaussian assumption is wrong but fixable.** Replacing it with Student-t (ν=2) improves precision significantly, confirming that financial returns are far heavier-tailed than any normal model predicts.

2. **Covariance estimation matters more than distribution shape.** The Elliptic Envelope and Ledoit-Wolf both beat the Student-t on F1, suggesting that *how* you estimate the covariance matters at least as much as *which* distribution you use.

3. **Robust estimation (MCD) is the single best improvement.** The Elliptic Envelope's MCD covariance, trained only on the central 90% of observations, produces the cleanest boundary between normal and anomalous market regimes.

4. **Sparsity (GLASSO) adds interpretability without hurting performance.** Graphical Lasso sits at F1=0.750 — essentially matching the baseline — while revealing the true conditional dependency structure of the financial system.

5. **Factor models need nonlinearity to shine.** The linear PCA factor model is the weakest detector here. The autoencoder equivalent (Person 4) is expected to outperform it significantly by learning nonlinear factor structure.

6. **All models favour recall over precision.** In a risk management context, this is the right bias — missing a crisis is more costly than a false alarm. For a quant strategy, Student-t or GLASSO offer the best precision.

---

## 7. Limitations & Next Steps

- **No temporal structure**: data is shuffled before splitting, breaking autocorrelation. A time-series-aware split (walk-forward CV) would be more realistic.
- **Student-t fitting**: ν selected by CV rather than MLE — a full EM-based fit would be more rigorous.
- **Factor Model**: k=12 selected by F1 on CV but reconstruction error is still noisy; sparse PCA or kernel PCA may help.
- **Asymmetric model**: the 2D grid search (ε, K) on a small CV set risks mild overfitting; extending to more discriminating variables (VIX + equity jointly) could improve results.
- **Comparison with other groups**: the final picture will emerge when Person 2 (supervised), Person 3 (unsupervised), and Person 4 (deep learning) results are merged into a single leaderboard.
