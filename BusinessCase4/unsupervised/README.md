# Unsupervised Models — Classical Detectors + PyOD

**Category:** Classical unsupervised anomaly detection (density, proximity, histogram)

---

## Approach

All models are trained on **normal weeks only** (novelty detection setting).
The anomaly threshold is tuned on the CV set by maximising F1.
Optuna is used for hyperparameter search on the three best-performing models.

A directional filter analysis is also included — result: the filter consistently
worsens F1 across all 11 models (boom/crash asymmetry is already implicitly
handled by these density-based methods).

---

## Files

```
unsupervised/
├── README.md                 ← this file
└── UnsupervisedModels.ipynb  ← full pipeline: 8 models + Optuna + analysis
```

---

## Models

| Model | Library | Type |
|---|---|---|
| Isolation Forest | sklearn | Tree-based isolation |
| One-Class SVM | sklearn | Kernel boundary |
| LOF | sklearn | Local density ratio |
| GMM (2 components) | sklearn | Probabilistic density |
| COPOD | pyod | Copula-based empirical CDF |
| ECOD | pyod | Empirical CDF |
| PCA Reconstruction | sklearn | Linear reconstruction error |
| HBOS | pyod | Histogram-based density |
| **LOF (Optuna)** | sklearn | Optimised k, metric |
| **IF (Optuna)** | sklearn | Optimised n_estimators, features |
| **OCSVM (Optuna)** | sklearn | Optimised nu, gamma |

---

## Results (shuffled split — professor's protocol)

### Baseline models

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| LOF | 0.673 | **0.832** | **0.744** | 0.742 |
| Isolation Forest | 0.684 | 0.782 | 0.729 | 0.697 |
| One-Class SVM | 0.664 | 0.798 | 0.725 | 0.724 |
| GMM 2 components | **0.726** | 0.647 | 0.684 | **0.756** |
| COPOD | 0.679 | 0.748 | 0.712 | 0.682 |
| HBOS | 0.660 | 0.782 | 0.715 | 0.672 |
| ECOD | 0.686 | 0.681 | 0.684 | 0.677 |
| PCA Reconstruction | 0.726 | 0.647 | 0.684 | 0.729 |

### After Optuna optimisation

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **OCSVM (Optuna)** | 0.624 | **0.950** | **0.753** | 0.722 |
| LOF (Optuna) | 0.674 | 0.815 | 0.738 | 0.747 |
| IF (Optuna) | 0.671 | 0.807 | 0.733 | 0.706 |

**Best Optuna hyperparameters:**
- LOF: n_neighbors=13, metric='manhattan'
- IF: n_estimators=331, max_samples=0.930, max_features=0.659
- OCSVM: nu=0.755, gamma='scale', kernel='rbf'

---

## Key Findings

1. **LOF is the most robust baseline** (F1=0.744, AUC=0.742). Local density estimation
   adapts to the heterogeneous density of financial data better than global methods.

2. **GMM has the best AUC (0.756)** despite mediocre F1 — it ranks weeks well
   but its threshold calibration is conservative (recall=0.647).

3. **OCSVM Optuna achieves the best overall F1 (0.753)** at the cost of very low
   precision (0.624 — 1 true alarm per 2.6 flags). Suitable for risk management
   (never miss a crisis), not for quant strategies (too many false alarms).

4. **Optuna consistently improves recall over baselines** (+14% for OCSVM, +2% for LOF)
   by pushing the decision boundary outward. The CV set has many anomalies (50%),
   which drives the optimiser toward high-recall solutions.

5. **Directional filter worsens F1 on all 11 models.** These density-based methods
   already have an implicit asymmetry: they flag points far from the training cloud
   regardless of direction. Projecting onto the risk-off direction removes valid flags
   that happen to point in an unexpected direction (e.g., COVID initial USD strengthening).
