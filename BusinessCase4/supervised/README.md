# Supervised Models — Random Forest, SVM, XGBoost

**Category:** Supervised classification with class-imbalance handling

---

## Approach

Unlike the other three sections, supervised models have access to **crisis labels
at training time**. They are trained on the combined train+CV set (with labels)
and evaluated on the held-out test set.

The main challenge is **class imbalance** — crisis weeks represent ~21% of the data,
causing classifiers to be biased toward the majority class. Three strategies are
compared: class weights, SMOTE oversampling, and cost-sensitive learning.

A temporal feature engineering step adds Dragon King build-up signals (rolling
Hawkes-style features) before the standard classification pipeline.

---

## Files

```
supervised/
├── README.md          ← this file
└── supervised.ipynb   ← full pipeline: baseline → imbalance → Optuna
```

---

## Models

| Model | Imbalance strategy | Hyperparameter search |
|---|---|---|
| Random Forest Baseline | `class_weight='balanced'` | Default |
| SVM Baseline | `class_weight='balanced'` | Default |
| Random Forest + SMOTE | SMOTE oversampling | Default |
| XGBoost | `scale_pos_weight` | Default |
| XGBoost + SMOTE | SMOTE oversampling | Default |
| **Best Supervised (Optuna)** | Best of above | Optuna (F1 on CV) |

---

## Results (shuffled split — professor's protocol)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| XGBoost | **0.946** | 0.609 | **0.741** | 0.907 |
| XGBoost + SMOTE | 0.922 | 0.617 | 0.740 | 0.902 |
| SVM Baseline | 0.851 | **0.644** | 0.733 | 0.867 |
| RF + SMOTE | 0.932 | 0.600 | 0.730 | **0.921** |
| Best Supervised (Optuna) | 0.928 | 0.557 | 0.696 | 0.911 |
| RF Baseline | 1.000 | 0.339 | 0.506 | 0.918 |

**Best Optuna hyperparameters (XGBoost):**
- n_estimators: 151 · max_depth: 9 · learning_rate: 0.025
- subsample: 0.896 · colsample_bytree: 0.926
- balancing: scale_pos_weight

---

## Key Findings

1. **Supervised models achieve the highest AUC (0.87–0.92) on the shuffled split.**
   However, this reflects look-ahead bias — the shuffled split allows future crisis
   weeks to appear in training. Direct comparison with unsupervised/statistical
   models on this split is not entirely fair.

2. **SMOTE does not consistently outperform scale_pos_weight.** XGBoost with
   scale_pos_weight achieves comparable F1 to XGBoost + SMOTE (0.741 vs 0.740)
   without inflating the training set.

3. **Random Forest Baseline achieves perfect precision (1.000) but very low recall
   (0.339).** The balanced class weight creates a very conservative classifier —
   only the most unambiguous crises are flagged.

4. **SHAP analysis** identifies the most informative features: VIX, MXUS (US equities),
   and credit spread indices dominate the supervised signal — consistent with the
   directional filter findings in the statistical module.

5. **Optuna does not improve over default XGBoost** on this dataset (F1: 0.696 vs 0.741).
   The CV set has too few anomalies (≈10% crisis rate) for reliable hyperparameter search.
