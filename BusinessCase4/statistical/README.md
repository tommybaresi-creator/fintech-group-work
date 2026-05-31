# Statistical Models — MVG Variants & Quantitative Finance Detectors

**Category:** Statistical anomaly detection (unsupervised, novelty detection)

---

## Approach

All models follow the **novelty detection** paradigm: trained exclusively on normal
market weeks, they learn the geometry of calm periods and flag deviations.

Two families of models are implemented:

**MVG Variants** — engineer the covariance matrix of the Multivariate Gaussian:
- Baseline sample covariance
- Ledoit-Wolf shrinkage (regularisation for p ≈ n settings)
- Elliptic Envelope (MCD — robust to contamination in training data)
- CDF-based scoring (replaces PDF with joint CDF)
- Asymmetric variant (equity-direction override via MXUS threshold)

**Quantitative Finance Models** — exploit financial structure:
- Student-t (ν=2) — fat-tailed distribution confirming financial leptokurtosis
- Factor Model (PCA reconstruction error) — Barra-style latent market factors
- Graphical Lasso — sparse precision matrix revealing conditional asset dependencies

---

## Files

```
statistical/
├── README.md
├── MVG.ipynb           ← 8 models × shuffled + walk-forward split + comparison
├── MVG_advanced.ipynb  ← EVT/POT threshold + directional filter + lead time
└── results/
    ├── results_simone_shuffled.csv     ← group leaderboard (professor's protocol)
    ├── results_simone_walkforward.csv  ← temporal honest evaluation
    └── results_simone_advanced.csv     ← advanced pipeline (EVT + filter)
```

The data path resolves automatically regardless of where Jupyter is launched.

---

## Results

### Shuffled split (professor's protocol — group leaderboard)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **Elliptic Envelope** | 0.677 | 0.899 | **0.773** | **0.768** |
| MVG Ledoit-Wolf | 0.635 | 0.966 | 0.767 | 0.750 |
| Student-t (ν=2) | **0.699** | 0.840 | 0.763 | 0.750 |
| Graphical Lasso | 0.701 | 0.807 | 0.750 | 0.749 |
| MVG Baseline | 0.603 | 0.983 | 0.748 | 0.750 |
| MVG Asymmetric | 0.612 | 0.916 | 0.734 | 0.710 |
| MVG CDF-based | 0.578 | 1.000 | 0.732 | 0.764 |
| Factor Model (k=12) | 0.698 | 0.740 | 0.718 | 0.730 |

### Advanced pipeline (walk-forward split — honest evaluation)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **GLASSO + EVT + Filter** | **0.889** | 0.276 | 0.421 | 0.777 |
| Student-t + EVT + Filter | 0.692 | 0.310 | **0.429** | **0.777** |
| EE + EVT + Filter | 0.667 | 0.276 | 0.390 | 0.728 |

---

## Key Findings

1. **Covariance estimation > distribution choice.** Elliptic Envelope (MCD) and
   Ledoit-Wolf beat Student-t on F1 despite the Gaussian assumption being wrong.

2. **Shuffled split inflates F1 by over 100%.** Average F1 drops from 0.748 (shuffled)
   to 0.337 (walk-forward). AUC remains stable at 0.73–0.78 — models rank correctly;
   the threshold is the problem. MVG Asymmetric is the most robust (walk-forward F1=0.522).

3. **EVT/POT threshold replaces grid search.** With only 6% anomalies in CV,
   F1 grid search is unreliable (precision ~0.15). GPD on the normal tail improves
   F1 by +39–73% at a stated 5% false-alarm rate.

4. **Directional filter removes boom false alarms.** GLASSO reaches precision=0.889
   (8/9 alarms are real crises). 25–29% of all flags were boom anomalies.

5. **Lead time up to 11 weeks.** Elliptic Envelope gives advance warning on 3/7
   crisis episodes. Median lead = 0 (exogenous crises not anticipated in advance).
