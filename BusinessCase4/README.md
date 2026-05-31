# Business Case 4 — Early Warning System for Financial Markets

**Politecnico di Milano · Fintech Course · A.Y. 2024–2025**
**Instructor:** Raffaele Zenti — Co-Founder & Chief AI Officer, Wealthype-AI B SpA

---

## Overview

This project builds an **Early Warning System (EWS)** for detecting abnormal financial
market regimes (risk-off / crisis periods) using multivariate anomaly detection on
weekly Bloomberg data spanning January 2000 – April 2021.

The core insight motivating the architecture is that the binary label "risk-off"
conflates three structurally different phenomena:

| Crisis type | Example | Information in prices | Detection approach |
|---|---|---|---|
| **Black Swan** | COVID-19 | None before onset | Change-point / static AE |
| **Dragon King** | GFC 2008 | Sequential build-up | LSTM / Hawkes branching ratio |
| **Boom anomaly** | 2017 bull run | Equidistant from crash | Directional filter |

Each model family addresses one or more of these problems from a different angle.

---

## Repository Structure

```
BusinessCase4/
│
├── README.md                          ← this file
├── CLAUDE.md                          ← full project context for AI assistants
├── pyproject.toml                     ← shared Python environment (uv)
├── uv.lock                            ← pinned dependency versions
│
├── data/
│   └── Dataset4_EWS.xlsx              ← Bloomberg weekly dataset (not in git)
│
├── EarlyWarningSystemPoliMI.ipynb     ← professor's reference notebook
├── Zenti_Business_Case_4.pdf          ← lecture slides
│
├── statistical/                       ← MVG variants + quant finance models
├── deep_learning/                     ← AE, LSTM-AE, HMM, Hawkes pipeline
├── supervised/                        ← RF, SVM, XGBoost
└── unsupervised/                      ← IF, OCSVM, LOF, GMM, COPOD, HBOS
```

> **Dataset:** place `Dataset4_EWS.xlsx` in `data/` before running any notebook.
> Always launch Jupyter from the `BusinessCase4/` root so all notebooks resolve
> the path `data/Dataset4_EWS.xlsx` correctly.

---

## Dataset

| Property | Value |
|---|---|
| Source | Bloomberg (weekly) |
| Period | January 2000 – April 2021 |
| Observations | 1,110 weeks |
| Features | 42 financial indicators |
| Label Y | 0 = normal (risk-on), 1 = anomalous (risk-off / crisis) |
| Crisis weeks | 237 (21.3%) |

**Feature groups:**

| Group | Tickers | N | Transformation |
|---|---|---|---|
| Equity indices | MXUS, MXEU, MXJP, MXCN, MXBR, MXIN, MXRU | 7 | Log-diff |
| Bond indices | EMUSTRUU, LF94TRUU, LF98TRUU, LG30TRUU, LMBITR, LP01TREU, LUACTRUU, LUMSTRUU | 8 | Log-diff |
| Interest rates / yields | EONIA, GT10, GTDEM*, GTGBP*, GTITL*, GTJPY*, US0001M, USGG* | 18 | First-diff |
| Currencies | DXY, GBP, JPY | 3 | Log-diff |
| Commodities | Cl1, CRY, BDIY, XAUBGNL | 4 | Log-diff |
| Volatility | VIX | 1 | Log-diff |
| Macro | ECSURPUS | 1 | As-is |

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for reproducible
dependency management.

```bash
# Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install all dependencies
cd BusinessCase4
uv sync

# Launch JupyterLab from project root
uv run jupyter lab
```

All notebooks are self-contained and can be run independently.
Run cells top-to-bottom; no inter-notebook dependencies.

---

## Results Summary

### Shuffled split (professor's protocol — model comparison baseline)

| Model | Category | F1 | AUC | Precision | Recall |
|---|---|---|---|---|---|
| SVM | Supervised | **0.733** | 0.867 | 0.851 | 0.644 |
| XGBoost | Supervised | 0.741 | 0.907 | **0.946** | 0.609 |
| RF + SMOTE | Supervised | 0.730 | **0.921** | 0.932 | 0.600 |
| LOF | Unsupervised | 0.744 | 0.742 | 0.673 | **0.832** |
| OCSVM Optuna | Unsupervised | **0.753** | 0.722 | 0.624 | 0.950 |
| Elliptic Envelope | Statistical | 0.773 | 0.768 | 0.677 | 0.899 |
| MVG Ledoit-Wolf | Statistical | 0.767 | 0.750 | 0.635 | 0.966 |
| Student-t (ν=2) | Statistical | 0.763 | 0.750 | 0.699 | 0.840 |
| Graphical Lasso | Statistical | 0.750 | 0.749 | 0.701 | 0.807 |

> **Note:** supervised models have access to labels at training time — direct F1 comparison
> with unsupervised/statistical models is not entirely fair.

### Temporal split (honest out-of-sample evaluation)

| Model | Category | F1 | AUC | Notes |
|---|---|---|---|---|
| LSTM-AE | Deep learning | 0.435 | **0.791** | Best AUC; +5w lag on COVID |
| AE Group-Decomposed | Deep learning | **0.507** | 0.773 | Best F1; Credit 14.9× sep. |
| HMM-Routed | Deep learning | 0.476 | 0.769 | Routing adds HMM signal |
| AE + EVT | Deep learning | 0.476 | 0.766 | Principled threshold |
| GLASSO + EVT + Filter | Statistical | 0.421 | 0.777 | Precision = **0.889** |
| Student-t + EVT + Filter | Statistical | 0.429 | 0.777 | Best statistical F1 |
| EE + EVT + Filter | Statistical | 0.390 | 0.728 | Best lead time (11w max) |

> **Key finding:** shuffled AUC for supervised models reaches 0.92 vs 0.77 for temporal
> deep learning — the gap is largely look-ahead bias, not model quality.

---

## Team

| Folder | Category |
|---|---|
| `statistical/` | MVG variants + quant finance models |
| `deep_learning/` | Deep learning pipeline (AE, LSTM, HMM, Hawkes) |
| `supervised/` | Supervised classifiers (RF, SVM, XGBoost) |
| `unsupervised/` | Classical unsupervised + PyOD |
