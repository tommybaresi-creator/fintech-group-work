# Deep Learning Early Warning Systems for Financial Market Stress

**Business Case 4 | Politecnico di Milano | Fintech Course**

---

## Overview

This project develops a deep learning pipeline for detecting financial market stress
(risk-off episodes) using a weekly cross-asset dataset of 42 Bloomberg indicators
spanning January 2000 to April 2021.

The pipeline is built around a central theoretical observation: the label "risk-off"
conflates three structurally different crisis types — exogenous shocks (Black Swans),
endogenous build-ups (Dragon Kings), and the boom/bust asymmetry problem. Each type
has a different information structure in the time series, and each requires a different
detection architecture. A single model optimised for one type will underperform on the others.

---

## Pipeline Architecture

```
Raw weekly returns (42 features)
           │
    ┌──────┴──────┐
    │   Stage 1   │  HMM (concurrent regime detection)
    │             │  Hawkes branching ratio n (endogeneity fraction)
    └──────┬──────┘
           │ routing
    ┌──────┴──────┐        ┌─────────────────────────┐
    │  Stage 2a   │        │       Stage 2b           │
    │  AE + group │ ←→     │  LSTM-AE (temporal)      │
    │  decomp.    │        │  Sequential detection     │
    └──────┬──────┘        └───────────┬─────────────┘
           └──────────┬────────────────┘
                      │
               ┌──────┴──────┐
               │   Stage 3   │  Directional filter (domain-knowledge vector)
               │             │  Pass only risk-off direction anomalies
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │   Stage 4   │  EVT/POT threshold (regime-conditional GPD)
               │             │  Threshold at stated false-alarm rate p
               └──────┬──────┘
                      │
               ┌──────┴──────┐
               │  Lead Time  │  Detection advance over crisis onset
               └─────────────┘
```

---

## Models and Results

| Model | AUC | F1 | Notes |
|---|---|---|---|
| LSTM-AE (temporal) | **0.791** | 0.435 | Best AUC; +5w lag on COVID |
| AE Group-Decomposed | 0.773 | **0.507** | Best F1; Credit 14.9× separability |
| HMM-Routed (AE+LSTM) | 0.769 | 0.476 | Between components — routing imprecise |
| AE k=12 | 0.766 | 0.507 | Scalar baseline |
| AE + EVT | 0.766 | 0.476 | Principled threshold calibration |
| AE + Directional Filter | 0.766 | 0.449 | Precision ↑, Recall ↓ |

All models use the temporal (chronological) split. No shuffling.
Test set: 2018–2021 (138 normal, 29 anomaly). Primary test crisis: COVID-19.

---

## Key Findings

**The LSTM-AE gap quantifies shuffling damage.** Under the shuffled protocol the
LSTM-AE achieves AUC = 0.568 (near-random); under the temporal protocol 0.791.
The 0.222 gap shows shuffling categorically destroys sequence model function —
not merely overstates performance.

**Group decomposition adds value (+0.007 AUC).** Credit bonds (HY, EM) show
14.9× separability on the test set (COVID-19 was a credit and liquidity crisis).
CV-based weights are Macro-dominated (EM sell-off dynamics); the improvement
persists despite this mismatch.

**Hawkes GFC pre-2008 signal.** The branching ratio $n$ rises and remains elevated
in 2007–08 before the acute phase — qualitative evidence of Dragon King endogeneity
detectable before the cross-sectional anomaly. Too noisy for operational use but
theoretically the most interesting result.

**COVID-19 lead time.** The AE fires at the ±26-week window boundary (likely a
late-2019 false positive). The LSTM-AE lags by +5 weeks — the predicted window
smoothing cost for sudden exogenous shocks, confirming the routing rationale.

---

## Project Structure

```
BusinessCase4/
├── BC4_P4_EarlyWarningSystem.ipynb    ← main notebook
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── Dataset4_EWS.xlsx              ← Bloomberg weekly data (not included)
└── outputs/
    ├── figures/                        ← all plots (PNG, 110 dpi)
    └── results_p4.csv                  ← model metrics table
```

---

## Setup

### Requirements
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)

### Installation

```bash
cd BusinessCase4
uv sync
```

### Running

```bash
# Interactive
uv run jupyter lab BC4_P4_EarlyWarningSystem.ipynb

# Headless (full execution)
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    BC4_P4_EarlyWarningSystem.ipynb
```

### Dataset

Place `Dataset4_EWS.xlsx` in `data/`. The notebook searches the following locations:

1. `data/Dataset4_EWS.xlsx` (recommended)
2. `Dataset4_EWS.xlsx`
3. `../Dataset4_EWS.xlsx`
4. `/mnt/user-data/uploads/Dataset4_EWS.xlsx`

---

## Theoretical Background

The pipeline is grounded in Sornette's (2009) Black Swan / Dragon King taxonomy
and Filimonov & Sornette's (2012) Hawkes branching ratio for endogeneity quantification:

- **Black Swan** (COVID-19): exogenous, unpredictable from prices → change-point detector (AE)
- **Dragon King** (GFC): endogenous build-up, detectable in sequential dynamics → LSTM-AE
- **Directional asymmetry**: booms and crashes are both anomalous cross-sectionally →
  directional gate using domain-knowledge risk-off vector

---

## Reproducibility

All random seeds fixed to 42. Full reproducibility guaranteed with `uv sync` + execution.

```python
RANDOM_SEED = 42
np.random.seed(42)
torch.manual_seed(42)
```

---

## Limitations

- Test set contains only COVID-19 (exogenous) — Dragon King routing cannot be validated
- CV has 10 anomaly weeks — threshold and weight estimates are noisy
- Hawkes implementation is a discrete-time approximation — daily data would improve precision
- AUC ceiling at 0.766–0.791 reflects the binary label conflating three crisis types
- Rolling refit (quarterly model updates) not implemented — production staleness unaddressed