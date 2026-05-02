# `utils/run_nn.py`

## What it does

Trains a **weight-generating neural network** that maps a sliding window of past futures returns directly to portfolio weights.  The loss function is defined on tracking quality — not on return prediction — so the network learns an end-to-end weighting policy.

---

## Architecture

```
Input:  (batch, window, n_futures)         ← last W weeks of returns
         │
         ├── MLP mode:  flatten → Dense → ReLU → Dropout → ... → Dense(n_futures)
         │
         └── LSTM mode: LSTM(hidden) → last hidden state → Linear(n_futures)
         │
Output: (batch, n_futures)                 ← raw, unconstrained weights
```

Weights are unconstrained (no softmax), so the model can go long and short.  Gross exposure is controlled via the VaR scaling layer applied **outside** the network.

---

## Loss function

```
L = MSE(replica_return, target_return) + λ × mean(|weights|)
```

- **MSE term**: minimises tracking error directly
- **L1 term** (coefficient `l1_penalty`): penalises gross exposure, encourages sparsity — analogous to LASSO regularisation

---

## Data split

| Segment | Default fraction | Purpose |
|---------|-----------------|---------|
| Train | 60 % | Model parameters |
| Validation | 15 % | Early stopping (no look-ahead) |
| Test | 25 % | Reported out-of-sample performance |

The test set is the only period that appears in the comparison plots.

---

## VaR scaling (post-inference)

After generating weights at each time step, the module checks:

```
VaR(1M, 99%) = -(μ × 4 + σ × √4 × Φ⁻¹(0.01))
```

If VaR > `max_var_threshold` (default 0.20 = 20 %), all weights are scaled down uniformly:

```
scale = max_var_threshold / VaR
weights_scaled = weights × scale
```

This mirrors the UCITS regulatory constraint and makes the NN directly comparable to the regularisation-based models.

---

## Default configurations

Four configs are tried by default:

| Label | Mode | Window | Hidden dims | L1 penalty |
|-------|------|--------|-------------|------------|
| MLP_w26_h64x32_l10.0 | MLP | 26 wks | [64, 32] | 0 |
| MLP_w52_h64x32_l10.001 | MLP | 52 wks | [64, 32] | 0.001 |
| MLP_w26_h128x64x32_l10.001 | MLP | 26 wks | [128, 64, 32] | 0.001 |
| LSTM_w52_h64_l10.0 | LSTM | 52 wks | [64] | 0 |

The best config is selected by **information ratio** on the test set.

---

## Plots generated

| File | Content |
|------|---------|
| `nn_cfg{N}_training_curves.png` | Train / val loss per epoch |
| `nn_cfg{N}_weights.png` | Top 8 portfolio weights over time |
| `nn_cfg{N}_gross_var.png` | Gross exposure + VaR with limit line |

---

## How to call

```python
from utils.run_nn import run_nn, DEFAULT_CONFIGS

nn_output = run_nn(
    X=data["X"],
    y=data["y"],
    configs=DEFAULT_CONFIGS,
    train_frac=0.60,
    val_frac=0.15,
    max_var_threshold=0.20,
)

best = nn_output["best_result"]
# best["replica_returns"] → pd.Series
# best["target_returns"]  → pd.Series
# best["weights_history"] → pd.DataFrame (dates × futures)
```

---

## Output pickle

`data/picklefiles/nn_results.pkl` contains:

```python
{
    "best_result":   dict,           # result dict for best config
    "all_results":   list[dict],     # one per config
    "metrics_df":    pd.DataFrame,   # metrics summary
    "best_config":   dict,           # hyperparameters of best config
    "feature_names": list[str],      # futures column names
}
```

---

## Requirements

```bash
pip install torch
```

PyTorch is not installed as part of the base requirements — the module raises an `ImportError` with installation instructions if it's missing.

---

## Extending to custom configs

```python
my_configs = [
    dict(mode="mlp", window=104, hidden_dims=[256, 128, 64],
         dropout=0.3, lr=5e-4, epochs=500,
         batch_size=16, l1_penalty=5e-4, patience=50),
]

nn_output = run_nn(data["X"], data["y"], configs=my_configs)
```
