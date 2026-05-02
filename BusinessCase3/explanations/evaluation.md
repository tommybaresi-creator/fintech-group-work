# `utils/evaluation.py`

## What it does

Computes all performance and replication-quality metrics for any set of model results, and generates six standardised comparison plots.  Designed to accept results from **any** model in the project (linear, Kalman, NN) through a common interface.

---

## Metrics computed

| Metric | Formula | Notes |
|--------|---------|-------|
| Annualised Return | `mean(r) × 52` | Weekly → annual |
| Annualised Volatility | `std(r) × √52` | |
| Sharpe Ratio | `ann_ret / ann_vol` | Risk-free = 0 |
| Tracking Error (TEV) | `std(r_replica − r_target) × √52` | Key replication metric |
| Information Ratio | `ann_active_return / TEV` | Primary ranking criterion |
| Correlation | Pearson(replica, target) | |
| Max Drawdown | `max(1 − cum / cummax)` | Positive fraction |

---

## Expected input format

Every model's result dict must contain at minimum:

```python
{
    "replica_returns": pd.Series,   # weekly returns, DatetimeIndex
    "target_returns":  pd.Series,   # same index as replica
}
```

Pass a dictionary of these dicts to `run_evaluation()`:

```python
results = {
    "OLS":     ols_result,
    "Ridge":   ridge_result,
    "LASSO":   lasso_result,
    "Kalman":  kalman_result,
    "NN_best": nn_output["best_result"],
}
```

---

## Plots generated

| File | Content |
|------|---------|
| `{prefix}_01_cumulative_returns.png` | Overlay of all models vs target |
| `{prefix}_02_tracking_metrics.png` | Tracking Error + Information Ratio bars |
| `{prefix}_03_drawdowns.png` | Drawdown series for all models |
| `{prefix}_04_scatter_returns.png` | Weekly return scatter plots |
| `{prefix}_05_rolling_correlation.png` | 26-week rolling correlation |
| `{prefix}_06_metrics_heatmap.png` | Heatmap with z-score normalisation |

---

## How to call

```python
from utils.evaluation import run_evaluation

metrics_df = run_evaluation(results, save_prefix="final")
```

Returns a `pd.DataFrame` with one row per model.

---

## VaR utility

`compute_var()` is also available standalone:

```python
from utils.evaluation import compute_var
import numpy as np

weekly_returns = np.random.normal(0.001, 0.02, 52)
var = compute_var(weekly_returns, confidence=0.99, horizon_weeks=4, method="parametric")
print(f"1M VaR(99%): {var:.2%}")
```

Supports `method='parametric'` (normal distribution) and `method='historical'`.

---

## Output pickle

`data/picklefiles/evaluation.pkl` contains:

```python
{
    "metrics": pd.DataFrame,   # metrics table
    "results": dict,           # full results dict (replica + target series)
}
```
