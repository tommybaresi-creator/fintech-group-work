# `utils/data_loader.py`

## What it does

Loads the raw Bloomberg weekly price data from the Excel file, computes returns, builds the synthetic **monster index**, and outputs the aligned feature matrix `X` and target vector `y` that every downstream model consumes.

---

## Pipeline (in order)

| Step | Function | Output |
|------|----------|--------|
| 1 | `load_raw_data()` | Price DataFrame with Bloomberg column names |
| 2 | `compute_returns()` | Weekly % returns (first row dropped) |
| 3 | `build_monster_index()` | Synthetic target: 50 % HFRX + 25 % MSCI World + 25 % Global Agg |
| 4 | `get_X_y()` | `X` (11 futures) and `y` (monster index), index-aligned |
| 5 | 5 diagnostic plots | Saved to `outputs/` |
| 6 | Pickle dump | `data/picklefiles/data_loader.pkl` |

---

## Monster index composition

```
HFRXGL Index    50%   ← global hedge fund benchmark
MXWO Index      25%   ← MSCI World (developed equities)
LEGATRUU Index  25%   ← Bloomberg Global Aggregate Bond
```

This is deliberately non-investable: it spans tens of thousands of securities across asset classes and currencies.  The replication task is to clone it using 11 liquid futures.

---

## Output pickle keys

| Key | Type | Description |
|-----|------|-------------|
| `prices` | `pd.DataFrame` | Raw price levels, DatetimeIndex |
| `returns` | `pd.DataFrame` | Weekly % returns, all series |
| `monster` | `pd.Series` | Monster index weekly returns |
| `X` | `pd.DataFrame` | Futures returns — feature matrix |
| `y` | `pd.Series` | Monster index returns — target |

---

## Plots generated

| File | Content |
|------|---------|
| `01_price_series.png` | Normalised prices for 4 target indices |
| `02_futures_series.png` | Normalised prices for 11 futures |
| `03_correlation_heatmap.png` | Full return correlation matrix |
| `04_monster_index.png` | Cumulative returns + return distribution |
| `05_return_stats.png` | Annualised return and vol bar charts |

---

## How to call

```python
from utils.data_loader import run_data_loader

data = run_data_loader(
    filepath="data/Dataset3_PortfolioReplicaStrategy.xlsx",
    monster_weights=None,   # None → use default 50/25/25
)

X, y = data["X"], data["y"]
```

Or from the command line:

```bash
python utils/data_loader.py data/Dataset3_PortfolioReplicaStrategy.xlsx
```

---

## Customising the monster index

```python
data = run_data_loader(
    filepath="...",
    monster_weights={
        "HFRXGL Index":   0.40,
        "MXWD Index":     0.40,   # MSCI ACWI instead of MSCI World
        "LEGATRUU Index": 0.20,
    }
)
```

Weights are automatically normalised if they don't sum to 1.
