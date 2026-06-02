# Business Case 3 - Portfolio Replication

Weekly-frequency portfolio replication of a **Monster Index** (50% HFRX Global Hedge Fund, 25% MSCI World, 25% Bloomberg Global Agg Bond) using 11 futures as replicating instruments.
Five linear estimators, a Kalman filter, and neural networks (MLP + LSTM) are compared on a shared out-of-sample test period with UCITS risk constraints and transaction-cost analysis.

> **To explore the raw dataset, run `results/eda.ipynb`. To visualise all model results, run `results/compare_all.ipynb` in order.**
> Every section calls a single function from `utils/`; figures and pickles are saved automatically to `results/outputs/` and `results/data/picklefiles/`.

---

## Structure

```
BusinessCase3/
|-- data/                          raw Excel data file
|-- results/
|   |-- eda.ipynb                  exploratory data analysis notebook
|   |-- compare_all.ipynb          master notebook - run this for modeling
|   |-- outputs/                   all generated figures (PNG)
|   `-- data/picklefiles/          cached results (pkl)
`-- utils/
    |-- __init__.py                logging setup
    |-- data_loader.py             price ingestion, return computation, monster index
    |-- models.py                  sklearn model factories + hyperparameter selectors
    |-- rolling_engine.py          shared walk-forward loop with VaR scaling
    |-- run_linear_models.py       OLS / Ridge / LASSO / ElasticNet / WOLS pipeline
    |-- run_kalman.py              Kalman filter (random-walk state-space weights)
    |-- run_nn.py                  MLP + LSTM weight-generators (PyTorch)
    |-- evaluation.py              metrics, comparison plots, regime analysis
    |-- transaction_costs.py       turnover, cost drag, multi-scenario analysis
    `-- risk.py                    UCITS VaR / leverage helpers
```

## Dependencies

```
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch openpyxl
```

## Notebook execution order

Start with `results/eda.ipynb` for standalone exploratory analysis of prices, returns, Monster Index construction, correlations, rolling relationships, and the pre/post-COVID regime context. It reuses `data_loader.pkl` when available and falls back to the raw Excel file if the cache is missing or incompatible. EDA figures are saved to `results/outputs/` with the `eda_` prefix.

Then run `results/compare_all.ipynb` for the full modeling workflow:

| Section | Function called | Output pickles |
|---------|----------------|----------------|
| 1 - Data Loader | `run_data_loader()` | `data_loader.pkl` |
| 2 - Linear Models | `run_linear_models()` | `linear_results.pkl` |
| 3 - Kalman Filter | `run_kalman()` | `kalman_results.pkl` |
| 4 - Neural Network | `run_nn()` | `nn_results.pkl` |
| 5 - Full Comparison | `run_evaluation()` | `evaluation.pkl` |
| 6 - Transaction Costs | `run_transaction_cost_analysis()` + `run_evaluation_with_costs()` | `transaction_costs.pkl`, `evaluation_with_costs.pkl` |
| 7 - Regime Analysis | `run_regime_analysis()` | `regime_analysis.pkl` |

Each section loads pre-computed pickles if they exist, so individual steps can be re-run independently.

## Key design decisions

- **No look-ahead**: scalers and hyperparameters are fit on training data only; the rolling loop predicts `y_t` using weights from the window `[t-W, t)`.
- **VaR constraint**: UCITS 1-month 99% VaR <= 20%; enforced at each step using the conservative max(backward, forward-looking) Cornish-Fisher + historical simulation estimate.
- **Cost model**: one-way turnover at user-specified bps; Kalman dominates (turnover approx. 0.03x/week) while OLS/WOLS are the most cost-sensitive (approx. 1.19x/week).
- **Regime split**: pre-COVID (up to 2020-02-21) vs post-COVID (from 2020-02-24); all models evaluated on aligned test windows for comparability.
