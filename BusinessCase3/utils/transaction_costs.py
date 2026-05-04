"""
Transaction cost analysis for the Portfolio Replication project.

Computes portfolio turnover from a weight history, applies a per-trade
cost in basis points, and returns net-of-cost return series. Works with
any model that produces a ``weights_history`` DataFrame (NN, linear
models, Kalman).

Cost model
----------
At each rebalancing step the traded quantity is the absolute change in
each weight.  Total one-way cost for period t is:

    cost_t = cost_bps / 10_000 * sum_j |w_{t,j} - w_{t-1,j}|

This is a conservative first-order approximation: it ignores bid-ask
spread vs. mid-price distinction, market impact, and margin financing
costs, all of which are small for liquid Futures at the fund sizes
discussed in the course (>50M EUR, automated program trading).

Realistic Futures cost range: 2--5 bps one-way.  Default is 5 bps,
which is deliberately conservative.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PICKLE_DIR = Path("data/picklefiles")
OUTPUTS_DIR = Path("outputs")

# Realistic cost scenarios in basis points (one-way, per notional traded)
COST_SCENARIOS_BPS: List[float] = [0.0, 2.0, 5.0, 10.0]
DEFAULT_COST_BPS: float = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(stem: str) -> None:
    path = OUTPUTS_DIR / f"{stem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


# ── Core functions ────────────────────────────────────────────────────────────

def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute one-way portfolio turnover at each rebalancing step.

    Turnover at time t is defined as the sum of absolute weight changes:

        turnover_t = sum_j |w_{t,j} - w_{t-1,j}|

    The first period is NaN (no prior weights to compare against).

    Parameters
    ----------
    weights : pd.DataFrame
        Weight history, shape ``(T, n_assets)``.  Index should be
        a ``DatetimeIndex``; columns are asset names.

    Returns
    -------
    pd.Series
        One-way turnover at each date, first value NaN.
    """
    diffs = weights.diff().abs().sum(axis=1)
    diffs.iloc[0] = np.nan
    diffs.name = "turnover"
    return diffs


def apply_transaction_costs(
    replica_returns: pd.Series,
    weights: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.Series:
    """
    Subtract transaction costs from gross replica returns.

    At each period, the cost equals ``cost_bps / 10_000`` times the
    one-way turnover.  The first period incurs the cost of the initial
    position (turnover from zero).

    Parameters
    ----------
    replica_returns : pd.Series
        Gross weekly replica returns.
    weights : pd.DataFrame
        Weight history aligned to ``replica_returns.index``.
        Shape ``(T, n_assets)``.
    cost_bps : float
        One-way transaction cost in basis points.  Default 5 bps.

    Returns
    -------
    pd.Series
        Net-of-cost weekly replica returns.

    Raises
    ------
    ValueError
        If ``replica_returns`` and ``weights`` do not share the same
        index.
    """
    common = replica_returns.index.intersection(weights.index)
    if len(common) == 0:
        raise ValueError("replica_returns and weights have no overlapping dates.")

    r  = replica_returns.loc[common]
    w  = weights.loc[common]

    # First period: cost of putting on the initial position from zero
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)

    cost_per_period = turnover * (cost_bps / 10_000.0)
    net_returns     = r - cost_per_period
    net_returns.name = f"replica_net_{cost_bps:.0f}bps"

    logger.debug(
        "cost_bps=%.1f | mean_turnover=%.4f | mean_cost_per_period=%.6f",
        cost_bps, turnover.mean(), cost_per_period.mean(),
    )
    return net_returns


def compute_cost_metrics(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    weights: pd.DataFrame,
    cost_scenarios: Optional[List[float]] = None,
    freq: int = 52,
) -> pd.DataFrame:
    """
    Compare performance across multiple cost scenarios for one model.

    Parameters
    ----------
    replica_returns : pd.Series
        Gross replica weekly returns.
    target_returns : pd.Series
        Target index weekly returns.
    weights : pd.DataFrame
        Weight history.
    cost_scenarios : list of float, optional
        Cost levels in bps to evaluate.  Defaults to
        ``COST_SCENARIOS_BPS`` (0, 2, 5, 10 bps).
    freq : int
        Periods per year for annualisation.

    Returns
    -------
    pd.DataFrame
        One row per cost scenario with columns:
        ``cost_bps``, ``ann_ret``, ``ann_vol``, ``sharpe``,
        ``tracking_error``, ``information_ratio``, ``correlation``,
        ``ann_cost_drag``, ``mean_turnover``.
    """
    if cost_scenarios is None:
        cost_scenarios = COST_SCENARIOS_BPS

    common = replica_returns.index.intersection(target_returns.index)
    r_gross = replica_returns.loc[common]
    t       = target_returns.loc[common]

    w_prev    = weights.loc[common].shift(1).fillna(0.0)
    turnover  = (weights.loc[common] - w_prev).abs().sum(axis=1)
    mean_to   = float(turnover.mean())

    rows = []
    for bps in cost_scenarios:
        net = apply_transaction_costs(r_gross, weights, cost_bps=bps)
        net = net.loc[common]

        active = net - t
        te     = float(active.std(ddof=1) * np.sqrt(freq))
        ir     = (float(active.mean() * freq) / te) if te > 0 else float("nan")
        vol    = float(net.std(ddof=1) * np.sqrt(freq))
        ann_r  = float(net.mean() * freq)
        sharpe = (ann_r / vol) if vol > 0 else float("nan")
        corr   = float(net.corr(t))

        gross_r    = float(r_gross.mean() * freq)
        ann_drag   = gross_r - ann_r

        rows.append({
            "cost_bps":          bps,
            "ann_ret":           ann_r,
            "ann_vol":           vol,
            "sharpe":            sharpe,
            "tracking_error":    te,
            "information_ratio": ir,
            "correlation":       corr,
            "ann_cost_drag":     ann_drag,
            "mean_turnover":     mean_to,
        })

    return pd.DataFrame(rows)


def build_cost_comparison(
    results: Dict[str, Dict],
    cost_scenarios: Optional[List[float]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run cost analysis for every model in a results dictionary.

    Parameters
    ----------
    results : dict
        Keys are model names.  Each value must contain:

        ``replica_returns``
            pd.Series of gross weekly returns.
        ``target_returns``
            pd.Series of weekly target returns.
        ``weights_history``
            pd.DataFrame of portfolio weights.

    cost_scenarios : list of float, optional
        Cost levels in bps.  Defaults to ``COST_SCENARIOS_BPS``.

    Returns
    -------
    dict
        Keys are model names; values are DataFrames from
        :func:`compute_cost_metrics`.
    """
    out = {}
    for name, res in results.items():
        if "weights_history" not in res:
            logger.warning("Model '%s' has no weights_history — skipping.", name)
            continue
        logger.info("Computing cost scenarios for '%s' …", name)
        out[name] = compute_cost_metrics(
            res["replica_returns"],
            res["target_returns"],
            res["weights_history"],
            cost_scenarios=cost_scenarios,
        )
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_turnover(
    results: Dict[str, Dict],
    save_name: str = "tc_01_turnover",
) -> None:
    """
    Plot weekly turnover time series for all models.

    Parameters
    ----------
    results : dict
    save_name : str
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for (name, res), color in zip(results.items(), palette):
        if "weights_history" not in res:
            continue
        to = compute_turnover(res["weights_history"])
        ax.plot(to, label=name, color=color, linewidth=1.2, alpha=0.8)

    ax.set_title("Weekly Portfolio Turnover by Model", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("One-way turnover (sum |Δw|)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_cost_drag(
    cost_tables: Dict[str, pd.DataFrame],
    save_name: str = "tc_02_cost_drag",
) -> None:
    """
    Bar chart of annualised cost drag at each scenario for all models.

    Parameters
    ----------
    cost_tables : dict
        Output of :func:`build_cost_comparison`.
    save_name : str
    """
    scenarios = COST_SCENARIOS_BPS[1:]   # skip 0 bps
    n_models  = len(cost_tables)
    x         = np.arange(len(scenarios))
    width     = 0.8 / max(n_models, 1)
    palette   = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate(cost_tables.items()):
        drag = [
            float(df.loc[df["cost_bps"] == bps, "ann_cost_drag"].iloc[0]) * 100
            for bps in scenarios
        ]
        ax.bar(x + i * width, drag, width, label=name, color=palette[i], alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"{b:.0f} bps" for b in scenarios])
    ax.set_title("Annualised Cost Drag by Model and Scenario", fontsize=13)
    ax.set_ylabel("Cost drag (% p.a.)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_ir_vs_cost(
    cost_tables: Dict[str, pd.DataFrame],
    save_name: str = "tc_03_ir_vs_cost",
) -> None:
    """
    Line chart of Information Ratio vs cost level for all models.

    Parameters
    ----------
    cost_tables : dict
    save_name : str
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(cost_tables)))

    for (name, df), color in zip(cost_tables.items(), palette):
        ax.plot(
            df["cost_bps"], df["information_ratio"],
            marker="o", label=name, color=color, linewidth=1.8,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Information Ratio vs Transaction Cost Level", fontsize=13)
    ax.set_xlabel("Cost (bps one-way)")
    ax.set_ylabel("Information Ratio")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_net_cumulative(
    results: Dict[str, Dict],
    cost_bps: float = DEFAULT_COST_BPS,
    save_name: str = "tc_04_net_cumulative",
) -> None:
    """
    Overlay gross vs net-of-cost cumulative returns for best model.

    Shows all models at the given cost level against the target.

    Parameters
    ----------
    results : dict
    cost_bps : float
    save_name : str
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    target = next(iter(results.values()))["target_returns"]
    ax.plot(
        (1 + target).cumprod(), color="black", linewidth=2.5,
        linestyle="--", label="Target",
    )

    for (name, res), color in zip(results.items(), palette):
        if "weights_history" not in res:
            continue
        gross = res["replica_returns"]
        net   = apply_transaction_costs(
            gross, res["weights_history"], cost_bps=cost_bps
        )
        ax.plot(
            (1 + gross).cumprod(), color=color,
            linewidth=1.2, linestyle=":", alpha=0.5,
        )
        ax.plot(
            (1 + net).cumprod(), color=color,
            linewidth=1.8, label=f"{name} net ({cost_bps:.0f}bps)",
        )

    ax.set_title(
        f"Cumulative Returns — Gross (dotted) vs Net of {cost_bps:.0f}bps Cost",
        fontsize=13,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (start = 1)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


def plot_summary_table_heatmap(
    cost_tables: Dict[str, pd.DataFrame],
    metric: str = "information_ratio",
    save_name: str = "tc_05_summary_heatmap",
) -> None:
    """
    Heatmap of a chosen metric across models (rows) and cost scenarios (cols).

    Parameters
    ----------
    cost_tables : dict
    metric : str
        Column name from :func:`compute_cost_metrics` output.
    save_name : str
    """
    import seaborn as sns

    pivot = pd.DataFrame({
        name: df.set_index("cost_bps")[metric]
        for name, df in cost_tables.items()
    }).T
    pivot.columns = [f"{c:.0f} bps" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.7 + 2)))
    sns.heatmap(
        pivot.round(4), annot=True, fmt="g",
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_title(f"{metric} across models and cost scenarios", fontsize=13)
    plt.tight_layout()
    _save_fig(save_name)
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_transaction_cost_analysis(
    results: Dict[str, Dict],
    cost_scenarios: Optional[List[float]] = None,
    save_prefix: str = "tc",
) -> Dict[str, pd.DataFrame]:
    """
    Full transaction cost analysis pipeline.

    Generates cost tables and five diagnostic plots for all models.
    Saves ``data/picklefiles/transaction_costs.pkl``.

    Parameters
    ----------
    results : dict
        Keys are model names.  Each value must contain
        ``replica_returns``, ``target_returns``, ``weights_history``.
    cost_scenarios : list of float, optional
        One-way cost levels in bps to evaluate.
        Defaults to [0, 2, 5, 10].
    save_prefix : str
        Filename prefix for all saved figures.

    Returns
    -------
    dict
        Keys are model names; values are cost metric DataFrames.
    """
    _ensure_dirs()
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    logger.info("=" * 60)
    logger.info("TRANSACTION COST ANALYSIS — START  (%d models)", len(results))
    logger.info("=" * 60)

    if cost_scenarios is None:
        cost_scenarios = COST_SCENARIOS_BPS

    cost_tables = build_cost_comparison(results, cost_scenarios)

    # ── Log summary at default cost ──────────────────────────────────────────
    logger.info("Summary at %.0f bps:", DEFAULT_COST_BPS)
    for name, df in cost_tables.items():
        row = df.loc[df["cost_bps"] == DEFAULT_COST_BPS].iloc[0]
        logger.info(
            "  %-30s | net IR=%.4f | drag=%.4f%% | mean TO=%.4f",
            name,
            row["information_ratio"],
            row["ann_cost_drag"] * 100,
            row["mean_turnover"],
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_turnover(results,                  save_name=f"{save_prefix}_01_turnover")
    plot_cost_drag(cost_tables,             save_name=f"{save_prefix}_02_cost_drag")
    plot_ir_vs_cost(cost_tables,            save_name=f"{save_prefix}_03_ir_vs_cost")
    plot_net_cumulative(results,            save_name=f"{save_prefix}_04_net_cumulative")
    plot_summary_table_heatmap(cost_tables, save_name=f"{save_prefix}_05_summary_heatmap")

    # ── Pickle ───────────────────────────────────────────────────────────────
    pkl_path = PICKLE_DIR / "transaction_costs.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"cost_tables": cost_tables, "cost_scenarios": cost_scenarios}, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("TRANSACTION COST ANALYSIS — DONE")
    logger.info("=" * 60)
    return cost_tables


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import setup_logging
    setup_logging()

    pkl = Path("data/picklefiles/nn_results.pkl")
    if not pkl.exists():
        logger.error("nn_results.pkl not found — run run_nn() first.")
        sys.exit(1)

    with open(pkl, "rb") as fh:
        nn = pickle.load(fh)

    results = {"NN_best": nn["best_result"]}
    run_transaction_cost_analysis(results)