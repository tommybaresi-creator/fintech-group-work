"""
Data loading and preprocessing pipeline for the Portfolio Replication project.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Directory constants ───────────────────────────────────────────────────────
PICKLE_DIR = Path("data/picklefiles")
OUTPUTS_DIR = Path("outputs")

# ── Bloomberg column naming ───────────────────────────────────────────────────
INDEX_TICKERS = ["MXWO", "MXWD", "LEGATRUU", "HFRXGL"]
FUTURES_TICKERS = [
    "RX1",
    "TY1",
    "GC1",
    "CO1",
    "ES1",
    "VG1",
    "NQ1",
    "LLL1",
    "TP1",
    "DU1",
    "TU2",
]

FUTURES_NAMES: Dict[str, str] = {
    "RX1 Comdty": "Bund 10Y (DE)",
    "TY1 Comdty": "US Treasury 10Y",
    "GC1 Comdty": "Gold",
    "CO1 Comdty": "Brent Oil",
    "ES1 Comdty": "S&P 500",
    "VG1 Comdty": "EuroStoxx 50",
    "NQ1 Comdty": "Nasdaq 100",
    "LLL1 Comdty": "MSCI EM",
    "TP1 Comdty": "TOPIX (JP)",
    "DU1 Comdty": "Schatz 2Y (DE)",
    "TU2 Comdty": "US Treasury 2Y",
}

# Default monster index composition
DEFAULT_MONSTER_WEIGHTS: Dict[str, float] = {
    "HFRXGL Index": 0.50,
    "MXWO Index": 0.25,
    "LEGATRUU Index": 0.25,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    """Create PICKLE_DIR and OUTPUTS_DIR if they do not exist."""
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(stem: str) -> None:
    """Save the current matplotlib figure to OUTPUTS_DIR/<stem>.png."""
    path = OUTPUTS_DIR / f"{stem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


# ── Core functions ────────────────────────────────────────────────────────────


def load_raw_data(
    filepath: str | Path,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    """
    Load and rename the Bloomberg weekly price data from an Excel file.

    The first column is treated as the date index. Columns in
    ``INDEX_TICKERS`` receive an ``' Index'`` suffix; columns in
    ``FUTURES_TICKERS`` receive a ``' Comdty'`` suffix.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.xlsx`` file.
    sheet_name : int or str, optional
        Sheet to read (default: first sheet).

    Returns
    -------
    pd.DataFrame
        Price levels with a ``DatetimeIndex``, sorted ascending.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``filepath``.
    ValueError
        If the file cannot be parsed or expected columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Reading Excel file: %s (sheet=%s)", filepath, sheet_name)
    try:
        raw = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as exc:
        raise ValueError(f"Could not read {filepath}: {exc}") from exc

    # First column → Date index
    raw = raw.rename(columns={raw.columns[0]: "Date"})

    # Rename tickers with Bloomberg suffixes
    new_cols = ["Date"]
    for col in raw.columns[1:]:
        if col in INDEX_TICKERS:
            new_cols.append(f"{col} Index")
        elif col in FUTURES_TICKERS:
            new_cols.append(f"{col} Comdty")
        else:
            logger.warning("Unrecognised column '%s' — kept as-is", col)
            new_cols.append(col)
    raw.columns = new_cols

    # Parse dates (handles both Excel serial numbers and ISO strings)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()

    # Validate presence of all expected columns
    expected = [f"{t} Index" for t in INDEX_TICKERS] + [
        f"{t} Comdty" for t in FUTURES_TICKERS
    ]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}")

    logger.info(
        "Loaded %d observations | %s → %s",
        len(raw),
        raw.index[0].date(),
        raw.index[-1].date(),
    )
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly percentage returns from price levels.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels with a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Percentage returns; the first row (NaN) is dropped.
    """
    logger.info("Computing weekly returns from price levels")
    rets = prices.pct_change().dropna()
    logger.info("Returns shape: %s", rets.shape)
    return rets


def build_monster_index(
    returns: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Build the synthetic monster index as a weighted sum of index returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Weekly return series; must contain all columns in ``weights``.
    weights : dict, optional
        ``{column_name: weight}`` mapping. Defaults to
        ``DEFAULT_MONSTER_WEIGHTS`` (50 % HFRX, 25 % MSCI World,
        25 % Global Agg Bond).

    Returns
    -------
    pd.Series
        Weekly returns of the monster index, named ``'MonsterIndex'``.

    Raises
    ------
    ValueError
        If any weight column is absent from ``returns``.
    """
    if weights is None:
        weights = DEFAULT_MONSTER_WEIGHTS

    missing = [c for c in weights if c not in returns.columns]
    if missing:
        raise ValueError(f"Weight columns not found in returns: {missing}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning("Weights sum to %.4f — normalising to 1.0", total)
        weights = {k: v / total for k, v in weights.items()}

    monster = sum(returns[col] * w for col, w in weights.items())
    monster.name = "MonsterIndex"

    logger.info(
        "Monster index composition: %s",
        " | ".join(f"{c}={w:.0%}" for c, w in weights.items()),
    )
    return monster


def get_X_y(
    returns: pd.DataFrame,
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract and align the futures feature matrix and target return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Full return DataFrame containing both index and futures columns.
    target : pd.Series
        Monster index weekly returns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X`` — futures returns only (columns ending in ``' Comdty'``).
        ``y`` — target returns, aligned to ``X``'s index.
    """
    futures_cols = [c for c in returns.columns if c.endswith("Comdty")]
    X = returns[futures_cols]
    common_idx = X.index.intersection(target.index)
    X, y = X.loc[common_idx], target.loc[common_idx]
    logger.info("X shape: %s | y shape: %s", X.shape, y.shape)
    return X, y


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot_price_series(prices: pd.DataFrame) -> None:
    """Normalised (base 100) price series for the four target indices."""
    index_cols = [c for c in prices.columns if c.endswith("Index")]
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in index_cols:
        ax.plot(prices[col] / prices[col].iloc[0] * 100, label=col, linewidth=1.8)
    ax.set_title("Target Indices — Normalised Price Series (Base 100)", fontsize=14)
    ax.set_ylabel("Level")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig("01_price_series")
    plt.close(fig)


def _plot_futures_series(prices: pd.DataFrame) -> None:
    """Normalised (base 100) price series for all futures contracts."""
    futures_cols = [c for c in prices.columns if c.endswith("Comdty")]
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in futures_cols:
        ax.plot(
            prices[col] / prices[col].iloc[0] * 100,
            label=FUTURES_NAMES.get(col, col),
            linewidth=1.2,
        )
    ax.set_title("Futures Contracts — Normalised Price Series (Base 100)", fontsize=14)
    ax.set_ylabel("Level")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig("02_futures_series")
    plt.close(fig)


def _plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    """Correlation heatmap for all return series with readable labels."""
    short = {
        c: FUTURES_NAMES.get(c, c.replace(" Index", "").replace(" Comdty", ""))
        for c in returns.columns
    }
    corr = returns.corr().rename(index=short, columns=short)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Return Correlation Matrix (weekly)", fontsize=14)
    plt.tight_layout()
    _save_fig("03_correlation_heatmap")
    plt.close(fig)


def _plot_monster_index(monster: pd.Series, returns: pd.DataFrame) -> None:
    """Cumulative returns and return distribution of the monster index."""
    component_cols = list(DEFAULT_MONSTER_WEIGHTS.keys())
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Cumulative returns ---
    ax = axes[0]
    ax.plot(
        (1 + monster).cumprod(),
        label="Monster Index",
        color="black",
        linewidth=2.5,
    )
    for col in component_cols:
        ax.plot(
            (1 + returns[col]).cumprod(),
            label=col,
            linewidth=1.2,
            alpha=0.7,
        )
    ax.set_title("Monster Index vs Components — Cumulative Returns", fontsize=14)
    ax.set_ylabel("Cumulative Return (start = 1)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Return distribution ---
    ax = axes[1]
    ax.hist(monster, bins=60, alpha=0.75, color="steelblue", edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(
        x=monster.mean(),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean={monster.mean():.4f}",
    )
    ax.set_title("Monster Index — Weekly Return Distribution", fontsize=14)
    ax.set_xlabel("Weekly Return")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_fig("04_monster_index")
    plt.close(fig)


def _plot_return_stats(returns: pd.DataFrame, monster: pd.Series) -> None:
    """Bar chart of annualised return and volatility for all series."""
    freq = 52
    all_rets = returns.copy()
    all_rets["MonsterIndex"] = monster
    all_rets = all_rets.dropna()

    ann_ret = all_rets.mean() * freq * 100
    ann_vol = all_rets.std() * np.sqrt(freq) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    short_labels = [
        FUTURES_NAMES.get(c, c.replace(" Index", "").replace(" Comdty", ""))
        for c in ann_ret.index
    ]

    for ax, data, title, color in zip(
        axes,
        [ann_ret, ann_vol],
        ["Annualised Return (%)", "Annualised Volatility (%)"],
        ["steelblue", "coral"],
    ):
        ax.barh(short_labels, data, color=color, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Return Statistics — All Series (Weekly Data)", fontsize=14)
    plt.tight_layout()
    _save_fig("05_return_stats")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────


def run_data_loader(
    filepath: str | Path = "data/Dataset3_PortfolioReplicaStrategy.xlsx",
    monster_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Full data pipeline: load → clean → returns → monster index → plots → pickle.

    Saves ``data/picklefiles/data_loader.pkl`` and five ``.png`` figures
    to ``outputs/``.

    Parameters
    ----------
    filepath : str or Path
        Path to the Bloomberg Excel dataset.
    monster_weights : dict, optional
        Custom composition for the monster index.
        Defaults to ``DEFAULT_MONSTER_WEIGHTS``.

    Returns
    -------
    dict
        Keys:

        ``prices``
            Raw price levels (pd.DataFrame).
        ``returns``
            Weekly percentage returns (pd.DataFrame).
        ``monster``
            Monster index weekly returns (pd.Series).
        ``X``
            Futures returns — feature matrix (pd.DataFrame).
        ``y``
            Monster index returns — target (pd.Series).

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found at ``filepath``.
    ValueError
        If data validation fails.
    """
    _ensure_dirs()
    logger.info("=" * 60)
    logger.info("DATA LOADER — START")
    logger.info("=" * 60)

    # 1. Load
    prices = load_raw_data(filepath)

    # 2. Returns
    returns = compute_returns(prices)

    # 3. Monster index
    monster = build_monster_index(returns, weights=monster_weights)

    # 4. Features / target
    X, y = get_X_y(returns, monster)

    # 5. Diagnostic plots
    logger.info("Generating diagnostic plots …")
    sns.set_theme(style="whitegrid")
    _plot_price_series(prices)
    _plot_futures_series(prices)
    _plot_correlation_heatmap(returns)
    _plot_monster_index(monster, returns)
    _plot_return_stats(returns, monster)

    # 6. Save pickle
    payload = {
        "prices": prices,
        "returns": returns,
        "monster": monster,
        "X": X,
        "y": y,
    }
    pkl_path = PICKLE_DIR / "data_loader.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Pickle saved → %s", pkl_path)

    logger.info("DATA LOADER — DONE")
    logger.info("=" * 60)
    return payload


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from utils import setup_logging

    setup_logging()
    fp = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/Dataset3_PortfolioReplicaStrategy.xlsx"
    )
    run_data_loader(filepath=fp)
