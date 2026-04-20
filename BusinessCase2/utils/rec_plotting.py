"""
Plotting utilities for the investment product recommendation pipeline.

Functions
---------
plot_client_distribution
    Three-panel figure: client need segments, propensity distributions,
    and coverage breakdown (served / no product / ineligible).

plot_product_distribution
    Three-panel figure: recommendation frequency per product (SVD vs AE),
    SRI profile with weighted-average overlays, and SVD↔AE agreement rate.

plot_risk_suitability
    Two-panel scatter: client RiskPropensity vs recommended SRI for each
    model, with the MiFID II diagonal (SRI = RiskPropensity) overlaid.

plot_matrix_heatmap
    Side-by-side heatmap of the observed sparse R and the SVD reconstruction
    R̂ for the first N clients — shows zero-inflation bias directly.

plot_roc_curves
    ROC curves for SVD and Autoencoder evaluated on all (client, product)
    pairs, treating R=1 as positive — visualises the AUC gap.

plot_recommendation_frequency
    Two-panel figure: recommendation count per product (SVD vs AE grouped
    bars) and cumulative share (Lorenz-style) to show SVD clustering vs AE
    spread.
"""

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc

_BLUE   = '#4C72B0'
_ORANGE = '#DD8452'
_GREEN  = '#55A868'
_RED    = '#C44E52'
_GRAY   = '#8C8C8C'


# ---------------------------------------------------------------------------
# Figure 1 — Client distribution
# ---------------------------------------------------------------------------


def plot_client_distribution(
    clients_df: pd.DataFrame,
    recs_svd: pd.DataFrame,
    recs_ae: pd.DataFrame,
    n_eligible: int,
    n_total: int,
) -> None:
    """
    Plot a three-panel client distribution summary.

    Panel 1: Pie chart of need segments (income-only, accum-only, both, none).
    Panel 2: Overlapping histograms of p_hat_income and p_hat_accum.
    Panel 3: Stacked bar — served / no compliant product / ineligible.

    Parameters
    ----------
    clients_df : pd.DataFrame
        Must contain: need_income, need_accum, p_hat_income, p_hat_accum.
    recs_svd, recs_ae : pd.DataFrame
        Recommendation outputs from recommend_svd() / recommend_ae().
    n_eligible : int
        Number of clients with at least one confirmed need.
    n_total : int
        Total clients in the dataset.
    """
    n_income_only = (clients_df['need_income'] & ~clients_df['need_accum']).sum()
    n_accum_only  = (~clients_df['need_income'] & clients_df['need_accum']).sum()
    n_both        = (clients_df['need_income'] & clients_df['need_accum']).sum()
    n_none        = (~clients_df['need_income'] & ~clients_df['need_accum']).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Client Distribution', fontsize=14, fontweight='bold', y=1.01)

    # Panel 1 — need segment pie
    ax = axes[0]
    sizes  = [n_income_only, n_accum_only, n_both, n_none]
    labels = [
        f'Income only\n({n_income_only:,})',
        f'Accumulation only\n({n_accum_only:,})',
        f'Both needs\n({n_both:,})',
        f'No need\n({n_none:,})',
    ]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        colors=[_BLUE, _ORANGE, _GREEN, _GRAY],
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title('Client need segments (N=5,000)', fontweight='bold')

    # Panel 2 — propensity histograms
    ax = axes[1]
    ax.hist(clients_df['p_hat_income'], bins=40, alpha=0.6, color=_BLUE,
            label='p̂ Income', density=True)
    ax.hist(clients_df['p_hat_accum'],  bins=40, alpha=0.6, color=_ORANGE,
            label='p̂ Accumulation', density=True)
    ax.set_xlabel('Propensity score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity score distributions', fontweight='bold')
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Panel 3 — coverage stacked bar
    ax = axes[2]
    served_svd   = recs_svd[recs_svd['status'] == 'ok']['client_idx'].nunique()
    served_ae    = recs_ae[recs_ae['status']  == 'ok']['client_idx'].nunique()
    n_no_prod    = (
        recs_svd[recs_svd['status'] == 'no_compliant_product']['client_idx'].nunique()
    )
    not_eligible = n_total - n_eligible

    categories     = ['SVD', 'Autoencoder']
    vals_served    = [served_svd, served_ae]
    vals_no_prod   = [n_no_prod, n_no_prod]
    vals_ineligible = [not_eligible, not_eligible]

    x = np.arange(2)
    w = 0.5
    b1 = ax.bar(x, vals_served,     w, label='Served',               color=_GREEN)
    b2 = ax.bar(x, vals_no_prod,    w, bottom=vals_served,            label='No compliant product', color=_RED)
    bot2 = [v + n for v, n in zip(vals_served, vals_no_prod)]
    ax.bar(x, vals_ineligible, w, bottom=bot2, label='No need flagged', color=_GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Number of clients')
    ax.set_title('Client coverage breakdown', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.legend(loc='upper right', fontsize=8)

    for bar_group in [b1, b2]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 50:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + h / 2,
                    f'{int(h):,}',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold',
                )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Product distribution
# ---------------------------------------------------------------------------


def plot_product_distribution(
    products_df: pd.DataFrame,
    recs_svd: pd.DataFrame,
    recs_ae: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """
    Plot a three-panel product distribution summary.

    Panel 1: Grouped bar — recommendation frequency per product (SVD vs AE).
    Panel 2: Horizontal bar — SRI per product with weighted-average overlays.
    Panel 3: Bar — SVD↔AE agreement rate per product (SVD top-1 grouping).

    Parameters
    ----------
    products_df : pd.DataFrame
        Product catalogue from get_products().
    recs_svd, recs_ae : pd.DataFrame
        Recommendation outputs from recommend_svd() / recommend_ae().
    merged : pd.DataFrame
        Side-by-side comparison DataFrame returned by
        compare_recommendation_models().
    """
    prod_order = sorted(products_df['product_id'].tolist())
    prod_type  = (
        products_df.set_index('product_id')['type']
        if 'type' in products_df.columns else None
    )

    svd_counts = (
        recs_svd[recs_svd['status'] == 'ok']
        .groupby('product_id').size()
        .reindex(prod_order, fill_value=0)
    )
    ae_counts = (
        recs_ae[recs_ae['status'] == 'ok']
        .groupby('product_id').size()
        .reindex(prod_order, fill_value=0)
    )

    sri_map  = products_df.set_index('product_id')['SRI']
    sri_vals = [sri_map[p] for p in prod_order]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Product Distribution', fontsize=14, fontweight='bold', y=1.01)

    # Panel 1 — recommendation frequency
    ax = axes[0]
    x, w = np.arange(len(prod_order)), 0.38
    ax.bar(x - w / 2, svd_counts.values, w, label='SVD',         color=_BLUE,   alpha=0.85)
    ax.bar(x + w / 2, ae_counts.values,  w, label='Autoencoder', color=_ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(prod_order, rotation=45, ha='right')
    ax.set_ylabel('Times recommended (top-1)')
    ax.set_title('Recommendation frequency per product\n(SVD vs Autoencoder)', fontweight='bold')

    if prod_type is not None:
        for tick, pid in zip(ax.get_xticklabels(), prod_order):
            tick.set_color(_BLUE if prod_type[pid] == 'Accumulation' else _ORANGE)
        ax.legend(handles=[
            mpatches.Patch(color=_BLUE,   label='SVD'),
            mpatches.Patch(color=_ORANGE, label='AE'),
            mpatches.Patch(color=_BLUE,   label='Accumulation product'),
            mpatches.Patch(color=_ORANGE, label='Income product'),
        ], fontsize=7)
    else:
        ax.legend()

    # Panel 2 — SRI profile with weighted averages
    ax = axes[1]
    svd_weighted_sri = (svd_counts * sri_vals).sum() / svd_counts.sum()
    ae_weighted_sri  = (ae_counts  * sri_vals).sum() / ae_counts.sum()

    bar_colors = [
        _BLUE if (prod_type is not None and prod_type.get(p) == 'Accumulation')
        else _ORANGE
        for p in prod_order
    ]
    ax.barh(prod_order, sri_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.axvline(svd_weighted_sri, color='navy',       linestyle='--', lw=1.5,
               label=f'Weighted avg SRI — SVD ({svd_weighted_sri:.2f})')
    ax.axvline(ae_weighted_sri,  color='darkorange',  linestyle='--', lw=1.5,
               label=f'Weighted avg SRI — AE  ({ae_weighted_sri:.2f})')
    ax.set_xlabel('SRI (Synthetic Risk Indicator)')
    ax.set_title('SRI per product\n(weighted avg by recommendation frequency)', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 3 — SVD↔AE agreement rate per product
    ax = axes[2]
    merged_ok = merged.dropna(subset=['product_svd', 'product_ae'])
    agree_by_prod = (
        merged_ok.groupby('product_svd')['agree']
        .mean()
        .reindex(prod_order, fill_value=np.nan)
    )
    overall_agree = merged_ok['agree'].mean()

    colors_bar = [_GREEN if (not np.isnan(v) and v >= 0.5) else _RED
                  for v in agree_by_prod]
    ax.bar(prod_order, agree_by_prod.values, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(overall_agree, color='black', linestyle='--', lw=1.5,
               label=f'Overall agreement ({overall_agree:.1%})')
    ax.set_xlabel('Product (SVD top-1)')
    ax.set_ylabel('Agreement rate with AE')
    ax.set_title('SVD ↔ AE agreement rate\nper product (SVD top-1)', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks(range(len(prod_order)))
    ax.set_xticklabels(prod_order, rotation=45, ha='right')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3 — Risk propensity vs recommended SRI
# ---------------------------------------------------------------------------


def plot_risk_suitability(
    recs_svd: pd.DataFrame,
    recs_ae: pd.DataFrame,
    clients_df: pd.DataFrame,
) -> None:
    """
    Scatter plot of client RiskPropensity vs recommended SRI for each model.

    Every point must lie on or below the diagonal (SRI ≤ RiskPropensity) to
    satisfy the MiFID II hard constraint.

    Parameters
    ----------
    recs_svd, recs_ae : pd.DataFrame
        Recommendation outputs from recommend_svd() / recommend_ae().
    clients_df : pd.DataFrame
        Must contain column RiskPropensity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Risk Propensity vs Recommended SRI', fontsize=14, fontweight='bold')

    for ax, (recs, label, color) in zip(axes, [
        (recs_svd, 'SVD',         _BLUE),
        (recs_ae,  'Autoencoder', _ORANGE),
    ]):
        ok = recs[recs['status'] == 'ok'].merge(
            clients_df[['RiskPropensity']], left_on='client_idx', right_index=True
        )
        ax.scatter(ok['RiskPropensity'], ok['SRI'],
                   alpha=0.15, s=6, color=color, rasterized=True)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='SRI = Risk Propensity')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Client Risk Propensity')
        ax.set_ylabel('Recommended SRI')
        ax.set_title(
            f'{label}: client risk vs recommended SRI\n'
            f'(all points must be below the diagonal)',
            fontweight='bold',
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4 — Observed R vs SVD reconstruction heatmap
# ---------------------------------------------------------------------------


def plot_matrix_heatmap(
    R: np.ndarray,
    R_hat_svd: np.ndarray,
    products_df: pd.DataFrame,
    n_clients: int = 100,
) -> None:
    """
    Side-by-side heatmap: observed sparse R vs SVD reconstruction R̂.

    The left panel shows the binary 0/1 interaction matrix (very sparse, ~8%
    positive).  The right panel shows the continuous SVD scores: zero-inflation
    bias is immediately visible — scores are pulled toward zero across all
    products even for clients who have interacted.

    Parameters
    ----------
    R : np.ndarray
        Observed binary interaction matrix (n_clients, n_products).
    R_hat_svd : np.ndarray
        SVD score matrix from score_all_svd() (n_clients, n_products).
    products_df : pd.DataFrame
        Product catalogue from get_products() — used for x-axis labels.
    n_clients : int
        Number of clients (rows) to display; default 100 for readability.
    """
    prod_labels = products_df['product_id'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Observed interactions vs SVD reconstruction (first {n_clients} clients)',
        fontsize=13, fontweight='bold',
    )

    ax = axes[0]
    im = ax.imshow(R[:n_clients], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xticks(range(len(prod_labels)))
    ax.set_xticklabels(prod_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Client index')
    ax.set_title('R \u2014 observed (binary, ~8% positive)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im2 = ax.imshow(R_hat_svd[:n_clients], aspect='auto', cmap='RdYlGn',
                    interpolation='nearest')
    ax.set_xticks(range(len(prod_labels)))
    ax.set_xticklabels(prod_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Client index')
    ax.set_title('R\u0302 \u2014 SVD reconstruction (zero-inflation bias visible)',
                 fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Score')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Figure 5 — Held-out AUC comparison
# ---------------------------------------------------------------------------


def plot_roc_curves(
    k_results: dict,
    ae_k_results: dict,
    K_STAR: int,
    K_STAR_AE: int,
    save_path: str = None,
) -> None:
    """
    Plot held-out AUC curves for SVD and Autoencoder across all evaluated ranks.

    Uses the AUC scores already computed during rank/bottleneck selection
    (20% of positive entries held out, evaluated against all zeros).  These
    are genuine out-of-sample scores — evaluating the final models on the full
    R would give trivially inflated values (AE \u2248 1.000) due to training-set
    overlap.

    The right panel shows the k\u2605 AUC values side by side as a bar chart for
    direct numerical comparison.

    Parameters
    ----------
    k_results : dict
        Output of select_k() — must contain 'auc_scores' (dict k\u2192auc).
    ae_k_results : dict
        Output of select_k_ae() — must contain 'auc_scores' (dict k\u2192auc).
    K_STAR : int
        Optimal SVD rank (marked with a vertical line).
    K_STAR_AE : int
        Optimal AE bottleneck size (marked with a vertical line).
    save_path : str, optional
        If provided, save the figure to this path (e.g. 'materials/figures/auc_comparison.png').
    """
    svd_scores = k_results['auc_scores']
    ae_scores  = ae_k_results['auc_scores']

    # Restrict AE x-axis to the range it was evaluated on
    ae_ks  = sorted(ae_scores.keys())
    svd_ks = sorted(svd_scores.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Held-out AUC: SVD vs Autoencoder', fontsize=13,
                 fontweight='bold')

    # Panel 1 — AUC curves across k
    ax = axes[0]
    ax.plot(svd_ks, [svd_scores[k] for k in svd_ks],
            marker='o', lw=2, color=_BLUE,   label='SVD')
    ax.plot(ae_ks,  [ae_scores[k]  for k in ae_ks],
            marker='s', lw=2, color=_ORANGE, label='Autoencoder')
    ax.axvline(K_STAR,    color=_BLUE,   linestyle='--', lw=1,
               label=f'SVD k\u2605 = {K_STAR}  (AUC = {svd_scores[K_STAR]:.4f})')
    ax.axvline(K_STAR_AE, color=_ORANGE, linestyle=':',  lw=1.5,
               label=f'AE  k\u2605 = {K_STAR_AE}  (AUC = {ae_scores[K_STAR_AE]:.4f})')
    ax.set_xlabel('Rank / bottleneck size k')
    ax.set_ylabel('Held-out AUC (positives vs all zeros)')
    ax.set_title('AUC vs k \u2014 both models', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=max(0, min(min(svd_scores.values()),
                                  min(ae_scores.values())) - 0.05))

    # Panel 2 — k* AUC bar chart
    ax = axes[1]
    best = {'SVD': svd_scores[K_STAR], 'Autoencoder': ae_scores[K_STAR_AE]}
    colors = [_BLUE, _ORANGE]
    bars = ax.bar(list(best.keys()), list(best.values()), color=colors,
                  alpha=0.85, width=0.4, edgecolor='white')
    for bar, val in zip(bars, best.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Held-out AUC at k\u2605')
    ax.set_title('Best held-out AUC at optimal k\u2605', fontweight='bold')
    delta = ae_scores[K_STAR_AE] - svd_scores[K_STAR]
    ax.text(0.98, 0.05,
            f'\u0394AUC (AE \u2212 SVD) = +{delta:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                      ec='gray', alpha=0.9))
    ax.set_ylim(0, min(1.05, max(best.values()) + 0.1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Figure 6 — Recommendation frequency detail
# ---------------------------------------------------------------------------


def plot_recommendation_frequency(
    products_df: pd.DataFrame,
    recs_svd: pd.DataFrame,
    recs_ae: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Focused two-panel plot showing SVD's conservative concentration vs AE spread.

    Panel 1: Grouped bar chart of recommendation counts per product.
    Panel 2: Cumulative share (Lorenz-style, sorted by SVD rank) — a steeper
             curve means higher concentration in fewer products.

    Parameters
    ----------
    products_df : pd.DataFrame
        Product catalogue from get_products().
    recs_svd, recs_ae : pd.DataFrame
        Recommendation outputs from recommend_svd() / recommend_ae().
    save_path : str, optional
        If provided, save the figure to this path (e.g. 'materials/figures/rec_frequency.png').
    """
    prod_order = sorted(products_df['product_id'].tolist())
    prod_type  = products_df.set_index('product_id')['type']

    svd_counts = (
        recs_svd[recs_svd['status'] == 'ok']
        .groupby('product_id').size()
        .reindex(prod_order, fill_value=0)
    )
    ae_counts = (
        recs_ae[recs_ae['status'] == 'ok']
        .groupby('product_id').size()
        .reindex(prod_order, fill_value=0)
    )

    total_svd = max(svd_counts.sum(), 1)
    total_ae  = max(ae_counts.sum(), 1)

    svd_sorted   = svd_counts.sort_values(ascending=False)
    ae_reindexed = ae_counts.reindex(svd_sorted.index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        'Recommendation frequency: SVD concentration vs AE spread',
        fontsize=13, fontweight='bold', y=1.02,
    )

    ax = axes[0]
    x, w = np.arange(len(prod_order)), 0.38
    ax.bar(x - w / 2, svd_counts.values, w, color=_BLUE,   alpha=0.85, label='SVD')
    ax.bar(x + w / 2, ae_counts.values,  w, color=_ORANGE, alpha=0.85, label='Autoencoder')
    ax.set_xticks(x)
    ax.set_xticklabels(prod_order, rotation=45, ha='right')
    for tick, pid in zip(ax.get_xticklabels(), prod_order):
        tick.set_color(_BLUE if prod_type[pid] == 'Accumulation' else _ORANGE)
    ax.set_ylabel('Number of clients recommended')
    ax.set_title('Recommendation count per product', fontweight='bold')
    ax.legend()
    top3_svd = svd_counts.nlargest(3).sum() / total_svd
    top3_ae  = ae_counts.nlargest(3).sum()  / total_ae
    ax.text(
        0.98, 0.96,
        f'Top-3 share:  SVD {top3_svd:.0%}  |  AE {top3_ae:.0%}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9),
    )

    ax = axes[1]
    cum_svd = svd_sorted.values.cumsum() / total_svd
    cum_ae  = ae_reindexed.values.cumsum() / total_ae
    x_pos   = np.arange(1, len(prod_order) + 1)
    ax.step(x_pos, cum_svd, where='post', color=_BLUE,   lw=2,
            marker='o', ms=6, label='SVD')
    ax.step(x_pos, cum_ae,  where='post', color=_ORANGE, lw=2,
            marker='s', ms=6, label='Autoencoder')
    ax.axhline(1.0, color=_GRAY, lw=0.8, linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(svd_sorted.index, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Product (sorted by SVD recommendation count, descending)')
    ax.set_ylabel('Cumulative share of all recommendations')
    ax.set_title(
        'Cumulative recommendation share\n(steeper = more concentrated in fewer products)',
        fontweight='bold',
    )
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
