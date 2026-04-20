"""
Evaluation utilities for the investment product recommendation pipeline.

Functions
---------
compute_propensity_on_full_dataset
    Apply the best pickled classifier per target to all 5,000 clients and
    return a DataFrame with calibrated propensity scores and binary need flags.

evaluate_recommendation_coverage
    Print coverage, suitability, and no-product metrics for one model's
    recommendation output.

compare_recommendation_models
    Compute and display per-model and cross-model comparison metrics for SVD
    vs Autoencoder, then print a side-by-side sample of top-1 recommendations.
    Returns the merged comparison DataFrame for downstream plotting.
"""

from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    build_baseline_features,
    build_features,
    get_propensity_scores,
    load_data,
)


# ---------------------------------------------------------------------------
# Stage 1 helper — propensity scores on full dataset
# ---------------------------------------------------------------------------


def compute_propensity_on_full_dataset(
    df: pd.DataFrame,
    best_results: dict,
) -> pd.DataFrame:
    """
    Apply the best pickled model per target to all clients and return
    a clients_df with propensity scores and binary need flags.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (output of load_data()).
    best_results : dict
        Mapping target_name → result dict (output of load_result()).
        Each result dict must contain: 'model', 'scaler', 'feature_names',
        and optionally 'threshold_metrics' or 'threshold_info'.

    Returns
    -------
    pd.DataFrame
        One row per client with columns:
        RiskPropensity, p_hat_income, need_income, p_hat_accum, need_accum.
    """
    X_fe = build_features(df)
    X_fb = build_baseline_features(df)

    clients_df = df[['RiskPropensity']].copy().reset_index(drop=True)

    _need_col = {
        'IncomeInvestment':       ('p_hat_income', 'need_income'),
        'AccumulationInvestment': ('p_hat_accum',  'need_accum'),
    }

    for target, (col_phat, col_need) in _need_col.items():
        if target not in best_results:
            print(f'Skipping {target} — pickle not loaded')
            continue

        r          = best_results[target]
        model      = r['model']
        scaler     = r['scaler']
        feat_names = r['feature_names']

        X = X_fb if feat_names == BASELINE_FEATURE_NAMES else X_fe

        if scaler is not None:
            X_in = pd.DataFrame(
                scaler.transform(X), columns=X.columns, index=X.index
            )
        else:
            X_in = X

        thr_info  = r.get('threshold_metrics') or r.get('threshold_info')
        threshold = thr_info['threshold'] if thr_info else 0.5

        clients_df[col_phat] = get_propensity_scores(model, X_in)
        clients_df[col_need] = (clients_df[col_phat] >= threshold).astype(bool)

    print(
        f"Clients with income need  : {clients_df['need_income'].sum()} "
        f"({clients_df['need_income'].mean():.1%})"
    )
    print(
        f"Clients with accum need   : {clients_df['need_accum'].sum()} "
        f"({clients_df['need_accum'].mean():.1%})"
    )
    print(
        f"Clients with either need  : "
        f"{(clients_df['need_income'] | clients_df['need_accum']).sum()}"
    )
    return clients_df


# ---------------------------------------------------------------------------
# Coverage metrics — one model
# ---------------------------------------------------------------------------


def evaluate_recommendation_coverage(
    recs_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    eligible_clients: list,
    n_total: int,
    score_col: str = 'svd_score',
) -> None:
    """
    Print coverage, suitability, and no-product metrics for one model.

    Parameters
    ----------
    recs_df : pd.DataFrame
        Output of recommend_svd() or recommend_ae().
    clients_df : pd.DataFrame
        Must contain column RiskPropensity.
    eligible_clients : list
        Client indices with at least one confirmed need.
    n_total : int
        Total number of clients in the dataset (used for coverage rate).
    score_col : str
        'svd_score' or 'ae_score' — used only to label the print header.
    """
    label     = 'SVD' if 'svd' in score_col else 'Autoencoder'
    ok        = recs_df[recs_df['status'] == 'ok']
    n_served  = ok['client_idx'].nunique()
    n_eligible = len(eligible_clients)
    n_no_prod = recs_df[
        recs_df['status'] == 'no_compliant_product'
    ]['client_idx'].nunique()

    merged_risk = ok.merge(
        clients_df[['RiskPropensity']], left_on='client_idx', right_index=True
    )
    suit_pass = (merged_risk['SRI'] <= merged_risk['RiskPropensity']).mean()

    print(f"[{label}]")
    print(f"  Eligible clients          : {n_eligible}")
    print(f"  Clients served (>= 1 rec) : {n_served}  ({n_served/n_eligible:.1%})")
    print(f"  Coverage (of {n_total:,})      : {n_served/n_total:.1%}")
    print(f"  Clients with no product   : {n_no_prod}")
    print(f"  Suitability pass rate     : {suit_pass:.1%}  (must be 100%)")


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def _top1_wide(recs: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Pivot top-1 ok rows to one row per (client_idx, need_type)."""
    tag = score_col.split('_')[0]
    ok  = recs[recs['status'] == 'ok'][
        ['client_idx', 'need_type', 'product_id', 'SRI', score_col]
    ]
    return ok.rename(columns={'product_id': f'product_{tag}', 'SRI': f'SRI_{tag}'})


def _top1_set(recs: pd.DataFrame) -> set:
    """Set of (client_idx, need_type, product_id) triples from top-1 ok rows."""
    return set(
        recs[recs['status'] == 'ok'][['client_idx', 'need_type', 'product_id']]
        .itertuples(index=False, name=None)
    )


def compare_recommendation_models(
    recs_svd: pd.DataFrame,
    recs_ae: pd.DataFrame,
    clients_df: pd.DataFrame,
    k_results: dict,
    ae_k_results: dict,
    K_STAR: int,
    K_STAR_AE: int,
    n_total: int = 5000,
) -> pd.DataFrame:
    """
    Compute and display per-model and cross-model comparison metrics, then
    print a side-by-side top-1 sample.

    Parameters
    ----------
    recs_svd, recs_ae : pd.DataFrame
        Recommendation outputs from recommend_svd() / recommend_ae().
    clients_df : pd.DataFrame
        Must contain columns: RiskPropensity, need_income, p_hat_income,
        p_hat_accum.
    k_results : dict
        Output of select_k() — must contain 'auc_scores'.
    ae_k_results : dict
        Output of select_k_ae() — must contain 'auc_scores'.
    K_STAR, K_STAR_AE : int
        Optimal rank / bottleneck size from the rank-selection step.
    n_total : int
        Total number of clients (default 5,000).

    Returns
    -------
    pd.DataFrame
        Merged side-by-side comparison (all ok clients, both models), with
        columns: client_idx, need_type, p_hat, product_svd, SRI_svd,
        product_ae, SRI_ae, agree.
        Used as input to plot_product_distribution().
    """
    svd_ok = recs_svd[recs_svd['status'] == 'ok']
    ae_ok  = recs_ae[recs_ae['status']  == 'ok']

    coverage_svd = svd_ok['client_idx'].nunique() / n_total
    coverage_ae  = ae_ok['client_idx'].nunique()  / n_total

    def _suit(ok):
        m = ok.merge(
            clients_df[['RiskPropensity']], left_on='client_idx', right_index=True
        )
        return (m['SRI'] <= m['RiskPropensity']).mean()

    suit_svd = _suit(svd_ok)
    suit_ae  = _suit(ae_ok)

    svd_served_idx = svd_ok['client_idx'].unique()
    ae_served_idx  = ae_ok['client_idx'].unique()

    avg_p_income_svd = clients_df.loc[
        clients_df.index.isin(svd_served_idx) & clients_df['need_income'],
        'p_hat_income',
    ].mean()
    avg_p_income_ae = clients_df.loc[
        clients_df.index.isin(ae_served_idx) & clients_df['need_income'],
        'p_hat_income',
    ].mean()

    svd_set = _top1_set(recs_svd)
    ae_set  = _top1_set(recs_ae)
    jaccard = (
        len(svd_set & ae_set) / len(svd_set | ae_set)
        if (svd_set | ae_set) else 0.0
    )

    best_auc_svd = k_results['auc_scores'][K_STAR]
    best_auc_ae  = ae_k_results['auc_scores'][K_STAR_AE]

    summary = pd.DataFrame({
        'Metric': [
            'Coverage rate (of 5,000 clients)',
            'Suitability pass rate (MiFID II)',
            'Avg p_hat_income of served clients',
            'Held-out AUC (rank selection)',
            'Bottleneck size k*',
        ],
        'SVD': [
            f"{coverage_svd:.1%}",
            f"{suit_svd:.1%}",
            f"{avg_p_income_svd:.3f}",
            f"{best_auc_svd:.4f}",
            str(K_STAR),
        ],
        'Autoencoder': [
            f"{coverage_ae:.1%}",
            f"{suit_ae:.1%}",
            f"{avg_p_income_ae:.3f}",
            f"{best_auc_ae:.4f}",
            str(K_STAR_AE),
        ],
    })

    cross = pd.DataFrame({
        'Cross-model metric': [
            'Recommendation overlap (Jaccard)',
            'Coverage delta (AE - SVD)',
            'AUC delta (AE - SVD)',
        ],
        'Value': [
            f"{jaccard:.3f}",
            f"{coverage_ae - coverage_svd:+.1%}",
            f"{best_auc_ae - best_auc_svd:+.4f}",
        ],
    })

    print("Per-model metrics:")
    display(summary.set_index('Metric'))
    print("\nCross-model metrics:")
    display(cross.set_index('Cross-model metric'))

    # Side-by-side top-1 sample
    svd_wide = _top1_wide(recs_svd, 'svd_score')
    ae_wide  = _top1_wide(recs_ae,  'ae_score')

    merged = svd_wide.merge(ae_wide, on=['client_idx', 'need_type'], how='outer')
    merged = merged.merge(
        clients_df[['RiskPropensity', 'p_hat_income', 'p_hat_accum']],
        left_on='client_idx', right_index=True,
    )
    merged['p_hat'] = np.where(
        merged['need_type'] == 'Income',
        merged['p_hat_income'],
        merged['p_hat_accum'],
    )
    merged['agree'] = merged['product_svd'] == merged['product_ae']

    sample = (
        merged[['client_idx', 'need_type', 'p_hat',
                'product_svd', 'SRI_svd', 'product_ae', 'SRI_ae', 'agree']]
        .sort_values('client_idx')
        .head(10)
        .reset_index(drop=True)
    )
    print("\nSide-by-side sample (10 clients):")
    display(sample)

    return merged
