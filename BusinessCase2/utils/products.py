"""
Product catalogue and client-product interaction matrix for the collaborative
filtering stage of the recommendation pipeline.

Theory
------
The collaborative filtering stage requires a user-item interaction matrix
R ∈ {0,1}^(n×m), where R[c, p] = 1 if client c purchased product p.

No explicit purchase history is available in the dataset, so R is constructed
via the **revealed preference principle**: the binary investment-need labels
(IncomeInvestment, AccumulationInvestment) encode that a trusted advisor
identified the need and sold a matching product. Under mean-variance utility
preferences, the product sold is the one whose Synthetic Risk Indicator (SRI)
is closest to the client's RiskPropensity — the utility-maximising choice
within the feasible product set.

Formally, for each need type t ∈ {Income, Accumulation}:

    R[c, p] = 1  iff  label_t[c] = 1
                  and  p = argmin_{p': type(p') = t} |SRI_{p'} − RiskPropensity_c|

This produces at most 2 interactions per client (one per need type) and a
matrix density of approximately 8%, which is sufficient for low-rank
collaborative filtering.

Product catalogue
-----------------
Eleven products are defined with SRI spanning [0.12, 0.88]:

- 6 Accumulation products: SRI in {0.15, 0.28, 0.42, 0.56, 0.72, 0.88}.
  Wider SRI range reflects the broader and more risk-diverse client base
  implied by the 51% AccumulationInvestment positive rate.

- 5 Income products: SRI in {0.12, 0.22, 0.35, 0.50, 0.65}.
  Conservative bias reflects the lifecycle hypothesis: income seekers are
  empirically older and wealthier, correlating with lower RiskPropensity.

Implementation
--------------
- get_products: returns the static product catalogue as a DataFrame.
- build_interaction_matrix: constructs R from the raw dataset and catalogue.
- check_interaction_matrix: validates structural properties of R and
  optionally plots product popularity.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INCOME       = "Income"
ACCUMULATION = "Accumulation"

NEED_TO_TYPE: dict = {
    "IncomeInvestment":       INCOME,
    "AccumulationInvestment": ACCUMULATION,
}

_CATALOGUE = [
    # Accumulation products — SRI spans [0.15, 0.88]
    {"product_id": "P01", "type": ACCUMULATION, "SRI": 0.15},
    {"product_id": "P02", "type": ACCUMULATION, "SRI": 0.28},
    {"product_id": "P03", "type": ACCUMULATION, "SRI": 0.42},
    {"product_id": "P04", "type": ACCUMULATION, "SRI": 0.56},
    {"product_id": "P05", "type": ACCUMULATION, "SRI": 0.72},
    {"product_id": "P06", "type": ACCUMULATION, "SRI": 0.88},
    # Income products — SRI spans [0.12, 0.65], conservative bias
    {"product_id": "P07", "type": INCOME, "SRI": 0.12},
    {"product_id": "P08", "type": INCOME, "SRI": 0.22},
    {"product_id": "P09", "type": INCOME, "SRI": 0.35},
    {"product_id": "P10", "type": INCOME, "SRI": 0.50},
    {"product_id": "P11", "type": INCOME, "SRI": 0.65},
]


# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------


def get_products() -> pd.DataFrame:
    """
    Return the static product catalogue.

    Returns
    -------
    pd.DataFrame
        Columns: product_id (str), type (str), SRI (float).
        Index: RangeIndex 0…10, where column p in the interaction matrix
        corresponds to row p of this DataFrame.
    """
    return pd.DataFrame(_CATALOGUE)


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------


def build_interaction_matrix(
    df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the client-product interaction matrix R via revealed preference.

    For each need type t with label_t[c] = 1, assign R[c, p] = 1 where p is
    the product of type t whose SRI minimises |SRI_p − RiskPropensity_c|.
    This is the utility-maximising product under mean-variance preferences.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.  Must contain columns: RiskPropensity, IncomeInvestment,
        AccumulationInvestment.
    products_df : pd.DataFrame
        Product catalogue with a RangeIndex 0…(m-1) and columns: type, SRI.
        Use the output of :func:`get_products`.

    Returns
    -------
    np.ndarray
        Binary matrix of shape (n_clients, n_products) with dtype int8.
        R[c, p] = 1 if client c is assigned to product p.

    Raises
    ------
    KeyError
        If any required columns are missing from df or products_df.
    """
    required_df = {"RiskPropensity", "IncomeInvestment", "AccumulationInvestment"}
    missing = required_df - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    required_prod = {"type", "SRI"}
    missing_prod = required_prod - set(products_df.columns)
    if missing_prod:
        raise KeyError(f"Missing columns in products_df: {missing_prod}")

    n_clients  = len(df)
    n_products = len(products_df)
    R = np.zeros((n_clients, n_products), dtype=np.int8)

    risk = df["RiskPropensity"].values  # (n_clients,)

    for target, prod_type in NEED_TO_TYPE.items():
        # Global positions (in products_df) of products with this type
        type_positions = np.where(products_df["type"] == prod_type)[0]
        if len(type_positions) == 0:
            continue

        product_sris = products_df.loc[type_positions, "SRI"].values  # (n_type_products,)

        # Clients who have this need
        client_positions = np.where(df[target].values == 1)[0]
        if len(client_positions) == 0:
            continue

        client_risks = risk[client_positions]  # (n_positive,)

        # Pairwise |SRI_p - risk_c|: shape (n_positive, n_type_products)
        diffs = np.abs(client_risks[:, None] - product_sris[None, :])

        # For each positive client: index within the type-subset → global position
        closest_local  = np.argmin(diffs, axis=1)
        closest_global = type_positions[closest_local]

        R[client_positions, closest_global] = 1

    return R


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_interaction_matrix(
    R: np.ndarray,
    products_df: pd.DataFrame,
    plot: bool = True,
) -> dict:
    """
    Validate structural properties of R and report summary statistics.

    Checks
    ------
    1. Column count matches the product catalogue.
    2. All values are binary (0 or 1).
    3. Each client has at most 2 interactions.
    4. Total interactions are in the expected range [4000, 5000].

    Parameters
    ----------
    R : np.ndarray
        Interaction matrix from :func:`build_interaction_matrix`.
    products_df : pd.DataFrame
        Product catalogue from :func:`get_products`.
    plot : bool
        If True, display a product-popularity bar chart.

    Returns
    -------
    dict
        Keys: shape, n_interactions, density, max_per_client,
        interactions_per_product, all_checks_passed.
    """
    interactions_per_client  = R.sum(axis=1)
    interactions_per_product = R.sum(axis=0)
    n_interactions  = int(R.sum())
    density         = n_interactions / R.size
    max_per_client  = int(interactions_per_client.max())

    checks = {
        "shape_ok":        R.ndim == 2 and R.shape[1] == len(products_df),
        "binary_ok":       bool(((R == 0) | (R == 1)).all()),
        "max_per_client":  max_per_client <= 2,
        "interactions_in_range": 4000 <= n_interactions <= 5000,
    }

    def _ok(flag: bool) -> str:
        return "OK" if flag else "FAIL"

    print("=" * 45)
    print("Interaction matrix checks")
    print("=" * 45)
    print(f"  Shape                   : {R.shape}  [{_ok(checks['shape_ok'])}]")
    print(f"  Binary values           : [{_ok(checks['binary_ok'])}]")
    print(f"  Max interactions/client : {max_per_client}  [{_ok(checks['max_per_client'])}]")
    print(f"  Total interactions      : {n_interactions}  [{_ok(checks['interactions_in_range'])}]")
    print(f"  Matrix density          : {density:.2%}")
    print("-" * 45)
    print(f"  Clients with 0 interactions : {(interactions_per_client == 0).sum()}")
    print(f"  Clients with 1 interaction  : {(interactions_per_client == 1).sum()}")
    print(f"  Clients with 2 interactions : {(interactions_per_client == 2).sum()}")

    all_passed = all(checks.values())
    print(f"\n  All checks passed: [{_ok(all_passed)}]")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = [
            "#4C72B0" if t == ACCUMULATION else "#DD8452"
            for t in products_df["type"]
        ]
        ax.bar(products_df["product_id"], interactions_per_product, color=colors)
        ax.set_xlabel("Product")
        ax.set_ylabel("Number of clients")
        ax.set_title("Product popularity — interactions per product")

        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor="#4C72B0", label="Accumulation"),
            Patch(facecolor="#DD8452", label="Income"),
        ]
        ax.legend(handles=legend)

        # Annotate SRI values on each bar
        for i, (sri, count) in enumerate(zip(products_df["SRI"], interactions_per_product)):
            ax.text(i, count + 5, f"SRI={sri:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.show()

    return {
        "shape":                    R.shape,
        "n_interactions":           n_interactions,
        "density":                  density,
        "max_per_client":           max_per_client,
        "interactions_per_product": interactions_per_product,
        "all_checks_passed":        all_passed,
    }