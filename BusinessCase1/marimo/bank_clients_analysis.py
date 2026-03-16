import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Bank Clients — EDA & K-Medoids Clustering")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    import marimo as mo

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    matplotlib.rcParams["figure.dpi"] = 110

    print("✅ Core libraries loaded")
    return Path, matplotlib, mo, np, pd, plt, sns, warnings


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 🏦 Bank Clients — Exploratory Data Analysis & K-Medoids Clustering

        This notebook combines two analyses:
        1. **EDA & Outlier Detection** — data quality, distributions, domain anomalies
        2. **K-Medoids Clustering** — mixed-type segmentation with Gower distance

        ---

        **Features Metadata:**

        | Feature | Type | Description |
        |---------|------|-------------|
        | `Age` | Numerical | Age in years |
        | `Gender` | Categorical | 0 = Male, 1 = Female |
        | `Job` | Categorical | 1=Unemployed, 2=Employee, 3=Manager, 4=Entrepreneur, 5=Retired |
        | `Area` | Categorical | 1=North, 2=Center, 3=South/Islands |
        | `CitySize` | Categorical | 1=Small, 2=Medium, 3=Large (>200k) |
        | `FamilySize` | Numerical | Number of family members |
        | `Income` | Numerical | Normalized income percentile $\in [0,1]$ |
        | `Wealth` | Numerical | Normalized wealth percentile $\in [0,1]$ |
        | `Debt` | Numerical | Normalized debt percentile $\in [0,1]$ |
        | `FinEdu` | Numerical | Normalized financial education $\in [0,1]$ |
        | `ESG` | Numerical | Normalized ESG propensity $\in [0,1]$ |
        | `Digital` | Numerical | Normalized digital propensity $\in [0,1]$ |
        | `BankFriend` | Numerical | Normalized bank friendliness $\in [0,1]$ |
        | `LifeStyle` | Numerical | Normalized lifestyle index $\in [0,1]$ |
        | `Luxury` | Numerical | Normalized luxury spending $\in [0,1]$ |
        | `Saving` | Numerical | Normalized saving propensity $\in [0,1]$ |
        | `Investments` | Categorical | 1=None, 2=Lump Sum, 3=Capital Accumulation |
        """
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# SVG ASSET PANEL
# ─────────────────────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ---
        ## 🖼️ SVG Assets

        Place any `.svg` files inside the `assets/` folder next to this notebook.
        They are discovered and rendered automatically.
        """
    )
    return


@app.cell
def __(Path, mo):
    _assets_dir = Path(__file__).parent / "assets"
    _svg_files = sorted(_assets_dir.glob("*.svg")) if _assets_dir.exists() else []

    if _svg_files:
        _cards = []
        for _f in _svg_files:
            _title = _f.stem.replace("_", " ").title()
            _svg = _f.read_text(encoding="utf-8")
            _cards.append(mo.vstack([mo.md(f"**{_title}**"), mo.Html(_svg)]))
        svg_panel = mo.hstack(_cards, wrap=True)
    else:
        svg_panel = mo.callout(
            mo.md("No SVG files found in `assets/`. Add `.svg` files there to display them."),
            kind="info",
        )

    svg_panel
    return (svg_panel,)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ---
        # Part 1 — Exploratory Data Analysis & Outlier Detection
        """
    )
    return


# ── 1. Data Loading ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1. Data Loading & Sanity Checks")
    return


@app.cell
def __(Path, pd):
    _data_path = Path(__file__).parent.parent / "Data" / "Dataset1_BankClients.xlsx"
    df_raw = pd.read_excel(_data_path)
    print(f"📁  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns loaded from {_data_path.name}")
    df_raw.head()
    return (df_raw,)


@app.cell
def __(df_raw, mo, pd):
    df = df_raw.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    _missing = df.isna().sum()
    _dupes = df.duplicated().sum()

    _tbl = pd.DataFrame({
        "Column": _missing.index,
        "Missing": _missing.values,
        "Missing %": (_missing.values / len(df) * 100).round(2),
    })

    mo.vstack([
        mo.md(f"**Shape (ID dropped):** {df.shape[0]:,} × {df.shape[1]}  |  **Duplicates:** {_dupes}"),
        mo.ui.table(_tbl, selection=None),
    ])
    return (df,)


# ── 2. Feature Separation ────────────────────────────────────────────────────
@app.cell
def __(df, mo):
    categorical_cols = ["Gender", "Job", "Area", "CitySize", "Investments"]
    numerical_cols = [c for c in df.columns if c not in categorical_cols]
    normalized_vars = [c for c in numerical_cols if c not in ["Age", "FamilySize"]]

    mo.vstack([
        mo.md(f"**Categorical ({len(categorical_cols)}):** {', '.join(categorical_cols)}"),
        mo.md(f"**Numerical ({len(numerical_cols)}):** {', '.join(numerical_cols)}"),
    ])
    return categorical_cols, normalized_vars, numerical_cols


# ── 3. Numerical Distributions ───────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2. Univariate Analysis — Numerical Features

        We inspect histograms + boxplots for each numerical feature.
        Percentile-normalized variables should lie in $[0,1]$.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols, pd):
    _desc = df[numerical_cols].describe().T.round(4).reset_index()
    _desc.columns = ["Feature"] + list(_desc.columns[1:])
    mo.ui.table(pd.DataFrame(_desc), selection=None)
    return


@app.cell
def __(df, numerical_cols, plt, sns):
    _n = len(numerical_cols)
    fig_num, axes_num = plt.subplots(_n, 2, figsize=(14, 4 * _n))

    for _i, _col in enumerate(numerical_cols):
        sns.histplot(df[_col], kde=True, ax=axes_num[_i, 0], color="#4A90D9")
        axes_num[_i, 0].set_title(f"Distribution — {_col}", fontweight="bold")
        sns.boxplot(x=df[_col], ax=axes_num[_i, 1], color="#7ED321")
        axes_num[_i, 1].set_title(f"Boxplot — {_col}", fontweight="bold")

    plt.suptitle("Numerical Features — Distributions & Boxplots", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    fig_num
    return axes_num, fig_num


# ── 4. Bound Checks ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Outlier Bound Checks

        - **Age** $\in (0, 120]$
        - **FamilySize** $\geq 1$
        - **Normalized features** $\in [0, 1]$
        """
    )
    return


@app.cell
def __(df, mo, normalized_vars, pd):
    _rows = []
    _rows.append({
        "Feature": "Age",
        "Min": round(float(df["Age"].min()), 2),
        "Max": round(float(df["Age"].max()), 2),
        "Expected": "[0, 120]",
        "OK?": "✅" if 0 <= df["Age"].min() and df["Age"].max() <= 120 else "❌",
    })
    _rows.append({
        "Feature": "FamilySize",
        "Min": int(df["FamilySize"].min()),
        "Max": int(df["FamilySize"].max()),
        "Expected": "≥ 1",
        "OK?": "✅" if df["FamilySize"].min() >= 1 else "❌",
    })
    for _col in normalized_vars:
        _mn, _mx = float(df[_col].min()), float(df[_col].max())
        _rows.append({
            "Feature": _col,
            "Min": round(_mn, 4),
            "Max": round(_mx, 4),
            "Expected": "[0, 1]",
            "OK?": "✅" if _mn >= 0 and _mx <= 1 else "❌",
        })

    _has_issues = any(r["OK?"] == "❌" for r in _rows)
    mo.vstack([
        mo.callout(
            mo.md("⚠️ Some features are out of expected bounds!" if _has_issues else "All feature bounds look healthy ✅"),
            kind="warn" if _has_issues else "success",
        ),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


# ── 5. Categorical Distributions ─────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3. Univariate Analysis — Categorical Features")
    return


@app.cell
def __(categorical_cols, df, plt, sns):
    _labels_map = {
        "Gender":      {0: "Male", 1: "Female"},
        "Job":         {1: "Unemployed", 2: "Employee", 3: "Manager", 4: "Entrepreneur", 5: "Retired"},
        "Area":        {1: "North", 2: "Center", 3: "South/Islands"},
        "CitySize":    {1: "Small", 2: "Medium", 3: "Large"},
        "Investments": {1: "None", 2: "Lump Sum", 3: "Cap. Accum."},
    }

    fig_cat, axes_cat = plt.subplots(len(categorical_cols), 1, figsize=(9, 4 * len(categorical_cols)))
    for _i, _col in enumerate(categorical_cols):
        _order = sorted(df[_col].dropna().unique())
        sns.countplot(x=df[_col], order=_order, ax=axes_cat[_i], palette="viridis")
        axes_cat[_i].set_title(f"Distribution — {_col}", fontweight="bold")
        axes_cat[_i].set_xticklabels(
            [_labels_map[_col].get(int(v), str(v)) for v in _order], rotation=20
        )

    plt.suptitle("Categorical Features — Count Distributions", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    fig_cat
    return axes_cat, fig_cat


# ── 6. Correlation Heatmap ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4. Bivariate Analysis — Pearson Correlation

        $$r_{XY} = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i-\bar{Y})^2}}$$

        Pairs with $|r| > 0.8$ are flagged — they risk providing duplicate information in
        distance-based clustering.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols, pd, plt, sns):
    corr_matrix = df[numerical_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
        vmin=-1, vmax=1, ax=ax_corr, linewidths=0.4, annot_kws={"size": 8},
    )
    ax_corr.set_title("Pearson Correlation Heatmap — Numerical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()

    _threshold = 0.8
    _pairs = []
    for _i in range(len(corr_matrix.columns)):
        for _j in range(_i):
            _v = corr_matrix.iloc[_i, _j]
            if abs(_v) > _threshold:
                _pairs.append({
                    "Feature A": corr_matrix.columns[_i],
                    "Feature B": corr_matrix.columns[_j],
                    "Pearson r": round(float(_v), 4),
                })

    mo.vstack([
        fig_corr,
        mo.md(f"### High-Correlation Pairs (|r| > {_threshold})"),
        mo.callout(
            mo.md(f"**{len(_pairs)} pair(s)** with |r| > {_threshold}." if _pairs else "No pairs above threshold ✅"),
            kind="warn" if _pairs else "success",
        ),
        mo.ui.table(pd.DataFrame(_pairs if _pairs else [{"Result": "None found"}]), selection=None),
    ])
    return ax_corr, corr_matrix, fig_corr


# ── 7. Advanced Duplicate Checks ─────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. Advanced Duplicate Checks

        Beyond perfectly identical rows, we search for clients sharing the same
        **complete financial profile** — a potential sign of synthetic data artefacts.
        """
    )
    return


@app.cell
def __(df, mo):
    _fin_cols = [
        "Income", "Wealth", "Debt", "FinEdu", "ESG",
        "Digital", "BankFriend", "LifeStyle", "Luxury", "Saving", "Investments",
    ]
    _n = df.duplicated(subset=_fin_cols, keep=False).sum()
    mo.callout(
        mo.md(f"**{_n}** rows share an identical financial profile." if _n > 0 else "No financial profile duplicates ✅"),
        kind="warn" if _n > 10 else "success",
    )
    return


# ── 8. Demographics vs Financials ────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 6. Demographics vs Financials

        Income and Wealth distributions across job categories should align with
        real-world expectations (managers & entrepreneurs → higher distributions).
        """
    )
    return


@app.cell
def __(df, plt, sns):
    fig_biv, axes_biv = plt.subplots(1, 2, figsize=(14, 5))
    _job_labels = ["Unemployed", "Employee", "Manager", "Entrepreneur", "Retired"]

    sns.boxplot(x="Job", y="Income", data=df, ax=axes_biv[0], palette="Set2")
    axes_biv[0].set_title("Income by Job Category", fontweight="bold")
    axes_biv[0].set_xticklabels(_job_labels, rotation=30)

    sns.boxplot(x="Job", y="Wealth", data=df, ax=axes_biv[1], palette="Set2")
    axes_biv[1].set_title("Wealth by Job Category", fontweight="bold")
    axes_biv[1].set_xticklabels(_job_labels, rotation=30)

    plt.tight_layout()
    fig_biv
    return axes_biv, fig_biv


# ── 9. Domain Knowledge Anomalies ────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 7. Domain Knowledge Anomaly Detection

        | Cohort | Rule | Concern |
        |--------|------|---------|
        | Young Retirees | Age < 50 & Job = Retired | Suspicious early retirement |
        | Working Minors | Age < 18 & Job ∈ {2,3,4} | Legal/data issue |
        | Rich Unemployed | Job = 1 & Income > 90th pct | Contradictory signals |
        | Wealthy, No Investments | Investments = 1 & Wealth > 95th pct | Missed opportunity |
        | Young Large Families | Age ≤ 20 & FamilySize ≥ 4 | Demographically unusual |
        """
    )
    return


@app.cell
def __(df, mo, pd):
    _cohorts = {
        "Young Retirees (Age < 50, Job=Retired)":   ((df["Age"] < 50) & (df["Job"] == 5)).sum(),
        "Working Minors (Age < 18, Job∈{2,3,4})":   ((df["Age"] < 18) & (df["Job"].isin([2,3,4]))).sum(),
        "Rich Unemployed (Job=1 & Income > 0.9)":   ((df["Job"] == 1) & (df["Income"] > 0.9)).sum(),
        "Wealthy No Invest (Inv=1 & Wealth > 0.95)": ((df["Investments"] == 1) & (df["Wealth"] > 0.95)).sum(),
        "Young Large Fam (Age≤20 & FamSz≥4)":       ((df["Age"] <= 20) & (df["FamilySize"] >= 4)).sum(),
    }
    _tbl = pd.DataFrame([{"Cohort": k, "Count": v, "%": f"{v/len(df)*100:.2f}%"} for k,v in _cohorts.items()])
    _total = sum(_cohorts.values())

    mo.vstack([
        mo.ui.table(_tbl, selection=None),
        mo.callout(
            mo.md(f"**{_total} total anomalous records** ({_total/len(df)*100:.2f}%). "
                  "Working minors and young retirees are candidates for removal before clustering."),
            kind="warn",
        ),
    ])
    return


# ── 10. Isolation Forest ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 8. Multivariate Outlier Detection — Isolation Forest

        Isolation Forest partitions the feature space randomly. Outliers require
        fewer splits to isolate. The anomaly score is:

        $$\text{score}(x, n) = 2^{-\,\dfrac{E[h(x)]}{c(n)}}$$

        where:
        - $h(x)$ = path length from root to leaf (average over trees)
        - $c(n) = 2H(n-1) - \dfrac{2(n-1)}{n}$  — expected path length under no structure ($H$ = harmonic number)

        A score $\approx 1$ indicates an **outlier**; $\approx 0.5$ indicates a **normal** observation.

        We flag the top **1%** ($\text{contamination} = 0.01$) as multivariate outliers.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols):
    from sklearn.ensemble import IsolationForest

    _iso = IsolationForest(contamination=0.01, random_state=42)
    _labels = _iso.fit_predict(df[numerical_cols])

    df_iso = df.copy()
    df_iso["Is_Outlier"] = (_labels == -1)
    df_clean = df_iso[~df_iso["Is_Outlier"]].drop(columns=["Is_Outlier"]).reset_index(drop=True)

    _n_out = int(df_iso["Is_Outlier"].sum())

    mo.callout(
        mo.md(f"**{_n_out} multivariate outliers** detected ({_n_out/len(df)*100:.2f}%). "
              f"Clean dataset: **{len(df_clean):,} clients**."),
        kind="warn",
    )
    return IsolationForest, df_clean, df_iso


# ── 11. PCA Visualization ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 9. Initial Dimensionality Reduction — PCA

        PCA finds the orthogonal directions of maximum variance:

        $$\mathbf{W}^* = \underset{\mathbf{W} : \mathbf{W}^\top\mathbf{W}=\mathbf{I}}{\arg\max}\;\operatorname{tr}\!\left(\mathbf{W}^\top \mathbf{\Sigma} \mathbf{W}\right)$$

        The $k$-th principal component explains a fraction of total variance:

        $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j} \lambda_j}$$

        where $\lambda_k$ is the $k$-th eigenvalue of the sample covariance $\mathbf{\Sigma}$.
        """
    )
    return


@app.cell
def __(df_clean, mo, numerical_cols, plt):
    from sklearn.decomposition import PCA

    _pca = PCA(n_components=2, random_state=42)
    _res = _pca.fit_transform(df_clean[numerical_cols])
    _ev1, _ev2 = _pca.explained_variance_ratio_

    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    ax_pca.scatter(_res[:, 0], _res[:, 1], alpha=0.35, s=10, color="#7B2D8B")
    ax_pca.set_title("PCA 2D Projection — Numerical Features", fontsize=14, fontweight="bold")
    ax_pca.set_xlabel(f"PC1  ({_ev1:.2%} variance)")
    ax_pca.set_ylabel(f"PC2  ({_ev2:.2%} variance)")
    ax_pca.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md(f"**Variance explained by PC1+PC2:** {(_ev1+_ev2):.2%}"),
        fig_pca,
    ])
    return PCA, ax_pca, fig_pca


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — K-MEDOIDS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ---
        # Part 2 — K-Medoids Clustering

        **Objective:** Identify distinct customer segments for targeted marketing strategies.
        **Approach:** K-Medoids with Gower distance + multi-metric validation.
        """
    )
    return


# ── Theory Section ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1. Distance Metrics — Theory

        Our dataset has **5 categorical** + **13 numerical** features — requiring a
        metric that handles mixed types natively.

        ---

        ### Euclidean Distance (L₂)

        $$d_{\text{Eucl}}(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

        ✅ Fast, intuitive geometry  
        ❌ **Cannot handle categorical variables** | Sensitive to outliers (squares amplify extremes)

        ---

        ### Manhattan Distance (L₁)

        $$d_{\text{Man}}(\mathbf{x},\mathbf{y}) = \sum_{i=1}^{n}|x_i - y_i|$$

        ✅ More robust than L₂ in high dimensions  
        ❌ **Still numerical only** | Assumes axis-aligned independence

        ---

        ### Jaccard Distance

        $$d_{\text{Jacc}}(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}$$

        ✅ Excellent for binary/set-valued features  
        ❌ **Ignores numerical magnitude** | Requires binarization

        ---

        ### Gower Distance ⭐ *Selected*

        $$d_{\text{Gower}}(\mathbf{x},\mathbf{y}) = \frac{1}{p}\sum_{j=1}^{p}\delta_j(\mathbf{x},\mathbf{y})$$

        Per-feature contributions:

        $$\delta_j = \begin{cases} \dfrac{|x_j - y_j|}{R_j} & \text{numerical} \quad (R_j = \max_j - \min_j) \\ \mathbf{1}[x_j \neq y_j] & \text{categorical (Hamming)} \end{cases}$$

        ✅ Native mixed-type support | Auto-normalization | Output $\in [0,1]$  
        ❌ O(n²) distance matrix — expensive for large $n$

        ---

        ### Why Gower for This Dataset?

        1. Categorical features (Gender, Job, Area, CitySize, Investments) = **28% of features** — cannot be
           discarded or naively encoded
        2. One-hot encoding creates sparse high-dimensional artefacts; Gower avoids this
        3. Numerical features span different scales (Age: 18–95 vs percentile [0,1]); Gower normalises by range
        4. Compatible with K-Medoids' precomputed distance matrix interface
        """
    )
    return


# ── Clustering Imports ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2. Clustering Libraries")
    return


@app.cell
def __(mo):
    try:
        import gower
        from sklearn_extra.cluster import KMedoids
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
        )
        from sklearn.manifold import TSNE
        from scipy.spatial.distance import pdist, squareform

        mo.callout(mo.md("✅ All clustering libraries loaded."), kind="success")
    except ImportError as _e:
        mo.callout(
            mo.md(f"❌ Missing: `{_e}`. Run `pip install gower scikit-learn-extra`."),
            kind="danger",
        )
        raise
    return (
        KMedoids,
        TSNE,
        calinski_harabasz_score,
        davies_bouldin_score,
        gower,
        pdist,
        silhouette_score,
        squareform,
    )


@app.cell
def __(categorical_cols, df_clean, mo, numerical_cols):
    df_cluster = df_clean.copy()
    mo.vstack([
        mo.md(f"**Dataset for clustering:** {df_cluster.shape[0]:,} clients × {df_cluster.shape[1]} features"),
        mo.md(f"- Categorical ({len(categorical_cols)}): {', '.join(categorical_cols)}"),
        mo.md(f"- Numerical ({len(numerical_cols)}): {', '.join(numerical_cols)}"),
    ])
    return (df_cluster,)


# ── Distance Metric Comparison ───────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3. Distance Metric Comparison (500-Client Sample)

        We compare pairwise distance distributions on a random 500-client sample
        for efficient visual comparison.
        """
    )
    return


@app.cell
def __(df_cluster, gower, mo, np, numerical_cols, pdist, plt, squareform):
    _n_s = min(500, len(df_cluster))
    _samp = df_cluster.sample(n=_n_s, random_state=42)
    _samp_num = _samp[numerical_cols]

    _dist_euc = squareform(pdist(_samp_num, metric="euclidean"))
    _dist_man = squareform(pdist(_samp_num, metric="cityblock"))
    _dist_gow = gower.gower_matrix(_samp)

    _specs = [
        ("Euclidean\n(Numerical Only)", _dist_euc, "#4A90D9"),
        ("Manhattan\n(Numerical Only)", _dist_man, "#F5A623"),
        ("Gower\n(Mixed-Type) ⭐", _dist_gow, "#7ED321"),
    ]

    fig_dcmp, axes_dcmp = plt.subplots(1, 3, figsize=(18, 5))
    for _ax, (_title, _dm, _col) in zip(axes_dcmp, _specs):
        _upper = _dm[np.triu_indices_from(_dm, k=1)]
        _ax.hist(_upper, bins=50, edgecolor="white", alpha=0.85, color=_col)
        _ax.axvline(_upper.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean={_upper.mean():.3f}")
        _ax.set_title(_title, fontweight="bold")
        _ax.set_xlabel("Distance")
        _ax.set_ylabel("Frequency")
        _ax.legend()
        _ax.grid(True, alpha=0.3)

    plt.suptitle(f"Pairwise Distance Distributions (n={_n_s} sample)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    mo.vstack([
        fig_dcmp,
        mo.callout(
            mo.md("**Gower** is normalized to $[0,1]$ and incorporates all 18 features. "
                  "Euclidean/Manhattan ignore the 5 categorical features (28% of the feature space)."),
            kind="info",
        ),
    ])
    return axes_dcmp, fig_dcmp


# ── Full Gower Matrix ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4. Full Gower Distance Matrix

        For $N = 5{,}000$ clients, the matrix has $\binom{5000}{2} = 12{,}497{,}500$ unique pairs.
        This is a one-time computation cached as `distance_matrix`.

        > *This cell may take 30–60 seconds to run on first execution.*
        """
    )
    return


@app.cell
def __(df_cluster, gower, mo, np, plt):
    distance_matrix = gower.gower_matrix(df_cluster)
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    fig_gdist, ax_gdist = plt.subplots(figsize=(12, 5))
    ax_gdist.hist(_upper, bins=100, edgecolor="white", alpha=0.8, color="#4A90D9")
    ax_gdist.axvline(_upper.mean(), color="red", linestyle="--", linewidth=2,
                     label=f"Mean = {_upper.mean():.3f}")
    ax_gdist.axvline(float(np.median(_upper)), color="green", linestyle="--", linewidth=2,
                     label=f"Median = {float(np.median(_upper)):.3f}")
    ax_gdist.set_xlabel("Gower Distance", fontsize=12)
    ax_gdist.set_ylabel("Frequency", fontsize=12)
    ax_gdist.set_title(
        f"Full Pairwise Gower Distance Distribution ({len(_upper):,} pairs)",
        fontsize=13, fontweight="bold",
    )
    ax_gdist.legend()
    ax_gdist.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md(
            f"**Shape:** {distance_matrix.shape} | "
            f"**Range:** [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}] | "
            f"**Mean:** {distance_matrix.mean():.4f} | "
            f"**Median:** {float(np.median(distance_matrix)):.4f}"
        ),
        fig_gdist,
    ])
    return ax_gdist, distance_matrix, fig_gdist


# ── K-Medoids Theory ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. K-Medoids Algorithm

        ### Why K-Medoids over K-Means?

        | | K-Means | K-Medoids |
        |---|---|---|
        | **Centroid** | Mean vector (may not exist in data) | Actual data point |
        | **Mixed data** | ❌ mean undefined for categoricals | ✅ precomputed distance matrix |
        | **Outliers** | Sensitive | Robust (median-based) |

        The medoid of cluster $C_k$ minimises intra-cluster distance:

        $$m_k = \underset{x_i \in C_k}{\arg\min} \sum_{x_j \in C_k} d(x_i, x_j)$$

        The objective (inertia) is:

        $$\mathcal{L} = \sum_{k=1}^{K} \sum_{x_i \in C_k} d(x_i, m_k)$$

        ---

        ### Validation Metrics — Voting Scheme

        We use **three complementary indices** and take the majority vote for $k$:

        #### Silhouette Coefficient $s(i) \in [-1, +1]$ — **Maximize ↑**

        $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\; b(i)\}}, \quad
          a(i) = \frac{1}{|C_k|-1}\sum_{j \in C_k, j\neq i} d(i,j), \quad
          b(i) = \min_{l \neq k} \frac{1}{|C_l|}\sum_{j \in C_l} d(i,j)$$

        #### Davies-Bouldin Index $\in [0, +\infty)$ — **Minimize ↓**

        $$DB = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i}\!\left(\frac{\sigma_i + \sigma_j}{d(m_i, m_j)}\right)$$

        where $\sigma_k = \frac{1}{|C_k|}\sum_{x \in C_k} d(x, m_k)$ is the average within-cluster distance.

        #### Calinski-Harabasz Index $\in [0, +\infty)$ — **Maximize ↑**

        $$CH = \frac{SS_B\,/\,(K-1)}{SS_W\,/\,(n-K)}, \quad
          SS_B = \sum_{k=1}^K |C_k|\,d^2(m_k, m), \quad
          SS_W = \sum_{k=1}^K \sum_{x \in C_k} d^2(x, m_k)$$

        where $m$ is the global centroid.
        """
    )
    return


# ── Run K-Medoids ─────────────────────────────────────────────────────────────
@app.cell
def __(
    KMedoids,
    calinski_harabasz_score,
    davies_bouldin_score,
    distance_matrix,
    mo,
    np,
    silhouette_score,
):
    k_range = range(3, 7)
    results = {}
    _log_lines = []

    for _k in k_range:
        _km = KMedoids(
            n_clusters=_k,
            metric="precomputed",
            init="k-medoids++",
            max_iter=300,
            random_state=42,
        )
        _clus = _km.fit_predict(distance_matrix)
        _unique, _counts = np.unique(_clus, return_counts=True)

        results[_k] = {
            "model": _km,
            "clusters": _clus,
            "medoid_indices": _km.medoid_indices_,
            "silhouette": float(silhouette_score(distance_matrix, _clus, metric="precomputed")),
            "davies_bouldin": float(davies_bouldin_score(distance_matrix, _clus)),
            "calinski_harabasz": float(calinski_harabasz_score(distance_matrix, _clus)),
            "inertia": float(_km.inertia_),
            "sizes": dict(zip(_unique.tolist(), _counts.tolist())),
        }

        _log_lines.append(
            f"k={_k}: sizes={results[_k]['sizes']}  "
            f"Sil={results[_k]['silhouette']:.4f}  "
            f"DB={results[_k]['davies_bouldin']:.4f}  "
            f"CH={results[_k]['calinski_harabasz']:.2f}"
        )

    mo.md("```\n" + "\n".join(_log_lines) + "\n```")
    return k_range, results


# ── Validation Plots ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 6. Cluster Validation Metrics")
    return


@app.cell
def __(k_range, mo, pd, plt, results):
    summary_df = pd.DataFrame({
        "k": list(k_range),
        "Silhouette": [results[k]["silhouette"] for k in k_range],
        "Davies-Bouldin": [results[k]["davies_bouldin"] for k in k_range],
        "Calinski-Harabasz": [results[k]["calinski_harabasz"] for k in k_range],
    })

    fig_val, axes_val = plt.subplots(1, 3, figsize=(18, 5))
    _specs = [
        ("Silhouette", "Silhouette Coefficient\n(Higher = Better)", "#1f77b4", "max"),
        ("Davies-Bouldin", "Davies-Bouldin Index\n(Lower = Better)", "#ff7f0e", "min"),
        ("Calinski-Harabasz", "Calinski-Harabász Index\n(Higher = Better)", "#2ca02c", "max"),
    ]

    for _ax, (_col, _title, _color, _optim) in zip(axes_val, _specs):
        _ax.plot(summary_df["k"], summary_df[_col], marker="o", linewidth=2,
                 markersize=10, color=_color)
        _best = (summary_df.loc[summary_df[_col].idxmax(), "k"]
                 if _optim == "max"
                 else summary_df.loc[summary_df[_col].idxmin(), "k"])
        _ax.axvline(_best, color="red", linestyle="--", alpha=0.6, label=f"Best k={int(_best)}")
        _ax.set_xlabel("k", fontsize=11)
        _ax.set_ylabel(_col, fontsize=11)
        _ax.set_title(_title, fontsize=12, fontweight="bold")
        _ax.legend()
        _ax.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack([
        fig_val,
        mo.ui.table(summary_df.round(4), selection=None),
    ])
    return axes_val, fig_val, summary_df


# ── Voting Scheme ─────────────────────────────────────────────────────────────
@app.cell
def __(mo, summary_df):
    _best_sil = int(summary_df.loc[summary_df["Silhouette"].idxmax(), "k"])
    _best_db  = int(summary_df.loc[summary_df["Davies-Bouldin"].idxmin(), "k"])
    _best_ch  = int(summary_df.loc[summary_df["Calinski-Harabasz"].idxmax(), "k"])

    _votes = {}
    for _v in [_best_sil, _best_db, _best_ch]:
        _votes[_v] = _votes.get(_v, 0) + 1

    optimal_k = max(_votes, key=lambda x: _votes[x])

    mo.callout(
        mo.md(
            "### 🗳️ Voting Scheme Result\n\n"
            f"| Metric | Winner |\n|---|---|\n"
            f"| Silhouette ↑ | k = {_best_sil} |\n"
            f"| Davies-Bouldin ↓ | k = {_best_db} |\n"
            f"| Calinski-Harabász ↑ | k = {_best_ch} |\n\n"
            f"**→ Optimal k = {optimal_k}** (by majority vote)\n\n"
            "*(k = 3 remains highly valid for business interpretability.)*"
        ),
        kind="success",
    )
    return (optimal_k,)


# ── Visualization: PCA + t-SNE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 7. Cluster Visualization — PCA & t-SNE

        **t-SNE** (t-distributed Stochastic Neighbor Embedding) minimises the
        KL-divergence between pairwise similarity distributions in high- and low-dimensional space:

        $$\mathcal{L} = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

        - $p_{ij}$: Gaussian-kernel similarity in original space
        - $q_{ij}$: Student-t (df=1) kernel similarity in 2D embedding

        > t-SNE is **visualization only** — global distances are not preserved.
        > PCA provides the complementary linear global-structure view.
        """
    )
    return


@app.cell
def __(PCA, df_cluster, mo, numerical_cols, optimal_k, plt, results):
    _clus_opt = results[optimal_k]["clusters"]
    _pal = plt.cm.tab10.colors

    _pca3 = PCA(n_components=2, random_state=42)
    _pca_r = _pca3.fit_transform(df_cluster[numerical_cols])

    fig_pca3, ax_pca3 = plt.subplots(figsize=(10, 7))
    for _c in range(optimal_k):
        _mask = (_clus_opt == _c)
        ax_pca3.scatter(
            _pca_r[_mask, 0], _pca_r[_mask, 1],
            s=12, alpha=0.5, color=_pal[_c], label=f"Cluster {_c}",
        )
    ax_pca3.set_title(f"PCA 2D — K-Medoids (k={optimal_k})", fontsize=13, fontweight="bold")
    ax_pca3.set_xlabel(f"PC1 ({_pca3.explained_variance_ratio_[0]:.2%} var.)")
    ax_pca3.set_ylabel(f"PC2 ({_pca3.explained_variance_ratio_[1]:.2%} var.)")
    ax_pca3.legend(markerscale=3)
    ax_pca3.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md(f"**Cluster sizes (k={optimal_k}):** {results[optimal_k]['sizes']}"),
        fig_pca3,
    ])
    return ax_pca3, fig_pca3


@app.cell(hide_code=True)
def __(mo):
    mo.md("### t-SNE Visualization (1,000-Client Sample)")
    return


@app.cell
def __(TSNE, distance_matrix, mo, np, optimal_k, plt, results):
    _n_s = min(1000, distance_matrix.shape[0])
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(distance_matrix.shape[0], size=_n_s, replace=False)
    _dm_s = distance_matrix[np.ix_(_idx, _idx)]
    _clus_s = results[optimal_k]["clusters"][_idx]
    _pal = plt.cm.tab10.colors

    _tsne = TSNE(n_components=2, metric="precomputed", perplexity=30, n_iter=1000, random_state=42)
    _tsne_r = _tsne.fit_transform(_dm_s)

    fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
    for _c in range(optimal_k):
        _mask = (_clus_s == _c)
        ax_tsne.scatter(_tsne_r[_mask, 0], _tsne_r[_mask, 1],
                        s=15, alpha=0.6, color=_pal[_c], label=f"Cluster {_c}")
    ax_tsne.set_title(
        f"t-SNE — K-Medoids Clusters (k={optimal_k}, n={_n_s} sample)",
        fontsize=13, fontweight="bold",
    )
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.legend(markerscale=3)
    ax_tsne.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_tsne
    return ax_tsne, fig_tsne


# ── Cluster Profiling ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 8. Cluster Profiling — Customer Personas

        For each cluster we report:
        - **Numerical:** mean feature values (percentiles already in [0,1])
        - **Categorical:** mode (most frequent category)
        - **Size** as absolute count and % of total
        """
    )
    return


@app.cell
def __(categorical_cols, df_cluster, mo, numerical_cols, optimal_k, pd, plt, results):
    _clus = results[optimal_k]["clusters"]
    df_profiled = df_cluster.copy()
    df_profiled["Cluster"] = _clus

    _job_map  = {1:"Unemployed", 2:"Employee", 3:"Manager", 4:"Entrepreneur", 5:"Retired"}
    _inv_map  = {1:"None", 2:"Lump Sum", 3:"Cap. Accum."}
    _area_map = {1:"North", 2:"Center", 3:"South/Islands"}
    _gen_map  = {0:"Male", 1:"Female"}
    _city_map = {1:"Small", 2:"Medium", 3:"Large"}

    _rows = []
    for _c in range(optimal_k):
        _g = df_profiled[df_profiled["Cluster"] == _c]
        _row = {
            "Cluster": _c,
            "N": len(_g),
            "%": f"{len(_g)/len(df_profiled)*100:.1f}%",
        }
        for _col in numerical_cols:
            _row[_col] = f"{_g[_col].mean():.3f}"
        _row["Job (mode)"]  = _job_map.get(int(_g["Job"].mode()[0]), "?")
        _row["Gender"]      = _gen_map.get(int(_g["Gender"].mode()[0]), "?")
        _row["Area"]        = _area_map.get(int(_g["Area"].mode()[0]), "?")
        _row["Investments"] = _inv_map.get(int(_g["Investments"].mode()[0]), "?")
        _rows.append(_row)

    _profile_df = pd.DataFrame(_rows)

    # Radar chart
    _radar_cols = numerical_cols
    _angles = [n / len(_radar_cols) * 2 * 3.14159 for n in range(len(_radar_cols))]
    _angles += _angles[:1]
    _pal = plt.cm.tab10.colors

    fig_radar, _axr = plt.subplots(
        1, optimal_k, figsize=(5 * optimal_k, 5), subplot_kw=dict(polar=True)
    )
    if optimal_k == 1:
        _axr = [_axr]

    for _c, _ax in enumerate(_axr):
        _g = df_profiled[df_profiled["Cluster"] == _c]
        _vals = _g[_radar_cols].mean().tolist() + [_g[_radar_cols].mean().tolist()[0]]
        _ax.plot(_angles, _vals, "o-", linewidth=2, color=_pal[_c])
        _ax.fill(_angles, _vals, alpha=0.25, color=_pal[_c])
        _ax.set_xticks(_angles[:-1])
        _ax.set_xticklabels(_radar_cols, size=7)
        _ax.set_ylim(0, 1)
        _ax.set_title(f"Cluster {_c}\n(n={len(_g)})", fontweight="bold", pad=15)

    plt.suptitle("Radar Charts — Mean Numerical Feature Values per Cluster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Cluster Summary"),
        mo.ui.table(_profile_df, selection=None),
        mo.md("### Radar Profiles"),
        fig_radar,
    ])
    return df_profiled, fig_radar


# ── Final Notes ───────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 9. Business Interpretation & Methodological Comparison

        > **Key finding:** While quantitative metrics often point to k=4 as technically
        > optimal, **k=3 offers superior business interpretability** — it maps cleanly
        > onto a Low / Medium / High value segmentation with actionable marketing implications.

        ### Recommended Personas (k=3)

        | Persona | Profile | Marketing Focus |
        |---------|---------|-----------------|
        | 🟢 **Premium Investor** | High Wealth & Income, high FinEdu, ESG-oriented | Sustainable portfolios, premium advisory |
        | 🟡 **Balanced Family** | Mid Income, family-focused, moderate Saving | Life insurance, education savings plans |
        | 🔴 **Developing Client** | Lower Income, higher Debt, lower FinEdu | Financial literacy programs, debt restructuring |

        ---

        ### Methodological Comparison

        | Aspect | Tommy — K-Prototypes | Giulia — K-Medoids + Gower |
        |--------|---------------------|----------------------------|
        | Distance metric | Euclidean + Hamming (γ-weighted) | Gower (range-normalised, equal weight) |
        | Centroid type | Computed (mean + mode) | Actual data point (medoid) |
        | Mixed-data handling | Requires tuning γ | Automatic |
        | Outlier robustness | Moderate | Higher (median-based) |
        | Library | `kmodes` | `sklearn_extra` + `gower` |
        | Output | Cluster centres | Representative clients |
        """
    )
    return


if __name__ == "__main__":
    app.run()
