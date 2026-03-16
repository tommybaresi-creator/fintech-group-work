import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Bank Clients — EDA & K-Medoids Clustering")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _load_svg(name: str) -> str:
    """Load an SVG from the assets/ folder by stem name. Returns empty string if not found."""
    from pathlib import Path
    p = Path(__file__).parent / "assets" / f"{name}.svg"
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from pathlib import Path
    import warnings
    import marimo as mo

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    matplotlib.rcParams["figure.dpi"] = 110
    np.random.seed(42)

    return Path, go, make_subplots, matplotlib, mo, np, pd, plt, px, sns, warnings


# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Bank Client Segmentation — EDA & K-Medoids Clustering

        **Objective:** Implement robust clustering for mixed-type financial data
        (5,000 clients, 18 features) to identify distinct customer segments for
        targeted marketing strategies.

        **Approach:** K-Medoids algorithm with Gower distance metric and validation
        through multiple internal cluster quality indices.

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
        | `ESG` | Numerical | ESG investing propensity $\in [0,1]$ |
        | `Digital` | Numerical | Digital channel adoption $\in [0,1]$ |
        | `BankFriend` | Numerical | Bank relationship orientation $\in [0,1]$ |
        | `LifeStyle` | Numerical | Lifestyle spending index $\in [0,1]$ |
        | `Luxury` | Numerical | Luxury goods propensity $\in [0,1]$ |
        | `Saving` | Numerical | Savings behaviour $\in [0,1]$ |
        | `Investments` | Categorical | 1=None, 2=Lump Sum, 3=Capital Accumulation |
        """
    )
    return


# ===========================================================================
# PART 1 — EDA
# ===========================================================================
@app.cell(hide_code=True)
def __(mo):
    mo.md("---\n# Part 1 — Exploratory Data Analysis & Outlier Detection")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1. Data Loading & Sanity Checks")
    return


@app.cell
def __(Path, pd):
    _data_path = Path(__file__).parent.parent / "Data" / "Dataset1_BankClients.xlsx"
    df_raw = pd.read_excel(_data_path)
    print(f"Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns from {_data_path.name}")
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
        mo.md(f"**Shape after dropping ID:** {df.shape[0]:,} rows x {df.shape[1]} columns  |  **Duplicates:** {_dupes}"),
        mo.ui.table(_tbl, selection=None),
    ])
    return (df,)


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


# -- Numerical distributions ------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2. Univariate Analysis — Numerical Features

        Histograms and boxplots for each numerical feature.
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
def __(df, numerical_cols, px):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _msp

    _n = len(numerical_cols)
    _fig = _msp(rows=_n, cols=2,
                subplot_titles=[f"{c} — hist" if i % 2 == 0 else f"{c} — box"
                                for c in numerical_cols for i in range(2)],
                vertical_spacing=0.02)

    _colors = px.colors.qualitative.Plotly
    for _i, _col in enumerate(numerical_cols):
        _r = _i + 1
        _c = _colors[_i % len(_colors)]
        _fig.add_trace(_go.Histogram(x=df[_col], marker_color=_c,
                                     opacity=0.75, showlegend=False,
                                     name=_col), row=_r, col=1)
        _fig.add_trace(_go.Box(x=df[_col], marker_color=_c,
                               showlegend=False, name=_col), row=_r, col=2)

    _fig.update_layout(height=220 * _n, title_text="Numerical Features — Distributions & Boxplots",
                       title_font_size=16)
    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Bound Checks

        - **Age** must be in $(0, 120]$
        - **FamilySize** must be $\geq 1$
        - Normalized features must lie in $[0, 1]$
        """
    )
    return


@app.cell
def __(df, mo, normalized_vars, pd):
    _rows = []
    _rows.append({"Feature": "Age",
                  "Min": round(float(df["Age"].min()), 2),
                  "Max": round(float(df["Age"].max()), 2),
                  "Expected": "[0, 120]",
                  "OK?": "Yes" if 0 <= df["Age"].min() and df["Age"].max() <= 120 else "No"})
    _rows.append({"Feature": "FamilySize",
                  "Min": int(df["FamilySize"].min()),
                  "Max": int(df["FamilySize"].max()),
                  "Expected": ">= 1",
                  "OK?": "Yes" if df["FamilySize"].min() >= 1 else "No"})
    for _col in normalized_vars:
        _mn, _mx = float(df[_col].min()), float(df[_col].max())
        _rows.append({"Feature": _col,
                      "Min": round(_mn, 4),
                      "Max": round(_mx, 4),
                      "Expected": "[0, 1]",
                      "OK?": "Yes" if _mn >= 0 and _mx <= 1 else "No"})
    _has_issues = any(r["OK?"] == "No" for r in _rows)
    mo.vstack([
        mo.callout(
            mo.md("Some features are out of expected bounds." if _has_issues else "All feature bounds look healthy."),
            kind="warn" if _has_issues else "success",
        ),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


# -- Categorical distributions ----------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3. Univariate Analysis — Categorical Features")
    return


@app.cell
def __(categorical_cols, df, mo, px):
    _labels_map = {
        "Gender":      {0: "Male", 1: "Female"},
        "Job":         {1: "Unemployed", 2: "Employee", 3: "Manager", 4: "Entrepreneur", 5: "Retired"},
        "Area":        {1: "North", 2: "Center", 3: "South/Islands"},
        "CitySize":    {1: "Small", 2: "Medium", 3: "Large"},
        "Investments": {1: "None", 2: "Lump Sum", 3: "Cap. Accum."},
    }
    _figs = []
    for _col in categorical_cols:
        _vc = df[_col].value_counts().sort_index()
        _labels = [_labels_map[_col].get(int(v), str(v)) for v in _vc.index]
        _fig = px.bar(x=_labels, y=_vc.values,
                      title=f"Distribution of {_col}",
                      labels={"x": _col, "y": "Count"},
                      color_discrete_sequence=px.colors.qualitative.Vivid)
        _fig.update_layout(height=350, margin=dict(t=50, b=30))
        _figs.append(_fig)

    mo.hstack(_figs, wrap=True)
    return


# -- Correlation heatmap ----------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4. Bivariate Analysis — Pearson Correlation

        $$r_{XY} = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i-\bar{Y})^2}}$$

        Pairs with $|r| > 0.8$ may provide duplicate information in
        distance-based clustering.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols, pd, px):
    corr_matrix = df[numerical_cols].corr()
    _fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Pearson Correlation Heatmap — Numerical Features")
    _fig_corr.update_layout(height=650)

    _threshold = 0.8
    _pairs = []
    for _i in range(len(corr_matrix.columns)):
        for _j in range(_i):
            _v = corr_matrix.iloc[_i, _j]
            if abs(_v) > _threshold:
                _pairs.append({"Feature A": corr_matrix.columns[_i],
                                "Feature B": corr_matrix.columns[_j],
                                "Pearson r": round(float(_v), 4)})

    mo.vstack([
        mo.ui.plotly(_fig_corr),
        mo.callout(
            mo.md(f"**{len(_pairs)} pair(s)** with |r| > {_threshold}." if _pairs else "No pairs above threshold."),
            kind="warn" if _pairs else "success",
        ),
        mo.ui.table(pd.DataFrame(_pairs if _pairs else [{"Result": "None found"}]), selection=None),
    ])
    return (corr_matrix,)


# -- Domain anomalies -------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. Domain Knowledge Anomaly Detection

        | Cohort | Rule |
        |--------|------|
        | Young Retirees | Age < 50 and Job = Retired |
        | Working Minors | Age < 18 and Job in {Employee, Manager, Entrepreneur} |
        | Rich Unemployed | Job = Unemployed and Income > 90th pct |
        | Wealthy, No Investments | Investments = 1 and Wealth > 95th pct |
        | Young Large Families | Age <= 20 and FamilySize >= 4 |
        """
    )
    return


@app.cell
def __(df, mo, pd):
    _cohorts = {
        "Young Retirees (Age < 50, Job=Retired)":    ((df["Age"] < 50) & (df["Job"] == 5)).sum(),
        "Working Minors (Age < 18, Job in {2,3,4})": ((df["Age"] < 18) & (df["Job"].isin([2,3,4]))).sum(),
        "Rich Unemployed (Job=1 & Income > 0.9)":    ((df["Job"] == 1) & (df["Income"] > 0.9)).sum(),
        "Wealthy No Invest (Inv=1 & Wealth > 0.95)": ((df["Investments"] == 1) & (df["Wealth"] > 0.95)).sum(),
        "Young Large Fam (Age<=20 & FamSz>=4)":      ((df["Age"] <= 20) & (df["FamilySize"] >= 4)).sum(),
    }
    _tbl = pd.DataFrame([{"Cohort": k, "Count": v, "%": f"{v/len(df)*100:.2f}%"}
                          for k, v in _cohorts.items()])
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


# -- Isolation Forest -------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 6. Multivariate Outlier Detection — Isolation Forest

        Isolation Forest partitions the feature space randomly. Outliers require
        fewer splits to isolate. The anomaly score is:

        $$\text{score}(x, n) = 2^{-\,\dfrac{E[h(x)]}{c(n)}}$$

        where $h(x)$ = average path length, $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$
        is the expected path length under no structure ($H$ = harmonic number).

        We flag the top **1%** as multivariate outliers (`contamination=0.01`).
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols):
    from sklearn.ensemble import IsolationForest

    _iso = IsolationForest(contamination=0.01, random_state=42)
    _labels = _iso.fit_predict(df[numerical_cols])

    df_clean = df[_labels == 1].copy().reset_index(drop=True)
    _n_out = int((_labels == -1).sum())

    mo.callout(
        mo.md(f"**{_n_out} multivariate outliers** flagged ({_n_out/len(df)*100:.2f}%). "
              f"Clean dataset: **{len(df_clean):,} clients**."),
        kind="warn",
    )
    return IsolationForest, df_clean


# ===========================================================================
# PART 2 — K-MEDOIDS CLUSTERING
# ===========================================================================
@app.cell(hide_code=True)
def __(mo):
    mo.md("---\n# Part 2 — K-Medoids Clustering")
    return


# ---------------------------------------------------------------------------
# SVG: Distance Metric Selection Guide
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1. Theoretical Background: Distance Metrics for Mixed-Type Data")
    return


@app.cell(hide_code=True)
def __(mo):
    _svg = _load_svg("distance_metric_selection_guide")
    if _svg:
        mo.vstack([
            mo.md("### Distance Metric Selection Guide"),
            mo.Html(_svg),
        ])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Our dataset contains **mixed-type features** (5 categorical + 13 numerical),
        requiring careful distance metric selection.

        ---

        ### Euclidean Distance (L2)

        $$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

        **Pros:** Computationally efficient; intuitive geometric interpretation;
        well-established theoretical properties.

        **Cons:** Cannot handle categorical variables (Job, Gender, Area, etc.);
        sensitive to outliers (squared differences amplify extreme values);
        assumes continuous metric space.

        ---

        ### Manhattan Distance (L1)

        $$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

        **Pros:** More robust than Euclidean in high dimensions;
        foundation for composite metrics (Gower, Sorensen).

        **Cons:** Still requires numerical data (incompatible with categorical features);
        assumes axis-aligned independence (unrealistic for correlated financial variables).

        ---

        ### Jaccard Distance (categorical)

        $$d(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}$$

        **Pros:** Excellent for binary/categorical features; measures set overlap semantically.

        **Cons:** Ignores numerical magnitudes (Income, Wealth differences lost);
        requires binarization of numerical data.

        ---

        ### Gower Distance (mixed-type) — Selected

        $$d_{\text{Gower}}(x, y) = \frac{1}{p}\sum_{j=1}^{p}\delta_j(x,y)$$

        where per-feature contribution $\delta_j$ is:

        $$\delta_j = \begin{cases}
        \dfrac{|x_j - y_j|}{R_j} & \text{numerical (normalized Manhattan, } R_j = \max_j - \min_j\text{)} \\[6pt]
        \mathbf{1}[x_j \neq y_j] & \text{categorical (Hamming)}
        \end{cases}$$

        **Pros:** Handles mixed categorical/numerical data natively; automatic normalization
        prevents scale bias; equal weighting of all features; range-normalized to $[0,1]$.

        **Cons:** Computationally expensive — $O(n^2)$ distance matrix; assumes equal feature importance.

        ---

        ### Selection Rationale

        For this dataset:
        - Categorical features (Gender, Job, Area, CitySize, Investments) represent **~28%** of features
        - Numerical features span different scales (Age: 19–95 vs normalized [0,1] scores)
        - Gower distance is optimal as it:
          1. Handles mixed types without preprocessing artefacts (one-hot encoding creates sparsity)
          2. Normalizes numerical features automatically (prevents Income/Wealth dominance)
          3. Treats categorical differences uniformly
        """
    )
    return


# ---------------------------------------------------------------------------
# SVG: Weighted Gower Decision
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    _svg = _load_svg("weighted_gower_decision")
    if _svg:
        mo.vstack([
            mo.md("### Weighted Gower — Decision Summary"),
            mo.Html(_svg),
        ])
    return


# ---------------------------------------------------------------------------
# Data Loading for Clustering
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2. Data Loading and Preparation")
    return


@app.cell
def __(mo):
    try:
        import gower
        from sklearn_extra.cluster import KMedoids
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from scipy.spatial.distance import pdist, squareform
        import umap
        mo.callout(mo.md("All clustering libraries loaded."), kind="success")
    except ImportError as _e:
        mo.callout(mo.md(f"Missing library: {_e}. Run `uv add gower scikit-learn-extra umap-learn`."), kind="danger")
        raise
    return (
        KMedoids,
        PCA,
        TSNE,
        calinski_harabasz_score,
        davies_bouldin_score,
        gower,
        pdist,
        silhouette_score,
        squareform,
        umap,
    )


@app.cell
def __(categorical_cols, df_clean, mo, numerical_cols):
    df_cluster = df_clean.copy()
    mo.vstack([
        mo.md(f"**Dataset for clustering:** {df_cluster.shape[0]:,} clients x {df_cluster.shape[1]} features"),
        mo.md(f"Categorical ({len(categorical_cols)}): {', '.join(categorical_cols)}"),
        mo.md(f"Numerical ({len(numerical_cols)}): {', '.join(numerical_cols)}"),
    ])
    return (df_cluster,)


# ---------------------------------------------------------------------------
# Distance Metric Comparison
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3. Distance Metric Comparison

        Comparing pairwise distance distributions on a 500-client random sample
        across Euclidean, Manhattan, and Gower to validate metric selection.
        """
    )
    return


@app.cell
def __(df_cluster, go, gower, make_subplots, mo, np, numerical_cols, pdist, squareform):
    _samp = df_cluster.sample(n=min(500, len(df_cluster)), random_state=42)
    _samp_num = _samp[numerical_cols]

    _dist_euc = squareform(pdist(_samp_num, metric="euclidean"))
    _dist_man = squareform(pdist(_samp_num, metric="cityblock"))
    _dist_gow = gower.gower_matrix(_samp)

    _specs = [
        ("Euclidean (Numerical Only)", _dist_euc, "#4A90D9"),
        ("Manhattan (Numerical Only)", _dist_man, "#F5A623"),
        ("Gower (Mixed-Type)", _dist_gow, "#7ED321"),
    ]

    _fig = make_subplots(rows=1, cols=3, subplot_titles=[s[0] for s in _specs])
    for _ci, (_title, _dm, _col) in enumerate(_specs):
        _upper = _dm[np.triu_indices_from(_dm, k=1)]
        _fig.add_trace(go.Histogram(x=_upper, marker_color=_col, opacity=0.8,
                                    name=_title, showlegend=False,
                                    nbinsx=50), row=1, col=_ci + 1)
        _fig.add_vline(x=float(_upper.mean()), line_dash="dash", line_color="red",
                       annotation_text=f"Mean={_upper.mean():.3f}",
                       row=1, col=_ci + 1)

    _fig.update_layout(height=400, title_text="Pairwise Distance Distributions (n=500 sample)",
                       title_font_size=14)
    _fig.update_xaxes(title_text="Distance")
    _fig.update_yaxes(title_text="Frequency", col=1)

    mo.vstack([
        mo.ui.plotly(_fig),
        mo.callout(
            mo.md("Gower distance is range-normalized [0,1] and incorporates categorical features "
                  "(Gender, Job, Area, CitySize, Investments), providing a comprehensive similarity measure. "
                  "Euclidean/Manhattan omit 28% of the feature space."),
            kind="info",
        ),
    ])
    return


# ---------------------------------------------------------------------------
# SVG: Gower Kmedoids Pipeline
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    _svg = _load_svg("gower_kmedoids_pipeline")
    if _svg:
        mo.vstack([
            mo.md("### Gower + K-Medoids Pipeline"),
            mo.Html(_svg),
        ])
    return


# ---------------------------------------------------------------------------
# Full Gower Distance Matrix
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4. Gower Distance Matrix Computation (Full Dataset)

        For $N = 5{,}000$ clients this yields $\binom{5000}{2} = 12{,}497{,}500$ unique pairs.

        > This cell may take 30–60 seconds on first execution.
        """
    )
    return


@app.cell
def __(df_cluster, go, gower, mo, np):
    distance_matrix = gower.gower_matrix(df_cluster)

    print(f"\nDistance matrix: {distance_matrix.shape}")
    print(f"Range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    print(f"Mean: {distance_matrix.mean():.4f}")
    print(f"Median: {np.median(distance_matrix):.4f}")

    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=_upper, nbinsx=100, marker_color="steelblue",
                                opacity=0.8, name="Gower distances"))
    _fig.add_vline(x=float(_upper.mean()), line_dash="dash", line_color="red",
                   annotation_text=f"Mean = {_upper.mean():.3f}")
    _fig.add_vline(x=float(np.median(_upper)), line_dash="dash", line_color="green",
                   annotation_text=f"Median = {float(np.median(_upper)):.3f}")
    _fig.update_layout(
        title=f"Pairwise Gower Distance Distribution ({len(_upper):,} pairs)",
        xaxis_title="Gower Distance",
        yaxis_title="Frequency",
        height=400,
    )
    mo.ui.plotly(_fig)
    return (distance_matrix,)


# ---------------------------------------------------------------------------
# SVG: Kmedoids Explainer
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    _svg = _load_svg("kmedoids_explainer")
    if _svg:
        mo.vstack([
            mo.md("### K-Medoids Algorithm Explainer"),
            mo.Html(_svg),
        ])
    return


# ---------------------------------------------------------------------------
# K-Medoids Theory
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. K-Medoids Clustering with Optimal k Selection

        ### 5.1 Algorithm Rationale

        **Why K-Medoids over K-Means?**
        - K-Means requires computing centroids as mean vectors — undefined for categorical features
        - K-Medoids selects actual data points as cluster representatives (medoids)
        - Compatible with precomputed distance matrices (Gower)
        - More robust to outliers (median-based vs mean-based)

        The medoid of cluster $C_k$ minimises intra-cluster distance:

        $$m_k = \underset{x_i \in C_k}{\arg\min} \sum_{x_j \in C_k} d(x_i, x_j)$$

        ---

        ### 5.2 Cluster Validation Metrics

        We employ a **voting scheme** across three complementary metrics:

        **Silhouette Coefficient** $\in [-1, 1]$ — maximize

        $$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

        where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance.
        Values near $+1$ indicate well-separated clusters.

        **Davies-Bouldin Index** $\in [0, \infty)$ — minimize

        $$DB = \frac{1}{K}\sum_{i=1}^{K}\max_{j \neq i}\!\left(\frac{\sigma_i + \sigma_j}{d(m_i, m_j)}\right)$$

        where $\sigma_k$ = average within-cluster distance. Lower values indicate better clustering.

        **Calinski-Harabasz Index** $\in [0, \infty)$ — maximize

        $$CH = \frac{SS_B\,/\,(K-1)}{SS_W\,/\,(n-K)}$$

        where $SS_B$ = between-cluster, $SS_W$ = within-cluster sum of squares.
        Higher values indicate denser, well-separated clusters.
        """
    )
    return


# ---------------------------------------------------------------------------
# Run K-Medoids
# ---------------------------------------------------------------------------
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

    print("K-Medoids clustering for k in {3, 4, 5, 6}\n")

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

        print(f"k={_k}: sizes={results[_k]['sizes']},  "
              f"Sil={results[_k]['silhouette']:.4f},  "
              f"DB={results[_k]['davies_bouldin']:.4f},  "
              f"CH={results[_k]['calinski_harabasz']:.2f}")

    return k_range, results


# ---------------------------------------------------------------------------
# Performance Metrics Plots
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("## 6. Cluster Validation — Performance Metrics")
    return


@app.cell
def __(go, k_range, make_subplots, mo, pd, results):
    summary_df = pd.DataFrame({
        "k": list(k_range),
        "Silhouette": [results[k]["silhouette"] for k in k_range],
        "Davies-Bouldin": [results[k]["davies_bouldin"] for k in k_range],
        "Calinski-Harabasz": [results[k]["calinski_harabasz"] for k in k_range],
    })

    _ks = summary_df["k"].tolist()

    _best_k_sil = int(summary_df.loc[summary_df["Silhouette"].idxmax(), "k"])
    _best_k_db  = int(summary_df.loc[summary_df["Davies-Bouldin"].idxmin(), "k"])
    _best_k_ch  = int(summary_df.loc[summary_df["Calinski-Harabasz"].idxmax(), "k"])

    _fig = make_subplots(rows=1, cols=3,
                         subplot_titles=[
                             "Silhouette Coefficient<br>(Higher = Better)",
                             "Davies-Bouldin Index<br>(Lower = Better)",
                             "Calinski-Harabasz Index<br>(Higher = Better)",
                         ])

    # Silhouette
    _fig.add_trace(go.Scatter(x=_ks, y=summary_df["Silhouette"].tolist(),
                               mode="lines+markers", marker=dict(size=10),
                               line=dict(color="#1f77b4", width=2),
                               name="Silhouette"), row=1, col=1)
    _fig.add_vline(x=_best_k_sil, line_dash="dash", line_color="red",
                   annotation_text=f"Optimal k={_best_k_sil}", row=1, col=1)

    # Davies-Bouldin
    _fig.add_trace(go.Scatter(x=_ks, y=summary_df["Davies-Bouldin"].tolist(),
                               mode="lines+markers", marker=dict(size=10),
                               line=dict(color="#ff7f0e", width=2),
                               name="Davies-Bouldin"), row=1, col=2)
    _fig.add_vline(x=_best_k_db, line_dash="dash", line_color="red",
                   annotation_text=f"Optimal k={_best_k_db}", row=1, col=2)

    # Calinski-Harabasz
    _fig.add_trace(go.Scatter(x=_ks, y=summary_df["Calinski-Harabasz"].tolist(),
                               mode="lines+markers", marker=dict(size=10),
                               line=dict(color="#2ca02c", width=2),
                               name="Calinski-Harabasz"), row=1, col=3)
    _fig.add_vline(x=_best_k_ch, line_dash="dash", line_color="red",
                   annotation_text=f"Optimal k={_best_k_ch}", row=1, col=3)

    _fig.update_xaxes(title_text="k (number of clusters)")
    _fig.update_layout(height=420, showlegend=False,
                       title_text="Cluster Validation Metrics", title_font_size=15)

    mo.vstack([
        mo.ui.plotly(_fig),
        mo.ui.table(summary_df.round(4), selection=None),
    ])
    return summary_df,


# -- Voting Scheme ----------------------------------------------------------
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
            "### Voting Scheme Result\n\n"
            f"| Metric | Optimal k |\n|---|---|\n"
            f"| Silhouette (maximize) | k = {_best_sil} |\n"
            f"| Davies-Bouldin (minimize) | k = {_best_db} |\n"
            f"| Calinski-Harabasz (maximize) | k = {_best_ch} |\n\n"
            f"**Optimal k by majority vote: k = {optimal_k}**\n\n"
            "While the numerical results suggest that k=4 may be technically optimal "
            "according to the majority of indices, a choice of k=3 remains highly valid. "
            "In a business context, k=3 often provides a more interpretable segmentation "
            "(e.g., Low, Medium, High value clients)."
        ),
        kind="success",
    )
    return (optimal_k,)


# ---------------------------------------------------------------------------
# SVG: PCA Gower PAM Compatibility
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    _svg = _load_svg("pca_gower_pam_compatibility")
    if _svg:
        mo.vstack([
            mo.md("### PCA / Gower / PAM Compatibility"),
            mo.Html(_svg),
        ])
    return


# ---------------------------------------------------------------------------
# Cluster Visualization — PCA
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 7. Cluster Visualization

        ### 7.1 PCA Projection

        PCA finds orthogonal directions of maximum variance:

        $$\mathbf{W}^* = \underset{\mathbf{W}:\mathbf{W}^\top\mathbf{W}=\mathbf{I}}{\arg\max}\;\operatorname{tr}\!\left(\mathbf{W}^\top \mathbf{\Sigma} \mathbf{W}\right), \quad
        \text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$
        """
    )
    return


@app.cell
def __(PCA, df_cluster, mo, numerical_cols, optimal_k, px, results):
    _clus = results[optimal_k]["clusters"]
    _pca = PCA(n_components=2, random_state=42)
    _pca_r = _pca.fit_transform(df_cluster[numerical_cols])
    _ev1, _ev2 = _pca.explained_variance_ratio_

    import pandas as _pd_local
    _pca_df = _pd_local.DataFrame({
        "PC1": _pca_r[:, 0],
        "PC2": _pca_r[:, 1],
        "Cluster": [f"Cluster {c}" for c in _clus],
    })

    _fig_pca = px.scatter(
        _pca_df, x="PC1", y="PC2", color="Cluster",
        title=f"PCA 2D — K-Medoids Clusters (k={optimal_k})",
        labels={"PC1": f"PC1 ({_ev1:.2%} var.)", "PC2": f"PC2 ({_ev2:.2%} var.)"},
        opacity=0.55,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    _fig_pca.update_traces(marker=dict(size=4))
    _fig_pca.update_layout(height=550)

    mo.vstack([
        mo.md(f"**Total variance explained by PC1+PC2:** {(_ev1+_ev2):.2%}"),
        mo.ui.plotly(_fig_pca),
    ])
    return


# ---------------------------------------------------------------------------
# Cluster Visualization — UMAP
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 7.2 UMAP Projection

        **UMAP** (Uniform Manifold Approximation and Projection) learns a low-dimensional
        representation by preserving the topological structure of the data manifold.
        It minimises the cross-entropy between fuzzy topological representations:

        $$\mathcal{L} = \sum_{e \in E} \left[w_h(e)\log\frac{w_h(e)}{w_l(e)} + (1-w_h(e))\log\frac{1-w_h(e)}{1-w_l(e)}\right]$$

        where $w_h$ = high-dimensional edge weights (fuzzy simplicial set), 
        $w_l$ = low-dimensional edge weights.

        UMAP is run on the **precomputed Gower distance matrix** (`metric='precomputed'`),
        making it fully consistent with the K-Medoids clustering.

        > UMAP on the full distance matrix may take ~1–2 minutes.
        """
    )
    return


@app.cell
def __(distance_matrix, mo, np, optimal_k, px, results, umap):
    _n_s = min(2000, distance_matrix.shape[0])
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(distance_matrix.shape[0], size=_n_s, replace=False)
    _dm_s = distance_matrix[np.ix_(_idx, _idx)]
    _clus_s = results[optimal_k]["clusters"][_idx]

    _reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    _umap_r = _reducer.fit_transform(_dm_s)

    import pandas as _pd_local2
    _umap_df = _pd_local2.DataFrame({
        "UMAP1": _umap_r[:, 0],
        "UMAP2": _umap_r[:, 1],
        "Cluster": [f"Cluster {c}" for c in _clus_s],
    })

    _fig_umap = px.scatter(
        _umap_df, x="UMAP1", y="UMAP2", color="Cluster",
        title=f"UMAP — K-Medoids Clusters (k={optimal_k}, n={_n_s} sample, Gower metric)",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    _fig_umap.update_traces(marker=dict(size=4))
    _fig_umap.update_layout(height=580)

    mo.ui.plotly(_fig_umap)
    return


# ---------------------------------------------------------------------------
# Cluster Visualization — UMAP 3D
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("### 7.3 UMAP 3D Projection")
    return


@app.cell
def __(distance_matrix, mo, np, optimal_k, px, results, umap):
    _n_s3 = min(1500, distance_matrix.shape[0])
    _rng3 = np.random.default_rng(99)
    _idx3 = _rng3.choice(distance_matrix.shape[0], size=_n_s3, replace=False)
    _dm_s3 = distance_matrix[np.ix_(_idx3, _idx3)]
    _clus_s3 = results[optimal_k]["clusters"][_idx3]

    _reducer3 = umap.UMAP(
        n_components=3,
        metric="precomputed",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    _umap_r3 = _reducer3.fit_transform(_dm_s3)

    import pandas as _pd_local3
    _umap_df3 = _pd_local3.DataFrame({
        "UMAP1": _umap_r3[:, 0],
        "UMAP2": _umap_r3[:, 1],
        "UMAP3": _umap_r3[:, 2],
        "Cluster": [f"Cluster {c}" for c in _clus_s3],
    })

    _fig3d = px.scatter_3d(
        _umap_df3, x="UMAP1", y="UMAP2", z="UMAP3", color="Cluster",
        title=f"UMAP 3D — K-Medoids Clusters (k={optimal_k}, n={_n_s3} sample)",
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    _fig3d.update_traces(marker=dict(size=3))
    _fig3d.update_layout(height=650)

    mo.ui.plotly(_fig3d)
    return


# ---------------------------------------------------------------------------
# Cluster Profiling
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 8. Cluster Profiling — Customer Personas

        For each cluster we compute:
        - **Numerical features:** mean value (percentiles in [0,1])
        - **Categorical features:** mode (most frequent value)
        - **Cluster size** as count and % of total
        """
    )
    return


@app.cell
def __(categorical_cols, df_cluster, go, make_subplots, mo, numerical_cols, optimal_k, pd, px, results):
    import pandas as _pd_prof
    _clus = results[optimal_k]["clusters"]
    _df_p = df_cluster.copy()
    _df_p["Cluster"] = _clus

    _job_map  = {1:"Unemployed", 2:"Employee", 3:"Manager", 4:"Entrepreneur", 5:"Retired"}
    _inv_map  = {1:"None", 2:"Lump Sum", 3:"Cap. Accum."}
    _area_map = {1:"North", 2:"Center", 3:"South/Islands"}
    _gen_map  = {0:"Male", 1:"Female"}
    _city_map = {1:"Small", 2:"Medium", 3:"Large"}

    _rows = []
    for _c in range(optimal_k):
        _g = _df_p[_df_p["Cluster"] == _c]
        _row = {"Cluster": _c, "N": len(_g), "%": f"{len(_g)/len(_df_p)*100:.1f}%"}
        for _col in numerical_cols:
            _row[_col] = f"{_g[_col].mean():.3f}"
        _row["Job (mode)"]  = _job_map.get(int(_g["Job"].mode()[0]), "?")
        _row["Gender"]      = _gen_map.get(int(_g["Gender"].mode()[0]), "?")
        _row["Area"]        = _area_map.get(int(_g["Area"].mode()[0]), "?")
        _row["Investments"] = _inv_map.get(int(_g["Investments"].mode()[0]), "?")
        _rows.append(_row)

    _profile_df = _pd_prof.DataFrame(_rows)

    # -- Radar chart (plotly) --
    _pal = px.colors.qualitative.Vivid
    _radar_fig = go.Figure()
    for _c in range(optimal_k):
        _g = _df_p[_df_p["Cluster"] == _c]
        _vals = _g[numerical_cols].mean().tolist()
        _vals_closed = _vals + [_vals[0]]
        _cats_closed = numerical_cols + [numerical_cols[0]]
        _radar_fig.add_trace(go.Scatterpolar(
            r=_vals_closed, theta=_cats_closed,
            fill="toself", name=f"Cluster {_c}",
            line_color=_pal[_c], opacity=0.7,
        ))
    _radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Radar Chart — Mean Numerical Profiles (k={optimal_k})",
        height=520,
    )

    # -- Box plots for key financial features --
    _key_cols = ["Income", "Wealth", "Debt", "Saving", "Luxury"]
    _box_fig = make_subplots(rows=1, cols=len(_key_cols),
                              subplot_titles=_key_cols)
    for _ci, _col in enumerate(_key_cols):
        for _c in range(optimal_k):
            _g = _df_p[_df_p["Cluster"] == _c]
            _box_fig.add_trace(go.Box(
                y=_g[_col].tolist(), name=f"Cluster {_c}",
                marker_color=_pal[_c], showlegend=(_ci == 0),
            ), row=1, col=_ci + 1)
    _box_fig.update_layout(height=450, title_text="Key Financial Features by Cluster",
                            boxmode="group")

    mo.vstack([
        mo.md("### Summary Table"),
        mo.ui.table(_profile_df, selection=None),
        mo.md("### Radar Profiles"),
        mo.ui.plotly(_radar_fig),
        mo.md("### Financial Feature Distributions by Cluster"),
        mo.ui.plotly(_box_fig),
    ])
    return


# ---------------------------------------------------------------------------
# Final Notes
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 9. Summary & Business Interpretation

        While the numerical results typically suggest that **k=4 is technically optimal**
        according to the majority of indices, a choice of **k=3 remains highly valid**.
        In a business context, k=3 often provides a more interpretable segmentation
        (e.g., Low, Medium, High value clients) and maintains consistency with
        previous exploratory models.

        ### Recommended Personas (k=3)

        | Persona | Profile | Marketing Focus |
        |---------|---------|-----------------|
        | Segment A | High Wealth & Income, high FinEdu, ESG-oriented | Premium portfolios, sustainable investments |
        | Segment B | Mid Income, family-focused, moderate Saving | Life insurance, education savings plans |
        | Segment C | Lower Income, higher Debt, lower FinEdu | Financial literacy programs, debt restructuring |
        """
    )
    return


if __name__ == "__main__":
    app.run()
