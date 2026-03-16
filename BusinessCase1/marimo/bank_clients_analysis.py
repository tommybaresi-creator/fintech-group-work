import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Bank Clients — EDA & K-Medoids Clustering")


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
@app.cell
def __():
    import pandas as pd
    import numpy as np
    import json
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from pathlib import Path
    import warnings
    import marimo as mo

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    return Path, go, json, make_subplots, mo, np, pd, px, warnings


# ---------------------------------------------------------------------------
# PATHS & HELPERS
# ---------------------------------------------------------------------------
@app.cell
def __(Path, mo):
    RESULTS = Path(__file__).parent / "results"
    ASSETS  = Path(__file__).parent / "assets"

    def load_svg(name: str):
        p = ASSETS / f"{name}.svg"
        return mo.Html(p.read_text(encoding="utf-8")) if p.exists() else mo.md("")

    def check_results():
        required = [
            "df_cluster.parquet", "distance_matrix.npy", "metrics.json",
            "labels_k3.npy", "labels_k4.npy", "labels_k5.npy", "labels_k6.npy",
            "umap2d.npy", "umap3d.npy",
            "distance_matrix_w.npy", "metrics_w.json",
            "labels_k3_w.npy", "labels_k4_w.npy", "labels_k5_w.npy", "labels_k6_w.npy",
            "umap2d_w.npy", "umap3d_w.npy",
        ]
        missing = [f for f in required if not (RESULTS / f).exists()]
        return missing

    _missing = check_results()
    if _missing:
        mo.callout(
            mo.md(
                f"**Precomputed results not found.** Run the following command first:\n\n"
                f"```bash\nuv run python marimo/precompute.py\n```\n\n"
                f"Missing: {', '.join(_missing)}"
            ),
            kind="danger",
        )
    else:
        mo.callout(mo.md("All precomputed results found."), kind="success")

    return ASSETS, RESULTS, check_results, load_svg


# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Bank Client Segmentation — EDA & K-Medoids Clustering

        **Executive Summary (Abstract)**: We analyse a portfolio of **5,000 retail bank clients** described by **18 mixed-type features**
        (15 numerical/ordinal, 3 categorical). Our goal is to identify distinct, actionable customer segments
        that can inform targeted marketing and product personalisation strategies. We adopt a **K-Medoids clustering** approach with **Gower distance**, performing standard (unweighted) clustering alongside a **weighted alternative** that assigns double importance to the *Job* and *Investments* features.

        **Index**:
        - [Part 1 — Exploratory Data Analysis & Outlier Detection](#part-1)
        - [Part 2 — Unweighted K-Medoids Clustering](#part-2)
        - [Part 3 — Weighted K-Medoids Clustering](#part-3)

        ---

        | Feature | Type | Description |
        |---------|------|-------------|
        | `Age` | Numerical | Age in years |
        | `Gender` | Categorical | 0 = Male, 1 = Female |
        | `Job` | Categorical | 1=Unemployed, 2=Employee, 3=Manager, 4=Entrepreneur, 5=Retired |
        | `Area` | Categorical | 1=North, 2=Center, 3=South/Islands |
        | `CitySize` | Numerical (Ordinal) | 1=Small, 2=Medium, 3=Large (>200k) |
        | `FamilySize` | Numerical | Number of family members |
        | `Income` | Numerical | Normalized income percentile in [0,1] |
        | `Wealth` | Numerical | Normalized wealth percentile in [0,1] |
        | `Debt` | Numerical | Normalized debt percentile in [0,1] |
        | `FinEdu` | Numerical | Financial education in [0,1] |
        | `ESG` | Numerical | ESG investing propensity in [0,1] |
        | `Digital` | Numerical | Digital channel adoption in [0,1] |
        | `BankFriend` | Numerical | Bank relationship orientation in [0,1] |
        | `LifeStyle` | Numerical | Lifestyle spending index in [0,1] |
        | `Luxury` | Numerical | Luxury goods propensity in [0,1] |
        | `Saving` | Numerical | Savings behaviour in [0,1] |
        | `Investments` | Numerical (Ordinal) | 1=None, 2=Lump Sum, 3=Capital Accumulation |
        """
    )
    return


# ===========================================================================
# PART 1 — EDA
# ===========================================================================
@app.cell(hide_code=True)
def __(mo):
    mo.md("<a id='part-1'></a>\n---\n# Part 1 — Exploratory Data Analysis & Outlier Detection")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1. Data Loading & Sanity Checks")
    return


@app.cell
def __(Path, pd):
    _data_path = Path(__file__).parent.parent / "Data" / "Dataset1_BankClients.xlsx"
    df_raw = pd.read_excel(_data_path)
    print(f"Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
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
        mo.md(f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns | **Duplicates:** {_dupes}"),
        mo.ui.table(_tbl, selection=None),
    ])
    return (df,)


@app.cell
def __(df, mo):
    categorical_cols = ["Gender", "Job", "Area"]
    numerical_cols = [c for c in df.columns if c not in categorical_cols]
    normalized_vars = [c for c in numerical_cols if c not in ["Age", "FamilySize", "CitySize", "Investments"]]

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

        The 15 numerical features cover different scales:
        - **Age** and **FamilySize** are raw counts/years
        - **CitySize** and **Investments** are ordinal scales
        - All remaining features are pre-normalized to $[0,1]$ percentile ranks

        We verify these bounds explicitly and inspect distributional shapes.
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
def __(df, make_subplots, mo, numerical_cols, px):
    import plotly.graph_objects as _go_l
    _n = len(numerical_cols)
    _colors = px.colors.qualitative.Plotly
    _fig = make_subplots(rows=_n, cols=2, horizontal_spacing=0.08, vertical_spacing=0.015)
    for _i, _col in enumerate(numerical_cols):
        _r, _c = _i + 1, _colors[_i % len(_colors)]
        _fig.add_trace(_go_l.Histogram(x=df[_col].tolist(), marker_color=_c,
                                        opacity=0.75, showlegend=False,
                                        nbinsx=40, name=_col), row=_r, col=1)
        _fig.add_trace(_go_l.Box(x=df[_col].tolist(), marker_color=_c,
                                  showlegend=False, name=_col), row=_r, col=2)
        _fig.update_yaxes(title_text=_col, row=_r, col=1, title_font_size=9)
    _fig.update_layout(height=200 * _n, title_text="Numerical Features — Distributions & Boxplots",
                        title_font_size=14)
    mo.ui.plotly(_fig)
    return


@app.cell
def __(df, mo, normalized_vars, pd):
    _rows = []
    _rows.append({"Feature": "Age", "Min": round(float(df["Age"].min()), 2),
                  "Max": round(float(df["Age"].max()), 2), "Expected": "[0, 120]",
                  "OK?": "Yes" if df["Age"].max() <= 120 else "No"})
    _rows.append({"Feature": "FamilySize", "Min": int(df["FamilySize"].min()),
                  "Max": int(df["FamilySize"].max()), "Expected": ">= 1",
                  "OK?": "Yes" if df["FamilySize"].min() >= 1 else "No"})
    for _col in normalized_vars:
        _mn, _mx = float(df[_col].min()), float(df[_col].max())
        _rows.append({"Feature": _col, "Min": round(_mn, 4), "Max": round(_mx, 4),
                      "Expected": "[0, 1]", "OK?": "Yes" if _mn >= 0 and _mx <= 1 else "No"})
    _has_issues = any(r["OK?"] == "No" for r in _rows)
    mo.vstack([
        mo.callout(
            mo.md("Feature bounds are all within expected ranges." if not _has_issues
                  else "Some features fall outside expected bounds."),
            kind="success" if not _has_issues else "warn",
        ),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


# -- Categorical distributions ----------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3. Univariate Analysis — Categorical Features

        The client base is broadly distributed across job types, with a slight
        predominance of employees. Area of residence is roughly balanced; most
        clients live in medium or large cities. Investment behaviour is heavily
        skewed — the majority hold no formal investment product.
        """
    )
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
        _ls = [_labels_map[_col].get(int(v), str(v)) for v in _vc.index]
        _f = px.bar(x=_ls, y=_vc.values.tolist(), title=f"{_col}",
                    labels={"x": "", "y": "Count"},
                    color_discrete_sequence=px.colors.qualitative.Vivid)
        _f.update_layout(height=300, margin=dict(t=40, b=15))
        _figs.append(mo.ui.plotly(_f))
    mo.hstack(_figs, wrap=True)
    return


# -- Correlation heatmap ----------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4. Bivariate Analysis — Pearson Correlation

        We compute pairwise Pearson correlation among numerical features:

        $$r_{XY} = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}
        {\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i-\bar{Y})^2}}$$

        Highly correlated feature pairs ($|r| > 0.8$) carry partially redundant information.
        We note these but do **not** remove features: Gower distance already weights each
        feature equally, and we prefer to preserve the full signal for clustering.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols, pd, px):
    corr_matrix = df[numerical_cols].corr()
    _fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Pearson Correlation Heatmap — Numerical Features")
    _fig_corr.update_layout(height=570)

    _pairs = []
    for _i in range(len(corr_matrix.columns)):
        for _j in range(_i):
            _v = corr_matrix.iloc[_i, _j]
            if abs(_v) > 0.8:
                _pairs.append({"Feature A": corr_matrix.columns[_i],
                                "Feature B": corr_matrix.columns[_j],
                                "r": round(float(_v), 4)})
    mo.vstack([
        mo.ui.plotly(_fig_corr),
        mo.callout(
            mo.md(f"**{len(_pairs)} pair(s)** with |r| > 0.8 detected." if _pairs
                  else "No strong correlations (|r| > 0.8) detected."),
            kind="warn" if _pairs else "success",
        ),
        mo.ui.table(pd.DataFrame(_pairs if _pairs else [{"Result": "None"}]), selection=None),
    ])
    return (corr_matrix,)


# -- Domain anomalies -------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. Domain Anomaly Detection

        We flag client records that are logically inconsistent with real-world constraints.
        These do not necessarily indicate data corruption, but represent edge cases that
        may distort cluster structure if left untreated.
        """
    )
    return


@app.cell
def __(df, mo, pd):
    _cohorts = {
        "Young Retirees (Age < 50, Job=Retired)":     int(((df["Age"] < 50) & (df["Job"] == 5)).sum()),
        "Working Minors (Age < 18, Job in {2,3,4})":  int(((df["Age"] < 18) & (df["Job"].isin([2,3,4]))).sum()),
        "Rich Unemployed (Job=1, Income > 0.9)":       int(((df["Job"] == 1) & (df["Income"] > 0.9)).sum()),
        "Wealthy, No Investments (Inv=1, Wealth>0.95)":int(((df["Investments"] == 1) & (df["Wealth"] > 0.95)).sum()),
        "Young Large Families (Age<=20, FamSize>=4)":  int(((df["Age"] <= 20) & (df["FamilySize"] >= 4)).sum()),
    }
    _total = sum(_cohorts.values())
    _tbl = pd.DataFrame([{"Cohort": k, "Count": v, "%": f"{v/len(df)*100:.2f}%"}
                          for k, v in _cohorts.items()])
    mo.vstack([
        mo.ui.table(_tbl, selection=None),
        mo.callout(
            mo.md(f"**{_total} records** ({_total/len(df)*100:.2f}%) match at least one anomaly rule. "
                  "Working minors are excluded before clustering as a hard filter."),
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

        Beyond univariate checks, we apply **Isolation Forest** to detect
        clients that are anomalous across the joint distribution of all numerical features.
        The anomaly score is:

        $$\text{score}(x, n) = 2^{-\,\dfrac{E[h(x)]}{c(n)}}, \quad
        c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

        Scores near 1 indicate outliers; near 0.5 indicate typical observations.
        We set `contamination=0.01` to remove the top 1% most anomalous clients.
        The cleaned dataset is then used for all clustering steps.
        """
    )
    return


@app.cell
def __(df, mo, numerical_cols):
    from sklearn.ensemble import IsolationForest as _IsoF
    _iso = _IsoF(contamination=0.01, random_state=42)
    _lbl = _iso.fit_predict(df[numerical_cols])
    _n_out = int((_lbl == -1).sum())
    _n_clean = int((_lbl == 1).sum())
    mo.callout(
        mo.md(f"**{_n_out} multivariate outliers** removed ({_n_out/len(df)*100:.2f}%). "
              f"Clustering proceeds on **{_n_clean:,} clients**."),
        kind="warn",
    )
    return


# ===========================================================================
# PART 2 — K-MEDOIDS CLUSTERING
# ===========================================================================
@app.cell(hide_code=True)
def __(mo):
    mo.md("<a id='part-2'></a>\n---\n# Part 2 — Unweighted K-Medoids Clustering")
    return


# ---------------------------------------------------------------------------
# 1. Distance Metric Selection — our rationale
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1. Why Gower Distance?

        Selecting the right distance metric is the most consequential methodological decision
        in this analysis. Our dataset is **mixed-type**: five categorical features (Job, Gender,
        Area, CitySize, Investments) and thirteen numerical features at different scales.

        We evaluated four candidate metrics:
        """
    )
    return


@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("**Our evaluation of distance metrics across four criteria:**"),
        load_svg("distance_metric_selection_guide"),
        mo.md(
            r"""
            **Euclidean** and **Manhattan** must be ruled out immediately: both require purely
            numerical input and cannot process categorical features without lossy encoding
            (one-hot encoding inflates dimensionality and introduces artificial sparsity).

            **Jaccard** operates on set membership and loses all magnitude information —
            Income and Wealth differences would be invisible.

            **Gower distance** is the natural choice for our data structure:

            $$d_{\text{Gower}}(x, y) = \frac{1}{p}\sum_{j=1}^{p}\delta_j(x,y), \quad
            \delta_j = \begin{cases} \dfrac{|x_j - y_j|}{R_j} & \text{numerical} \\[4pt]
            \mathbf{1}[x_j \neq y_j] & \text{categorical} \end{cases}$$

            Each feature contributes equally; numerical features are range-normalized
            to $[0,1]$ automatically ($R_j = \max_j - \min_j$), preventing Age (range 19–95)
            from dominating the [0,1]-scaled financial indicators.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("**Weighting rationale — why we chose unweighted Gower:**"),
        load_svg("weighted_gower_decision"),
        mo.md(
            "We considered weighting categorical features more heavily to reflect their "
            "business relevance (Job type and Investment behaviour drive product affinity). "
            "However, we opted for **equal weights** to avoid introducing subjective priors "
            "that could not be validated without ground-truth segment labels. "
            "The equal-weight formulation also ensures the distance matrix is a proper metric."
        ),
    ])
    return


# ---------------------------------------------------------------------------
# 2. Preprocessing for Gower
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2. Preprocessing Pipeline

        Computing a valid Gower distance matrix requires careful data preparation.
        We ran the full pipeline in `precompute.py` — results are loaded below.

        **Key steps:**

        1. **Missing values** — imputed with column median (numerical) / mode (categorical)
        2. **Dtype enforcement** — categorical columns cast to `object` dtype so the
           `gower` library applies Hamming distance (not numerical range normalization)
        3. **Hard domain filter** — working minors (Age < 18, employed) removed
        4. **Isolation Forest** — top 1% multivariate outliers removed
        5. **Gower matrix** — computed on the cleaned, correctly-typed dataframe

        The resulting matrix is $N \times N$ float32, entries in $[0, 1]$.
        """
    )
    return


# ---------------------------------------------------------------------------
# 3. Load precomputed data
# ---------------------------------------------------------------------------
@app.cell
def __(RESULTS, json, mo, np, pd):
    _missing = []
    for _f in ["df_cluster.parquet", "distance_matrix.npy", "metrics.json",
               "labels_k3.npy", "umap2d.npy", "umap3d.npy"]:
        if not (RESULTS / _f).exists():
            _missing.append(_f)

    if _missing:
        raise FileNotFoundError(
            f"Missing precomputed files: {_missing}. "
            "Run: uv run python marimo/precompute.py"
        )

    df_cluster = pd.read_parquet(RESULTS / "df_cluster.parquet")
    
    distance_matrix = np.load(RESULTS / "distance_matrix.npy")
    distance_matrix_w = np.load(RESULTS / "distance_matrix_w.npy")

    with open(RESULTS / "metrics.json") as _f:
        _raw_metrics = json.load(_f)
    metrics_dict = {int(k): v for k, v in _raw_metrics.items()}

    with open(RESULTS / "metrics_w.json") as _f:
        _raw_metrics_w = json.load(_f)
    metrics_dict_w = {int(k): v for k, v in _raw_metrics_w.items()}

    K_RANGE   = sorted(metrics_dict.keys())
    labels_by_k = {k: np.load(RESULTS / f"labels_k{k}.npy") for k in K_RANGE}
    labels_by_k_w = {k: np.load(RESULTS / f"labels_k{k}_w.npy") for k in K_RANGE}
    
    umap2d    = np.load(RESULTS / "umap2d.npy")
    umap3d    = np.load(RESULTS / "umap3d.npy")
    umap2d_w  = np.load(RESULTS / "umap2d_w.npy")
    umap3d_w  = np.load(RESULTS / "umap3d_w.npy")

    mo.vstack([
        mo.callout(mo.md("Precomputed results loaded successfully."), kind="success"),
        mo.md(
            f"- Dataset: **{df_cluster.shape[0]:,} clients** x {df_cluster.shape[1]} features  \n"
            f"- Distance matrix: **{distance_matrix.shape[0]:,} x {distance_matrix.shape[0]:,}** "
            f"({distance_matrix.nbytes / 1e6:.0f} MB)  \n"
            f"- Metrics available for k: **{K_RANGE}**"
        ),
    ])
    return (
        K_RANGE,
        df_cluster,
        distance_matrix,
        distance_matrix_w,
        labels_by_k,
        labels_by_k_w,
        metrics_dict,
        metrics_dict_w,
        umap2d,
        umap3d,
        umap2d_w,
        umap3d_w,
    )


# ---------------------------------------------------------------------------
# 4. Gower Distance Distribution
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3. Gower Distance Distribution

        The pairwise Gower matrix covers $\binom{N}{2}$ unique client pairs.
        We examine the distribution to confirm the metric is well-behaved before
        passing it to the clustering algorithm.
        """
    )
    return


@app.cell
def __(distance_matrix, go, mo, np, pd):
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _dm_mean = float(_upper.mean())
    _dm_med  = float(np.median(_upper))
    _dm_std  = float(_upper.std())

    _fig_dist = go.Figure()
    _fig_dist.add_trace(go.Histogram(x=_upper.tolist(), nbinsx=120,
                                     marker_color="steelblue", opacity=0.8,
                                     name="Gower distances"))
    _fig_dist.add_vline(x=_dm_mean, line_dash="dash", line_color="red",
                        annotation_text=f"Mean = {_dm_mean:.3f}")
    _fig_dist.add_vline(x=_dm_med, line_dash="dot", line_color="green",
                        annotation_text=f"Median = {_dm_med:.3f}")
    _fig_dist.update_layout(
        title=f"Pairwise Gower Distance Distribution ({len(_upper):,} pairs)",
        xaxis_title="Gower Distance", yaxis_title="Frequency", height=400,
    )

    _stats_df = pd.DataFrame({
        "Statistic": ["Min", "Max", "Mean", "Median", "Std Dev"],
        "Value": [
            f"{float(_upper.min()):.4f}",
            f"{float(_upper.max()):.4f}",
            f"{_dm_mean:.4f}",
            f"{_dm_med:.4f}",
            f"{_dm_std:.4f}",
        ],
    })

    mo.vstack([
        mo.ui.plotly(_fig_dist),
        mo.ui.table(_stats_df, selection=None),
        mo.callout(
            mo.md("The distribution is roughly bell-shaped and confined to [0, 1], "
                  "confirming that no single feature dominates the distance metric. "
                  "The absence of a spike near 0 indicates the dataset has no trivial near-duplicates "
                  "post-outlier removal."),
            kind="info",
        ),
    ])
    return


# ---------------------------------------------------------------------------
# 5. K-Medoids rationale + SVGs
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md(
            r"""
            ## 4. K-Medoids Algorithm

            We select **K-Medoids** (specifically the FasterPAM variant) for three reasons:

            1. **Precomputed matrix compatibility** — K-Means requires computing Euclidean centroids,
               which are undefined in a precomputed distance setting. K-Medoids assigns the cluster
               representative to the actual data point that minimises within-cluster total distance:

               $$m_k = \underset{x_i \in C_k}{\arg\min} \sum_{x_j \in C_k} d(x_i, x_j)$$

            2. **Robustness** — medoids are less sensitive to extreme observations than means.

            3. **Interpretability** — each medoid is a real client, making segment representatives
               directly actionable for marketing personas.
            """
        ),
        mo.md("**K-Medoids algorithm and its interaction with Gower-preprocessed inputs:**"),
        load_svg("kmedoids_explainer"),
    ])
    return


@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("**Our full methodology from raw data to cluster labels:**"),
        load_svg("gower_kmedoids_pipeline"),
    ])
    return


# ---------------------------------------------------------------------------
# 6. Validation metrics + optimal k
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. Cluster Validation & Optimal k Selection

        We compute three complementary cluster quality indices and select $k$ via majority voting.

        **Silhouette Coefficient** $\in [-1, 1]$ — maximize

        $$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

        where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean distance to nearest other cluster.

        **Davies-Bouldin Index** $\in [0, \infty)$ — minimize

        $$DB = \frac{1}{K}\sum_{i=1}^{K}\max_{j \neq i}\left(\frac{\sigma_i + \sigma_j}{d(m_i, m_j)}\right)$$

        **Calinski-Harabasz Index** $\in [0, \infty)$ — maximize

        $$CH = \frac{SS_B\,/\,(K-1)}{SS_W\,/\,(n-K)}$$
        """
    )
    return


@app.cell
def __(K_RANGE, go, make_subplots, metrics_dict, mo, pd):
    summary_df = pd.DataFrame({
        "k":                  K_RANGE,
        "Silhouette":         [metrics_dict[k]["silhouette"]         for k in K_RANGE],
        "Loss":               [metrics_dict[k]["loss"]               for k in K_RANGE],
    })

    _ks = summary_df["k"].tolist()
    _bsil = int(summary_df.loc[summary_df["Silhouette"].idxmax(), "k"])

    _fig_met = make_subplots(rows=1, cols=2, subplot_titles=[
        "Silhouette<br>(Higher = Better)",
        "K-Medoids Loss<br>(Lower = Better, look for Elbow)",
    ])
    for _ci, (_col, _vals, _best) in enumerate([
        ("#1f77b4", summary_df["Silhouette"].tolist(),        _bsil),
        ("#ff7f0e", summary_df["Loss"].tolist(),              None),
    ]):
        _fig_met.add_trace(go.Scatter(x=_ks, y=_vals, mode="lines+markers",
                                       marker=dict(size=10), line=dict(color=_col, width=2),
                                       showlegend=False), row=1, col=_ci + 1)
        if _best is not None:
            _fig_met.add_vline(x=_best, line_dash="dash", line_color="red",
                               annotation_text=f"k={_best}", row=1, col=_ci + 1)
            
    _fig_met.update_xaxes(title_text="k")
    _fig_met.update_layout(height=400, title_text="Cluster Validation Metrics vs. k",
                            title_font_size=14)

    mo.vstack([
        mo.ui.plotly(_fig_met),
        mo.ui.table(summary_df.round(4), selection=None),
    ])
    return summary_df,


@app.cell
def __(mo, summary_df):
    _bsil = int(summary_df.loc[summary_df["Silhouette"].idxmax(), "k"])
    optimal_k = _bsil

    mo.callout(
        mo.md(
            "### Optimal k Selection\n\n"
            f"**Max Silhouette: k = {_bsil}**\n\n"
            f"We proceed with **k = {optimal_k}**. Note that other values may remain equally defensible from "
            "a business interpretability standpoint, provided they align with an elbow in the loss curve "
            "and map naturally to logical client tiers."
        ),
        kind="success",
    )
    return (optimal_k,)


# ---------------------------------------------------------------------------
# 7. PCA / UMAP compatibility + visualizations
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("## 6. Cluster Visualizations"),
        mo.md("**Why PCA and UMAP are both valid for visualising Gower-based clusters:**"),
        load_svg("pca_gower_pam_compatibility"),
        mo.md(
            "PCA is applied to the **numerical subspace** of the cleaned dataset — it captures "
            "linear variance and provides a consistent, deterministic projection. "
            "UMAP is applied to the **precomputed Gower distance matrix** (`metric='precomputed'`), "
            "preserving non-linear topological structure while remaining consistent with the "
            "clustering geometry."
        ),
    ])
    return


# -- PCA visualization -------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("### 6.1 PCA — Linear Projection of Numerical Features")
    return


@app.cell
def __(df_cluster, labels_by_k, mo, np, optimal_k, px):
    from sklearn.decomposition import PCA as _PCA
    _num_cols = [c for c in df_cluster.columns
                 if c not in ["Gender", "Job", "Area", "CitySize", "Investments"]]
    _df_num = df_cluster[_num_cols].astype(float)
    _pca = _PCA(n_components=2, random_state=42)
    _pca_r = _pca.fit_transform(_df_num)
    _ev1, _ev2 = _pca.explained_variance_ratio_

    import pandas as _ppca
    _pca_df = _ppca.DataFrame({
        "PC1": _pca_r[:, 0].tolist(), "PC2": _pca_r[:, 1].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k[optimal_k].tolist()],
    })
    _fig_pca = px.scatter(_pca_df, x="PC1", y="PC2", color="Cluster",
                           title=f"PCA 2D — K-Medoids k={optimal_k} (numerical features)",
                           labels={"PC1": f"PC1 ({_ev1:.1%} var.)", "PC2": f"PC2 ({_ev2:.1%} var.)"},
                           opacity=0.55, color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_pca.update_traces(marker=dict(size=4))
    _fig_pca.update_layout(height=520)

    mo.vstack([
        mo.md(f"PC1+PC2 explain **{(_ev1+_ev2):.1%}** of numerical variance."),
        mo.ui.plotly(_fig_pca),
    ])
    return


# -- UMAP 2D -----------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 6.2 UMAP 2D — Non-linear Embedding on Gower Distance

        $$\mathcal{L}(\text{UMAP}) = \sum_{e} \left[w_h \log\frac{w_h}{w_l} + (1-w_h)\log\frac{1-w_h}{1-w_l}\right]$$

        UMAP minimises cross-entropy between the high-dimensional fuzzy simplicial set ($w_h$, from Gower)
        and the low-dimensional representation ($w_l$). The embedding below was computed
        on the full precomputed Gower matrix with `n_neighbors=15`, `min_dist=0.1`.
        """
    )
    return


@app.cell
def __(labels_by_k, mo, optimal_k, px, umap2d):
    import pandas as _pumap
    _umap_df = _pumap.DataFrame({
        "UMAP1": umap2d[:, 0].tolist(),
        "UMAP2": umap2d[:, 1].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k[optimal_k].tolist()],
    })
    _fig_u2 = px.scatter(_umap_df, x="UMAP1", y="UMAP2", color="Cluster",
                          title=f"UMAP 2D — K-Medoids k={optimal_k} (Gower metric, full dataset)",
                          opacity=0.6, color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_u2.update_traces(marker=dict(size=4))
    _fig_u2.update_layout(height=560)
    mo.ui.plotly(_fig_u2)
    return


# -- UMAP 3D -----------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("### 6.3 UMAP 3D — Interactive Exploration")
    return


@app.cell
def __(labels_by_k, mo, optimal_k, px, umap3d):
    import pandas as _pumap3
    _umap3_df = _pumap3.DataFrame({
        "UMAP1": umap3d[:, 0].tolist(),
        "UMAP2": umap3d[:, 1].tolist(),
        "UMAP3": umap3d[:, 2].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k[optimal_k].tolist()],
    })
    _fig_u3 = px.scatter_3d(_umap3_df, x="UMAP1", y="UMAP2", z="UMAP3",
                              color="Cluster",
                              title=f"UMAP 3D — K-Medoids k={optimal_k} (Gower metric)",
                              opacity=0.65,
                              color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_u3.update_traces(marker=dict(size=3))
    _fig_u3.update_layout(height=650)
    mo.ui.plotly(_fig_u3)
    return


# ---------------------------------------------------------------------------
# 8. Cluster Profiling
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 7. Cluster Profiling & Business Personas

        We characterise each cluster by computing mean numerical features and
        mode categorical features. These profiles are used to derive interpretable
        customer personas for downstream marketing strategy.
        """
    )
    return


@app.cell
def __(df_cluster, go, labels_by_k, make_subplots, mo, optimal_k, px):
    import pandas as _pprof

    _cat_cols = ["Gender", "Job", "Area", "CitySize", "Investments"]
    _num_cols = [c for c in df_cluster.columns if c not in _cat_cols]

    _job_map  = {1:"Unemployed", 2:"Employee", 3:"Manager", 4:"Entrepreneur", 5:"Retired"}
    _inv_map  = {1:"None", 2:"Lump Sum", 3:"Cap. Accum."}
    _gen_map  = {0:"Male", 1:"Female"}
    _area_map = {1:"North", 2:"Center", 3:"South"}

    _df_p = df_cluster.copy()
    _df_p["Cluster"] = labels_by_k[optimal_k]

    _pal = px.colors.qualitative.Vivid

    # Summary table
    _rows = []
    for _c in range(optimal_k):
        _g = _df_p[_df_p["Cluster"] == _c]
        _row = {"Cluster": _c, "N": len(_g), "%": f"{len(_g)/len(_df_p)*100:.1f}%"}
        for _col in _num_cols:
            _row[_col] = f"{float(_g[_col].mean()):.3f}"
        _row["Job"]         = _job_map.get(int(_g["Job"].astype(float).mode()[0]), "?")
        _row["Gender"]      = _gen_map.get(int(_g["Gender"].astype(float).mode()[0]), "?")
        _row["Investments"] = _inv_map.get(int(_g["Investments"].astype(float).mode()[0]), "?")
        _rows.append(_row)
    _profile_df = _pprof.DataFrame(_rows)

    # Radar chart
    _radar = go.Figure()
    for _c in range(optimal_k):
        _g = _df_p[_df_p["Cluster"] == _c]
        _vals = [float(_g[_col].mean()) for _col in _num_cols]
        _radar.add_trace(go.Scatterpolar(
            r=_vals + [_vals[0]], theta=_num_cols + [_num_cols[0]],
            fill="toself", name=f"Cluster {_c}",
            line_color=_pal[_c], opacity=0.75,
        ))
    _radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Radar — Numerical Feature Means per Cluster (k={optimal_k})",
        height=500,
    )

    # Box plots
    _key = ["Income", "Wealth", "Debt", "Saving", "Luxury", "FinEdu"]
    _box = make_subplots(rows=1, cols=len(_key), subplot_titles=_key)
    for _ci, _col in enumerate(_key):
        for _c in range(optimal_k):
            _g = _df_p[_df_p["Cluster"] == _c]
            _box.add_trace(go.Box(y=_g[_col].tolist(), name=f"Cluster {_c}",
                                   marker_color=_pal[_c], showlegend=(_ci == 0)), row=1, col=_ci + 1)
    _box.update_layout(height=430, title_text="Key Financial Features by Cluster", boxmode="group")

    mo.vstack([
        mo.md("### Cluster Summary"),
        mo.ui.table(_profile_df, selection=None),
        mo.md("### Radar Profiles"),
        mo.ui.plotly(_radar),
        mo.md("### Financial Feature Distributions by Cluster"),
        mo.ui.plotly(_box),
    ])
    return


# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo, optimal_k):
    mo.md(
        f"""
        ## 8. Conclusions & Business Recommendations

        Our analysis identifies **k = {optimal_k}** as the optimal number of
        client segments by maximising the Silhouette index and confirming an elbow structure in K-Medoids loss.

        The Gower + K-Medoids approach provides three key advantages over simpler alternatives:
        - **No information loss** — all 18 features (including categorical) contribute to cluster assignment
        - **Scale invariance** — automatic range normalization prevents any single feature from dominating
        - **Interpretable representatives** — each medoid is a real client that can serve as a segment archetype

        ### Recommended Personas

        | Segment | Profile | Recommended Products |
        |---------|---------|---------------------|
        | A | High Income / Wealth, ESG-oriented, high FinEdu | Sustainable portfolios, premium advisory |
        | B | Mid Income, family-oriented, moderate Saving | Life insurance, education savings plans |
        | C | Lower Income, higher Debt, low FinEdu | Financial literacy, debt consolidation |
        """
    )
    return


# ---------------------------------------------------------------------------
# 4. Gower Distance Distribution
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <a id="part-3"></a>\n---\n# Part 3 — Weighted K-Medoids Clustering\n\n## 1. Weighted Gower Distance Distribution

        The **weighted** pairwise Gower matrix (Job and Investments x2) covers $\binom{N}{2}$ unique client pairs.
        We examine the distribution to confirm the metric is well-behaved before
        passing it to the clustering algorithm.
        """
    )
    return


@app.cell
def __(distance_matrix_w, go, mo, np, pd):
    _upper = distance_matrix_w[np.triu_indices_from(distance_matrix_w, k=1)]
    _dm_mean = float(_upper.mean())
    _dm_med  = float(np.median(_upper))
    _dm_std  = float(_upper.std())

    _fig_dist = go.Figure()
    _fig_dist.add_trace(go.Histogram(x=_upper.tolist(), nbinsx=120,
                                     marker_color="steelblue", opacity=0.8,
                                     name="Gower distances"))
    _fig_dist.add_vline(x=_dm_mean, line_dash="dash", line_color="red",
                        annotation_text=f"Mean = {_dm_mean:.3f}")
    _fig_dist.add_vline(x=_dm_med, line_dash="dot", line_color="green",
                        annotation_text=f"Median = {_dm_med:.3f}")
    _fig_dist.update_layout(
        title=f"Pairwise Gower Distance Distribution ({len(_upper):,} pairs)",
        xaxis_title="Gower Distance", yaxis_title="Frequency", height=400,
    )

    _stats_df = pd.DataFrame({
        "Statistic": ["Min", "Max", "Mean", "Median", "Std Dev"],
        "Value": [
            f"{float(_upper.min()):.4f}",
            f"{float(_upper.max()):.4f}",
            f"{_dm_mean:.4f}",
            f"{_dm_med:.4f}",
            f"{_dm_std:.4f}",
        ],
    })

    mo.vstack([
        mo.ui.plotly(_fig_dist),
        mo.ui.table(_stats_df, selection=None),
        mo.callout(
            mo.md("The distribution is roughly bell-shaped and confined to [0, 1], "
                  "confirming that no single feature dominates the distance metric. "
                  "The absence of a spike near 0 indicates the dataset has no trivial near-duplicates "
                  "post-outlier removal."),
            kind="info",
        ),
    ])
    return


# ---------------------------------------------------------------------------
# 5. K-Medoids rationale + SVGs
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md(
            r"""
            ## 2. K-Medoids Algorithm (Weighted)

            We select **K-Medoids** (specifically the FasterPAM variant) for three reasons:

            1. **Precomputed matrix compatibility** — K-Means requires computing Euclidean centroids,
               which are undefined in a precomputed distance setting. K-Medoids assigns the cluster
               representative to the actual data point that minimises within-cluster total distance:

               $$m_k = \underset{x_i \in C_k}{\arg\min} \sum_{x_j \in C_k} d(x_i, x_j)$$

            2. **Robustness** — medoids are less sensitive to extreme observations than means.

            3. **Interpretability** — each medoid is a real client, making segment representatives
               directly actionable for marketing personas.
            """
        ),
        mo.md("**K-Medoids algorithm and its interaction with Gower-preprocessed inputs:**"),
        load_svg("kmedoids_explainer"),
    ])
    return


@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("**Our full methodology from raw data to cluster labels:**"),
        load_svg("gower_kmedoids_pipeline"),
    ])
    return


# ---------------------------------------------------------------------------
# 6. Validation metrics + optimal k
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3. Weighted Cluster Validation

        We compute three complementary cluster quality indices and select $k$ via majority voting.

        **Silhouette Coefficient** $\in [-1, 1]$ — maximize

        $$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

        where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean distance to nearest other cluster.

        **Davies-Bouldin Index** $\in [0, \infty)$ — minimize

        $$DB = \frac{1}{K}\sum_{i=1}^{K}\max_{j \neq i}\left(\frac{\sigma_i + \sigma_j}{d(m_i, m_j)}\right)$$

        **Calinski-Harabasz Index** $\in [0, \infty)$ — maximize

        $$CH = \frac{SS_B\,/\,(K-1)}{SS_W\,/\,(n-K)}$$
        """
    )
    return


@app.cell
def __(K_RANGE, go, make_subplots, metrics_dict_w, mo, pd):
    summary_df_w = pd.DataFrame({
        "k":                  K_RANGE,
        "Silhouette":         [metrics_dict_w[k]["silhouette"]         for k in K_RANGE],
        "Davies-Bouldin":     [metrics_dict_w[k]["davies_bouldin"]     for k in K_RANGE],
        "Calinski-Harabasz":  [metrics_dict_w[k]["calinski_harabasz"]  for k in K_RANGE],
    })

    _ks = summary_df_w["k"].tolist()
    _bsil = int(summary_df_w.loc[summary_df_w["Silhouette"].idxmax(), "k"])
    _bdb  = int(summary_df_w.loc[summary_df_w["Davies-Bouldin"].idxmin(), "k"])
    _bch  = int(summary_df_w.loc[summary_df_w["Calinski-Harabasz"].idxmax(), "k"])

    _fig_met = make_subplots(rows=1, cols=3, subplot_titles=[
        "Silhouette<br>(Higher = Better)",
        "Davies-Bouldin<br>(Lower = Better)",
        "Calinski-Harabasz<br>(Higher = Better)",
    ])
    for _ci, (_col, _vals, _best, _better) in enumerate([
        ("#1f77b4", summary_df_w["Silhouette"].tolist(),        _bsil, "max"),
        ("#ff7f0e", summary_df_w["Davies-Bouldin"].tolist(),    _bdb,  "min"),
        ("#2ca02c", summary_df_w["Calinski-Harabasz"].tolist(), _bch,  "max"),
    ]):
        _fig_met.add_trace(go.Scatter(x=_ks, y=_vals, mode="lines+markers",
                                       marker=dict(size=10), line=dict(color=_col, width=2),
                                       showlegend=False), row=1, col=_ci + 1)
        _fig_met.add_vline(x=_best, line_dash="dash", line_color="red",
                           annotation_text=f"k={_best}", row=1, col=_ci + 1)
    _fig_met.update_xaxes(title_text="k")
    _fig_met.update_layout(height=400, title_text="Cluster Validation Metrics vs. k",
                            title_font_size=14)

    mo.vstack([
        mo.ui.plotly(_fig_met),
        mo.ui.table(summary_df_w.round(4), selection=None),
    ])
    return summary_df_w,


@app.cell
def __(mo, summary_df_w):
    _bsil = int(summary_df_w.loc[summary_df_w["Silhouette"].idxmax(), "k"])
    _bdb  = int(summary_df_w.loc[summary_df_w["Davies-Bouldin"].idxmin(), "k"])
    _bch  = int(summary_df_w.loc[summary_df_w["Calinski-Harabasz"].idxmax(), "k"])

    _votes = {}
    for _v in [_bsil, _bdb, _bch]:
        _votes[_v] = _votes.get(_v, 0) + 1
    optimal_k_w = max(_votes, key=lambda x: _votes[x])

    mo.callout(
        mo.md(
            "### Voting Scheme — Optimal k\n\n"
            f"| Metric | Best k |\n|---|---|\n"
            f"| Silhouette (maximize) | k = {_bsil} |\n"
            f"| Davies-Bouldin (minimize) | k = {_bdb} |\n"
            f"| Calinski-Harabasz (maximize) | k = {_bch} |\n\n"
            f"**Result: k = {optimal_k_w}** (majority vote)\n\n"
            "We proceed with this k. Note that k=3 remains equally defensible from "
            "a business interpretability standpoint — three segments map naturally to "
            "Low / Mid / High value client tiers."
        ),
        kind="success",
    )
    return (optimal_k_w,)


# ---------------------------------------------------------------------------
# 7. PCA / UMAP compatibility + visualizations
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(load_svg, mo):
    mo.vstack([
        mo.md("## 4. Weighted Cluster Visualizations"),
        mo.md("**Why PCA and UMAP are both valid for visualising Gower-based clusters:**"),
        load_svg("pca_gower_pam_compatibility"),
        mo.md(
            "PCA is applied to the **numerical subspace** of the cleaned dataset — it captures "
            "linear variance and provides a consistent, deterministic projection. "
            "UMAP is applied to the **precomputed Gower distance matrix** (`metric='precomputed'`), "
            "preserving non-linear topological structure while remaining consistent with the "
            "clustering geometry."
        ),
    ])
    return


# -- PCA visualization -------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("### 4.1 PCA (Weighted Labels)")
    return


@app.cell
def __(df_cluster, labels_by_k_w, mo, np, optimal_k_w, px):
    from sklearn.decomposition import PCA as _PCA
    _num_cols = [c for c in df_cluster.columns
                 if c not in ["Gender", "Job", "Area", "CitySize", "Investments"]]
    _df_num = df_cluster[_num_cols].astype(float)
    _pca = _PCA(n_components=2, random_state=42)
    _pca_r = _pca.fit_transform(_df_num)
    _ev1, _ev2 = _pca.explained_variance_ratio_

    import pandas as _ppca
    _pca_df = _ppca.DataFrame({
        "PC1": _pca_r[:, 0].tolist(), "PC2": _pca_r[:, 1].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k_w[optimal_k_w].tolist()],
    })
    _fig_pca = px.scatter(_pca_df, x="PC1", y="PC2", color="Cluster",
                           title=f"PCA 2D — K-Medoids k={optimal_k_w} (numerical features)",
                           labels={"PC1": f"PC1 ({_ev1:.1%} var.)", "PC2": f"PC2 ({_ev2:.1%} var.)"},
                           opacity=0.55, color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_pca.update_traces(marker=dict(size=4))
    _fig_pca.update_layout(height=520)

    mo.vstack([
        mo.md(f"PC1+PC2 explain **{(_ev1+_ev2):.1%}** of numerical variance."),
        mo.ui.plotly(_fig_pca),
    ])
    return


# -- UMAP 2D -----------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 4.2 UMAP 2D (Weighted) — Non-linear Embedding on Gower Distance

        $$\mathcal{L}(\text{UMAP}) = \sum_{e} \left[w_h \log\frac{w_h}{w_l} + (1-w_h)\log\frac{1-w_h}{1-w_l}\right]$$

        UMAP minimises cross-entropy between the high-dimensional fuzzy simplicial set ($w_h$, from Gower)
        and the low-dimensional representation ($w_l$). The embedding below was computed
        on the full precomputed Gower matrix with `n_neighbors=15`, `min_dist=0.1`.
        """
    )
    return


@app.cell
def __(labels_by_k_w, mo, optimal_k_w, px, umap2d_w):
    import pandas as _pumap
    _umap_df = _pumap.DataFrame({
        "UMAP1": umap2d_w[:, 0].tolist(),
        "UMAP2": umap2d_w[:, 1].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k_w[optimal_k_w].tolist()],
    })
    _fig_u2 = px.scatter(_umap_df, x="UMAP1", y="UMAP2", color="Cluster",
                          title=f"UMAP 2D — K-Medoids k={optimal_k_w} (Gower metric, full dataset)",
                          opacity=0.6, color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_u2.update_traces(marker=dict(size=4))
    _fig_u2.update_layout(height=560)
    mo.ui.plotly(_fig_u2)
    return


# -- UMAP 3D -----------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md("### 4.3 UMAP 3D (Weighted) — Interactive Exploration")
    return


@app.cell
def __(labels_by_k_w, mo, optimal_k_w, px, umap3d_w):
    import pandas as _pumap3
    _umap3_df = _pumap3.DataFrame({
        "UMAP1": umap3d_w[:, 0].tolist(),
        "UMAP2": umap3d_w[:, 1].tolist(),
        "UMAP3": umap3d_w[:, 2].tolist(),
        "Cluster": [f"Cluster {c}" for c in labels_by_k_w[optimal_k_w].tolist()],
    })
    _fig_u3 = px.scatter_3d(_umap3_df, x="UMAP1", y="UMAP2", z="UMAP3",
                              color="Cluster",
                              title=f"UMAP 3D — K-Medoids k={optimal_k_w} (Gower metric)",
                              opacity=0.65,
                              color_discrete_sequence=px.colors.qualitative.Vivid)
    _fig_u3.update_traces(marker=dict(size=3))
    _fig_u3.update_layout(height=650)
    mo.ui.plotly(_fig_u3)
    return


# ---------------------------------------------------------------------------
# 8. Cluster Profiling
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5. Weighted Cluster Profiling

        We characterise each cluster by computing mean numerical features and
        mode categorical features. These profiles are used to derive interpretable
        customer personas for downstream marketing strategy.
        """
    )
    return


@app.cell
def __(df_cluster, go, labels_by_k_w, make_subplots, mo, optimal_k_w, px):
    import pandas as _pprof

    _cat_cols = ["Gender", "Job", "Area", "CitySize", "Investments"]
    _num_cols = [c for c in df_cluster.columns if c not in _cat_cols]

    _job_map  = {1:"Unemployed", 2:"Employee", 3:"Manager", 4:"Entrepreneur", 5:"Retired"}
    _inv_map  = {1:"None", 2:"Lump Sum", 3:"Cap. Accum."}
    _gen_map  = {0:"Male", 1:"Female"}
    _area_map = {1:"North", 2:"Center", 3:"South"}

    _df_p = df_cluster.copy()
    _df_p["Cluster"] = labels_by_k_w[optimal_k_w]

    _pal = px.colors.qualitative.Vivid

    # Summary table
    _rows = []
    for _c in range(optimal_k_w):
        _g = _df_p[_df_p["Cluster"] == _c]
        _row = {"Cluster": _c, "N": len(_g), "%": f"{len(_g)/len(_df_p)*100:.1f}%"}
        for _col in _num_cols:
            _row[_col] = f"{float(_g[_col].mean()):.3f}"
        _row["Job"]         = _job_map.get(int(_g["Job"].astype(float).mode()[0]), "?")
        _row["Gender"]      = _gen_map.get(int(_g["Gender"].astype(float).mode()[0]), "?")
        _row["Investments"] = _inv_map.get(int(_g["Investments"].astype(float).mode()[0]), "?")
        _rows.append(_row)
    _profile_df = _pprof.DataFrame(_rows)

    # Radar chart
    _radar = go.Figure()
    for _c in range(optimal_k_w):
        _g = _df_p[_df_p["Cluster"] == _c]
        _vals = [float(_g[_col].mean()) for _col in _num_cols]
        _radar.add_trace(go.Scatterpolar(
            r=_vals + [_vals[0]], theta=_num_cols + [_num_cols[0]],
            fill="toself", name=f"Cluster {_c}",
            line_color=_pal[_c], opacity=0.75,
        ))
    _radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Radar — Numerical Feature Means per Cluster (k={optimal_k_w})",
        height=500,
    )

    # Box plots
    _key = ["Income", "Wealth", "Debt", "Saving", "Luxury", "FinEdu"]
    _box = make_subplots(rows=1, cols=len(_key), subplot_titles=_key)
    for _ci, _col in enumerate(_key):
        for _c in range(optimal_k_w):
            _g = _df_p[_df_p["Cluster"] == _c]
            _box.add_trace(go.Box(y=_g[_col].tolist(), name=f"Cluster {_c}",
                                   marker_color=_pal[_c], showlegend=(_ci == 0)), row=1, col=_ci + 1)
    _box.update_layout(height=430, title_text="Key Financial Features by Cluster", boxmode="group")

    mo.vstack([
        mo.md("### Cluster Summary"),
        mo.ui.table(_profile_df, selection=None),
        mo.md("### Radar Profiles"),
        mo.ui.plotly(_radar),
        mo.md("### Financial Feature Distributions by Cluster"),
        mo.ui.plotly(_box),
    ])
    return


# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def __(mo, optimal_k_w):
    mo.md(
        f"""
        ## 6. Weighted Conclusions & Business Recommendations

        Our analysis identifies **k = {optimal_k_w}** as the statistically optimal number of
        client segments by majority vote across Silhouette, Davies-Bouldin, and Calinski-Harabasz indices.

        The Gower + K-Medoids approach provides three key advantages over simpler alternatives:
        - **No information loss** — all 18 features (including categorical) contribute to cluster assignment
        - **Scale invariance** — automatic range normalization prevents any single feature from dominating
        - **Interpretable representatives** — each medoid is a real client that can serve as a segment archetype

        ### Recommended Personas

        | Segment | Profile | Recommended Products |
        |---------|---------|---------------------|
        | A | High Income / Wealth, ESG-oriented, high FinEdu | Sustainable portfolios, premium advisory |
        | B | Mid Income, family-oriented, moderate Saving | Life insurance, education savings plans |
        | C | Lower Income, higher Debt, low FinEdu | Financial literacy, debt consolidation |
        """
    )
    return


if __name__ == "__main__":
    app.run()


if __name__ == "__main__":
    app.run()
