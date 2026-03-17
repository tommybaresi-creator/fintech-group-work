import re

with open("bank_clients_analysis.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Title / Index
old_title = r'''        # Bank Client Segmentation — EDA & K-Medoids Clustering

        We analyse a portfolio of \*\*5,000 retail bank clients\*\* described by \*\*18 mixed-type features\*\*
        \(13 numerical, 5 categorical\)\. Our goal is to identify distinct, actionable customer segments
        that can inform targeted marketing and product personalisation strategies\.

        We adopt a \*\*K-Medoids clustering\*\* approach with \*\*Gower distance\*\* — a deliberate methodological
        choice driven by the mixed-type structure of the data and the need for interpretable, robust segments\.'''

new_title = r'''        # Bank Client Segmentation — EDA & K-Medoids Clustering

        **Executive Summary (Abstract)**: We analyse a portfolio of **5,000 retail bank clients** described by **18 mixed-type features**
        (13 numerical, 5 categorical). Our goal is to identify distinct, actionable customer segments
        that can inform targeted marketing and product personalisation strategies. We adopt a **K-Medoids clustering** approach with **Gower distance**, performing standard (unweighted) clustering alongside a **weighted alternative** that assigns double importance to the *Job* and *Investments* features.

        **Index**:
        - [Part 1 — Exploratory Data Analysis & Outlier Detection](#part-1)
        - [Part 2 — Unweighted K-Medoids Clustering](#part-2)
        - [Part 3 — Weighted K-Medoids Clustering](#part-3)'''

content = re.sub(old_title, new_title, content)

# 2. Add anchors to parts
content = content.replace(
    'mo.md("---\\n# Part 1 — Exploratory Data Analysis & Outlier Detection")',
    'mo.md("<a id=\'part-1\'></a>\\n---\\n# Part 1 — Exploratory Data Analysis & Outlier Detection")'
)

content = content.replace(
    'mo.md("---\\n# Part 2 — K-Medoids Clustering")',
    'mo.md("<a id=\'part-2\'></a>\\n---\\n# Part 2 — Unweighted K-Medoids Clustering")'
)

# 3. Update paths / load logic
load_func_old = r'''        required = \[
            "df_cluster\.parquet", "distance_matrix\.npy", "metrics\.json",
            "labels_k3\.npy", "labels_k4\.npy", "labels_k5\.npy", "labels_k6\.npy",
            "umap2d\.npy", "umap3d\.npy",
        \]'''

load_func_new = r'''        required = [
            "df_cluster.parquet", "distance_matrix.npy", "metrics.json",
            "labels_k3.npy", "labels_k4.npy", "labels_k5.npy", "labels_k6.npy",
            "umap2d.npy", "umap3d.npy",
            "distance_matrix_w.npy", "metrics_w.json",
            "labels_k3_w.npy", "labels_k4_w.npy", "labels_k5_w.npy", "labels_k6_w.npy",
            "umap2d_w.npy", "umap3d_w.npy",
        ]'''
content = re.sub(load_func_old, load_func_new, content)

load_vars_old = r'''    df_cluster = pd\.read_parquet\(RESULTS / "df_cluster\.parquet"\)
    distance_matrix = np\.load\(RESULTS / "distance_matrix\.npy"\)

    with open\(RESULTS / "metrics\.json"\) as _f:
        _raw_metrics = json\.load\(_f\)
    metrics_dict = \{int\(k\): v for k, v in _raw_metrics\.items\(\)\}

    K_RANGE   = sorted\(metrics_dict\.keys\(\)\)
    labels_by_k = \{k: np\.load\(RESULTS / f"labels_k\{k\}\.npy"\) for k in K_RANGE\}
    umap2d    = np\.load\(RESULTS / "umap2d\.npy"\)
    umap3d    = np\.load\(RESULTS / "umap3d\.npy"\)'''

load_vars_new = r'''    df_cluster = pd.read_parquet(RESULTS / "df_cluster.parquet")
    
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
    umap3d_w  = np.load(RESULTS / "umap3d_w.npy")'''
content = re.sub(load_vars_old, load_vars_new, content)

return_old = r'''    return \(
        K_RANGE,
        df_cluster,
        distance_matrix,
        labels_by_k,
        metrics_dict,
        umap2d,
        umap3d,
    \)'''

return_new = r'''    return (
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
    )'''
content = re.sub(return_old, return_new, content)

# Extract cells to copy. We will regex search for @app.cell and extract all cells starting from Gower Distribution to Summary.
cells = content.split('\n@app.cell\n')
# cell 0 is everything before first app.cell. cells[1:] are the actual cells.

part3_cells = []
start_copying = False
for c in cells[1:]:
    if 'mo.md(\n        r"""\n        ## 3. Gower Distance Distribution' in c:
        start_copying = True
    if start_copying:
        # copy cell text
        c_w = c
        # Replace Markdown headers
        c_w = c_w.replace('## 3. Gower Distance Distribution', '<a id="part-3"></a>\\n---\\n# Part 3 — Weighted K-Medoids Clustering\\n\\n## 1. Weighted Gower Distance Distribution')
        c_w = c_w.replace('## 4. K-Medoids Algorithm', '## 2. K-Medoids Algorithm (Weighted)')
        c_w = c_w.replace('## 5. Cluster Validation & Optimal k Selection', '## 3. Weighted Cluster Validation')
        c_w = c_w.replace('## 6. Cluster Visualizations', '## 4. Weighted Cluster Visualizations')
        c_w = c_w.replace('### 6.1 PCA — Linear Projection of Numerical Features', '### 4.1 PCA (Weighted Labels)')
        c_w = c_w.replace('### 6.2 UMAP 2D', '### 4.2 UMAP 2D (Weighted)')
        c_w = c_w.replace('### 6.3 UMAP 3D', '### 4.3 UMAP 3D (Weighted)')
        c_w = c_w.replace('## 7. Cluster Profiling & Business Personas', '## 5. Weighted Cluster Profiling')
        c_w = c_w.replace('## 8. Conclusions & Business Recommendations', '## 6. Weighted Conclusions & Business Recommendations')
        
        c_w = c_w.replace('The pairwise Gower matrix', 'The **weighted** pairwise Gower matrix (Job and Investments x2)')

        # Replace variables mapping
        c_w = c_w.replace('distance_matrix', 'distance_matrix_w')
        c_w = c_w.replace('metrics_dict', 'metrics_dict_w')
        c_w = c_w.replace('labels_by_k', 'labels_by_k_w')
        c_w = c_w.replace('umap2d', 'umap2d_w')
        c_w = c_w.replace('umap3d', 'umap3d_w')
        
        c_w = c_w.replace('summary_df', 'summary_df_w')
        c_w = c_w.replace('optimal_k', 'optimal_k_w')
        
        # fix the def call signature
        # We replace "def __(distance_matrix_w," with "def __(distance_matrix_w," which is fine, marimo auto-detects dependencies from the parameter list.
        # But any `summary_df` returned must be renamed `summary_df_w`.
        c_w = c_w.replace('return summary_df_w,', 'return summary_df_w,')
        c_w = c_w.replace('return (optimal_k_w,)', 'return (optimal_k_w,)')
        
        # we append it to part3_cells
        part3_cells.append('\n@app.cell\n' + c_w)

# Rejoin original content (which has cells[0])
base_content = cells[0] + '\n@app.cell\n' + '\n@app.cell\n'.join(cells[1:])

# Remove the `if __name__ == "__main__":` from base_content to append new cells before it
base_content = base_content.replace('if __name__ == "__main__":\n    app.run()\n', '')

# Append part3_cells
final_content = base_content + "".join(part3_cells) + '\n\nif __name__ == "__main__":\n    app.run()\n'

with open("bank_clients_analysis_new.py", "w", encoding="utf-8") as f:
    f.write(final_content)
print("Generated bank_clients_analysis_new.py with valid app.cells!")
