"""
precompute.py — Run once to compute Gower distance matrix + K-Medoids results.

Saves to results/ folder:
  df_cluster.parquet       — preprocessed dataset used for clustering
  distance_matrix.npy      — Gower pairwise distance matrix (N x N float32)
  labels_k{k}.npy          — cluster labels for k = 3, 4, 5, 6
  medoids_k{k}.npy         — medoid row indices
  metrics.json             — Silhouette, Davies-Bouldin, Calinski-Harabasz per k
  umap2d.npy               — UMAP 2D embedding (N x 2)
  umap3d.npy               — UMAP 3D embedding (N x 3)

Usage:
    uv run python marimo/precompute.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import gower
import kmedoids
import umap

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_PATH = HERE.parent / "Data" / "Dataset1_BankClients.xlsx"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Gender", "Job", "Area", "CitySize", "Investments"]
NUMERICAL_COLS = [
    "Age", "FamilySize", "Income", "Wealth", "Debt",
    "FinEdu", "ESG", "Digital", "BankFriend",
    "LifeStyle", "Luxury", "Saving",
]

# ==========================================================================
# STEP 1 — Load raw data
# ==========================================================================
print("Step 1: Loading data...")
df_raw = pd.read_excel(DATA_PATH)
print(f"  Raw shape: {df_raw.shape}")

df = df_raw.copy()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# ==========================================================================
# STEP 2 — Sanity checks
# ==========================================================================
print("\nStep 2: Sanity checks...")
n_missing = df.isna().sum().sum()
n_dupes = df.duplicated().sum()
print(f"  Missing values: {n_missing}")
print(f"  Duplicate rows: {n_dupes}")

# Fill any missing values (defensive) using column median/mode
for col in NUMERICAL_COLS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
        print(f"  Filled NaN in {col} with median")

for col in CATEGORICAL_COLS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])
        print(f"  Filled NaN in {col} with mode")

# Drop duplicates
if n_dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Dropped {n_dupes} duplicates. New shape: {df.shape}")

# ==========================================================================
# STEP 3 — Enforce correct dtypes for Gower
#
# The `gower` library auto-detects categorical columns by dtype:
#   - object / category  → treated as categorical (Hamming distance)
#   - numeric (int/float) → treated as numerical (range-normalized Manhattan)
#
# IMPORTANT: categorical cols must be cast to `object` before passing to
# gower.gower_matrix() so they aren't mistakenly treated as ordered integers.
# ==========================================================================
print("\nStep 3: Enforcing correct dtypes for Gower distance...")
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(object)   # <-- critical: string/object → Hamming in Gower
    print(f"  {col}: cast to object (categorical)")

for col in NUMERICAL_COLS:
    df[col] = df[col].astype(float)    # ensure float for range-normalized Manhattan
    print(f"  {col}: cast to float64 (numerical)")

# Verify no remaining issues
assert df[CATEGORICAL_COLS].dtypes.apply(lambda d: d == object).all(), \
    "Some categorical cols are not object dtype!"
assert df[NUMERICAL_COLS].dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all(), \
    "Some numerical cols are not float dtype!"
print("  Dtype verification passed.")

# ==========================================================================
# STEP 4 — Domain anomaly removal (optional — working minors as hard filter)
# ==========================================================================
print("\nStep 4: Removing hard domain anomalies...")
n_before = len(df)

# Working minors (age < 18 in a financial dataset is a data error)
# Job column is now object dtype — compare as float
working_minors_mask = (
    df["Age"].astype(float) < 18
) & (
    df["Job"].astype(float).isin([2.0, 3.0, 4.0])
)
df = df[~working_minors_mask].reset_index(drop=True)

print(f"  Removed {n_before - len(df)} working minors. Shape: {df.shape}")

# ==========================================================================
# STEP 5 — Multivariate outlier removal (Isolation Forest)
#
# Isolation Forest uses only numerical features for scoring (it cannot
# handle mixed types natively). The top 1% most anomalous clients are flagged.
# ==========================================================================
print("\nStep 5: Isolation Forest outlier removal (numerical features, contamination=0.01)...")
iso = IsolationForest(contamination=0.01, random_state=42)
iso_labels = iso.fit_predict(df[NUMERICAL_COLS])
n_outliers = (iso_labels == -1).sum()
df = df[iso_labels == 1].reset_index(drop=True)
print(f"  Removed {n_outliers} multivariate outliers. Final shape: {df.shape}")

# ==========================================================================
# STEP 6 — Save preprocessed dataset
# ==========================================================================
print("\nStep 6: Saving preprocessed dataset...")
# Cast categoricals to str for parquet compatibility
df_save = df.copy()
for col in CATEGORICAL_COLS:
    df_save[col] = df_save[col].astype(str)
df_save.to_parquet(RESULTS_DIR / "df_cluster.parquet", index=False)
print(f"  Saved df_cluster.parquet ({len(df_save):,} rows)")

# ==========================================================================
# STEP 7 — Compute Gower distance matrices (Unweighted & Weighted)
#
# Weighted Gower gives 2x importance to 'Job' and 'Investments'
# ==========================================================================
print(f"\nStep 7: Computing Gower distance matrices ({len(df):,} x {len(df):,})...")
print("  This may take 2–5 minutes...")

weight_array = np.ones(df.shape[1], dtype=np.float32)
col_idx_job = df.columns.get_loc("Job")
col_idx_inv = df.columns.get_loc("Investments")
weight_array[col_idx_job] = 2.0
weight_array[col_idx_inv] = 2.0

matrices = {
    "": gower.gower_matrix(df).astype(np.float32),
    "_w": gower.gower_matrix(df, weight=weight_array).astype(np.float32)
}

for suffix, dist_matrix in matrices.items():
    print(f"\n  Matrix '{suffix or 'unweighted'}':")
    print(f"    Shape: {dist_matrix.shape}")
    print(f"    Range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    print(f"    Mean:  {dist_matrix.mean():.4f}")
    print(f"    Median:{float(np.median(dist_matrix)):.4f}")
    np.save(RESULTS_DIR / f"distance_matrix{suffix}.npy", dist_matrix)
    print(f"    Saved distance_matrix{suffix}.npy")

# ==========================================================================
# STEP 8 — K-Medoids (FasterPAM) for k = 3, 4, 5, 6
# ==========================================================================
print("\nStep 8: Running FasterPAM K-Medoids for k in {3, 4, 5, 6}...")
K_RANGE = [3, 4, 5, 6]

for suffix, dist_matrix in matrices.items():
    print(f"\n  --- {'WEIGHTED' if suffix else 'UNWEIGHTED'} ---")
    metrics_dict = {}
    for k in K_RANGE:
        print(f"\n    k={k}...")
        res = kmedoids.fasterpam(dist_matrix, k, random_state=42)
        labels = np.array(res.labels, dtype=np.int32)
        
        sil  = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
        db   = float(davies_bouldin_score(dist_matrix, labels))
        ch   = float(calinski_harabasz_score(dist_matrix, labels))
        
        unique, counts = np.unique(labels, return_counts=True)
        sizes = dict(zip(unique.tolist(), counts.tolist()))
        
        print(f"      sizes={sizes}  Sil={sil:.4f}  DB={db:.4f}  CH={ch:.2f}")
        
        np.save(RESULTS_DIR / f"labels_k{k}{suffix}.npy", labels)
        np.save(RESULTS_DIR / f"medoids_k{k}{suffix}.npy", np.array(res.medoids, dtype=np.int32))
        
        metrics_dict[k] = {
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "loss": float(res.loss),
            "sizes": sizes,
        }
        
    with open(RESULTS_DIR / f"metrics{suffix}.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved metrics{suffix}.json")

# ==========================================================================
# STEP 9 — UMAP embeddings
# ==========================================================================
print("\nStep 9: Computing UMAP embeddings (precomputed Gower)...")

for suffix, dist_matrix in matrices.items():
    print(f"\n  --- {'WEIGHTED' if suffix else 'UNWEIGHTED'} ---")
    print("    UMAP 2D...")
    reducer2 = umap.UMAP(n_components=2, metric="precomputed", n_neighbors=15, min_dist=0.1, random_state=42)
    emb2 = reducer2.fit_transform(dist_matrix).astype(np.float32)
    np.save(RESULTS_DIR / f"umap2d{suffix}.npy", emb2)

    print("    UMAP 3D...")
    reducer3 = umap.UMAP(n_components=3, metric="precomputed", n_neighbors=15, min_dist=0.1, random_state=42)
    emb3 = reducer3.fit_transform(dist_matrix).astype(np.float32)
    np.save(RESULTS_DIR / f"umap3d{suffix}.npy", emb3)

# ==========================================================================
# DONE
# ==========================================================================
print("\n=== Precompute complete ===")
print(f"Results saved to: {RESULTS_DIR}")
for f in sorted(RESULTS_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:35s}  {size_mb:6.1f} MB")
