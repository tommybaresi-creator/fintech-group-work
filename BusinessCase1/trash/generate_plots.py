import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(r"c:\Users\giuli\Repositories\fintech-group-work\BusinessCase1\marimo\results")
ASSETS_DIR = Path(r"c:\Users\giuli\Repositories\fintech-group-work\BusinessCase1\marimo\assets")

# Load data
print("Loading data...")
umap2d = np.load(RESULTS_DIR / "umap2d.npy")
labels_k4 = np.load(RESULTS_DIR / "labels_k4.npy")
labels_k5 = np.load(RESULTS_DIR / "labels_k5.npy")

# Set aesthetic style
sns.set_theme(style="whitegrid")

# Generate k=4 plot
print("Generating k=4 plot...")
plt.figure(figsize=(12, 8))
scatter4 = sns.scatterplot(
    x=umap2d[:, 0], y=umap2d[:, 1], 
    hue=labels_k4, palette="deep", 
    s=40, alpha=0.8, edgecolor="w", linewidth=0.5
)
plt.title("UMAP Projection of Bank Client Clusters (k=4)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.legend(title="Client Segment", title_fontsize='13', fontsize='12', frameon=True, shadow=True)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "umap_clusters_k4.png", dpi=300, bbox_inches='tight')
plt.close()

# Generate k=5 plot
print("Generating k=5 plot...")
plt.figure(figsize=(12, 8))
scatter5 = sns.scatterplot(
    x=umap2d[:, 0], y=umap2d[:, 1], 
    hue=labels_k5, palette="magma", 
    s=40, alpha=0.8, edgecolor="w", linewidth=0.5
)
plt.title("UMAP Projection of Bank Client Clusters (k=5)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.legend(title="Client Segment", title_fontsize='13', fontsize='12', frameon=True, shadow=True)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "umap_clusters_k5.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved successfully in assets directory!")
