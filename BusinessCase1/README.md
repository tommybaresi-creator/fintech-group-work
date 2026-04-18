<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/UV-FF4B4B?style=for-the-badge&logo=rust&logoColor=white" alt="UV Badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML Badge"/>

  <h1>Bank Client Segmentation With Unsupervised Machine Learning</h1>
  <p><b>Mixed-type clustering, dimensionality reduction on a 5,000-client retail bank portfolio.</b></p>
</div>

---

## Overview

The pipeline segments 5,000 retail bank clients (18 features: continuous financial metrics + categorical demographics) into interpretable *Financial Personas* using distance metrics that handle mixed data correctly — no Euclidean means applied to categorical variables.

## Method

**Distance metrics.** Six mixed-type metrics are evaluated, combining numerical norms (L1, L2, Canberra) with categorical dissimilarities (Hamming, Tanimoto), benchmarked against Gower's distance as baseline.

**Alpha sweep.** A systematic sweep over the numerical/categorical weight parameter α identifies the optimal balance. Categorical features dominate cluster structure: L1+Hamming at α=0.2 achieves the best silhouette (0.308, k=8), more than doubling Gower's baseline (0.141, k=3).

**Clustering.** K-Medoids (FasterPAM) assigns actual data points as cluster exemplars, preserving interpretability for business profiling.

**UMAP validation.** A 15-dimensional UMAP embedding is clustered independently (sil=0.431, k=10, stability ARI=0.891) and compared against the distance-space solution. Distance-space methods agree strongly with each other (ARI=0.824); UMAP finds different structure (ARI≈0.21), so the two approaches are treated as complementary views rather than converging evidence.

**Economic Valuation.** To assign tangible financial margins to behavioral segments, financial variables are projected into a non-leaking Customer Economic Score (CES) via PCA, providing objective high-value thresholds. Robust, leakage-free logistic classifiers map structural product propensities, calculating exact expected net € revenues by substituting global AUM assumptions with localized empirical cluster means.



## Notebook Structure

| Notebook | Focus |
|----------|-------|
| `clustering.ipynb` | Distance metric definitions, sanity checks, neutral-alpha clustering |
| `clustering_weighted.ipynb` | Alpha sweep and optimal metric selection |
| `clustering_umap.ipynb` | UMAP embedding, clustering, and stability testing |
| `comparison.ipynb` | ARI matrix and co-assignment analysis across all three methods |
| `economic_value_v2.ipynb` | Maps cluster variances into physical Euros via PCA scoring (CES), empirical AUM margins, and leakage-free propensities. |

## Setup

Built with [uv](https://github.com/astral-sh/uv). Pre-computed distance matrices (base computation: ~2–5 min per config) are available on Google Drive:

<div align="center">
  <a href="https://drive.google.com/drive/folders/1WHr3FO-oPKigeAqhSIEJamlWKMWciQyu?usp=sharing">
    <img src="https://img.shields.io/badge/Google%20Drive-Cached%20Distances-4285F4?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive Cache"/>
  </a>
</div>

<div align="center">
  <i>Developed for Fintech Course</i>
</div>