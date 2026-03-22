<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/UV-FF4B4B?style=for-the-badge&logo=rust&logoColor=white" alt="UV Badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML Badge"/>

  <h1>Bank Client Segmentation With Unsupervised Machine Learning Methods</h1>
  
  <p><b>Data-driven customer insights using mixed-data clustering, dimensionality reduction, and Bayesian profile updating.</b></p>
</div>

---

## Abstract

This repository presents a comprehensive unsupervised machine learning pipeline designed to organically partition a portfolio of **5,000 retail bank clients** into interpretable segments. The dataset encompasses 18 features—predominantly continuous financial metrics interwoven with categorical demographic profiles (e.g., gender, profession, area). 

The primary objective is to reliably extract actionable, cross-sectional archetypes (hereafter termed *Financial Personas*) while strictly adhering to rigorous distance metric formalizations that do not rely on mathematically incongruous assumptions (such as evaluating the Euclidean mean of categorical variables).

## Methodological Framework

### 1. Mixed-Type Distance Metrics
To mitigate the distortion induced by standard Euclidean norms on categorical variables, our clustering approach fundamentally relies on robust mixed-data formulations. 
Specifically, the distance between two multivariate observations leverages varying combinations of numerical norms (L1/L2/Canberra) and categorical dissimilarities (Hamming/Tanimoto).
- **Gower's Distances**: Initially explored as a baseline for naturally evaluating numerical and categorical data within a structurally consistent range $[0, 1]$.
- **Hyperparameter Optimization ($\alpha$-Sweeping)**: A systematic tuning of the $\alpha$ coefficient—thereby balancing the weight of continuous versus discrete features—identified that categorical signals predominantly dictate meaningful structural parcellation, confirming **L1 + Tanimoto** combinations with $\alpha = 0.2$ as the highest-performing metric.

### 2. Clustering Approaches
- **K-Medoids (FasterPAM)**: Distinct from native K-Means, K-Medoids designates an actual data point as the core cluster exemplar (the medoid). This preserves the interpretability needed for constructing concrete business profiles.

### 3. Dimensionality Reduction & Validation
- **UMAP (Uniform Manifold Approximation and Projection)**: Deployed to compress the 17-dimensional space while preserving local topological structures. K-medoids is then applied directly to this non-linearly mapped latent space.
- **Adjusted Rand Index (ARI)**: Adopted as a strict empirical measure to quantify alignment across distance-space and latent-space clustered sets. The findings exhibit an ARI of up to $0.988$, evidencing near-perfect methodological consistency. Additional visualization is handled via **PCA** and **t-SNE**.

### 4. Bayesian Inference & Persona Updating
The formulated static *Financial Personas* serve as a prior distribution for iterative refinement. Through a **Beta-Binomial Bayesian framework**, client attributes undergo continuous updates. This recursive methodology dynamically aligns broader segment-level profiles directly with incoming subject-specific numerical data.

---

## Key Findings & Results

- **Categorical Dominance**: Expanding cross-sectional data through $\alpha$-sweeping conclusively confirmed that discrete demographic traits heavily outweigh continuous variables in structuring behavior. Assigning purely categorical blocks an 80% operational weight ($\alpha=0.2$) drastically increased partition quality.
- **Optimum Metric Isolation**: **L1 norm + Tanimoto coefficient** emerged as the universally superior mixed-metric configuration. By actively demanding positive evidence of shared class membership, Tanimoto drastically outperformed basic Hamming overlaps, yielding an impressive **Silhouette Score of ~0.69** ($k=10$) in the native high-dimensional distance space.
- **Cross-Space Stability**: Stress-testing conventional 17-dimensional distance-space clusters against a compressed 12-dimensional UMAP manifold (which achieved a **Silhouette Score of ~0.85** for $k=6$) produced an Adjusted Rand Index (ARI) of **0.82–0.98**. This extraordinary coherence confirms the integrity of the segments as genuine client groups rather than geometric artifacts.
- **Primary Segments**: The optimal configuration reliably isolates three dominant, highly actionable client portfolios that together encompass roughly 75% of the examined data subset.

---

## Technical Appendix: Notebook Structure

The analysis is procedurally structured across several interconnected notebooks housed in the `BusinessCase1/` directory:

| Notebook | Focus Area |
|----------|------------|
| `clustering.ipynb` | Formalization of distance definitions. Computes alpha weightings and rigorously compares Tanimoto/Hamming categorizations against L1/L2 numerical norms. |
| `clustering_weighted.ipynb` | Executes exhaustive hyperparameter tuning ($\alpha$-sweeping) to deduce the exact optimum weight distributions across distance functions. |
| `clustering_umap.ipynb` | Leverages UMAP manifold compressions as an independent clustering benchmark to aggressively strain-test distance-space assignments. |
| `comparison.ipynb` | Yields the definitive ARI matrix demonstrating near-perfect congruence among unweighted, weighted, and latent-space methodologies. |

---

## Quickstart: Execution Environment

This project is built atop [uv](https://github.com/astral-sh/uv) to securely bundle dependency scopes safely.

### 1. Bootstrapping & Feature Precomputation

First, ascertain that `uv` is installed on your local environment. 

> **Note:** Base distance matrix computation typically evaluates over a 1–3 minute runtime epoch. Output caches will selectively populate within the operational directory.

To replicate the full analysis, execute the notebooks in sequential order. For computational efficiency, pre-computed distance matrices are available for download:

<div align="center">
  <a href="https://drive.google.com/drive/folders/1oXuBDXjM8oniLixNXOPpyIpglZ2fk1R0?usp=sharing">
    <img src="https://img.shields.io/badge/Google%20Drive-Cached%20Distances-4285F4?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive Cache"/>
  </a>
</div>

<div align="center">
  <i>Developed for Fintech Course</i>
</div>
