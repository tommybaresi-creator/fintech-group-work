<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Marimo-000000?style=for-the-badge&logo=jupyter&logoColor=white" alt="Marimo Badge"/>
  <img src="https://img.shields.io/badge/UV-FF4B4B?style=for-the-badge&logo=rust&logoColor=white" alt="UV Badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML Badge"/>

  <h1>Bank Client Segmentation With Unsupervised Machine Learning Methods</h1>
  
  <p><b>Data-driven customer insights using K-Medoids Clustering and Gower Distance.</b></p>
</div>

---

## Overview

This project presents a comprehensive analysis of a portfolio of **5,000 retail bank clients**, described by 18 mixed-type features (numerical and categorical).

The objective is to identify distinct, actionable customer segments to inform targeted marketing and product personalisation strategies.

## Quickstart: Running the Marimo Notebook

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [marimo](https://marimo.io/) as a reactive notebook environment.

### 1. Setup & Precompute

Ensure `uv` is installed. The raw data must be preprocessed prior to viewing the notebook. The following script computes the Gower distance matrix and K-Medoids results:

```bash
uv run python marimo/precompute.py
```
> **Note:** Execution takes approximately 1–3 minutes and will populate the `marimo/results/` directory with distance matrices and clustering labels.

### 2. Run the Notebook

Launch the interactive Marimo notebook:

```bash
uv run marimo edit marimo/bank_clients_analysis.py
```

---

## Methodology & Pipeline

Mixed-type data is handled via **Gower Distance**, and clients are clustered using **K-Medoids** (FasterPAM algorithm).

<p align="center">
  <img src="./marimo/assets/gower_kmedoids_pipeline.svg" width="80%" alt="Pipeline">
</p>

### Rationale for Gower Distance & K-Medoids

- **Gower Distance**: Natively accommodates mixed data types (categorical and numerical) without requiring one-hot encoding, preserving the ordinal structure of variables.
- **K-Medoids**: Selects actual data points (clients) as cluster representatives (medoids), producing interpretable segment profiles suitable for business use.

<p align="center">
  <img src="./marimo/assets/kmedoids_explainer.svg" width="80%" alt="K-Medoids Explainer">
</p>

---

## Client Segments (UMAP Projections)

Clusters were evaluated for $k \in \{3, 4, 5, 6\}$ using the Silhouette Score and K-Medoids inertia (elbow method) to determine the optimal number of segments.

### $k = 4$ Client Clusters

<p align="center">
  <img src="./marimo/assets/umap_clusters_k4.png" width="85%" alt="UMAP k=4">
</p>

### $k = 5$ Client Clusters

<p align="center">
  <img src="./marimo/assets/umap_clusters_k5.png" width="85%" alt="UMAP k=5">
</p>

---

<div align="center">
  <i>Developed for Fintech Course</i>
</div>
