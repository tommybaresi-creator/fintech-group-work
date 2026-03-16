<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Marimo-000000?style=for-the-badge&logo=jupyter&logoColor=white" alt="Marimo Badge"/>
  <img src="https://img.shields.io/badge/UV-FF4B4B?style=for-the-badge&logo=rust&logoColor=white" alt="UV Badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML Badge"/>

  <h1>🏦 Bank Client Segmentation & EDA</h1>
  
  <p><b>Data-driven customer insights using K-Medoids Clustering and Gower Distance.</b></p>
</div>

---

## 📖 Overview

Welcome to **Business Case 1**. This project involves the comprehensive analysis of a portfolio of **5,000 retail bank clients**, described by 18 mixed-type features (numerical and categorical). 

Our goal is to identify distinct, actionable customer segments to inform targeted marketing and product personalization strategies. 

## 🚀 Quickstart: Running the Marimo Notebook

This project uses [uv](https://github.com/astral-sh/uv) for blazingly fast dependency management and [marimo](https://marimo.io/) as a reactive notebook environment.

### 1️⃣ Setup & Precompute
Ensure you have `uv` installed. The raw data needs to be preprocessed before viewing the notebook. We compute the Gower distance matrix and K-Medoids results using a standalone script:

```bash
uv run python marimo/precompute.py
```
> **Note:** This takes 1-3 minutes and will populate the `marimo/results/` folder with distance matrices and clustering labels.

### 2️⃣ Run the Notebook
Launch the interactive Marimo notebook:

```bash
uv run marimo edit marimo/bank_clients_analysis.py
```

---

## 🧠 Methodology & Pipeline

We handle the mixed-type data using **Gower Distance** and cluster the clients with **K-Medoids** (FasterPAM).

<p align="center">
  <img src="./marimo/assets/gower_kmedoids_pipeline.svg" width="80%" alt="Pipeline">
</p>

### Why Gower & K-Medoids?
- **Gower Distance**: Naturally handles mixed data types (categorical + numerical) without explosive one-hot encoding.
- **K-Medoids**: Chooses actual data points (clients) as cluster centers (medoids), making segment representatives highly interpretable for marketing personas.

<p align="center">
  <img src="./marimo/assets/kmedoids_explainer.svg" width="80%" alt="K-Medoids Explainer">
</p>

---

## 📊 Client Segments (UMAP Projections)

We evaluated clusters for $k \in \{3, 4, 5, 6\}$. By utilizing Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics, we identified optimal cluster numbers.

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
  <i>Developed with ❤️ for Fintech Group Work</i>
</div>
