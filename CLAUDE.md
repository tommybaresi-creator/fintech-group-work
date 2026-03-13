# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **group project** (4 team members) for fintech customer segmentation, analyzing 5,000 bank clients across 18 features (demographics, financial metrics, behavioral scores). Each team member has a specific task to implement different clustering methodologies.

**Technology Stack:** Python 3.13, Jupyter notebooks, scikit-learn, pandas, Gower distance metrics

### Team Task Division

1. **Simo** - Exploratory Data Analysis (EDA)
   - Notebook: `EDA_BankClients_simo.ipynb`
   - Completed: Data quality checks, outlier detection, domain anomaly analysis

2. **Tommy** - K-Prototypes Clustering
   - Notebook: `Analysis.ipynb`
   - Completed: K-Prototypes implementation with Euclidean + Hamming distance

3. **Alberto (You)** - Main Clustering Implementation (K-Medoids + Gower Distance)
   - Notebook: `Clustering_Alberto.ipynb` (to be created)
   - Task: Implement K-Medoids clustering with Gower distance for mixed data
   - This is the recommended approach for handling mixed categorical/numerical data

4. **Team Member 4** - TBD (possibly Bayesian updates or alternative methods)

**Reference Materials (PoliMI Course):**
- `SegmentingClientsPoliMI.ipynb` - Complete clustering pipeline reference
- `Updating_PersonasPoliMI.ipynb` - Bayesian persona updating
- `AdditionalContentPoliMI.ipynb` - LVQ alternative approach

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate

# CRITICAL: Install numpy<2.0.0 first (scikit-learn-extra compatibility)
pip install 'numpy<2.0.0'

# Install dependencies
pip install -r BusinessCase1/requirements.txt
```

### Running Jupyter Notebooks
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Launch Jupyter
jupyter notebook

# Team notebooks (in execution order):
# 1. EDA_BankClients_simo.ipynb - Simo's EDA work
# 2. Analysis.ipynb - Tommy's K-Prototypes clustering
# 3. Clustering_Alberto.ipynb - Alberto's K-Medoids + Gower (YOUR TASK)

# Reference notebooks from PoliMI course:
# - SegmentingClientsPoliMI.ipynb - Complete clustering reference
# - Updating_PersonasPoliMI.ipynb - Bayesian updates
# - AdditionalContentPoliMI.ipynb - LVQ approach
```

### Git Workflow
```bash
# Main branch for PRs: main
# Current collaborative branches: alberto, tommy, simonezani

# Standard workflow
git checkout -b your-branch
# Make changes...
git add .
git commit -m "Description"
git push origin your-branch
```

## Project Architecture

### Core Data Structures

**Dataset:** 5,000 bank clients with 18 features
- **Categorical (5):** Gender, Job, Area, CitySize, Investments
- **Numerical (13):** Age, FamilySize, Income, Wealth, Debt, FinEdu, ESG, Digital, BankFriend, LifeStyle, Luxury, Saving
- **Key insight:** Numerical features are normalized to [0,1] percentiles, representing relative position in distribution

**Data Location:** `BusinessCase1/Data/Dataset1_BankClients.xlsx`

### High-Level Pipeline Flow

```
Raw Data → EDA (outlier detection, validation)
    ↓
Distance Matrix Computation (Gower metric)
    ↓
Three Clustering Approaches:
├─ K-Medoids + Gower (primary) → Personas
├─ K-Prototypes (alternative)  → Clusters
└─ LVQ (supervised alternative) → Prototypes
    ↓
Bayesian Updates (progressive personalization)
```

### Critical Design Decisions

#### 1. Mixed-Data Clustering Strategy
**Problem:** Standard distance metrics (Euclidean, cosine) don't handle mixed categorical/numerical data properly.

**Solution:** Gower distance + K-Medoids
- Gower distance: Combines normalized Euclidean (numerical) + Jaccard (categorical)
- K-Medoids: Uses actual data points as centroids (unlike K-means, which computes means that don't work for categoricals)
- Implementation: `metric='precomputed'` with precomputed Gower distance matrix

**Alternatives implemented:**
- K-Prototypes: Euclidean + Hamming hybrid (simpler but less flexible)
- LVQ: One-hot encode categoricals, standard scale (requires labels)

#### 2. Optimal k Selection via Voting Scheme
Rather than choosing a single metric, implements democratic validation:
- Compute Calinski-Harabasz (maximize), Davies-Bouldin (minimize), Silhouette (maximize)
- Each metric "votes" for optimal k
- Final decision: median of recommendations
- **Rationale:** Prevents metric-specific biases, increases robustness

#### 3. Distance Metric Polyarchy
The codebase doesn't commit to a single distance metric. t-SNE visualization is performed with:
- Cityblock (L1), Cosine, Euclidean (L2), Gower
- **Insight:** Distance metric choice is a modeling decision, not technical—different metrics reveal different cluster structures

#### 4. Bayesian Persona Updates
**Purpose:** Transition from generic "cluster persona" to personalized "individual profile" as new data arrives.

**Beta-Binomial Approach (univariate):**
```
Prior: α = persona_value × N, β = (1 - persona_value) × N
Update: α' = α + x × N, β' = β + (1 - x) × N
(N = "information weight", initially 20)
```

**Multivariate Gaussian (with Kalman gain):**
- Problem: Features in [0,1] but Gaussian assumes ℝ
- Solution: Logit transform → Bayesian update → Inverse logit
- Accounts for feature correlations via covariance matrix

#### 5. Dimensionality Reduction Philosophy
Two complementary approaches:
- **t-SNE:** Non-linear, preserves local structure, used for cluster visualization
- **PCA:** Linear, interpretable variance, used for global structure
- **Why both?** t-SNE can create misleading "clusters" that don't exist; PCA validates global patterns

### Key Business Domain Concepts

**Financial Personas:** Human-interpretable customer archetypes with:
- Demographic profile (age, job, location, family)
- Financial capacity (income, wealth, debt)
- Behavioral propensities (digital adoption, ESG affinity, luxury consumption)
- Product affinity (investment type preferences)

**Behavioral Features (normalized [0,1]):**
- **FinEdu:** Financial education/sophistication level
- **ESG:** Environmental, Social, Governance investing propensity
- **Digital:** Digital banking channel adoption
- **BankFriend:** Relationship orientation with bank
- **LifeStyle:** Lifestyle spending index
- **Luxury:** Luxury goods consumption propensity
- **Saving:** Savings behavior intensity

**Clustering Challenge:** Categorical variables (e.g., Job) create "natural clusters" that can dominate distance metrics, obscuring meaningful behavioral patterns. This is why Gower distance (equal weighting) and multiple validation metrics are critical.

### EDA Pipeline (EDA_BankClients_simo.ipynb)

**Data Quality Checks:**
- Missing values: 0
- Duplicates: 0
- Range validation for normalized features [0,1]

**Domain Anomaly Detection (logical inconsistencies):**
- Young retirees (Age < 50, Job=Retired): 57 cases
- Rich unemployed: 28 cases
- Wealthy with no investments: 8 cases
- Young large families: 24 cases

**Outlier Detection:**
- Isolation Forest with contamination=0.01
- Flags 50 multivariate outliers for review

### Main Clustering Pipeline (SegmentingClientsPoliMI.ipynb)

**Step-by-step process:**
1. Load data, drop ID column
2. Separate categorical vs numerical features
3. One-hot encode categoricals (drop_first=True)
4. Normalize numericals (MinMaxScaler or StandardScaler)
5. Compute Gower distance matrix (N×N for 5,000 clients)
6. Run K-Medoids for k ∈ {3, 4, 5, 6}
7. Evaluate each k with Calinski-Harabasz, Davies-Bouldin, Silhouette
8. Apply voting scheme to select optimal k
9. Profile clusters: compute means, medians, modes per cluster
10. Visualize: radar charts, box plots, pair plots, t-SNE/PCA projections
11. Define business personas from statistical profiles

**Alternative: DBSCAN**
- Density-based clustering, handles noise/outliers
- Epsilon tuning via grid search on Gower distance matrix
- Use when number of clusters unknown or outliers significant

## Important Notes

### This is Research/Exploratory Code
- No production infrastructure (APIs, model serialization, orchestration)
- No unit tests or CI/CD pipelines
- Jupyter-centric architecture emphasizing narrative documentation
- Likely academic project (PoliMI references in filenames)

### Collaborative Development Pattern
- Multiple team members working on different methodological approaches
- Each notebook represents a distinct analytical perspective
- Integration happens at git merge level, not code level

### Data Directory
The `Data` folder is gitignored but contains critical files:
- `Dataset1_BankClients.xlsx` (938KB)
- `BankClients_Metadata.xlsx`
If working from a fresh clone, ensure data files are available locally.

### NumPy Version Constraint
Must use `numpy<2.0.0` due to scikit-learn-extra compatibility. Always install numpy first before other dependencies.

## Alberto's Task: K-Medoids Clustering with Gower Distance

### Objective
Implement the main clustering solution using K-Medoids algorithm with Gower distance metric. This approach properly handles mixed categorical/numerical data, unlike standard K-means.

### Why K-Medoids + Gower?
1. **Gower Distance** handles mixed data types correctly:
   - Numerical features: Normalized Euclidean distance
   - Categorical features: Jaccard/Hamming distance (matching)
   - Equal weighting prevents categorical dominance

2. **K-Medoids** uses actual data points as centroids:
   - Works with precomputed distance matrices
   - No "mean" calculation needed (which fails for categorical data)
   - More robust to outliers than K-means

3. **Different from Tommy's K-Prototypes**:
   - K-Prototypes: Euclidean + Hamming hybrid with gamma weighting
   - K-Medoids + Gower: Precomputed distance matrix, more flexible

### Implementation Steps (Your Notebook)

**Step 1: Load Clean Data from Simo's EDA**
```python
# Load the dataset
import pandas as pd
from pathlib import Path

data_path = Path("Data") / "Dataset1_BankClients.xlsx"
df = pd.read_excel(data_path)

# Drop ID column
df = df.drop(columns=['ID'])

# Define feature types (from Simo's work)
categorical_cols = ['Gender', 'Job', 'Area', 'CitySize', 'Investments']
numerical_cols = ['Age', 'FamilySize', 'Income', 'Wealth', 'Debt',
                  'FinEdu', 'ESG', 'Digital', 'BankFriend',
                  'LifeStyle', 'Luxury', 'Saving']
```

**Step 2: Compute Gower Distance Matrix**
```python
import gower

# Gower distance automatically detects categorical vs numerical
# For 5,000 clients, this creates a 5000x5000 matrix
distance_matrix = gower.gower_matrix(df)

# This is a symmetric matrix where distance_matrix[i,j] = distance between client i and j
```

**Step 3: Run K-Medoids for Multiple k Values**
```python
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

results = {}

for k in range(3, 7):  # Test k = 3, 4, 5, 6
    # K-Medoids with precomputed Gower distances
    kmedoids = KMedoids(
        n_clusters=k,
        metric='precomputed',
        init='k-medoids++',
        random_state=42
    )

    clusters = kmedoids.fit_predict(distance_matrix)

    # Compute validation metrics
    silhouette = silhouette_score(distance_matrix, clusters, metric='precomputed')
    davies_bouldin = davies_bouldin_score(distance_matrix, clusters)
    calinski = calinski_harabasz_score(distance_matrix, clusters)

    results[k] = {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski,
        'clusters': clusters
    }
```

**Step 4: Apply Voting Scheme to Select Optimal k**
```python
# For each metric, which k is best?
votes = {}

# Silhouette: maximize
best_k_silhouette = max(results.keys(), key=lambda k: results[k]['silhouette'])
votes[best_k_silhouette] = votes.get(best_k_silhouette, 0) + 1

# Davies-Bouldin: minimize
best_k_db = min(results.keys(), key=lambda k: results[k]['davies_bouldin'])
votes[best_k_db] = votes.get(best_k_db, 0) + 1

# Calinski-Harabasz: maximize
best_k_ch = max(results.keys(), key=lambda k: results[k]['calinski_harabasz'])
votes[best_k_ch] = votes.get(best_k_ch, 0) + 1

# Optimal k = median of votes
optimal_k = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]
print(f"Optimal k by voting: {optimal_k}")
```

**Step 5: Visualize Clusters with t-SNE**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# t-SNE with Gower distances for 2D visualization
tsne = TSNE(n_components=2, metric='precomputed', random_state=42)
tsne_result = tsne.fit_transform(distance_matrix)

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    c=results[optimal_k]['clusters'],
    cmap='viridis',
    alpha=0.6
)
plt.colorbar(scatter, label='Cluster')
plt.title(f't-SNE Visualization of K-Medoids Clusters (k={optimal_k})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

**Step 6: Profile Each Cluster (Create Personas)**
```python
# Add cluster assignments to dataframe
df['Cluster'] = results[optimal_k]['clusters']

# For each cluster, compute statistics
for cluster_id in range(optimal_k):
    print(f"\n=== Cluster {cluster_id} Profile ===")
    cluster_data = df[df['Cluster'] == cluster_id]

    print(f"Size: {len(cluster_data)} clients ({len(cluster_data)/len(df)*100:.1f}%)")

    # Numerical features: mean
    print("\nNumerical Features (Mean):")
    print(cluster_data[numerical_cols].mean())

    # Categorical features: mode
    print("\nCategorical Features (Most Common):")
    for col in categorical_cols:
        mode_val = cluster_data[col].mode()[0]
        print(f"{col}: {mode_val}")
```

**Step 7: Compare with Tommy's K-Prototypes Results**
```python
# Load Tommy's results (if he saved cluster assignments)
# Compare cluster compositions, silhouette scores, etc.
# Document differences in your notebook
```

**Step 8: Save Results**
```python
# Save clustered data
df.to_csv("BankClients_Clustered_Alberto.csv", index=False)

# Save visualization
plt.savefig("cluster_visualization_alberto.png", dpi=300, bbox_inches='tight')
```

### Expected Deliverables

1. **Jupyter Notebook**: `Clustering_Alberto.ipynb` with:
   - All steps documented with markdown explanations
   - Gower distance matrix computation
   - K-Medoids implementation for k ∈ {3, 4, 5, 6}
   - Voting scheme for optimal k selection
   - Cluster visualizations (t-SNE, PCA)
   - Persona profiles for each cluster
   - Comparison with Tommy's K-Prototypes approach

2. **Output Files**:
   - `BankClients_Clustered_Alberto.csv` - Dataset with cluster assignments
   - `cluster_visualization_alberto.png` - Main cluster visualization

3. **Analysis**:
   - Which k is optimal according to the voting scheme?
   - How do your clusters differ from Tommy's K-Prototypes clusters?
   - What are the business interpretations of each cluster (financial personas)?

### Key Differences from Tommy's Work

| Aspect | Tommy (K-Prototypes) | Alberto (K-Medoids + Gower) |
|--------|---------------------|---------------------------|
| Distance Metric | Euclidean + Hamming with gamma | Gower (normalized, equal weight) |
| Centroid Type | Computed (mean for num, mode for cat) | Actual data points (medoids) |
| Implementation | `kmodes` library | `sklearn_extra` + `gower` |
| Flexibility | Fixed gamma weighting | Automatic equal weighting |

### Working in Notebook Format

Since you prefer notebooks over .py files:
- Create a new notebook: `Clustering_Alberto.ipynb`
- Use markdown cells generously to explain each step
- Include visualizations inline
- Document your thought process and findings
- Make it readable for your teammates and professors
