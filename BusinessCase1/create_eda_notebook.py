import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = r"""
# Bank Clients - Exploratory Data Analysis (EDA) & Outlier Detection
This notebook performs Initial Data Exploration (EDA) and Outlier Detection based on the dataset and its metadata.

**Features MetaData:**
* `ID`: Numerical ID
* `Age`: Age, in years
* `Gender`: Female = 1, Male = 0
* `Job`: 1 = Unemployed, 2 = Employee/Worker, 3 = Manager/Executive, 4 = Entrepreneur/Freelancer, 5 = Retired
* `Area`: 1 = North, 2 = Center, 3 = South/Islands
* `CitySize`: 1 = Small town, 2 = Medium-sized city, 3 = Large city (>200k population)
* `FamilySize`: Number of family members
* `Income`: Normalized Income (percentiles)
* `Wealth`: Normalized Wealth (percentiles)
* `Debt`: Normalized Debt (percentiles)
* `FinEdu`: Normalized Financial Education (percentiles)
* `ESG`: Normalized ESG propensity (percentiles)
* `Digital`: Normalized Digital propensity (percentiles)
* `BankFriend`: Normalized Bank Friendliness (percentiles)
* `LifeStyle`: Normalized Lifestyle Index (percentiles)
* `Luxury`: Normalized Luxury spending (percentiles)
* `Saving`: Normalized Saving propensity (percentiles)
* `Investments`: 1 = no investments; 2 = mostly lump sum; 3 = mostly capital accumulation

Let's begin by importing the necessary libraries.
"""

code_1 = r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
"""

markdown_2 = r"""
## 1. Data Loading and Sanity Checks
First, we load the dataset and perform basic sanity checks for missing values and duplicates.
"""

code_2 = r"""
data_path = Path("Data") / "Dataset1_BankClients.xlsx"

# Load data
df = pd.read_excel(data_path)

print(f"Dataset Shape: {df.shape}")
df.head()
"""

code_3 = r"""
# Drop the ID column as it's not useful for analysis
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Check for missing values
print("Missing values in each column:\n", df.isna().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
"""

markdown_3 = r"""
## 2. Univariate Analysis (Numerical)
Let's analyze the distribution of numerical variables using histograms and boxplots. 
We'll check if the supposedly normalized variables really lie in the `[0, 1]` range, and if `Age` and `FamilySize` are reasonable.
"""

code_4 = r"""
# Separate numerical and categorical columns based on metadata
categorical_cols = ['Gender', 'Job', 'Area', 'CitySize', 'Investments']
numerical_cols = [col for col in df.columns if col not in categorical_cols]

# Summary statistics for numerical columns
df[numerical_cols].describe()
"""

code_5 = r"""
# Plot Histograms and Boxplots for numerical variables
normalized_vars = [col for col in numerical_cols if col not in ['Age', 'FamilySize']]

fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(14, 4 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    # Histogram
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()
"""

markdown_4 = r"""
### Outlier Checks on Numerical Features
We should explicitly verify `Age` bounds and bounds for normalized `[0,1]` variables.
"""

code_6 = r"""
# Check Age bounds
print(f"Age range: {df['Age'].min()} to {df['Age'].max()}")
if df['Age'].min() < 0 or df['Age'].max() > 120:
    print("Warning: Age contains unrealistic values.")

# Check FamilySize bounds
print(f"FamilySize range: {df['FamilySize'].min()} to {df['FamilySize'].max()}")

# Check normalized variables bounds
for col in normalized_vars:
    min_val, max_val = df[col].min(), df[col].max()
    if min_val < 0 or max_val > 1:
        print(f"Warning: {col} is out of [0, 1] bounds! Min: {min_val}, Max: {max_val}")
"""

markdown_5 = r"""
## 3. Univariate Analysis (Categorical)
Now, let's explore categorical variables with bar charts and verify they align with the metadata encoding.
"""

code_7 = r"""
# Plot count distributions for categorical features
fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(8, 4 * len(categorical_cols)))

for i, col in enumerate(categorical_cols):
    sns.countplot(x=df[col], ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {col}')
    
    # Check for invalid categories based on metadata
    unique_vals = sorted(df[col].dropna().unique())
    print(f"{col} unique values: {unique_vals}")

plt.tight_layout()
plt.show()
"""

markdown_6 = r"""
## 4. Bivariate Analysis & Correlations
Using a correlation heatmap to look for highly correlated features (Pearson's coefficient > 0.8 or < -0.8). Highly correlated features might provide duplicate information for distance-based clustering.
"""

code_8 = r"""
# Correlation heatmap for numerical variables
plt.figure(figsize=(12, 10))
corr_matrix = df[numerical_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
"""

code_9 = r"""
# Identify highly correlated pairs
threshold = 0.8
highly_correlated = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            highly_correlated.append((colname_i, colname_j, corr_val))

if highly_correlated:
    print(f"Highly correlated features (|r| > {threshold}):")
    for pair in highly_correlated:
        print(f"{pair[0]} & {pair[1]}: {pair[2]:.3f}")
else:
    print(f"No highly correlated numerical features found above |r| = {threshold}.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(markdown_5),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_markdown_cell(markdown_6),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_code_cell(code_9)
]

with open('EDA_BankClients.ipynb', 'w') as f:
    nbf.write(nb, f)

print("EDA notebook generated!")
