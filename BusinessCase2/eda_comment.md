## EDA & Feature Engineering — Rationale

---

### What we explored and why

**Descriptive statistics + class balance**
Both targets are moderately imbalanced: `IncomeInvestment` 38/62%, `AccumulationInvestment` 51/49%.

**Box plots — Wealth & Income**
Both variables show severe right skew and extreme outliers (Wealth up to €2.2M, Income up to €365k).
This is not noise! It reflects real-world wealth distribution, a power low more than a gaussian.
Thus, log transformation is the appropriate response.

**Violin plots — RiskPropensity by age group × target**
Chosen over simple bar charts because they show the full distribution shape.
Key finding: among clients aged 65+, those with `IncomeInvestment = 1` show higher risk propensity than those with `= 0`.
This is consistent with the lifecycle theory: wealthier elderly clients can afford more risk while still seeking income.
For `AccumulationInvestment`, age group has virtually no effect. Distribution shapes are nearly identical across all bins.

**Scatter plots — Age × Wealth and Income × Wealth**
Direct visual motivation for the two engineered interaction features.
Age x Wealth: orange points (Income need = Yes) cluster top-right, so it's saying that old and wealthy clients drive this target.
Income x Wealth: separation follows the horizontal axis (Income), not the vertical (Wealth), this confirms that Income is the primary accumulation driver.

**What we did not include**
- *Q-Q plots*: normality is not assumed by any model in our pipeline (SVM, XGBoost, MLP, Random Forest). Testing for it adds no actionable information.
- *Pairplots*: with 10 engineered features, a full pairplot is 100 subplots, which is visually intractable and redundant with the correlation matrix already in the notebook.

---

### Feature engineering decisions

All features are motivated by financial theory first, then validated by correlation with the targets.

| Feature | Formula | Motivation |
|---|---|---|
| `Wealth_log` | log(1 + Wealth) | Corrects extreme right skew; box plot confirmed necessity |
| `Income_log` | log(1 + Income) | Same rationale |
| `Age_sq` | Age^2 | IncomeInvestment need accelerates non-linearly after 65 |
| `Age_x_Wealth` | Age x Wealth_log | Lifecycle interaction: elderly and wealthy = strongest income need signal (r = 0.45 with target) |
| `Income_per_FM_log` | log(1 + Income/FamilyMembers) | Per-capita income is more informative than gross income for multi-member households |
| `Wealth_per_FM_log` | log(1 + Wealth/FamilyMembers) | Same rationale applied to patrimony |
| `FinEdu_x_Risk` | FinancialEducation x RiskPropensity | Captures the "sophisticated investor" profile: risk-aware clients with financial literacy |

**What we excluded**
- *Gender*: correlation with both targets is < 0.015 — pure noise.
- *FamilyMembers (raw)*: the information it carries is better expressed by the per-capita ratios above.

**Data leakage prevention**
Scaling (MinMaxScaler) is fitted exclusively on the training set and applied to the test set.
The raw feature construction (log transforms, ratios, interactions) involves no statistical fitting and is therefore safe to apply on the full dataset before splitting.