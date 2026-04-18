# Methodology: How We Create the Revenue Stream

Creating a robust revenue stream estimation (Customer Lifetime Value - CLV) for the bank's client clusters requires bridging the gap between raw behavioral data (Machine Learning) and real-world financial economics. 

In our notebook (`economic_value_v2.ipynb`), the revenue stream formula is built from the ground up using four core pillars:

## The Core Revenue Equation
For any given client, the Expected Annual Gross Revenue is the sum of the expected margins across all product families:

$$ \text{Gross Revenue} = \sum_{p \in \text{Products}} (\text{Propensity}_p \times \text{Volume Proxy}_p \times \text{Market Margin}_p) $$

Once Gross Revenue is calculated, we adjust for the cost to serve the client based on their preferred channels to find the True Net Margin:

$$ \text{Net Revenue} = \text{Gross Revenue} - \text{Estimated Service Cost} $$

---

## Pillar 1: Machine Learning Propensity (The "Will They Buy?")
Instead of assuming a static engagement rate, we trained **Logistic Regression Classifiers** on historical client data. 
*   **What it does:** It takes a client's sociodemographic and financial indicators (Income, Debt, ESG affinity, Family Size) and predicts a probability score `[0 - 1]` for whether they hold or will purchase a specific product (Savings, Investments, ESG Funds, Mortgages).
*   *Example:* Cluster 8 has a `0.36` (36%) propensity to take out a mortgage, whereas Cluster 3 has a `0.05` (5%) propensity.

## Pillar 2: Volume/AUM Proxies (The "How Big is the Ticket?")
Propensity alone is not money. We must multiply the probability of purchase by the average ticket size of that product category. We used standard Italian retail banking baselines:
*   **Savings:** € 50,000 average deposit.
*   **Traditional Investments:** € 60,000 average portfolio.
*   **ESG Investments:** € 60,000 average portfolio.
*   **Mortgages:** € 150,000 average residential loan.

## Pillar 3: Real-World Market Margins (The "Bank's Slice")
This is the critical multiplier that turns an Asset Under Management (AUM) or a Loan into actual *bottom-line revenue* for the bank. We anchored these to current European/Italian macroeconomic rates:
*   **Savings NIM (Net Interest Margin): ~1.8%**
    *   *Logic:* The spread between the interest the bank pays the retail depositor and the rate at which the bank re-deploys those funds in the market (e.g., via ECB deposit facilities or corporate lending).
*   **Mortgage Spread: ~1.5%**
    *   *Logic:* On a variable rate mortgage (Euribor + Spread), the bank captures an all-in margin of about 140-160 basis points upon origination.
*   **Standard Managed Funds Fee: ~1.2%**
    *   *Logic:* Average Italian retail TERs (Total Expense Ratios) are ~1.5%-2.2%, of which the distributing bank retains a 50-60% retrocession fee.
*   **ESG Funds Fee: ~1.4%**
    *   *Logic:* A slight premium over standard funds due to high retail demand and regulatory alignment.

## Pillar 4: Cost Assumptions & Channel Adjustments (The "Net Profit")
Gross revenue is meaningless without accounting for the Cost to Serve (CTS) and Customer Acquisition/Maintenance Costs (CAC). We established strict service cost assumptions based on the primary interaction channel of each cluster. 

We assume a **Baseline Physical Service Cost of € 300 per client annually**. This baseline accounts for branch real estate overhead, teller personnel salaries, paper compliance, and face-to-face advisory time. We then scale this cost down using specific assumptions based on digital adoption:

1.  **Digital-Native Cost Assumption (0.70x Multiplier = € 210/year)**
    *   *Applied To:* **Clusters 5 & 6** (The Volume Engines)
    *   *Rationale:* These clients exhibit the highest digital scores in the database. Because they self-serve via the mobile banking app and robo-advisors, they bypass expensive branch tellers. The 30% cost reduction reflects structural IT scaling efficiencies versus physical real estate.
2.  **Hybrid Service Cost Assumption (0.85x Multiplier = € 255/year)**
    *   *Applied To:* **Clusters 0, 1, 2, 4, 8, & 9** (Stable Core, Borrowers, Transitional)
    *   *Rationale:* A blend of digital and physical. These clients use the app for daily transactions (transfers, balance checks) but require expensive human advisory time for complex milestones (taking out a mortgage, structuring family investments). The 15% discount reflects partial automation.
3.  **Branch-Dependent Cost Assumption (1.0x Multiplier = € 300/year)**
    *   *Applied To:* **Clusters 3 & 7** (ESG Retirees)
    *   *Rationale:* Traditional segments with virtually zero digital footprint. They demand high-touch, face-to-face service for even basic operations (e.g., cashing checks, updating libretti). They bear the full weight of physical retail banking overhead.

---

## Example: Calculating the Revenue Stream for an Average "Cluster 5" Client
1. **Savings Check:** 61% Propensity $\times$ €50k $\times$ 1.8% Margin = **€ 549**
2. **Investments Check:** 52% Propensity $\times$ €60k $\times$ 1.2% Margin = **€ 374**
3. **ESG Check:** 53% Propensity $\times$ €60k $\times$ 1.4% Margin = **€ 445**
4. **Mortgage Check:** 25% Propensity $\times$ €150k $\times$ 1.5% Margin = **€ 562**
*   **GROSS REVENUE** = €1,930
*   **Service Cost (Digital 0.70)** = - €210
*   **FINAL NET REVENUE STREAM (CLV)** = **€1,720 per annum**

*(Note: Multiplied by the 1,314 clients in Cluster 5, this single segment generates a predictable stream of over **€2.26 Million** annually).*
