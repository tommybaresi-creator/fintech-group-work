## Individual Model Performance

### XGBoost
* **IncomeInvestment:** The model shows high precision (0.854 on Test) but struggles significantly with recall (0.503), suggesting it is very selective and misses nearly half of the positive cases. The calibration process successfully lowered the Brier score (0.1682), indicating better-aligned probability estimates, though the negative ablation delta suggests recent feature engineering didn't provide a lift for this specific target.
* **AccumulationInvestment:** This is a strong, balanced performer with an F1-score of 0.770 on the test set. Unlike the Income target, it maintains a much healthier recall (0.708). The PR-curve thresholding allows for a precision of 0.808 while keeping recall high at 0.772, making this a reliable model for identifying accumulation-focused investors.

### Logistic Regression
* **IncomeInvestment:** This model underperforms compared to tree-based methods, with an accuracy of 0.706 on the test set. It suffers from a generalization gap, as test metrics are consistently lower than CV means. However, the positive ablation delta (+0.019) indicates that the experimental features are providing a meaningful boost to this linear learner.
* **AccumulationInvestment:** Performance is moderately better here than on the Income target, achieving an F1 of 0.713. While it has decent recall (0.752), its precision is relatively low (0.678), and attempting to force a precision of 0.75 via thresholding results in a massive collapse of recall to 0.407, suggesting the model's decision boundary is not well-defined.

### GaussianNB
* **IncomeInvestment:** The Naive Bayes approach acts as a mediocre baseline, with an F1 of 0.593 on test data. It is particularly weak at higher precision requirements; to reach the 0.75 precision threshold, recall drops to 0.393, making it inefficient for targeted campaigns.
* **AccumulationInvestment:** While this model achieves a high recall (0.737), its overall accuracy and precision are the lowest among all tested models for this target. The Brier score (0.2436) is dangerously close to the 0.25 baseline, meaning its probability predictions are barely more informative than a random guess.

### RandomForest
* **IncomeInvestment:** This model is highly conservative, yielding the highest precision in cross-validation (0.848). Like XGBoost, it trades off recall (0.516) for accuracy. It is well-calibrated (Brier 0.1673), but the ablation study shows that the current feature set is likely plateaued, as changes resulted in a negligible ΔF1.
* **AccumulationInvestment:** This is arguably the top performer for this target, achieving a very high F1 at the adjusted threshold (0.785). It shows excellent stability between CV and Test results, and the calibration further refined its predictive power, making it the most robust choice for this segment.

### MLP (Multi-Layer Perceptron)
* **IncomeInvestment:** The neural network shows a significant drop-off from CV (0.792 accuracy) to Test (0.757), suggesting slight overfitting. While it captures the general trend, its recall (0.529) remains a bottleneck. Interestingly, it benefited more from feature engineering (+0.011 ΔF1) than the tree-based models.
* **AccumulationInvestment:** The MLP performs competitively here with an F1 of 0.765. It maintains high precision (0.809 on Test) and shows good generalization. Like the Income target, it responded positively to the experimental feature set, suggesting it can extract more value from new data than simpler models.

### ClassifierChain (XGBoost)
* **IncomeInvestment:** By utilizing the relationship between labels, this model achieved the highest test accuracy (0.796) and a very strong F1 (0.722) for this target. This suggests that knowing whether a user is interested in "Accumulation" is a powerful predictor for "Income" investment interest, significantly outperforming the standalone XGBoost model.

### Soft/Hard Voting Ensembles
* **IncomeInvestment (Soft & Hard):** Both ensembles provide a "smoothing" effect, resulting in balanced but not necessarily record-breaking performance. The Hard Voting ensemble produced a slightly higher F1 (0.640) than Soft Voting on the test set, suggesting that a majority-rules approach is more effective than averaging probabilities for this specific label.
* **AccumulationInvestment (Soft & Hard):** The Soft Voting ensemble is particularly effective here, benefiting from a +0.013 lift from feature engineering and maintaining a strong F1 of 0.766. The Hard Voting ensemble saw a significant ablation boost (+0.031), indicating that the combination of models is much more sensitive to the refined feature set than individual components.

---

## Metrics Comparison Table

| Model | Target | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | Income | 0.776 | 0.854 | 0.503 | 0.633 |
| | Accumulation | 0.783 | 0.844 | 0.708 | 0.770 |
| **RandomForest** | Income | 0.776 | 0.839 | 0.516 | 0.639 |
| | Accumulation | 0.778 | 0.836 | 0.706 | 0.765 |
| **ClassifierChain** | Income | **0.796** | 0.803 | **0.656** | **0.722** |
| **MLP** | Income | 0.757 | 0.766 | 0.529 | 0.626 |
| | Accumulation | 0.771 | 0.809 | 0.725 | 0.765 |
| **Soft Voting** | Income | 0.767 | 0.792 | 0.534 | 0.638 |
| | Accumulation | 0.763 | 0.777 | 0.754 | 0.766 |
| **GaussianNB** | Income | 0.709 | 0.640 | 0.552 | 0.593 |
| | Accumulation | 0.601 | 0.589 | 0.737 | 0.655 |

---

## Overall Conclusions

1.  **Best Model for Income Investment:** **ClassifierChain (XGBoost)**. It significantly outperforms all other models in F1 (0.722) and Recall (0.656). This indicates that the inter-dependency between investment types is a crucial signal that standalone models miss.
2.  **Best Model for Accumulation Investment:** **XGBoost** or **RandomForest**. Both provide high accuracy (~0.78) and strong F1 scores (~0.77). RandomForest is slightly more robust when calibrated, but XGBoost offers a marginal lead in test accuracy.
3.  **Key Challenge (Recall):** Across almost all models for "Income Investment," Recall is significantly lower than Precision. The models are good at identifying *who* is a definite "Income" lead, but they are missing a large portion (up to 50%) of the actual audience.
4.  **Feature Impact:** The Ablation studies show that new features consistently help linear models (Logistic Regression) and Neural Networks (MLP), while tree-based models (XGBoost/RF) are already nearing their performance ceiling with the current feature set.