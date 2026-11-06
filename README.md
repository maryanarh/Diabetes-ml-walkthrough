# Pima Diabetes: PCA, Ensembles, and Logistic Regression

A single, reproducible Jupyter notebook that walks through three foundational machine-learning ideas on the **Pima Indians Diabetes** dataset (768 records, 8 clinical predictors, 1 binary target):

1. **Principal Component Analysis** — standardize the predictors, inspect the scree plot, and select the minimum number of components needed to retain 90% of variance (7 of 8 in this dataset). A 2-D scatter of PC1/PC2 shows how much — or how little — the top two components separate the observed diabetes classes.
2. **Ensemble learning** — compare a single Decision Tree, a Random Forest (bagging), and a Gradient Boosting model (boosting) on the same stratified 80/20 split. Boosting and bagging both substantially beat the single tree on accuracy, F1, and ROC-AUC.
3. **Diabetes prediction** — train a leakage-safe `Pipeline` (median imputation → `StandardScaler` → balanced `LogisticRegression`) and evaluate with a confusion matrix, precision/recall/F1, and ROC-AUC, with an explicit discussion of the false-negative vs. false-positive trade-off in a screening context.

All preprocessing is fitted on the training fold only. Results are fully reproducible with `RANDOM_STATE = 42`.

## Key results (held-out 20% test set)

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Decision Tree | 0.682 | 0.481 | 0.515 | 0.636 |
| Random Forest (bagging) | 0.740 | 0.537 | 0.592 | 0.816 |
| Gradient Boosting | **0.760** | 0.574 | **0.626** | **0.831** |
| Logistic Regression (balanced) | 0.734 | **0.704** | 0.650 | 0.813 |

**Model-selection note:** Gradient Boosting leads on accuracy and ROC-AUC, but its recall (0.574) is substantially lower than the balanced Logistic Regression's (0.704). In a screening context where false negatives are costlier than false positives, the balanced Logistic Regression is the more defensible choice despite lower overall accuracy.
