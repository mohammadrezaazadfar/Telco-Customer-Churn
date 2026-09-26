# Telco Customer Churn — Classification

Predicting whether a telecom customer will churn, using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (7,043 customers, 21 features).

## Problem & Business Framing

Losing a customer (churn) is expensive, and it's much cheaper to catch a likely churner in advance than to win a customer back. Because of that, this project optimizes for **Recall on the churn class**, not raw Accuracy — missing an actual churner is treated as more costly than a false alarm. The modeling target is `Churn` (binary), and the final model/threshold is chosen with a hard requirement of **Recall ≥ 0.80** for the churn class, then the best F1 among the thresholds that meet it.

## Repository Structure

```
project__Telco-Customer-Churn/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   ├── Telco-Customer-preprocessing-Churn.csv
│   ├── x_test_raw.csv          # for Decision Tree / Random Forest
│   ├── x_test_scaled.csv       # for SVC / KNN / Logistic Regression
│   └── y_test.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svc.pkl
│   ├── knn.pkl
│   └── logistic_regression.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── requirements.txt
└── README.md
```

## Methodology

1. **EDA** — distribution of `Churn`, relationship of every feature with churn, correlation analysis, data quality check.
2. **Preprocessing** — drop `customerID`, one-hot encode multi-category features, map binary Yes/No columns to 0/1, fix the `TotalCharges` data issue.
3. **Modeling** — 5 classifiers (Decision Tree, Random Forest, SVC, KNN, Logistic Regression), tuned with `GridSearchCV` scored on Recall for the churn class.
4. **Evaluation** — threshold sweep per model (0.20–0.70), final model/threshold picked under a Recall ≥ 0.80 constraint.

## Key EDA Findings

- Target is imbalanced: 73.5% No-churn / 26.5% Churn.
- **Contract type** is the strongest driver: Month-to-month churn = 42.7%, vs 11.3% (One year) and 2.8% (Two year).
- **tenure** is the strongest numeric correlate with churn (r = -0.35): churners average ~18 months tenure vs ~38 for retained customers.
- **PaymentMethod**: Electronic check churns at 45.3%, ~2.5x the other payment methods.
- **Fiber optic internet** churns more than DSL (41.9% vs 19.0%) despite being the premium tier.
- Missing add-ons (OnlineSecurity, OnlineBackup) correlate with higher churn.
- `StreamingTV` and `StreamingMovies` are highly redundant with each other (Cramér's V = 0.77).
- `gender` and `PhoneService` show almost no relationship with churn.

Full detail in `notebooks/01_EDA.ipynb` → EDA Conclusions section.

## Model Comparison

Best threshold per model under the Recall ≥ 0.80 constraint:

| Model | Best Threshold | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.810 | 0.525 | 0.637 | 0.755 |
| SVC | 0.25 | 0.821 | 0.470 | 0.598 | 0.707 |
| KNN | 0.30 | 0.807 | 0.457 | 0.584 | 0.694 |
| Decision Tree | — | — | — | — | *pending re-run, see note below* |
| Random Forest | — | — | — | — | *pending re-run, see note below* |

**Selected model: Logistic Regression @ threshold 0.50** (highest F1 among models meeting Recall ≥ 0.80).

> **Note on Decision Tree / Random Forest:** an earlier version of this pipeline had a scaling bug — the tree models were trained on unscaled features but evaluated on scaled test data (see the fix explained in `03_Modeling.ipynb`). That made their previous numbers invalid. The bug is fixed in the current notebooks; the table above will be complete once `03_Modeling.ipynb` and `04_Evaluation.ipynb` are re-run end to end with the raw dataset.

## Business Trade-off

At Recall ≥ 0.80, Precision is well below 1.0 (0.525 for the chosen model) — a meaningful share of customers flagged as "will churn" won't actually churn. This is an accepted trade-off for a retention-campaign use case: a false alarm (unnecessary retention offer) is assumed cheaper than missing a real churner. If that assumption is wrong for a given business, the `MIN_RECALL` constraint in `04_Evaluation.ipynb` should be revisited, not the model.

## How to Run

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `01_EDA.ipynb` → `02_Preprocessing.ipynb` → `03_Modeling.ipynb` → `04_Evaluation.ipynb`.

### Using a saved model for inference

```python
import joblib
import pandas as pd

model = joblib.load("models/logistic_regression.pkl")

# new_customer must go through the same preprocessing + StandardScaler
# fit on the training data in 03_Modeling.ipynb before calling predict_proba
proba_churn = model.predict_proba(new_customer_scaled)[:, 1]
prediction = (proba_churn >= 0.50).astype(int)  # using the selected threshold
```

## Known Issues / Next Steps

- Re-run `03_Modeling.ipynb` → `04_Evaluation.ipynb` with the raw CSV to get valid Decision Tree / Random Forest numbers (see note above).
- `pd.get_dummies` currently keeps all category levels (no `drop_first=True`); harmless for tree models, worth revisiting for Logistic Regression's interpretability.
- No `Pipeline`/`ColumnTransformer` yet — preprocessing and modeling are separate notebooks connected by an intermediate CSV. A future iteration could wrap this in a single `sklearn.Pipeline` for cleaner deployment.
