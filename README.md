# Customer Churn Prediction — Telco Dataset

End-to-end machine learning project predicting customer churn for a telecom company. Built in Python with logistic regression and random forest, achieving **~79% accuracy** and **0.84 ROC-AUC** on the test set.

## Overview

This project walks through a classic supervised classification problem: identifying customers likely to cancel their service so the business can target them with retention offers. The notebook covers the full workflow — data extraction with SQL, exploratory analysis, feature engineering, model training, and comparative evaluation.

## Key results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.79 | 0.58 | 0.83 |
| Random Forest | 0.79 | 0.56 | 0.84 |

Both models perform in the same ballpark, with logistic regression slightly ahead on F1 — making it the better choice here given its interpretability advantage.

## Top churn drivers identified

1. **Contract type** — month-to-month customers churn at ~43% vs ~3% for two-year contracts
2. **Tenure** — first-year customers are far more likely to leave than long-tenured ones
3. **Internet service** — fiber optic customers churn more than DSL (worth investigating with the product team)
4. **Add-on services** — customers without tech support or online security churn at roughly 2x the rate

## Tech stack

- **Python** — pandas, numpy
- **SQL** — SQLite for data extraction queries
- **Modeling** — scikit-learn (LogisticRegression, RandomForestClassifier)
- **Visualization** — matplotlib, seaborn
- **Environment** — Jupyter Notebook

## Project structure

```
.
├── customer_churn_prediction.ipynb   # Main analysis notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/zainali_rajabali/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies (a virtual environment is recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook customer_churn_prediction.ipynb
   ```

The notebook fetches the dataset directly from its source URL — no manual download needed.

## Approach

The notebook is structured as follows:

1. **Data extraction with SQL** — load the CSV into SQLite and run analytical queries to understand the data shape and overall churn rate
2. **Exploratory data analysis** — distribution plots, churn rates by segment, correlation analysis
3. **Feature engineering** — tenure buckets, services count, average monthly spend
4. **Modeling** — train logistic regression and random forest on the same train/test split
5. **Evaluation** — compare both models on accuracy, precision, recall, F1, ROC-AUC, and confusion matrices
6. **Interpretation** — feature importance from random forest, signed coefficients from logistic regression, and business recommendations

## Why F1 matters more than accuracy here

With ~27% churn in the dataset, a naive model that always predicts "no churn" achieves 73% accuracy while being completely useless. F1 balances precision (don't waste retention budget on customers who weren't going to leave) with recall (catch as many at-risk customers as possible) — a much more meaningful metric for this problem.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers with 21 features covering demographics, service subscriptions, account information, and churn status. Publicly available and widely used as a benchmark for churn prediction tutorials.

## Possible extensions

- Try gradient boosting (XGBoost or LightGBM) — typically pushes AUC into the 0.85–0.87 range on this dataset
- Tune the classification threshold based on the cost ratio of false positives (wasted retention spend) vs false negatives (lost customer lifetime value)
- Add SHAP values for per-customer prediction explanations
- Build a simple Streamlit app for interactive predictions

## License

MIT
