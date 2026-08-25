# Loan Approval Prediction System

Predicts whether a loan application should be **Approved** or **Rejected**
based on applicant income, credit history, education and other details.

## How it works
1. Generates a synthetic loan dataset (`dataset/loan_data.csv`)
2. Handles missing values, encodes categorical variables
3. Runs EDA (income vs status, credit history vs approval)
4. Trains **Logistic Regression, Decision Tree, Random Forest, Naive Bayes**
5. Evaluates with Accuracy / Precision / Recall / F1 / Confusion Matrix

## Run it
```bash
pip install -r ../requirements.txt
python loan_approval_prediction.py
```

## Outputs (in `outputs/`)
- `income_vs_status.png`, `credit_history_vs_status.png` — EDA plots
- `model_comparison.png` / `.csv` — model accuracy comparison
- `results.txt` — full metrics report

## Tech Stack
Python, Pandas, Scikit-learn (Logistic Regression, Decision Tree, Random Forest)
