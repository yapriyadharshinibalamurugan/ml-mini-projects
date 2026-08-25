"""
Loan Approval Prediction System using Machine Learning
-----------------------------------------------------------
Predicts whether a loan application should be Approved or Rejected.

Run:  python loan_approval_prediction.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic loan dataset)
# ---------------------------------------------------------------
np.random.seed(42)
n = 600

gender = np.random.choice(["Male", "Female"], n)
married = np.random.choice(["Yes", "No"], n)
dependents = np.random.choice([0, 1, 2, 3], n)
education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.75, 0.25])
self_employed = np.random.choice(["Yes", "No"], n, p=[0.15, 0.85])
applicant_income = np.round(np.random.normal(5000, 2000, n).clip(1000, 20000))
coapplicant_income = np.round(np.random.normal(1500, 1200, n).clip(0, 10000))
loan_amount = np.round(np.random.normal(150, 60, n).clip(20, 500))
loan_term = np.random.choice([120, 180, 240, 300, 360], n)
credit_history = np.random.choice([1, 0], n, p=[0.8, 0.2])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], n)

score = (credit_history * 40 + (education == "Graduate") * 10 +
          (applicant_income + coapplicant_income) / 300 - loan_amount / 20 +
          np.random.normal(0, 8, n))
loan_status = np.where(score >= 25, "Approved", "Rejected")

df = pd.DataFrame({
    "Loan_ID": [f"LN{i+1:04d}" for i in range(n)],
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history, "Property_Area": property_area,
    "Loan_Status": loan_status
})

# a few missing values, like real datasets
for col in ["LoanAmount", "Credit_History"]:
    idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[idx, col] = np.nan

df.to_csv(os.path.join(DATASET_DIR, "loan_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/loan_data.csv")

# ---------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

encode_map = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}
for col, mapping in encode_map.items():
    df[col + "_Enc"] = df[col].map(mapping)

df["Loan_Status_Label"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

# ---------------------------------------------------------------
# STEP 3: EDA
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome")
plt.title("Applicant Income vs Loan Status")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "income_vs_status.png"))
plt.close()

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Approval")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "credit_history_vs_status.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 4: FEATURE SELECTION
# ---------------------------------------------------------------
features = ["Credit_History", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Education_Enc", "Married_Enc", "Gender_Enc",
            "Self_Employed_Enc", "Property_Area_Enc", "Dependents"]
X = df[features]
y = df["Loan_Status_Label"]

# ---------------------------------------------------------------
# STEP 5 & 6: TRAIN/TEST SPLIT + MODEL TRAINING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    "Naive Bayes": GaussianNB(),
}

results = []
report_lines = []
best_model_name, best_acc, best_model = None, -1, None

for name, model in models.items():
    if name in ("Logistic Regression",):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    results.append([name, acc, prec, rec, f1])
    report_lines.append(f"\n=== {name} ===")
    report_lines.append(f"Accuracy : {acc:.3f}")
    report_lines.append(f"Precision: {prec:.3f}")
    report_lines.append(f"Recall   : {rec:.3f}")
    report_lines.append(f"F1-Score : {f1:.3f}")
    report_lines.append(f"Confusion Matrix:\n{cm}")

    if acc > best_acc:
        best_acc, best_model_name, best_model = acc, name, model

res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
res_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

plt.figure(figsize=(6, 4))
sns.barplot(data=res_df, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"))
plt.close()

with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
    f.write("LOAN APPROVAL PREDICTION - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines))
    f.write(f"\n\nBest Model: {best_model_name} (Accuracy: {best_acc:.3f})\n")

print(f"\nBest model: {best_model_name} with accuracy {best_acc:.3f}")
print("All outputs saved in outputs/ folder.")
