"""
Disease Prediction System using Machine Learning (Healthcare Analytics)
---------------------------------------------------------------------------
Predicts whether a patient likely has heart disease based on medical
parameters (age, blood pressure, sugar level, cholesterol, etc.).

Run:  python disease_prediction.py
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
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic patient dataset)
# ---------------------------------------------------------------
np.random.seed(42)
n = 500

age = np.random.randint(20, 80, n)
gender = np.random.choice(["Male", "Female"], n)
blood_pressure = np.round(np.random.normal(125, 18, n).clip(90, 200))
sugar_level = np.round(np.random.normal(110, 30, n).clip(70, 300))
cholesterol = np.round(np.random.normal(200, 40, n).clip(120, 400))
symptom_score = np.random.randint(0, 10, n)  # count of reported symptoms
medical_history = np.random.choice(["None", "Diabetes", "Hypertension", "Other"], n)

risk = (age * 0.3 + blood_pressure * 0.3 + sugar_level * 0.15 +
        cholesterol * 0.15 + symptom_score * 3 + np.random.normal(0, 12, n))
disease = np.where(risk >= np.percentile(risk, 55), "Disease", "No Disease")

df = pd.DataFrame({
    "Patient_ID": [f"P{i+1:04d}" for i in range(n)],
    "Age": age, "Gender": gender, "Blood_Pressure": blood_pressure,
    "Sugar_Level": sugar_level, "Cholesterol": cholesterol,
    "Symptom_Score": symptom_score, "Medical_History": medical_history,
    "Disease": disease
})
df.to_csv(os.path.join(DATASET_DIR, "patient_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/patient_data.csv")

# ---------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------
df["Gender_Enc"] = df["Gender"].map({"Male": 1, "Female": 0})
history_dummies = pd.get_dummies(df["Medical_History"], prefix="History")
df = pd.concat([df, history_dummies], axis=1)
df["Disease_Label"] = df["Disease"].map({"Disease": 1, "No Disease": 0})

# ---------------------------------------------------------------
# STEP 3: EDA
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Disease", y="Age")
plt.title("Age vs Disease")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "age_vs_disease.png"))
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Blood_Pressure", y="Cholesterol", hue="Disease")
plt.title("Blood Pressure vs Cholesterol by Disease")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bp_vs_cholesterol.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING / SELECTION
# ---------------------------------------------------------------
features = ["Age", "Gender_Enc", "Blood_Pressure", "Sugar_Level", "Cholesterol",
            "Symptom_Score"] + list(history_dummies.columns)
X = df[features]
y = df["Disease_Label"]

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
    "SVM": SVC(),
}

results = []
report_lines = []
best_model_name, best_acc, best_model = None, -1, None

for name, model in models.items():
    if name in ("Logistic Regression", "SVM"):
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
    f.write("DISEASE PREDICTION - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines))
    f.write(f"\n\nBest Model: {best_model_name} (Accuracy: {best_acc:.3f})\n")

print(f"\nBest model: {best_model_name} with accuracy {best_acc:.3f}")
print("All outputs saved in outputs/ folder.")
