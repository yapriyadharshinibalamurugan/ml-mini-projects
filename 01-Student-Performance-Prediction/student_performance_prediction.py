"""
Student Performance Prediction System using Machine Learning
--------------------------------------------------------------
Predicts whether a student will Pass or Fail based on study hours,
attendance, previous marks, assignments and internal marks.

Run:  python student_performance_prediction.py
Outputs are saved in the outputs/ folder (plots + results.txt)
and the generated dataset is saved in dataset/student_data.csv
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic dataset generation)
# ---------------------------------------------------------------
np.random.seed(42)
n = 500

study_hours = np.round(np.random.normal(4, 1.8, n).clip(0, 10), 1)
attendance = np.round(np.random.normal(75, 15, n).clip(30, 100), 1)
previous_marks = np.round(np.random.normal(60, 15, n).clip(20, 100), 1)
assignments = np.round(np.random.normal(70, 12, n).clip(0, 100), 1)
internal_marks = np.round(np.random.normal(65, 14, n).clip(0, 100), 1)

# Final score is a weighted combination + noise -> decides Pass/Fail
score = (0.25 * study_hours * 10 + 0.25 * attendance + 0.2 * previous_marks +
          0.15 * assignments + 0.15 * internal_marks + np.random.normal(0, 5, n))
final_result = np.where(score >= 60, "Pass", "Fail")

df = pd.DataFrame({
    "Student_ID": [f"S{i+1:04d}" for i in range(n)],
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Assignments": assignments,
    "Internal_Marks": internal_marks,
    "Final_Result": final_result
})

# introduce a few missing values, like a real dataset would have
for col in ["Attendance", "Previous_Marks"]:
    idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[idx, col] = np.nan

df.to_csv(os.path.join(DATASET_DIR, "student_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/student_data.csv")

# ---------------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------------------
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Previous_Marks"] = df["Previous_Marks"].fillna(df["Previous_Marks"].mean())
df["Final_Result_Label"] = df["Final_Result"].map({"Pass": 1, "Fail": 0})

# ---------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Study_Hours", y="Previous_Marks", hue="Final_Result")
plt.title("Study Hours vs Previous Marks (by Result)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "study_hours_vs_marks.png"))
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Final_Result", y="Attendance")
plt.title("Attendance vs Result")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attendance_vs_result.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 4: FEATURE SELECTION
# ---------------------------------------------------------------
features = ["Study_Hours", "Attendance", "Previous_Marks", "Assignments", "Internal_Marks"]
X = df[features]
y = df["Final_Result_Label"]

# ---------------------------------------------------------------
# STEP 5 & 6: TRAIN/TEST SPLIT + MODEL TRAINING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

results = []
report_lines = []
best_model_name, best_acc, best_model = None, -1, None

for name, model in models.items():
    if name == "Logistic Regression":
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

# ---------------------------------------------------------------
# STEP 7: MODEL COMPARISON PLOT
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# STEP 8: LIST AT-RISK STUDENTS (predicted Fail) using best model
# ---------------------------------------------------------------
if best_model_name == "Logistic Regression":
    all_preds = best_model.predict(scaler.transform(X))
else:
    all_preds = best_model.predict(X)

df["Predicted_Result"] = np.where(all_preds == 1, "Pass", "Fail")
at_risk = df[df["Predicted_Result"] == "Fail"][["Student_ID", "Study_Hours", "Attendance", "Predicted_Result"]]
at_risk.to_csv(os.path.join(OUT_DIR, "at_risk_students.csv"), index=False)

with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
    f.write("STUDENT PERFORMANCE PREDICTION - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines))
    f.write(f"\n\nBest Model: {best_model_name} (Accuracy: {best_acc:.3f})\n")
    f.write(f"Number of at-risk (predicted Fail) students: {len(at_risk)}\n")

print(f"\nBest model: {best_model_name} with accuracy {best_acc:.3f}")
print(f"At-risk students identified: {len(at_risk)}")
print("All outputs saved in outputs/ folder.")
