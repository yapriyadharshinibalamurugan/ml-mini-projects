# Student Performance Prediction System

Predicts whether a student will **Pass** or **Fail** using study hours, attendance,
previous marks, assignments and internal marks.

## How it works
1. Generates a realistic synthetic dataset (`dataset/student_data.csv`)
2. Cleans missing values
3. Runs EDA (study hours vs marks, attendance vs result)
4. Trains **Logistic Regression, Decision Tree, Random Forest**
5. Evaluates with Accuracy / Precision / Recall / F1 / Confusion Matrix
6. Outputs a list of **at-risk (predicted Fail) students**

## Run it
```bash
pip install -r ../requirements.txt
python student_performance_prediction.py
```

## Outputs (in `outputs/`)
- `study_hours_vs_marks.png`, `attendance_vs_result.png` — EDA plots
- `model_comparison.png` / `.csv` — model accuracy comparison
- `at_risk_students.csv` — students predicted to fail
- `results.txt` — full metrics report

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
