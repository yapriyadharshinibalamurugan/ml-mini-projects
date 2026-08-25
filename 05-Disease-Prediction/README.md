# Disease Prediction System (Healthcare Analytics)

Predicts whether a patient is likely to have a disease based on age, blood
pressure, sugar level, cholesterol, symptoms and medical history.

## How it works
1. Generates a synthetic patient dataset (`dataset/patient_data.csv`)
2. Encodes categorical fields (gender, medical history)
3. Runs EDA (age vs disease, blood pressure vs cholesterol)
4. Trains **Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM**
5. Evaluates with Accuracy / Precision / Recall / F1 / Confusion Matrix

## Run it
```bash
pip install -r ../requirements.txt
python disease_prediction.py
```

## Outputs (in `outputs/`)
- `age_vs_disease.png`, `bp_vs_cholesterol.png` — EDA plots
- `model_comparison.png` / `.csv` — model accuracy comparison
- `results.txt` — full metrics report

## Tech Stack
Python, Pandas, Scikit-learn

**Disclaimer:** This is an educational project using synthetic data — it is
not a medical diagnostic tool.
