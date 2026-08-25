# ML Mini Projects — 6 Machine Learning Projects

A collection of 6 complete, runnable machine learning mini-projects built with
Python and Scikit-learn. Each project includes its own synthetic dataset,
end-to-end pipeline script, saved outputs (plots + metrics), and a README.

| # | Project | Type | Key Techniques |
|---|---------|------|-----------------|
| 1 | [Student Performance Prediction](./01-Student-Performance-Prediction) | Classification | Logistic Regression, Decision Tree, Random Forest |
| 2 | [Fake News Detection](./02-Fake-News-Detection) | NLP / Classification | TF-IDF, Naive Bayes, SVM |
| 3 | [Customer Segmentation](./03-Customer-Segmentation) | Clustering | K-Means, Elbow Method |
| 4 | [Loan Approval Prediction](./04-Loan-Approval-Prediction) | Classification | Logistic Regression, Random Forest, Naive Bayes |
| 5 | [Disease Prediction](./05-Disease-Prediction) | Classification | Logistic Regression, Random Forest, SVM |
| 6 | [Sales Forecasting](./06-Sales-Forecasting) | Time Series / Regression | Linear Regression, Random Forest Regressor |

## Project Structure
```
ml-mini-projects/
├── 01-Student-Performance-Prediction/
│   ├── dataset/            # generated dataset (CSV)
│   ├── outputs/             # plots, results.txt, comparison CSVs
│   ├── student_performance_prediction.py
│   └── README.md
├── 02-Fake-News-Detection/
│   └── ... (same structure)
├── ... (03 to 06 follow the same structure)
├── requirements.txt
└── README.md   <- you are here
```

## How to run any project
```bash
git clone <your-repo-url>
cd ml-mini-projects
pip install -r requirements.txt
cd 01-Student-Performance-Prediction
python student_performance_prediction.py
```
Each script is self-contained — it generates its own dataset, trains multiple
models, evaluates them, and saves all plots/results into that project's
`outputs/` folder.

## Note on datasets
Each project uses a **synthetically generated dataset** (created inside the
script itself with NumPy, using realistic distributions and relationships)
so that the projects run instantly with no external downloads. If you have
real datasets (e.g. from Kaggle/UCI), you can drop them into the `dataset/`
folder and point the script's `pd.read_csv(...)` call to that file instead.

## Author
Add your name here before submitting.
