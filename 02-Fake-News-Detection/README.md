# Fake News Detection System

Classifies news articles as **Real** or **Fake** using NLP + Machine Learning.

## How it works
1. Generates a synthetic news dataset (`dataset/news_data.csv`)
2. Cleans text (lowercase, remove punctuation/stopwords)
3. Extracts features using **TF-IDF**
4. Trains **Naive Bayes, Logistic Regression, Linear SVM**
5. Evaluates with Accuracy / Precision / Recall / F1 / Confusion Matrix

## Run it
```bash
pip install -r ../requirements.txt
python fake_news_detection.py
```

## Outputs (in `outputs/`)
- `class_distribution.png` — Real vs Fake counts
- `model_comparison.png` / `.csv` — model accuracy comparison
- `results.txt` — full metrics report

## Tech Stack
Python, NLTK-style text cleaning, Scikit-learn (TF-IDF, Naive Bayes, SVM)
