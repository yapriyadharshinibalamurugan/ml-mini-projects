"""
Fake News Detection System using Machine Learning & NLP
----------------------------------------------------------
Classifies news articles as Real or Fake using TF-IDF + classical ML models.

Run:  python fake_news_detection.py
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic news dataset)
# ---------------------------------------------------------------
random.seed(42)
np.random.seed(42)

real_templates = [
    "The government announced a new policy on {topic} today after a cabinet meeting.",
    "Scientists at a leading university published a peer reviewed study on {topic}.",
    "The stock market showed a moderate change following news about {topic}.",
    "Officials confirmed that the {topic} project will proceed as planned next quarter.",
    "A new report from the health ministry outlines updated guidelines for {topic}.",
    "The company released its quarterly earnings report showing growth in {topic}.",
]
fake_templates = [
    "SHOCKING: You won't believe what {topic} is secretly hiding from you!!!",
    "Doctors HATE this one trick about {topic} that big companies don't want you to know.",
    "BREAKING: Aliens confirmed to be behind the {topic} conspiracy, insiders claim.",
    "This miracle cure for {topic} was banned by the government, sources say.",
    "Celebrity secretly reveals {topic} is a total hoax, fans are furious.",
    "Experts are TERRIFIED after new claims about {topic} go viral overnight.",
]
topics = ["the economy", "climate change", "the new vaccine", "space exploration",
          "the election", "artificial intelligence", "the housing market",
          "renewable energy", "the education system", "cryptocurrency"]

rows = []
for i in range(400):
    topic = random.choice(topics)
    template = random.choice(real_templates)
    rows.append([f"N{i+1:04d}", f"Update on {topic}", template.format(topic=topic), "Real Author", 1])
for i in range(400):
    topic = random.choice(topics)
    template = random.choice(fake_templates)
    rows.append([f"N{i+401:04d}", f"You need to see this about {topic}", template.format(topic=topic), "Unknown", 0])

df = pd.DataFrame(rows, columns=["id", "title", "text", "author", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv(os.path.join(DATASET_DIR, "news_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/news_data.csv")

# ---------------------------------------------------------------
# STEP 2 & 3: PREPROCESSING + TEXT CLEANING
# ---------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = (df["title"] + " " + df["text"]).apply(clean_text)

# ---------------------------------------------------------------
# STEP 4: FEATURE EXTRACTION (TF-IDF)
# ---------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------
# STEP 5 & 6: MODEL SELECTION + TRAINING
# ---------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Linear SVM": LinearSVC(),
}

results = []
report_lines = []
best_model_name, best_acc, best_model = None, -1, None

for name, model in models.items():
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

plt.figure(figsize=(4, 4))
df["label"].map({1: "Real", 0: "Fake"}).value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"))
plt.close()

with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
    f.write("FAKE NEWS DETECTION - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines))
    f.write(f"\n\nBest Model: {best_model_name} (Accuracy: {best_acc:.3f})\n")

print(f"\nBest model: {best_model_name} with accuracy {best_acc:.3f}")
print("All outputs saved in outputs/ folder.")
