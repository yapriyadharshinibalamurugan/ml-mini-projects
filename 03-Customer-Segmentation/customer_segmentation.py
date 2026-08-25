"""
Customer Segmentation using K-Means Clustering
--------------------------------------------------
Groups customers into segments based on Age, Annual Income, Spending Score
and Purchase Frequency.

Run:  python customer_segmentation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic customer dataset)
# ---------------------------------------------------------------
np.random.seed(42)
n = 400

age = np.random.randint(18, 70, n)
gender = np.random.choice(["Male", "Female"], n)
annual_income = np.round(np.random.normal(50000, 20000, n).clip(10000, 150000), -2)
spending_score = np.random.randint(1, 100, n)
purchase_frequency = np.random.randint(1, 50, n)

df = pd.DataFrame({
    "Customer_ID": [f"C{i+1:04d}" for i in range(n)],
    "Age": age,
    "Gender": gender,
    "Annual_Income": annual_income,
    "Spending_Score": spending_score,
    "Purchase_Frequency": purchase_frequency
})
df.to_csv(os.path.join(DATASET_DIR, "customer_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/customer_data.csv")

# ---------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

# ---------------------------------------------------------------
# STEP 3: FEATURE SELECTION + SCALING
# ---------------------------------------------------------------
features = ["Annual_Income", "Spending_Score", "Age"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# STEP 4: ELBOW METHOD TO FIND OPTIMAL K
# ---------------------------------------------------------------
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(K_range), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "elbow_method.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 5 & 6: APPLY K-MEANS (K chosen = 4) + TRAIN
# ---------------------------------------------------------------
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------------
# STEP 7: VISUALIZATION
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score", hue="Cluster", palette="tab10")
plt.title("Customer Segments: Income vs Spending Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "clusters_income_vs_spending.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 8: INTERPRETATION
# ---------------------------------------------------------------
cluster_summary = df.groupby("Cluster")[["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency"]].mean().round(1)
cluster_summary["Count"] = df["Cluster"].value_counts().sort_index()
cluster_summary.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))

df.to_csv(os.path.join(OUT_DIR, "customers_with_clusters.csv"), index=False)

with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
    f.write("CUSTOMER SEGMENTATION - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Number of clusters used: {K_OPTIMAL}\n\n")
    f.write("Cluster Summary (mean values per cluster):\n")
    f.write(cluster_summary.to_string())

print("\nCluster summary:")
print(cluster_summary)
print("\nAll outputs saved in outputs/ folder.")
