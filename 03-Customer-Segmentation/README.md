# Customer Segmentation using K-Means Clustering

Groups customers into meaningful segments (e.g. high spenders, budget customers)
based on income, spending score, age and purchase frequency.

## How it works
1. Generates a synthetic customer dataset (`dataset/customer_data.csv`)
2. Scales features with StandardScaler
3. Uses the **Elbow Method** to choose the optimal number of clusters
4. Applies **K-Means Clustering** (K = 4)
5. Visualizes and interprets each cluster

## Run it
```bash
pip install -r ../requirements.txt
python customer_segmentation.py
```

## Outputs (in `outputs/`)
- `elbow_method.png` — WCSS vs K plot
- `clusters_income_vs_spending.png` — cluster visualization
- `cluster_summary.csv` — mean values per cluster
- `customers_with_clusters.csv` — full dataset with assigned cluster
- `results.txt` — summary report

## Tech Stack
Python, Pandas, Scikit-learn (KMeans), Matplotlib, Seaborn
