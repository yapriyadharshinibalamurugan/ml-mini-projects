"""
Sales Forecasting System using Machine Learning and Time Series Analysis
------------------------------------------------------------------------------
Predicts future daily sales using historical sales data with trend +
seasonality, comparing Linear Regression, Random Forest Regressor and ARIMA.

Run:  python sales_forecasting.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: DATA COLLECTION (synthetic daily sales dataset, 2 years)
# ---------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=730, freq="D")
t = np.arange(len(dates))

trend = 200 + 0.3 * t
seasonality = 40 * np.sin(2 * np.pi * t / 365) + 15 * np.sin(2 * np.pi * t / 7)
noise = np.random.normal(0, 15, len(dates))
promotion = np.random.choice([0, 1], len(dates), p=[0.85, 0.15])
holiday = np.where(pd.Series(dates).dt.weekday.isin([5, 6]), 1, 0)

units_sold = (trend + seasonality + noise + promotion * 30 + holiday * 20).clip(min=0).round()
revenue = (units_sold * np.random.uniform(8, 12, len(dates))).round(2)

df = pd.DataFrame({
    "Date": dates, "Product_ID": "PRD001", "Store_ID": "ST01",
    "Units_Sold": units_sold, "Revenue": revenue,
    "Promotion": promotion, "Holiday": holiday
})
df.to_csv(os.path.join(DATASET_DIR, "sales_data.csv"), index=False)
print(f"Dataset created: {len(df)} rows -> dataset/sales_data.csv")

# ---------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------
# STEP 3: EDA - trend & seasonality
# ---------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(df["Date"], df["Units_Sold"])
plt.title("Daily Units Sold Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sales_trend.png"))
plt.close()

monthly = df.set_index("Date")["Units_Sold"].resample("ME").mean()
plt.figure(figsize=(8, 4))
monthly.plot(marker="o")
plt.title("Average Monthly Sales (Seasonality)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "monthly_seasonality.png"))
plt.close()

# ---------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING (day index, day-of-week, month, lag features)
# ---------------------------------------------------------------
df["day_index"] = np.arange(len(df))
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["lag_1"] = df["Units_Sold"].shift(1)
df["lag_7"] = df["Units_Sold"].shift(7)
df = df.dropna().reset_index(drop=True)

features = ["day_index", "day_of_week", "month", "Promotion", "Holiday", "lag_1", "lag_7"]
X = df[features]
y = df["Units_Sold"]

# Time-based split: last 60 days as test set (no shuffling, this is a time series)
split_point = len(df) - 60
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
test_dates = df["Date"].iloc[split_point:]

# ---------------------------------------------------------------
# STEP 5 & 6: MODEL SELECTION + TRAINING
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = []
report_lines = []
best_model_name, best_mae, best_preds = None, float("inf"), None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    results.append([name, mae, mse, rmse])
    report_lines.append(f"\n=== {name} ===")
    report_lines.append(f"MAE : {mae:.2f}")
    report_lines.append(f"MSE : {mse:.2f}")
    report_lines.append(f"RMSE: {rmse:.2f}")

    if mae < best_mae:
        best_mae, best_model_name, best_preds = mae, name, preds

res_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE"])
res_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

# ---------------------------------------------------------------
# STEP 7: FORECAST VISUALIZATION (best model, last 60 days)
# ---------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(test_dates, y_test.values, label="Actual", marker="o", markersize=3)
plt.plot(test_dates, best_preds, label=f"Predicted ({best_model_name})", marker="x", markersize=3)
plt.title("Sales Forecast vs Actual (Last 60 Days)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forecast_vs_actual.png"))
plt.close()

with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
    f.write("SALES FORECASTING - RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write("\n".join(report_lines))
    f.write(f"\n\nBest Model: {best_model_name} (MAE: {best_mae:.2f})\n")

print(f"\nBest model: {best_model_name} with MAE {best_mae:.2f}")
print("All outputs saved in outputs/ folder.")
