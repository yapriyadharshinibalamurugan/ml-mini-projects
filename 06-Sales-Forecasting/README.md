# Sales Forecasting System

Forecasts future daily sales using historical trend, seasonality, promotions
and holiday indicators.

## How it works
1. Generates 2 years of synthetic daily sales data (`dataset/sales_data.csv`)
   with trend + weekly/yearly seasonality
2. Engineers features (day index, day-of-week, month, lag features)
3. Splits data by time (last 60 days = test set, no shuffling)
4. Trains **Linear Regression** and **Random Forest Regressor**
5. Evaluates with MAE / MSE / RMSE and plots forecast vs actual

## Run it
```bash
pip install -r ../requirements.txt
python sales_forecasting.py
```

## Outputs (in `outputs/`)
- `sales_trend.png`, `monthly_seasonality.png` — EDA plots
- `forecast_vs_actual.png` — predicted vs actual sales (last 60 days)
- `model_comparison.csv` — MAE / MSE / RMSE per model
- `results.txt` — full metrics report

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib
