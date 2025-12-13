import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load data
# -----------------------------
sales = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])
gold = pd.read_csv("data/raw/gold_price.csv", parse_dates=["date"])

df = sales.merge(gold, on="date")

# -----------------------------
# Weekly aggregation
# -----------------------------
df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

weekly = (
    df.groupby(["week", "category"])
      .agg(
          units_sold=("units_sold", "sum"),
          gold_price=("gold_price_aed", "mean")
      )
      .reset_index()
)

# -----------------------------
# Feature engineering
# -----------------------------
weekly = weekly.sort_values(["category", "week"])

for lag in [1, 2, 4]:
    weekly[f"lag_{lag}"] = weekly.groupby("category")["units_sold"].shift(lag)

weekly["rolling_4w"] = (
    weekly.groupby("category")["units_sold"]
          .shift(1)
          .rolling(4)
          .mean()
)

weekly = weekly.dropna()

# -----------------------------
# Train / test split
# -----------------------------
train = weekly[weekly["week"] < "2024-06-01"]
test = weekly[weekly["week"] >= "2024-06-01"]

features = ["gold_price", "lag_1", "lag_2", "lag_4", "rolling_4w"]

# -----------------------------
# Model training
# -----------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(train[features], train["units_sold"])

# -----------------------------
# Forecast & evaluation
# -----------------------------
test["forecast_units"] = model.predict(test[features]).round()
mae = mean_absolute_error(test["units_sold"], test["forecast_units"])

print("MAE:", round(mae, 2))

# -----------------------------
# Save forecast
# -----------------------------
test[["week", "category", "units_sold", "forecast_units"]].to_csv(
    "reports/weekly_forecast.csv",
    index=False
)

print("✅ Weekly forecast saved to reports/weekly_forecast.csv")