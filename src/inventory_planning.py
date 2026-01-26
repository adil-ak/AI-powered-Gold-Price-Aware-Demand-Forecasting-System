import pandas as pd
import numpy as np

# -----------------------------
# Load forecast
# -----------------------------
forecast = pd.read_csv("reports/weekly_forecast.csv", parse_dates=["week"])

# -----------------------------
# Business assumptions
# -----------------------------
LEAD_TIME_WEEKS = 2          # supplier lead time
SERVICE_LEVEL_Z = 1.65       # ~95% service level

# -----------------------------
# Aggregate by 
# -----------------------------
plan = (
    forecast.groupby("category")
    .agg(
        avg_weekly_demand=("forecast_units", "mean"),
        demand_std=("forecast_units", "std")
    )
    .reset_index()
)

# -----------------------------
# Inventory formulas
# -----------------------------
plan["lead_time_demand"] = plan["avg_weekly_demand"] * LEAD_TIME_WEEKS
plan["safety_stock"] = (
    SERVICE_LEVEL_Z * plan["demand_std"] * np.sqrt(LEAD_TIME_WEEKS)
)

plan["recommended_purchase_units"] = (
    plan["lead_time_demand"] + plan["safety_stock"]
).round()

# -----------------------------
# Save purchase plan
# -----------------------------
plan.to_csv("reports/purchase_plan.csv", index=False)

print("✅ Inventory purchase plan saved to reports/purchase_plan.csv")