import os
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Gold Price Demand Forecasting", layout="wide")
st.title("💍 Gold Price–Aware Demand Forecasting & Inventory Planning")

st.markdown("Upload your **sales.csv** and **gold_price.csv**, then generate forecasts and a purchase plan.")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
lead_time_weeks = st.sidebar.number_input("Lead time (weeks)", min_value=1, max_value=12, value=2)
service_z = st.sidebar.number_input("Service level Z (95%≈1.65)", min_value=0.5, max_value=3.0, value=1.65, step=0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Required columns:")
st.sidebar.code("sales.csv: date, category, units_sold\ngold_price.csv: date, gold_price_aed")

# ---------- Upload files ----------
sales_file = st.file_uploader("Upload sales.csv", type=["csv"])
gold_file = st.file_uploader("Upload gold_price.csv", type=["csv"])

def make_week_start(d):
    return d.dt.to_period("W").apply(lambda r: r.start_time)

def train_and_forecast(sales_df, gold_df):
    df = sales_df.merge(gold_df, on="date", how="left")
    df["week"] = make_week_start(df["date"])

    weekly = (
        df.groupby(["week", "category"])
          .agg(units_sold=("units_sold", "sum"),
               gold_price=("gold_price_aed", "mean"))
          .reset_index()
          .sort_values(["category", "week"])
    )

    for lag in [1, 2, 4]:
        weekly[f"lag_{lag}"] = weekly.groupby("category")["units_sold"].shift(lag)

    weekly["rolling_4w"] = (
        weekly.groupby("category")["units_sold"]
              .shift(1)
              .rolling(4)
              .mean()
    )

    weekly = weekly.dropna()

    # Time split
    split_date = weekly["week"].max() - pd.Timedelta(weeks=26)
    train = weekly[weekly["week"] < split_date].copy()
    test = weekly[weekly["week"] >= split_date].copy()

    features = ["gold_price", "lag_1", "lag_2", "lag_4", "rolling_4w"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(train[features], train["units_sold"])
    test["forecast_units"] = model.predict(test[features]).clip(min=0).round()

    mae = mean_absolute_error(test["units_sold"], test["forecast_units"])
    return test[["week", "category", "units_sold", "forecast_units"]], mae

def build_purchase_plan(forecast_df, lead_time_weeks, service_z):
    plan = (
        forecast_df.groupby("category")
        .agg(
            avg_weekly_demand=("forecast_units", "mean"),
            demand_std=("forecast_units", "std")
        )
        .reset_index()
    )

    plan["lead_time_demand"] = plan["avg_weekly_demand"] * lead_time_weeks
    plan["safety_stock"] = service_z * plan["demand_std"] * np.sqrt(lead_time_weeks)
    plan["recommended_purchase_units"] = (plan["lead_time_demand"] + plan["safety_stock"]).round()
    return plan

if sales_file and gold_file:
    sales = pd.read_csv(sales_file, parse_dates=["date"])
    gold = pd.read_csv(gold_file, parse_dates=["date"])

    # Basic checks
    missing = []
    for c in ["date", "category", "units_sold"]:
        if c not in sales.columns: missing.append(f"sales.csv missing: {c}")
    for c in ["date", "gold_price_aed"]:
        if c not in gold.columns: missing.append(f"gold_price.csv missing: {c}")

    if missing:
        st.error("\n".join(missing))
        st.stop()

    if st.button("🚀 Train & Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            forecast_df, mae = train_and_forecast(sales, gold)
            plan_df = build_purchase_plan(forecast_df, lead_time_weeks, service_z)

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("📈 Weekly Forecast")
            st.write(forecast_df)
        with col2:
            st.subheader("✅ Model Quality")
            st.metric("MAE", f"{mae:.2f}")

        st.subheader("📦 Inventory Purchase Plan")
        st.write(plan_df)

        # Download buttons
        st.download_button(
            "Download weekly_forecast.csv",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="weekly_forecast.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download purchase_plan.csv",
            plan_df.to_csv(index=False).encode("utf-8"),
            file_name="purchase_plan.csv",
            mime="text/csv"
        )
else:
    st.info("Upload both files to enable forecasting.")