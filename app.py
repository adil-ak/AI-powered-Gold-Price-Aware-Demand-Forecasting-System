import os
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Gold Price Demand Forecasting", layout="wide")
st.title("💍 Gold Price–Aware Demand Forecasting & Inventory Planning")
st.markdown("Upload your **sales.csv** and **gold_price.csv**, then generate forecasts and a purchase plan.")

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")
lead_time_weeks = st.sidebar.number_input("Lead time (weeks)", min_value=1, max_value=24, value=2)
service_z = st.sidebar.number_input("Service level Z (95%≈1.65)", min_value=0.5, max_value=3.0, value=1.65, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Required columns (after mapping)")
st.sidebar.code(
    "sales.csv: date, category, units_sold\n"
    "gold_price.csv: date, gold_price_aed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional: Column mapping")
st.sidebar.caption("If your file uses different column names, fill these in (case-insensitive).")

# Sales mapping inputs
sales_date_col = st.sidebar.text_input("Sales date column name", value="date")
sales_cat_col = st.sidebar.text_input("Sales category column name", value="category")
sales_units_col = st.sidebar.text_input("Sales units column name", value="units_sold")

# Gold mapping inputs
gold_date_col = st.sidebar.text_input("Gold date column name", value="date")
gold_price_col = st.sidebar.text_input("Gold price column name", value="gold_price_aed")

st.sidebar.markdown("---")
dayfirst = st.sidebar.checkbox("Day-first dates (DD/MM/YYYY)", value=True)
st.sidebar.caption("Keep ON for UAE-style dates like 13/12/2024. Turn OFF for US-style 12/13/2024.")

# -----------------------------
# Helpers
# -----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def safe_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # mapping keys/values must be lowercase
    df = df.copy()
    df = df.rename(columns=mapping)
    return df

def parse_dates_or_stop(df: pd.DataFrame, col: str, label: str, dayfirst_flag: bool) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst_flag)

    bad = df[df[col].isna()]
    if len(bad) > 0:
        st.error(f"❌ {label}: Some '{col}' values could not be parsed as dates.")
        st.write("Here are sample bad rows (fix these values in your CSV):")
        st.dataframe(bad.head(20))
        st.stop()
    return df

def make_week_start(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d, errors="coerce")
    return d.dt.to_period("W").apply(lambda r: r.start_time)

def validate_required_cols_or_stop(df: pd.DataFrame, required: set, label: str):
    missing = sorted(list(required - set(df.columns)))
    if missing:
        st.error(f"❌ {label} is missing required columns: {missing}")
        st.write("Found columns:", list(df.columns))
        st.stop()

def train_and_forecast(sales_df: pd.DataFrame, gold_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    # Merge sales with gold on date
    df = sales_df.merge(gold_df, on="date", how="left")

    # If gold has missing values after merge, fill forward/backward
    if df["gold_price_aed"].isna().any():
        df = df.sort_values("date")
        df["gold_price_aed"] = df["gold_price_aed"].ffill().bfill()

    # Weekly aggregation
    df["week"] = make_week_start(df["date"])

    weekly = (
        df.groupby(["week", "category"])
          .agg(
              units_sold=("units_sold", "sum"),
              gold_price=("gold_price_aed", "mean")
          )
          .reset_index()
          .sort_values(["category", "week"])
    )

    # Feature engineering
    for lag in [1, 2, 4]:
        weekly[f"lag_{lag}"] = weekly.groupby("category")["units_sold"].shift(lag)

    weekly["rolling_4w"] = (
        weekly.groupby("category")["units_sold"]
              .shift(1)
              .rolling(4)
              .mean()
    )

    weekly = weekly.dropna()

    if weekly.empty:
        st.error("❌ Not enough data after feature creation. You need more historical weeks to build lags/rolling features.")
        st.stop()

    # Time-based split (last ~26 weeks as test)
    split_date = weekly["week"].max() - pd.Timedelta(weeks=26)
    train = weekly[weekly["week"] < split_date].copy()
    test = weekly[weekly["week"] >= split_date].copy()

    if train.empty or test.empty:
        st.error("❌ Not enough history to split into train/test. Please upload more data (at least ~30-40 weeks).")
        st.stop()

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

    forecast_df = test[["week", "category", "units_sold", "forecast_units"]].copy()
    return forecast_df, float(mae)

def build_purchase_plan(forecast_df: pd.DataFrame, lead_time: int, z: float) -> pd.DataFrame:
    plan = (
        forecast_df.groupby("category")
        .agg(
            avg_weekly_demand=("forecast_units", "mean"),
            demand_std=("forecast_units", "std")
        )
        .reset_index()
    )

    # std can be NaN if category has too few rows in test; replace with 0
    plan["demand_std"] = plan["demand_std"].fillna(0)

    plan["lead_time_demand"] = plan["avg_weekly_demand"] * lead_time
    plan["safety_stock"] = z * plan["demand_std"] * np.sqrt(lead_time)
    plan["recommended_purchase_units"] = (plan["lead_time_demand"] + plan["safety_stock"]).round()

    return plan

# -----------------------------
# Upload UI
# -----------------------------
sales_file = st.file_uploader("Upload sales.csv", type=["csv"])
gold_file = st.file_uploader("Upload gold_price.csv", type=["csv"])

if not (sales_file and gold_file):
    st.info("Upload both files to enable forecasting.")
    st.stop()

# -----------------------------
# Read CSVs
# -----------------------------
sales = pd.read_csv(sales_file)
gold = pd.read_csv(gold_file)

# Normalize column names to lowercase
sales = normalize_cols(sales)
gold = normalize_cols(gold)

# Build mapping based on sidebar input (case-insensitive)
mapping_sales = {
    str(sales_date_col).strip().lower(): "date",
    str(sales_cat_col).strip().lower(): "category",
    str(sales_units_col).strip().lower(): "units_sold",
}
mapping_gold = {
    str(gold_date_col).strip().lower(): "date",
    str(gold_price_col).strip().lower(): "gold_price_aed",
}

# Apply renaming
sales = safe_rename(sales, mapping_sales)
gold = safe_rename(gold, mapping_gold)

# Validate required columns
validate_required_cols_or_stop(sales, {"date", "category", "units_sold"}, "sales.csv")
validate_required_cols_or_stop(gold, {"date", "gold_price_aed"}, "gold_price.csv")

# Parse dates robustly (stop & show bad rows if invalid)
sales = parse_dates_or_stop(sales, "date", "sales.csv", dayfirst)
gold = parse_dates_or_stop(gold, "date", "gold_price.csv", dayfirst)

# Clean category + units
sales["category"] = sales["category"].astype(str).str.strip().str.upper()
sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce")

bad_units = sales[sales["units_sold"].isna()]
if len(bad_units) > 0:
    st.error("❌ sales.csv: Some 'units_sold' values are not numeric.")
    st.write("Here are sample bad rows:")
    st.dataframe(bad_units.head(20))
    st.stop()

sales["units_sold"] = sales["units_sold"].astype(float)

# -----------------------------
# Action button
# -----------------------------
if st.button("🚀 Train & Generate Forecast"):
    with st.spinner("Training model and generating forecast..."):
        forecast_df, mae = train_and_forecast(sales, gold)
        plan_df = build_purchase_plan(forecast_df, lead_time_weeks, service_z)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Weekly Forecast")
        st.dataframe(forecast_df, use_container_width=True)

    with col2:
        st.subheader("✅ Model Quality")
        st.metric("MAE", f"{mae:.2f}")

    st.subheader("📦 Inventory Purchase Plan")
    st.dataframe(plan_df, use_container_width=True)

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
    st.info("Click **Train & Generate Forecast** to run the model.")