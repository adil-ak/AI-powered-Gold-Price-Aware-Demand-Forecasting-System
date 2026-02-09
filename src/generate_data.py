import pandas as pd
import numpy as np

np.random.seed(42)

# Generate daily dates
dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")

# ---------------------------
# Gold price (AED / gram)
# ---------------------------
gold_price = 200 + np.cumsum(np.random.normal(0, 0.6, len(dates)))

gold_df = pd.DataFrame({
    "date": dates,
    "gold_price_aed": gold_price.round(2)
})

# ---------------------------
# ---------------------------
categories = ["22K", "24K", "DIAMONDS", "COINS"]
rows = []

for cat in categories:
    base = {"22K":30, "24K":22, "DIAMONDS":10, "COINS":15}[cat]
    sensitivity = {"22K":-0.4, "24K":-0.5, "DIAMONDS":-0.1, "COINS":0.25}[cat]

    demand = base + sensitivity * (gold_price - 200)
    demand += np.random.normal(0, 3, len(dates))
    demand = np.maximum(demand, 0).round()

    for d, u in zip(dates, demand):
        rows.append([d, cat, int(u)])

sales_df = pd.DataFrame(rows, columns=["date", "category", "units_sold"])

# ---------------------------
# Save files
# ---------------------------
gold_df.to_csv("data/raw/gold_price.csv", index=False)
sales_df.to_csv("data/raw/sales.csv", index=False)

print("✅ Gold price and sales data generated successfully")