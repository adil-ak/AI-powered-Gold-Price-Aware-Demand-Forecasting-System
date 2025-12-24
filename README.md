# AI Powered Gold Price Aware Demand Forecasting & Inventory Planning (Streamlit)

This project forecasts weekly jewellery demand by category (22K, 24K, Diamonds, Coins) using gold price as an external driver, and converts forecasts into inventory purchase recommendations using lead time demand + safety stock.

## Features
- Upload `sales.csv` and `gold_price.csv`
- Train model + generate weekly forecasts
- Generate purchase plan using:
  - Lead time demand
  - Safety stock (service level Z)

## Tech Stack
- Python 3.11
- pandas, 
- XGBoost
- Streamlit

## Required CSV Columns
### sales.csv
- `date` (YYYY-MM-DD recommended)
- `category`
- `units_sold`

### gold_price.csv
- `date` (YYYY-MM-DD recommended)
- `gold_price_aed`

## Run Locally
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
