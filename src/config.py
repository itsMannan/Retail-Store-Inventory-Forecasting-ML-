"""Project-wide constants and default modeling choices."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_store_inventory.csv"
RESULTS_DIR = ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"

TARGET = "Units Sold"
DATE_COL = "Date"
GROUP_COLS = ["Store ID", "Product ID"]

CATEGORICAL_COLS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]
BINARY_COLS = ["Holiday/Promotion"]
NUMERIC_BASE_COLS = [
    "Inventory Level",
    "Price",
    "Discount",
    "Competitor Pricing",
]

# Known at forecast time only if a vendor forecast already exists.
VENDOR_FORECAST_COL = "Demand Forecast"
# Contemporaneous replenishment decision — excluded from predictors.
EXCLUDE_FROM_FEATURES = ["Units Ordered", TARGET]

ENGINEERED_NUMERIC = [
    "day_of_week",
    "month",
    "weekofyear",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "lag_1_sales",
    "lag_7_sales",
    "lag_30_sales",
    "ma_7_sales",
    "ma_30_sales",
    "std_7_sales",
    "price_to_competitor",
    "discount_rate",
    "promo_discount",
    "inventory_to_ma7",
    "price_discount",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
LAG_MIN_PERIODS = 1

# Inventory economics (documented assumptions).
ANNUAL_HOLDING_RATE = 0.25
STOCKOUT_MARGIN = 0.45
# Daily on-hand snapshot, so the recommended position covers one selling day.
LEAD_TIME_DAYS = 1
Z_CANDIDATES = [0.25, 0.52, 0.84, 1.04, 1.28, 1.65, 1.96, 2.33]

DEMAND_BINS = {
    "LOW": 0.25,
    "HIGH": 0.75,
}
