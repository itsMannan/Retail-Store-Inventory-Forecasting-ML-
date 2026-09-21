"""03 — Feature engineering: calendar, lags, rolling history, price ratios.

Lag and rolling features are computed per store-product and shifted by one
day so today's sales cannot leak into today's features.

Run from the repo root:

    python3 scripts/03_feature_engineering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.feature_engineer import engineer_features
from src.utils import print_section, save_csv


def main() -> None:
    df = load_dataset()
    feat = engineer_features(df)

    print_section("Engineered columns (sample)")
    cols = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "lag_1_sales",
        "lag_7_sales",
        "ma_7_sales",
        "day_sin",
        "price_to_competitor",
    ]
    print(feat[cols].head(10).to_string(index=False))

    print_section("Sanity check: lag_1 equals yesterday's units sold")
    sku = feat[(feat["Store ID"] == "S001") & (feat["Product ID"] == "P0001")].head(8)
    print(sku[["Date", "Units Sold", "lag_1_sales", "ma_7_sales"]].to_string(index=False))

    preview = feat[cols].head(50)
    save_csv(preview, "feature_engineering_preview.csv")

    print_section("Feature groups")
    print("Calendar: day of week, month, weekofyear, sin/cos encodings")
    print("History: lag 1/7/30, rolling mean 7/30, rolling std 7 (all shifted)")
    print("Price: price, discount, competitor, price/competitor, promo x discount")
    print("Operations: inventory level, inventory / MA7")
    print("Context: store, product, category, region, weather, seasonality, holiday")
    print("Vendor track only: Demand Forecast")


if __name__ == "__main__":
    main()
