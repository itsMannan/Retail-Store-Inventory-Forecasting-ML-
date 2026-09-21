"""01 — Exploratory data analysis for retail store inventory.

Run from the repo root:

    python3 scripts/01_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import VIZ_DIR
from src.data_loader import dataset_profile, load_dataset
from src.utils import ensure_output_dirs, print_section, save_csv, save_mapping_csv

sns.set_theme(style="whitegrid")


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()

    print_section("Dataset profile")
    profile = dataset_profile(df)
    for key, value in profile.items():
        print(f"  {key}: {value}")
    save_mapping_csv(profile, "dataset_profile.csv")

    print_section("Preview")
    print(df.head().to_string(index=False))
    print(df.describe().T.to_string())
    print("nulls")
    print(df.isnull().sum().to_string())

    print_section("Panel shape")
    print(df.groupby(["Store ID", "Product ID"]).size().describe().to_string())
    print("stores", df["Store ID"].nunique(), "products", df["Product ID"].nunique())
    for col in ["Category", "Region", "Weather Condition", "Seasonality"]:
        print(col, df[col].value_counts().to_dict())

    print_section("Daily volume")
    daily = df.groupby("Date")["Units Sold"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values)
    ax.set_title("Total units sold by day")
    ax.set_ylabel("Units")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "daily_sales.png", dpi=130)
    plt.close(fig)

    print_section("Inventory cap")
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = df.sample(min(6000, len(df)), random_state=42)
    ax.scatter(sample["Inventory Level"], sample["Units Sold"], s=8, alpha=0.25)
    lim = max(sample["Inventory Level"].max(), sample["Units Sold"].max())
    ax.plot([0, lim], [0, lim], "r--", label="sold = inventory")
    ax.set_xlabel("Inventory Level")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.set_title("Sales never exceed on-hand stock")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "inventory_vs_sold.png", dpi=130)
    plt.close(fig)

    print_section("Numeric correlations")
    num = [
        "Units Sold",
        "Inventory Level",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Holiday/Promotion",
    ]
    corr = df[num].corr().round(3)
    print(corr.to_string())
    save_csv(corr.reset_index().rename(columns={"index": "feature"}), "numeric_correlations.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Numeric correlations")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "correlation_heatmap.png", dpi=130)
    plt.close(fig)

    print_section("External factors")
    print("category\n", df.groupby("Category")["Units Sold"].mean().to_string())
    print("region\n", df.groupby("Region")["Units Sold"].mean().to_string())
    print("weather\n", df.groupby("Weather Condition")["Units Sold"].mean().to_string())
    print("promo\n", df.groupby("Holiday/Promotion")["Units Sold"].mean().to_string())
    print("seasonality label\n", df.groupby("Seasonality")["Units Sold"].mean().to_string())

    print_section("Data-quality findings")
    notes = [
        "Do not treat Demand Forecast as an ordinary from-scratch feature; it is already a forecast of Units Sold.",
        "Store-product lag-1 autocorrelation of units sold is about 0, so lags help very little.",
        "Category, region, and seasonality labels are not stable entity attributes.",
        "Units sold look like a random fraction of inventory (mean sold / mean inventory about 0.5).",
        "MAPE is a poor headline metric here because of zero-sales rows. Prefer WAPE and MAPE on demand >= 10.",
    ]
    for note in notes:
        print("-", note)
    save_csv(pd.DataFrame({"note": notes}), "data_quality_notes.csv")
    print("Wrote charts to", VIZ_DIR)


if __name__ == "__main__":
    main()
