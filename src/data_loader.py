"""Load and validate the retail inventory dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, DATE_COL, TARGET

EXPECTED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality",
]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.exists():
        fallback = csv_path.parent.parent / "retail_store_inventory.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the CSV, parse dates, and sort by SKU then date."""
    df = load_raw(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(["Store ID", "Product ID", DATE_COL]).reset_index(drop=True)
    if df[TARGET].isna().any():
        raise ValueError("Target Units Sold contains missing values")
    return df


def dataset_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "n_dates": int(df[DATE_COL].nunique()),
        "n_stores": int(df["Store ID"].nunique()),
        "n_products": int(df["Product ID"].nunique()),
        "n_store_product": int(df.groupby(["Store ID", "Product ID"]).ngroups),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "units_sold_zero": int((df[TARGET] == 0).sum()),
        "stockout_exact": int((df["Inventory Level"] <= df[TARGET]).sum()),
    }
