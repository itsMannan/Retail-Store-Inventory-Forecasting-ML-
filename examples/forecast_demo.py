"""Starter demand-forecasting demo for the retail store inventory dataset.

Loads ``retail_store_inventory.csv``, engineers a few features, trains a
gradient-boosted model to predict ``Units Sold`` using a time-based
train/test split, prints evaluation metrics, and saves diagnostic plots.

Run from the repository root inside the project virtual environment:

    python examples/forecast_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "retail_store_inventory.csv"
DEFAULT_OUTDIR = REPO_ROOT / "outputs"

TARGET = "Units Sold"
CATEGORICAL = ["Category", "Region", "Weather Condition", "Seasonality"]
NUMERIC = [
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Competitor Pricing",
    "Holiday/Promotion",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["day_of_year"] = out["Date"].dt.dayofyear
    out = pd.get_dummies(out, columns=CATEGORICAL, drop_first=True)
    return out


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    cutoff = df["Date"].quantile(1 - test_frac)
    train = df[df["Date"] <= cutoff]
    test = df[df["Date"] > cutoff]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    raw = load_data(args.data)
    print(f"  rows={len(raw):,}  date range={raw['Date'].min().date()} -> {raw['Date'].max().date()}")

    featured = build_features(raw)
    feature_cols = [
        c
        for c in featured.columns
        if c not in {TARGET, "Date", "Store ID", "Product ID", "Demand Forecast"}
    ]

    train_df, test_df = time_split(featured)
    print(f"  train rows={len(train_df):,}  test rows={len(test_df):,}")

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print("\nEvaluation on held-out (future) period:")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  R^2  = {r2:.4f}")

    # Predicted vs actual scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, s=6, alpha=0.3)
    lim = [0, max(y_test.max(), preds.max())]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("Actual Units Sold")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title(f"Predicted vs Actual (R2={r2:.3f})")
    fig.tight_layout()
    scatter_path = args.outdir / "predicted_vs_actual.png"
    fig.savefig(scatter_path, dpi=120)
    print(f"\nSaved plot: {scatter_path}")

    # Aggregate daily demand trend
    daily = raw.groupby("Date")[TARGET].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax2)
    ax2.set_ylabel("Total Units Sold")
    ax2.set_title("Total daily units sold over time")
    fig2.tight_layout()
    trend_path = args.outdir / "daily_demand_trend.png"
    fig2.savefig(trend_path, dpi=120)
    print(f"Saved plot: {trend_path}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
