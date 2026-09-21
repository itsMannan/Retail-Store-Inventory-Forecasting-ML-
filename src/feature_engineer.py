"""Calendar features, lag/rolling history, and price interactions.

Lag and rolling statistics are computed per store-product and always
shift by one day first so the current target cannot leak into features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATE_COL, GROUP_COLS, TARGET


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df[DATE_COL]
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df.drop(columns=["day_of_year"])


def _grouped_shift(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    return df.groupby(GROUP_COLS, sort=False)[col].shift(periods)


def _grouped_shifted_rolling(df: pd.DataFrame, col: str, window: int, stat: str) -> pd.Series:
    def _apply(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        roll = shifted.rolling(window=window, min_periods=1)
        if stat == "mean":
            return roll.mean()
        if stat == "std":
            return roll.std(ddof=0)
        raise ValueError(stat)

    return df.groupby(GROUP_COLS, sort=False)[col].transform(_apply)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1_sales"] = _grouped_shift(df, TARGET, 1)
    df["lag_7_sales"] = _grouped_shift(df, TARGET, 7)
    df["lag_30_sales"] = _grouped_shift(df, TARGET, 30)
    df["ma_7_sales"] = _grouped_shifted_rolling(df, TARGET, 7, "mean")
    df["ma_30_sales"] = _grouped_shifted_rolling(df, TARGET, 30, "mean")
    df["std_7_sales"] = _grouped_shifted_rolling(df, TARGET, 7, "std")
    return df


def add_price_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    competitor = df["Competitor Pricing"].replace(0, np.nan)
    df["price_to_competitor"] = df["Price"] / competitor
    df["discount_rate"] = df["Discount"] / 100.0
    df["promo_discount"] = df["Holiday/Promotion"] * df["discount_rate"]
    df["price_discount"] = df["Price"] * (1.0 - df["discount_rate"])
    df["inventory_to_ma7"] = df["Inventory Level"] / df["ma_7_sales"].replace(0, np.nan)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature set. Call on the chronologically sorted frame before splitting."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_price_inventory_features(df)
    # First day per SKU has no lag history; later lags may still be null.
    history_cols = ["lag_1_sales", "lag_7_sales", "lag_30_sales", "ma_7_sales", "ma_30_sales", "std_7_sales"]
    df[history_cols] = df[history_cols].fillna(0.0)
    df["price_to_competitor"] = df["price_to_competitor"].fillna(1.0)
    df["inventory_to_ma7"] = df["inventory_to_ma7"].fillna(df["inventory_to_ma7"].median())
    return df
