"""Cleaning, temporal split, and encoding. Scalers are train-only."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    DATE_COL,
    ENGINEERED_NUMERIC,
    NUMERIC_BASE_COLS,
    TARGET,
    TEST_SIZE,
    VENDOR_FORECAST_COL,
)


OPERATIONAL_NUMERIC = NUMERIC_BASE_COLS + ENGINEERED_NUMERIC + BINARY_COLS
VENDOR_NUMERIC = OPERATIONAL_NUMERIC + [VENDOR_FORECAST_COL]


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def temporal_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> TemporalSplit:
    """Hold out the most recent `test_size` fraction of calendar dates."""
    dates = pd.Series(df[DATE_COL].sort_values().unique())
    cutoff_idx = int((1.0 - test_size) * len(dates))
    cutoff = pd.Timestamp(dates.iloc[cutoff_idx])
    train = df[df[DATE_COL] < cutoff].copy()
    test = df[df[DATE_COL] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test frame")
    return TemporalSplit(train=train, test=test, cutoff=cutoff)


class FeatureEncoder:
    """One-hot encode categoricals, optionally scale numerics (linear models)."""

    def __init__(self, feature_set: str = "operational"):
        if feature_set not in {"operational", "vendor"}:
            raise ValueError(feature_set)
        self.feature_set = feature_set
        self.numeric_cols = VENDOR_NUMERIC if feature_set == "vendor" else OPERATIONAL_NUMERIC
        self.categorical_cols = list(CATEGORICAL_COLS)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.feature_names_: list[str] | None = None

    def _select(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        missing_num = [c for c in self.numeric_cols if c not in df.columns]
        missing_cat = [c for c in self.categorical_cols if c not in df.columns]
        if missing_num or missing_cat:
            raise KeyError(f"Missing numeric={missing_num} categorical={missing_cat}")
        return df[self.numeric_cols].astype(float), df[self.categorical_cols].astype(str)

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        X_num, X_cat = self._select(df)
        self.encoder.fit(X_cat)
        self.scaler.fit(X_num)
        cat_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names_ = list(self.numeric_cols) + cat_names
        return self

    def transform(self, df: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        X_num, X_cat = self._select(df)
        if scale:
            X_num_out = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=self.numeric_cols,
                index=df.index,
            )
        else:
            X_num_out = X_num.copy()
        X_cat_out = pd.DataFrame(
            self.encoder.transform(X_cat),
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
            index=df.index,
        )
        X = pd.concat([X_num_out, X_cat_out], axis=1)
        return X[self.feature_names_]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df, df[TARGET].astype(float)
