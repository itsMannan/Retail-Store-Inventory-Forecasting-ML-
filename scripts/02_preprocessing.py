"""02 — Preprocessing: dates, missing values, temporal split, encoding.

Run from the repo root:

    python3 scripts/02_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.feature_engineer import engineer_features
from src.preprocessor import FeatureEncoder, temporal_split
from src.utils import print_section


def main() -> None:
    df = load_dataset()
    print_section("Dtypes and missing values")
    print(df.dtypes.to_string())
    print("null count", int(df.isna().sum().sum()))

    print_section("Temporal split (most recent 20 percent of dates)")
    featured = engineer_features(df)
    split = temporal_split(featured)
    print("cutoff", split.cutoff.date(), "train", len(split.train), "test", len(split.test))
    if split.train["Date"].max() >= split.test["Date"].min():
        raise AssertionError("Train dates leaked into the test window")

    print_section("Train-only one-hot encoding")
    enc = FeatureEncoder("operational").fit(split.train)
    X_train = enc.transform(split.train, scale=False)
    X_test = enc.transform(split.test, scale=False)
    X_train_scaled = enc.transform(split.train, scale=True)
    print("X_train", X_train.shape, "X_test", X_test.shape)
    print("scaled mean of first numeric col", float(X_train_scaled.iloc[:, 0].mean()))
    print(X_train.head().to_string())

    print_section("Columns excluded from operational predictors")
    print("Units Sold — target")
    print("Units Ordered — replenishment decision, near-zero correlation with sales")
    print("Demand Forecast — reserved for the vendor-refinement track")


if __name__ == "__main__":
    main()
