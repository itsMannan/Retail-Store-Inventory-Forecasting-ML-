from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineer import engineer_features
from src.metrics import mape, mape_at_least, wape
from src.preprocessor import temporal_split


def test_mape_handles_zeros():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([1.0, 10.0, 22.0])
    value = mape(y, p, epsilon=1.0)
    assert np.isfinite(value)
    assert value > 0


def test_wape_and_filtered_mape():
    y = np.array([0.0, 50.0, 100.0])
    p = np.array([5.0, 50.0, 80.0])
    assert abs(wape(y, p) - (25 / 150 * 100)) < 1e-6
    filtered = mape_at_least(y, p, min_actual=10)
    assert abs(filtered - (0.5 * (0 + 20 / 100) * 100)) < 1e-6


def test_temporal_split_is_forward_looking():
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Store ID": ["S1"] * 10 + ["S2"] * 10,
            "Product ID": ["P1"] * 20,
            "Units Sold": range(20),
        }
    )
    split = temporal_split(df, test_size=0.2)
    assert split.train["Date"].max() < split.test["Date"].min()
    assert split.test["Date"].min() == split.cutoff


def test_rolling_mean_does_not_include_current_sales():
    n = 10
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "Store ID": ["S1"] * n,
            "Product ID": ["P1"] * n,
            "Units Sold": np.arange(n, dtype=float) + 1,
            "Inventory Level": 100,
            "Price": 10.0,
            "Discount": 0,
            "Competitor Pricing": 10.0,
            "Holiday/Promotion": 0,
            "Category": ["Toys"] * n,
            "Region": ["North"] * n,
            "Weather Condition": ["Sunny"] * n,
            "Seasonality": ["Summer"] * n,
            "Units Ordered": 5,
            "Demand Forecast": 3.0,
        }
    )
    out = engineer_features(df)
    # ma_7 on day index 3 uses sales from days 0-2 shifted, never today's 4.
    # After shift(1), rolling mean of first 3 known past values at row 3:
    # past = [1,2,3] mean = 2.0 (row 0 has no past → 0 fill after fillna on earlier rows)
    assert out.loc[0, "lag_1_sales"] == 0.0
    assert out.loc[1, "lag_1_sales"] == 1.0
    assert out.loc[3, "lag_1_sales"] == 3.0
    # Rolling mean at row 3: shift then roll → mean(1,2,3) = 2.0
    assert abs(out.loc[3, "ma_7_sales"] - 2.0) < 1e-9
    assert out.loc[3, "ma_7_sales"] != out.loc[3, "Units Sold"]
