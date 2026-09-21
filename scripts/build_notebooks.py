"""Generate the five project notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def cell_md(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": _split(lines)}


def cell_code(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(lines),
    }


def _split(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def write(name: str, cells: list[dict]) -> None:
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1))
    print("wrote", path)


def nb01():
    write(
        "01_eda.ipynb",
        [
            cell_md(
                """# 01 — Exploratory Data Analysis

Retail store inventory (`data/retail_store_inventory.csv`): daily store-product rows with sales, on-hand stock, pricing, weather, and a vendor demand forecast.

This notebook profiles the table and records the data-quality facts that drive the modeling design."""
            ),
            cell_code(
                """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset, dataset_profile

sns.set_theme(style="whitegrid")
df = load_dataset()
profile = dataset_profile(df)
profile"""
            ),
            cell_code(
                """df.head()"""
            ),
            cell_code(
                """df.describe().T"""
            ),
            cell_code(
                """df.isnull().sum()"""
            ),
            cell_md(
                """## Shape of the panel

Five stores × 20 products × 731 dates. The file covers 2022-01-01 through 2024-01-01, not a single calendar year."""
            ),
            cell_code(
                """print(df.groupby(["Store ID", "Product ID"]).size().describe())
print("stores", df["Store ID"].nunique(), "products", df["Product ID"].nunique())
for col in ["Category", "Region", "Weather Condition", "Seasonality"]:
    print(col, df[col].value_counts().to_dict())"""
            ),
            cell_md("## Daily volume and the inventory cap"),
            cell_code(
                """daily = df.groupby("Date")["Units Sold"].sum()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily.values)
ax.set_title("Total units sold by day")
ax.set_ylabel("Units")
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
sample = df.sample(6000, random_state=42)
ax.scatter(sample["Inventory Level"], sample["Units Sold"], s=8, alpha=0.25)
lim = max(sample["Inventory Level"].max(), sample["Units Sold"].max())
ax.plot([0, lim], [0, lim], "r--", label="sold = inventory")
ax.set_xlabel("Inventory Level")
ax.set_ylabel("Units Sold")
ax.legend()
ax.set_title("Sales never exceed on-hand stock")
plt.show()"""
            ),
            cell_md(
                """## Correlations

`Demand Forecast` is almost collinear with `Units Sold`. `Inventory Level` is the only other strong numeric signal. Price, discount, competitor price, and units ordered are essentially uncorrelated with sales. Competitor price itself is just price ± about $5."""
            ),
            cell_code(
                """num = ["Units Sold", "Inventory Level", "Units Ordered", "Demand Forecast", "Price", "Discount", "Competitor Pricing", "Holiday/Promotion"]
print(df[num].corr().round(3))
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Numeric correlations")
plt.show()"""
            ),
            cell_md("## External factors barely move the mean"),
            cell_code(
                """display(df.groupby("Category")["Units Sold"].mean())
display(df.groupby("Region")["Units Sold"].mean())
display(df.groupby("Weather Condition")["Units Sold"].mean())
display(df.groupby("Holiday/Promotion")["Units Sold"].mean())
display(df.groupby("Seasonality")["Units Sold"].mean())"""
            ),
            cell_md(
                """## Data-quality findings (modeling implications)

1. **Do not treat `Demand Forecast` as an ordinary feature in an "from scratch" model.** It is already a forecast of the target (corr ≈ 0.997, MAE ≈ 8). Using it is *forecast refinement*, which is valid, but it is not independent demand prediction.
2. **Lags will not help much.** Store-product lag-1 autocorrelation of units sold is ≈ 0.
3. **Category / region / seasonality labels are not entity attributes.** Each product appears in every category; each store appears in every region; the seasonality flag does not match calendar season.
4. **Units sold look like a random fraction of inventory** (mean sold / mean inventory ≈ 0.5, sold ≤ inventory always). Operational models that include inventory will capture that cap and little else.
5. **MAPE is a poor headline metric here.** 360 zero-sales rows make classic MAPE explode. Report WAPE, MAE, RMSE, R², and MAPE on demand ≥ 10 alongside it.
"""
            ),
        ],
    )


def nb02():
    write(
        "02_preprocessing.ipynb",
        [
            cell_md(
                """# 02 — Preprocessing

Parse dates, confirm there are no missing values, and apply a **temporal** train/test split (most recent 20% of dates). Random row-wise splits leak the future into the past."""
            ),
            cell_code(
                """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.feature_engineer import engineer_features
from src.preprocessor import FeatureEncoder, temporal_split

df = load_dataset()
print(df.dtypes)
print("nulls", int(df.isna().sum().sum()))"""
            ),
            cell_md(
                """One-hot encoding and scaling are **fit on train only**. The `FeatureEncoder` does that. Trees do not need scaled inputs; linear models do (`scale=True`)."""
            ),
            cell_code(
                """featured = engineer_features(df)
split = temporal_split(featured)
print("cutoff", split.cutoff.date(), "train", len(split.train), "test", len(split.test))
assert split.train["Date"].max() < split.test["Date"].min()

enc = FeatureEncoder("operational").fit(split.train)
X_train = enc.transform(split.train, scale=False)
X_train_scaled = enc.transform(split.train, scale=True)
X_test = enc.transform(split.test, scale=False)
print(X_train.shape, X_test.shape)
X_train.head()"""
            ),
            cell_md(
                """Columns **not** used as operational predictors:

- `Units Sold` (target)
- `Units Ordered` (replenishment decision, ~0 correlation with sales)
- `Demand Forecast` (reserved for the vendor-refinement track)
"""
            ),
        ],
    )


def nb03():
    write(
        "03_feature_engineering.ipynb",
        [
            cell_md(
                """# 03 — Feature engineering

All lag and rolling features are computed **per store-product** and **shifted by one day** so today's sales cannot leak into today's features."""
            ),
            cell_code(
                """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.feature_engineer import engineer_features

df = load_dataset()
feat = engineer_features(df)
feat[["Date", "Store ID", "Product ID", "Units Sold", "lag_1_sales", "lag_7_sales", "ma_7_sales", "day_sin", "price_to_competitor"]].head(10)"""
            ),
            cell_md("Sanity check: lag_1 of a SKU equals yesterday's units sold."),
            cell_code(
                """sku = feat[(feat["Store ID"] == "S001") & (feat["Product ID"] == "P0001")].head(8)
sku[["Date", "Units Sold", "lag_1_sales", "ma_7_sales"]]"""
            ),
            cell_md(
                """## Feature groups

| Group | Columns |
|---|---|
| Calendar | day of week, month, weekofyear, sin/cos encodings |
| History | lag 1/7/30, rolling mean 7/30, rolling std 7 (all shifted) |
| Price | price, discount, competitor, price/competitor, promo × discount |
| Operations | inventory level, inventory / MA7 |
| Context | store, product, category, region, weather, seasonality, holiday flag |
| Vendor track only | `Demand Forecast` |
"""
            ),
        ],
    )


def nb04():
    write(
        "04_modeling.ipynb",
        [
            cell_md(
                """# 04 — Modeling

Train regression and classification models on both feature tracks.

This notebook retrains a **small** set of models so it stays fast. For the full catalog (Linear / Ridge / Random Forest / XGBoost, both tracks) run:

```bash
python3 -m src.pipeline
```

and read `results/model_performance.csv`."""
            ),
            cell_code(
                """import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.feature_engineer import engineer_features
from src.metrics import regression_report
from src.models import make_demand_labels, train_classifiers, train_regressors
from src.preprocessor import temporal_split
from src.config import TARGET

featured = engineer_features(load_dataset())
split = temporal_split(featured)
y_train, y_test = split.train[TARGET], split.test[TARGET]
print("cutoff", split.cutoff.date())"""
            ),
            cell_md("## Approach 1 — demand regression"),
            cell_code(
                """op = train_regressors(split.train, split.test, y_train, y_test, "operational")
vendor = train_regressors(split.train, split.test, y_train, y_test, "vendor")
pd.DataFrame([m.metrics for m in op + vendor])"""
            ),
            cell_md(
                """Operational models should land near the "half of inventory" baseline (R² ≈ 0.33). Vendor-track models should match the given forecast (R² ≈ 0.99, WAPE ≈ 6%). That is Forecast Value Added: does ML beat the forecast you already have?"""
            ),
            cell_md("## Approach 2 — LOW / MEDIUM / HIGH classification"),
            cell_code(
                """y_tr_c, y_te_c, low_q, high_q = make_demand_labels(y_train, y_test)
print("LOW <", low_q, "HIGH >", high_q)
clfs = train_classifiers(split.train, split.test, y_tr_c, y_te_c, "vendor")
pd.DataFrame([m.metrics for m in clfs])"""
            ),
        ],
    )


def nb05():
    write(
        "05_evaluation.ipynb",
        [
            cell_md(
                """# 05 — Evaluation, inventory policy, and business takeaways

Load artifacts from `python -m src.pipeline` (run that once if `results/` is empty)."""
            ),
            cell_code(
                """import sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
VIZ = RESULTS / "visualizations"

perf = pd.read_csv(RESULTS / "model_performance.csv")
clf = pd.read_csv(RESULTS / "classification_performance.csv")
inv = pd.read_csv(RESULTS / "inventory_optimization.csv")
insights = pd.read_json(RESULTS / "business_insights.json", typ="series")
perf"""
            ),
            cell_md("## Regression leaderboard"),
            cell_code(
                """display(perf.sort_values("WAPE"))
for name in ["regression_wape.png", "regression_mape.png", "scatter_vendor.png", "scatter_operational.png", "pred_vs_actual_vendor.png", "feature_importance.png"]:
    path = VIZ / name
    if path.exists():
        display(Markdown(f"**{name}**"))
        display(Image(str(path)))"""
            ),
            cell_md("## Classification"),
            cell_code(
                """display(clf.sort_values("accuracy", ascending=False))
display(Image(str(VIZ / "confusion_matrix.png")))
display(Image(str(VIZ / "classification_accuracy.png")))"""
            ),
            cell_md(
                """## Approach 3 — inventory optimization

Assumptions (also in `src/config.py`):

- Holding cost = 25% of price per year, charged daily on on-hand units
- Stockout cost = 45% of price per lost unit (lost margin)
- Lead time = 1 day (the dataset is a daily on-hand snapshot)
- Safety stock = z × residual std × √lead_time
- z is chosen on the **train** period to minimize total cost
"""
            ),
            cell_code(
                """display(inv)
display(Image(str(VIZ / "inventory_cost_comparison.png")))
pd.read_csv(RESULTS / "safety_stock_grid.csv")"""
            ),
            cell_md("## Clustering (segmentation check)"),
            cell_code(
                """skus = pd.read_csv(RESULTS / "sku_segments.csv")
display(skus.groupby("segment")[["mean_sales", "mean_price", "mean_inventory", "turnover", "revenue"]].mean())
display(Image(str(VIZ / "sku_clusters.png")))
print("Because SKU means are almost identical in this file, clusters will be weak. That is a finding, not a failure.")"""
            ),
            cell_md(
                """## Takeaways

1. **Forecasting:** Use the vendor `Demand Forecast` as the production signal (WAPE ≈ 6%). Operational-only ML cannot honestly beat ~R² 0.35 on this file.
2. **Classification:** Vendor-track models exceed 75% accuracy because they can read the existing forecast. Operational-only classifiers sit near 55–60%.
3. **Inventory:** Current on-hand is far above realized daily demand. A forecast + safety-stock target cuts holding cost; watch the stockout rate as z changes.
4. **Pricing / weather / promotions:** No material mean shift in this dataset. Do not recommend discount policy changes from these columns.
5. **Metrics:** Prefer WAPE and MAE for reporting. Quote MAPE only on days with demand ≥ 10, and say so.
"""
            ),
        ],
    )


if __name__ == "__main__":
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
