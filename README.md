# Retail Store Inventory Forecasting & Optimization

Machine learning project on daily store-product inventory: forecast demand, classify demand intensity, and set on-hand targets that cut holding cost without a real fill-rate collapse.

## Dataset

| | |
|---|---|
| File | [`data/retail_store_inventory.csv`](data/retail_store_inventory.csv) |
| Rows | 73,100 (5 stores × 20 products × 731 days) |
| Dates | 2022-01-01 → 2024-01-01 |
| Target | `Units Sold` |
| Split | Temporal: train through 2023-08-07 (58,400 rows), test from 2023-08-08 (14,700 rows) |

Columns: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality.

No missing values. Units sold never exceed on-hand inventory.

## What this repo actually finds

This file is weakly structured outside two columns:

1. **`Demand Forecast` is already a forecast of the target** (correlation 0.997, MAE ≈ 8.4). Using it as an ordinary feature would be circular. The project treats it as a *vendor forecast to beat or refine* (forecast value added).
2. **`Inventory Level` is the only other real numeric signal** (correlation 0.59). Mean sold / mean inventory ≈ 0.5, which is why a “half of inventory” baseline already matches operational ML (R² ≈ 0.34).
3. Price, discount, competitor price, weather, promotions, and the seasonality *label* do not move mean sales. Competitor price is just price ± about $5. Category is not a stable product attribute; region is not a stable store attribute.

Lag-1 autocorrelation of units sold is ~0, so time-series lags add almost nothing. That is reported, not papered over.

## Approaches

| Approach | Goal | Production track |
|---|---|---|
| 1. Regression | Predict `Units Sold` | Random Forest using the vendor forecast + operations features |
| 2. Classification | LOW / MEDIUM / HIGH demand | XGBoost on the same vendor track (thresholds from **train**: LOW < 49, HIGH > 203) |
| 3. Inventory policy | On-hand = predicted demand × lead time + safety stock | z = 1.96 chosen on train to minimize total cost |

Two feature tracks are trained for every supervised model:

- **Operational** — inventory, price, discount, weather, calendar, leakage-safe lags/rolling means. Does **not** use `Demand Forecast` or `Units Ordered`.
- **Vendor refinement** — operational features plus `Demand Forecast`.

Lag and rolling features are computed per store-product and **shifted one day** so today’s sales cannot leak.

## Results (held-out test dates)

### Demand forecasting

Success target from the brief: MAPE < 15% on usable demand, predictions within ±20%.

| Model | Track | WAPE | MAPE (y≥10) | MAE | RMSE | R² | Within 20% |
|---|---|---:|---:|---:|---:|---:|---:|
| Mean baseline | — | 65.3% | 139.4% | 88.8 | 108.2 | 0.00 | 16% |
| Half of inventory | — | 50.5% | 101.6% | 68.6 | 87.6 | 0.34 | 21% |
| Linear / Ridge / RF / XGB | operational | ~50.5–50.9% | ~101% | ~69 | ~88 | ~0.34 | ~21% |
| Vendor `Demand Forecast` | baseline | 6.15% | 12.59% | 8.36 | 10.05 | 0.991 | 79% |
| **Random Forest** | **vendor** | **5.35%** | **10.47%** | **7.26** | **8.50** | **0.994** | **82%** |
| XGBoost | vendor | 5.40% | 10.54% | 7.33 | 8.62 | 0.994 | 82% |
| Linear Regression | vendor | 5.47% | 11.22% | 7.43 | 8.62 | 0.994 | 81% |

Headline MAPE with a 1-unit floor is 22.7% for the best model because 360 zero-sales rows explode percentage error. **Retail WAPE is 5.35%** and **MAPE on days with demand ≥ 10 is 10.47%**, both under the 15% bar. Operational-only models cannot honestly beat the inventory cap (R² ceiling ≈ 0.35).

### Demand classification

Success target: accuracy > 75%.

| Model | Track | Accuracy | F1 macro | ROC-AUC (OVR) |
|---|---|---:|---:|---:|
| Random Forest / XGBoost | operational | 57–58% | 0.53 | 0.74 |
| Logistic Regression | vendor | 94.7% | 0.947 | 0.996 |
| **XGBoost** | **vendor** | **94.7%** | **0.947** | **0.996** |

LOW and HIGH are almost never confused with each other on the vendor track.

### Inventory optimization

Assumptions (in `src/config.py`): 25% annual holding cost charged daily on on-hand units; stockout costs 45% of price per lost unit; 1-day lead time (the table is a daily snapshot). Safety-stock z is tuned on **train** residuals (σ ≈ 7.17).

| Policy | Mean on-hand | Holding $ | Stockout $ | Total $ | Stockout rate | vs current |
|---|---:|---:|---:|---:|---:|---:|
| Current on-hand | 274 | 150,892 | 0 | 150,892 | 0.0% | — |
| Match forecast, no safety stock | 136 | 74,943 | 1,305,538 | 1,380,481 | 50.1% | −815% |
| **Recommended (z = 1.96)** | **150** | **82,677** | **12,220** | **94,897** | **3.4%** | **−37% cost** |

Fill rate on the recommended policy is 99.97%. Cutting to the raw forecast without safety stock is much cheaper to hold and disastrous on lost sales.

SKU k-means (k=3) silhouette is 0.23 — product/store means are almost identical in this file, so ABC-style clusters are weak. That is a dataset finding.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 -m src.pipeline
python3 -m pytest tests/ -q
```

Writes `results/*.csv` and `results/visualizations/*.png`. The project is Python (`.py`) plus CSV data/results — no notebooks and no JSON.

Python walkthrough (same `src/` package):

```bash
python3 scripts/01_eda.py
python3 scripts/02_preprocessing.py
python3 scripts/03_feature_engineering.py
python3 scripts/04_modeling.py
python3 scripts/05_evaluation.py
```

Or one shot: `python3 scripts/run_all.py`

## Layout

```
data/retail_store_inventory.csv
scripts/        01_eda.py … 05_evaluation.py, run_all.py
src/            data_loader, preprocessor, feature_engineer, models, inventory, clustering, pipeline
results/        CSV metrics, predictions, PNG charts
tests/          MAPE/WAPE, temporal split, lag leakage
```

## Why these metrics

- **WAPE** = Σ|error| / Σ|actual| — the usual retail forecast KPI; not destroyed by zero-sales days.
- **MAPE (y≥10)** — percentage error on days with real volume; this is the number compared to the “MAPE < 15%” target.
- **MAE / RMSE / R²** — absolute fit.
- Classification uses train-only percentiles so the test labels are not fit on future data.

## Business takeaways

1. Ship the vendor forecast, lightly refined (Random Forest WAPE 5.35% vs 6.15% raw). Do not rebuild demand from price/weather/promo on this file — those columns have no signal.
2. Do not set on-hand equal to the forecast. A z ≈ 2 safety buffer is what makes the cheaper inventory policy safe.
3. Current mean on-hand (274) is about **2×** mean daily sales (136). The recommended mean on-hand is 150.
4. Do not change discount or promotional strategy from this extract; promotion vs non-promotion mean sales are 136.4 vs 136.5.
