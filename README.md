"""
Retail Store Inventory Forecasting, Classification, and Optimization
====================================================================

End-to-end machine learning project on `data/retail_store_inventory.csv`.

Approaches implemented
----------------------
1. **Demand forecasting (regression)** — predict `Units Sold`
2. **Demand-level classification** — LOW / MEDIUM / HIGH
3. **Inventory optimization** — safety stock and on-hand targets that cut holding cost

Two feature tracks
------------------
- **Operational** — price, discount, inventory, weather, calendar, lag/rolling sales. Does **not** use `Demand Forecast`.
- **Vendor-forecast refinement** — same features plus the existing `Demand Forecast` column (forecast value-added).

The vendor track is the production recommendation because that column is already a near-copy of realized sales in this dataset. The operational track is the honest ablation: it shows how far you can get from store operations data alone.

Quick start
-----------
```bash
python3 -m pip install -r requirements.txt
python3 -m src.pipeline
```

Artifacts land in `results/` (metrics CSVs) and `results/visualizations/` (plots).

Notebooks under `notebooks/` walk through EDA, preprocessing, features, modeling, and evaluation. They import the same `src/` package as the pipeline.

Project layout
--------------
```
data/retail_store_inventory.csv
notebooks/01_eda.ipynb … 05_evaluation.ipynb
src/          data_loader, preprocessor, feature_engineer, models, inventory, pipeline
results/      metrics, predictions, charts
tests/        leakage and metric checks
```
