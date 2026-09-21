"""End-to-end training pipeline: EDA → features → models → inventory policy."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m src.pipeline` and `python src/pipeline.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.clustering import cluster_skus, sku_features
from src.config import RESULTS_DIR, TARGET, VENDOR_FORECAST_COL
from src.data_loader import dataset_profile, load_dataset
from src.feature_engineer import engineer_features
from src.inventory import compare_policies, residual_std, tune_z
from src.metrics import regression_report
from src.models import baseline_predictions, make_demand_labels, train_classifiers, train_regressors
from src.preprocessor import temporal_split
from src.utils import ensure_output_dirs, print_section, save_csv, save_json
from src import visualize as viz


def business_insights(df: pd.DataFrame, importance: pd.DataFrame | None) -> dict:
    return {
        "top_features": (
            importance.head(8)[["feature", "importance"]].to_dict(orient="records") if importance is not None else []
        ),
        "mean_units_by_category": df.groupby("Category")[TARGET].mean().sort_values(ascending=False).round(3).to_dict(),
        "mean_units_by_region": df.groupby("Region")[TARGET].mean().sort_values(ascending=False).round(3).to_dict(),
        "promotion_impact": df.groupby("Holiday/Promotion")[TARGET].mean().round(3).to_dict(),
        "weather_impact": df.groupby("Weather Condition")[TARGET].mean().round(3).to_dict(),
        "seasonality_label_impact": df.groupby("Seasonality")[TARGET].mean().round(3).to_dict(),
        "inventory_sold_corr": round(float(df["Inventory Level"].corr(df[TARGET])), 4),
        "vendor_forecast_corr": round(float(df[VENDOR_FORECAST_COL].corr(df[TARGET])), 4),
        "price_competitor_corr": round(float(df["Price"].corr(df["Competitor Pricing"])), 4),
        "mean_inventory": round(float(df["Inventory Level"].mean()), 2),
        "mean_units_sold": round(float(df[TARGET].mean()), 2),
        "mean_leftover": round(float((df["Inventory Level"] - df[TARGET]).mean()), 2),
        "data_quality_notes": [
            "Category is not a stable product attribute (each Product ID appears in all 5 categories).",
            "Region is not a stable store attribute (each Store ID appears in all 4 regions).",
            "Seasonality labels do not line up with calendar seasons.",
            "Units sold are always <= inventory; mean sold / mean inventory ≈ 0.5.",
            "Vendor Demand Forecast is nearly collinear with Units Sold (corr ≈ 0.997).",
        ],
    }


def run(data_path: Path | None = None) -> dict:
    ensure_output_dirs()
    print_section("1. Load & profile")
    df = load_dataset(data_path)
    profile = dataset_profile(df)
    print(profile)
    save_json(profile, "dataset_profile.json")

    print_section("2. Exploratory charts")
    viz.plot_daily_sales(df)
    viz.plot_category_region(df)
    viz.plot_external_factors(df)
    viz.plot_inventory_vs_sold(df)
    viz.plot_correlation(df)

    print_section("3. Feature engineering")
    featured = engineer_features(df)
    split = temporal_split(featured)
    print(f"Cutoff date: {split.cutoff.date()}  train={len(split.train):,}  test={len(split.test):,}")

    y_train = split.train[TARGET]
    y_test = split.test[TARGET]

    print_section("4. Demand forecasting (regression)")
    baseline_rows = []
    baseline_preds = baseline_predictions(y_train, split.test)
    for name, preds in baseline_preds.items():
        row = {"model": name, "feature_set": "baseline", **regression_report(y_test, preds)}
        baseline_rows.append(row)
        print(row)

    op_models = train_regressors(split.train, split.test, y_train, y_test, "operational")
    vendor_models = train_regressors(split.train, split.test, y_train, y_test, "vendor")
    all_reg = op_models + vendor_models
    reg_metrics = pd.DataFrame(baseline_rows + [m.metrics for m in all_reg])
    save_csv(reg_metrics, "model_performance.csv")
    print(reg_metrics.to_string(index=False))

    viz.plot_model_bars(reg_metrics, "WAPE", "Regression WAPE (lower is better)", "regression_wape.png")
    viz.plot_model_bars(reg_metrics, "MAPE_y>=10", "Regression MAPE on demand ≥ 10", "regression_mape.png")

    best_vendor = min(
        [m for m in vendor_models],
        key=lambda m: m.metrics["WAPE"],
    )
    best_operational = min(op_models, key=lambda m: m.metrics["WAPE"])
    print("Best vendor-track model:", best_vendor.name, best_vendor.metrics)
    print("Best operational model:", best_operational.name, best_operational.metrics)

    viz.plot_pred_vs_actual(
        y_test, best_vendor.predictions, f"{best_vendor.name} (vendor track) vs actual", "pred_vs_actual_vendor.png"
    )
    viz.plot_pred_vs_actual(
        y_test,
        best_operational.predictions,
        f"{best_operational.name} (operational track) vs actual",
        "pred_vs_actual_operational.png",
    )
    viz.plot_scatter_pred(
        y_test, best_vendor.predictions, f"{best_vendor.name} vendor track", "scatter_vendor.png"
    )
    viz.plot_scatter_pred(
        y_test, best_operational.predictions, f"{best_operational.name} operational track", "scatter_operational.png"
    )
    viz.plot_residuals(y_test, best_vendor.predictions, "residuals_vendor.png")

    importance_frames = [m.importances for m in all_reg if m.importances is not None]
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    if not importance.empty:
        save_csv(importance, "feature_importance.csv")
        best_imp = importance[importance["model"] == best_vendor.name]
        if best_imp.empty:
            best_imp = importance
        viz.plot_feature_importance(
            best_imp[best_imp["model"] == best_imp["model"].iloc[0]],
            f"Feature importance — {best_vendor.name}",
            "feature_importance.png",
        )

    pred_out = split.test[["Date", "Store ID", "Product ID", "Category", "Region", TARGET, "Demand Forecast", "Inventory Level", "Price"]].copy()
    pred_out["pred_operational"] = best_operational.predictions
    pred_out["pred_vendor_track"] = best_vendor.predictions
    pred_out["residual_vendor_track"] = pred_out[TARGET] - pred_out["pred_vendor_track"]
    save_csv(pred_out, "predictions.csv")

    print_section("5. Demand-level classification")
    y_train_c, y_test_c, low_q, high_q = make_demand_labels(y_train, y_test)
    print(f"LOW < {low_q:.1f}   HIGH > {high_q:.1f}")
    clf_op = train_classifiers(split.train, split.test, y_train_c, y_test_c, "operational")
    clf_vendor = train_classifiers(split.train, split.test, y_train_c, y_test_c, "vendor")
    clf_metrics = pd.DataFrame([m.metrics for m in clf_op + clf_vendor])
    save_csv(clf_metrics, "classification_performance.csv")
    print(clf_metrics.to_string(index=False))
    viz.plot_model_bars(clf_metrics, "accuracy", "Classification accuracy", "classification_accuracy.png")
    best_clf = max(clf_vendor, key=lambda m: m.metrics["accuracy"])
    viz.plot_confusion(y_test_c, best_clf.predictions, f"{best_clf.name} vendor track", "confusion_matrix.png")

    print_section("6. Inventory optimization")
    # Fit residual scale on TRAIN using the vendor-track model applied to train.
    X_train_v = best_vendor.encoder.transform(split.train, scale=best_vendor.scale)
    train_pred = np.clip(best_vendor.estimator.predict(X_train_v), 0, None)
    sigma = residual_std(y_train, train_pred)
    z_star, z_table = tune_z(
        y_train.to_numpy(),
        train_pred,
        split.train["Price"].to_numpy(dtype=float),
        sigma,
    )
    save_csv(z_table, "safety_stock_grid.csv")
    print(f"Train residual std={sigma:.3f}  selected z={z_star}")
    print(z_table.to_string(index=False))
    policy_df = compare_policies(split.test, best_vendor.predictions, sigma, z_star)
    save_csv(policy_df, "inventory_optimization.csv")
    print(policy_df.to_string(index=False))
    viz.plot_inventory_costs(policy_df)

    print_section("7. SKU clustering")
    skus = sku_features(df)
    clustered, cluster_metrics = cluster_skus(skus, n_clusters=3)
    save_csv(clustered, "sku_segments.csv")
    viz.plot_clusters(clustered)
    print(cluster_metrics)

    print_section("8. Business insights")
    vendor_imp = None
    if not importance.empty:
        vendor_imp = importance[
            (importance["model"] == best_vendor.name)
        ]
        # Feature importance table is concatenated across feature sets; keep vendor-track rows
        # by matching the length of vendor encoder names when possible.
        vendor_imp = vendor_imp.drop_duplicates(subset=["feature"]).sort_values(
            "importance", ascending=False
        )
    insights = business_insights(df, vendor_imp)
    insights["temporal_cutoff"] = str(split.cutoff.date())
    insights["demand_class_thresholds"] = {"LOW_lt": low_q, "HIGH_gt": high_q}
    insights["best_regression"] = best_vendor.metrics
    insights["best_operational_regression"] = best_operational.metrics
    insights["best_classifier"] = best_clf.metrics
    insights["inventory_policy"] = policy_df.to_dict(orient="records")
    insights["selected_z"] = z_star
    insights["residual_std"] = sigma
    insights["clustering"] = {
        "silhouette": cluster_metrics["silhouette"],
        "davies_bouldin": cluster_metrics["davies_bouldin"],
        "n_clusters": cluster_metrics["n_clusters"],
    }
    save_json(insights, "business_insights.json")

    summary = {
        "profile": profile,
        "cutoff": str(split.cutoff.date()),
        "best_vendor_regression": best_vendor.metrics,
        "best_operational_regression": best_operational.metrics,
        "best_classifier": best_clf.metrics,
        "inventory": policy_df.to_dict(orient="records"),
        "clustering": insights["clustering"],
        "results_dir": str(RESULTS_DIR),
    }
    save_json(summary, "run_summary.json")
    print_section("Done")
    print(f"Wrote artifacts to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    run()
