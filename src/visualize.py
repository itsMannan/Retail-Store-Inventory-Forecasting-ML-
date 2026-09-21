"""Chart helpers. All figures are written under results/visualizations/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import TARGET, VIZ_DIR
from src.utils import ensure_output_dirs

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, name: str) -> Path:
    ensure_output_dirs()
    path = VIZ_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_daily_sales(df: pd.DataFrame) -> Path:
    daily = df.groupby("Date")[TARGET].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily.values, color="#1f4e79", linewidth=1.2)
    ax.set_title("Total units sold by day")
    ax.set_ylabel("Units sold")
    ax.set_xlabel("Date")
    return _save(fig, "daily_sales.png")


def plot_category_region(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cat = df.groupby("Category")[TARGET].mean().sort_values(ascending=False)
    region = df.groupby("Region")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=cat.index, y=cat.values, ax=axes[0], color="#2a9d8f")
    sns.barplot(x=region.index, y=region.values, ax=axes[1], color="#e76f51")
    axes[0].set_title("Average units sold by category")
    axes[1].set_title("Average units sold by region")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("")
    axes[0].set_ylabel("Mean units sold")
    return _save(fig, "category_region_sales.png")


def plot_external_factors(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    weather = df.groupby("Weather Condition")[TARGET].mean()
    promo = df.groupby("Holiday/Promotion")[TARGET].mean()
    season = df.groupby("Seasonality")[TARGET].mean()
    sns.barplot(x=weather.index, y=weather.values, ax=axes[0], color="#457b9d")
    sns.barplot(x=["No promo", "Promo"], y=promo.values, ax=axes[1], color="#e9c46a")
    sns.barplot(x=season.index, y=season.values, ax=axes[2], color="#9b5de5")
    axes[0].set_title("Weather")
    axes[1].set_title("Promotion")
    axes[2].set_title("Seasonality label")
    for ax in axes:
        ax.set_ylabel("Mean units sold")
    return _save(fig, "external_factors.png")


def plot_inventory_vs_sold(df: pd.DataFrame, sample: int = 8000) -> Path:
    plot_df = df.sample(min(sample, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(
        plot_df["Inventory Level"],
        plot_df[TARGET],
        s=8,
        alpha=0.25,
        color="#1f4e79",
    )
    max_v = max(plot_df["Inventory Level"].max(), plot_df[TARGET].max())
    ax.plot([0, max_v], [0, max_v], color="crimson", linestyle="--", linewidth=1, label="Sold = inventory")
    ax.set_xlabel("Inventory level")
    ax.set_ylabel("Units sold")
    ax.set_title("Sales are capped by on-hand inventory")
    ax.legend()
    return _save(fig, "inventory_vs_sold.png")


def plot_correlation(df: pd.DataFrame) -> Path:
    cols = [
        TARGET,
        "Inventory Level",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Holiday/Promotion",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Numeric feature correlations")
    return _save(fig, "correlation_heatmap.png")


def plot_pred_vs_actual(y_true, y_pred, title: str, name: str) -> Path:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(400, len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(y_true[:n], label="Actual", linewidth=1.4)
    ax.plot(y_pred[:n], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Test row (first 400)")
    ax.set_ylabel("Units sold")
    ax.legend()
    return _save(fig, name)


def plot_scatter_pred(y_true, y_pred, title: str, name: str) -> Path:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(6000, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.25, color="#1f4e79")
    lo = 0
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--")
    ax.set_xlabel("Actual units sold")
    ax.set_ylabel("Predicted units sold")
    ax.set_title(title)
    return _save(fig, name)


def plot_residuals(y_true, y_pred, name: str) -> Path:
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(resid, bins=50, color="#2a9d8f", edgecolor="white")
    ax.set_title("Prediction residual distribution")
    ax.set_xlabel("Actual − predicted")
    ax.set_ylabel("Count")
    return _save(fig, name)


def plot_feature_importance(importance: pd.DataFrame, title: str, name: str, top: int = 15) -> Path:
    top_df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"], color="#1f4e79")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _save(fig, name)


def plot_model_bars(metrics_df: pd.DataFrame, value_col: str, title: str, name: str) -> Path:
    plot_df = metrics_df.copy()
    plot_df["label"] = plot_df["model"] + "\n(" + plot_df["feature_set"] + ")"
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="label", y=value_col, ax=ax, color="#457b9d")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, name)


def plot_confusion(y_true, y_pred, title: str, name: str) -> Path:
    labels = ["LOW", "MEDIUM", "HIGH"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    return _save(fig, name)


def plot_inventory_costs(policy_df: pd.DataFrame) -> Path:
    melted = policy_df.melt(
        id_vars=["policy"],
        value_vars=["holding_cost", "stockout_cost"],
        var_name="cost_type",
        value_name="cost",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="policy", y="cost", hue="cost_type", ax=ax)
    ax.set_title("Inventory cost by policy (test period)")
    ax.set_xlabel("")
    ax.set_ylabel("Cost ($)")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "inventory_cost_comparison.png")


def plot_clusters(sku_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=sku_df,
        x="mean_price",
        y="mean_sales",
        hue="segment",
        size="mean_inventory",
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Store-product segments")
    return _save(fig, "sku_clusters.png")
