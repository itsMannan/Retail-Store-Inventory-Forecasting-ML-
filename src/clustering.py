"""Product/store clustering for inventory segmentation (ABC-style)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET


def sku_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["Store ID", "Product ID"], as_index=False).agg(
        mean_sales=(TARGET, "mean"),
        std_sales=(TARGET, "std"),
        mean_price=("Price", "mean"),
        mean_inventory=("Inventory Level", "mean"),
        mean_discount=("Discount", "mean"),
        promo_rate=("Holiday/Promotion", "mean"),
        revenue=(TARGET, "sum"),
    )
    grouped["turnover"] = grouped["mean_sales"] / grouped["mean_inventory"]
    grouped["revenue"] = grouped["mean_price"] * grouped["revenue"]
    grouped["cv_sales"] = grouped["std_sales"] / grouped["mean_sales"].replace(0, pd.NA)
    grouped["cv_sales"] = grouped["cv_sales"].fillna(0.0)
    return grouped


def cluster_skus(sku_df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    feature_cols = ["mean_sales", "std_sales", "mean_price", "mean_inventory", "turnover", "cv_sales"]
    X = sku_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(Xs)
    out = sku_df.copy()
    out["cluster"] = labels
    # Map clusters to A/B/C by mean revenue (A = highest).
    rank = out.groupby("cluster")["revenue"].mean().sort_values(ascending=False)
    mapping = {cluster: letter for letter, cluster in zip(["A", "B", "C", "D", "E"], rank.index)}
    out["segment"] = out["cluster"].map(mapping)
    segment_means = out.groupby("segment")[feature_cols + ["revenue"]].mean().round(2).reset_index()
    metrics = {
        "n_clusters": n_clusters,
        "silhouette": round(float(silhouette_score(Xs, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Xs, labels)), 4),
        "inertia": round(float(model.inertia_), 4),
    }
    return out, metrics, segment_means
