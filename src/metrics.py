"""Evaluation metrics for regression and classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def _as_float(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    """Mean absolute percentage error in percent, with a floor on |y|.

    Classic MAPE is undefined / explosive when units sold are 0. Retail
    reporting typically floors the denominator (here, 1 unit).
    """
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)) * 100)


def mape_at_least(y_true, y_pred, min_actual: float = 10.0) -> float:
    """MAPE on rows where actual demand is at least `min_actual` units."""
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    mask = np.abs(y_true) >= min_actual
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error in percent (retail standard)."""
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def smape(y_true, y_pred) -> float:
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denom, 1e-9)) * 100)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def within_tolerance(y_true, y_pred, pct: float = 0.20) -> float:
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= pct * np.maximum(np.abs(y_true), 1.0)) * 100)


def regression_report(y_true, y_pred) -> dict:
    y_true = _as_float(y_true)
    y_pred = _as_float(y_pred)
    return {
        "MAPE": round(mape(y_true, y_pred), 4),
        "MAPE_y>=10": round(mape_at_least(y_true, y_pred, 10), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
        "WAPE": round(wape(y_true, y_pred), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "within_20pct": round(within_tolerance(y_true, y_pred, 0.20), 4),
        "n": int(len(y_true)),
    }


def classification_report_dict(y_true, y_pred, y_proba=None, labels=None) -> dict:
    labels = labels or ["LOW", "MEDIUM", "HIGH"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    report = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }
    for label, p, r, f in zip(labels, precision, recall, f1):
        report[f"precision_{label}"] = round(float(p), 4)
        report[f"recall_{label}"] = round(float(r), 4)
        report[f"f1_{label}"] = round(float(f), 4)
    if y_proba is not None:
        try:
            report["roc_auc_ovr"] = round(
                float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)),
                4,
            )
        except ValueError:
            report["roc_auc_ovr"] = float("nan")
    return report


def demand_classes(series: pd.Series, low_q: float, high_q: float) -> pd.Series:
    return pd.Series(
        np.where(series < low_q, "LOW", np.where(series > high_q, "HIGH", "MEDIUM")),
        index=series.index,
        name="demand_class",
    )
