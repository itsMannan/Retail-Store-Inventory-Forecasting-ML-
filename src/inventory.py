"""Inventory policy simulation: holding vs stockout cost trade-off."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ANNUAL_HOLDING_RATE, LEAD_TIME_DAYS, STOCKOUT_MARGIN, Z_CANDIDATES


@dataclass
class PolicyResult:
    name: str
    z: float
    total_cost: float
    holding_cost: float
    stockout_cost: float
    stockout_rate: float
    mean_inventory: float
    leftover: float
    metrics: dict


def unit_costs(price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit daily holding cost and per-unit stockout (lost margin) cost."""
    holding = (ANNUAL_HOLDING_RATE * price) / 365.0
    stockout = STOCKOUT_MARGIN * price
    return holding, stockout


def evaluate_inventory(demand: np.ndarray, inventory: np.ndarray, price: np.ndarray) -> dict:
    demand = np.asarray(demand, dtype=float)
    inventory = np.asarray(inventory, dtype=float)
    price = np.asarray(price, dtype=float)
    leftover = np.maximum(inventory - demand, 0.0)
    lost = np.maximum(demand - inventory, 0.0)
    holding, stockout = unit_costs(price)
    # Holding is charged on the on-hand position (units sitting on the shelf).
    holding_cost = float(np.sum(inventory * holding))
    stockout_cost = float(np.sum(lost * stockout))
    return {
        "holding_cost": round(holding_cost, 2),
        "stockout_cost": round(stockout_cost, 2),
        "total_cost": round(holding_cost + stockout_cost, 2),
        "stockout_rate": round(float(np.mean(lost > 0)), 4),
        "lost_units": round(float(np.sum(lost)), 2),
        "leftover_units": round(float(np.sum(leftover)), 2),
        "mean_inventory": round(float(np.mean(inventory)), 2),
        "mean_demand": round(float(np.mean(demand)), 2),
        "fill_rate": round(float(1.0 - np.sum(lost) / max(np.sum(demand), 1e-9)), 4),
    }


def safety_stock(residual_std: float, z: float, lead_time: int = LEAD_TIME_DAYS) -> float:
    return z * residual_std * np.sqrt(lead_time)


def recommended_inventory(predicted_demand: np.ndarray, residual_std: float, z: float) -> np.ndarray:
    """Target on-hand stock covering lead-time demand plus safety stock.

    The dataset is a daily on-hand snapshot, so the policy is expressed as a
    recommended on-hand level = daily demand × lead time + safety stock.
    """
    predicted_demand = np.maximum(np.asarray(predicted_demand, dtype=float), 0.0)
    rec = predicted_demand * LEAD_TIME_DAYS + safety_stock(residual_std, z)
    return np.maximum(rec, 0.0)


def tune_z(
    demand: np.ndarray,
    predicted_demand: np.ndarray,
    price: np.ndarray,
    residual_std: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    best_z = Z_CANDIDATES[0]
    best_cost = float("inf")
    for z in Z_CANDIDATES:
        rec = recommended_inventory(predicted_demand, residual_std, z)
        stats = evaluate_inventory(demand, rec, price)
        rows.append({"z": z, **stats})
        if stats["total_cost"] < best_cost:
            best_cost = stats["total_cost"]
            best_z = z
    return best_z, pd.DataFrame(rows)


def compare_policies(
    test_df: pd.DataFrame,
    predicted_demand: np.ndarray,
    residual_std: float,
    z: float,
) -> pd.DataFrame:
    demand = test_df["Units Sold"].to_numpy(dtype=float)
    price = test_df["Price"].to_numpy(dtype=float)
    current = evaluate_inventory(demand, test_df["Inventory Level"].to_numpy(dtype=float), price)
    rec_inv = recommended_inventory(predicted_demand, residual_std, z)
    recommended = evaluate_inventory(demand, rec_inv, price)
    naive = evaluate_inventory(demand, predicted_demand, price)

    current["policy"] = "current_on_hand"
    recommended["policy"] = f"recommended_z{z:.2f}"
    naive["policy"] = "match_forecast_no_safety"

    frame = pd.DataFrame([current, naive, recommended])
    base = current["total_cost"]
    frame["cost_reduction_vs_current_pct"] = np.where(
        base == 0,
        0.0,
        (base - frame["total_cost"]) / base * 100.0,
    ).round(2)
    return frame


def residual_std(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.std(y_true - y_pred, ddof=1))
