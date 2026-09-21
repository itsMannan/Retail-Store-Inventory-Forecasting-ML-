"""05 — Evaluation, inventory policy, and business takeaways.

Reads CSV artifacts from python3 -m src.pipeline (or scripts/04_modeling.py).

Run from the repo root:

    python3 scripts/05_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, VIZ_DIR
from src.utils import print_section


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python3 -m src.pipeline` first."
        )
    return pd.read_csv(path)


def main() -> None:
    perf = _read("model_performance.csv")
    clf = _read("classification_performance.csv")
    inv = _read("inventory_optimization.csv")

    print_section("Regression leaderboard (lower WAPE is better)")
    print(perf.sort_values("WAPE").to_string(index=False))

    print_section("Classification leaderboard (higher accuracy is better)")
    print(clf.sort_values("accuracy", ascending=False).to_string(index=False))

    print_section("Inventory policies")
    print(inv.to_string(index=False))
    grid_path = RESULTS_DIR / "safety_stock_grid.csv"
    if grid_path.exists():
        print_section("Safety-stock grid (train)")
        print(pd.read_csv(grid_path).to_string(index=False))

    sku_path = RESULTS_DIR / "sku_segment_means.csv"
    if sku_path.exists():
        print_section("SKU segment means")
        print(pd.read_csv(sku_path).to_string(index=False))

    notes_path = RESULTS_DIR / "data_quality_notes.csv"
    if notes_path.exists():
        print_section("Data-quality notes")
        for note in pd.read_csv(notes_path)["note"]:
            print("-", note)

    print_section("Charts written by the pipeline")
    for name in [
        "regression_wape.png",
        "regression_mape.png",
        "scatter_vendor.png",
        "scatter_operational.png",
        "pred_vs_actual_vendor.png",
        "feature_importance.png",
        "confusion_matrix.png",
        "classification_accuracy.png",
        "inventory_cost_comparison.png",
        "sku_clusters.png",
    ]:
        path = VIZ_DIR / name
        print(("OK  " if path.exists() else "MISS"), path)

    print_section("Takeaways")
    print("1. Use the vendor Demand Forecast as the production signal, optionally refined by Random Forest.")
    print("2. Operational-only ML cannot honestly beat about R2 0.35 on this file.")
    print("3. Vendor-track classifiers exceed 75 percent accuracy; operational-only sit near 55-60 percent.")
    print("4. Current on-hand is far above realized daily demand. Forecast + safety stock cuts holding cost.")
    print("5. Pricing / weather / promotions show no material mean shift. Do not change discount policy from this extract.")
    print("6. Prefer WAPE and MAE for reporting. Quote MAPE only on days with demand >= 10.")


if __name__ == "__main__":
    main()
