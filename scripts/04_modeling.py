"""04 — Modeling: demand regression and LOW/MEDIUM/HIGH classification.

Trains Linear Regression, Ridge, Random Forest, and XGBoost on both
feature tracks (operational vs vendor forecast). This is the same catalog
as python3 -m src.pipeline.

Run from the repo root:

    python3 scripts/04_modeling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET
from src.data_loader import load_dataset
from src.feature_engineer import engineer_features
from src.models import make_demand_labels, train_classifiers, train_regressors
from src.preprocessor import temporal_split
from src.utils import print_section, save_csv


def main() -> None:
    featured = engineer_features(load_dataset())
    split = temporal_split(featured)
    y_train, y_test = split.train[TARGET], split.test[TARGET]
    print_section("Split")
    print("cutoff", split.cutoff.date(), "train", len(split.train), "test", len(split.test))

    print_section("Approach 1 — demand regression")
    op = train_regressors(split.train, split.test, y_train, y_test, "operational")
    vendor = train_regressors(split.train, split.test, y_train, y_test, "vendor")
    reg = pd.DataFrame([m.metrics for m in op + vendor])
    print(reg.to_string(index=False))
    save_csv(reg, "model_performance.csv")

    print_section("Approach 2 — LOW / MEDIUM / HIGH classification")
    y_tr_c, y_te_c, low_q, high_q = make_demand_labels(y_train, y_test)
    print("LOW <", low_q, "HIGH >", high_q)
    clf_op = train_classifiers(split.train, split.test, y_tr_c, y_te_c, "operational")
    clf_vendor = train_classifiers(split.train, split.test, y_tr_c, y_te_c, "vendor")
    clf = pd.DataFrame([m.metrics for m in clf_op + clf_vendor])
    print(clf.to_string(index=False))
    save_csv(clf, "classification_performance.csv")

    print_section("How to read this")
    print("Operational models should land near the half-of-inventory baseline (R2 about 0.33).")
    print("Vendor-track models should match or slightly beat the given forecast (R2 about 0.99, WAPE about 6 percent).")
    print("That is forecast value added: does ML beat the forecast you already have?")


if __name__ == "__main__":
    main()
