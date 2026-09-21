"""Regression and classification model factories plus training loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from src.config import RANDOM_STATE
from src.metrics import classification_report_dict, demand_classes, regression_report
from src.preprocessor import FeatureEncoder


@dataclass
class TrainedModel:
    name: str
    feature_set: str
    estimator: object
    encoder: FeatureEncoder
    scale: bool
    predictions: np.ndarray
    metrics: dict
    importances: pd.DataFrame | None = None
    proba: np.ndarray | None = None


def _regression_catalog() -> dict:
    return {
        "Linear Regression": (
            LinearRegression(),
            True,
        ),
        "Ridge": (
            Ridge(alpha=1.0),
            True,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
            ),
            False,
        ),
    }


def _classification_catalog() -> dict:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=400,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
            ),
            False,
        ),
    }


def _feature_importance(name: str, estimator, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(values) != len(feature_names):
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": values, "model": name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_regressors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_set: str,
) -> list[TrainedModel]:
    encoder = FeatureEncoder(feature_set=feature_set).fit(train_df)
    trained: list[TrainedModel] = []
    for name, (estimator, scale) in _regression_catalog().items():
        X_train = encoder.transform(train_df, scale=scale)
        X_test = encoder.transform(test_df, scale=scale)
        estimator.fit(X_train, y_train)
        preds = np.clip(estimator.predict(X_test), 0, None)
        trained.append(
            TrainedModel(
                name=name,
                feature_set=feature_set,
                estimator=estimator,
                encoder=encoder,
                scale=scale,
                predictions=preds,
                metrics={"model": name, "feature_set": feature_set, **regression_report(y_test, preds)},
                importances=_feature_importance(name, estimator, list(X_train.columns)),
            )
        )
    return trained


def train_classifiers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_set: str,
) -> list[TrainedModel]:
    encoder = FeatureEncoder(feature_set=feature_set).fit(train_df)
    labels = ["LOW", "MEDIUM", "HIGH"]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    trained: list[TrainedModel] = []
    for name, (estimator, scale) in _classification_catalog().items():
        X_train = encoder.transform(train_df, scale=scale)
        X_test = encoder.transform(test_df, scale=scale)
        if name == "XGBoost":
            estimator.fit(X_train, y_train.map(label_to_idx))
            proba = estimator.predict_proba(X_test)
            idx = np.argmax(proba, axis=1)
            preds = np.array(labels)[idx]
        else:
            estimator.fit(X_train, y_train)
            preds = estimator.predict(X_test)
            proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
            if proba is not None:
                # Align probability columns to LOW/MEDIUM/HIGH.
                class_order = list(estimator.classes_)
                aligned = np.zeros((len(preds), len(labels)))
                for j, cls in enumerate(class_order):
                    aligned[:, label_to_idx[cls]] = proba[:, j]
                proba = aligned
        trained.append(
            TrainedModel(
                name=name,
                feature_set=feature_set,
                estimator=estimator,
                encoder=encoder,
                scale=scale,
                predictions=preds,
                proba=proba,
                metrics={
                    "model": name,
                    "feature_set": feature_set,
                    **classification_report_dict(y_test, preds, proba, labels=labels),
                },
                importances=_feature_importance(name, estimator, list(X_train.columns)),
            )
        )
    return trained


def make_demand_labels(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, pd.Series, float, float]:
    low_q = float(y_train.quantile(0.25))
    high_q = float(y_train.quantile(0.75))
    return demand_classes(y_train, low_q, high_q), demand_classes(y_test, low_q, high_q), low_q, high_q


def baseline_predictions(train_y: pd.Series, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    mean_pred = np.full(len(test_df), float(train_y.mean()))
    vendor = test_df["Demand Forecast"].to_numpy(dtype=float)
    half_inventory = test_df["Inventory Level"].to_numpy(dtype=float) * 0.5
    return {
        "Mean Baseline": mean_pred,
        "Vendor Demand Forecast": vendor,
        "Half of Inventory": half_inventory,
    }
