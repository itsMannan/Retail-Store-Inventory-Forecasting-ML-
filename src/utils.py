"""Filesystem helpers and small shared utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, VIZ_DIR


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str) -> Path:
    ensure_output_dirs()
    path = RESULTS_DIR / name
    df.to_csv(path, index=False)
    return path


def save_json(payload: dict, name: str) -> Path:
    ensure_output_dirs()
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")
