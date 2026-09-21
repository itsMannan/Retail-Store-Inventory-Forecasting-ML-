"""Filesystem helpers and small shared utilities. Results are CSV only."""

from __future__ import annotations

from pathlib import Path

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


def flatten_mapping(mapping: dict, prefix: str = "") -> list[dict]:
    """Turn nested dict/list values into key,value rows for CSV."""
    rows: list[dict] = []
    for key, value in mapping.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(flatten_mapping(value, full_key))
        elif isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    rows.extend(flatten_mapping(item, f"{full_key}[{i}]"))
            else:
                rows.append({"key": full_key, "value": " | ".join(str(v) for v in value)})
        else:
            rows.append({"key": full_key, "value": value})
    return rows


def save_mapping_csv(mapping: dict, name: str) -> Path:
    return save_csv(pd.DataFrame(flatten_mapping(mapping)), name)


def print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")
