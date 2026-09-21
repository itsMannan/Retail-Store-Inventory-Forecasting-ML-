"""Run the numbered Python scripts in order.

    python3 scripts/run_all.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run as run_pipeline
from src.utils import print_section


def main() -> None:
    print_section("Full pipeline (EDA, features, models, inventory)")
    run_pipeline()
    print_section("Evaluation script")
    runpy.run_path(str(Path(__file__).with_name("05_evaluation.py")), run_name="__main__")


if __name__ == "__main__":
    main()
