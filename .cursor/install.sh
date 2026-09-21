#!/usr/bin/env bash
# Idempotent environment bootstrap for the Retail Store Inventory Forecasting project.
# Installs the Python venv toolchain (system) and project dependencies into ./.venv.
set -euo pipefail

cd "$(dirname "$0")/.."

# System packages required to create virtual environments and build native wheels.
# Guarded so the script stays fast and safe to re-run.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    python3.12-venv python3-dev build-essential
fi

# Create the virtual environment if it does not already exist.
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

# Install / refresh Python dependencies.
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
