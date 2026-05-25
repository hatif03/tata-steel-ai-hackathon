#!/usr/bin/env bash
# Create and bootstrap the project virtual environment (Linux/macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then train:"
echo "  python models/xgboost-baseline/train.py"
