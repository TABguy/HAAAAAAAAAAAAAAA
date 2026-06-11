#!/usr/bin/env bash
# HAKS 2026 — environment setup. OS-aware. Always work in the venv.
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
echo "Detected OS: $OS"

if [ ! -d venv ]; then
  python3 -m venv venv
  echo "Created venv/"
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup done. Next:"
echo "  source venv/bin/activate"
echo "  scripts/run.sh cv        # grouped-CV Brier sanity check"
echo "Note: Docling (doc parsing) is a heavy optional extra -> pip install -r requirements-docling.txt"
