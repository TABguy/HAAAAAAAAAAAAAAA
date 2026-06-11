#!/usr/bin/env bash
# HAKS 2026 — run helper. OS-aware. macOS reserves port 5000 (AirDrop) -> use 8501.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source venv/bin/activate

OS="$(uname -s)"
PORT=8501   # safe on macOS (never 5000) and Linux
MODE="${1:-cv}"

case "$MODE" in
  cv)      python src/corrosion_model.py ;;
  submit)  python src/corrosion_model.py --submit ;;
  app)     echo "Streamlit on :$PORT ($OS)"; streamlit run app.py --server.port "$PORT" ;;
  *)       echo "usage: scripts/run.sh [cv|submit|app]"; exit 1 ;;
esac
