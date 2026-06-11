#!/usr/bin/env bash
# HAKS 2026 — Run helper script
# Usage: ./scripts/run.sh [cv|submit]

set -euo pipefail
cd "$(dirname "$0")/.."

# Check if venv exists
if [ ! -d venv ]; then
    echo "❌ Error: Virtual environment not found"
    echo "Run './scripts/setup.sh' first"
    exit 1
fi

# Activate virtual environment
# shellcheck disable=SC1091
source venv/bin/activate

# Get mode (default: cv)
MODE="${1:-cv}"

echo "🚀 HAKS 2026 - Wing Corrosion"
echo "===================================="
echo

case "$MODE" in
    cv)
        echo "📊 Running cross-validation..."
        echo
        python src/corrosion_model.py
        ;;
    
    submit)
        echo "📝 Generating submission file..."
        echo
        python src/corrosion_model.py --submit
        ;;
    
    *)
        echo "❌ Error: Unknown mode '$MODE'"
        echo
        echo "Usage: ./scripts/run.sh [cv|submit]"
        echo
        echo "Modes:"
        echo "  cv      - Run cross-validation (default)"
        echo "  submit  - Generate submission file"
        exit 1
        ;;
esac

echo
echo "===================================="
echo "✅ Done!"

# Made with Bob
