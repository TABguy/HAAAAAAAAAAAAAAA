#!/usr/bin/env bash
# HAKS 2026 — Environment setup script
# Creates virtual environment and installs dependencies

set -euo pipefail
cd "$(dirname "$0")/.."

echo "🚀 HAKS 2026 - Wing Corrosion Setup"
echo "===================================="
echo

# Detect OS
OS="$(uname -s)"
echo "📍 Detected OS: $OS"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d venv ]; then
    echo
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo
echo "🔧 Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# Upgrade pip
echo
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install requirements
echo
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Verify installation
echo
echo "🔍 Verifying installation..."
python -c "import pandas, numpy, sklearn; print('✅ All core packages installed')"

# Check if data files exist
echo
echo "📊 Checking data files..."
if [ -f "input/environment_training.csv" ]; then
    echo "✅ Training data found"
else
    echo "⚠️  Warning: input/environment_training.csv not found"
fi

if [ -f "input/corrosions_training.csv" ]; then
    echo "✅ Corrosion data found"
else
    echo "⚠️  Warning: input/corrosions_training.csv not found"
fi

# Success message
echo
echo "===================================="
echo "✅ Setup complete!"
echo
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source venv/bin/activate"
echo
echo "  2. Run cross-validation:"
echo "     python src/corrosion_model.py"
echo
echo "  3. Generate submission:"
echo "     python src/corrosion_model.py --submit"
echo
echo "Or use the run script:"
echo "  ./scripts/run.sh cv       # Cross-validation"
echo "  ./scripts/run.sh submit   # Generate submission"
echo "===================================="

# Made with Bob
