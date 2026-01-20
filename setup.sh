#!/bin/bash
set -euo pipefail

echo "======================================"
echo "  txttmd Setup"
echo "======================================"
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check Python version
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or later is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Docker (optional)
echo ""
if command -v docker &> /dev/null; then
    echo "Docker found: $(docker --version)"
    DOCKER_AVAILABLE=true
else
    echo "Docker not found (optional, for containerized deployment)"
    DOCKER_AVAILABLE=false
fi

# Run setup wizard
echo ""
echo "======================================"
echo "  Starting Setup Wizard"
echo "======================================"
echo ""

python3 -m src.setup.wizard

# Post-wizard options
echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "To start txttmd:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run: python -m src.main"
echo ""

if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "Or use Docker:"
    echo "  cd docker && docker-compose up -d"
    echo ""
fi

echo "For help: python -m src.cli.notectl --help"
