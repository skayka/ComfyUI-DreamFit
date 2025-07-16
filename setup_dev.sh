#!/bin/bash

# Setup development environment for DreamFit

echo "Setting up DreamFit development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks if using pre-commit
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pip install pre-commit
    pre-commit install
fi

echo "Development environment setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"