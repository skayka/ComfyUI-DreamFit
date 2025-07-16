#!/bin/bash
# Quick lint script for DreamFit
echo "🔍 Running flake8 linter..."
python3 -m flake8 nodes/ dreamfit_core/ \
  --max-line-length=120 \
  --ignore=E501,W503,W293,W291,W292 \
  --exclude=__pycache__,*.pyc,.git \
  --show-source \
  --statistics

echo ""
echo "💡 Key error codes to watch for:"
echo "  F401 - Unused imports"
echo "  F821 - Undefined variable (like our recent bug!)"
echo "  F841 - Unused local variables"
echo "  E999 - Syntax errors"