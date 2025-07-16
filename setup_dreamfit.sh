#!/bin/bash

echo "Setting up DreamFit patches..."

# Check if DreamFit-official exists
if [ ! -d "DreamFit-official" ]; then
    echo "Error: DreamFit-official directory not found!"
    echo "Please run: git clone https://github.com/bytedance/DreamFit DreamFit-official"
    exit 1
fi

# Backup original math.py if it exists and hasn't been backed up
if [ -f "DreamFit-official/src/flux/math.py" ] && [ ! -f "DreamFit-official/src/flux/math_original.py" ]; then
    echo "Backing up original math.py..."
    cp DreamFit-official/src/flux/math.py DreamFit-official/src/flux/math_original.py
fi

# Copy our patched math.py
echo "Applying flash_attn optional patch..."
cp dreamfit_patches/math.py DreamFit-official/src/flux/math.py

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Setup complete! DreamFit is now configured to work without flash_attn."
echo "You can now restart ComfyUI."