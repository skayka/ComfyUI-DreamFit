#!/bin/bash
echo "Starting flash-attn installation with verbose output..."
echo "This will take 10-20 minutes. Progress will be shown throughout."
echo "=========================================="

# Function to keep connection alive
keep_alive() {
    while kill -0 $1 2>/dev/null; do
        echo -n "."
        sleep 10
    done
}

# Install ninja first
echo "Step 1/3: Installing ninja..."
pip install ninja -v

# Set architecture for RTX 5090
echo "Step 2/3: Setting CUDA architecture..."
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4  # Limit parallel jobs to avoid memory issues

# Install flash-attn with maximum verbosity
echo "Step 3/3: Installing flash-attn (this will take 10-20 minutes)..."
echo "Starting at: $(date)"

# Run pip install in background and monitor
pip install flash-attn --no-build-isolation -vvv 2>&1 | while IFS= read -r line; do
    echo "[$(date +%H:%M:%S)] $line"
done &

PID=$!
keep_alive $PID &
KEEP_ALIVE_PID=$!

wait $PID
RESULT=$?

kill $KEEP_ALIVE_PID 2>/dev/null

echo "=========================================="
echo "Installation completed at: $(date)"
echo "Exit code: $RESULT"

# Test if it worked
echo "Testing installation..."
python -c "import flash_attn; print(f'SUCCESS: flash-attn {flash_attn.__version__} installed!')" || echo "FAILED: flash-attn not installed"