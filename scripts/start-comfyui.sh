#!/bin/bash
# Start ComfyUI server for local development

cd "$(dirname "$0")/.."

echo "Starting ComfyUI..."
echo "================================"

source comfyui-env/bin/activate

cd ComfyUI

# Check if running on Mac with MPS support
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Running on Mac - using CPU mode"
    echo "Access the UI at: http://127.0.0.1:8188"
    python3 main.py --cpu
else
    echo "Running with GPU"
    python3 main.py --listen 0.0.0.0 --port 8188
fi
