#!/bin/bash
# Start all services for Virtual Try-On

SCRIPT_DIR="$(dirname "$0")"

echo "=========================================="
echo "  Virtual Try-On - Starting All Services"
echo "=========================================="
echo ""

# Check if ComfyUI models exist
CHECKPOINT_DIR="$SCRIPT_DIR/../ComfyUI/models/checkpoints"
if [ ! -f "$CHECKPOINT_DIR/sd-v1-5-inpainting.safetensors" ] && [ ! -f "$CHECKPOINT_DIR/sd-v1-5-inpainting.ckpt" ]; then
    echo "WARNING: No inpainting model found!"
    echo "Please download SD 1.5 Inpainting model to:"
    echo "  $CHECKPOINT_DIR/"
    echo ""
    echo "Run: cd $CHECKPOINT_DIR && curl -L -o sd-v1-5-inpainting.ckpt 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start ComfyUI in background
echo "1. Starting ComfyUI..."
osascript -e "tell app \"Terminal\" to do script \"cd $SCRIPT_DIR/.. && ./scripts/start-comfyui.sh\"" 2>/dev/null || \
    (cd "$SCRIPT_DIR/.." && ./scripts/start-comfyui.sh &)

sleep 5

# Start Backend in background
echo "2. Starting Backend API..."
osascript -e "tell app \"Terminal\" to do script \"cd $SCRIPT_DIR/.. && ./scripts/start-backend.sh\"" 2>/dev/null || \
    (cd "$SCRIPT_DIR/.." && ./scripts/start-backend.sh &)

sleep 3

# Start Frontend
echo "3. Starting Frontend..."
osascript -e "tell app \"Terminal\" to do script \"cd $SCRIPT_DIR/.. && ./scripts/start-frontend.sh\"" 2>/dev/null || \
    (cd "$SCRIPT_DIR/.." && ./scripts/start-frontend.sh &)

echo ""
echo "=========================================="
echo "  All services starting!"
echo "=========================================="
echo ""
echo "  ComfyUI:  http://127.0.0.1:8188"
echo "  Backend:  http://127.0.0.1:8000/docs"
echo "  Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop this script"
echo "(Services will continue running in their terminals)"
echo ""

# Keep script running
wait
