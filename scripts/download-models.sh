#!/bin/bash
# Download AI Models for Boutique Try-On System
# These models are unrestricted and suitable for all garment types

set -e

SCRIPT_DIR="$(dirname "$0")"
MODELS_DIR="$SCRIPT_DIR/../ComfyUI/models"
CHECKPOINTS_DIR="$MODELS_DIR/checkpoints"
CONTROLNET_DIR="$MODELS_DIR/controlnet"
VAE_DIR="$MODELS_DIR/vae"

echo "=============================================="
echo "  Boutique Try-On - Model Downloader"
echo "=============================================="
echo ""

# Create directories
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$CONTROLNET_DIR"
mkdir -p "$VAE_DIR"

# Function to download with progress
download_model() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "$dest" ]; then
        echo "[SKIP] $name already exists"
        return 0
    fi

    echo "[DOWNLOADING] $name..."
    curl -L --progress-bar -o "$dest" "$url"
    echo "[DONE] $name"
}

echo ""
echo "=== CHECKPOINT MODELS ==="
echo "These are the main image generation models"
echo ""

# ============================================
# RECOMMENDED MODELS FOR BOUTIQUE USE
# ============================================

# 1. Realistic Vision V6.0 - Best for realistic fashion photography
# Unrestricted, excellent for all garment types including lingerie, swimwear
echo "1. Realistic Vision V6.0 (Recommended for fashion photography)"
download_model \
    "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors" \
    "$CHECKPOINTS_DIR/realisticVisionV60B1_v51VAE.safetensors" \
    "Realistic Vision V6.0"

# 2. SD 1.5 Inpainting - Required for inpainting workflows
echo ""
echo "2. Stable Diffusion 1.5 Inpainting (Required for try-on)"
download_model \
    "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt" \
    "$CHECKPOINTS_DIR/sd-v1-5-inpainting.safetensors" \
    "SD 1.5 Inpainting"

# 3. Deliberate V3 - High quality, artistic renders
echo ""
echo "3. Deliberate V3 (Artistic fashion renders)"
download_model \
    "https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v3.safetensors" \
    "$CHECKPOINTS_DIR/deliberate_v3.safetensors" \
    "Deliberate V3"

# 4. DreamShaper 8 - Versatile, good for creative styling
echo ""
echo "4. DreamShaper 8 (Creative styling)"
download_model \
    "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors" \
    "$CHECKPOINTS_DIR/dreamshaper_8.safetensors" \
    "DreamShaper 8"

# 5. ChilloutMix - Excellent for Asian fashion, lingerie, swimwear
echo ""
echo "5. ChilloutMix (Fashion-focused, unrestricted)"
download_model \
    "https://huggingface.co/emilianJR/chilloutmix_NiPrunedFp32Fix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors" \
    "$CHECKPOINTS_DIR/chilloutmix.safetensors" \
    "ChilloutMix"

echo ""
echo "=== VAE MODELS ==="
echo "These improve image quality and colors"
echo ""

# VAE for better image quality
download_model \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
    "$VAE_DIR/vae-ft-mse-840000-ema-pruned.safetensors" \
    "SD VAE ft-mse"

echo ""
echo "=== CONTROLNET MODELS ==="
echo "These help maintain body pose and structure"
echo ""

# ControlNet OpenPose - For body pose preservation
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
    "$CONTROLNET_DIR/control_v11p_sd15_openpose.pth" \
    "ControlNet OpenPose"

# ControlNet Canny - For edge detection
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" \
    "$CONTROLNET_DIR/control_v11p_sd15_canny.pth" \
    "ControlNet Canny"

# ControlNet Depth - For depth preservation
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
    "$CONTROLNET_DIR/control_v11f1p_sd15_depth.pth" \
    "ControlNet Depth"

echo ""
echo "=============================================="
echo "  Download Complete!"
echo "=============================================="
echo ""
echo "Models installed to:"
echo "  Checkpoints: $CHECKPOINTS_DIR"
echo "  ControlNet:  $CONTROLNET_DIR"
echo "  VAE:         $VAE_DIR"
echo ""
echo "Recommended model for boutique use:"
echo "  - Realistic Vision V6.0 (best quality)"
echo "  - ChilloutMix (fashion-focused)"
echo ""
echo "To start the system, run:"
echo "  cd ~/comify && ./scripts/start-all.sh"
echo ""
