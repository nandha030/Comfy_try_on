#!/bin/bash
# ============================================================
# Model Download Script — Virtual Try-On (comify)
# Target: /Volumes/T7 Shield/ComfyUI/comify/ComfyUI/models
# ============================================================

set -e

MODELS_DIR="/Volumes/T7 Shield/ComfyUI/comify/ComfyUI/models"
VENV="/Volumes/T7 Shield/ComfyUI/comify/comfyui-env"

# ── helpers ──────────────────────────────────────────────────
ok()   { echo "  [ok]  $1"; }
skip() { echo "  [--]  already exists: $1"; }
dl()   { echo "  [dl]  downloading: $1 (~$2)"; }

curl_dl() {
    local dest="$1"; local url="$2"; local label="$3"; local size="$4"
    if [ -f "$dest" ]; then
        skip "$label"
    else
        dl "$label" "$size"
        curl -L --progress-bar -o "$dest" "$url"
        ok "$label"
    fi
}

hf_snapshot() {
    local repo="$1"; local dest="$2"; local label="$3"
    if [ -f "$dest/config.json" ] || [ -f "$dest/model_index.json" ]; then
        skip "$label (already has config)"
    else
        dl "$label" "varies"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', local_dir='$dest', local_dir_use_symlinks=False)
"
        ok "$label"
    fi
}

echo ""
echo "============================================================"
echo " Virtual Try-On Model Download Script"
echo " Models dir: $MODELS_DIR"
echo "============================================================"
echo ""

# ── activate venv & install huggingface_hub ───────────────────
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "[env] Using venv at $VENV"
else
    echo "[env] No venv found, using system Python"
fi
pip install -q huggingface_hub

# ── 1. IDM-VTON ───────────────────────────────────────────────
echo ""
echo "=== 1/8  IDM-VTON (yisol/IDM-VTON) ==="
hf_snapshot "yisol/IDM-VTON" "$MODELS_DIR/IDM-VTON" "IDM-VTON"

# ── 2. CatVTON ───────────────────────────────────────────────
echo ""
echo "=== 2/8  CatVTON (zhengchong/CatVTON) ==="
hf_snapshot "zhengchong/CatVTON" "$MODELS_DIR/CatVTON" "CatVTON"

# ── 3. Checkpoints ────────────────────────────────────────────
echo ""
echo "=== 3/8  Checkpoints ==="
mkdir -p "$MODELS_DIR/checkpoints"

curl_dl \
    "$MODELS_DIR/checkpoints/sd-v1-5-inpainting.ckpt" \
    "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt" \
    "SD 1.5 Inpainting checkpoint" "4.0 GB"

curl_dl \
    "$MODELS_DIR/checkpoints/realisticVisionV60B1_v51VAE.safetensors" \
    "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors" \
    "Realistic Vision V6 checkpoint" "2.1 GB"

# ── 4. ControlNet ─────────────────────────────────────────────
echo ""
echo "=== 4/8  ControlNet ==="
mkdir -p "$MODELS_DIR/controlnet"

curl_dl \
    "$MODELS_DIR/controlnet/control_v11p_sd15_openpose.pth" \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
    "ControlNet OpenPose SD1.5" "1.4 GB"

# ── 5. SAM (Segment Anything) ─────────────────────────────────
echo ""
echo "=== 5/8  SAM models ==="
mkdir -p "$MODELS_DIR/sams"

# ViT-H (standard) — required by IDM-VTON workflow
curl_dl \
    "$MODELS_DIR/sams/sam_vit_h_4b8939.pth" \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
    "SAM ViT-H (standard)" "2.4 GB"

# ViT-B (lightweight fallback)
curl_dl \
    "$MODELS_DIR/sams/sam_vit_b_01ec64.pth" \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
    "SAM ViT-B (lightweight)" "375 MB"

# SAM-HQ ViT-H
curl_dl \
    "$MODELS_DIR/sams/sam_hq_vit_h.pth" \
    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth" \
    "SAM-HQ ViT-H" "2.6 GB"

# ── 6. GroundingDINO ─────────────────────────────────────────
echo ""
echo "=== 6/8  GroundingDINO ==="
mkdir -p "$MODELS_DIR/grounding-dino"

curl_dl \
    "$MODELS_DIR/grounding-dino/groundingdino_swint_ogc.pth" \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
    "GroundingDINO SwinT (small)" "694 MB"

curl_dl \
    "$MODELS_DIR/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py" \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py" \
    "GroundingDINO SwinT config" "tiny"

curl_dl \
    "$MODELS_DIR/grounding-dino/groundingdino_swinb_cogcoor.pth" \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth" \
    "GroundingDINO SwinB (large)" "938 MB"

curl_dl \
    "$MODELS_DIR/grounding-dino/GroundingDINO_SwinB.cfg.py" \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py" \
    "GroundingDINO SwinB config" "tiny"

# ── 7. Face models ────────────────────────────────────────────
echo ""
echo "=== 7/8  Face detection & restoration ==="
mkdir -p "$MODELS_DIR/facedetection"
mkdir -p "$MODELS_DIR/facerestore_models"

curl_dl \
    "$MODELS_DIR/facedetection/detection_Resnet50_Final.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
    "RetinaFace detector" "104 MB"

curl_dl \
    "$MODELS_DIR/facedetection/parsing_parsenet.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
    "ParseNet face parser" "85 MB"

curl_dl \
    "$MODELS_DIR/facerestore_models/codeformer-v0.1.0.pth" \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" \
    "CodeFormer face restoration" "376 MB"

# ── 8. Upscale / misc ─────────────────────────────────────────
echo ""
echo "=== 8/8  Upscale & misc models ==="
mkdir -p "$MODELS_DIR/realesrgan"
mkdir -p "$MODELS_DIR/insightface"
mkdir -p "$MODELS_DIR/dwpose"

curl_dl \
    "$MODELS_DIR/realesrgan/RealESRGAN_x4.pth" \
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" \
    "RealESRGAN x4 upscaler" "67 MB"

curl_dl \
    "$MODELS_DIR/insightface/inswapper_128.onnx" \
    "https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx" \
    "InsightFace inswapper" "270 MB"

# DWPose (used by comfyui_controlnet_aux for pose detection)
curl_dl \
    "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx" \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx" \
    "DWPose body keypoints model" "270 MB"

curl_dl \
    "$MODELS_DIR/dwpose/yolox_l.onnx" \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx" \
    "DWPose person detector" "207 MB"

# ── done ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All models downloaded!"
echo " Location: $MODELS_DIR"
echo "============================================================"
echo ""
echo "Model summary:"
echo "  IDM-VTON          → $MODELS_DIR/IDM-VTON/"
echo "  CatVTON           → $MODELS_DIR/CatVTON/"
echo "  Checkpoints       → $MODELS_DIR/checkpoints/"
echo "  ControlNet        → $MODELS_DIR/controlnet/"
echo "  SAM models        → $MODELS_DIR/sams/"
echo "  GroundingDINO     → $MODELS_DIR/grounding-dino/"
echo "  Face detection    → $MODELS_DIR/facedetection/"
echo "  Face restoration  → $MODELS_DIR/facerestore_models/"
echo "  RealESRGAN        → $MODELS_DIR/realesrgan/"
echo "  InsightFace       → $MODELS_DIR/insightface/"
echo "  DWPose            → $MODELS_DIR/dwpose/"
echo ""
