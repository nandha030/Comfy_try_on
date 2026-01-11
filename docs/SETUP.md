# ComfyUI Virtual Try-On Setup Guide

## Quick Start

### 1. Start ComfyUI
```bash
cd ~/comify
./scripts/start-comfyui.sh
```
Access at: http://127.0.0.1:8188

### 2. Start Backend API
```bash
./scripts/start-backend.sh
```
Access at: http://127.0.0.1:8000/docs

---

## Phase 2: Model Downloads

### Required Models

Place models in `ComfyUI/models/checkpoints/`:

| Model | Size | Purpose | Download |
|-------|------|---------|----------|
| SD 1.5 Inpainting | ~4GB | Main try-on model | [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt) |
| SD 1.5 Base | ~4GB | Fallback/testing | [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors) |

### Download Commands

```bash
cd ~/comify/ComfyUI/models/checkpoints

# SD 1.5 Inpainting (RECOMMENDED)
curl -L -o sd-v1-5-inpainting.safetensors \
  "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"

# SD 1.5 Base (Optional)
curl -L -o v1-5-pruned-emaonly.safetensors \
  "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
```

---

## Phase 4: ControlNet (Optional)

### Install ControlNet Nodes
```bash
cd ~/comify/ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd ..
source ../comfyui-env/bin/activate
pip install -r custom_nodes/comfyui_controlnet_aux/requirements.txt
```

### ControlNet Models

Place in `ComfyUI/models/controlnet/`:

| Model | Purpose | Download |
|-------|---------|----------|
| OpenPose | Body pose control | [HuggingFace](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth) |

```bash
cd ~/comify/ComfyUI/models/controlnet
curl -L -o control_v11p_sd15_openpose.pth \
  "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth"
```

---

## Phase 5: GPU VM Setup (Production)

### RunPod/Vast.ai Setup

1. **Rent GPU VM**
   - GPU: RTX 3090 or 4090
   - Disk: 80GB+
   - Expose port: 8188

2. **Install on VM**
```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py --listen 0.0.0.0 --port 8188
```

3. **SDXL Models for Higher Quality**

Place in `models/checkpoints/`:
- `sd_xl_base_1.0.safetensors`
- SDXL Inpainting model

---

## Project Structure

```
comify/
├── ComfyUI/                 # ComfyUI installation
│   ├── models/
│   │   ├── checkpoints/     # SD models go here
│   │   ├── controlnet/      # ControlNet models
│   │   └── ...
│   └── output/              # Generated images
├── backend/                 # FastAPI server
│   ├── main.py
│   ├── uploads/             # User uploads
│   └── outputs/             # Processed results
├── frontend/                # Next.js app (TBD)
├── workflows/               # ComfyUI workflow JSONs
├── scripts/                 # Startup scripts
└── docs/                    # Documentation
```

---

## Troubleshooting

### ComfyUI won't start
- Ensure Python 3.10+ is installed
- Check PyTorch compatibility: `python -c "import torch; print(torch.__version__)"`
- Use `--cpu` flag on Mac

### Model not loading
- Verify model file is in correct folder
- Check file isn't corrupted (compare file size)
- Restart ComfyUI after adding models

### Out of memory
- Use `--lowvram` flag
- Reduce resolution to 512x512
- Use `--cpu` mode (slower but works)

---

## API Usage Example

```python
import requests

# Submit try-on job
files = {
    'person_image': open('person.jpg', 'rb'),
    'mask_image': open('mask.png', 'rb'),
}
data = {
    'prompt': 'person wearing elegant dress',
    'consent': True
}

response = requests.post('http://localhost:8000/api/tryon', files=files, data=data)
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/api/result/{job_id}')
print(status.json())
```
