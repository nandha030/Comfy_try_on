# Comify Virtual Try-On - Installation Guide

## Overview

Comify is a professional virtual try-on system for fashion boutiques that runs entirely locally. It features:

- **Face Preservation**: Maintains original face identity across generations
- **Body/Skin Tone Matching**: Matches skin tones and body proportions
- **Uncensored Models**: Supports all garment types (lingerie, swimwear, etc.)
- **Hardware Auto-Detection**: Automatically configures for your GPU/CPU
- **Cross-Platform Support**: Works on Windows, macOS (Intel & Apple Silicon), and Linux

---

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB free space
- **GPU**: Optional but recommended

### Platform Support

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| Windows 10/11 | NVIDIA CUDA, AMD DirectML | Best NVIDIA support |
| macOS (Apple Silicon) | MPS (M1/M2/M3/M4) | Native ARM support |
| macOS (Intel) | CPU only | Limited performance |
| Linux (Ubuntu 20.04+) | NVIDIA CUDA, AMD ROCm | Server deployments |

### Recommended GPU Configurations
| GPU Type | VRAM | Performance | Platform |
|----------|------|-------------|----------|
| RTX 4090 | 24GB | Best | Windows/Linux |
| RTX 3090 | 24GB | Excellent | Windows/Linux |
| RTX 4080 | 16GB | Very Good | Windows/Linux |
| RTX 3080 | 10GB | Good | Windows/Linux |
| Apple M3 Max | 32GB+ | Excellent | macOS |
| Apple M2 Pro/Max | 16-32GB | Very Good | macOS |
| Apple M1/M2/M3 | 8-16GB | Good | macOS |
| AMD RX 7900 | 24GB | Good | Windows (DirectML) |

---

## Cross-Platform Installation

### Quick Install (Automatic)

```bash
# Clone the repository
git clone https://github.com/nandha030/Comfy_try_on.git
cd Comfy_try_on

# Run cross-platform setup (auto-detects your system)
python setup.py
```

The setup script will:
- Detect your platform (Windows/macOS/Linux)
- Detect your GPU type (NVIDIA/AMD/Apple Silicon/CPU)
- Install PyTorch with correct backend
- Install all dependencies
- Configure the system

---

## Platform-Specific Installation

### Windows Installation

```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install PyTorch (choose one):

# For NVIDIA GPU (CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD GPU (DirectML):
pip install torch torchvision
pip install torch-directml

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install requirements
pip install -r requirements-base.txt
pip install -r requirements-windows.txt
```

### macOS Installation (Apple Silicon)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (MPS support included automatically)
pip install torch torchvision torchaudio

# 3. Install requirements
pip install -r requirements-base.txt
pip install -r requirements-mac.txt
```

### macOS Installation (Intel)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (CPU only for Intel Mac)
pip install torch torchvision torchaudio

# 3. Install requirements
pip install -r requirements-base.txt
pip install -r requirements-mac.txt
```

### Linux Installation (Ubuntu/Debian)

```bash
# 0. Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (choose one):

# For NVIDIA GPU (CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD GPU (ROCm):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install requirements
pip install -r requirements-base.txt
pip install -r requirements-linux.txt
```

---

## Requirements Files

| File | Description |
|------|-------------|
| `requirements.txt` | Main file with all dependencies |
| `requirements-base.txt` | Platform-agnostic dependencies |
| `requirements-windows.txt` | Windows-specific (NVIDIA/AMD/CPU) |
| `requirements-mac.txt` | macOS-specific (Apple Silicon/Intel) |
| `requirements-linux.txt` | Linux-specific (NVIDIA/ROCm/CPU) |

---

## Download AI Models

After installation, download the AI models:

```bash
# Run the installer to download models
python install.py
```

---

## Start Application

```bash
# Linux/macOS:
./start.sh

# Windows:
start.bat
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## AI Models

### Essential Models (Required)

| Model | Description | Size | Auto-Download |
|-------|-------------|------|---------------|
| **InsightFace Antelopev2** | Face detection & embedding | ~300MB | Yes |
| **DWPose** | Body pose estimation | ~200MB | Yes |

### Base Models (Choose at least one)

| Model | Description | Size | Type |
|-------|-------------|------|------|
| **Realistic Vision V6** | Photorealistic generations | ~2GB | Uncensored |
| **Deliberate V6** | High quality, detailed | ~2GB | Uncensored |
| **epiCRealism** | Natural lighting focus | ~2GB | Uncensored |
| **CyberRealistic** | Enhanced realism | ~2GB | Uncensored |
| **SD 1.5 Inpainting** | For garment replacement | ~2GB | Required |

### Optional Enhancement Models

| Model | Description | Size |
|-------|-------------|------|
| **SD VAE (ft-mse)** | Better color reproduction | ~300MB |
| **GFPGAN** | Face restoration | ~350MB |
| **RealESRGAN x4** | Image upscaling | ~64MB |
| **Segment Anything** | Automatic masking | ~2.4GB |

---

## Model Download

### Automatic Download (Recommended)
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Download all essential models
python -c "from installer.model_downloader import ModelDownloader; d = ModelDownloader('./models'); d.download_all()"
```

### Download Specific Categories
```python
from installer.model_downloader import ModelDownloader

downloader = ModelDownloader('./models')

# Download only essential (face/body detection)
downloader.download_category('essential')

# Download uncensored base models
downloader.download_category('uncensored')

# Download enhancement models
downloader.download_category('enhancement')
```

### Manual Download

If automatic download fails, download models manually:

1. **InsightFace Antelopev2**
   - Download from: https://github.com/deepinsight/insightface
   - Place in: `models/insightface/models/antelopev2/`

2. **Realistic Vision V6**
   - Download from: https://civitai.com/models/4201/realistic-vision-v60-b1
   - Place in: `models/base_models/`

3. **SD 1.5 Inpainting**
   - Download from: https://huggingface.co/runwayml/stable-diffusion-inpainting
   - Place in: `models/base_models/`

---

## Directory Structure

```
comify/
├── ai_engine/          # AI processing modules
│   ├── pipeline.py     # Main generation pipeline
│   ├── face_processor.py
│   ├── body_processor.py
│   └── config.py
├── backend/            # FastAPI backend
│   ├── main.py         # Main API endpoints
│   ├── api_v2.py       # V2 advanced endpoints
│   └── database.py
├── frontend/           # Next.js frontend
│   └── src/
│       ├── app/        # App pages
│       ├── components/ # React components
│       └── lib/        # API client
├── installer/          # Installation scripts
│   ├── __init__.py
│   ├── hardware_detect.py
│   └── model_downloader.py
├── models/             # AI models (not in git)
│   ├── insightface/
│   ├── dwpose/
│   ├── base_models/
│   └── enhancement/
├── install.py          # Main installer
├── start.sh            # Start script
└── requirements.txt
```

---

## Configuration

### Hardware Configuration
The system auto-detects hardware. To manually configure:

```python
# config/hardware.json
{
    "device_type": "cuda",  # or "mps", "directml", "cpu"
    "device_name": "NVIDIA RTX 3090",
    "vram_gb": 24,
    "compute_backend": "cuda",
    "use_fp16": true,
    "model_profile": "full"
}
```

### Model Profile Options
- **full**: All features enabled (24GB+ VRAM)
- **balanced**: Most features (12-24GB VRAM)
- **light**: Essential features only (8-12GB VRAM)
- **minimal**: CPU-only mode

---

## RunPod Deployment

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for cloud deployment instructions.

### Quick RunPod Setup
```bash
# SSH into your RunPod
ssh your-pod@ssh.runpod.io

# Run setup script
curl -sSL https://raw.githubusercontent.com/nandha030/comify/main/runpod_setup.sh | bash

# Start application
cd /workspace/comify
./start_comify.sh
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
    "status": "healthy",
    "comfyui": "disconnected",
    "v2_engine": "ready",
    "v2_models": {
        "available": true,
        "face_detection": true,
        "body_pose": true,
        "base_model": true
    },
    "models": [...]
}
```

### V2 Advanced Try-On
```bash
curl -X POST http://localhost:8000/api/v2/tryon/advanced \
  -F "person_image=@person.jpg" \
  -F "mask_image=@mask.png" \
  -F "preserve_face=true" \
  -F "preserve_skin_tone=true" \
  -F "steps=20"
```

### Extract Features
```bash
curl -X POST http://localhost:8000/api/v2/extract-features \
  -F "image=@person.jpg"
```

---

## Troubleshooting

### "Generate Try-On" Button Greyed Out
1. Check if models are downloaded: `ls models/`
2. Check backend logs: `tail -f logs/backend.log`
3. Check health endpoint: `curl http://localhost:8000/api/health`

### Out of Memory (OOM)
- Reduce steps to 15
- Disable upscaling
- Use a lighter model profile
- Close other GPU applications

### Slow Generation
- Ensure GPU is being used (check hardware detection)
- Enable FP16: `use_fp16: true`
- Reduce image resolution

### Model Loading Errors
- Verify models are in correct directory
- Check file integrity (re-download if corrupt)
- Ensure sufficient disk space

---

## Support

- GitHub Issues: https://github.com/nandha030/comify/issues
- Documentation: This file and RUNPOD_DEPLOYMENT.md

---

## License

This software is for authorized use only. All processing is done locally - no data is sent to external servers.
