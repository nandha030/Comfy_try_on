# Comify Virtual Try-On - Installation Guide

## Overview

Comify is a professional virtual try-on system for fashion boutiques that runs entirely locally. It features:

- **Face Preservation**: Maintains original face identity across generations
- **Body/Skin Tone Matching**: Matches skin tones and body proportions
- **Uncensored Models**: Supports all garment types (lingerie, swimwear, etc.)
- **Hardware Auto-Detection**: Automatically configures for your GPU/CPU

---

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB free space
- **GPU**: Optional but recommended

### Recommended for GPU Acceleration
| GPU Type | VRAM | Performance |
|----------|------|-------------|
| RTX 3090 | 24GB | Excellent |
| RTX 4090 | 24GB | Best |
| RTX 4080 | 16GB | Very Good |
| RTX 3080 | 10GB | Good |
| Apple M1/M2/M3 | 8-32GB | Good (MPS) |

---

## Quick Start (Local Development)

### 1. Clone Repository
```bash
git clone https://github.com/nandha030/comify.git
cd comify
```

### 2. Run Installer
```bash
python install.py
```

The installer will:
- Detect your hardware (GPU/CPU)
- Create a Python virtual environment
- Install dependencies
- Download required AI models
- Set up the database

### 3. Start Application
```bash
./start.sh
# or on Windows:
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
