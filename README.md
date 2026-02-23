# Comify - Virtual Try-On System

A professional virtual try-on solution for fashion boutiques and e-commerce. Upload a person's photo and a garment image, and see how the clothing looks on them - all processed locally on your machine.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)
![License](https://img.shields.io/badge/License-Private-red.svg)

---

## What is Comify?

Comify is a complete virtual try-on system that lets you:

- **Try clothes on anyone** - Upload a person's photo and see how different garments look on them
- **Works with all clothing types** - Shirts, dresses, pants, swimwear, lingerie - no restrictions
- **Preserves identity** - Face and body features remain intact after try-on
- **Runs locally** - All processing happens on your computer, no data sent to external servers
- **Supports any hardware** - Works on NVIDIA GPUs, AMD GPUs, Apple Silicon Macs, or CPU-only

### Who is this for?

- **Fashion boutiques** - Let customers see how clothes look before buying
- **E-commerce stores** - Create product images with virtual models
- **Photographers** - Quick clothing mockups for clients
- **Fashion designers** - Visualize designs on different body types

---

## Quick Start

### Prerequisites

- **Python 3.9 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **Node.js 18+** (for the web interface) - [Download Node.js](https://nodejs.org/)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/nandha030/Comfy_try_on.git
cd Comfy_try_on

# Run the setup script - it handles everything!
python setup.py
```

The setup script will automatically:
1. Detect your operating system (Windows, macOS, or Linux)
2. Detect your GPU (NVIDIA, AMD, Apple Silicon, or CPU)
3. Create a Python virtual environment
4. Install PyTorch with the correct GPU support
5. Install all dependencies
6. Download ComfyUI and required extensions
7. Download AI models (optional, ~10-15GB)
8. Create start/stop scripts

### Start the Application

```bash
# Windows
start.bat

# macOS / Linux
./start.sh
```

Then open your browser to: **http://localhost:3000**

---

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Person Photo   │     │  Garment Image  │     │   Try-On Result │
│                 │ ──► │                 │ ──► │                 │
│  (Full body)    │     │  (Clothing)     │     │  (Combined)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Upload a person photo** - Full body shot works best
2. **Upload a garment** - The clothing item you want to try on
3. **Select the area** - Mark where the garment should go (optional, auto-detected)
4. **Generate** - AI processes the images and creates the try-on result
5. **Download** - Save the result or try another garment

---

## System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| CPU | 4 cores |
| RAM | 16 GB |
| Storage | 50 GB free space |
| Python | 3.9 or higher |

### Recommended Hardware

| Hardware | VRAM/Memory | Performance | Best For |
|----------|-------------|-------------|----------|
| NVIDIA RTX 4090 | 24 GB | Excellent | Production use |
| NVIDIA RTX 3080 | 10 GB | Very Good | Regular use |
| Apple M2 Pro/Max | 16-32 GB | Very Good | Mac users |
| Apple M1/M2 | 8-16 GB | Good | Casual use |
| AMD RX 7900 | 24 GB | Good | Windows AMD users |
| CPU Only | - | Slow | Testing only |

### Supported Platforms

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| Windows 10/11 | NVIDIA (CUDA), AMD (DirectML) | Best NVIDIA support |
| macOS 12+ (Apple Silicon) | MPS | Native M1/M2/M3 support |
| macOS (Intel) | CPU only | Limited performance |
| Ubuntu 20.04+ | NVIDIA (CUDA), AMD (ROCm) | Server deployments |

---

## Project Structure

```
Comfy_try_on/
├── setup.py                 # One-click installer
├── start.sh / start.bat     # Start scripts
├── requirements-*.txt       # Platform-specific dependencies
│
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main API server
│   └── database.py         # SQLite database
│
├── frontend/                # Next.js web interface
│   └── src/
│       ├── app/            # Pages
│       └── components/     # React components
│
├── ai_engine/              # AI processing modules
│   ├── pipeline.py         # Main processing pipeline
│   ├── face_processor.py   # Face detection/preservation
│   └── body_processor.py   # Body/pose detection
│
├── installer/              # Installation utilities
│   ├── hardware_detect.py  # GPU detection
│   └── model_downloader.py # AI model downloader
│
├── ComfyUI/                # (Created by setup.py)
│   └── custom_nodes/       # Try-on extensions
│
├── models/                 # (Downloaded by setup.py)
│   ├── checkpoints/        # Base AI models
│   └── insightface/        # Face detection models
│
└── workflows/              # ComfyUI workflow files
```

---

## Configuration

### Hardware Auto-Detection

The system automatically detects your hardware and configures optimal settings:

```bash
# View detected hardware
python -c "from installer.hardware_detect import detect_hardware, print_hardware_info; print_hardware_info(detect_hardware())"
```

### Manual Configuration

Edit `config/hardware.json` to override settings:

```json
{
  "device_type": "cuda",
  "device_name": "NVIDIA RTX 3080",
  "vram_gb": 10,
  "compute_backend": "cuda",
  "model_profile": "medium"
}
```

### Model Profiles

| Profile | VRAM Required | Features | Speed |
|---------|--------------|----------|-------|
| `high` | 12+ GB | All features, best quality | ~15 sec |
| `medium` | 8-12 GB | Most features | ~25 sec |
| `low` | 6-8 GB | Basic features | ~40 sec |
| `cpu_optimized` | - | Minimal features | ~5 min |

---

## Downloading AI Models

Models are downloaded during setup, but you can also download them manually:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Download all required models for try-on
python installer/model_downloader.py --tryon

# Download specific categories
python installer/model_downloader.py --category essential      # Face/body detection
python installer/model_downloader.py --category base_models    # Main AI models
python installer/model_downloader.py --category enhancement    # Upscaling, face restore

# List available models
python installer/model_downloader.py --list
```

### Model Sizes

| Category | Size | Description |
|----------|------|-------------|
| Essential | ~500 MB | Face and body detection |
| Base Models | ~6 GB | Main generation models |
| Enhancement | ~500 MB | Upscaling and face restoration |
| Full Setup | ~10-15 GB | Everything included |

---

## API Usage

The backend provides a REST API for integration:

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Try-On Generation
```bash
curl -X POST http://localhost:8000/api/v2/tryon/advanced \
  -F "person_image=@person.jpg" \
  -F "garment_image=@shirt.jpg" \
  -F "preserve_face=true" \
  -F "steps=20"
```

### API Documentation
Visit **http://localhost:8000/docs** for interactive API documentation.

---

## Troubleshooting

### "Setup fails to detect GPU"

**Windows (NVIDIA):**
```bash
# Check if NVIDIA driver is installed
nvidia-smi
```
If not found, install [NVIDIA drivers](https://www.nvidia.com/drivers).

**macOS (Apple Silicon):**
MPS is automatically available on M1/M2/M3 Macs with macOS 12+.

**Force CPU mode:**
```bash
python setup.py --cpu
```

### "Out of memory error"

Reduce the number of inference steps or use a smaller model profile:
```bash
# Edit config/ai_config.json
{
  "num_inference_steps": 12,
  "model_profile": "low"
}
```

### "Models fail to download"

Download manually from these sources:
- [Hugging Face](https://huggingface.co/) - Most AI models
- [CivitAI](https://civitai.com/) - Stable Diffusion models

Place downloaded files in the `models/` directory.

### "Frontend won't start"

```bash
cd frontend
npm install
npm run dev
```

Make sure Node.js 18+ is installed.

---

## Development

### Running in Development Mode

```bash
# Terminal 1: Backend
cd backend
source ../venv/bin/activate
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: ComfyUI (optional)
cd ComfyUI
source ../venv/bin/activate
python main.py
```

### Adding Custom Nodes

Edit `custom_nodes.json` to add new ComfyUI extensions:

```json
{
  "nodes": [
    {
      "name": "MyCustomNode",
      "url": "https://github.com/user/repo.git",
      "required": false
    }
  ]
}
```

Then run:
```bash
python setup.py --no-models
```

---

## Cloud Deployment

### RunPod

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for deploying to RunPod GPU cloud.

### Docker

```bash
docker build -t comify .
docker run -p 3000:3000 -p 8000:8000 comify
```

---

## Credits

This project uses several open-source technologies:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based AI workflow engine
- [IDM-VTON](https://github.com/yisol/IDM-VTON) - Virtual try-on model
- [CatVTON](https://github.com/Zheng-Chong/CatVTON) - Alternative try-on model
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection
- [Stable Diffusion](https://stability.ai/) - Image generation backbone

---

## License

This software is for authorized use only. All processing is done locally - no data is sent to external servers.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/nandha030/Comfy_try_on/issues)
- **Documentation**: See `INSTALLATION.md` and `RUNPOD_DEPLOYMENT.md`

---

Made with care for fashion boutiques everywhere.
