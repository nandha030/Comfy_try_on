# RunPod Deployment Guide for Comify Virtual Try-On

## RunPod Configuration Options

### Option 1: GPU Pod (Recommended for Development/Testing)

Use this for persistent deployment with a web interface.

#### Recommended GPU Configurations:

| GPU Type | VRAM | Speed | Cost/hr | Best For |
|----------|------|-------|---------|----------|
| RTX 3090 | 24GB | Fast | ~$0.44 | Development, testing |
| RTX 4090 | 24GB | Very Fast | ~$0.74 | Production, fast inference |
| A100 40GB | 40GB | Very Fast | ~$1.64 | High throughput, batch processing |
| RTX 4080 | 16GB | Fast | ~$0.54 | Budget production |
| RTX 3080 | 10GB | Moderate | ~$0.30 | Budget option (may need optimization) |

#### Minimum Requirements:
- **VRAM**: 12GB+ (16GB+ recommended)
- **RAM**: 32GB+
- **Storage**: 50GB+ (for models)
- **CUDA**: 12.0+

### Option 2: Serverless Endpoint (Recommended for Production)

Use this for pay-per-request pricing with auto-scaling.

---

## Deployment Steps

### Option 1: GPU Pod Deployment

#### Step 1: Create Pod on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy"
3. Select your GPU (RTX 3090 or RTX 4090 recommended)
4. Choose template: **RunPod Pytorch 2.1.0**
5. Configure:
   - Container Disk: 50GB
   - Volume Disk: 100GB (for models)
   - Expose HTTP Ports: 8000, 3000

#### Step 2: Setup the Application

SSH into your pod and run:

```bash
# Clone the repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/comify.git
cd comify

# Run the installer
python install.py

# When prompted:
# - Select "cuda" for device
# - Choose to download models

# Start the services
./start.sh
```

#### Step 3: Access the Application

- Frontend: `https://YOUR_POD_ID-3000.proxy.runpod.net`
- Backend API: `https://YOUR_POD_ID-8000.proxy.runpod.net`

---

### Option 2: Serverless Endpoint Deployment

#### Step 1: Build and Push Docker Image

On your local machine:

```bash
# Login to Docker Hub (or your registry)
docker login

# Build the RunPod-optimized image
docker build -f Dockerfile.runpod -t YOUR_DOCKERHUB/comify-runpod:latest .

# Push to registry
docker push YOUR_DOCKERHUB/comify-runpod:latest
```

#### Step 2: Create Serverless Endpoint on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: comify-tryon
   - **Container Image**: `YOUR_DOCKERHUB/comify-runpod:latest`
   - **GPU Type**: RTX 3090 or RTX 4090
   - **Max Workers**: 3 (adjust based on load)
   - **Idle Timeout**: 60 seconds
   - **Flash Boot**: Enable (for faster cold starts)

4. Add Network Volume (for models):
   - Create a Network Volume (100GB)
   - Mount at `/app/models`
   - Pre-download models to this volume

#### Step 3: Download Models to Network Volume

Create a temporary pod to download models:

```bash
# SSH into a temporary pod with the volume attached
cd /app/models

# Run the model downloader
python -c "
from installer.model_downloader import ModelDownloader
downloader = ModelDownloader('/app/models')
downloader.download_all()
"
```

#### Step 4: Use the API

```python
import requests
import base64

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

# Encode your image
with open("person.jpg", "rb") as f:
    person_b64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "action": "tryon",
            "person_image": person_b64,
            "preserve_face": True,
            "preserve_skin_tone": True,
            "steps": 20
        }
    }
)

result = response.json()
if "output" in result:
    # Decode result image
    result_bytes = base64.b64decode(result["output"]["result_image"])
    with open("result.png", "wb") as f:
        f.write(result_bytes)
```

---

## API Reference (Serverless)

### Try-On Generation

```json
POST /runsync or /run

{
    "input": {
        "action": "tryon",
        "person_image": "<base64 encoded image>",
        "garment_image": "<base64 encoded image>",  // optional
        "mask_image": "<base64 encoded image>",     // optional, auto-generated
        "profile_id": "abc123",                     // optional, for consistent results
        "preserve_face": true,
        "preserve_skin_tone": true,
        "upscale": false,
        "face_restore": true,
        "steps": 20,
        "denoise": 0.85,
        "prompt": "person wearing elegant dress",
        "negative_prompt": "blurry, distorted",
        "seed": -1
    }
}

Response:
{
    "output": {
        "result_image": "<base64 encoded result>",
        "seed": 12345,
        "generation_time": 5.2,
        "face_detected": true,
        "body_detected": true
    }
}
```

### Extract Features

```json
{
    "input": {
        "action": "extract_features",
        "image": "<base64 encoded image>"
    }
}

Response:
{
    "output": {
        "face_detected": true,
        "body_detected": true,
        "body_shape": "average",
        "pose_type": "front",
        "skin_colors": {
            "face": [200, 170, 150],
            "arms": [195, 165, 145]
        }
    }
}
```

### Create Profile

```json
{
    "input": {
        "action": "create_profile",
        "image": "<base64 encoded image>",
        "name": "Model A"
    }
}

Response:
{
    "output": {
        "profile_id": "abc123def456",
        "name": "Model A",
        "face_detected": true,
        "body_shape": "average",
        "created_at": "2024-01-15T10:30:00"
    }
}
```

---

## Cost Optimization Tips

### For GPU Pods:
1. Use **Spot Instances** for development (up to 80% cheaper)
2. **Stop pods** when not in use
3. Use **Volume storage** to persist models between restarts

### For Serverless:
1. Enable **Flash Boot** to reduce cold start time
2. Set appropriate **Idle Timeout** (60-120 seconds)
3. Use **Network Volumes** for model storage
4. Pre-load models during build (increases image size but faster cold starts)

### Model Optimization:
1. Use **FP16** precision (enabled by default)
2. Enable **xformers** for memory-efficient attention
3. Reduce **inference steps** (15-20 is usually sufficient)

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size to 1
- Lower inference steps to 15
- Disable upscaling
- Use a GPU with more VRAM

### Slow Generation
- Ensure CUDA is being used (not CPU)
- Enable xformers
- Use Flash Boot for serverless
- Pre-load models

### Model Loading Errors
- Verify models are downloaded to correct path
- Check disk space
- Ensure network volume is mounted

### Connection Issues
- Check port exposure settings
- Verify proxy URL format
- Check RunPod status page

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/app/models` | Path to AI models |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) to use |
| `HF_HOME` | `/app/.cache` | HuggingFace cache directory |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDA memory config |

---

## Support

For issues specific to this deployment:
1. Check the logs in RunPod console
2. Verify model downloads completed
3. Test with the health check endpoint first

```bash
# Health check
curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'
```
