# Comify Virtual Try-On - RunPod Deployment
# Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install additional AI dependencies
RUN pip install \
    insightface \
    onnxruntime-gpu \
    segment-anything \
    controlnet-aux \
    accelerate \
    transformers \
    diffusers \
    safetensors \
    xformers \
    triton

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/backend/data/results /app/backend/data/uploads

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Back to app root
WORKDIR /app

# Expose ports
# 8000 - Backend API
# 3000 - Frontend
# 8188 - ComfyUI (optional)
EXPOSE 8000 3000 8188

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Comify Virtual Try-On..."\n\
\n\
# Start backend\n\
cd /app/backend\n\
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
\n\
# Start frontend\n\
cd /app/frontend\n\
npm start &\n\
FRONTEND_PID=$!\n\
\n\
echo "Backend running on port 8000"\n\
echo "Frontend running on port 3000"\n\
\n\
# Wait for any process to exit\n\
wait -n\n\
\n\
# Exit with status of process that exited first\n\
exit $?\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]
