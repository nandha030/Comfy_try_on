#!/usr/bin/env python3
"""
Comify Cross-Platform Setup
Run this once on any machine (Mac / Windows / Linux) to get started.
Models are already on the drive — no re-downloading needed.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path

BASE_DIR   = Path(__file__).parent.absolute()
COMFYUI    = BASE_DIR / "ComfyUI"
MODELS_DIR = COMFYUI / "models"

SYSTEM = platform.system()          # Darwin | Windows | Linux
ARCH   = platform.machine()         # arm64 | x86_64 | AMD64

# Platform-specific venv names so they can coexist on the same drive
VENV_NAMES = {"Darwin": "comfyui-env-mac", "Windows": "comfyui-env-win", "Linux": "comfyui-env-linux"}
VENV_DIR   = BASE_DIR / VENV_NAMES.get(SYSTEM, "comfyui-env")


# ── helpers ────────────────────────────────────────────────────────────────

def ok(msg):   print(f"  \033[92m✓\033[0m {msg}")
def info(msg): print(f"  \033[94m→\033[0m {msg}")
def warn(msg): print(f"  \033[93m⚠\033[0m {msg}")
def header(msg):
    print(f"\n\033[1;95m{'─'*55}\033[0m")
    print(f"\033[1;95m  {msg}\033[0m")
    print(f"\033[1;95m{'─'*55}\033[0m\n")


def python_exe():
    if SYSTEM == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python3"


def pip_exe():
    if SYSTEM == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip3"


def run(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ── hardware detection ──────────────────────────────────────────────────────

def detect_hardware():
    info(f"OS: {SYSTEM}  Arch: {ARCH}")

    # NVIDIA GPU (Windows / Linux)
    if SYSTEM in ("Windows", "Linux"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            name, vram = out.split(",")
            vram_gb = float(vram.strip()) / 1024
            ok(f"NVIDIA GPU: {name.strip()} ({vram_gb:.1f} GB VRAM)")
            return {"device": "cuda", "torch_url": "https://download.pytorch.org/whl/cu121",
                    "torch_pkgs": ["torch", "torchvision", "torchaudio"]}
        except Exception:
            pass

    # Apple Silicon MPS
    if SYSTEM == "Darwin" and ARCH == "arm64":
        ok("Apple Silicon — MPS acceleration")
        return {"device": "mps", "torch_url": None,
                "torch_pkgs": ["torch", "torchvision", "torchaudio"]}

    # Intel Mac
    if SYSTEM == "Darwin":
        warn("Intel Mac — CPU mode")
        return {"device": "cpu", "torch_url": "https://download.pytorch.org/whl/cpu",
                "torch_pkgs": ["torch", "torchvision", "torchaudio"]}

    # AMD GPU on Windows (DirectML)
    if SYSTEM == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL, text=True
            )
            if "AMD" in out or "Radeon" in out:
                ok("AMD GPU — DirectML acceleration")
                return {"device": "directml", "torch_url": "https://download.pytorch.org/whl/cpu",
                        "torch_pkgs": ["torch", "torchvision"]}
        except Exception:
            pass

    # CPU fallback
    warn("No GPU detected — CPU mode")
    return {"device": "cpu", "torch_url": "https://download.pytorch.org/whl/cpu",
            "torch_pkgs": ["torch", "torchvision", "torchaudio"]}


# ── venv ───────────────────────────────────────────────────────────────────

def venv_healthy():
    py = python_exe()
    if not py.exists():
        return False
    try:
        subprocess.check_output([str(py), "-c", "import torch"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def create_venv():
    header("Creating Python environment")
    info(f"Location: {VENV_DIR}")
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    run(sys.executable, "-m", "venv", VENV_DIR)
    ok(f"Virtual environment created ({SYSTEM})")


def install_pytorch(hw):
    header("Installing PyTorch")
    pip = pip_exe()
    run(pip, "install", "--upgrade", "pip", "wheel", "setuptools")

    pkgs = hw["torch_pkgs"]
    if hw["torch_url"]:
        run(pip, "install", *pkgs, "--index-url", hw["torch_url"])
    else:
        run(pip, "install", *pkgs)

    if hw["device"] == "directml":
        run(pip, "install", "torch-directml")

    ok("PyTorch installed")


def install_comfyui_requirements():
    header("Installing ComfyUI requirements")
    pip = pip_exe()
    req = COMFYUI / "requirements.txt"
    if req.exists():
        run(pip, "install", "-r", req)
        ok("ComfyUI requirements installed")
    else:
        warn("ComfyUI/requirements.txt not found — skipping")


def install_custom_node_requirements():
    header("Installing custom node requirements")
    pip = pip_exe()
    cn_dir = COMFYUI / "custom_nodes"
    if not cn_dir.exists():
        return
    count = 0
    for node_dir in cn_dir.iterdir():
        req = node_dir / "requirements.txt"
        if req.exists():
            info(f"Installing: {node_dir.name}")
            try:
                run(pip, "install", "-r", req)
                count += 1
            except subprocess.CalledProcessError:
                warn(f"Some deps in {node_dir.name} failed — continuing")
    ok(f"Installed requirements for {count} custom nodes")


# ── model check ────────────────────────────────────────────────────────────

def check_models():
    header("Checking models (no downloads — already on drive)")
    checks = [
        ("VAE",            MODELS_DIR / "vae"               / "Wan2.1_VAE.pth"),
        ("Text encoder",   MODELS_DIR / "text_encoders"     / "models_t5_umt5-xxl-enc-bf16.pth"),
        ("BiRefNet",       MODELS_DIR / "BiRefNet"          / "model.safetensors"),
        ("SAM ViT-H",      MODELS_DIR / "sams"              / "sam_vit_h_4b8939.pth"),
        ("GFPGANv1.4",     MODELS_DIR / "facerestore_models" / "GFPGANv1.4.pth"),
        ("IC-Light FC",    MODELS_DIR / "unet"              / "iclight_sd15_fc_unet_ldm.safetensors"),
        ("YOLOv8 face",    MODELS_DIR / "ultralytics" / "bbox" / "face_yolov8m.pt"),
        ("buffalo_l 1k3d", MODELS_DIR / "insightface" / "models" / "buffalo_l" / "1k3d68.onnx"),
    ]
    missing = []
    for label, path in checks:
        if path.exists():
            ok(f"{label}  ({path.stat().st_size / 1e9:.2f} GB)")
        else:
            warn(f"{label}: NOT FOUND — {path}")
            missing.append(label)

    wan = MODELS_DIR / "diffusion_models" / "Wan2.2-T2V-A14B" / "high_noise_model" / "diffusion_pytorch_model-00001-of-00006.safetensors"
    if wan.exists():
        ok("Wan2.2-T2V model shards: present")
    else:
        warn("Wan2.2-T2V model: NOT FOUND (still downloading?)")
        missing.append("Wan2.2-T2V")

    if missing:
        warn(f"Missing models: {', '.join(missing)}")
    else:
        ok("All models present")
    return missing


# ── start scripts ──────────────────────────────────────────────────────────

def write_start_scripts(hw):
    header("Writing launch scripts")
    py = python_exe()
    extra = "--cpu" if hw["device"] == "cpu" else ""

    if SYSTEM == "Windows":
        s = BASE_DIR / "START_COMFYUI.bat"
        s.write_text(
            f"@echo off\n"
            f"echo Starting ComfyUI...\n"
            f"cd /d \"{COMFYUI}\"\n"
            f"\"{py}\" main.py {extra} --listen 0.0.0.0 --port 8188\n"
            f"pause\n"
        )
        ok(f"Created: START_COMFYUI.bat")
    else:
        s = BASE_DIR / "START_COMFYUI.sh"
        s.write_text(
            f"#!/bin/bash\nset -e\n"
            f"echo 'Starting ComfyUI...'\n"
            f"cd \"{COMFYUI}\"\n"
            f"\"{py}\" main.py {extra} --listen 0.0.0.0 --port 8188\n"
        )
        s.chmod(0o755)
        ok(f"Created: START_COMFYUI.sh")

    info("Browser: http://localhost:8188")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    header(f"Comify Setup  ·  {SYSTEM} / {ARCH}")

    hw = detect_hardware()
    info(f"Compute: {hw['device'].upper()}")

    if venv_healthy():
        ok(f"Venv already healthy — skipping install  ({VENV_DIR.name})")
    else:
        create_venv()
        install_pytorch(hw)
        install_comfyui_requirements()
        install_custom_node_requirements()

    check_models()
    write_start_scripts(hw)

    header("Done!")
    if SYSTEM == "Windows":
        print("  Double-click: START_COMFYUI.bat\n")
    else:
        print("  Run:  ./START_COMFYUI.sh\n")
        print("  Then open: http://localhost:8188\n")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9+ required. Install from python.org")
    main()
