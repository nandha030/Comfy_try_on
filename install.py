#!/usr/bin/env python3
"""
Comify Installer
Cross-platform installer with automatic hardware detection
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, Tuple

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(text: str):
    print(f"{Colors.CYAN}>>> {text}{Colors.END}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")


def get_base_dir() -> Path:
    """Get the base directory for installation"""
    return Path(__file__).parent.absolute()


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ required. Found: {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def detect_hardware() -> dict:
    """Detect hardware configuration"""
    print_step("Detecting hardware...")

    system = platform.system()
    arch = platform.machine()

    result = {
        "system": system,
        "arch": arch,
        "device_type": "cpu",
        "device_name": "CPU",
        "vram_gb": 0,
        "compute_backend": "onnx",
        "model_profile": "cpu_optimized",
        "torch_index_url": "https://download.pytorch.org/whl/cpu"
    }

    # Check for NVIDIA GPU
    if system in ["Windows", "Linux"]:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if output.returncode == 0:
                parts = output.stdout.strip().split(',')
                result["device_name"] = parts[0].strip()
                result["vram_gb"] = float(parts[1].strip()) / 1024
                result["device_type"] = "cuda"
                result["compute_backend"] = "cuda"

                if result["vram_gb"] >= 12:
                    result["model_profile"] = "high"
                    result["torch_index_url"] = "https://download.pytorch.org/whl/cu121"
                elif result["vram_gb"] >= 8:
                    result["model_profile"] = "medium"
                    result["torch_index_url"] = "https://download.pytorch.org/whl/cu121"
                else:
                    result["model_profile"] = "low"
                    result["torch_index_url"] = "https://download.pytorch.org/whl/cu121"

                print_success(f"NVIDIA GPU: {result['device_name']} ({result['vram_gb']:.1f}GB)")
                return result
        except:
            pass

    # Check for Apple Silicon
    if system == "Darwin" and arch == "arm64":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            result["device_name"] = output.stdout.strip()
            result["device_type"] = "mps"
            result["compute_backend"] = "mps"

            # Get RAM for unified memory estimate
            ram_output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            ram_gb = int(ram_output.stdout.strip()) / (1024**3)
            result["vram_gb"] = ram_gb * 0.7  # ~70% available for GPU

            if result["vram_gb"] >= 24:
                result["model_profile"] = "high"
            elif result["vram_gb"] >= 12:
                result["model_profile"] = "medium"
            else:
                result["model_profile"] = "low"

            result["torch_index_url"] = "https://download.pytorch.org/whl/cpu"  # MPS uses default

            print_success(f"Apple Silicon: {result['device_name']}")
            print_success(f"Unified Memory: {ram_gb:.1f}GB (~{result['vram_gb']:.1f}GB for GPU)")
            return result
        except:
            pass

    # Check for AMD GPU on Windows
    if system == "Windows":
        try:
            output = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True
            )
            if "AMD" in output.stdout or "Radeon" in output.stdout:
                result["device_type"] = "directml"
                result["compute_backend"] = "directml"
                result["model_profile"] = "medium"
                result["device_name"] = "AMD GPU"
                print_success("AMD GPU detected (DirectML)")
                return result
        except:
            pass

    # Intel Mac
    if system == "Darwin" and arch == "x86_64":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            result["device_name"] = output.stdout.strip()
            print_warning(f"Intel Mac: {result['device_name']}")
            print_warning("No GPU acceleration - will use CPU (slower)")
            return result
        except:
            pass

    # Default CPU
    print_warning("No GPU detected - will use CPU mode")
    return result


def create_virtual_environment(base_dir: Path) -> Path:
    """Create Python virtual environment"""
    print_step("Creating virtual environment...")

    venv_path = base_dir / "comify-env"

    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Recreate? (y/N): ").strip().lower()
        if response == 'y':
            shutil.rmtree(venv_path)
        else:
            return venv_path

    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print_success(f"Created virtual environment: {venv_path}")

    return venv_path


def get_pip_path(venv_path: Path) -> Path:
    """Get pip executable path"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_python_path(venv_path: Path) -> Path:
    """Get Python executable path"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_pytorch(pip_path: Path, hardware: dict):
    """Install PyTorch with appropriate backend"""
    print_step("Installing PyTorch...")

    backend = hardware["compute_backend"]
    index_url = hardware["torch_index_url"]

    if backend == "cuda":
        print(f"Installing PyTorch with CUDA support...")
        subprocess.run([
            str(pip_path), "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ], check=True)
    elif backend == "mps":
        print(f"Installing PyTorch with MPS (Apple Silicon) support...")
        subprocess.run([
            str(pip_path), "install",
            "torch", "torchvision", "torchaudio"
        ], check=True)
    elif backend == "directml":
        print(f"Installing PyTorch with DirectML (AMD) support...")
        subprocess.run([
            str(pip_path), "install",
            "torch", "torchvision",
            "--index-url", index_url
        ], check=True)
        subprocess.run([
            str(pip_path), "install", "torch-directml"
        ], check=True)
    else:
        print(f"Installing PyTorch (CPU only)...")
        subprocess.run([
            str(pip_path), "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ], check=True)

    print_success("PyTorch installed")


def install_dependencies(pip_path: Path, base_dir: Path):
    """Install all required dependencies"""
    print_step("Installing dependencies...")

    # Core dependencies
    core_deps = [
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "python-multipart>=0.0.6",
        "aiohttp>=3.9.0",
        "websockets>=12.0",
        "pydantic>=2.5.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    ]

    # AI dependencies
    ai_deps = [
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "onnxruntime>=1.16.0",
        "insightface>=0.7.3",
        "segment-anything",
        "rembg>=2.0.50",
    ]

    # Install core dependencies
    print("Installing core dependencies...")
    subprocess.run([str(pip_path), "install"] + core_deps, check=True)

    # Install AI dependencies
    print("Installing AI dependencies...")
    for dep in ai_deps:
        try:
            subprocess.run([str(pip_path), "install", dep], check=True)
        except subprocess.CalledProcessError:
            print_warning(f"Could not install {dep} - some features may be limited")

    # Install from requirements.txt if exists
    requirements_file = base_dir / "backend" / "requirements.txt"
    if requirements_file.exists():
        print("Installing from requirements.txt...")
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)

    print_success("Dependencies installed")


def install_frontend_dependencies(base_dir: Path):
    """Install frontend (Next.js) dependencies"""
    print_step("Installing frontend dependencies...")

    frontend_dir = base_dir / "frontend"
    if not frontend_dir.exists():
        print_warning("Frontend directory not found")
        return

    # Check for npm
    npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"

    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except:
        print_warning("npm not found - please install Node.js to use the frontend")
        return

    # Install dependencies
    subprocess.run([npm_cmd, "install"], cwd=str(frontend_dir), check=True)
    print_success("Frontend dependencies installed")


def save_configuration(base_dir: Path, hardware: dict):
    """Save hardware configuration"""
    print_step("Saving configuration...")

    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save hardware config
    with open(config_dir / "hardware.json", 'w') as f:
        json.dump(hardware, f, indent=2)

    # Create AI config
    ai_config = {
        "device": hardware["device_type"],
        "compute_backend": hardware["compute_backend"],
        "batch_size": 1 if hardware["model_profile"] in ["low", "cpu_optimized"] else 2,
        "num_inference_steps": 20 if hardware["model_profile"] == "high" else 15,
        "use_fp16": hardware["device_type"] in ["cuda", "mps"],
        "enable_face_preservation": True,
        "enable_skin_tone_matching": True,
        "enable_upscaling": hardware["model_profile"] in ["high", "medium"],
        "upscale_factor": 2
    }

    with open(config_dir / "ai_config.json", 'w') as f:
        json.dump(ai_config, f, indent=2)

    print_success("Configuration saved")


def create_start_scripts(base_dir: Path, venv_path: Path):
    """Create start scripts for easy launching"""
    print_step("Creating start scripts...")

    python_path = get_python_path(venv_path)

    if platform.system() == "Windows":
        # Windows batch script
        script = f"""@echo off
echo Starting Comify...
echo.

REM Start ComfyUI
start "ComfyUI" cmd /k "cd /d {base_dir}\\ComfyUI && {python_path} main.py --cpu"

REM Wait for ComfyUI to start
timeout /t 10 /nobreak

REM Start Backend
start "Backend" cmd /k "cd /d {base_dir}\\backend && {python_path} main.py"

REM Wait for Backend to start
timeout /t 5 /nobreak

REM Start Frontend
start "Frontend" cmd /k "cd /d {base_dir}\\frontend && npm run dev"

echo.
echo Comify is starting...
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo ComfyUI:  http://localhost:8188
echo.
pause
"""
        with open(base_dir / "start.bat", 'w') as f:
            f.write(script)

        # Stop script
        stop_script = """@echo off
echo Stopping Comify...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
echo Done.
pause
"""
        with open(base_dir / "stop.bat", 'w') as f:
            f.write(stop_script)

    else:
        # Unix shell script
        script = f"""#!/bin/bash
echo "Starting Comify..."
echo

# Start ComfyUI
cd "{base_dir}/ComfyUI"
"{python_path}" main.py --cpu &
COMFYUI_PID=$!
echo "ComfyUI starting (PID: $COMFYUI_PID)"

# Wait for ComfyUI
sleep 10

# Start Backend
cd "{base_dir}/backend"
"{python_path}" main.py &
BACKEND_PID=$!
echo "Backend starting (PID: $BACKEND_PID)"

# Wait for Backend
sleep 5

# Start Frontend
cd "{base_dir}/frontend"
npm run dev &
FRONTEND_PID=$!
echo "Frontend starting (PID: $FRONTEND_PID)"

echo
echo "Comify is starting..."
echo
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "ComfyUI:  http://localhost:8188"
echo
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $COMFYUI_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
"""
        start_path = base_dir / "start.sh"
        with open(start_path, 'w') as f:
            f.write(script)
        os.chmod(start_path, 0o755)

        # Stop script
        stop_script = """#!/bin/bash
echo "Stopping Comify..."
pkill -f "python.*main.py" 2>/dev/null
pkill -f "next dev" 2>/dev/null
echo "Done."
"""
        stop_path = base_dir / "stop.sh"
        with open(stop_path, 'w') as f:
            f.write(stop_script)
        os.chmod(stop_path, 0o755)

    print_success("Start scripts created")


def download_models_prompt(base_dir: Path, hardware: dict, pip_path: Path):
    """Prompt user to download AI models"""
    print_step("AI Models Setup")

    total_size = {
        "high": "~15GB",
        "medium": "~10GB",
        "low": "~6GB",
        "cpu_optimized": "~4GB"
    }

    profile = hardware["model_profile"]

    print(f"\nRecommended model profile: {profile}")
    print(f"Estimated download size: {total_size.get(profile, '~10GB')}")
    print("\nModels will be downloaded to: models/")

    response = input("\nDownload models now? (Y/n): ").strip().lower()

    if response != 'n':
        print_step("Downloading models...")
        python_path = get_python_path(Path(pip_path).parent.parent)
        downloader_path = base_dir / "installer" / "model_downloader.py"

        if downloader_path.exists():
            subprocess.run([
                str(python_path),
                str(downloader_path),
                "--profile", profile
            ])
        else:
            print_warning("Model downloader not found - please download models manually")
    else:
        print_warning("Skipping model download - you can run this later:")
        print(f"  python installer/model_downloader.py --profile {profile}")


def main():
    """Main installation function"""
    print_header("COMIFY INSTALLER")
    print("Professional Virtual Try-On System")
    print(f"Platform: {platform.system()} ({platform.machine()})")

    base_dir = get_base_dir()
    print(f"Install directory: {base_dir}")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Detect hardware
    hardware = detect_hardware()

    print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Device: {hardware['device_name']}")
    print(f"  Backend: {hardware['compute_backend']}")
    print(f"  Profile: {hardware['model_profile']}")

    # Confirm installation
    response = input(f"\nProceed with installation? (Y/n): ").strip().lower()
    if response == 'n':
        print("Installation cancelled.")
        sys.exit(0)

    try:
        # Create virtual environment
        venv_path = create_virtual_environment(base_dir)
        pip_path = get_pip_path(venv_path)

        # Upgrade pip
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)

        # Install PyTorch
        install_pytorch(pip_path, hardware)

        # Install dependencies
        install_dependencies(pip_path, base_dir)

        # Install frontend
        install_frontend_dependencies(base_dir)

        # Save configuration
        save_configuration(base_dir, hardware)

        # Create start scripts
        create_start_scripts(base_dir, venv_path)

        # Download models
        download_models_prompt(base_dir, hardware, pip_path)

        # Done!
        print_header("INSTALLATION COMPLETE")
        print(f"{Colors.GREEN}Comify has been installed successfully!{Colors.END}")
        print(f"\nTo start Comify:")
        if platform.system() == "Windows":
            print(f"  Double-click start.bat")
            print(f"  Or run: .\\start.bat")
        else:
            print(f"  Run: ./start.sh")
        print(f"\nThen open: http://localhost:3000")

    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
