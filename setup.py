#!/usr/bin/env python3
"""
Comify - Complete Cross-Platform Setup Script
==============================================
Automatically sets up the entire Comify Virtual Try-On system:
- Creates virtual environment (venv)
- Detects platform (Windows/macOS/Linux) and GPU (NVIDIA/AMD/Apple Silicon/CPU)
- Installs PyTorch with correct backend
- Installs all dependencies
- Clones ComfyUI and required custom nodes
- Downloads AI models
- Configures the system

Usage:
    python setup.py                    # Full interactive setup
    python setup.py --quick            # Quick setup (skip optional components)
    python setup.py --no-models        # Skip model download
    python setup.py --no-comfyui       # Skip ComfyUI setup
    python setup.py --cpu              # Force CPU mode
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============== Configuration ==============
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
CUSTOM_NODES_CONFIG = "custom_nodes.json"
VENV_NAME = "venv"


# ============== Colors ==============
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def disable():
        """Disable colors for Windows without ANSI support"""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.END = ''
        Colors.BOLD = ''


# Check Windows ANSI support
if platform.system() == 'Windows':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(text: str):
    print(f"{Colors.CYAN}>>> {text}{Colors.END}")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.WARNING}[!] {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.FAIL}[X] {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}[i] {text}{Colors.END}")


# ============== Platform Detection ==============
class PlatformInfo:
    """Stores detected platform information"""
    def __init__(self):
        self.system = platform.system()  # Windows, Darwin, Linux
        self.arch = platform.machine()   # x86_64, arm64, AMD64
        self.is_windows = self.system == 'Windows'
        self.is_mac = self.system == 'Darwin'
        self.is_linux = self.system == 'Linux'
        self.is_arm = self.arch in ['arm64', 'aarch64']
        self.is_x64 = self.arch in ['x86_64', 'AMD64']
        self.gpu_type = 'cpu'
        self.gpu_name = 'CPU'
        self.vram_gb = 0
        self.ram_gb = self._get_ram_gb()

    def _get_ram_gb(self) -> float:
        """Get system RAM in GB"""
        try:
            if self.is_mac:
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                       capture_output=True, text=True)
                return int(result.stdout.strip()) / (1024**3)
            elif self.is_linux:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            return int(line.split()[1]) / (1024**2)
            elif self.is_windows:
                result = subprocess.run(['wmic', 'OS', 'get', 'TotalVisibleMemorySize'],
                                       capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return int(lines[1].strip()) / (1024**2)
        except:
            pass
        return 16.0  # Default

    def detect_gpu(self):
        """Detect GPU type and capabilities"""
        # Check for NVIDIA GPU
        if self._detect_nvidia():
            return

        # Check for Apple Silicon
        if self._detect_apple_silicon():
            return

        # Check for AMD GPU
        if self._detect_amd():
            return

        # Default to CPU
        self.gpu_type = 'cpu'
        self.gpu_name = 'CPU (No GPU acceleration)'

    def _detect_nvidia(self) -> bool:
        """Detect NVIDIA GPU"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                self.gpu_name = parts[0].strip()
                self.vram_gb = float(parts[1].strip()) / 1024
                self.gpu_type = 'cuda'
                return True
        except:
            pass
        return False

    def _detect_apple_silicon(self) -> bool:
        """Detect Apple Silicon"""
        if self.is_mac and self.is_arm:
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                       capture_output=True, text=True)
                self.gpu_name = result.stdout.strip()
                self.vram_gb = self.ram_gb * 0.7  # Unified memory
                self.gpu_type = 'mps'
                return True
            except:
                pass
        return False

    def _detect_amd(self) -> bool:
        """Detect AMD GPU"""
        if self.is_windows:
            try:
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True, text=True, timeout=10
                )
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    self.gpu_type = 'directml'
                    self.gpu_name = 'AMD GPU (DirectML)'
                    self.vram_gb = 8  # Estimate
                    return True
            except:
                pass
        elif self.is_linux and Path('/opt/rocm').exists():
            self.gpu_type = 'rocm'
            self.gpu_name = 'AMD GPU (ROCm)'
            self.vram_gb = 8  # Estimate
            return True
        return False

    def __str__(self):
        return f"""Platform Information:
  System: {self.system} ({self.arch})
  GPU: {self.gpu_name}
  GPU Type: {self.gpu_type.upper()}
  VRAM: {self.vram_gb:.1f} GB
  RAM: {self.ram_gb:.1f} GB"""


# ============== Virtual Environment ==============
class VenvManager:
    """Manages Python virtual environment"""

    def __init__(self, base_dir: Path, venv_name: str = "venv"):
        self.base_dir = base_dir
        self.venv_path = base_dir / venv_name
        self.platform_info = PlatformInfo()

    @property
    def python_path(self) -> Path:
        """Get Python executable path in venv"""
        if self.platform_info.is_windows:
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    @property
    def pip_path(self) -> Path:
        """Get pip executable path in venv"""
        if self.platform_info.is_windows:
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    @property
    def activate_script(self) -> str:
        """Get activation script command"""
        if self.platform_info.is_windows:
            return f"{self.venv_path}\\Scripts\\activate"
        return f"source {self.venv_path}/bin/activate"

    def exists(self) -> bool:
        """Check if venv exists"""
        return self.python_path.exists()

    def create(self, force: bool = False) -> bool:
        """Create virtual environment"""
        if self.exists():
            if force:
                print_warning("Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            else:
                print_success(f"Virtual environment already exists: {self.venv_path}")
                return True

        print_step(f"Creating virtual environment: {self.venv_path}")
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], check=True)
            print_success("Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to create virtual environment: {e}")
            return False

    def run_pip(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run pip command in venv"""
        cmd = [str(self.pip_path)] + args
        return subprocess.run(cmd, check=check)

    def run_python(self, args: List[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
        """Run Python command in venv"""
        cmd = [str(self.python_path)] + args
        return subprocess.run(cmd, check=check, **kwargs)

    def upgrade_pip(self):
        """Upgrade pip to latest version"""
        print_step("Upgrading pip...")
        self.run_pip(['install', '--upgrade', 'pip'])


# ============== PyTorch Installation ==============
def get_pytorch_command(platform_info: PlatformInfo) -> List[str]:
    """Get PyTorch installation command based on platform"""
    gpu = platform_info.gpu_type

    if gpu == 'cuda':
        return ['torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu121']
    elif gpu == 'mps':
        return ['torch', 'torchvision', 'torchaudio']
    elif gpu == 'directml':
        return ['torch', 'torchvision']  # torch-directml installed separately
    elif gpu == 'rocm':
        return ['torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/rocm5.6']
    else:  # CPU
        return ['torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu']


def install_pytorch(venv: VenvManager, platform_info: PlatformInfo) -> bool:
    """Install PyTorch with correct backend"""
    print_step("Installing PyTorch...")

    gpu_names = {
        'cuda': 'NVIDIA CUDA',
        'mps': 'Apple Silicon MPS',
        'directml': 'AMD DirectML',
        'rocm': 'AMD ROCm',
        'cpu': 'CPU only'
    }

    print_info(f"Installing for: {gpu_names.get(platform_info.gpu_type, 'Unknown')}")

    try:
        packages = get_pytorch_command(platform_info)
        venv.run_pip(['install'] + packages)

        # Install torch-directml for AMD on Windows
        if platform_info.gpu_type == 'directml':
            print_step("Installing torch-directml...")
            venv.run_pip(['install', 'torch-directml'])

        print_success("PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"PyTorch installation failed: {e}")
        return False


# ============== Requirements Installation ==============
def get_requirements_file(platform_info: PlatformInfo) -> str:
    """Get platform-specific requirements file"""
    if platform_info.is_windows:
        return 'requirements-windows.txt'
    elif platform_info.is_mac:
        return 'requirements-mac.txt'
    elif platform_info.is_linux:
        return 'requirements-linux.txt'
    return 'requirements.txt'


def get_onnxruntime_package(platform_info: PlatformInfo) -> str:
    """Get correct onnxruntime package"""
    if platform_info.gpu_type == 'cuda':
        return 'onnxruntime-gpu>=1.16.0'
    elif platform_info.gpu_type == 'directml':
        return 'onnxruntime-directml>=1.16.0'
    return 'onnxruntime>=1.16.0'


def install_requirements(venv: VenvManager, base_dir: Path, platform_info: PlatformInfo) -> bool:
    """Install all requirements"""
    print_step("Installing base requirements...")

    try:
        # Install base requirements
        base_req = base_dir / 'requirements-base.txt'
        if base_req.exists():
            venv.run_pip(['install', '-r', str(base_req)])
            print_success("Base requirements installed")

        # Install platform-specific requirements
        print_step("Installing platform-specific requirements...")
        platform_req = base_dir / get_requirements_file(platform_info)
        if platform_req.exists():
            venv.run_pip(['install', '-r', str(platform_req)])
            print_success(f"Platform requirements installed ({platform_req.name})")
        else:
            # Fallback to main requirements
            main_req = base_dir / 'requirements.txt'
            if main_req.exists():
                venv.run_pip(['install', '-r', str(main_req)])

        # Install correct onnxruntime
        print_step("Installing ONNX Runtime...")
        onnx_pkg = get_onnxruntime_package(platform_info)
        venv.run_pip(['install', onnx_pkg], check=False)
        print_success(f"ONNX Runtime installed")

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Requirements installation failed: {e}")
        return False


# ============== ComfyUI Setup ==============
def clone_comfyui(base_dir: Path) -> bool:
    """Clone ComfyUI repository"""
    comfyui_dir = base_dir / 'ComfyUI'

    if comfyui_dir.exists():
        print_success("ComfyUI already exists")
        # Update ComfyUI
        print_step("Updating ComfyUI...")
        try:
            subprocess.run(['git', 'pull'], cwd=str(comfyui_dir), check=True,
                          capture_output=True)
            print_success("ComfyUI updated")
        except:
            print_warning("Could not update ComfyUI")
        return True

    print_step("Cloning ComfyUI...")
    try:
        subprocess.run(['git', 'clone', COMFYUI_REPO, str(comfyui_dir)], check=True)
        print_success("ComfyUI cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to clone ComfyUI: {e}")
        return False


def install_custom_nodes(base_dir: Path, required_only: bool = False) -> bool:
    """Install custom nodes from config"""
    config_path = base_dir / CUSTOM_NODES_CONFIG
    custom_nodes_dir = base_dir / 'ComfyUI' / 'custom_nodes'

    if not config_path.exists():
        print_warning("Custom nodes config not found")
        return True

    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    nodes = config.get('nodes', [])
    print_step(f"Installing {len(nodes)} custom nodes...")

    success = True
    for node in nodes:
        name = node['name']
        url = node['url']
        required = node.get('required', False)

        if required_only and not required:
            continue

        node_dir = custom_nodes_dir / name

        if node_dir.exists():
            print_info(f"  {name} - already exists")
            continue

        print_info(f"  Cloning {name}...")
        try:
            subprocess.run(['git', 'clone', url, str(node_dir)],
                          check=True, capture_output=True)
            print_success(f"  {name} installed")
        except subprocess.CalledProcessError:
            if required:
                print_error(f"  Failed to install required node: {name}")
                success = False
            else:
                print_warning(f"  Could not install optional node: {name}")

    return success


def install_comfyui_requirements(venv: VenvManager, base_dir: Path) -> bool:
    """Install ComfyUI requirements"""
    comfyui_req = base_dir / 'ComfyUI' / 'requirements.txt'

    if not comfyui_req.exists():
        return True

    print_step("Installing ComfyUI requirements...")
    try:
        venv.run_pip(['install', '-r', str(comfyui_req)])
        print_success("ComfyUI requirements installed")
        return True
    except subprocess.CalledProcessError:
        print_warning("Some ComfyUI requirements could not be installed")
        return True


# ============== Model Download ==============
def download_models(venv: VenvManager, base_dir: Path, platform_info: PlatformInfo) -> bool:
    """Download AI models using model_downloader"""
    downloader_path = base_dir / 'installer' / 'model_downloader.py'

    if not downloader_path.exists():
        print_warning("Model downloader not found")
        return True

    # Determine profile based on hardware
    if platform_info.gpu_type in ['cuda', 'mps'] and platform_info.vram_gb >= 12:
        profile = 'high'
    elif platform_info.gpu_type in ['cuda', 'mps', 'directml'] and platform_info.vram_gb >= 8:
        profile = 'medium'
    elif platform_info.gpu_type != 'cpu':
        profile = 'low'
    else:
        profile = 'cpu_optimized'

    print_step(f"Downloading AI models (profile: {profile})...")
    print_info("This may take a while depending on your internet connection")

    try:
        venv.run_python([str(downloader_path), '--profile', profile, '--tryon'])
        print_success("Models downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print_warning("Some models could not be downloaded")
        print_info("You can run model download manually later:")
        print_info(f"  python installer/model_downloader.py --profile {profile}")
        return True


# ============== Frontend Setup ==============
def install_frontend(base_dir: Path, platform_info: PlatformInfo) -> bool:
    """Install frontend (Next.js) dependencies"""
    frontend_dir = base_dir / 'frontend'

    if not frontend_dir.exists():
        print_warning("Frontend directory not found")
        return True

    print_step("Installing frontend dependencies...")

    # Check for npm
    npm_cmd = 'npm.cmd' if platform_info.is_windows else 'npm'

    try:
        subprocess.run([npm_cmd, '--version'], capture_output=True, check=True)
    except:
        print_warning("npm not found - please install Node.js to use the frontend")
        print_info("Download from: https://nodejs.org/")
        return True

    try:
        subprocess.run([npm_cmd, 'install'], cwd=str(frontend_dir), check=True)
        print_success("Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print_warning("Frontend installation had issues")
        return True


# ============== Configuration ==============
def save_configuration(base_dir: Path, platform_info: PlatformInfo):
    """Save hardware and AI configuration"""
    print_step("Saving configuration...")

    config_dir = base_dir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save hardware config
    hardware_config = {
        'system': platform_info.system,
        'arch': platform_info.arch,
        'device_type': platform_info.gpu_type,
        'device_name': platform_info.gpu_name,
        'vram_gb': platform_info.vram_gb,
        'ram_gb': platform_info.ram_gb,
        'compute_backend': platform_info.gpu_type,
    }

    with open(config_dir / 'hardware.json', 'w') as f:
        json.dump(hardware_config, f, indent=2)

    # Determine model profile
    if platform_info.vram_gb >= 12:
        model_profile = 'high'
    elif platform_info.vram_gb >= 8:
        model_profile = 'medium'
    elif platform_info.gpu_type != 'cpu':
        model_profile = 'low'
    else:
        model_profile = 'cpu_optimized'

    # Save AI config
    ai_config = {
        'device': platform_info.gpu_type,
        'compute_backend': platform_info.gpu_type,
        'batch_size': 2 if model_profile == 'high' else 1,
        'num_inference_steps': 20 if model_profile == 'high' else 15,
        'use_fp16': platform_info.gpu_type in ['cuda', 'mps'],
        'model_profile': model_profile,
    }

    with open(config_dir / 'ai_config.json', 'w') as f:
        json.dump(ai_config, f, indent=2)

    print_success("Configuration saved")


def create_start_scripts(base_dir: Path, venv: VenvManager, platform_info: PlatformInfo):
    """Create start/stop scripts"""
    print_step("Creating start scripts...")

    if platform_info.is_windows:
        # Windows batch script
        start_script = f"""@echo off
echo Starting Comify Virtual Try-On...
echo.

REM Activate virtual environment
call "{venv.venv_path}\\Scripts\\activate.bat"

REM Start ComfyUI
start "ComfyUI" cmd /k "cd /d {base_dir}\\ComfyUI && python main.py"

REM Wait for ComfyUI
timeout /t 10 /nobreak

REM Start Backend
start "Backend" cmd /k "cd /d {base_dir}\\backend && python main.py"

REM Wait for Backend
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
        with open(base_dir / 'start.bat', 'w') as f:
            f.write(start_script)

        # Stop script
        stop_script = """@echo off
echo Stopping Comify...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
echo Done.
pause
"""
        with open(base_dir / 'stop.bat', 'w') as f:
            f.write(stop_script)

    else:
        # Unix shell script
        start_script = f"""#!/bin/bash
echo "Starting Comify Virtual Try-On..."
echo

# Activate virtual environment
source "{venv.venv_path}/bin/activate"

# Start ComfyUI
cd "{base_dir}/ComfyUI"
python main.py &
COMFYUI_PID=$!
echo "ComfyUI starting (PID: $COMFYUI_PID)"

sleep 10

# Start Backend
cd "{base_dir}/backend"
python main.py &
BACKEND_PID=$!
echo "Backend starting (PID: $BACKEND_PID)"

sleep 5

# Start Frontend
cd "{base_dir}/frontend"
npm run dev &
FRONTEND_PID=$!
echo "Frontend starting (PID: $FRONTEND_PID)"

echo
echo "Comify is running!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "ComfyUI:  http://localhost:8188"
echo
echo "Press Ctrl+C to stop all services"

trap "kill $COMFYUI_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
"""
        start_path = base_dir / 'start.sh'
        with open(start_path, 'w') as f:
            f.write(start_script)
        os.chmod(start_path, 0o755)

        # Stop script
        stop_script = """#!/bin/bash
echo "Stopping Comify..."
pkill -f "python.*main.py" 2>/dev/null
pkill -f "next dev" 2>/dev/null
echo "Done."
"""
        stop_path = base_dir / 'stop.sh'
        with open(stop_path, 'w') as f:
            f.write(stop_script)
        os.chmod(stop_path, 0o755)

    print_success("Start scripts created")


# ============== Main Setup ==============
def main():
    parser = argparse.ArgumentParser(description='Comify Setup Script')
    parser.add_argument('--quick', action='store_true', help='Quick setup (required nodes only)')
    parser.add_argument('--no-models', action='store_true', help='Skip model download')
    parser.add_argument('--no-comfyui', action='store_true', help='Skip ComfyUI setup')
    parser.add_argument('--no-frontend', action='store_true', help='Skip frontend setup')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--force-venv', action='store_true', help='Recreate virtual environment')
    args = parser.parse_args()

    print_header("COMIFY VIRTUAL TRY-ON SETUP")

    base_dir = Path(__file__).parent.absolute()
    print(f"Installation directory: {base_dir}")

    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print_error(f"Python 3.9+ required. Found: {py_version.major}.{py_version.minor}")
        sys.exit(1)
    print_success(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Detect platform and GPU
    platform_info = PlatformInfo()
    platform_info.detect_gpu()

    if args.cpu:
        platform_info.gpu_type = 'cpu'
        platform_info.gpu_name = 'CPU (forced)'
        platform_info.vram_gb = 0

    print(f"\n{platform_info}\n")

    # Confirm installation
    response = input("Proceed with setup? (Y/n): ").strip().lower()
    if response == 'n':
        print("Setup cancelled.")
        sys.exit(0)

    try:
        # Create virtual environment
        venv = VenvManager(base_dir, VENV_NAME)
        if not venv.create(force=args.force_venv):
            sys.exit(1)

        # Upgrade pip
        venv.upgrade_pip()

        # Install PyTorch
        if not install_pytorch(venv, platform_info):
            print_error("PyTorch installation failed")
            sys.exit(1)

        # Install requirements
        if not install_requirements(venv, base_dir, platform_info):
            print_warning("Some requirements could not be installed")

        # Setup ComfyUI
        if not args.no_comfyui:
            if not clone_comfyui(base_dir):
                print_warning("ComfyUI setup incomplete")
            else:
                install_comfyui_requirements(venv, base_dir)
                install_custom_nodes(base_dir, required_only=args.quick)

        # Download models
        if not args.no_models:
            response = input("\nDownload AI models now? This may take a while. (Y/n): ").strip().lower()
            if response != 'n':
                download_models(venv, base_dir, platform_info)

        # Install frontend
        if not args.no_frontend:
            install_frontend(base_dir, platform_info)

        # Save configuration
        save_configuration(base_dir, platform_info)

        # Create start scripts
        create_start_scripts(base_dir, venv, platform_info)

        # Done!
        print_header("SETUP COMPLETE")
        print_success("Comify has been set up successfully!")
        print(f"""
Next steps:
  1. Activate the virtual environment:
     {venv.activate_script}

  2. Start Comify:
     {"start.bat" if platform_info.is_windows else "./start.sh"}

  3. Open in browser:
     http://localhost:3000

For more information, see INSTALLATION.md
""")

    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
