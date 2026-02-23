#!/usr/bin/env python3
"""
Comify Cross-Platform Setup Script
Automatically detects your platform and installs the correct dependencies.
Works on Windows, macOS (Apple Silicon & Intel), and Linux.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(text):
    print(f"{Colors.BLUE}>>> {text}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.WARNING}[!] {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.FAIL}[X] {text}{Colors.END}")


def get_platform_info():
    """Detect platform and architecture"""
    system = platform.system()
    arch = platform.machine()

    info = {
        'system': system,
        'arch': arch,
        'is_windows': system == 'Windows',
        'is_mac': system == 'Darwin',
        'is_linux': system == 'Linux',
        'is_arm': arch in ['arm64', 'aarch64'],
        'is_x64': arch in ['x86_64', 'AMD64'],
    }

    # Determine GPU type
    info['gpu_type'] = detect_gpu_type(system)

    return info


def detect_gpu_type(system):
    """Detect GPU type (NVIDIA, AMD, Apple Silicon, or CPU)"""

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return 'nvidia'
    except:
        pass

    # Check for Apple Silicon
    if system == 'Darwin':
        arch = platform.machine()
        if arch == 'arm64':
            return 'apple_silicon'
        return 'intel_mac'

    # Check for AMD GPU on Windows
    if system == 'Windows':
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, text=True, timeout=10
            )
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                return 'amd'
        except:
            pass

    # Check for AMD GPU on Linux (ROCm)
    if system == 'Linux':
        if Path('/opt/rocm').exists():
            return 'amd_rocm'

    return 'cpu'


def get_pytorch_install_command(platform_info):
    """Get the correct PyTorch installation command"""
    gpu = platform_info['gpu_type']
    system = platform_info['system']

    if gpu == 'nvidia':
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ]
    elif gpu == 'apple_silicon':
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio'
        ]
    elif gpu == 'intel_mac':
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio'
        ]
    elif gpu == 'amd' and system == 'Windows':
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision'
        ]
    elif gpu == 'amd_rocm':
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/rocm5.6'
        ]
    else:  # CPU
        return [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]


def get_onnxruntime_package(platform_info):
    """Get the correct onnxruntime package name"""
    gpu = platform_info['gpu_type']
    system = platform_info['system']

    if gpu == 'nvidia':
        return 'onnxruntime-gpu>=1.16.0'
    elif gpu == 'amd' and system == 'Windows':
        return 'onnxruntime-directml>=1.16.0'
    else:
        return 'onnxruntime>=1.16.0'


def get_requirements_file(platform_info):
    """Get the platform-specific requirements file"""
    if platform_info['is_windows']:
        return 'requirements-windows.txt'
    elif platform_info['is_mac']:
        return 'requirements-mac.txt'
    elif platform_info['is_linux']:
        return 'requirements-linux.txt'
    return 'requirements.txt'


def check_python_version():
    """Check if Python version is compatible"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print_error(f"Python 3.9+ required. Found: {major}.{minor}")
        return False
    print_success(f"Python version: {major}.{minor}")
    return True


def install_pytorch(platform_info):
    """Install PyTorch with appropriate backend"""
    print_step("Installing PyTorch...")

    gpu_names = {
        'nvidia': 'NVIDIA CUDA',
        'apple_silicon': 'Apple Silicon MPS',
        'intel_mac': 'Intel Mac (CPU)',
        'amd': 'AMD DirectML',
        'amd_rocm': 'AMD ROCm',
        'cpu': 'CPU only'
    }

    gpu = platform_info['gpu_type']
    print(f"  Detected: {gpu_names.get(gpu, 'Unknown')}")

    cmd = get_pytorch_install_command(platform_info)
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print_warning("PyTorch installation had issues. You may need to install manually.")
        return False

    # Install torch-directml for AMD on Windows
    if gpu == 'amd' and platform_info['is_windows']:
        print_step("Installing torch-directml for AMD GPU...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch-directml'])

    print_success("PyTorch installed")
    return True


def install_requirements(platform_info, base_dir):
    """Install all requirements"""
    print_step("Installing base requirements...")

    # Install base requirements first
    base_req = base_dir / 'requirements-base.txt'
    if base_req.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(base_req)], check=True)
        print_success("Base requirements installed")

    # Install platform-specific requirements
    print_step("Installing platform-specific requirements...")
    platform_req = base_dir / get_requirements_file(platform_info)
    if platform_req.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(platform_req)], check=True)
        print_success(f"Platform requirements installed ({platform_req.name})")

    # Install correct onnxruntime
    print_step("Installing ONNX Runtime...")
    onnx_pkg = get_onnxruntime_package(platform_info)
    subprocess.run([sys.executable, '-m', 'pip', 'install', onnx_pkg], check=True)
    print_success(f"Installed: {onnx_pkg}")


def install_frontend(base_dir):
    """Install frontend dependencies"""
    print_step("Installing frontend dependencies...")

    frontend_dir = base_dir / 'frontend'
    if not frontend_dir.exists():
        print_warning("Frontend directory not found")
        return

    # Check for npm
    npm_cmd = 'npm.cmd' if platform.system() == 'Windows' else 'npm'

    try:
        subprocess.run([npm_cmd, '--version'], capture_output=True, check=True)
    except:
        print_warning("npm not found - please install Node.js to use the frontend")
        return

    subprocess.run([npm_cmd, 'install'], cwd=str(frontend_dir), check=True)
    print_success("Frontend dependencies installed")


def main():
    """Main setup function"""
    print_header("COMIFY CROSS-PLATFORM SETUP")

    base_dir = Path(__file__).parent.absolute()
    print(f"Install directory: {base_dir}")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Detect platform
    platform_info = get_platform_info()
    print(f"\nPlatform: {platform_info['system']} ({platform_info['arch']})")
    print(f"GPU Type: {platform_info['gpu_type']}")

    # Confirm installation
    response = input("\nProceed with installation? (Y/n): ").strip().lower()
    if response == 'n':
        print("Installation cancelled.")
        sys.exit(0)

    try:
        # Upgrade pip first
        print_step("Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)

        # Install PyTorch
        install_pytorch(platform_info)

        # Install requirements
        install_requirements(platform_info, base_dir)

        # Install frontend
        install_frontend(base_dir)

        print_header("SETUP COMPLETE")
        print_success("Comify has been set up successfully!")
        print(f"\nNext steps:")
        print(f"  1. Run: python install.py  (to download AI models)")
        print(f"  2. Start backend: cd backend && python main.py")
        print(f"  3. Start frontend: cd frontend && npm run dev")
        print(f"  4. Open: http://localhost:3000")

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
