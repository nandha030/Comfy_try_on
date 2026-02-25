"""
Hardware Detection System for Comify
Automatically detects GPU/CPU and configures optimal settings
"""

import platform
import subprocess
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

@dataclass
class HardwareProfile:
    """Hardware configuration profile"""
    system: str  # Darwin, Windows, Linux
    arch: str  # arm64, x86_64
    device_type: str  # cuda, mps, directml, cpu
    device_name: str  # GPU name or CPU name
    vram_gb: float  # GPU VRAM in GB
    ram_gb: float  # System RAM in GB
    compute_backend: str  # cuda, mps, directml, onnx, cpu
    model_profile: str  # high, medium, low, cpu_optimized
    recommended_batch_size: int
    recommended_steps: int
    estimated_speed: str  # "~10 sec", "~5 min", etc.


def get_system_info() -> Tuple[str, str]:
    """Get OS and architecture"""
    system = platform.system()
    arch = platform.machine()
    return system, arch


def get_ram_gb() -> float:
    """Get total system RAM in GB"""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            return int(result.stdout.strip()) / (1024**3)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong)
                ]
            memoryStatus = MEMORYSTATUS()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
            return memoryStatus.dwTotalPhys / (1024**3)
        else:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) / (1024**2)
    except:
        pass
    return 8.0  # Default assumption


def detect_nvidia_gpu() -> Tuple[bool, str, float]:
    """Detect NVIDIA GPU and VRAM"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                vram_gb = vram_mb / 1024
                return True, name, vram_gb
    except:
        pass
    return False, "", 0.0


def detect_apple_silicon() -> Tuple[bool, str, float]:
    """Detect Apple Silicon and unified memory"""
    system, arch = get_system_info()
    if system == "Darwin" and arch == "arm64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            cpu_name = result.stdout.strip()
            ram_gb = get_ram_gb()
            # Apple Silicon uses unified memory, allocate ~70% for GPU
            gpu_memory = ram_gb * 0.7
            return True, cpu_name, gpu_memory
        except:
            pass
    return False, "", 0.0


def detect_amd_gpu() -> Tuple[bool, str, float]:
    """Detect AMD GPU (Windows DirectML)"""
    if platform.system() != "Windows":
        return False, "", 0.0

    try:
        # Use WMI to detect AMD GPU
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
            capture_output=True, text=True, timeout=10
        )
        if "AMD" in result.stdout or "Radeon" in result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:
                if "AMD" in line or "Radeon" in line:
                    parts = line.split()
                    name = " ".join([p for p in parts if not p.isdigit()])
                    # Try to extract VRAM
                    vram_bytes = [int(p) for p in parts if p.isdigit()]
                    vram_gb = vram_bytes[0] / (1024**3) if vram_bytes else 8.0
                    return True, name, vram_gb
    except:
        pass
    return False, "", 0.0


def detect_intel_mac() -> Tuple[bool, str]:
    """Detect Intel Mac"""
    system, arch = get_system_info()
    if system == "Darwin" and arch == "x86_64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return True, result.stdout.strip()
        except:
            pass
    return False, ""


def get_cpu_name() -> str:
    """Get CPU name for fallback"""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().split('\n')
            return lines[1].strip() if len(lines) > 1 else "Unknown CPU"
        else:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
    except:
        pass
    return "Unknown CPU"


def detect_hardware() -> HardwareProfile:
    """
    Main detection function - returns optimal hardware profile
    """
    system, arch = get_system_info()
    ram_gb = get_ram_gb()

    # Check for NVIDIA GPU first (best option)
    has_nvidia, nvidia_name, nvidia_vram = detect_nvidia_gpu()
    if has_nvidia:
        if nvidia_vram >= 12:
            return HardwareProfile(
                system=system,
                arch=arch,
                device_type="cuda",
                device_name=nvidia_name,
                vram_gb=nvidia_vram,
                ram_gb=ram_gb,
                compute_backend="cuda",
                model_profile="high",
                recommended_batch_size=4,
                recommended_steps=20,
                estimated_speed="~10-15 seconds"
            )
        elif nvidia_vram >= 8:
            return HardwareProfile(
                system=system,
                arch=arch,
                device_type="cuda",
                device_name=nvidia_name,
                vram_gb=nvidia_vram,
                ram_gb=ram_gb,
                compute_backend="cuda",
                model_profile="medium",
                recommended_batch_size=2,
                recommended_steps=15,
                estimated_speed="~20-30 seconds"
            )
        else:
            return HardwareProfile(
                system=system,
                arch=arch,
                device_type="cuda",
                device_name=nvidia_name,
                vram_gb=nvidia_vram,
                ram_gb=ram_gb,
                compute_backend="cuda",
                model_profile="low",
                recommended_batch_size=1,
                recommended_steps=12,
                estimated_speed="~30-45 seconds"
            )

    # Check for Apple Silicon (second best for Mac)
    has_apple, apple_name, apple_memory = detect_apple_silicon()
    if has_apple:
        if apple_memory >= 24:
            profile = "high"
            batch = 2
            steps = 20
            speed = "~15-20 seconds"
        elif apple_memory >= 12:
            profile = "medium"
            batch = 1
            steps = 15
            speed = "~25-35 seconds"
        else:
            profile = "low"
            batch = 1
            steps = 12
            speed = "~40-60 seconds"

        return HardwareProfile(
            system=system,
            arch=arch,
            device_type="mps",
            device_name=apple_name,
            vram_gb=apple_memory,
            ram_gb=ram_gb,
            compute_backend="mps",
            model_profile=profile,
            recommended_batch_size=batch,
            recommended_steps=steps,
            estimated_speed=speed
        )

    # Check for AMD GPU on Windows
    has_amd, amd_name, amd_vram = detect_amd_gpu()
    if has_amd:
        return HardwareProfile(
            system=system,
            arch=arch,
            device_type="directml",
            device_name=amd_name,
            vram_gb=amd_vram,
            ram_gb=ram_gb,
            compute_backend="directml",
            model_profile="medium",
            recommended_batch_size=1,
            recommended_steps=15,
            estimated_speed="~25-40 seconds"
        )

    # Check for Intel Mac
    has_intel_mac, intel_name = detect_intel_mac()
    if has_intel_mac:
        return HardwareProfile(
            system=system,
            arch=arch,
            device_type="cpu",
            device_name=intel_name,
            vram_gb=0,
            ram_gb=ram_gb,
            compute_backend="onnx",
            model_profile="cpu_optimized",
            recommended_batch_size=1,
            recommended_steps=8,
            estimated_speed="~3-5 minutes"
        )

    # Fallback to CPU
    cpu_name = get_cpu_name()
    return HardwareProfile(
        system=system,
        arch=arch,
        device_type="cpu",
        device_name=cpu_name,
        vram_gb=0,
        ram_gb=ram_gb,
        compute_backend="onnx",
        model_profile="cpu_optimized",
        recommended_batch_size=1,
        recommended_steps=8,
        estimated_speed="~5-10 minutes"
    )


def save_hardware_config(profile: HardwareProfile, config_path: Path):
    """Save hardware configuration to JSON file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(asdict(profile), f, indent=2)
    print(f"Configuration saved to: {config_path}")


def load_hardware_config(config_path: Path) -> Optional[HardwareProfile]:
    """Load hardware configuration from JSON file"""
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            return HardwareProfile(**data)
    return None


def print_hardware_info(profile: HardwareProfile):
    """Pretty print hardware information"""
    print("\n" + "="*60)
    print("           COMIFY HARDWARE DETECTION")
    print("="*60)
    print(f"\n  System:          {profile.system} ({profile.arch})")
    print(f"  Device:          {profile.device_name}")
    print(f"  Device Type:     {profile.device_type.upper()}")
    if profile.vram_gb > 0:
        print(f"  GPU Memory:      {profile.vram_gb:.1f} GB")
    print(f"  System RAM:      {profile.ram_gb:.1f} GB")
    print(f"\n  Compute Backend: {profile.compute_backend}")
    print(f"  Model Profile:   {profile.model_profile}")
    print(f"  Batch Size:      {profile.recommended_batch_size}")
    print(f"  Steps:           {profile.recommended_steps}")
    print(f"  Est. Speed:      {profile.estimated_speed}")
    print("\n" + "="*60 + "\n")


def get_torch_device(profile: HardwareProfile) -> str:
    """Get PyTorch device string"""
    if profile.compute_backend == "cuda":
        return "cuda"
    elif profile.compute_backend == "mps":
        return "mps"
    else:
        return "cpu"


def get_model_variant(profile: HardwareProfile) -> str:
    """Get model variant based on profile"""
    if profile.model_profile == "high":
        return "full"
    elif profile.model_profile == "medium":
        return "optimized"
    elif profile.model_profile == "low":
        return "lite"
    else:
        return "onnx"


# Main execution
if __name__ == "__main__":
    profile = detect_hardware()
    print_hardware_info(profile)

    # Save configuration
    config_dir = Path(__file__).parent.parent / "config"
    save_hardware_config(profile, config_dir / "hardware.json")
