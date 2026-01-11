"""
AI Engine Configuration
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class AIConfig:
    """AI Engine configuration"""
    # Device settings
    device: str = "cpu"  # cuda, mps, cpu
    compute_backend: str = "onnx"  # cuda, mps, directml, onnx

    # Model paths
    models_dir: Path = None

    # Performance settings
    batch_size: int = 1
    num_inference_steps: int = 15
    use_fp16: bool = False

    # Feature toggles
    enable_face_preservation: bool = True
    enable_skin_tone_matching: bool = True
    enable_pose_transfer: bool = True
    enable_upscaling: bool = True
    enable_face_restore: bool = True

    # Quality settings
    upscale_factor: int = 2
    face_restore_strength: float = 0.7
    denoise_strength: float = 0.85

    # Try-on settings
    tryon_engine: str = "idm-vton"  # idm-vton, ootdiffusion, inpainting
    preserve_background: bool = True

    def __post_init__(self):
        if self.models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        elif isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['models_dir'] = str(self.models_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'AIConfig':
        if 'models_dir' in data:
            data['models_dir'] = Path(data['models_dir'])
        return cls(**data)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'AIConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_config(config_path: Optional[Path] = None) -> AIConfig:
    """Load AI configuration from file or create default"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "ai_config.json"

    if config_path.exists():
        return AIConfig.load(config_path)

    # Load hardware config to set defaults
    hardware_path = config_path.parent / "hardware.json"
    if hardware_path.exists():
        with open(hardware_path, 'r') as f:
            hw = json.load(f)

        config = AIConfig(
            device=hw.get('device_type', 'cpu'),
            compute_backend=hw.get('compute_backend', 'onnx'),
            batch_size=hw.get('recommended_batch_size', 1),
            num_inference_steps=hw.get('recommended_steps', 15),
            use_fp16=hw.get('device_type') in ['cuda', 'mps']
        )
    else:
        config = AIConfig()

    # Save for future use
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)

    return config


def create_config_from_hardware(hardware_profile: dict) -> AIConfig:
    """Create AI config from hardware detection results"""
    device_type = hardware_profile.get('device_type', 'cpu')
    compute_backend = hardware_profile.get('compute_backend', 'onnx')
    model_profile = hardware_profile.get('model_profile', 'cpu_optimized')

    # Set quality based on profile
    if model_profile == 'high':
        steps = 20
        upscale = 4
        batch = 2
    elif model_profile == 'medium':
        steps = 15
        upscale = 2
        batch = 1
    elif model_profile == 'low':
        steps = 12
        upscale = 2
        batch = 1
    else:  # cpu_optimized
        steps = 8
        upscale = 1
        batch = 1

    return AIConfig(
        device=device_type,
        compute_backend=compute_backend,
        batch_size=batch,
        num_inference_steps=steps,
        use_fp16=device_type in ['cuda', 'mps'],
        upscale_factor=upscale,
        enable_upscaling=upscale > 1
    )
