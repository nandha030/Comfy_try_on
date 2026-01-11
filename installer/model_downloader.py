"""
Model Downloader for Comify
Downloads and manages AI models based on hardware profile
"""

import os
import sys
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Model definitions with working URLs - Uncensored models for professional use
MODELS = {
    "face_detection": {
        "name": "InsightFace Antelopev2",
        "files": [
            {
                "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip",
                "path": "insightface/models/antelopev2.zip",
                "size_mb": 360,
                "extract": True
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "body_pose": {
        "name": "DWPose",
        "files": [
            {
                "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
                "path": "dwpose/dw-ll_ucoco_384.onnx",
                "size_mb": 200
            },
            {
                "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
                "path": "dwpose/yolox_l.onnx",
                "size_mb": 200
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    # Uncensored Base Models
    "base_realistic": {
        "name": "Realistic Vision V6 (Uncensored)",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/245598?type=Model&format=SafeTensor&size=full&fp=fp16",
                "path": "checkpoints/realisticVisionV60B1_v51VAE.safetensors",
                "size_mb": 2000
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "base_deliberate": {
        "name": "Deliberate V6 (Uncensored)",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/274039?type=Model&format=SafeTensor&size=full&fp=fp16",
                "path": "checkpoints/deliberate_v6.safetensors",
                "size_mb": 2000
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    "base_epicrealism": {
        "name": "epiCRealism (Natural Uncensored)",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/143906?type=Model&format=SafeTensor&size=full&fp=fp16",
                "path": "checkpoints/epicrealism_naturalSinRC1VAE.safetensors",
                "size_mb": 2000
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    "base_cyberrealistic": {
        "name": "CyberRealistic (Uncensored)",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/138176?type=Model&format=SafeTensor&size=full&fp=fp16",
                "path": "checkpoints/cyberrealistic_v42.safetensors",
                "size_mb": 2000
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    # Inpainting model for try-on
    "inpainting": {
        "name": "SD 1.5 Inpainting",
        "files": [
            {
                "url": "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",
                "path": "checkpoints/sd-v1-5-inpainting.ckpt",
                "size_mb": 4000
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    # VAE for better quality
    "vae": {
        "name": "SD VAE (ft-mse)",
        "files": [
            {
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                "path": "vae/vae-ft-mse-840000-ema-pruned.safetensors",
                "size_mb": 335
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    # ControlNet for pose
    "controlnet_openpose": {
        "name": "ControlNet OpenPose",
        "files": [
            {
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
                "path": "controlnet/control_v11p_sd15_openpose.pth",
                "size_mb": 1450
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    "controlnet_inpaint": {
        "name": "ControlNet Inpaint",
        "files": [
            {
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
                "path": "controlnet/control_v11p_sd15_inpaint.pth",
                "size_mb": 1450
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    # Enhancement models
    "face_restore": {
        "name": "CodeFormer",
        "files": [
            {
                "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                "path": "codeformer/codeformer.pth",
                "size_mb": 400
            }
        ],
        "required": False,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "upscaler": {
        "name": "RealESRGAN x4",
        "files": [
            {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "path": "realesrgan/RealESRGAN_x4plus.pth",
                "size_mb": 64
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    # CLIP and IP-Adapter for better garment transfer
    "clip_vision": {
        "name": "CLIP ViT-H",
        "files": [
            {
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
                "path": "clip/clip_vision_h.safetensors",
                "size_mb": 2500
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    "ip_adapter": {
        "name": "IP-Adapter",
        "files": [
            {
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
                "path": "ip_adapter/ip-adapter_sd15.safetensors",
                "size_mb": 44
            },
            {
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
                "path": "ip_adapter/ip-adapter-plus_sd15.safetensors",
                "size_mb": 98
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    # LoRA for better body/skin
    "lora_detail": {
        "name": "Detail Tweaker LoRA",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/62833?type=Model&format=SafeTensor",
                "path": "loras/add_detail.safetensors",
                "size_mb": 144
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    }
}

# Category mappings for easier downloads
CATEGORIES = {
    "insightface": ["face_detection"],
    "dwpose": ["body_pose"],
    "essential": ["face_detection", "body_pose"],
    "base_models": ["base_realistic", "inpainting", "vae"],
    "uncensored": ["base_realistic", "base_deliberate", "base_epicrealism", "base_cyberrealistic"],
    "controlnet": ["controlnet_openpose", "controlnet_inpaint"],
    "enhancement": ["face_restore", "upscaler"],
    "ip_adapter": ["clip_vision", "ip_adapter"],
    "loras": ["lora_detail"],
    "full": ["face_detection", "body_pose", "base_realistic", "inpainting", "vae",
             "controlnet_openpose", "face_restore", "upscaler", "clip_vision", "ip_adapter"],
    "all": list(MODELS.keys())
}


@dataclass
class DownloadProgress:
    """Track download progress"""
    model_name: str
    file_name: str
    total_size: int
    downloaded: int
    speed: float  # bytes per second
    status: str  # downloading, completed, failed, skipped


class ModelDownloader:
    """Handles downloading and managing AI models"""

    def __init__(self, base_path: Path, profile: str = "medium"):
        self.base_path = Path(base_path)
        self.profile = profile
        self.progress_callbacks: List[Callable] = []
        self.current_progress: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()

    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)

    def _notify_progress(self, progress: DownloadProgress):
        """Notify all callbacks of progress"""
        with self._lock:
            self.current_progress[progress.file_name] = progress
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except:
                pass

    def get_required_models(self) -> List[str]:
        """Get list of models required for current profile"""
        required = []
        for model_id, model_info in MODELS.items():
            if self.profile in model_info.get("profiles", []):
                if model_info.get("required", False):
                    required.append(model_id)
        return required

    def get_optional_models(self) -> List[str]:
        """Get list of optional models for current profile"""
        optional = []
        for model_id, model_info in MODELS.items():
            if self.profile in model_info.get("profiles", []):
                if not model_info.get("required", False):
                    optional.append(model_id)
        return optional

    def get_total_download_size(self, model_ids: List[str]) -> int:
        """Calculate total download size in MB"""
        total = 0
        for model_id in model_ids:
            if model_id in MODELS:
                for file_info in MODELS[model_id]["files"]:
                    file_path = self.base_path / file_info["path"]
                    if not file_path.exists():
                        total += file_info.get("size_mb", 0)
        return total

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if model is already downloaded"""
        if model_id not in MODELS:
            return False

        for file_info in MODELS[model_id]["files"]:
            file_path = self.base_path / file_info["path"]
            # For zip files, check if extracted folder exists
            if file_info.get("extract") and file_path.suffix == '.zip':
                extract_dir = file_path.parent / file_path.stem
                if not extract_dir.exists():
                    return False
            elif not file_path.exists():
                return False
        return True

    def download_file(self, url: str, dest_path: Path, model_name: str) -> bool:
        """Download a single file with progress tracking"""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Start download with headers to handle some servers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks for faster downloads
            start_time = __import__('time').time()

            # Create temp file
            temp_path = dest_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Calculate speed
                        elapsed = __import__('time').time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0

                        # Update progress
                        progress = DownloadProgress(
                            model_name=model_name,
                            file_name=dest_path.name,
                            total_size=total_size,
                            downloaded=downloaded,
                            speed=speed,
                            status="downloading"
                        )
                        self._notify_progress(progress)

            # Rename temp to final
            temp_path.rename(dest_path)

            # Notify completion
            progress = DownloadProgress(
                model_name=model_name,
                file_name=dest_path.name,
                total_size=total_size,
                downloaded=total_size,
                speed=0,
                status="completed"
            )
            self._notify_progress(progress)

            return True

        except Exception as e:
            print(f"\nError downloading {url}: {e}")
            # Clean up temp file if exists
            temp_path = dest_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()

            progress = DownloadProgress(
                model_name=model_name,
                file_name=dest_path.name,
                total_size=0,
                downloaded=0,
                speed=0,
                status="failed"
            )
            self._notify_progress(progress)
            return False

    def download_model(self, model_id: str) -> bool:
        """Download a specific model"""
        if model_id not in MODELS:
            print(f"Unknown model: {model_id}")
            return False

        model_info = MODELS[model_id]
        print(f"\nDownloading: {model_info['name']}")

        success = True
        for file_info in model_info["files"]:
            file_path = self.base_path / file_info["path"]

            # Check if specific profile needed
            file_profiles = file_info.get("profiles", model_info.get("profiles", []))
            if self.profile not in file_profiles:
                continue

            # For zip files, check if already extracted
            if file_info.get("extract") and file_path.suffix == '.zip':
                extract_dir = file_path.parent / file_path.stem
                if extract_dir.exists():
                    print(f"  Already extracted: {extract_dir.name}")
                    continue

            if file_path.exists() and not file_info.get("extract"):
                print(f"  Already exists: {file_path.name}")
                progress = DownloadProgress(
                    model_name=model_info['name'],
                    file_name=file_path.name,
                    total_size=0,
                    downloaded=0,
                    speed=0,
                    status="skipped"
                )
                self._notify_progress(progress)
                continue

            print(f"  Downloading: {file_path.name} ({file_info.get('size_mb', '?')} MB)")
            if not self.download_file(file_info["url"], file_path, model_info['name']):
                success = False
                continue

            # Handle extraction if needed
            if file_info.get("extract") and file_path.suffix == '.zip':
                import zipfile
                extract_dir = file_path.parent
                print(f"  Extracting to: {extract_dir}")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"  Extraction complete!")
                except Exception as e:
                    print(f"  Extraction error: {e}")
                    success = False

        return success

    def download_category(self, category: str) -> bool:
        """Download all models in a category"""
        if category not in CATEGORIES:
            print(f"Unknown category: {category}")
            print(f"Available categories: {list(CATEGORIES.keys())}")
            return False

        model_ids = CATEGORIES[category]
        print(f"\nDownloading category '{category}': {len(model_ids)} models")

        total_size = self.get_total_download_size(model_ids)
        print(f"Estimated download size: ~{total_size} MB")

        success = True
        for model_id in model_ids:
            if not self.download_model(model_id):
                success = False

        return success

    def download_all_required(self) -> bool:
        """Download all required models for current profile"""
        required = self.get_required_models()
        print(f"\nRequired models for '{self.profile}' profile: {len(required)}")

        total_size = self.get_total_download_size(required)
        print(f"Total download size: ~{total_size} MB\n")

        success = True
        for model_id in required:
            if not self.download_model(model_id):
                success = False

        return success

    def download_all(self, include_optional: bool = False) -> bool:
        """Download all models"""
        models_to_download = self.get_required_models()
        if include_optional:
            models_to_download.extend(self.get_optional_models())

        print(f"\nDownloading {len(models_to_download)} models...")
        total_size = self.get_total_download_size(models_to_download)
        print(f"Estimated total size: ~{total_size} MB")

        success = True
        for model_id in models_to_download:
            if not self.download_model(model_id):
                success = False

        return success

    def download_essential(self) -> bool:
        """Download just the essential models (face + body detection)"""
        return self.download_category("essential")

    def download_for_tryon(self) -> bool:
        """Download all models needed for virtual try-on"""
        print("\n" + "="*60)
        print("Downloading models for Virtual Try-On")
        print("="*60)

        categories = ["essential", "base_models"]
        success = True
        for cat in categories:
            if not self.download_category(cat):
                success = False

        print("\n" + "="*60)
        print("Download complete! You can now use the try-on feature.")
        print("="*60)
        return success

    def get_status(self) -> Dict:
        """Get status of all models"""
        status = {}
        for model_id, model_info in MODELS.items():
            status[model_id] = {
                "name": model_info["name"],
                "downloaded": self.is_model_downloaded(model_id),
                "required": model_info.get("required", False),
                "profiles": model_info.get("profiles", [])
            }
        return status


def print_download_progress(progress: DownloadProgress):
    """Print download progress to console"""
    if progress.status == "downloading":
        percent = (progress.downloaded / progress.total_size * 100) if progress.total_size > 0 else 0
        speed_mb = progress.speed / (1024 * 1024)
        downloaded_mb = progress.downloaded / (1024 * 1024)
        total_mb = progress.total_size / (1024 * 1024)
        print(f"\r  [{progress.file_name}] {downloaded_mb:.0f}/{total_mb:.0f} MB ({percent:.1f}%) - {speed_mb:.1f} MB/s", end="", flush=True)
    elif progress.status == "completed":
        print(f"\r  [{progress.file_name}] Complete!                                        ")
    elif progress.status == "failed":
        print(f"\r  [{progress.file_name}] FAILED                                           ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Comify AI models")
    parser.add_argument("--profile", default="medium", choices=["high", "medium", "low", "cpu_optimized"])
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--category", help="Download category: essential, base_models, uncensored, controlnet, enhancement, full, all")
    parser.add_argument("--tryon", action="store_true", help="Download all models needed for try-on")
    parser.add_argument("--list", action="store_true", help="List models without downloading")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / "models"
    downloader = ModelDownloader(base_path, profile=args.profile)
    downloader.add_progress_callback(print_download_progress)

    if args.list:
        print("\nModel Status:")
        print("=" * 60)
        status = downloader.get_status()
        for model_id, info in status.items():
            downloaded = "YES" if info["downloaded"] else "NO"
            required = "Required" if info["required"] else "Optional"
            profiles = ", ".join(info["profiles"])
            print(f"  {info['name']}: {downloaded} ({required})")
        print("\nCategories:", list(CATEGORIES.keys()))
    elif args.tryon:
        downloader.download_for_tryon()
    elif args.category:
        print(f"\nDownloading category: {args.category}")
        print("=" * 60)
        downloader.download_category(args.category)
        print("\nDownload complete!")
    else:
        print(f"\nDownloading models for profile: {args.profile}")
        print("=" * 60)
        downloader.download_all(include_optional=args.include_optional)
        print("\nDownload complete!")
