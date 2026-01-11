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

# Model definitions with HuggingFace URLs
MODELS = {
    "face_detection": {
        "name": "InsightFace Buffalo_L",
        "files": [
            {
                "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/insightface/models/buffalo_l.zip",
                "path": "models/insightface/buffalo_l.zip",
                "size_mb": 500,
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
                "path": "models/dwpose/dw-ll_ucoco_384.onnx",
                "size_mb": 200
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "segmentation": {
        "name": "Segment Anything (SAM)",
        "files": [
            {
                "url": "https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth",
                "path": "models/sam/sam_vit_h.pth",
                "size_mb": 2500,
                "profiles": ["high"]
            },
            {
                "url": "https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_b_01ec64.pth",
                "path": "models/sam/sam_vit_b.pth",
                "size_mb": 375,
                "profiles": ["medium", "low", "cpu_optimized"]
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "tryon_idmvton": {
        "name": "IDM-VTON",
        "files": [
            {
                "url": "https://huggingface.co/yisol/IDM-VTON/resolve/main/unet/diffusion_pytorch_model.safetensors",
                "path": "models/idm-vton/unet/diffusion_pytorch_model.safetensors",
                "size_mb": 1700
            },
            {
                "url": "https://huggingface.co/yisol/IDM-VTON/resolve/main/image_encoder/model.safetensors",
                "path": "models/idm-vton/image_encoder/model.safetensors",
                "size_mb": 1200
            }
        ],
        "required": True,
        "profiles": ["high", "medium"]
    },
    "tryon_ootd": {
        "name": "OOTDiffusion",
        "files": [
            {
                "url": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/ootd_hd.safetensors",
                "path": "models/ootdiffusion/ootd_hd.safetensors",
                "size_mb": 2000
            }
        ],
        "required": False,
        "profiles": ["high"]
    },
    "face_restore": {
        "name": "CodeFormer",
        "files": [
            {
                "url": "https://huggingface.co/spaces/sczhou/CodeFormer/resolve/main/weights/CodeFormer/codeformer.pth",
                "path": "models/codeformer/codeformer.pth",
                "size_mb": 400
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "upscaler": {
        "name": "RealESRGAN x4",
        "files": [
            {
                "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
                "path": "models/realesrgan/RealESRGAN_x4.pth",
                "size_mb": 64
            }
        ],
        "required": False,
        "profiles": ["high", "medium"]
    },
    "base_sd15": {
        "name": "Realistic Vision V6",
        "files": [
            {
                "url": "https://civitai.com/api/download/models/245598",
                "path": "models/checkpoints/realisticVisionV60B1_v51VAE.safetensors",
                "size_mb": 2000
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    },
    "clip_vision": {
        "name": "CLIP Vision",
        "files": [
            {
                "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
                "path": "models/clip/clip-vit-large-patch14.bin",
                "size_mb": 600
            }
        ],
        "required": True,
        "profiles": ["high", "medium", "low", "cpu_optimized"]
    }
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
            if not file_path.exists():
                return False
        return True

    def download_file(self, url: str, dest_path: Path, model_name: str) -> bool:
        """Download a single file with progress tracking"""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
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
            print(f"Error downloading {url}: {e}")
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

            if file_path.exists():
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

            # Handle extraction if needed
            if file_info.get("extract") and file_path.suffix == '.zip':
                import zipfile
                extract_dir = file_path.parent
                print(f"  Extracting to: {extract_dir}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

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

        success = True
        for model_id in models_to_download:
            if not self.download_model(model_id):
                success = False

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
        print(f"\r  [{progress.file_name}] {percent:.1f}% ({speed_mb:.1f} MB/s)", end="", flush=True)
    elif progress.status == "completed":
        print(f"\r  [{progress.file_name}] Complete!                    ")
    elif progress.status == "failed":
        print(f"\r  [{progress.file_name}] FAILED                       ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Comify AI models")
    parser.add_argument("--profile", default="medium", choices=["high", "medium", "low", "cpu_optimized"])
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--list", action="store_true", help="List models without downloading")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
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
            print(f"  {info['name']}: {downloaded} ({required}) [{profiles}]")
    else:
        print(f"\nDownloading models for profile: {args.profile}")
        print("=" * 60)
        downloader.download_all(include_optional=args.include_optional)
        print("\nDownload complete!")
