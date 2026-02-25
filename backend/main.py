"""
Boutique Virtual Try-On System
Professional local-only solution for fashion boutiques
Supports all garment categories - no content restrictions
All data stored locally with SQLite
"""

import asyncio
import json
import uuid
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import aiohttp
import websockets

from database import (
    init_database, ClientDB, GarmentDB, TryOnSessionDB,
    TryOnResultDB, SettingsDB, get_db
)

# ============== Configuration ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
GARMENTS_DIR = DATA_DIR / "garments"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

# Create directories
for dir_path in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, GARMENTS_DIR, THUMBNAILS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_WS_URL = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")
COMFYUI_INPUT_DIR = BASE_DIR.parent / "ComfyUI" / "input"
COMFYUI_OUTPUT_DIR = BASE_DIR.parent / "ComfyUI" / "output"

# Ensure ComfyUI directories exist
COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMFYUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== FastAPI App ==============
app = FastAPI(
    title="Boutique Try-On System",
    description="Professional virtual try-on for fashion boutiques. Local-only, unrestricted.",
    version="2.0.0"
)

# CORS - local only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (also persisted to DB)
active_jobs = {}


# ============== Pydantic Models ==============

class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class GarmentCreate(BaseModel):
    name: str
    category_id: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    sku: Optional[str] = None
    tags: Optional[List[str]] = None


class TryOnSettings(BaseModel):
    prompt: str = "person wearing the garment, professional fashion photo, high quality, detailed fabric texture"
    negative_prompt: str = "blurry, distorted, low quality, deformed, bad anatomy, ugly"
    steps: int = 35
    cfg_scale: float = 2.5
    sampler: str = "euler_ancestral"
    seed: int = -1
    denoise: float = 0.85
    model: str = "realisticVision"
    # Engine and workflow selection
    engine: str = "catvton"              # "catvton" | "idmvton" | "inpainting"
    face_restore: bool = True
    codeformer_fidelity: float = 0.7     # 0.0=quality, 1.0=fidelity to original
    mask_grow: int = 25
    garment_type: str = "top"            # "top" | "bottom" | "dress" | "full" | "none"
    controlnet_strength: float = 0.8     # For pose ControlNet in no-garment mode


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    result_url: Optional[str] = None
    result_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ============== Utility Functions ==============

def generate_id() -> str:
    return uuid.uuid4().hex[:12]


async def save_upload_file(file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    content = await file.read()
    with open(destination, "wb") as f:
        f.write(content)
    return destination


def copy_to_comfyui_input(source_path: Path, prefix: str = "") -> str:
    """Copy file to ComfyUI input directory and return filename"""
    filename = f"{prefix}_{generate_id()}{source_path.suffix}"
    dest = COMFYUI_INPUT_DIR / filename
    shutil.copy(source_path, dest)
    return filename


def get_available_models() -> List[dict]:
    """Get list of available checkpoint models from both ComfyUI and V2 models directories"""
    models = []

    # Check ComfyUI models directory
    comfyui_models_dir = BASE_DIR.parent / "ComfyUI" / "models" / "checkpoints"
    if comfyui_models_dir.exists():
        for f in comfyui_models_dir.glob("*.safetensors"):
            models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "comfyui"})
        for f in comfyui_models_dir.glob("*.ckpt"):
            models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "comfyui"})

    # Check V2 models directory (our downloaded models)
    v2_models_dir = BASE_DIR.parent / "models"
    if v2_models_dir.exists():
        # Check for base models
        base_models_dir = v2_models_dir / "base_models"
        if base_models_dir.exists():
            for f in base_models_dir.glob("*.safetensors"):
                models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "v2"})
            for f in base_models_dir.glob("*.ckpt"):
                models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "v2"})

        # Check for checkpoints directory
        checkpoints_dir = v2_models_dir / "checkpoints"
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.safetensors"):
                models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "v2"})
            for f in checkpoints_dir.glob("*.ckpt"):
                models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "v2"})

    return models


def check_v2_models_available() -> dict:
    """Check which V2 AI models are available"""
    v2_models_dir = BASE_DIR.parent / "models"
    status = {
        "available": False,
        "face_detection": False,
        "body_pose": False,
        "base_model": False,
        "inpainting": False,
        "models_found": [],
        "models_dir": str(v2_models_dir)
    }

    if not v2_models_dir.exists():
        return status

    # Check for InsightFace models (antelopev2 folder with .onnx files)
    insightface_dir = v2_models_dir / "insightface"
    if insightface_dir.exists():
        # Check for .onnx files recursively (they're in antelopev2 subfolder)
        onnx_files = list(insightface_dir.glob("**/*.onnx"))
        if onnx_files:
            status["face_detection"] = True
            status["models_found"].append("insightface")

    # Check for DWPose models
    dwpose_dir = v2_models_dir / "dwpose"
    if dwpose_dir.exists():
        onnx_files = list(dwpose_dir.glob("*.onnx"))
        if onnx_files:
            status["body_pose"] = True
            status["models_found"].append("dwpose")

    # Check for base models in checkpoints directory (where model_downloader puts them)
    checkpoints_dir = v2_models_dir / "checkpoints"
    base_models_dir = v2_models_dir / "base_models"

    for models_dir in [checkpoints_dir, base_models_dir]:
        if models_dir and models_dir.exists():
            safetensor_files = list(models_dir.glob("*.safetensors"))
            ckpt_files = list(models_dir.glob("*.ckpt"))
            if safetensor_files or ckpt_files:
                status["base_model"] = True
                status["models_found"].append(f"base_model ({len(safetensor_files) + len(ckpt_files)} files)")

                # Check specifically for inpainting model
                inpaint_files = list(models_dir.glob("*inpaint*"))
                if inpaint_files:
                    status["inpainting"] = True
                    status["models_found"].append("inpainting")
                break

    # V2 is available if we have at least face detection or base model
    status["available"] = status["face_detection"] or status["base_model"]

    return status


def _has_face_restore_models() -> bool:
    """Check if CodeFormer face restoration models are available"""
    models_dir = BASE_DIR.parent / "ComfyUI" / "models"
    return (
        any((models_dir / "facerestore_models").glob("codeformer*"))
        and any((models_dir / "facedetection").glob("*.pth"))
    )


def _has_controlnet_openpose() -> bool:
    """Check if ControlNet OpenPose model is available"""
    return (BASE_DIR.parent / "ComfyUI" / "models" / "controlnet" / "control_v11p_sd15_openpose.pth").exists()


def _add_face_restore_nodes(workflow: dict, source_image_node: str, save_node_id: str, settings: TryOnSettings) -> str:
    """Add CodeFormer face restoration nodes to a workflow. Returns the node ID that produces the final image."""
    if not settings.face_restore or not _has_face_restore_models():
        return source_image_node

    workflow["90"] = {
        "class_type": "FaceRestoreModelLoader",
        "inputs": {"model_name": "codeformer-v0.1.0.pth"}
    }
    workflow["91"] = {
        "class_type": "FaceRestoreCFWithModel",
        "inputs": {
            "facerestore_model": ["90", 0],
            "image": [source_image_node, 0],
            "facedetection": "retinaface_resnet50",
            "codeformer_fidelity": settings.codeformer_fidelity,
        }
    }
    return "91"


def build_catvton_workflow(
    person_image: str,
    mask_image: str,
    garment_image: str,
    settings: TryOnSettings,
) -> tuple:
    """CatVTON garment transfer + DWPose pose extraction + CodeFormer face restore.

    Best for: garment try-on with identity preservation. Uses built-in DensePose
    and SCHP (human parsing) for high-quality garment articulation.
    """
    seed = settings.seed if settings.seed != -1 else int(time.time() * 1000) % (2**32)

    workflow = {
        # Load person image
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image}
        },
        # Load garment reference image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": garment_image}
        },
        # Load mask image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image}
        },
        # Convert mask to MASK type
        "4": {
            "class_type": "ImageToMask",
            "inputs": {
                "image": ["3", 0],
                "channel": "red"
            }
        },
        # DWPose: extract pose for reference (body + hand + face keypoints)
        "10": {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["1", 0],
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
            }
        },
        # CatVTON: garment transfer with internal DensePose + SCHP
        "5": {
            "class_type": "CatVTONWrapper",
            "inputs": {
                "image": ["1", 0],
                "refer_image": ["2", 0],
                "mask": ["4", 0],
                "mask_grow": settings.mask_grow,
                "mixed_precision": "fp32",
                "seed": seed,
                "steps": settings.steps,
                "cfg": settings.cfg_scale,
            }
        },
    }

    # Add face restoration if available
    final_image_node = _add_face_restore_nodes(workflow, "5", "99", settings)

    # Save result
    workflow["99"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "boutique_result",
            "images": [final_image_node, 0]
        }
    }

    return workflow, seed


def build_nogarment_workflow(
    person_image: str,
    mask_image: str,
    settings: TryOnSettings,
) -> tuple:
    """Pose-conditioned inpainting without garment transfer.

    Uses ControlNet OpenPose to preserve the model's pose while inpainting
    the masked region based on the text prompt. Falls back to plain inpainting
    if ControlNet model is not available.
    """
    seed = settings.seed if settings.seed != -1 else int(time.time() * 1000) % (2**32)

    model_map = {
        "realisticVision": "realisticVisionV60B1_v51VAE.safetensors",
        "sd15_inpainting": "sd-v1-5-inpainting.ckpt",
        "deliberate": "deliberate_v3.safetensors",
        "dreamshaper": "dreamshaper_8.safetensors",
    }
    checkpoint = model_map.get(settings.model, "realisticVisionV60B1_v51VAE.safetensors")
    use_controlnet = _has_controlnet_openpose()
    # Use higher cfg for inpainting (not CatVTON's 2.5 default)
    inpaint_cfg = max(settings.cfg_scale, 6.0)

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load person image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image}
        },
        # Load mask
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image}
        },
        "3b": {
            "class_type": "ImageToMask",
            "inputs": {
                "image": ["3", 0],
                "channel": "red"
            }
        },
        # Positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": settings.prompt,
                "clip": ["1", 1]
            }
        },
        # Negative prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": settings.negative_prompt,
                "clip": ["1", 1]
            }
        },
        # Encode person image to latent
        "6": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2]
            }
        },
        # Apply mask to latent
        "7": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["6", 0],
                "mask": ["3b", 0]
            }
        },
    }

    # Determine the positive conditioning node (with or without ControlNet)
    positive_node = "4"

    if use_controlnet:
        # DWPose: extract pose from person image
        workflow["10"] = {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["2", 0],
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
            }
        }
        # Load ControlNet OpenPose
        workflow["15"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"}
        }
        # Apply ControlNet to positive conditioning
        workflow["16"] = {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],
                "control_net": ["15", 0],
                "image": ["10", 0],
                "strength": settings.controlnet_strength,
            }
        }
        positive_node = "16"

    # KSampler
    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": min(settings.steps, 25),
            "cfg": inpaint_cfg,
            "sampler_name": settings.sampler,
            "scheduler": "normal",
            "denoise": settings.denoise,
            "model": ["1", 0],
            "positive": [positive_node, 0],
            "negative": ["5", 0],
            "latent_image": ["7", 0]
        }
    }

    # VAE Decode
    workflow["9"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],
            "vae": ["1", 2]
        }
    }

    # Add face restoration if available
    final_image_node = _add_face_restore_nodes(workflow, "9", "99", settings)

    # Save result
    workflow["99"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "boutique_result",
            "images": [final_image_node, 0]
        }
    }

    return workflow, seed


def build_idmvton_workflow(
    person_image: str,
    mask_image: str,
    garment_image: str,
    settings: TryOnSettings,
) -> tuple:
    """IDM-VTON high-quality garment transfer + DWPose + CodeFormer face restore.

    Best quality garment articulation (SDXL-based, uses IP-Adapter).
    WARNING: Very slow on CPU (~15-30 min), requires ~12GB RAM.
    """
    seed = settings.seed if settings.seed != -1 else int(time.time() * 1000) % (2**32)

    workflow = {
        # Load IDM-VTON pipeline (MUST be float32 for CPU)
        "1": {
            "class_type": "PipelineLoader",
            "inputs": {"weight_dtype": "float32"}
        },
        # Load person image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image}
        },
        # Load garment image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": garment_image}
        },
        # Load mask image
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image}
        },
        # DWPose: extract pose (required input for IDM-VTON)
        "10": {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["2", 0],
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
            }
        },
        # IDM-VTON inference
        "5": {
            "class_type": "IDM-VTON",
            "inputs": {
                "pipeline": ["1", 0],
                "human_img": ["2", 0],
                "pose_img": ["10", 0],
                "mask_img": ["4", 0],
                "garment_img": ["3", 0],
                "garment_description": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "width": 768,
                "height": 1024,
                "num_inference_steps": min(settings.steps, 20),
                "guidance_scale": 2.0,
                "strength": 1.0,
                "seed": seed,
            }
        },
    }

    # Add face restoration if available
    final_image_node = _add_face_restore_nodes(workflow, "5", "99", settings)

    # Save result
    workflow["99"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "boutique_result",
            "images": [final_image_node, 0]
        }
    }

    return workflow, seed


def build_remove_clothing_workflow(
    person_image: str,
    mask_image: str,
    settings: TryOnSettings,
) -> tuple:
    """Remove all clothing and generate bare skin while preserving identity.

    Uses ControlNet OpenPose for pose/body shape preservation, high denoise
    for full clothing replacement, and CodeFormer for face restoration.
    Prompt is overridden to target bare skin with matching skin tone.
    """
    seed = settings.seed if settings.seed != -1 else int(time.time() * 1000) % (2**32)

    model_map = {
        "realisticVision": "realisticVisionV60B1_v51VAE.safetensors",
        "sd15_inpainting": "sd-v1-5-inpainting.ckpt",
        "deliberate": "deliberate_v3.safetensors",
        "dreamshaper": "dreamshaper_8.safetensors",
    }
    checkpoint = model_map.get(settings.model, "realisticVisionV60B1_v51VAE.safetensors")
    use_controlnet = _has_controlnet_openpose()

    # Override prompts for bare-skin generation with skin tone & body preservation
    positive_prompt = (
        "bare skin, natural body, same skin tone, same skin color, "
        "photorealistic human body, consistent skin texture, "
        "smooth natural skin, anatomically correct, same body proportions, "
        "professional photography, high quality, 8k, detailed skin"
    )
    negative_prompt = (
        "clothing, fabric, garment, shirt, pants, dress, underwear, "
        "blurry, distorted, deformed, bad anatomy, disfigured, "
        "low quality, artifacts, unnatural skin, extra limbs, "
        "different skin color, tattoo, text, watermark"
    )

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load person image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image}
        },
        # Load mask
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image}
        },
        "3b": {
            "class_type": "ImageToMask",
            "inputs": {
                "image": ["3", 0],
                "channel": "red"
            }
        },
        # Positive prompt - bare skin
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_prompt,
                "clip": ["1", 1]
            }
        },
        # Negative prompt - no clothing
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1]
            }
        },
        # Encode person image to latent
        "6": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2]
            }
        },
        # Apply mask to latent
        "7": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["6", 0],
                "mask": ["3b", 0]
            }
        },
    }

    # Positive conditioning node (with or without ControlNet)
    positive_node = "4"

    if use_controlnet:
        # DWPose: preserve exact body pose and proportions
        workflow["10"] = {
            "class_type": "DWPreprocessor",
            "inputs": {
                "image": ["2", 0],
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
            }
        }
        # Load ControlNet OpenPose
        workflow["15"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"}
        }
        # Strong ControlNet to lock body shape and pose
        workflow["16"] = {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],
                "control_net": ["15", 0],
                "image": ["10", 0],
                "strength": min(settings.controlnet_strength + 0.1, 1.5),  # Stronger for body preservation
            }
        }
        positive_node = "16"

    # KSampler - high denoise to fully replace clothing with skin
    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": max(settings.steps, 30),   # At least 30 steps for clean skin
            "cfg": 7.0,                          # Higher CFG for faithful prompt following
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 0.95,                     # Near-full denoise to completely replace clothing
            "model": ["1", 0],
            "positive": [positive_node, 0],
            "negative": ["5", 0],
            "latent_image": ["7", 0]
        }
    }

    # VAE Decode
    workflow["9"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],
            "vae": ["1", 2]
        }
    }

    # Face restoration - higher fidelity to preserve identity
    final_image_node = "9"
    if settings.face_restore and _has_face_restore_models():
        workflow["90"] = {
            "class_type": "FaceRestoreModelLoader",
            "inputs": {"model_name": "codeformer-v0.1.0.pth"}
        }
        workflow["91"] = {
            "class_type": "FaceRestoreCFWithModel",
            "inputs": {
                "facerestore_model": ["90", 0],
                "image": ["9", 0],
                "facedetection": "retinaface_resnet50",
                "codeformer_fidelity": max(settings.codeformer_fidelity, 0.8),  # High fidelity for identity
            }
        }
        final_image_node = "91"

    # Save result
    workflow["99"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "boutique_result",
            "images": [final_image_node, 0]
        }
    }

    return workflow, seed


def build_faceswap_workflow(
    target_image: str,
    source_image: str,
    codeformer_fidelity: float = 0.95,
) -> tuple:
    """Face swap using ReActor (InsightFace inswapper_128) + CodeFormer face restore.

    Swaps the face from source_image onto target_image while preserving the
    target's body, pose, and background.

    Args:
        target_image: ComfyUI filename of the body/target image (keeps body)
        source_image: ComfyUI filename of the face source image (takes face from)
        codeformer_fidelity: CodeFormer fidelity (0=quality, 1=identity preservation)
    """
    seed = int(time.time() * 1000) % (2**32)

    workflow = {
        # Load target image (body to keep)
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": target_image}
        },
        # Load source image (face to use)
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": source_image}
        },
        # ReActor face swap
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "codeformer-v0.1.0.pth",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            }
        },
    }

    # Additional CodeFormer pass for extra quality
    if _has_face_restore_models():
        workflow["90"] = {
            "class_type": "FaceRestoreModelLoader",
            "inputs": {"model_name": "codeformer-v0.1.0.pth"}
        }
        workflow["91"] = {
            "class_type": "FaceRestoreCFWithModel",
            "inputs": {
                "facerestore_model": ["90", 0],
                "image": ["3", 0],
                "facedetection": "retinaface_resnet50",
                "codeformer_fidelity": codeformer_fidelity,
            }
        }
        final_node = "91"
    else:
        final_node = "3"

    # Save result
    workflow["99"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "boutique_faceswap",
            "images": [final_node, 0]
        }
    }

    return workflow, seed


def build_tryon_workflow(
    person_image: str,
    mask_image: str,
    settings: TryOnSettings,
    garment_image: str = None,
) -> tuple:
    """Build ComfyUI workflow for try-on - dispatches to the appropriate engine.

    Engines:
    - catvton: CatVTON garment transfer (default, fastest on CPU)
    - idmvton: IDM-VTON high-quality transfer (slow on CPU, best quality)
    - inpainting: Plain inpainting fallback

    If garment_type is 'remove', strips all clothing preserving body identity.
    If no garment_image is provided, uses pose-conditioned inpainting.
    """
    # Remove clothing mode - dedicated workflow
    if settings.garment_type == "remove":
        print(f"[Workflow] Dispatching: REMOVE CLOTHING (garment_type={settings.garment_type})")
        return build_remove_clothing_workflow(person_image, mask_image, settings)

    if garment_image:
        if settings.engine == "idmvton":
            print(f"[Workflow] Dispatching: IDM-VTON (engine={settings.engine})")
            return build_idmvton_workflow(person_image, mask_image, garment_image, settings)
        else:
            print(f"[Workflow] Dispatching: CatVTON (engine={settings.engine})")
            return build_catvton_workflow(person_image, mask_image, garment_image, settings)
    else:
        print(f"[Workflow] Dispatching: NO-GARMENT inpainting (garment_type={settings.garment_type})")
        return build_nogarment_workflow(person_image, mask_image, settings)


async def queue_comfyui_prompt(workflow: dict) -> str:
    """Queue workflow in ComfyUI"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=500, detail=f"ComfyUI error: {error_text}")
            result = await response.json()
            return result.get("prompt_id")


async def wait_for_completion(prompt_id: str, job_id: str):
    """Wait for ComfyUI completion via WebSocket with robust timeout handling"""
    try:
        async with websockets.connect(
            f"{COMFYUI_WS_URL}?clientId={job_id}",
            ping_interval=30,
            ping_timeout=60,
            close_timeout=10
        ) as ws:
            while True:
                try:
                    # Long timeout for CPU processing - 30 minutes per message
                    message = await asyncio.wait_for(ws.recv(), timeout=1800)
                    data = json.loads(message)

                    if data.get("type") == "executing":
                        node = data.get("data", {}).get("node")
                        if node is None:
                            # Execution completed
                            active_jobs[job_id]["status"] = "completed"
                            active_jobs[job_id]["progress"] = 100
                            break
                        else:
                            # Update which node is processing
                            active_jobs[job_id]["current_node"] = node

                    elif data.get("type") == "progress":
                        value = data.get("data", {}).get("value", 0)
                        max_val = data.get("data", {}).get("max", 100)
                        progress = int((value / max_val) * 100)
                        active_jobs[job_id]["progress"] = progress

                    elif data.get("type") == "execution_error":
                        error_msg = data.get("data", {}).get("exception_message", "Unknown error")
                        active_jobs[job_id]["status"] = "failed"
                        active_jobs[job_id]["error"] = error_msg
                        break

                    active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

                except asyncio.TimeoutError:
                    # Check if job is still in ComfyUI queue
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{COMFYUI_URL}/queue") as resp:
                                queue = await resp.json()
                                running = queue.get("queue_running", [])
                                pending = queue.get("queue_pending", [])
                                if not running and not pending:
                                    # Queue empty, job might be done
                                    active_jobs[job_id]["status"] = "completed"
                                    active_jobs[job_id]["progress"] = 100
                                    break
                                # Still processing, continue waiting
                                continue
                    except:
                        continue

    except websockets.exceptions.ConnectionClosed:
        # Connection closed, check if job completed
        if active_jobs[job_id]["status"] != "completed":
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = "WebSocket connection closed"
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "service": "Boutique Try-On System",
        "version": "2.0.0",
        "mode": "local-only",
        "status": "running"
    }


# ============== Auto-Mask Generation ==============

# Human parsing model (lazy loaded)
_parsing_session = None

def get_parsing_session():
    """Lazy-load the ATR human parsing ONNX model"""
    global _parsing_session
    if _parsing_session is None:
        import onnxruntime as ort
        model_path = BASE_DIR.parent / "ComfyUI" / "models" / "IDM-VTON" / "humanparsing" / "parsing_atr.onnx"
        if model_path.exists():
            _parsing_session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        else:
            raise FileNotFoundError(f"Human parsing model not found at {model_path}")
    return _parsing_session


# ATR label map: 0-Background, 1-Hat, 2-Hair, 3-Sunglasses, 4-Upper-clothes, 5-Skirt, 6-Pants, 7-Dress, 8-Belt,
#                9-Left-shoe, 10-Right-shoe, 11-Face, 12-Left-leg, 13-Right-leg, 14-Left-arm, 15-Right-arm, 16-Bag, 17-Scarf
ATR_REGION_LABELS = {
    'top': [4, 17],           # Upper-clothes, Scarf
    'bottom': [5, 6],         # Skirt, Pants
    'dress': [7],             # Dress
    'full': [4, 5, 6, 7, 17], # All clothing
    'remove': [4, 5, 6, 7, 8, 9, 10, 16, 17],  # All clothing + belt + shoes + bag + scarf
}


def generate_auto_mask(image_bytes: bytes, region: str) -> bytes:
    """Generate a mask for the specified clothing region using human parsing"""
    import numpy as np
    from PIL import Image
    import io

    # Load and preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size

    # Resize to model input size (512x512)
    input_size = (512, 512)
    img_resized = img.resize(input_size, Image.BILINEAR)

    # Normalize to [0, 1] and then to model expected format
    img_np = np.array(img_resized, dtype=np.float32)
    # Standard normalization: (img - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np / 255.0 - mean) / std
    # NCHW format
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    # Run inference
    session = get_parsing_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_np})

    # Get parsing result (argmax of logits)
    parsing = outputs[0]
    if len(parsing.shape) == 4:
        parsing = np.argmax(parsing[0], axis=0)  # [H, W]
    elif len(parsing.shape) == 3:
        parsing = np.argmax(parsing, axis=0) if parsing.shape[0] < parsing.shape[1] else parsing[0]

    # Get target labels for the region
    target_labels = ATR_REGION_LABELS.get(region, ATR_REGION_LABELS['full'])

    # Create binary mask (white = replace, black = keep)
    mask = np.zeros(parsing.shape, dtype=np.uint8)
    for label in target_labels:
        mask[parsing == label] = 255

    # Dilate mask slightly for better coverage
    try:
        import cv2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    except ImportError:
        pass

    # Resize mask back to original image size
    mask_img = Image.fromarray(mask, mode='L')
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

    # Convert to PNG bytes
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    return buf.getvalue()


@app.post("/api/auto-mask")
async def auto_mask(
    person_image: UploadFile = File(...),
    region: str = Form("top"),
):
    """Generate an automatic mask for the specified clothing region.

    Regions: top, bottom, dress, full
    Returns a PNG mask image (white=replace, black=keep)
    """
    if region not in ATR_REGION_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}. Must be one of: {list(ATR_REGION_LABELS.keys())}")

    try:
        image_bytes = await person_image.read()
        mask_bytes = generate_auto_mask(image_bytes, region)

        from fastapi.responses import Response
        return Response(content=mask_bytes, media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-mask generation failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Check system health - returns healthy if either ComfyUI OR V2 models are available"""
    comfyui_status = "disconnected"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    comfyui_status = "connected"
    except:
        pass

    # Check V2 models availability
    v2_models = check_v2_models_available()

    # System is healthy if ComfyUI is connected OR V2 models are available
    is_healthy = comfyui_status == "connected" or v2_models["available"]

    return {
        "status": "healthy" if is_healthy else "degraded",
        "comfyui": comfyui_status,
        "v2_engine": "ready" if v2_models["available"] else "not_available",
        "v2_models": v2_models,
        "database": "connected",
        "models": get_available_models()
    }


@app.get("/api/models/status")
async def check_model_status():
    """Check availability of all required models for each workflow engine"""
    models_dir = BASE_DIR.parent / "ComfyUI" / "models"
    has_controlnet = _has_controlnet_openpose()
    has_face_restore = _has_face_restore_models()
    has_catvton = (models_dir / "CatVTON" / "stable-diffusion-inpainting").exists()
    has_idmvton = (models_dir / "IDM-VTON" / "unet").exists()
    has_checkpoint = (models_dir / "checkpoints" / "realisticVisionV60B1_v51VAE.safetensors").exists()
    has_sam = any((models_dir / "sams").glob("*.pth"))
    has_reactor = (models_dir / "insightface" / "inswapper_128.onnx").exists()

    engines = []
    if has_catvton:
        engines.append("catvton")
    if has_idmvton:
        engines.append("idmvton")
    if has_checkpoint:
        engines.append("inpainting")
    if has_reactor:
        engines.append("faceswap")

    return {
        "controlnet_openpose": has_controlnet,
        "codeformer": has_face_restore,
        "catvton": has_catvton,
        "idmvton": has_idmvton,
        "checkpoint": has_checkpoint,
        "sam": has_sam,
        "reactor_faceswap": has_reactor,
        "available_engines": engines,
    }


# ============== Client Endpoints ==============

@app.post("/api/clients")
async def create_client(client: ClientCreate):
    """Create new client"""
    client_id = generate_id()
    result = ClientDB.create(
        id=client_id,
        name=client.name,
        email=client.email,
        phone=client.phone,
        notes=client.notes
    )
    return result


@app.get("/api/clients")
async def list_clients():
    """List all clients"""
    return {"clients": ClientDB.list_all()}


@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get client by ID"""
    client = ClientDB.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return client


# ============== Garment Catalog Endpoints ==============

@app.get("/api/categories")
async def list_categories():
    """List all garment categories"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM garment_categories ORDER BY sort_order"
        ).fetchall()
        return {"categories": [dict(row) for row in rows]}


@app.post("/api/garments")
async def upload_garment(
    image: UploadFile = File(...),
    name: str = Form(...),
    category_id: str = Form(None),
    description: str = Form(None),
    price: float = Form(None),
    sku: str = Form(None),
    tags: str = Form(None)  # Comma-separated
):
    """Upload new garment to catalog"""
    garment_id = generate_id()

    # Save image
    ext = Path(image.filename).suffix or ".jpg"
    image_path = GARMENTS_DIR / f"{garment_id}{ext}"
    await save_upload_file(image, image_path)

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    result = GarmentDB.create(
        id=garment_id,
        name=name,
        image_path=str(image_path),
        category_id=category_id,
        description=description,
        price=price,
        sku=sku,
        tags=tag_list
    )

    return result


@app.get("/api/garments")
async def list_garments(
    category_id: str = Query(None),
    search: str = Query(None)
):
    """List garments, optionally filtered by category or search"""
    if search:
        garments = GarmentDB.search(search)
    else:
        garments = GarmentDB.list_by_category(category_id)
    return {"garments": garments}


@app.get("/api/garments/{garment_id}")
async def get_garment(garment_id: str):
    """Get garment by ID"""
    garment = GarmentDB.get(garment_id)
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    return garment


@app.get("/api/garments/{garment_id}/image")
async def get_garment_image(garment_id: str):
    """Get garment image"""
    garment = GarmentDB.get(garment_id)
    if not garment or not garment.get("image_path"):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(garment["image_path"])


# ============== Try-On Endpoints ==============

@app.post("/api/tryon")
async def start_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    mask_image: UploadFile = File(None),
    garment_image: UploadFile = File(None),
    garment_id: str = Form(None),
    client_id: str = Form(None),
    prompt: str = Form("person wearing elegant garment, professional fashion photography, high quality, detailed"),
    negative_prompt: str = Form("blurry, distorted, low quality, deformed, bad anatomy, ugly, disfigured"),
    steps: int = Form(35),
    cfg_scale: float = Form(2.5),
    sampler: str = Form("euler_ancestral"),
    seed: int = Form(-1),
    denoise: float = Form(0.85),
    model: str = Form("realisticVision"),
    engine: str = Form("catvton"),
    face_restore: bool = Form(True),
    codeformer_fidelity: float = Form(0.7),
    mask_grow: int = Form(25),
    garment_type: str = Form("top"),
    controlnet_strength: float = Form(0.8),
    auto_mask: bool = Form(True),
):
    """Start a try-on generation job.

    Supports three engines:
    - catvton: CatVTON garment transfer (default, recommended)
    - idmvton: IDM-VTON high-quality transfer (slow on CPU)
    - inpainting: Basic inpainting fallback

    If mask_image is not provided and auto_mask=True, a mask is generated
    automatically based on garment_type using ATR human parsing.
    """

    job_id = generate_id()
    session_id = generate_id()
    now = datetime.now().isoformat()

    # Save person image
    person_ext = Path(person_image.filename).suffix or ".png"
    person_path = UPLOADS_DIR / f"person_{job_id}{person_ext}"
    await save_upload_file(person_image, person_path)

    # Handle mask: auto-generate if not provided
    if mask_image and mask_image.filename:
        mask_ext = Path(mask_image.filename).suffix or ".png"
        mask_path = UPLOADS_DIR / f"mask_{job_id}{mask_ext}"
        await save_upload_file(mask_image, mask_path)
    elif auto_mask:
        # Auto-generate mask from person image using human parsing
        with open(person_path, "rb") as f:
            person_bytes = f.read()
        region = garment_type if garment_type in ATR_REGION_LABELS else "full"
        mask_bytes = generate_auto_mask(person_bytes, region)
        mask_path = UPLOADS_DIR / f"mask_{job_id}.png"
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)
    else:
        raise HTTPException(status_code=400, detail="Either mask_image or auto_mask=True is required")

    # Save garment image if provided
    garment_path = None
    if garment_image and garment_image.filename:
        garment_ext = Path(garment_image.filename).suffix or ".png"
        garment_path = UPLOADS_DIR / f"garment_{job_id}{garment_ext}"
        await save_upload_file(garment_image, garment_path)

    # Copy to ComfyUI input directory
    person_comfy = copy_to_comfyui_input(person_path, "person")
    mask_comfy = copy_to_comfyui_input(mask_path, "mask")
    garment_comfy = None
    if garment_path:
        garment_comfy = copy_to_comfyui_input(garment_path, "garment")

    # Create settings
    settings = TryOnSettings(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        seed=seed,
        denoise=denoise,
        model=model,
        engine=engine,
        face_restore=face_restore,
        codeformer_fidelity=codeformer_fidelity,
        mask_grow=mask_grow,
        garment_type=garment_type,
        controlnet_strength=controlnet_strength,
    )

    # Save session to database
    TryOnSessionDB.create(
        id=session_id,
        client_photo_id=str(person_path),
        garment_id=garment_id,
        client_id=client_id,
        prompt=prompt,
        settings=settings.model_dump()
    )

    # Initialize job tracking
    active_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "result_url": None,
        "result_id": None,
        "error": None,
        "created_at": now,
        "updated_at": now
    }

    # Background processing
    async def process_job():
        try:
            active_jobs[job_id]["status"] = "processing"
            TryOnSessionDB.update_status(session_id, "processing")

            # Build and queue workflow
            workflow, used_seed = build_tryon_workflow(
                person_image=person_comfy,
                mask_image=mask_comfy,
                settings=settings,
                garment_image=garment_comfy
            )

            start_time = time.time()
            prompt_id = await queue_comfyui_prompt(workflow)
            await wait_for_completion(prompt_id, job_id)

            generation_time = time.time() - start_time

            if active_jobs[job_id]["status"] == "completed":
                # Find and save result
                result_files = sorted(
                    COMFYUI_OUTPUT_DIR.glob("boutique_result_*.png"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if result_files:
                    # Copy result to our results directory
                    result_id = generate_id()
                    result_path = RESULTS_DIR / f"{result_id}.png"
                    shutil.copy(result_files[0], result_path)

                    # Save to database
                    TryOnResultDB.create(
                        id=result_id,
                        session_id=session_id,
                        image_path=str(result_path),
                        seed=used_seed,
                        generation_time=generation_time,
                        model_used=settings.model
                    )

                    active_jobs[job_id]["result_id"] = result_id
                    active_jobs[job_id]["result_url"] = f"/api/results/{result_id}/image"

                TryOnSessionDB.update_status(session_id, "completed")
            else:
                TryOnSessionDB.update_status(session_id, "failed")

        except Exception as e:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            TryOnSessionDB.update_status(session_id, "failed")

    background_tasks.add_task(process_job)

    return {"job_id": job_id, "session_id": session_id, "status": "queued"}


@app.post("/api/faceswap")
async def start_faceswap(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(...),
    source_image: UploadFile = File(...),
    codeformer_fidelity: float = Form(0.95),
    client_id: str = Form(None),
):
    """Start a face swap job.

    Swaps the face from source_image onto target_image using ReActor (InsightFace).
    - target_image: The body/background image (face will be replaced)
    - source_image: The face source image (face will be taken from here)
    - codeformer_fidelity: Face restore quality (0.0=quality, 1.0=identity). Default 0.95
    """
    job_id = generate_id()
    session_id = generate_id()
    now = datetime.now().isoformat()

    # Save images
    target_ext = Path(target_image.filename).suffix or ".png"
    target_path = UPLOADS_DIR / f"faceswap_target_{job_id}{target_ext}"
    await save_upload_file(target_image, target_path)

    source_ext = Path(source_image.filename).suffix or ".png"
    source_path = UPLOADS_DIR / f"faceswap_source_{job_id}{source_ext}"
    await save_upload_file(source_image, source_path)

    # Copy to ComfyUI input directory
    target_comfy = copy_to_comfyui_input(target_path, "fstarget")
    source_comfy = copy_to_comfyui_input(source_path, "fssource")

    # Save session to database
    TryOnSessionDB.create(
        id=session_id,
        client_photo_id=str(source_path),
        garment_id=None,
        client_id=client_id,
        prompt="face_swap",
        settings={"type": "faceswap", "codeformer_fidelity": codeformer_fidelity}
    )

    # Initialize job tracking
    active_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "result_url": None,
        "result_id": None,
        "error": None,
        "created_at": now,
        "updated_at": now
    }

    async def process_faceswap():
        try:
            active_jobs[job_id]["status"] = "processing"
            TryOnSessionDB.update_status(session_id, "processing")

            workflow, used_seed = build_faceswap_workflow(
                target_image=target_comfy,
                source_image=source_comfy,
                codeformer_fidelity=codeformer_fidelity,
            )

            start_time = time.time()
            prompt_id = await queue_comfyui_prompt(workflow)
            await wait_for_completion(prompt_id, job_id)

            generation_time = time.time() - start_time

            if active_jobs[job_id]["status"] == "completed":
                result_files = sorted(
                    COMFYUI_OUTPUT_DIR.glob("boutique_faceswap_*.png"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if result_files:
                    result_id = generate_id()
                    result_path = RESULTS_DIR / f"{result_id}.png"
                    shutil.copy(result_files[0], result_path)

                    TryOnResultDB.create(
                        id=result_id,
                        session_id=session_id,
                        image_path=str(result_path),
                        seed=used_seed,
                        generation_time=generation_time,
                        model_used="reactor_faceswap"
                    )

                    active_jobs[job_id]["result_id"] = result_id
                    active_jobs[job_id]["result_url"] = f"/api/results/{result_id}/image"

                TryOnSessionDB.update_status(session_id, "completed")
            else:
                TryOnSessionDB.update_status(session_id, "failed")

        except Exception as e:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            TryOnSessionDB.update_status(session_id, "failed")

    background_tasks.add_task(process_faceswap)

    return {"job_id": job_id, "session_id": session_id, "status": "queued"}


@app.get("/api/tryon/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**active_jobs[job_id])


# ============== Results Endpoints ==============

@app.get("/api/results/{result_id}/image")
async def get_result_image(result_id: str):
    """Get result image"""
    result_path = RESULTS_DIR / f"{result_id}.png"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_path)


@app.get("/api/results")
async def list_results(limit: int = Query(50)):
    """List recent results"""
    sessions = TryOnSessionDB.list_recent(limit)
    results = []
    for session in sessions:
        session_results = TryOnResultDB.get_by_session(session["id"])
        for r in session_results:
            r["session"] = session
            results.append(r)
    return {"results": results}


@app.post("/api/results/{result_id}/favorite")
async def toggle_favorite(result_id: str, is_favorite: bool = Form(True)):
    """Toggle favorite status"""
    TryOnResultDB.set_favorite(result_id, is_favorite)
    return {"status": "updated", "is_favorite": is_favorite}


@app.get("/api/results/favorites")
async def get_favorites():
    """Get favorite results"""
    return {"favorites": TryOnResultDB.get_favorites()}


# ============== Settings Endpoints ==============

@app.get("/api/settings")
async def get_settings():
    """Get all settings"""
    return {"settings": SettingsDB.get_all()}


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update settings"""
    for key, value in settings.items():
        SettingsDB.set(key, str(value))
    return {"status": "updated"}


@app.get("/api/models")
async def list_models():
    """List available AI models"""
    return {"models": get_available_models()}


# ============== Data Management ==============

@app.delete("/api/data/clear-results")
async def clear_results():
    """Clear all generated results (keeps garments and clients)"""
    # Clear results directory
    for f in RESULTS_DIR.glob("*"):
        if f.is_file():
            f.unlink()

    # Clear database results
    with get_db() as conn:
        conn.execute("DELETE FROM tryon_results")
        conn.execute("DELETE FROM tryon_sessions")

    # Clear active jobs
    global active_jobs
    active_jobs = {}

    return {"status": "cleared", "message": "All results cleared"}


@app.delete("/api/data/clear-uploads")
async def clear_uploads():
    """Clear temporary uploads"""
    for f in UPLOADS_DIR.glob("*"):
        if f.is_file():
            f.unlink()
    return {"status": "cleared"}


@app.get("/api/data/stats")
async def get_stats():
    """Get system statistics"""
    with get_db() as conn:
        clients = conn.execute("SELECT COUNT(*) as count FROM clients").fetchone()["count"]
        garments = conn.execute("SELECT COUNT(*) as count FROM garments").fetchone()["count"]
        sessions = conn.execute("SELECT COUNT(*) as count FROM tryon_sessions").fetchone()["count"]
        results = conn.execute("SELECT COUNT(*) as count FROM tryon_results").fetchone()["count"]
        favorites = conn.execute("SELECT COUNT(*) as count FROM tryon_results WHERE is_favorite = 1").fetchone()["count"]

    return {
        "clients": clients,
        "garments": garments,
        "sessions": sessions,
        "results": results,
        "favorites": favorites
    }


# ============== Static Files ==============

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/garments", StaticFiles(directory=str(GARMENTS_DIR)), name="garments")


# ============== V2 API Integration ==============

try:
    from api_v2 import include_v2_router
    include_v2_router(app)
    V2_API_AVAILABLE = True
    print("V2 API (Advanced AI Engine) loaded")
except ImportError as e:
    V2_API_AVAILABLE = False
    print(f"V2 API not available: {e}")


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    init_database()
    print("\n" + "="*50)
    print("   COMIFY - Virtual Try-On System")
    print("="*50)
    print(f"\nDatabase: {BASE_DIR / 'data' / 'boutique.db'}")
    print(f"Results: {RESULTS_DIR}")
    print(f"ComfyUI: {COMFYUI_URL}")
    print(f"V2 API: {'Enabled' if V2_API_AVAILABLE else 'Disabled'}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
