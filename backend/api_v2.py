"""
Comify API v2 - Advanced Try-On Endpoints
Integrates with the new AI engine for face/body preservation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import uuid
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2

# Import AI engine components
try:
    from ai_engine.pipeline import TryOnPipeline, TryOnRequest, create_pipeline
    from ai_engine.config import AIConfig, load_config, create_config_from_hardware
    from installer.hardware_detect import detect_hardware, HardwareProfile
    AI_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"AI Engine not fully available: {e}")
    AI_ENGINE_AVAILABLE = False

# ============== Configuration ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
PROFILES_DIR = DATA_DIR / "profiles"

for dir_path in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, PROFILES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== Router ==============
router = APIRouter(prefix="/api/v2", tags=["v2"])

# ============== Global State ==============
pipeline: Optional[TryOnPipeline] = None
hardware_profile: Optional[HardwareProfile] = None
active_jobs: Dict[str, Dict] = {}


# ============== Pydantic Models ==============

class SystemStatus(BaseModel):
    status: str
    hardware: Dict[str, Any]
    ai_engine: str
    models_loaded: List[str]
    version: str = "2.0.0"


class ModelProfileCreate(BaseModel):
    name: str
    notes: Optional[str] = None


class ModelProfileResponse(BaseModel):
    id: str
    name: str
    skin_colors: Dict[str, List[int]]
    body_shape: str
    created_at: str
    has_face_embedding: bool


class TryOnSettingsV2(BaseModel):
    prompt: str = "person wearing garment, professional fashion photo"
    negative_prompt: str = "blurry, distorted, bad anatomy, deformed"
    steps: int = 20
    seed: int = -1
    denoise: float = 0.85
    preserve_face: bool = True
    preserve_skin_tone: bool = True
    upscale: bool = False
    face_restore: bool = True


class TryOnRequestV2(BaseModel):
    model_profile_id: Optional[str] = None
    target_pose: Optional[str] = None
    settings: TryOnSettingsV2 = TryOnSettingsV2()


class JobStatusV2(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    result_url: Optional[str] = None
    result_id: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    model_used: Optional[str] = None
    created_at: str
    updated_at: str


# ============== Helper Functions ==============

def get_pipeline() -> TryOnPipeline:
    """Get or create the AI pipeline"""
    global pipeline, hardware_profile

    if pipeline is None:
        if not AI_ENGINE_AVAILABLE:
            raise HTTPException(status_code=503, detail="AI Engine not available")

        # Detect hardware
        hardware_profile = detect_hardware()

        # Create config from hardware
        config = create_config_from_hardware(hardware_profile.__dict__)

        # Initialize pipeline
        pipeline = TryOnPipeline(config)

    return pipeline


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


async def save_upload(file: UploadFile, prefix: str) -> Path:
    """Save uploaded file and return path"""
    ext = Path(file.filename).suffix or ".png"
    filename = f"{prefix}_{generate_id()}{ext}"
    filepath = UPLOADS_DIR / filename

    content = await file.read()
    with open(filepath, 'wb') as f:
        f.write(content)

    return filepath


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array"""
    return cv2.imread(str(path))


def save_image(image: np.ndarray, prefix: str) -> Path:
    """Save image and return path"""
    filename = f"{prefix}_{generate_id()}.png"
    filepath = RESULTS_DIR / filename
    cv2.imwrite(str(filepath), image)
    return filepath


# ============== Endpoints ==============

@router.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status including hardware detection"""
    global hardware_profile

    if hardware_profile is None:
        if AI_ENGINE_AVAILABLE:
            hardware_profile = detect_hardware()
        else:
            hardware_profile = None

    hw_info = {}
    if hardware_profile:
        hw_info = {
            "device": hardware_profile.device_name,
            "device_type": hardware_profile.device_type,
            "vram_gb": hardware_profile.vram_gb,
            "ram_gb": hardware_profile.ram_gb,
            "compute_backend": hardware_profile.compute_backend,
            "model_profile": hardware_profile.model_profile,
            "estimated_speed": hardware_profile.estimated_speed
        }

    return SystemStatus(
        status="ready" if AI_ENGINE_AVAILABLE else "limited",
        hardware=hw_info,
        ai_engine="full" if AI_ENGINE_AVAILABLE else "comfyui_only",
        models_loaded=[]
    )


@router.post("/system/detect-hardware")
async def detect_system_hardware():
    """Re-detect hardware and update configuration"""
    global hardware_profile, pipeline

    if not AI_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Engine not available")

    hardware_profile = detect_hardware()

    # Save config
    config_path = BASE_DIR.parent / "config" / "hardware.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(hardware_profile.__dict__, f, indent=2)

    # Reset pipeline to use new config
    pipeline = None

    return {
        "status": "success",
        "hardware": hardware_profile.__dict__
    }


@router.post("/profiles", response_model=ModelProfileResponse)
async def create_model_profile(
    name: str = Form(...),
    notes: Optional[str] = Form(None),
    reference_image: UploadFile = File(...)
):
    """Create a model profile from reference image"""
    pipe = get_pipeline()

    # Save and load image
    image_path = await save_upload(reference_image, "profile")
    image = load_image(image_path)

    # Create profile
    profile = pipe.create_profile_from_image(image, name)

    # Save reference image path
    profile.reference_images.append(str(image_path))
    pipe.save_profile(profile)

    return ModelProfileResponse(
        id=profile.id,
        name=profile.name,
        skin_colors={k: list(v) for k, v in profile.skin_colors.items()},
        body_shape=profile.body_shape,
        created_at=profile.created_at,
        has_face_embedding=profile.face_embedding is not None
    )


@router.get("/profiles", response_model=List[ModelProfileResponse])
async def list_model_profiles():
    """List all model profiles"""
    pipe = get_pipeline()

    profiles = []
    for profile in pipe.profiles.values():
        profiles.append(ModelProfileResponse(
            id=profile.id,
            name=profile.name,
            skin_colors={k: list(v) for k, v in profile.skin_colors.items()},
            body_shape=profile.body_shape,
            created_at=profile.created_at,
            has_face_embedding=profile.face_embedding is not None
        ))

    return profiles


@router.get("/profiles/{profile_id}", response_model=ModelProfileResponse)
async def get_model_profile(profile_id: str):
    """Get a specific model profile"""
    pipe = get_pipeline()

    if profile_id not in pipe.profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = pipe.profiles[profile_id]

    return ModelProfileResponse(
        id=profile.id,
        name=profile.name,
        skin_colors={k: list(v) for k, v in profile.skin_colors.items()},
        body_shape=profile.body_shape,
        created_at=profile.created_at,
        has_face_embedding=profile.face_embedding is not None
    )


@router.delete("/profiles/{profile_id}")
async def delete_model_profile(profile_id: str):
    """Delete a model profile"""
    pipe = get_pipeline()

    if profile_id not in pipe.profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    del pipe.profiles[profile_id]

    # Delete file
    profile_path = pipe.profiles_dir / f"{profile_id}.json"
    if profile_path.exists():
        profile_path.unlink()

    return {"status": "deleted", "id": profile_id}


@router.post("/tryon/advanced")
async def advanced_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    garment_image: Optional[UploadFile] = File(None),
    mask_image: Optional[UploadFile] = File(None),
    model_profile_id: Optional[str] = Form(None),
    target_pose: Optional[str] = Form(None),
    prompt: str = Form("person wearing garment, professional fashion photo"),
    negative_prompt: str = Form("blurry, distorted, bad anatomy"),
    steps: int = Form(20),
    seed: int = Form(-1),
    denoise: float = Form(0.85),
    preserve_face: bool = Form(True),
    preserve_skin_tone: bool = Form(True),
    upscale: bool = Form(False),
    face_restore: bool = Form(True)
):
    """
    Advanced try-on with face/body preservation

    This endpoint uses the new AI engine for better results
    """
    job_id = generate_id()

    # Save uploaded files
    person_path = await save_upload(person_image, "person")
    garment_path = await save_upload(garment_image, "garment") if garment_image else None
    mask_path = await save_upload(mask_image, "mask") if mask_image else None

    # Initialize job
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    # Run generation in background
    background_tasks.add_task(
        run_advanced_tryon,
        job_id,
        person_path,
        garment_path,
        mask_path,
        model_profile_id,
        target_pose,
        prompt,
        negative_prompt,
        steps,
        seed,
        denoise,
        preserve_face,
        preserve_skin_tone,
        upscale,
        face_restore
    )

    return {"job_id": job_id, "status": "queued"}


async def run_advanced_tryon(
    job_id: str,
    person_path: Path,
    garment_path: Optional[Path],
    mask_path: Optional[Path],
    model_profile_id: Optional[str],
    target_pose: Optional[str],
    prompt: str,
    negative_prompt: str,
    steps: int,
    seed: int,
    denoise: float,
    preserve_face: bool,
    preserve_skin_tone: bool,
    upscale: bool,
    face_restore: bool
):
    """Background task for advanced try-on"""
    try:
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["progress"] = 10
        active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

        # Get pipeline
        pipe = get_pipeline()

        # Load images
        person_image = load_image(person_path)
        garment_image = load_image(garment_path) if garment_path else None
        mask_image = load_image(mask_path) if mask_path else None

        # Convert mask to grayscale if needed
        if mask_image is not None and len(mask_image.shape) == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        active_jobs[job_id]["progress"] = 20

        # Create request
        request = TryOnRequest(
            person_image=person_image,
            garment_image=garment_image,
            mask=mask_image,
            model_profile_id=model_profile_id,
            target_pose=target_pose,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            denoise=denoise,
            preserve_face=preserve_face,
            preserve_skin_tone=preserve_skin_tone,
            upscale=upscale,
            face_restore=face_restore
        )

        active_jobs[job_id]["progress"] = 30

        # Generate
        response = pipe.generate(request)

        active_jobs[job_id]["progress"] = 90

        if response.success:
            # Save result
            result_path = save_image(response.result_image, "result")
            result_id = result_path.stem

            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["progress"] = 100
            active_jobs[job_id]["result_url"] = f"/api/v2/results/{result_id}/image"
            active_jobs[job_id]["result_id"] = result_id
            active_jobs[job_id]["generation_time"] = response.generation_time
            active_jobs[job_id]["model_used"] = response.model_used
        else:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = response.error

        active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["updated_at"] = datetime.now().isoformat()


@router.get("/tryon/{job_id}", response_model=JobStatusV2)
async def get_tryon_status(job_id: str):
    """Get status of a try-on job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusV2(**active_jobs[job_id])


@router.get("/results/{result_id}/image")
async def get_result_image(result_id: str):
    """Get result image"""
    # Look for result file
    for ext in ['.png', '.jpg', '.jpeg']:
        result_path = RESULTS_DIR / f"{result_id}{ext}"
        if result_path.exists():
            return FileResponse(result_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Result not found")


@router.post("/extract-features")
async def extract_features(
    image: UploadFile = File(...)
):
    """
    Extract face and body features from an image

    Returns face embedding, skin colors, body keypoints, etc.
    """
    pipe = get_pipeline()

    # Save and load image
    image_path = await save_upload(image, "extract")
    img = load_image(image_path)

    # Extract features
    face_data = pipe.face_processor.extract_face_data(img)
    body_data = pipe.body_processor.extract_body_data(img)

    result = {
        "face": None,
        "body": None
    }

    if face_data:
        result["face"] = {
            "detected": True,
            "bbox": list(face_data.bbox),
            "angle": face_data.angle,
            "skin_color": list(face_data.skin_color),
            "has_embedding": face_data.embedding is not None
        }

    if body_data:
        result["body"] = {
            "detected": True,
            "pose_type": body_data.pose_type,
            "body_shape": body_data.body_shape,
            "skin_colors": {k: list(v) for k, v in body_data.skin_colors.items()},
            "keypoints_count": len(body_data.keypoints) if body_data.keypoints is not None else 0
        }

    # Clean up
    image_path.unlink()

    return result


@router.post("/process-garment")
async def process_garment(
    image: UploadFile = File(...),
    remove_background: bool = Form(True)
):
    """
    Process a garment image

    Removes background and classifies garment type
    """
    pipe = get_pipeline()

    # Save and load image
    image_path = await save_upload(image, "garment")
    img = load_image(image_path)

    # Process garment
    processed, mask, info = pipe.process_garment(img, remove_bg=remove_background)

    # Save processed garment
    result_path = save_image(processed, "garment_processed")

    return {
        "garment_url": f"/api/v2/results/{result_path.stem}/image",
        "garment_info": info
    }


@router.post("/batch-tryon")
async def batch_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    garment_images: List[UploadFile] = File(...),
    prompt: str = Form("person wearing garment, professional fashion photo"),
    steps: int = Form(15),
    preserve_face: bool = Form(True)
):
    """
    Generate try-ons for multiple garments at once
    """
    batch_id = generate_id()

    # Save person image
    person_path = await save_upload(person_image, "person")

    # Save garment images
    garment_paths = []
    for garment in garment_images:
        garment_path = await save_upload(garment, "garment")
        garment_paths.append(garment_path)

    # Initialize batch job
    active_jobs[batch_id] = {
        "job_id": batch_id,
        "status": "queued",
        "progress": 0,
        "total": len(garment_paths),
        "completed": 0,
        "results": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    # Run in background
    background_tasks.add_task(
        run_batch_tryon,
        batch_id,
        person_path,
        garment_paths,
        prompt,
        steps,
        preserve_face
    )

    return {"batch_id": batch_id, "status": "queued", "total": len(garment_paths)}


async def run_batch_tryon(
    batch_id: str,
    person_path: Path,
    garment_paths: List[Path],
    prompt: str,
    steps: int,
    preserve_face: bool
):
    """Background task for batch try-on"""
    try:
        pipe = get_pipeline()
        person_image = load_image(person_path)

        active_jobs[batch_id]["status"] = "processing"

        for i, garment_path in enumerate(garment_paths):
            garment_image = load_image(garment_path)

            request = TryOnRequest(
                person_image=person_image,
                garment_image=garment_image,
                prompt=prompt,
                steps=steps,
                preserve_face=preserve_face,
                seed=i  # Different seed for each
            )

            response = pipe.generate(request)

            if response.success:
                result_path = save_image(response.result_image, f"batch_{batch_id}")
                active_jobs[batch_id]["results"].append({
                    "index": i,
                    "result_url": f"/api/v2/results/{result_path.stem}/image",
                    "success": True
                })
            else:
                active_jobs[batch_id]["results"].append({
                    "index": i,
                    "error": response.error,
                    "success": False
                })

            active_jobs[batch_id]["completed"] = i + 1
            active_jobs[batch_id]["progress"] = int((i + 1) / len(garment_paths) * 100)
            active_jobs[batch_id]["updated_at"] = datetime.now().isoformat()

        active_jobs[batch_id]["status"] = "completed"

    except Exception as e:
        active_jobs[batch_id]["status"] = "failed"
        active_jobs[batch_id]["error"] = str(e)
        active_jobs[batch_id]["updated_at"] = datetime.now().isoformat()


# ============== Include router in main app ==============
def include_v2_router(app):
    """Include v2 router in main FastAPI app"""
    app.include_router(router)
