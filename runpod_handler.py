"""
RunPod Serverless Handler for Comify Virtual Try-On
Handles inference requests in a serverless environment
"""

import runpod
import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent / "ai_engine"))

import numpy as np
from PIL import Image
import cv2

# Global pipeline instance (loaded once, reused across requests)
pipeline = None


def load_pipeline():
    """Load the AI pipeline (called once at cold start)"""
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading AI pipeline...")
    start = time.time()

    try:
        from ai_engine.pipeline import TryOnPipeline
        from ai_engine.config import AIConfig

        # Configure for GPU
        config = AIConfig(
            device="cuda",
            compute_backend="cuda",
            use_fp16=True,
            num_inference_steps=20,
            enable_upscaling=True,
            enable_face_restore=True
        )

        models_dir = Path(os.environ.get("MODELS_DIR", "/app/models"))
        pipeline = TryOnPipeline(config, models_dir)

        print(f"Pipeline loaded in {time.time() - start:.2f}s")
        return pipeline

    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        traceback.print_exc()
        raise


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (BGR)"""
    # Remove data URI prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to BGR for OpenCV
    if image.mode == "RGBA":
        image = image.convert("RGB")

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode_image_base64(image: np.ndarray, format: str = "PNG") -> str:
    """Encode numpy array (BGR) to base64 string"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=95)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def handler(job):
    """
    RunPod serverless handler

    Expected input format:
    {
        "input": {
            "action": "tryon" | "extract_features" | "create_profile",

            # For tryon:
            "person_image": "<base64>",
            "garment_image": "<base64>" (optional),
            "mask_image": "<base64>" (optional, auto-generated if not provided),
            "profile_id": "string" (optional),
            "preserve_face": true,
            "preserve_skin_tone": true,
            "upscale": false,
            "face_restore": true,
            "steps": 20,
            "denoise": 0.85,
            "prompt": "string",
            "negative_prompt": "string",
            "seed": -1,

            # For extract_features:
            "image": "<base64>",

            # For create_profile:
            "image": "<base64>",
            "name": "string"
        }
    }

    Returns:
    {
        "result_image": "<base64>",
        "seed": 12345,
        "generation_time": 5.2,
        "face_detected": true,
        "body_detected": true,
        ...
    }
    """

    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "tryon")

        # Load pipeline
        pipe = load_pipeline()

        if action == "tryon":
            return handle_tryon(pipe, job_input)
        elif action == "extract_features":
            return handle_extract_features(pipe, job_input)
        elif action == "create_profile":
            return handle_create_profile(pipe, job_input)
        elif action == "health":
            return handle_health(pipe)
        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def handle_tryon(pipe, job_input):
    """Handle try-on generation request"""
    from ai_engine.pipeline import TryOnRequest

    start_time = time.time()

    # Decode images
    person_image = decode_base64_image(job_input["person_image"])

    garment_image = None
    if job_input.get("garment_image"):
        garment_image = decode_base64_image(job_input["garment_image"])

    mask_image = None
    if job_input.get("mask_image"):
        mask_image = decode_base64_image(job_input["mask_image"])
        # Convert to grayscale if needed
        if len(mask_image.shape) == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Create request
    request = TryOnRequest(
        person_image=person_image,
        garment_image=garment_image,
        mask=mask_image,
        model_profile_id=job_input.get("profile_id"),
        prompt=job_input.get("prompt", "person wearing garment, professional photo"),
        negative_prompt=job_input.get("negative_prompt", "blurry, distorted, bad anatomy"),
        steps=job_input.get("steps", 20),
        seed=job_input.get("seed", -1),
        denoise=job_input.get("denoise", 0.85),
        preserve_face=job_input.get("preserve_face", True),
        preserve_skin_tone=job_input.get("preserve_skin_tone", True),
        upscale=job_input.get("upscale", False),
        face_restore=job_input.get("face_restore", True)
    )

    # Generate
    response = pipe.generate(request)

    if not response.success:
        return {"error": response.error or "Generation failed"}

    # Encode result
    result_base64 = encode_image_base64(response.result_image)

    return {
        "result_image": result_base64,
        "seed": response.seed,
        "generation_time": response.generation_time,
        "model_used": response.model_used,
        "face_detected": response.face_data is not None,
        "body_detected": response.body_data is not None
    }


def handle_extract_features(pipe, job_input):
    """Handle feature extraction request"""

    image = decode_base64_image(job_input["image"])

    # Extract face data
    face_data = pipe.face_processor.extract_face_data(image)

    # Extract body data
    body_data = pipe.body_processor.extract_body_data(image)

    result = {
        "face_detected": face_data is not None,
        "body_detected": body_data is not None
    }

    if face_data:
        result["face_bbox"] = list(face_data.bbox)
        result["face_angle"] = face_data.angle
        result["skin_color"] = list(face_data.skin_color)

    if body_data:
        result["body_shape"] = body_data.body_shape
        result["pose_type"] = body_data.pose_type
        result["skin_colors"] = {k: list(v) for k, v in body_data.skin_colors.items()}

    return result


def handle_create_profile(pipe, job_input):
    """Handle profile creation request"""

    image = decode_base64_image(job_input["image"])
    name = job_input.get("name", "Unnamed Profile")

    profile = pipe.create_profile_from_image(image, name)

    return {
        "profile_id": profile.id,
        "name": profile.name,
        "face_detected": profile.face_embedding is not None,
        "body_shape": profile.body_shape,
        "skin_colors": profile.skin_colors,
        "created_at": profile.created_at
    }


def handle_health(pipe):
    """Handle health check request"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipe is not None,
        "device": str(pipe.config.device) if pipe else "unknown"
    }


# For local testing
if __name__ == "__main__":
    # Test with a sample request
    import json

    # Create a simple test image (black square)
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gray

    # Encode to base64
    _, buffer = cv2.imencode('.png', test_image)
    test_base64 = base64.b64encode(buffer).decode('utf-8')

    test_job = {
        "input": {
            "action": "health"
        }
    }

    result = handler(test_job)
    print(json.dumps(result, indent=2))


# RunPod entry point
runpod.serverless.start({"handler": handler})
