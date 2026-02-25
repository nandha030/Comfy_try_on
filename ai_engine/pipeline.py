"""
Complete Try-On Pipeline
Orchestrates all AI components for virtual try-on
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from PIL import Image
import cv2
import json
import time

from .config import AIConfig, load_config
from .face_processor import FaceProcessor, FaceData
from .body_processor import BodyProcessor, BodyData
from .tryon_engine import TryOnEngine, TryOnResult, GarmentProcessor


@dataclass
class ModelProfile:
    """Stored model profile for consistent results"""
    id: str
    name: str
    face_embedding: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None
    skin_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    body_shape: str = "average"
    reference_images: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "face_embedding": self.face_embedding.tolist() if self.face_embedding is not None else None,
            "skin_colors": self.skin_colors,
            "body_shape": self.body_shape,
            "reference_images": self.reference_images,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelProfile':
        if data.get("face_embedding"):
            data["face_embedding"] = np.array(data["face_embedding"])
        return cls(**data)


@dataclass
class TryOnRequest:
    """Request for try-on generation"""
    person_image: np.ndarray
    garment_image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    model_profile_id: Optional[str] = None
    target_pose: Optional[str] = None  # "original", "front", "side", "custom"
    pose_reference: Optional[np.ndarray] = None  # Custom pose reference image

    # Generation settings
    prompt: str = "person wearing garment, professional photo"
    negative_prompt: str = "blurry, distorted, bad anatomy"
    steps: int = 20
    seed: int = -1
    denoise: float = 0.85

    # Feature flags
    preserve_face: bool = True
    preserve_skin_tone: bool = True
    upscale: bool = False
    face_restore: bool = True


@dataclass
class TryOnResponse:
    """Response from try-on generation"""
    success: bool
    result_image: Optional[np.ndarray] = None
    result_path: Optional[str] = None
    seed: int = 0
    generation_time: float = 0.0
    model_used: str = ""
    error: Optional[str] = None

    # Extracted data (for saving profile)
    face_data: Optional[FaceData] = None
    body_data: Optional[BodyData] = None


class TryOnPipeline:
    """
    Main pipeline orchestrating all AI components
    """

    def __init__(self, config: Optional[AIConfig] = None, models_dir: Optional[Path] = None):
        self.config = config or load_config()
        self.models_dir = models_dir or self.config.models_dir

        # Initialize processors
        self.face_processor = FaceProcessor(self.models_dir, self.config.device)
        self.body_processor = BodyProcessor(self.models_dir, self.config.device)
        self.tryon_engine = TryOnEngine(self.models_dir, self.config.device)
        self.garment_processor = GarmentProcessor()

        # Model profiles storage
        self.profiles_dir = self.models_dir.parent / "data" / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, ModelProfile] = {}

        self._load_profiles()

    def _load_profiles(self):
        """Load saved model profiles"""
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = ModelProfile.from_dict(data)
                    self.profiles[profile.id] = profile
            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")

    def save_profile(self, profile: ModelProfile):
        """Save model profile"""
        self.profiles[profile.id] = profile
        profile_path = self.profiles_dir / f"{profile.id}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)

    def create_profile_from_image(
        self,
        image: np.ndarray,
        name: str,
        profile_id: Optional[str] = None
    ) -> ModelProfile:
        """
        Create a model profile from reference image

        Args:
            image: BGR reference image
            name: Name for the profile
            profile_id: Optional ID (generated if not provided)

        Returns:
            ModelProfile object
        """
        import uuid
        from datetime import datetime

        if profile_id is None:
            profile_id = uuid.uuid4().hex[:12]

        # Extract face data
        face_data = self.face_processor.extract_face_data(image)

        # Extract body data
        body_data = self.body_processor.extract_body_data(image)

        profile = ModelProfile(
            id=profile_id,
            name=name,
            face_embedding=face_data.embedding if face_data else None,
            face_landmarks=face_data.landmarks if face_data else None,
            skin_colors=body_data.skin_colors if body_data else {},
            body_shape=body_data.body_shape if body_data else "average",
            created_at=datetime.now().isoformat()
        )

        self.save_profile(profile)
        return profile

    def process_garment(
        self,
        garment_image: np.ndarray,
        remove_bg: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process garment image for try-on

        Args:
            garment_image: BGR garment image
            remove_bg: Whether to remove background

        Returns:
            (processed_garment, mask, garment_info)
        """
        if remove_bg:
            processed, mask = self.garment_processor.remove_background(garment_image)
        else:
            processed = garment_image
            mask = np.ones(garment_image.shape[:2], dtype=np.uint8) * 255

        info = self.garment_processor.classify_garment(processed)

        return processed, mask, info

    def generate(self, request: TryOnRequest) -> TryOnResponse:
        """
        Main generation method

        Args:
            request: TryOnRequest object

        Returns:
            TryOnResponse object
        """
        start_time = time.time()

        try:
            # Extract face data
            print("Extracting face data...")
            face_data = self.face_processor.extract_face_data(request.person_image)

            # Extract body data
            print("Extracting body data...")
            body_data = self.body_processor.extract_body_data(request.person_image)

            # Use profile data if specified
            if request.model_profile_id and request.model_profile_id in self.profiles:
                profile = self.profiles[request.model_profile_id]
                if face_data and profile.face_embedding is not None:
                    # Could use profile embedding for consistency
                    pass
                if profile.skin_colors:
                    body_data.skin_colors = profile.skin_colors

            # Process mask
            if request.mask is None:
                # Auto-generate mask from body data
                print("Auto-generating mask...")
                request.mask = self._generate_auto_mask(
                    request.person_image, body_data,
                    request.garment_image
                )

            # Generate try-on
            print(f"Generating try-on with {self.config.num_inference_steps} steps...")
            result = self.tryon_engine.generate_tryon(
                person_image=request.person_image,
                garment_image=request.garment_image if request.garment_image is not None
                else request.person_image,
                mask=request.mask,
                face_data=face_data if request.preserve_face else None,
                body_data=body_data if request.preserve_skin_tone else None,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps or self.config.num_inference_steps,
                seed=request.seed,
                denoise=request.denoise,
                preserve_face=request.preserve_face,
                preserve_skin_tone=request.preserve_skin_tone
            )

            # Optional upscaling
            if request.upscale and self.config.enable_upscaling:
                print("Upscaling result...")
                result.image = self._upscale(result.image)

            # Optional face restoration
            if request.face_restore and self.config.enable_face_restore:
                print("Restoring face quality...")
                result.image = self._restore_face_quality(result.image, face_data)

            generation_time = time.time() - start_time

            return TryOnResponse(
                success=True,
                result_image=result.image,
                seed=result.seed,
                generation_time=generation_time,
                model_used=result.model_used,
                face_data=face_data,
                body_data=body_data
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TryOnResponse(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )

    def _generate_auto_mask(
        self,
        person_image: np.ndarray,
        body_data: Optional[BodyData],
        garment_image: Optional[np.ndarray]
    ) -> np.ndarray:
        """Generate mask automatically based on garment type"""
        h, w = person_image.shape[:2]

        if body_data is not None and body_data.body_mask is not None:
            # Use body segmentation
            mask = body_data.body_mask.copy()

            # Exclude face region if we have keypoints
            if body_data.keypoints is not None:
                nose = body_data.keypoints[0]
                if nose[2] > 0.3:
                    # Create face exclusion circle
                    face_radius = h // 8
                    cv2.circle(mask, (int(nose[0]), int(nose[1])), face_radius, 0, -1)

            return mask

        # Fallback: create rectangular mask for upper body
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 255, -1)
        return mask

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using RealESRGAN"""
        try:
            # Try loading RealESRGAN
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = self.models_dir / "realesrgan" / "RealESRGAN_x4.pth"
            if not model_path.exists():
                return image

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            upsampler = RealESRGANer(
                scale=self.config.upscale_factor,
                model_path=str(model_path),
                model=model,
                device=self.config.device
            )

            output, _ = upsampler.enhance(image, outscale=self.config.upscale_factor)
            return output

        except Exception as e:
            print(f"Upscaling error: {e}")
            # Fallback to simple resize
            scale = self.config.upscale_factor
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    def _restore_face_quality(
        self,
        image: np.ndarray,
        face_data: Optional[FaceData]
    ) -> np.ndarray:
        """Enhance face quality using CodeFormer"""
        try:
            # CodeFormer integration would go here
            # For now, return original
            return image

        except Exception as e:
            print(f"Face restoration error: {e}")
            return image

    def generate_pose_variants(
        self,
        request: TryOnRequest,
        poses: List[str] = ["front", "side_left", "side_right", "back"]
    ) -> List[TryOnResponse]:
        """
        Generate multiple pose variants

        Args:
            request: Base TryOnRequest
            poses: List of pose names to generate

        Returns:
            List of TryOnResponse objects
        """
        results = []

        for pose in poses:
            pose_request = TryOnRequest(
                person_image=request.person_image,
                garment_image=request.garment_image,
                mask=request.mask,
                target_pose=pose,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                seed=request.seed + len(results) if request.seed != -1 else -1,
                preserve_face=request.preserve_face,
                preserve_skin_tone=request.preserve_skin_tone
            )

            result = self.generate(pose_request)
            results.append(result)

        return results

    def batch_generate(
        self,
        person_image: np.ndarray,
        garments: List[np.ndarray],
        **kwargs
    ) -> List[TryOnResponse]:
        """
        Generate try-ons for multiple garments

        Args:
            person_image: BGR person image
            garments: List of garment images
            **kwargs: Additional arguments for TryOnRequest

        Returns:
            List of TryOnResponse objects
        """
        results = []

        for i, garment in enumerate(garments):
            request = TryOnRequest(
                person_image=person_image,
                garment_image=garment,
                seed=kwargs.get('seed', -1) + i if kwargs.get('seed', -1) != -1 else -1,
                **{k: v for k, v in kwargs.items() if k != 'seed'}
            )

            result = self.generate(request)
            results.append(result)

        return results


# Convenience functions
def create_pipeline(config_path: Optional[Path] = None) -> TryOnPipeline:
    """Create and return a TryOnPipeline instance"""
    config = load_config(config_path) if config_path else load_config()
    return TryOnPipeline(config)


def quick_tryon(
    person_image_path: str,
    garment_image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Quick try-on from file paths

    Args:
        person_image_path: Path to person image
        garment_image_path: Optional path to garment image
        output_path: Optional path to save result
        **kwargs: Additional arguments

    Returns:
        Result image as numpy array
    """
    # Load images
    person = cv2.imread(person_image_path)
    garment = cv2.imread(garment_image_path) if garment_image_path else None

    # Create pipeline and generate
    pipeline = create_pipeline()
    request = TryOnRequest(
        person_image=person,
        garment_image=garment,
        **kwargs
    )

    response = pipeline.generate(request)

    if response.success and output_path:
        cv2.imwrite(output_path, response.result_image)

    return response.result_image if response.success else person
