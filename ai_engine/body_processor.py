"""
Body Processing Module
Handles body pose detection, segmentation, and skin tone analysis
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from PIL import Image
import cv2

# Body keypoint definitions (COCO format)
BODY_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Body part connections for skeleton
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


@dataclass
class BodyData:
    """Extracted body data"""
    keypoints: np.ndarray  # 17x3 array (x, y, confidence)
    skeleton: List[Tuple[int, int]]  # Connections
    body_mask: np.ndarray  # Full body segmentation mask
    skin_regions: Dict[str, np.ndarray]  # Skin region masks
    skin_colors: Dict[str, Tuple[int, int, int]]  # Skin colors per region
    body_shape: str  # Body shape classification
    pose_type: str  # standing, sitting, etc.


class BodyProcessor:
    """
    Body pose detection and segmentation
    """

    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.pose_detector = None
        self.segmentation_model = None
        self._initialized = False

    def initialize(self):
        """Load body processing models"""
        if self._initialized:
            return

        try:
            # Try to load DWPose or OpenPose
            self._load_pose_model()
            self._load_segmentation_model()
            self._initialized = True
            print("Body processor initialized successfully")

        except Exception as e:
            print(f"Warning: Body processor initialization: {e}")
            self._initialized = True

    def _load_pose_model(self):
        """Load pose detection model"""
        try:
            import onnxruntime as ort

            model_path = self.models_dir / "dwpose" / "dw-ll_ucoco_384.onnx"
            if model_path.exists():
                providers = ['CPUExecutionProvider']
                if self.device == 'cuda':
                    providers = ['CUDAExecutionProvider'] + providers
                elif self.device == 'mps':
                    providers = ['CoreMLExecutionProvider'] + providers

                self.pose_detector = ort.InferenceSession(
                    str(model_path), providers=providers
                )
                print(f"Loaded DWPose model from {model_path}")
            else:
                print(f"DWPose model not found at {model_path}")

        except Exception as e:
            print(f"Could not load pose model: {e}")

    def _load_segmentation_model(self):
        """Load segmentation model (SAM or alternatives)"""
        try:
            # Try loading SAM
            sam_path = self.models_dir / "sam" / "sam_vit_b.pth"
            if not sam_path.exists():
                sam_path = self.models_dir / "sam" / "sam_vit_h.pth"

            if sam_path.exists():
                # SAM will be loaded on-demand due to memory
                self.sam_model_path = sam_path
                print(f"SAM model available at {sam_path}")
            else:
                print("SAM model not found, using fallback segmentation")

        except Exception as e:
            print(f"Could not load segmentation model: {e}")

    def detect_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect body pose keypoints

        Args:
            image: BGR numpy array

        Returns:
            17x3 numpy array of keypoints (x, y, confidence)
        """
        self.initialize()

        if self.pose_detector is not None:
            return self._detect_pose_dwpose(image)
        else:
            return self._detect_pose_fallback(image)

    def _detect_pose_dwpose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect pose using DWPose"""
        try:
            # Preprocess image
            h, w = image.shape[:2]
            input_size = 384

            # Resize maintaining aspect ratio
            scale = input_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))

            # Pad to square
            padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            padded[:new_h, :new_w] = resized

            # Normalize
            input_data = padded.astype(np.float32) / 255.0
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)

            # Run inference
            input_name = self.pose_detector.get_inputs()[0].name
            output = self.pose_detector.run(None, {input_name: input_data})

            # Process output to keypoints
            keypoints = self._process_pose_output(output[0], (w, h), scale)
            return keypoints

        except Exception as e:
            print(f"DWPose detection error: {e}")
            return self._detect_pose_fallback(image)

    def _detect_pose_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback pose detection using simple heuristics"""
        # Return None - caller should handle missing pose
        return None

    def _process_pose_output(
        self, output: np.ndarray, original_size: Tuple[int, int], scale: float
    ) -> np.ndarray:
        """Process model output to keypoints"""
        # This is a simplified version - actual implementation depends on model output format
        keypoints = np.zeros((17, 3))

        try:
            if output.shape[-1] >= 17:
                for i in range(17):
                    keypoints[i, 0] = output[0, i, 0] / scale
                    keypoints[i, 1] = output[0, i, 1] / scale
                    keypoints[i, 2] = output[0, i, 2] if output.shape[-1] > 2 else 1.0
        except:
            pass

        return keypoints

    def segment_body(self, image: np.ndarray, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment body from background

        Args:
            image: BGR numpy array
            keypoints: Optional pose keypoints to guide segmentation

        Returns:
            Binary mask of body region
        """
        self.initialize()

        # Try SAM-based segmentation
        if hasattr(self, 'sam_model_path') and self.sam_model_path.exists():
            return self._segment_with_sam(image, keypoints)

        # Fallback to GrabCut
        return self._segment_grabcut(image, keypoints)

    def _segment_with_sam(self, image: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """Segment using Segment Anything Model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Load SAM (cached if possible)
            if not hasattr(self, '_sam_predictor'):
                model_type = "vit_h" if "vit_h" in str(self.sam_model_path) else "vit_b"
                sam = sam_model_registry[model_type](checkpoint=str(self.sam_model_path))
                if self.device == 'cuda':
                    sam = sam.cuda()
                self._sam_predictor = SamPredictor(sam)

            # Set image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._sam_predictor.set_image(image_rgb)

            # Use keypoints as prompts if available
            if keypoints is not None and len(keypoints) > 0:
                valid_points = keypoints[keypoints[:, 2] > 0.5][:, :2]
                if len(valid_points) > 0:
                    masks, _, _ = self._sam_predictor.predict(
                        point_coords=valid_points,
                        point_labels=np.ones(len(valid_points)),
                        multimask_output=False
                    )
                    return (masks[0] * 255).astype(np.uint8)

            # Use center point as prompt
            h, w = image.shape[:2]
            masks, _, _ = self._sam_predictor.predict(
                point_coords=np.array([[w // 2, h // 2]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            return (masks[0] * 255).astype(np.uint8)

        except Exception as e:
            print(f"SAM segmentation error: {e}")
            return self._segment_grabcut(image, keypoints)

    def _segment_grabcut(self, image: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """Fallback segmentation using GrabCut"""
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            # Define initial rectangle
            if keypoints is not None and len(keypoints) > 0:
                valid = keypoints[keypoints[:, 2] > 0.3]
                if len(valid) > 0:
                    x_min = max(0, int(valid[:, 0].min()) - 20)
                    x_max = min(w, int(valid[:, 0].max()) + 20)
                    y_min = max(0, int(valid[:, 1].min()) - 20)
                    y_max = min(h, int(valid[:, 1].max()) + 20)
                    rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                else:
                    rect = (w // 4, h // 8, w // 2, 3 * h // 4)
            else:
                rect = (w // 4, h // 8, w // 2, 3 * h // 4)

            # Run GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Extract foreground
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            return mask2

        except Exception as e:
            print(f"GrabCut error: {e}")
            # Return center rectangle as fallback
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (w // 4, 0), (3 * w // 4, h), 255, -1)
            return mask

    def extract_skin_colors(
        self, image: np.ndarray, body_mask: np.ndarray, keypoints: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Extract skin colors from different body regions

        Returns:
            Dictionary mapping region names to RGB colors
        """
        h, w = image.shape[:2]
        skin_colors = {}

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define regions based on keypoints or heuristics
        regions = self._define_skin_regions(h, w, keypoints)

        for region_name, region_mask in regions.items():
            # Combine with body mask
            combined_mask = cv2.bitwise_and(region_mask, body_mask)

            if np.sum(combined_mask) > 100:
                # Extract pixels in region
                pixels = image_rgb[combined_mask > 0]
                if len(pixels) > 0:
                    # Filter for skin-like colors (simple heuristic)
                    skin_pixels = self._filter_skin_pixels(pixels)
                    if len(skin_pixels) > 0:
                        mean_color = np.mean(skin_pixels, axis=0).astype(int)
                        skin_colors[region_name] = tuple(mean_color)

        # Fallback if no skin detected
        if not skin_colors:
            skin_colors['body'] = (200, 170, 150)

        return skin_colors

    def _define_skin_regions(
        self, h: int, w: int, keypoints: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Define skin sampling regions"""
        regions = {}

        if keypoints is not None and len(keypoints) >= 17:
            # Face region (around nose)
            nose = keypoints[0]
            if nose[2] > 0.3:
                face_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(face_mask, (int(nose[0]), int(nose[1])), h // 8, 255, -1)
                regions['face'] = face_mask

            # Arms (wrist to elbow)
            for side, (wrist_idx, elbow_idx) in [('left', (9, 7)), ('right', (10, 8))]:
                wrist = keypoints[wrist_idx]
                elbow = keypoints[elbow_idx]
                if wrist[2] > 0.3 and elbow[2] > 0.3:
                    arm_mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array([[wrist[:2], elbow[:2]]], dtype=np.int32)
                    cv2.polylines(arm_mask, pts, False, 255, thickness=30)
                    regions[f'{side}_arm'] = arm_mask

        else:
            # Fallback: general body region
            body_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(body_mask, (w // 3, h // 4), (2 * w // 3, 3 * h // 4), 255, -1)
            regions['body'] = body_mask

        return regions

    def _filter_skin_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Filter pixels to keep only skin-like colors"""
        # Simple skin color filter in RGB
        # Skin typically has R > G > B and certain ranges
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

        # Basic skin color rules
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g) & (g > b) &
            (np.abs(r.astype(int) - g.astype(int)) > 10) &
            (r < 250) & (g < 230)
        )

        return pixels[skin_mask]

    def extract_body_data(self, image: np.ndarray) -> Optional[BodyData]:
        """
        Extract comprehensive body data

        Args:
            image: BGR numpy array

        Returns:
            BodyData object
        """
        self.initialize()

        # Detect pose
        keypoints = self.detect_pose(image)

        # Segment body
        body_mask = self.segment_body(image, keypoints)

        # Extract skin colors
        skin_colors = self.extract_skin_colors(image, body_mask, keypoints)

        # Classify pose type
        pose_type = self._classify_pose(keypoints)

        # Classify body shape (simplified)
        body_shape = self._classify_body_shape(keypoints, body_mask)

        return BodyData(
            keypoints=keypoints if keypoints is not None else np.zeros((17, 3)),
            skeleton=SKELETON_CONNECTIONS,
            body_mask=body_mask,
            skin_regions={},  # Can be populated if needed
            skin_colors=skin_colors,
            body_shape=body_shape,
            pose_type=pose_type
        )

    def _classify_pose(self, keypoints: Optional[np.ndarray]) -> str:
        """Classify pose type"""
        if keypoints is None:
            return "unknown"

        try:
            # Check if sitting (hip-knee angle)
            left_hip = keypoints[11]
            left_knee = keypoints[13]
            right_hip = keypoints[12]
            right_knee = keypoints[14]

            if left_hip[2] > 0.3 and left_knee[2] > 0.3:
                hip_knee_dist = abs(left_hip[1] - left_knee[1])
                if hip_knee_dist < 50:  # Knees roughly at hip level
                    return "sitting"

            # Check if lying down
            nose = keypoints[0]
            ankle = keypoints[15] if keypoints[15][2] > keypoints[16][2] else keypoints[16]
            if nose[2] > 0.3 and ankle[2] > 0.3:
                if abs(nose[1] - ankle[1]) < 100:  # Head and feet at similar height
                    return "lying"

            return "standing"

        except:
            return "standing"

    def _classify_body_shape(self, keypoints: Optional[np.ndarray], body_mask: np.ndarray) -> str:
        """Simple body shape classification"""
        # This is a placeholder - real implementation would be more sophisticated
        return "average"

    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw skeleton on image for visualization"""
        result = image.copy()

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                cv2.circle(result, (int(kp[0]), int(kp[1])), 5, color, -1)

        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(result, pt1, pt2, color, 2)

        return result
