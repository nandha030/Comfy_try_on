"""
Face Processing Module
Handles face detection, embedding extraction, and preservation
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from PIL import Image
import cv2

@dataclass
class FaceData:
    """Extracted face data for preservation"""
    embedding: np.ndarray  # 512-dim face embedding
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: np.ndarray  # 5 or 68 point landmarks
    angle: float  # face rotation angle
    skin_color: Tuple[int, int, int]  # RGB skin color
    mask: np.ndarray  # face mask


class FaceProcessor:
    """
    Face detection and embedding extraction using InsightFace
    """

    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.face_analyzer = None
        self.face_swapper = None
        self._initialized = False

    def initialize(self):
        """Load face analysis models"""
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Initialize face analyzer
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root=str(self.models_dir / "insightface"),
                providers=['CPUExecutionProvider'] if self.device == 'cpu'
                else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device != 'cpu' else -1)

            self._initialized = True
            print("Face processor initialized successfully")

        except ImportError:
            print("Warning: InsightFace not installed. Using fallback face detection.")
            self._initialized = True

        except Exception as e:
            print(f"Warning: Could not initialize face processor: {e}")
            self._initialized = True

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image

        Args:
            image: BGR numpy array

        Returns:
            List of face dictionaries with bbox, landmarks, embedding
        """
        self.initialize()

        if self.face_analyzer is None:
            return self._fallback_detect(image)

        try:
            faces = self.face_analyzer.get(image)
            results = []

            for face in faces:
                results.append({
                    'bbox': face.bbox.astype(int).tolist(),
                    'landmarks': face.landmark_2d_106 if hasattr(face, 'landmark_2d_106')
                    else face.landmark_3d_68[:, :2] if hasattr(face, 'landmark_3d_68')
                    else face.kps,
                    'embedding': face.embedding,
                    'age': face.age if hasattr(face, 'age') else None,
                    'gender': face.gender if hasattr(face, 'gender') else None,
                    'score': face.det_score
                })

            return results

        except Exception as e:
            print(f"Face detection error: {e}")
            return self._fallback_detect(image)

    def _fallback_detect(self, image: np.ndarray) -> List[Dict]:
        """Fallback face detection using OpenCV Haar cascades"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': [x, y, x + w, y + h],
                    'landmarks': None,
                    'embedding': None,
                    'score': 1.0
                })
            return results

        except Exception as e:
            print(f"Fallback face detection error: {e}")
            return []

    def extract_face_data(self, image: np.ndarray) -> Optional[FaceData]:
        """
        Extract comprehensive face data for preservation

        Args:
            image: BGR numpy array

        Returns:
            FaceData object or None if no face detected
        """
        faces = self.detect_faces(image)

        if not faces:
            return None

        # Use the largest/most confident face
        face = max(faces, key=lambda f: f.get('score', 0))

        # Extract skin color from face region
        bbox = face['bbox']
        face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        skin_color = self._extract_skin_color(face_region)

        # Create face mask
        mask = self._create_face_mask(image.shape[:2], bbox, face.get('landmarks'))

        # Calculate face angle
        angle = self._calculate_face_angle(face.get('landmarks'))

        return FaceData(
            embedding=face.get('embedding'),
            bbox=tuple(bbox),
            landmarks=face.get('landmarks'),
            angle=angle,
            skin_color=skin_color,
            mask=mask
        )

    def _extract_skin_color(self, face_region: np.ndarray) -> Tuple[int, int, int]:
        """Extract dominant skin color from face region"""
        if face_region.size == 0:
            return (200, 170, 150)  # Default skin tone

        try:
            # Convert to RGB
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

            # Get center region (more likely to be skin)
            h, w = face_rgb.shape[:2]
            center = face_rgb[h // 4:3 * h // 4, w // 4:3 * w // 4]

            if center.size == 0:
                center = face_rgb

            # Calculate mean color
            mean_color = np.mean(center.reshape(-1, 3), axis=0).astype(int)
            return tuple(mean_color)

        except Exception:
            return (200, 170, 150)

    def _create_face_mask(
        self,
        image_shape: Tuple[int, int],
        bbox: List[int],
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create a mask for the face region"""
        mask = np.zeros(image_shape, dtype=np.uint8)

        if landmarks is not None and len(landmarks) > 0:
            # Create mask from landmarks
            try:
                points = np.array(landmarks, dtype=np.int32)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 2)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 255)
            except:
                # Fallback to bbox
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        else:
            # Use bbox as mask
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)

        # Feather the edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def _calculate_face_angle(self, landmarks: Optional[np.ndarray]) -> float:
        """Calculate face rotation angle from landmarks"""
        if landmarks is None or len(landmarks) < 2:
            return 0.0

        try:
            # Use eye positions to calculate angle
            landmarks = np.array(landmarks)
            if len(landmarks) >= 5:
                left_eye = landmarks[0]
                right_eye = landmarks[1]
            else:
                left_eye = landmarks[0]
                right_eye = landmarks[-1]

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            return float(angle)

        except Exception:
            return 0.0

    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Cosine similarity
            dot = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot / (norm1 * norm2)
            return float((similarity + 1) / 2)  # Normalize to 0-1

        except Exception:
            return 0.0

    def align_face(self, image: np.ndarray, face_data: FaceData) -> np.ndarray:
        """Align face to standard position"""
        if face_data.angle == 0:
            return image

        try:
            h, w = image.shape[:2]
            center = ((face_data.bbox[0] + face_data.bbox[2]) // 2,
                      (face_data.bbox[1] + face_data.bbox[3]) // 2)

            M = cv2.getRotationMatrix2D(center, face_data.angle, 1.0)
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            return aligned

        except Exception:
            return image
