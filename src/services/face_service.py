import cv2
import numpy as np
from typing import Tuple, Optional
from src.core.config import settings
from src.core.logging import logger

# Coordenadas ideales de referencia para 112x112 (InsightFace / SCRFD Standard)
REFERENCE_LANDMARKS_112 = np.array([
    [30.2946, 51.6963],  # Ojo izquierdo
    [65.5318, 51.5014],  # Ojo derecho
    [48.0252, 71.7366],  # Nariz
    [33.5493, 92.3655],  # Esquina izquierda de la boca
    [62.7299, 92.2041]   # Esquina derecha de la boca
], dtype=np.float32)


class FaceService:
    """Service to handle facial image processing operations including alignment and normalized cropping."""

    @staticmethod
    def get_reference_landmarks(target_size: int) -> np.ndarray:
        """Scales ideal facial landmark coordinates to the target cropping size."""
        scale = target_size / 112.0
        return REFERENCE_LANDMARKS_112 * scale

    def align_and_crop(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray, 
        landmarks: Optional[np.ndarray], 
        target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Aligns and crops a face from the source image.
        If landmarks are provided, performs similarity transform (alignment).
        Otherwise, falls back to a clean box crop with safety padding.
        
        Args:
            image: Full original BGR image.
            bbox: Bounding box as [x1, y1, x2, y2].
            landmarks: Optional 5 keypoints matrix [5, 2].
            target_size: Final output dimensions (defaults to config.CROP_SIZE).
            
        Returns:
            A clean BGR NumPy array representing the normalized face crop.
        """
        if target_size is None:
            target_size = settings.CROP_SIZE
            
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Scenario A: Precise Alignment using 5-point facial landmarks
        if landmarks is not None and landmarks.shape == (5, 2):
            try:
                ref_pts = self.get_reference_landmarks(target_size)
                # Calculate optimal similarity transformation matrix (rigid: scale, rotation, translation)
                tfm, inliers = cv2.estimateAffinePartial2D(landmarks, ref_pts)
                
                if tfm is not None:
                    # Apply warp transformation to generate aligned crop
                    aligned_crop = cv2.warpAffine(image, tfm, (target_size, target_size), flags=cv2.INTER_CUBIC)
                    return aligned_crop
                
            except Exception as e:
                logger.warning("Similarity transform failed; falling back to direct cropped box", error=str(e))
                
        # Scenario B: Fallback/Bounding Box Crop (with expanded context padding)
        # Models like MiVOLO benefit immensely from seeing hair and upper chest context.
        # We expand the crop by 25% symmetrically.
        w = x2 - x1
        h = y2 - y1
        
        # Center of bounding box
        cx = x1 + w // 2
        cy = y1 + h // 2
        
        # Determine crop side length with 25% padding
        side = int(max(w, h) * 1.25)
        
        # Source box coordinates (clamped to image boundaries to avoid padding with black pixels)
        nx1 = max(0, cx - side // 2)
        ny1 = max(0, cy - side // 2)
        nx2 = min(w_img, cx + side // 2)
        ny2 = min(h_img, cy + side // 2)
        
        crop = image[ny1:ny2, nx1:nx2]
        
        # Resize to targeted dimension
        if crop.size > 0:
            return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            # Absolute fallback if bounding box is outside image
            # Return an empty but valid placeholder to prevent pipeline failure
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)
