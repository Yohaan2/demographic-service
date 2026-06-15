from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class FaceDetectionResult:
    """Class representing a single face detection result."""
    def __init__(
        self, 
        bbox: np.ndarray,      # [x1, y1, x2, y2]
        confidence: float,     # Confidence score [0.0, 1.0]
        landmarks: np.ndarray = None  # Optional facial keypoints [5, 2]
    ):
        self.bbox = bbox.astype(np.float32)
        self.confidence = float(confidence)
        self.landmarks = landmarks.astype(np.float32) if landmarks is not None else None


class FaceDetector(ABC):
    """Abstract base class for all face detection strategies."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Loads the detector model from the specified file path."""
        pass
        
    @abstractmethod
    def detect(self, image: np.ndarray, threshold: float = 0.5) -> List[FaceDetectionResult]:
        """
        Detects faces in the given image.
        
        Args:
            image: A NumPy array in BGR format (OpenCV standard).
            threshold: Confidence threshold for detections.
            
        Returns:
            A list of FaceDetectionResult instances.
        """
        pass
