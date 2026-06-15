from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class AgeGenderResult:
    """Class representing the result of age and gender estimation."""
    def __init__(self, age: float, gender: str, gender_confidence: float):
        self.age = age
        self.gender = gender  # 'male' or 'female'
        self.gender_confidence = float(gender_confidence)


class AgeGenderModel(ABC):
    """Abstract base class for age and gender estimation models (Strategy Pattern)."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Loads the age and gender model from the specified path."""
        pass
        
    @abstractmethod
    def estimate(self, face_crop: np.ndarray) -> AgeGenderResult:
        """
        Estimates the age and gender of a normalized/aligned face crop.
        
        Args:
            face_crop: An aligned face image of fixed size (BGR format).
            
        Returns:
            An AgeGenderResult containing predicted age, gender, and gender confidence.
        """
        pass
