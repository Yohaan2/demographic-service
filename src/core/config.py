import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Core Pipeline settings
    MODEL_TYPE: str = "mivolo"  # options: mivolo | insightface
    DETECTOR: str = "scrfd"      # options: scrfd
    
    # Models path
    DETECTOR_MODEL_PATH: str = "models/weights/scrfd_2.5g_bnkps.onnx"
    AGE_GENDER_MODEL_PATH: str = "models/weights/mivolo_v2.onnx"
    INSIGHTFACE_MODEL_PATH: str = "models/weights/genderage.onnx"
    
    # Trackers configuration
    MAX_TRACKERS: int = 500
    TRACKER_TTL: int = 300  # seconds to keep an inactive tracker in memory
    
    # Aggregation window (sliding window)
    AGGREGATION_WINDOW: int = 10  # number of samples: 5, 10, 20, 30
    
    # Crop size Configuration
    CROP_SIZE: int = 224  # 112 or 224
    
    # Backpressure and Rate Limiting
    MAX_PENDING_REQUESTS: int = 500
    
    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


# Instancia única de configuración (Singleton)
settings = Settings()
