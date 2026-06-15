from pydantic import BaseModel, Field
from typing import List


class FaceDemographics(BaseModel):
    """Demographics and spatial tracking information for a single detected face."""
    track_id: int = Field(..., description="Unique track ID assigned to this face trajectory.")
    gender: str = Field(..., description="Estimated gender: 'male' or 'female'.")
    gender_confidence: float = Field(..., description="Gender estimation confidence score [0.0, 1.0].")
    age: int = Field(..., description="Estimated age in years.")
    age_range: str = Field(..., description="Marketing/sociological age bracket (e.g. '25-34').")
    confidence: float = Field(..., description="Tracking confidence score [0.0, 1.0].")
    bbox: List[int] = Field(..., description="Bounding Box coordinates [x1, y1, x2, y2] in pixels.")


class AnalyzeResponse(BaseModel):
    """Root demographic analytics response schema."""
    camera_id: str = Field(..., description="The unique camera source identifier.")
    processing_time_ms: int = Field(..., description="Total pipeline frame execution latency in milliseconds.")
    faces: List[FaceDemographics] = Field(..., description="List of all tracked faces and their demographical predictions.")
