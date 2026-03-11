# schemas.py
# Defines the shape of all API request/response data
# Using Pydantic — FastAPI uses this for automatic validation

from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    damage_type: str          # e.g. "dent", "scratch_crack"
    confidence: float         # 0.0 - 1.0
    severity_score: float     # 0 - 100
    severity_label: str       # Minor / Moderate / Severe
    bbox: List[int]           # [x1, y1, x2, y2]

class AnalysisResponse(BaseModel):
    image_id: str
    processing_time_ms: float
    total_damages: int
    overall_severity: str     # Minor / Moderate / Severe / No Damage
    overall_score: float      # 0 - 100
    detections: List[DetectionResult]
    heatmap_url: str
    annotated_url: str

class HealthResponse(BaseModel):
    status: str
    model: str
    dataset: str
    classes: List[str]