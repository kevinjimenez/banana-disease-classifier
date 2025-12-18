from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    class_name: str
    confidence: float
    class_id: int
    all_probabilities: Optional[Dict[str, float]] = None


class PredictionHistoryResponse(BaseModel):
    """Respuesta de historial de predicción desde DB."""
    id: int
    image_name: str
    predicted_class: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]]
    device_used: Optional[str]
    inference_time_ms: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionFeedbackCreate(BaseModel):
    """Schema para crear feedback."""
    prediction_id: int
    correct_class: Optional[str] = None
    is_correct: int = Field(..., ge=0, le=1)  # 0 o 1
    comment: Optional[str] = None


class PredictionFeedbackResponse(BaseModel):
    """Respuesta de feedback."""
    id: int
    prediction_id: int
    correct_class: Optional[str]
    is_correct: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ModelInfo(BaseModel):
    """Información del modelo."""
    model_name: str
    num_classes: int
    classes: list[str]
    device: str
    checkpoint_path: str


class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str
    model_loaded: bool
    device: str
    database_connected: bool


class StatsResponse(BaseModel):
    """Estadísticas del sistema."""
    total_predictions: int
    predictions_today: int
    avg_confidence: float
    most_common_class: str
    class_distribution: Dict[str, int]
