"""
Modelos de base de datos para el historial de predicciones.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from database.connection import Base


class Prediction(Base):
    """
    Tabla para almacenar historial de predicciones.

    Campos:
        - id: ID único
        - image_name: Nombre del archivo
        - image_hash: Hash de la imagen (para detectar duplicados)
        - predicted_class: Clase predicha
        - confidence: Confianza de la predicción (0-1)
        - all_probabilities: JSON con todas las probabilidades
        - device_used: Device usado (cpu/mps/cuda)
        - inference_time_ms: Tiempo de inferencia en milisegundos
        - created_at: Timestamp de creación
        - metadata: Campos adicionales (JSON)
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    image_hash = Column(String(64), index=True)  # SHA256 hash
    predicted_class = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    all_probabilities = Column(JSON)
    device_used = Column(String(20))
    inference_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    extra_metadata = Column(JSON)  # Campos extras (user_id, location, etc.)

    def __repr__(self):
        return f"<Prediction(id={self.id}, class={self.predicted_class}, confidence={self.confidence:.2f})>"


class PredictionFeedback(Base):
    """
    Tabla para almacenar feedback de usuarios sobre predicciones.

    Útil para reentrenamiento y mejora continua del modelo.
    """

    __tablename__ = "prediction_feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, index=True)  # Referencia a Prediction
    correct_class = Column(String(50))  # Clase correcta según usuario
    is_correct = Column(Integer)  # 1=correcto, 0=incorrecto
    comment = Column(Text)  # Comentarios del usuario
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    extra_metadata = Column(JSON)

    def __repr__(self):
        return f"<Feedback(prediction_id={self.prediction_id}, is_correct={self.is_correct})>"


class ModelMetrics(Base):
    """
    Tabla para almacenar métricas agregadas del modelo.

    Útil para monitoreo y análisis de rendimiento en producción.
    """

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), index=True)
    total_predictions = Column(Integer)
    avg_confidence = Column(Float)
    class_distribution = Column(JSON)  # Conteo por clase
    avg_inference_time_ms = Column(Float)
    device_used = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelMetrics(date={self.date}, predictions={self.total_predictions})>"
