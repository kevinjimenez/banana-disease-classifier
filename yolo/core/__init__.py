"""
Módulo de integración YOLOv8 para clasificación de enfermedades de banano.
Usa Ultralytics YOLOv8 en modo clasificación (no detección).
"""

from .yolo_classifier import YOLOClassifier, YOLOTrainer
from .yolo_utils import convert_dataset_for_yolo, validate_yolo_dataset

__all__ = [
    "YOLOClassifier",
    "YOLOTrainer",
    "convert_dataset_for_yolo",
    "validate_yolo_dataset",
]
