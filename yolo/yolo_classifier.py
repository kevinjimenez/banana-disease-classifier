"""
Clasificador YOLOv8 para enfermedades de banano.
Usa Ultralytics YOLOv8-cls (modo clasificación).

Ventajas vs CNN tradicional:
- Más ligero y rápido
- Fácil despliegue (ONNX, TensorRT, etc.)
- Buenos resultados con datos limitados
- Training pipeline optimizado
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass
import torch
import numpy as np

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[WARN] Ultralytics no instalado. Ejecuta: pip install ultralytics")


@dataclass
class YOLOTrainingConfig:
    """Configuración específica para entrenamiento YOLOv8."""

    # Modelo base
    model_size: str = "n"  # n (nano), s (small), m (medium), l (large), x (xlarge)

    # Entrenamiento
    epochs: int = 100
    batch_size: int = 32
    image_size: int = 224

    # Optimizador
    optimizer: str = "AdamW"  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.937

    # Scheduler
    lr_scheduler: str = "cos"  # linear, cos
    warmup_epochs: float = 3.0

    # Augmentation
    augment: bool = True
    hsv_h: float = 0.015  # Hue
    hsv_s: float = 0.7  # Saturation
    hsv_v: float = 0.4  # Value
    degrees: float = 10.0  # Rotation
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    flipud: float = 0.0  # No flip vertical para plantas
    fliplr: float = 0.5  # Flip horizontal OK
    mosaic: float = 0.0  # Desactivar mosaic para clasificación
    mixup: float = 0.1

    # Regularización
    dropout: float = 0.0
    label_smoothing: float = 0.1

    # Hardware
    device: str = "0"  # GPU ID o "cpu"
    workers: int = 8

    # Logging
    project: str = "runs/yolo_classify"
    name: str = "banana_disease"
    exist_ok: bool = True

    # Early stopping
    patience: int = 20

    # Guardado
    save: bool = True
    save_period: int = 10

    def to_dict(self) -> Dict:
        """Convierte a diccionario para YOLO."""
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "optimizer": self.optimizer,
            "lr0": self.learning_rate,
            "lrf": 0.01,  # Final LR factor
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "warmup_epochs": self.warmup_epochs,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "device": self.device,
            "workers": self.workers,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "patience": self.patience,
            "save": self.save,
            "save_period": self.save_period,
        }


class YOLOClassifier:
    """
    Wrapper para YOLOv8 en modo clasificación.

    Modelos disponibles:
    - yolov8n-cls: Nano (1.4M params) - Más rápido
    - yolov8s-cls: Small (3.2M params)
    - yolov8m-cls: Medium (6.4M params)
    - yolov8l-cls: Large (16.5M params)
    - yolov8x-cls: XLarge (57.4M params) - Más preciso

    Args:
        model_size: 'n', 's', 'm', 'l', 'x'
        num_classes: Número de clases
        pretrained: Usar pesos preentrenados de ImageNet
    """

    # Tamaños de modelo disponibles con parámetros aproximados
    MODEL_SIZES = {
        "n": {"name": "yolov8n-cls", "params": "1.4M", "speed": "fastest"},
        "s": {"name": "yolov8s-cls", "params": "3.2M", "speed": "fast"},
        "m": {"name": "yolov8m-cls", "params": "6.4M", "speed": "medium"},
        "l": {"name": "yolov8l-cls", "params": "16.5M", "speed": "slow"},
        "x": {"name": "yolov8x-cls", "params": "57.4M", "speed": "slowest"},
    }

    def __init__(
        self,
        model_size: str = "s",
        num_classes: int = 4,
        pretrained: bool = True,
        model_path: Optional[str] = None,
    ):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics no instalado. Ejecuta: pip install ultralytics"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Cargar modelo
        if model_path and Path(model_path).exists():
            # Cargar modelo entrenado
            self.model = YOLO(model_path)
            print(f"[INFO] Modelo cargado desde: {model_path}")
        else:
            # Crear nuevo modelo
            model_name = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES["s"])["name"]
            if pretrained:
                self.model = YOLO(f"{model_name}.pt")
            else:
                # Cargar sin pesos (solo arquitectura)
                self.model = YOLO(f"{model_name}.yaml")
            print(f"[INFO] Modelo {model_name} inicializado (pretrained={pretrained})")

        self.classes: Optional[List[str]] = None
        self.trained = False

    def train(
        self,
        data_dir: str,
        config: Optional[YOLOTrainingConfig] = None,
        resume: bool = False,
    ) -> Dict:
        """
        Entrena el modelo en el dataset.

        Args:
            data_dir: Directorio con estructura train/val/test con subdirectorios de clase
            config: Configuración de entrenamiento
            resume: Reanudar entrenamiento previo

        Returns:
            Resultados del entrenamiento
        """
        if config is None:
            config = YOLOTrainingConfig()

        # Actualizar tamaño de modelo en config
        config.model_size = self.model_size

        # Obtener parámetros
        train_args = config.to_dict()
        train_args["data"] = data_dir
        train_args["resume"] = resume

        print("\n" + "=" * 70)
        print("ENTRENAMIENTO YOLOv8 CLASIFICACIÓN")
        print("=" * 70)
        print(f"Modelo: {self.MODEL_SIZES[self.model_size]['name']}")
        print(f"Dataset: {data_dir}")
        print(f"Épocas: {config.epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Image size: {config.image_size}")
        print("=" * 70 + "\n")

        # Entrenar
        results = self.model.train(**train_args)

        self.trained = True

        # Obtener nombres de clases
        if hasattr(self.model, "names"):
            self.classes = list(self.model.names.values())

        return results

    def predict(
        self,
        source: Union[str, Path, np.ndarray, List],
        conf_threshold: float = 0.25,
        save: bool = False,
        show: bool = False,
    ) -> List[Dict]:
        """
        Realiza predicciones.

        Args:
            source: Ruta a imagen(es), directorio, o array numpy
            conf_threshold: Umbral de confianza
            save: Guardar resultados
            show: Mostrar resultados

        Returns:
            Lista de predicciones con clase y confianza
        """
        results = self.model.predict(
            source=source, conf=conf_threshold, save=save, show=show
        )

        predictions = []
        for r in results:
            probs = r.probs
            pred = {
                "class_id": int(probs.top1),
                "class_name": (
                    r.names[probs.top1] if hasattr(r, "names") else str(probs.top1)
                ),
                "confidence": float(probs.top1conf),
                "top5_classes": probs.top5.tolist() if hasattr(probs, "top5") else [],
                "top5_confs": (
                    probs.top5conf.tolist() if hasattr(probs, "top5conf") else []
                ),
            }
            predictions.append(pred)

        return predictions

    def evaluate(self, data_dir: str, split: str = "test") -> Dict:
        """
        Evalúa el modelo en un split del dataset.

        Args:
            data_dir: Directorio del dataset
            split: Split a evaluar ('val' o 'test')

        Returns:
            Métricas de evaluación
        """
        results = self.model.val(data=data_dir, split=split)

        metrics = {
            "accuracy_top1": results.top1 if hasattr(results, "top1") else None,
            "accuracy_top5": results.top5 if hasattr(results, "top5") else None,
            "fitness": results.fitness if hasattr(results, "fitness") else None,
        }

        return metrics

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 224,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """
        Exporta modelo a formato optimizado.

        Args:
            format: 'onnx', 'torchscript', 'tflite', 'engine' (TensorRT)
            imgsz: Tamaño de imagen
            simplify: Simplificar modelo (ONNX)
            dynamic: Shapes dinámicos

        Returns:
            Ruta al modelo exportado
        """
        export_path = self.model.export(
            format=format, imgsz=imgsz, simplify=simplify, dynamic=dynamic
        )

        print(f"[INFO] Modelo exportado a: {export_path}")
        return export_path

    def get_model_info(self) -> Dict:
        """Retorna información del modelo."""
        info = self.MODEL_SIZES.get(self.model_size, {})

        # Contar parámetros reales
        if hasattr(self.model, "model"):
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.model.parameters() if p.requires_grad
            )
        else:
            total_params = 0
            trainable_params = 0

        return {
            "architecture": info.get("name", "unknown"),
            "estimated_params": info.get("params", "unknown"),
            "actual_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "speed_category": info.get("speed", "unknown"),
            "trained": self.trained,
            "num_classes": self.num_classes,
            "classes": self.classes,
        }

    def print_info(self):
        """Imprime información del modelo."""
        info = self.get_model_info()

        print("\n" + "=" * 60)
        print("INFORMACIÓN DEL MODELO YOLOv8")
        print("=" * 60)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")


class YOLOTrainer:
    """
    Clase para orquestar entrenamiento YOLOv8 con configuración avanzada.
    Permite comparaciones justas con otros modelos (CNN, transformers).
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "runs/yolo",
        model_sizes: List[str] = None,
    ):
        """
        Args:
            data_dir: Directorio del dataset
            output_dir: Directorio de salida
            model_sizes: Lista de tamaños a evaluar ['n', 's', 'm', 'l', 'x']
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_sizes = model_sizes or ["s", "m"]
        self.results: Dict[str, Dict] = {}

    def run_experiments(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        image_size: int = 224,
        repetitions: int = 3,
    ) -> Dict:
        """
        Ejecuta experimentos con múltiples configuraciones.

        Args:
            epochs: Número de épocas
            batch_size: Tamaño de batch
            image_size: Tamaño de imagen
            repetitions: Repeticiones por configuración (para promediar)

        Returns:
            Resultados agregados
        """
        all_results = {}

        for size in self.model_sizes:
            print(f"\n{'='*70}")
            print(f"EXPERIMENTO: YOLOv8{size.upper()}-cls")
            print(f"{'='*70}")

            size_results = []

            for rep in range(repetitions):
                print(f"\n[Rep {rep+1}/{repetitions}]")

                config = YOLOTrainingConfig(
                    model_size=size,
                    epochs=epochs,
                    batch_size=batch_size,
                    image_size=image_size,
                    project=str(self.output_dir),
                    name=f"yolov8{size}_rep{rep+1}",
                )

                classifier = YOLOClassifier(model_size=size)

                try:
                    results = classifier.train(str(self.data_dir), config)

                    # Evaluar
                    eval_metrics = classifier.evaluate(str(self.data_dir))

                    size_results.append(
                        {
                            "rep": rep + 1,
                            "train_results": results,
                            "eval_metrics": eval_metrics,
                        }
                    )

                except Exception as e:
                    print(f"[ERROR] Rep {rep+1} falló: {e}")
                    continue

            # Agregar resultados
            if size_results:
                top1_accs = [
                    r["eval_metrics"]["accuracy_top1"]
                    for r in size_results
                    if r["eval_metrics"]["accuracy_top1"] is not None
                ]

                all_results[f"yolov8{size}"] = {
                    "model_size": size,
                    "repetitions": len(size_results),
                    "mean_top1_accuracy": np.mean(top1_accs) if top1_accs else None,
                    "std_top1_accuracy": np.std(top1_accs) if top1_accs else None,
                    "detailed_results": size_results,
                }

        self.results = all_results
        return all_results

    def save_results(self, filename: str = "yolo_results.json"):
        """Guarda resultados a JSON."""
        import json

        output_path = self.output_dir / filename

        # Convertir a serializable
        def make_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "__dict__"):
                return str(obj)
            return obj

        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: make_serializable(v)
                    for k, v in value.items()
                    if k != "detailed_results"  # Excluir detalles muy largos
                }
            else:
                serializable_results[key] = make_serializable(value)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"[INFO] Resultados guardados en: {output_path}")

    def print_summary(self):
        """Imprime resumen de resultados."""
        print("\n" + "=" * 70)
        print("RESUMEN DE EXPERIMENTOS YOLOv8")
        print("=" * 70)

        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Repeticiones: {results['repetitions']}")
            if results["mean_top1_accuracy"] is not None:
                print(
                    f"  Top-1 Accuracy: {results['mean_top1_accuracy']:.4f} ± {results['std_top1_accuracy']:.4f}"
                )

        print("\n" + "=" * 70)


# Ejemplo de uso
if __name__ == "__main__":
    # Verificar instalación
    if not ULTRALYTICS_AVAILABLE:
        print("Instala ultralytics: pip install ultralytics")
        sys.exit(1)

    # Ejemplo básico
    print("=== YOLOv8 Classification Demo ===")

    # Crear clasificador
    classifier = YOLOClassifier(model_size="s")
    classifier.print_info()

    # Para entrenar (descomentar con datos reales):
    # config = YOLOTrainingConfig(epochs=50, batch_size=32)
    # results = classifier.train("./data", config)
