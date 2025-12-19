#!/usr/bin/env python3
"""
Script de entrenamiento YOLOv8 para clasificación de enfermedades de banano.

Uso:
    python yolo/train.py --data_dir ./data --model_size s --epochs 100

Modelos disponibles:
    - n (nano):   1.4M params - Más rápido, menos preciso
    - s (small):  3.2M params - Balance velocidad/precisión  [RECOMENDADO]
    - m (medium): 6.4M params - Buena precisión
    - l (large):  16.5M params - Alta precisión
    - x (xlarge): 57.4M params - Máxima precisión, más lento
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
from datetime import datetime
import torch

try:
    from yolo.core.yolo_classifier import YOLOClassifier, YOLOTrainingConfig
    from yolo.core.yolo_utils import validate_yolo_dataset, print_dataset_summary
except ImportError as e:
    print(f"[ERROR] No se pudo importar módulo YOLO: {e}")
    print("Asegúrate de que el directorio 'yolo/' existe con los archivos correctos")
    sys.exit(1)


def detect_device():
    """
    Detecta automáticamente el mejor dispositivo disponible.

    Returns:
        str: 'cuda', 'mps', o 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU CUDA detectada: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"[INFO] Apple Silicon (MPS) detectado")
    else:
        device = "cpu"
        print(f"[INFO] Usando CPU (no se detectó GPU)")

    return device


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento YOLOv8 para clasificación de enfermedades de banano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Entrenamiento básico
  python yolo/train.py --data_dir ./data --model_size s --epochs 50
  
  # Entrenamiento con configuración avanzada
  python yolo/train.py --data_dir ./data --model_size m --epochs 100 \\
      --batch_size 64 --lr 0.001 --image_size 224
  
  # Prueba rápida
  python yolo/train.py --data_dir ./data --model_size n --epochs 5 --quick_test
        """,
    )

    # Datos
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con datos (estructura: train/val/test con subdirs de clase)",
    )

    # Modelo
    parser.add_argument(
        "--model_size",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Tamaño del modelo: n=nano, s=small, m=medium, l=large, x=xlarge",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Usar pesos preentrenados de ImageNet",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a checkpoint para reanudar entrenamiento",
    )

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate inicial")
    parser.add_argument("--image_size", type=int, default=224, help="Tamaño de imagen")

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        help="Optimizador a usar",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="Momentum (para SGD)"
    )

    # Augmentation
    parser.add_argument(
        "--no_augment", action="store_true", help="Desactivar augmentation"
    )
    parser.add_argument("--mixup", type=float, default=0.1, help="Mixup alpha (0=off)")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing"
    )

    # Regularización
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (detecta automáticamente), '0' para GPU 0, 'mps' para Apple Silicon, 'cpu' para CPU, '0,1' para multi-GPU",
    )
    parser.add_argument("--workers", type=int, default=8, help="Número de workers")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/yolo_classify",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nombre del experimento (default: auto-generado)",
    )

    # Modo rápido
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Modo prueba rápida (menos épocas, batch pequeño)",
    )

    # Export
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "torchscript", "tflite", "engine"],
        help="Exportar modelo después del entrenamiento",
    )

    args = parser.parse_args()

    # Detectar dispositivo si es 'auto'
    if args.device == "auto":
        args.device = detect_device()
        print(f"[INFO] Dispositivo seleccionado automáticamente: {args.device}")
    else:
        print(f"[INFO] Usando dispositivo especificado: {args.device}")

    # Validar dataset
    print("\n" + "=" * 70)
    print("VALIDANDO DATASET")
    print("=" * 70)

    validation = validate_yolo_dataset(args.data_dir)

    if not validation["valid"]:
        print("\n❌ Dataset inválido:")
        for error in validation["errors"]:
            print(f"   - {error}")
        sys.exit(1)

    if validation["warnings"]:
        print("\n⚠️  Advertencias:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")

    print(f"\n✅ Dataset válido")
    print(f"   Clases: {validation['classes']}")
    print(f"   Total imágenes: {validation['total_images']}")

    for split, info in validation["splits"].items():
        print(f"   {split}: {info['total_images']} imágenes")

    # Modo prueba rápida
    if args.quick_test:
        print("\n🚀 Modo prueba rápida activado")
        args.epochs = 5
        args.batch_size = min(16, args.batch_size)
        args.patience = 3

    # Nombre del experimento
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"yolov8{args.model_size}_{timestamp}"

    # Crear configuración
    config = YOLOTrainingConfig(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        augment=not args.no_augment,
        mixup=args.mixup,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.output_dir,
        name=args.name,
    )

    # Crear clasificador
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN")
    print("=" * 70)

    classifier = YOLOClassifier(
        model_size=args.model_size,
        num_classes=len(validation["classes"]),
        pretrained=args.pretrained,
    )

    classifier.print_info()

    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Augmentation: {not args.no_augment}")
    print(f"Device: {args.device}")
    print("=" * 70 + "\n")

    # Entrenar
    try:
        results = classifier.train(
            data_dir=args.data_dir, config=config, resume=args.resume is not None
        )

        print("\n✅ Entrenamiento completado!")

        # Evaluar en test si existe
        if "test" in validation["splits"]:
            print("\n[INFO] Evaluando en test set...")
            eval_metrics = classifier.evaluate(args.data_dir, split="test")

            print("\n" + "=" * 60)
            print("MÉTRICAS EN TEST SET")
            print("=" * 60)
            for metric, value in eval_metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
            print("=" * 60)

        # Exportar si se solicita
        if args.export:
            print(f"\n[INFO] Exportando modelo a {args.export}...")
            export_path = classifier.export(format=args.export, imgsz=args.image_size)
            print(f"✅ Modelo exportado: {export_path}")

        # Guardar resumen
        summary_path = Path(args.output_dir) / args.name / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "model": f"yolov8{args.model_size}-cls",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": args.image_size,
            "classes": validation["classes"],
            "num_classes": len(validation["classes"]),
            "total_images": validation["total_images"],
            "augmentation": not args.no_augment,
            "device": args.device,
            "timestamp": datetime.now().isoformat(),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📁 Resultados guardados en: {Path(args.output_dir) / args.name}")

    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
