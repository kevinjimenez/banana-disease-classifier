#!/usr/bin/env python3
"""
Genera comando de entrenamiento óptimo basado en análisis de dataset.
NO usa YAML, genera comando directamente.

Uso:
    python generate_train_command.py \
        --data_dir ./data/train \
        --device mps \
        --experiment_name mi_experimento \
        --output train_command.sh
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
from collections import Counter
import torch


def count_images_per_class(data_dir):
    """Cuenta imágenes por clase."""
    data_dir = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    class_counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            count = sum(1 for f in class_dir.iterdir() if f.suffix in image_extensions)
            if count > 0:
                class_counts[class_dir.name] = count

    return class_counts


def get_gpu_memory():
    """Obtiene memoria GPU disponible en GB."""
    if not torch.cuda.is_available():
        return 0

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return gpu_mem


def generate_train_command(
    data_dir, device="auto", experiment_name=None, no_wandb=True, checkpoint_dir=None
):
    """
    Genera comando de train.py basado en análisis de dataset.

    Args:
        data_dir: Directorio con datos
        device: Device a usar (auto, cuda, mps, cpu)
        experiment_name: Nombre del experimento
        no_wandb: Desactivar wandb

    Returns:
        Comando como string
    """
    # Analizar dataset
    class_counts = count_images_per_class(data_dir)

    if not class_counts:
        raise ValueError(f"No se encontraron imágenes en {data_dir}")

    total_images = sum(class_counts.values())
    avg_images_per_class = total_images / len(class_counts)
    min_images = min(class_counts.values())
    max_images = max(class_counts.values())
    imbalance_ratio = max_images / min_images

    # Detectar GPU si device es auto
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_memory = get_gpu_memory()
        elif torch.backends.mps.is_available():
            device = "mps"
            gpu_memory = 16  # Estimado para Mac
        else:
            device = "cpu"
            gpu_memory = 0
    else:
        gpu_memory = (
            get_gpu_memory() if device == "cuda" else 16 if device == "mps" else 0
        )

    print("=" * 70)
    print("ANÁLISIS DE DATASET")
    print("=" * 70)
    print(f"\nDataset:")
    print(f"  Total imágenes: {total_images}")
    print(f"  Clases: {len(class_counts)}")
    for cls, count in sorted(class_counts.items()):
        print(f"    - {cls}: {count} imágenes")
    print(f"  Promedio imgs/clase: {avg_images_per_class:.0f}")
    print(f"  Ratio desbalance: {imbalance_ratio:.2f}x")

    print(f"\nHardware:")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {gpu_memory:.1f} GB")
    elif device == "mps":
        print(f"  Apple Silicon (MPS)")

    # ===== SELECCIÓN DE MODELO =====
    if avg_images_per_class < 500:
        # Dataset pequeño
        model = "convnext_tiny"
        epochs = 30
        augmentation = "heavy"
        expected_acc = "85-90%"
    elif avg_images_per_class < 1000:
        # Dataset mediano
        model = "convnext_small"
        epochs = 50
        augmentation = "medium"
        expected_acc = "90-93%"
    elif avg_images_per_class < 2000:
        # Dataset grande
        model = "convnext_base"
        epochs = 75
        augmentation = "medium"
        expected_acc = "93-96%"
    else:
        # Dataset muy grande
        model = "convnext_large"
        epochs = 100
        augmentation = "medium"
        expected_acc = "96-98%"

    # ===== BATCH SIZE SEGÚN GPU =====
    if device == "cpu":
        batch_size = 8
        model = "convnext_tiny"  # Forzar modelo pequeño en CPU
    elif device == "mps":
        # Apple Silicon
        if gpu_memory < 16:
            batch_size = 16
        elif gpu_memory < 24:
            batch_size = 32
        else:
            batch_size = 48
    elif gpu_memory < 6:
        # GPU pequeña (<6GB)
        batch_size = 8
        model = "convnext_tiny"
    elif gpu_memory < 8:
        # GPU mediana (6-8GB)
        batch_size = 16
        if model == "convnext_large":
            model = "convnext_base"
    elif gpu_memory < 12:
        # GPU buena (8-12GB) - RTX 4060, 3060
        batch_size = 32
        if model == "convnext_large":
            batch_size = 16
    else:
        # GPU excelente (>12GB)
        if model == "convnext_large":
            batch_size = 32
        else:
            batch_size = 64

    # ===== LEARNING RATE =====
    if avg_images_per_class < 500:
        lr = 5e-5
    elif avg_images_per_class < 2000:
        lr = 1e-4
    else:
        lr = 8e-5

    # Nombre del experimento
    if not experiment_name:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"pipeline_auto_{timestamp}"

    print("\n" + "=" * 70)
    print("CONFIGURACIÓN GENERADA")
    print("=" * 70)
    print(f"\nModelo:")
    print(f"  Arquitectura: {model}")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Augmentation: {augmentation}")

    print(f"\nPrecisión esperada: {expected_acc}")
    print("=" * 70 + "\n")

    # ===== CONSTRUIR COMANDO =====
    cmd_parts = [
        "python scripts/train.py \\",
        f"    --data_dir {data_dir} \\",
        f"    --model {model} \\",
        f"    --batch_size {batch_size} \\",
        f"    --epochs {epochs} \\",
        f"    --lr {lr} \\",
        f"    --augmentation {augmentation} \\",
        f"    --device {device} \\",
        f"    --experiment_name {experiment_name}",
    ]

    # Agregar checkpoint_dir si se especifica
    if checkpoint_dir:
        cmd_parts[-1] += " \\"
        cmd_parts.append(f"    --checkpoint_dir {checkpoint_dir}")

    if no_wandb:
        cmd_parts[-1] += " \\"
        cmd_parts.append("    --no_wandb")

    return "\n".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Genera comando de entrenamiento óptimo (sin YAML)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con datos de entrenamiento",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device para entrenamiento",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Nombre del experimento"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=True,
        help="Desactivar wandb (default: True)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directorio para guardar checkpoints",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Guardar comando en archivo"
    )

    args = parser.parse_args()

    # Generar comando
    command = generate_train_command(
        data_dir=args.data_dir,
        device=args.device,
        experiment_name=args.experiment_name,
        no_wandb=args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Imprimir
    print("\n" + "=" * 70)
    print("COMANDO DE ENTRENAMIENTO")
    print("=" * 70)
    print()
    print(command)
    print()
    print("=" * 70)

    # Guardar si se especificó output
    if args.output:
        with open(args.output, "w") as f:
            f.write(command + "\n")
        print(f"\n✅ Comando guardado en: {args.output}")
        print(f"\nPara ejecutar:")
        print(f"  bash {args.output}")
    else:
        print("\nPara ejecutar, copia y pega el comando de arriba.")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Cancelado")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
