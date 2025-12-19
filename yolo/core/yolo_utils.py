"""
Utilidades para preparar y validar datasets para YOLOv8 clasificación.

YOLOv8 clasificación espera esta estructura:
    dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── class2/
    │       └── img3.jpg
    ├── val/
    │   └── ...
    └── test/
        └── ...

Esta estructura es COMPATIBLE con ImageFolder de PyTorch,
por lo que el mismo dataset funciona para ambos frameworks.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

try:
    from PIL import Image
except ImportError:
    Image = None

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_yolo_dataset(data_dir: str) -> Dict:
    """
    Valida que el dataset tenga la estructura correcta para YOLOv8.

    Args:
        data_dir: Directorio raíz del dataset

    Returns:
        Diccionario con estadísticas y errores
    """
    data_dir = Path(data_dir)

    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "splits": {},
        "classes": set(),
        "total_images": 0,
    }

    # Verificar existencia
    if not data_dir.exists():
        result["valid"] = False
        result["errors"].append(f"Directorio no existe: {data_dir}")
        return result

    # Verificar splits
    required_splits = ["train"]
    optional_splits = ["val", "test"]

    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            result["valid"] = False
            result["errors"].append(f"Split requerido no existe: {split}")

    # Analizar cada split
    for split in required_splits + optional_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        split_info = {"exists": True, "classes": {}, "total_images": 0}

        # Obtener clases (subdirectorios)
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            result["warnings"].append(
                f"Split '{split}' no tiene subdirectorios de clase"
            )
            continue

        for class_dir in class_dirs:
            class_name = class_dir.name
            result["classes"].add(class_name)

            # Contar imágenes
            images = [
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTS
            ]

            split_info["classes"][class_name] = len(images)
            split_info["total_images"] += len(images)
            result["total_images"] += len(images)

        result["splits"][split] = split_info

    # Verificar consistencia de clases entre splits
    all_classes = list(result["classes"])
    for split, info in result["splits"].items():
        if "classes" in info:
            split_classes = set(info["classes"].keys())
            missing = set(all_classes) - split_classes
            if missing:
                result["warnings"].append(
                    f"Split '{split}' no tiene las clases: {missing}"
                )

    # Verificar balance de clases
    if "train" in result["splits"]:
        class_counts = result["splits"]["train"]["classes"]
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            if imbalance_ratio > 5:
                result["warnings"].append(
                    f"Desbalance significativo de clases (ratio {imbalance_ratio:.1f}x). "
                    "Considera usar class weights o data augmentation."
                )

    result["classes"] = sorted(list(result["classes"]))

    return result


def convert_dataset_for_yolo(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_files: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Convierte un dataset a la estructura de YOLOv8.

    Si el source_dir ya tiene estructura train/val/test, los copia directamente.
    Si es un directorio plano con subcarpetas de clase, hace el split.

    Args:
        source_dir: Directorio fuente
        output_dir: Directorio de salida
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        copy_files: True=copiar, False=crear symlinks
        seed: Semilla para reproducibilidad

    Returns:
        Estadísticas de la conversión
    """
    random.seed(seed)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
    ), "Los ratios deben sumar 1.0"

    stats = {
        "source": str(source_dir),
        "output": str(output_dir),
        "mode": None,
        "classes": [],
        "splits": {},
    }

    # Determinar estructura del source
    has_splits = all(
        (source_dir / split).exists()
        for split in ["train"]  # Al menos train debe existir
    )

    if has_splits:
        # Copiar estructura existente
        stats["mode"] = "copy_existing"
        _copy_existing_structure(source_dir, output_dir, copy_files, stats)
    else:
        # Crear splits desde directorio plano
        stats["mode"] = "create_splits"
        _create_splits_from_flat(
            source_dir,
            output_dir,
            train_ratio,
            val_ratio,
            test_ratio,
            copy_files,
            stats,
        )

    return stats


def _copy_existing_structure(
    source_dir: Path, output_dir: Path, copy_files: bool, stats: Dict
):
    """Copia estructura existente de splits."""

    for split in ["train", "val", "test"]:
        split_src = source_dir / split
        if not split_src.exists():
            continue

        split_dst = output_dir / split
        split_dst.mkdir(parents=True, exist_ok=True)

        split_stats = {"classes": {}, "total": 0}

        for class_dir in split_src.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in stats["classes"]:
                stats["classes"].append(class_name)

            class_dst = split_dst / class_name
            class_dst.mkdir(parents=True, exist_ok=True)

            count = 0
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTS:
                    dst_path = class_dst / img_path.name
                    if copy_files:
                        shutil.copy2(img_path, dst_path)
                    else:
                        if dst_path.exists():
                            dst_path.unlink()
                        dst_path.symlink_to(img_path.absolute())
                    count += 1

            split_stats["classes"][class_name] = count
            split_stats["total"] += count

        stats["splits"][split] = split_stats

    print(f"[INFO] Estructura copiada a {output_dir}")


def _create_splits_from_flat(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    copy_files: bool,
    stats: Dict,
):
    """Crea splits desde directorio plano con subdirectorios de clase."""

    # Recolectar todas las imágenes por clase
    class_images: Dict[str, List[Path]] = {}

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        # Saltar directorios de split si existen
        if class_dir.name in ["train", "val", "test"]:
            continue

        class_name = class_dir.name
        images = [
            f
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        ]

        if images:
            class_images[class_name] = images
            stats["classes"].append(class_name)

    if not class_images:
        raise ValueError(f"No se encontraron clases con imágenes en {source_dir}")

    # Crear splits
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        stats["splits"][split] = {"classes": {}, "total": 0}

    # Dividir imágenes por clase
    for class_name, images in class_images.items():
        random.shuffle(images)
        n = len(images)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits_data = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, split_images in splits_data.items():
            class_dst = output_dir / split / class_name
            class_dst.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dst_path = class_dst / img_path.name
                if copy_files:
                    shutil.copy2(img_path, dst_path)
                else:
                    if dst_path.exists():
                        dst_path.unlink()
                    dst_path.symlink_to(img_path.absolute())

            stats["splits"][split]["classes"][class_name] = len(split_images)
            stats["splits"][split]["total"] += len(split_images)

    print(f"[INFO] Splits creados en {output_dir}")


def print_dataset_summary(stats: Dict):
    """Imprime resumen del dataset."""
    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET")
    print("=" * 60)
    print(f"Directorio: {stats.get('output', stats.get('source', 'N/A'))}")
    print(f"Modo: {stats.get('mode', 'N/A')}")
    print(f"Clases: {stats.get('classes', [])}")

    print("\nDistribución por split:")
    for split, info in stats.get("splits", {}).items():
        print(f"\n  {split.upper()}:")
        print(f"    Total: {info.get('total', 0)} imágenes")
        for class_name, count in info.get("classes", {}).items():
            print(f"    - {class_name}: {count}")

    print("=" * 60 + "\n")


def calculate_class_weights(data_dir: str, split: str = "train") -> Dict[str, float]:
    """
    Calcula pesos de clase basados en frecuencia inversa.
    Útil para datasets desbalanceados.

    Args:
        data_dir: Directorio del dataset
        split: Split a analizar

    Returns:
        Diccionario {clase: peso}
    """
    data_dir = Path(data_dir) / split

    if not data_dir.exists():
        raise ValueError(f"Split no existe: {data_dir}")

    class_counts = {}
    total = 0

    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        count = len(
            [
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTS
            ]
        )

        class_counts[class_dir.name] = count
        total += count

    if total == 0:
        return {}

    # Calcular pesos: peso = total / (n_clases * count_clase)
    n_classes = len(class_counts)
    weights = {
        name: total / (n_classes * count) if count > 0 else 0
        for name, count in class_counts.items()
    }

    return weights


def verify_images(data_dir: str, fix_corrupt: bool = False) -> Dict:
    """
    Verifica integridad de imágenes en el dataset.

    Args:
        data_dir: Directorio del dataset
        fix_corrupt: Si True, elimina imágenes corruptas

    Returns:
        Reporte de verificación
    """
    if Image is None:
        raise ImportError("PIL requerido: pip install Pillow")

    data_dir = Path(data_dir)

    report = {"total_checked": 0, "valid": 0, "corrupt": [], "fixed": []}

    for img_path in data_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue

        report["total_checked"] += 1

        try:
            with Image.open(img_path) as img:
                img.verify()
            report["valid"] += 1
        except Exception as e:
            report["corrupt"].append({"path": str(img_path), "error": str(e)})

            if fix_corrupt:
                img_path.unlink()
                report["fixed"].append(str(img_path))

    print(f"[INFO] Verificadas {report['total_checked']} imágenes")
    print(f"[INFO] Válidas: {report['valid']}")
    print(f"[INFO] Corruptas: {len(report['corrupt'])}")

    if fix_corrupt and report["fixed"]:
        print(f"[INFO] Eliminadas: {len(report['fixed'])}")

    return report


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Utilidades para preparar datasets YOLOv8"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando")

    # Validate
    val_parser = subparsers.add_parser("validate", help="Validar dataset")
    val_parser.add_argument("data_dir", help="Directorio del dataset")

    # Convert
    conv_parser = subparsers.add_parser("convert", help="Convertir dataset")
    conv_parser.add_argument("source_dir", help="Directorio fuente")
    conv_parser.add_argument("output_dir", help="Directorio de salida")
    conv_parser.add_argument("--train-ratio", type=float, default=0.7)
    conv_parser.add_argument("--val-ratio", type=float, default=0.15)
    conv_parser.add_argument("--symlinks", action="store_true", help="Usar symlinks")

    # Verify
    ver_parser = subparsers.add_parser("verify", help="Verificar imágenes")
    ver_parser.add_argument("data_dir", help="Directorio del dataset")
    ver_parser.add_argument("--fix", action="store_true", help="Eliminar corruptas")

    args = parser.parse_args()

    if args.command == "validate":
        result = validate_yolo_dataset(args.data_dir)
        print(f"\nDataset válido: {result['valid']}")
        if result["errors"]:
            print("\nErrores:")
            for e in result["errors"]:
                print(f"  - {e}")
        if result["warnings"]:
            print("\nAdvertencias:")
            for w in result["warnings"]:
                print(f"  - {w}")
        print_dataset_summary(
            {"splits": result["splits"], "classes": result["classes"]}
        )

    elif args.command == "convert":
        stats = convert_dataset_for_yolo(
            args.source_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            copy_files=not args.symlinks,
        )
        print_dataset_summary(stats)

    elif args.command == "verify":
        report = verify_images(args.data_dir, fix_corrupt=args.fix)

    else:
        parser.print_help()
