"""
Script completo de análisis de dataset.
Identifica problemas de calidad, balance, y duplicados.

Uso:
    python analyze_dataset.py --data_dir ./data/train
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
from PIL import Image
import hashlib
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suprimir warnings de PIL para imágenes con problemas menores de formato
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def check_image_quality(img_path):
    """
    Verifica la calidad de una imagen.

    Returns:
        dict con métricas de calidad
    """
    try:
        img = Image.open(img_path)
        img.verify()  # Verificar integridad

        img = Image.open(img_path)  # Reabrir después de verify
        width, height = img.size

        # Convertir a numpy para análisis
        img_array = np.array(img)

        # Calcular métricas
        is_grayscale = len(img_array.shape) == 2 or img_array.shape[2] == 1
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)

        # Detectar si es muy oscura o muy clara
        is_too_dark = mean_brightness < 50
        is_too_bright = mean_brightness > 200

        # Detectar bajo contraste
        is_low_contrast = std_brightness < 20

        return {
            "valid": True,
            "size": (width, height),
            "format": img.format,
            "mode": img.mode,
            "is_grayscale": is_grayscale,
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright,
            "is_low_contrast": is_low_contrast,
            "file_size": img_path.stat().st_size,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def find_duplicates(image_paths):
    """Encuentra imágenes duplicadas usando hash MD5."""
    hashes = {}
    duplicates = []

    print("\n[INFO] Buscando duplicados...")
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(
                    {"original": hashes[file_hash], "duplicate": img_path}
                )
            else:
                hashes[file_hash] = img_path
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")

    return duplicates


def analyze_dataset(data_dir, output_dir="./dataset_analysis"):
    """
    Análisis completo del dataset.

    Args:
        data_dir: Directorio con estructura class_name/images.jpg
        output_dir: Directorio para guardar reportes
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("ANÁLISIS DE DATASET")
    print("=" * 70)
    print(f"Directorio: {data_dir}")

    # 1. Recolectar todas las imágenes
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images_by_class = defaultdict(list)
    all_images = []

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix in image_extensions:
                    images_by_class[class_name].append(img_path)
                    all_images.append(img_path)

    if not all_images:
        print("❌ No se encontraron imágenes!")
        return

    print(f"\n✅ Total imágenes encontradas: {len(all_images)}")
    print(f"✅ Total clases: {len(images_by_class)}")

    # 2. Análisis de distribución de clases
    print("\n" + "=" * 70)
    print("DISTRIBUCIÓN DE CLASES")
    print("=" * 70)

    class_counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
    total_images = sum(class_counts.values())

    print(f"\n{'Clase':<25} {'Cantidad':>10} {'Porcentaje':>12} {'Estado':>15}")
    print("-" * 70)

    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100

        # Estado
        if count < 100:
            status = "⚠️ Muy Pequeño"
        elif count < 500:
            status = "⚠️ Pequeño"
        elif count < 1000:
            status = "✅ Aceptable"
        else:
            status = "✅ Excelente"

        print(f"{cls:<25} {count:>10d} {percentage:>11.1f}% {status:>15s}")

    # Detectar desbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count

    print(f"\nRatio de desbalance: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 3:
        print("⚠️ Dataset DESBALANCEADO - Considera:")
        print("  - Recolectar más imágenes de clases minoritarias")
        print("  - Usar augmentation 'heavy' en clases pequeñas")
        print("  - Class weights están activos automáticamente")
    elif imbalance_ratio > 1.5:
        print("⚠️ Desbalance moderado - Class weights recomendados (ya activos)")
    else:
        print("✅ Dataset bien balanceado")

    # 3. Análisis de calidad de imágenes
    print("\n" + "=" * 70)
    print("ANÁLISIS DE CALIDAD DE IMÁGENES")
    print("=" * 70)

    quality_issues = {
        "corrupted": [],
        "too_dark": [],
        "too_bright": [],
        "low_contrast": [],
        "grayscale": [],
        "small_size": [],
    }

    sizes = []
    formats = Counter()

    print("\n[INFO] Analizando calidad de imágenes (esto puede tomar un momento)...")

    for i, img_path in enumerate(all_images):
        if (i + 1) % 100 == 0:
            print(f"  Procesadas: {i+1}/{len(all_images)}", end="\r")

        quality = check_image_quality(img_path)

        if not quality["valid"]:
            quality_issues["corrupted"].append(img_path)
            continue

        # Recolectar métricas
        sizes.append(quality["size"])
        formats[quality["format"]] += 1

        # Detectar problemas
        if quality["is_too_dark"]:
            quality_issues["too_dark"].append(img_path)
        if quality["is_too_bright"]:
            quality_issues["too_bright"].append(img_path)
        if quality["is_low_contrast"]:
            quality_issues["low_contrast"].append(img_path)
        if quality["is_grayscale"]:
            quality_issues["grayscale"].append(img_path)
        if quality["size"][0] < 224 or quality["size"][1] < 224:
            quality_issues["small_size"].append(img_path)

    print(f"  Procesadas: {len(all_images)}/{len(all_images)}")

    # Reporte de calidad
    print("\n📊 Problemas de Calidad Detectados:")
    print(f"  Corruptas:        {len(quality_issues['corrupted']):>6d}")
    print(f"  Muy oscuras:      {len(quality_issues['too_dark']):>6d}")
    print(f"  Muy claras:       {len(quality_issues['too_bright']):>6d}")
    print(f"  Bajo contraste:   {len(quality_issues['low_contrast']):>6d}")
    print(f"  En escala grises: {len(quality_issues['grayscale']):>6d}")
    print(f"  Tamaño <224px:    {len(quality_issues['small_size']):>6d}")

    # 4. Análisis de duplicados
    duplicates = find_duplicates(all_images)
    print(f"\n📊 Duplicados encontrados: {len(duplicates)}")

    # 5. Estadísticas de tamaño
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]

        print("\n" + "=" * 70)
        print("ESTADÍSTICAS DE TAMAÑO")
        print("=" * 70)
        print(
            f"Ancho  - Min: {min(widths):>4d}px | Max: {max(widths):>4d}px | Media: {np.mean(widths):>6.1f}px"
        )
        print(
            f"Alto   - Min: {min(heights):>4d}px | Max: {max(heights):>4d}px | Media: {np.mean(heights):>6.1f}px"
        )

        print(f"\nFormatos encontrados:")
        for fmt, count in formats.most_common():
            print(f"  {fmt}: {count}")

    # 6. Generar visualizaciones
    print("\n" + "=" * 70)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 70)

    # Gráfico de distribución de clases
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 6.1 Distribución de clases (barras)
    ax = axes[0, 0]
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    bars = ax.bar(classes, counts)

    # Colorear barras según estado
    for i, (cls, count) in enumerate(zip(classes, counts)):
        if count < 500:
            bars[i].set_color("red")
        elif count < 1000:
            bars[i].set_color("orange")
        else:
            bars[i].set_color("green")

    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de Imágenes")
    ax.set_title("Distribución de Clases")
    ax.tick_params(axis="x", rotation=45)

    # 6.2 Distribución de clases (pie)
    ax = axes[0, 1]
    ax.pie(counts, labels=classes, autopct="%1.1f%%", startangle=90)
    ax.set_title("Proporción de Clases")

    # 6.3 Distribución de tamaños
    ax = axes[1, 0]
    ax.hist(widths, bins=30, alpha=0.5, label="Ancho", color="blue")
    ax.hist(heights, bins=30, alpha=0.5, label="Alto", color="red")
    ax.axvline(224, color="green", linestyle="--", label="Tamaño target (224px)")
    ax.set_xlabel("Píxeles")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Tamaños de Imagen")
    ax.legend()

    # 6.4 Resumen de problemas
    ax = axes[1, 1]
    problem_labels = [
        "Corruptas",
        "Oscuras",
        "Claras",
        "Bajo contraste",
        "Grises",
        "Pequeñas",
    ]
    problem_counts = [
        len(quality_issues["corrupted"]),
        len(quality_issues["too_dark"]),
        len(quality_issues["too_bright"]),
        len(quality_issues["low_contrast"]),
        len(quality_issues["grayscale"]),
        len(quality_issues["small_size"]),
    ]
    ax.barh(problem_labels, problem_counts)
    ax.set_xlabel("Cantidad")
    ax.set_title("Problemas de Calidad Detectados")

    plt.tight_layout()

    plot_path = output_dir / "dataset_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Gráficos guardados en: {plot_path}")

    # 7. Guardar reporte detallado
    report_path = output_dir / "dataset_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DETALLADO DE ANÁLISIS DE DATASET\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total imágenes: {len(all_images)}\n")
        f.write(f"Total clases: {len(images_by_class)}\n\n")

        f.write("Distribución por clase:\n")
        for cls, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"  {cls}: {count}\n")

        f.write(f"\nRatio de desbalance: {imbalance_ratio:.2f}x\n\n")

        if quality_issues["corrupted"]:
            f.write("Imágenes corruptas:\n")
            for img in quality_issues["corrupted"]:
                f.write(f"  {img}\n")

        if duplicates:
            f.write("\nDuplicados:\n")
            for dup in duplicates:
                f.write(f"  {dup['original']} == {dup['duplicate']}\n")

    print(f"✅ Reporte guardado en: {report_path}")

    # 8. Recomendaciones finales
    print("\n" + "=" * 70)
    print("RECOMENDACIONES")
    print("=" * 70)

    total_images_per_class = np.mean(list(class_counts.values()))

    if total_images_per_class < 500:
        print("\n⚠️ Dataset PEQUEÑO (<500 imgs/clase)")
        print("Recomendaciones:")
        print("  1. Usa: --model convnext_tiny")
        print("  2. Usa: --augmentation heavy")
        print("  3. Reduce epochs a 30-50")
        print("  4. Precision esperada: 85-90%")
    elif total_images_per_class < 1000:
        print("\n✅ Dataset MEDIANO (500-1000 imgs/clase)")
        print("Recomendaciones:")
        print("  1. Usa: --model convnext_small o convnext_base")
        print("  2. Usa: --augmentation medium")
        print("  3. Epochs: 50-75")
        print("  4. Precision esperada: 90-93%")
    else:
        print("\n✅ Dataset GRANDE (>1000 imgs/clase)")
        print("Recomendaciones:")
        print("  1. Usa: --model convnext_base o convnext_large")
        print("  2. Usa: --augmentation medium")
        print("  3. Epochs: 75-100")
        print("  4. Precision esperada: 93-98%")

    if len(quality_issues["corrupted"]) > 0:
        print(
            f"\n❌ Eliminar {len(quality_issues['corrupted'])} imágenes corruptas antes de entrenar"
        )

    if len(duplicates) > 0:
        print(f"\n⚠️ Considera eliminar {len(duplicates)} duplicados")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Análisis completo de dataset para entrenamiento"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con datos (estructura: class/images.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset_analysis",
        help="Directorio para guardar reportes",
    )

    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.output_dir)

    print("\n✅ Análisis completado!")
    print(f"📁 Resultados en: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Análisis interrumpido")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
