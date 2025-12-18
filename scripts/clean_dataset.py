"""
Script para limpiar y mejorar el dataset automáticamente.

Funciones:
- Eliminar imágenes corruptas
- Eliminar duplicados
- Mover imágenes con problemas de calidad a carpeta de revisión
- Balancear clases (opcional)
- Generar reporte de cambios

Uso:
    # Modo dry-run (solo muestra qué haría)
    python clean_dataset.py --data_dir ./data/train --dry_run

    # Modo real (hace los cambios)
    python clean_dataset.py --data_dir ./data/train

    # Con balanceo de clases
    python clean_dataset.py --data_dir ./data/train --balance_classes
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import shutil
import hashlib
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import warnings

# Suprimir warnings de PIL para imágenes con problemas menores de formato
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


def check_image(img_path):
    """
    Verifica si una imagen es válida y su calidad.

    Returns:
        dict con status y problemas detectados
    """
    result = {
        "valid": True,
        "corrupted": False,
        "too_dark": False,
        "too_bright": False,
        "low_contrast": False,
        "too_small": False,
        "error": None,
    }

    try:
        # Intentar abrir y verificar
        img = Image.open(img_path)
        img.verify()

        # Re-abrir para analizar (verify() cierra el file)
        img = Image.open(img_path)

        # Check tamaño mínimo
        width, height = img.size
        if width < 100 or height < 100:
            result["too_small"] = True
            result["valid"] = False

        # Convertir a array para análisis
        img_array = np.array(img.convert("RGB"))

        # Análisis de brillo y contraste
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)

        # Umbrales
        if mean_brightness < 30:
            result["too_dark"] = True

        if mean_brightness > 225:
            result["too_bright"] = True

        if std_brightness < 15:
            result["low_contrast"] = True

    except Exception as e:
        result["valid"] = False
        result["corrupted"] = True
        result["error"] = str(e)

    return result


def compute_hash(file_path):
    """Calcula MD5 hash de un archivo."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(image_paths):
    """
    Encuentra imágenes duplicadas usando hash MD5.

    Returns:
        dict: {hash: [list of duplicate paths]}
    """
    hashes = defaultdict(list)

    print("\n[INFO] Buscando duplicados...")
    for img_path in tqdm(image_paths, desc="Calculando hashes"):
        try:
            file_hash = compute_hash(img_path)
            hashes[file_hash].append(img_path)
        except Exception as e:
            print(f"⚠️ Error hasheando {img_path}: {e}")

    # Filtrar solo duplicados
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}

    return duplicates


def clean_dataset(
    data_dir, dry_run=True, balance_classes=False, remove_quality_issues=False
):
    """
    Limpia el dataset según configuración.

    Args:
        data_dir: Directorio con el dataset
        dry_run: Si True, solo muestra qué haría sin hacer cambios
        balance_classes: Si True, balancea las clases
        remove_quality_issues: Si True, elimina imgs con problemas de calidad
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"❌ Directorio no existe: {data_dir}")
        return

    # Crear directorios de salida
    review_dir = data_dir.parent / f"{data_dir.name}_review"
    backup_dir = data_dir.parent / f"{data_dir.name}_backup"

    print("=" * 70)
    print("LIMPIEZA DE DATASET")
    print("=" * 70)
    print(f"\nDirectorio: {data_dir}")
    print(f"Modo: {'DRY RUN (no hace cambios)' if dry_run else 'REAL (hace cambios)'}")
    print(f"Balance de clases: {'Sí' if balance_classes else 'No'}")
    print(f"Eliminar problemas de calidad: {'Sí' if remove_quality_issues else 'No'}")

    # Encontrar todas las imágenes
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    all_images = []

    for ext in image_extensions:
        all_images.extend(data_dir.rglob(f"*{ext}"))

    print(f"\nTotal imágenes encontradas: {len(all_images)}")

    # Organizar por clase
    class_images = defaultdict(list)
    for img_path in all_images:
        # Asumir estructura: data_dir/class_name/image.jpg
        class_name = img_path.parent.name
        class_images[class_name].append(img_path)

    print(f"Clases encontradas: {len(class_images)}")
    for class_name, images in class_images.items():
        print(f"  • {class_name}: {len(images)} imágenes")

    # Estadísticas de limpieza
    stats = {
        "total_images": len(all_images),
        "corrupted": [],
        "duplicates": [],
        "quality_issues": [],
        "too_small": [],
        "removed_for_balance": [],
    }

    # 1. VERIFICAR CORRUPTAS Y CALIDAD
    print("\n" + "=" * 70)
    print("FASE 1: VERIFICACIÓN DE CALIDAD")
    print("=" * 70)

    corrupted = []
    quality_issues = []
    too_small = []

    for img_path in tqdm(all_images, desc="Verificando imágenes"):
        result = check_image(img_path)

        if result["corrupted"]:
            corrupted.append((img_path, result["error"]))
            stats["corrupted"].append(str(img_path))
        elif result["too_small"]:
            too_small.append(img_path)
            stats["too_small"].append(str(img_path))
        elif result["too_dark"] or result["too_bright"] or result["low_contrast"]:
            issues = []
            if result["too_dark"]:
                issues.append("muy_oscura")
            if result["too_bright"]:
                issues.append("muy_brillante")
            if result["low_contrast"]:
                issues.append("bajo_contraste")

            quality_issues.append((img_path, issues))
            stats["quality_issues"].append({"path": str(img_path), "issues": issues})

    print(f"\n❌ Imágenes corruptas: {len(corrupted)}")
    print(f"📏 Imágenes muy pequeñas (<100px): {len(too_small)}")
    print(f"⚠️ Imágenes con problemas de calidad: {len(quality_issues)}")

    # 2. BUSCAR DUPLICADOS
    print("\n" + "=" * 70)
    print("FASE 2: DETECCIÓN DE DUPLICADOS")
    print("=" * 70)

    # Excluir las ya marcadas como corruptas o muy pequeñas
    valid_images = [
        img
        for img in all_images
        if img not in [c[0] for c in corrupted] and img not in too_small
    ]

    duplicates = find_duplicates(valid_images)

    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())
    print(f"\n🔍 Grupos de duplicados encontrados: {len(duplicates)}")
    print(f"📊 Imágenes duplicadas totales: {duplicate_count}")

    if len(duplicates) > 0:
        print("\nEjemplos de duplicados:")
        for i, (hash_val, paths) in enumerate(list(duplicates.items())[:3]):
            print(f"\nGrupo {i+1} ({len(paths)} copias):")
            for p in paths[:3]:
                print(f"  • {p}")
            if len(paths) > 3:
                print(f"  ... y {len(paths) - 3} más")

    # 3. BALANCEO DE CLASES (si se solicita)
    to_remove_for_balance = []

    if balance_classes:
        print("\n" + "=" * 70)
        print("FASE 3: BALANCEO DE CLASES")
        print("=" * 70)

        # Calcular clase minoritaria
        class_counts = {cls: len(imgs) for cls, imgs in class_images.items()}
        min_count = min(class_counts.values())

        print(f"\nClase minoritaria: {min_count} imágenes")
        print("\nBalanceando a este número...")

        for class_name, images in class_images.items():
            current_count = len(images)

            if current_count > min_count:
                # Elegir aleatoriamente cuáles remover
                np.random.seed(42)
                valid_class_images = [
                    img
                    for img in images
                    if img not in [c[0] for c in corrupted]
                    and img not in too_small
                    and not any(img in dup_group for dup_group in duplicates.values())
                ]

                to_remove_count = current_count - min_count
                if to_remove_count > 0 and len(valid_class_images) >= min_count:
                    remove_indices = np.random.choice(
                        len(valid_class_images),
                        size=min(to_remove_count, len(valid_class_images) - min_count),
                        replace=False,
                    )

                    to_remove = [valid_class_images[i] for i in remove_indices]
                    to_remove_for_balance.extend(to_remove)
                    stats["removed_for_balance"].extend([str(p) for p in to_remove])

                    print(
                        f"  • {class_name}: {current_count} → {min_count} (-{len(to_remove)})"
                    )

    # 4. APLICAR CAMBIOS
    print("\n" + "=" * 70)
    print("FASE 4: APLICAR CAMBIOS")
    print("=" * 70)

    if dry_run:
        print("\n⚠️ DRY RUN - No se harán cambios reales")
    else:
        # Crear backup
        print(f"\n[INFO] Creando backup en: {backup_dir}")
        if not backup_dir.exists():
            shutil.copytree(data_dir, backup_dir)
            print("✅ Backup creado")

        # Crear directorio de revisión
        review_dir.mkdir(exist_ok=True)

    # Eliminar corruptas y muy pequeñas
    to_delete = corrupted + [(img, "too_small") for img in too_small]

    print(f"\n🗑️ Eliminando {len(to_delete)} imágenes corruptas/muy pequeñas...")
    for img_path, reason in to_delete:
        if dry_run:
            print(f"  [DRY RUN] Eliminaría: {img_path}")
        else:
            try:
                img_path.unlink()
                print(f"  ✅ Eliminada: {img_path}")
            except Exception as e:
                print(f"  ❌ Error eliminando {img_path}: {e}")

    # Manejar duplicados (conservar el primero, eliminar el resto)
    print(f"\n🔍 Procesando {len(duplicates)} grupos de duplicados...")
    for hash_val, dup_paths in duplicates.items():
        # Conservar el primero, eliminar el resto
        to_keep = dup_paths[0]
        to_remove = dup_paths[1:]

        for img_path in to_remove:
            stats["duplicates"].append(str(img_path))

            if dry_run:
                print(f"  [DRY RUN] Eliminaría duplicado: {img_path}")
            else:
                try:
                    img_path.unlink()
                    print(f"  ✅ Duplicado eliminado: {img_path}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")

    # Mover problemas de calidad a carpeta de revisión
    if remove_quality_issues and len(quality_issues) > 0:
        print(
            f"\n📋 Moviendo {len(quality_issues)} imágenes con problemas de calidad..."
        )

        for img_path, issues in quality_issues:
            if dry_run:
                print(
                    f"  [DRY RUN] Movería a revisión: {img_path} ({', '.join(issues)})"
                )
            else:
                # Crear estructura de carpetas en review
                relative_path = img_path.relative_to(data_dir)
                dest_path = review_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.move(str(img_path), str(dest_path))
                    print(f"  ✅ Movida: {img_path} → {dest_path}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")

    # Balanceo
    if balance_classes and len(to_remove_for_balance) > 0:
        print(f"\n⚖️ Removiendo {len(to_remove_for_balance)} imágenes para balanceo...")

        balance_dir = review_dir / "removed_for_balance"

        for img_path in to_remove_for_balance:
            if dry_run:
                print(f"  [DRY RUN] Movería para balanceo: {img_path}")
            else:
                relative_path = img_path.relative_to(data_dir)
                dest_path = balance_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.move(str(img_path), str(dest_path))
                except Exception as e:
                    print(f"  ❌ Error: {e}")

    # 5. RESUMEN FINAL
    print("\n" + "=" * 70)
    print("RESUMEN DE LIMPIEZA")
    print("=" * 70)

    print(f"\n📊 Imágenes originales: {stats['total_images']}")
    print(f"❌ Corruptas eliminadas: {len(stats['corrupted'])}")
    print(f"📏 Muy pequeñas eliminadas: {len(stats['too_small'])}")
    print(f"🔍 Duplicados eliminados: {len(stats['duplicates'])}")

    if remove_quality_issues:
        print(f"⚠️ Problemas de calidad movidas: {len(stats['quality_issues'])}")

    if balance_classes:
        print(f"⚖️ Removidas para balanceo: {len(stats['removed_for_balance'])}")

    total_removed = (
        len(stats["corrupted"])
        + len(stats["too_small"])
        + len(stats["duplicates"])
        + (len(stats["quality_issues"]) if remove_quality_issues else 0)
        + (len(stats["removed_for_balance"]) if balance_classes else 0)
    )

    print(f"\n🗑️ Total removidas: {total_removed}")
    print(f"✅ Imágenes restantes: {stats['total_images'] - total_removed}")

    # Guardar reporte
    if not dry_run:
        report_path = data_dir.parent / f"{data_dir.name}_cleaning_report.json"
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n📄 Reporte guardado en: {report_path}")

        if backup_dir.exists():
            print(f"💾 Backup disponible en: {backup_dir}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Limpia y mejora el dataset automáticamente"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con el dataset a limpiar",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Solo mostrar qué haría, sin hacer cambios",
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Balancear clases al tamaño de la clase minoritaria",
    )
    parser.add_argument(
        "--remove_quality_issues",
        action="store_true",
        help="Mover imágenes con problemas de calidad a carpeta de revisión",
    )

    args = parser.parse_args()

    clean_dataset(
        args.data_dir,
        dry_run=args.dry_run,
        balance_classes=args.balance_classes,
        remove_quality_issues=args.remove_quality_issues,
    )

    print("\n✅ Limpieza completada!")

    if args.dry_run:
        print("\n💡 Para aplicar los cambios, ejecuta sin --dry_run")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Limpieza interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
