"""
Script para comparar múltiples experimentos de entrenamiento.
Analiza checkpoints y genera comparaciones visuales.

Uso:
    python scripts/compare_experiments.py \
        --checkpoint_dirs ./checkpoints/exp1 ./checkpoints/exp2 ./checkpoints/exp3 \
        --output_dir ./comparison
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict

sns.set_style("whitegrid")


def load_checkpoint_info(checkpoint_path):
    """Carga información de un checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        info = {
            "epoch": checkpoint.get("epoch", "N/A"),
            "best_val_acc": checkpoint.get("best_val_acc", 0),
            "train_acc": checkpoint.get("train_acc", 0),
            "val_acc": checkpoint.get("val_acc", 0),
            "train_loss": checkpoint.get("train_loss", 0),
            "val_loss": checkpoint.get("val_loss", 0),
        }

        # Config si existe
        if "config" in checkpoint:
            config = checkpoint["config"]
            info["model_name"] = config.get("model", {}).get("name", "unknown")
            info["batch_size"] = config.get("training", {}).get("batch_size", "N/A")
            info["learning_rate"] = config.get("training", {}).get(
                "learning_rate", "N/A"
            )
            info["augmentation"] = config.get("data", {}).get(
                "augmentation_level", "N/A"
            )
        else:
            info["model_name"] = "unknown"
            info["batch_size"] = "N/A"
            info["learning_rate"] = "N/A"
            info["augmentation"] = "N/A"

        return info
    except Exception as e:
        print(f"⚠️ Error cargando {checkpoint_path}: {e}")
        return None


def find_best_checkpoint(checkpoint_dir):
    """Encuentra el mejor checkpoint en un directorio."""
    checkpoint_dir = Path(checkpoint_dir)

    # Buscar best_model.pth primero
    best_path = checkpoint_dir / "best_model.pth"
    if best_path.exists():
        return best_path

    # Si no, buscar todos los checkpoints y elegir el mejor
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None

    best_checkpoint = None
    best_acc = 0

    for ckpt in checkpoints:
        info = load_checkpoint_info(ckpt)
        if info and info["best_val_acc"] > best_acc:
            best_acc = info["best_val_acc"]
            best_checkpoint = ckpt

    return best_checkpoint


def load_training_history(checkpoint_dir):
    """Carga el historial de entrenamiento si existe."""
    history_file = Path(checkpoint_dir) / "training_history.json"

    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)

    return None


def compare_experiments(checkpoint_dirs, output_dir):
    """Compara múltiples experimentos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPARACIÓN DE EXPERIMENTOS")
    print("=" * 70)

    experiments = []

    # Cargar información de cada experimento
    for i, ckpt_dir in enumerate(checkpoint_dirs, 1):
        ckpt_dir = Path(ckpt_dir)
        exp_name = ckpt_dir.name

        print(f"\n[{i}/{len(checkpoint_dirs)}] Analizando: {exp_name}")

        best_ckpt = find_best_checkpoint(ckpt_dir)

        if best_ckpt is None:
            print(f"  ⚠️ No se encontraron checkpoints en {ckpt_dir}")
            continue

        info = load_checkpoint_info(best_ckpt)

        if info is None:
            print(f"  ⚠️ Error cargando checkpoint")
            continue

        info["experiment"] = exp_name
        info["checkpoint_path"] = str(best_ckpt)

        experiments.append(info)

        print(f"  ✅ Cargado: {best_ckpt.name}")
        print(f"     Modelo: {info['model_name']}")
        print(f"     Best Val Acc: {info['best_val_acc']:.4f}")

    if not experiments:
        print("\n❌ No se encontraron experimentos válidos")
        return

    # Crear DataFrame
    df = pd.DataFrame(experiments)

    # Ordenar por best_val_acc
    df = df.sort_values("best_val_acc", ascending=False)

    print("\n" + "=" * 70)
    print("RANKING DE EXPERIMENTOS")
    print("=" * 70)
    print(
        df[
            [
                "experiment",
                "model_name",
                "best_val_acc",
                "train_acc",
                "batch_size",
                "learning_rate",
            ]
        ].to_string(index=False)
    )

    # Guardar tabla
    df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"\n✅ Tabla guardada en: {output_dir / 'comparison_table.csv'}")

    # Visualizaciones
    create_comparison_plots(df, output_dir)

    # Análisis de overfitting
    print("\n" + "=" * 70)
    print("ANÁLISIS DE OVERFITTING")
    print("=" * 70)

    for _, row in df.iterrows():
        gap = row["train_acc"] - row["val_acc"]
        status = (
            "✅ OK"
            if gap < 0.05
            else "⚠️ OVERFITTING" if gap < 0.10 else "❌ OVERFITTING SEVERO"
        )
        print(f"{row['experiment']:30s} | Gap: {gap:+.4f} | {status}")

    # Mejor configuración
    print("\n" + "=" * 70)
    print("🏆 MEJOR EXPERIMENTO")
    print("=" * 70)

    best = df.iloc[0]
    print(f"\nNombre: {best['experiment']}")
    print(f"Modelo: {best['model_name']}")
    print(f"Val Accuracy: {best['best_val_acc']:.4f} ({best['best_val_acc']*100:.2f}%)")
    print(f"Train Accuracy: {best['train_acc']:.4f} ({best['train_acc']*100:.2f}%)")
    print(f"Overfitting Gap: {best['train_acc'] - best['val_acc']:+.4f}")
    print(f"\nHiperparámetros:")
    print(f"  • Batch size: {best['batch_size']}")
    print(f"  • Learning rate: {best['learning_rate']}")
    print(f"  • Augmentation: {best['augmentation']}")
    print(f"\nCheckpoint: {best['checkpoint_path']}")

    # Recomendaciones
    print("\n" + "=" * 70)
    print("💡 RECOMENDACIONES")
    print("=" * 70)

    # Comparar modelos
    model_accs = df.groupby("model_name")["best_val_acc"].agg(["mean", "max", "count"])
    if len(model_accs) > 1:
        print("\nRendimiento por modelo:")
        print(model_accs.to_string())
        best_model = model_accs["mean"].idxmax()
        print(f"\n✅ Mejor modelo en promedio: {best_model}")

    # Comparar augmentation
    if df["augmentation"].nunique() > 1:
        aug_accs = df.groupby("augmentation")["best_val_acc"].agg(
            ["mean", "max", "count"]
        )
        print("\nRendimiento por augmentation level:")
        print(aug_accs.to_string())
        best_aug = aug_accs["mean"].idxmax()
        print(f"\n✅ Mejor augmentation en promedio: {best_aug}")

    # Identificar overfitting patterns
    overfit_exps = df[df["train_acc"] - df["val_acc"] > 0.05]
    if len(overfit_exps) > 0:
        print(f"\n⚠️ {len(overfit_exps)} experimentos con overfitting detectado")
        print("   Considera:")
        print("   • Aumentar augmentation level")
        print("   • Aumentar dropout/regularización")
        print("   • Usar modelo más pequeño")

    print("\n" + "=" * 70)


def create_comparison_plots(df, output_dir):
    """Crea gráficos de comparación."""

    # 1. Gráfico de barras - Val Accuracy
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(df))
    bars = plt.bar(df["experiment"], df["best_val_acc"], color=colors)
    plt.xlabel("Experimento", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(
        "Comparación de Validation Accuracy por Experimento",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim([df["best_val_acc"].min() - 0.05, 1.0])

    # Añadir valores encima de barras
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "val_accuracy_comparison.png", dpi=300)
    plt.close()
    print(f"✅ Gráfico guardado: {output_dir / 'val_accuracy_comparison.png'}")

    # 2. Train vs Val Accuracy (overfitting detection)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(
        x - width / 2, df["train_acc"], width, label="Train Accuracy", color="skyblue"
    )
    plt.bar(x + width / 2, df["val_acc"], width, label="Val Accuracy", color="coral")

    plt.xlabel("Experimento", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Train vs Val Accuracy (Detección de Overfitting)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, df["experiment"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_vs_val_accuracy.png", dpi=300)
    plt.close()
    print(f"✅ Gráfico guardado: {output_dir / 'train_vs_val_accuracy.png'}")

    # 3. Overfitting Gap
    plt.figure(figsize=(12, 6))
    gaps = df["train_acc"] - df["val_acc"]
    colors = [
        "green" if gap < 0.05 else "orange" if gap < 0.10 else "red" for gap in gaps
    ]

    plt.bar(df["experiment"], gaps, color=colors)
    plt.xlabel("Experimento", fontsize=12)
    plt.ylabel("Overfitting Gap (Train - Val)", fontsize=12)
    plt.title("Análisis de Overfitting por Experimento", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0.05, color="orange", linestyle="--", label="Umbral aceptable (0.05)")
    plt.axhline(y=0.10, color="red", linestyle="--", label="Umbral severo (0.10)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overfitting_gap.png", dpi=300)
    plt.close()
    print(f"✅ Gráfico guardado: {output_dir / 'overfitting_gap.png'}")

    # 4. Loss comparison (si está disponible)
    if df["val_loss"].notna().any():
        plt.figure(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.35

        plt.bar(
            x - width / 2,
            df["train_loss"],
            width,
            label="Train Loss",
            color="lightblue",
        )
        plt.bar(x + width / 2, df["val_loss"], width, label="Val Loss", color="salmon")

        plt.xlabel("Experimento", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Train vs Val Loss", fontsize=14, fontweight="bold")
        plt.xticks(x, df["experiment"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_comparison.png", dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {output_dir / 'loss_comparison.png'}")

    # 5. Comparación por modelo (si hay múltiples)
    if df["model_name"].nunique() > 1:
        plt.figure(figsize=(10, 6))
        model_stats = df.groupby("model_name")["best_val_acc"].agg(
            ["mean", "std", "count"]
        )

        plt.bar(
            model_stats.index,
            model_stats["mean"],
            yerr=model_stats["std"],
            capsize=5,
            color="mediumpurple",
            alpha=0.8,
        )

        plt.xlabel("Modelo", fontsize=12)
        plt.ylabel("Val Accuracy (promedio)", fontsize=12)
        plt.title("Comparación de Modelos", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")

        # Añadir número de experimentos
        for i, (model, row) in enumerate(model_stats.iterrows()):
            plt.text(
                i,
                row["mean"] + row["std"] + 0.01,
                f"n={int(row['count'])}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {output_dir / 'model_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara múltiples experimentos de entrenamiento"
    )
    parser.add_argument(
        "--checkpoint_dirs",
        nargs="+",
        required=True,
        help="Directorios con checkpoints a comparar",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison",
        help="Directorio de salida para resultados",
    )

    args = parser.parse_args()

    compare_experiments(args.checkpoint_dirs, args.output_dir)

    print("\n✅ Análisis completado!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Comparación interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
