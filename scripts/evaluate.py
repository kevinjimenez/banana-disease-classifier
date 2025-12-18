"""
Script de evaluación para modelo entrenado de clasificación de enfermedades de banano.

Ejemplo de uso:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir ./data/test
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from deep.config.config import Config
from deep.core.dataset import BananaDiseaseDataset
from deep.core.transforms import get_transforms, AugmentationPolicy
from deep.core.inference import BananaClassifier


class ModelEvaluator:
    """
    Evaluador de modelo con múltiples métricas y visualizaciones.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        classes: list,
        device: str = "cuda",
    ):
        self.model = model.eval()
        self.dataloader = dataloader
        self.classes = classes
        self.device = device

        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    @torch.no_grad()
    def evaluate(self, use_tta: bool = False) -> dict:
        """
        Evalúa el modelo en el dataset.

        Args:
            use_tta: Usar Test-Time Augmentation

        Returns:
            Diccionario con métricas
        """
        print("\n" + "=" * 70)
        print("EVALUANDO MODELO")
        print("=" * 70)

        for images, labels in tqdm(self.dataloader, desc="Evaluando"):
            images = images.to(self.device)
            labels = labels.cpu().numpy()

            if use_tta:
                # Test-Time Augmentation
                outputs = self._predict_with_tta(images)
            else:
                # Predicción normal
                outputs = self.model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels)
            self.all_probs.extend(probs.cpu().numpy())

        self.all_preds = np.array(self.all_preds)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)

        # Calcular métricas
        metrics = self._compute_metrics()

        return metrics

    def _predict_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predicción con Test-Time Augmentation.
        Promedia predicciones de múltiples augmentaciones.
        """
        tta_transforms = AugmentationPolicy.get_test_time_augmentation()

        # Lista para acumular logits
        all_logits = []

        # Predicción original
        logits = self.model(images)
        all_logits.append(logits)

        # Predicciones con augmentations
        # (Nota: para TTA completo, necesitarías re-aplicar transforms)
        # Por simplicidad, aquí solo hacemos flip horizontal
        images_flipped = torch.flip(images, dims=[3])
        logits_flipped = self.model(images_flipped)
        all_logits.append(logits_flipped)

        # Promediar logits
        mean_logits = torch.stack(all_logits).mean(dim=0)

        return mean_logits

    def _compute_metrics(self) -> dict:
        """Calcula múltiples métricas de evaluación."""
        # Accuracy global
        accuracy = accuracy_score(self.all_labels, self.all_preds)

        # Precision, Recall, F1 por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average=None, zero_division=0
        )

        # Promedios
        precision_macro = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average="macro", zero_division=0
        )[0]

        recall_macro = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average="macro", zero_division=0
        )[1]

        f1_macro = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average="macro", zero_division=0
        )[2]

        # Métricas por clase
        per_class_metrics = {}
        for idx, class_name in enumerate(self.classes):
            per_class_metrics[class_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "per_class": per_class_metrics,
        }

        return metrics

    def print_metrics(self, metrics: dict):
        """Imprime métricas de forma legible."""
        print("\n" + "=" * 70)
        print("RESULTADOS DE EVALUACIÓN")
        print("=" * 70)
        print(f"\n📊 Métricas Globales:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

        print(f"\n📈 Métricas por Clase:")
        print(
            f"{'Clase':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        )
        print("-" * 70)

        for class_name, class_metrics in metrics["per_class"].items():
            print(
                f"{class_name:<20} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1']:>10.4f} "
                f"{class_metrics['support']:>10d}"
            )

        print("=" * 70 + "\n")

    def plot_confusion_matrix(self, save_path: str = None):
        """Genera matriz de confusión."""
        cm = confusion_matrix(self.all_labels, self.all_preds)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Matriz de confusión guardada en {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_per_class_metrics(self, metrics: dict, save_path: str = None):
        """Gráfico de barras con métricas por clase."""
        classes = list(metrics["per_class"].keys())
        precision = [metrics["per_class"][c]["precision"] for c in classes]
        recall = [metrics["per_class"][c]["recall"] for c in classes]
        f1 = [metrics["per_class"][c]["f1"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - width, precision, width, label="Precision", color="skyblue")
        ax.bar(x, recall, width, label="Recall", color="lightcoral")
        ax.bar(x + width, f1, width, label="F1-Score", color="lightgreen")

        ax.set_xlabel("Clase", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Métricas por Clase", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Gráfico guardado en {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, metrics: dict, save_path: str):
        """Guarda resultados en JSON."""
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Resultados guardados en {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de modelo de enfermedades de banano"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Ruta al checkpoint del modelo"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directorio con datos de test"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--use_tta", action="store_true", help="Usar Test-Time Augmentation"
    )
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"]
    )

    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar checkpoint
    print(f"[INFO] Cargando checkpoint desde {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Cargar config
    if "config" in checkpoint and checkpoint["config"]:
        config = Config()
        config_dict = checkpoint["config"]
        # Reconstruir config desde dict
        classes = config_dict["data"]["classes"]
        # Obtener parámetros del modelo desde config
        model_config = config_dict.get("model", {})
        model_name = model_config.get("name", "convnext_base")
        use_custom_head = model_config.get("use_custom_head", True)
        hidden_dims = model_config.get("hidden_dims", None)
        drop_rate = model_config.get("drop_rate", 0.2)
        drop_path_rate = model_config.get("drop_path_rate", 0.1)
    else:
        # Default classes
        classes = ["cordana", "fusarium_r4t", "healthy", "moko", "pestalotiopsis", "sigatoka"]
        model_name = "convnext_base"
        use_custom_head = True
        hidden_dims = None
        drop_rate = 0.2
        drop_path_rate = 0.1

    print(f"[INFO] Clases: {classes}")
    print(f"[INFO] Modelo: {model_name}")
    print(f"[INFO] Custom head: {use_custom_head}, Hidden dims: {hidden_dims}")

    # Crear modelo
    # Detectar arquitectura desde checkpoint
    state_dict = checkpoint["model_state_dict"]
    num_classes = len(classes)

    model = BananaClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        use_custom_head=use_custom_head,
        hidden_dims=hidden_dims,
    )

    # Cargar state_dict con strict=False para manejar diferencias en classifier
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(
            f"[WARNING] Missing keys: {missing_keys[:5]}..."
        )  # Mostrar solo primeras 5
    if unexpected_keys:
        print(
            f"[WARNING] Unexpected keys: {unexpected_keys[:5]}..."
        )  # Mostrar solo primeras 5

    model = model.to(args.device)
    model.eval()

    print(f"[INFO] Modelo cargado: {model_name}")

    # Crear dataset
    transform = get_transforms("test", image_size=(224, 224))

    test_dataset = BananaDiseaseDataset(
        root_dir=args.data_dir, transform=transform, classes=classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[INFO] Dataset de test: {len(test_dataset)} imágenes")

    # Evaluar
    evaluator = ModelEvaluator(
        model=model, dataloader=test_loader, classes=classes, device=args.device
    )

    metrics = evaluator.evaluate(use_tta=args.use_tta)

    # Imprimir resultados
    evaluator.print_metrics(metrics)

    # Guardar visualizaciones
    evaluator.plot_confusion_matrix(save_path=output_dir / "confusion_matrix.png")

    evaluator.plot_per_class_metrics(
        metrics, save_path=output_dir / "per_class_metrics.png"
    )

    # Guardar resultados JSON
    evaluator.save_results(metrics, save_path=output_dir / "evaluation_metrics.json")

    print(f"\n Evaluación completada!")
    print(f" Resultados en: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
