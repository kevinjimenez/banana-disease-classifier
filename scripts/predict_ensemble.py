"""
Ensemble de múltiples modelos para máxima precisión.
Combina predicciones de diferentes arquitecturas.

Uso:
    python scripts/ensemble_predict.py \
        --checkpoints model1.pth model2.pth model3.pth \
        --data_dir ./data/test \
        --output results_ensemble.json
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from deep.config.config import Config
from deep.core.dataset import BananaDiseaseDataset
from deep.core.transforms import get_transforms
from deep.core.inference import BananaClassifier


def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """Carga modelo desde checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Cargar config si existe
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        classes = config_dict["data"]["classes"]
        model_name = config_dict["model"].get("name", "convnext_base")
    else:
        classes = ["cordana", "fusarium_r4t", "healthy", "moko", "pestalotiopsis", "sigatoka"]
        model_name = "convnext_base"

    # Crear modelo
    model = BananaClassifier(
        model_name=model_name, num_classes=len(classes), pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, classes


@torch.no_grad()
def ensemble_predict(models, data_loader, device="cuda", method="average"):
    """
    Predicción con ensemble de modelos.

    Args:
        models: Lista de modelos
        data_loader: DataLoader con datos
        device: Device
        method: 'average' (promedio) o 'voting' (votación)

    Returns:
        predictions, labels, probabilities
    """
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for images, labels in tqdm(data_loader, desc="Ensemble prediction"):
        images = images.to(device)

        # Recolectar predicciones de cada modelo
        model_outputs = []
        for model in models:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            model_outputs.append(probs)

        # Combinar predicciones
        if method == "average":
            # Promedio de probabilidades
            ensemble_probs = torch.stack(model_outputs).mean(dim=0)
        elif method == "voting":
            # Voto mayoritario
            votes = torch.stack([probs.argmax(dim=1) for probs in model_outputs])
            # Convertir a one-hot y promediar
            one_hot = torch.zeros(
                votes.size(0), images.size(0), probs.size(1), device=device
            )
            one_hot.scatter_(2, votes.unsqueeze(-1), 1)
            ensemble_probs = one_hot.sum(dim=0) / len(models)
        else:
            raise ValueError(f"Unknown method: {method}")

        predictions = ensemble_probs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probabilities.extend(ensemble_probs.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def main():
    parser = argparse.ArgumentParser(description="Ensemble de múltiples modelos")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Lista de checkpoints a usar en el ensemble",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directorio con datos de test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_results.json",
        help="Archivo de salida con resultados",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["average", "voting"],
        help="Método de ensemble",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ENSEMBLE PREDICTION")
    print("=" * 70)
    print(f"\nModelos: {len(args.checkpoints)}")
    for i, ckpt in enumerate(args.checkpoints, 1):
        print(f"  {i}. {ckpt}")
    print(f"\nMétodo: {args.method}")
    print(f"Device: {device}")

    # Cargar modelos
    print("\n[INFO] Cargando modelos...")
    models = []
    classes = None

    for ckpt_path in args.checkpoints:
        model, model_classes = load_model_from_checkpoint(ckpt_path, device)
        models.append(model)

        # Verificar que todos los modelos usan las mismas clases
        if classes is None:
            classes = model_classes
        elif classes != model_classes:
            raise ValueError(
                f"Los modelos tienen clases diferentes: {classes} vs {model_classes}"
            )

    print(f"✅ {len(models)} modelos cargados")
    print(f"   Clases: {classes}")

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

    print(f"\n[INFO] Dataset: {len(test_dataset)} imágenes")

    # Hacer predicciones
    predictions, labels, probabilities = ensemble_predict(
        models, test_loader, device, args.method
    )

    # Evaluar
    accuracy = accuracy_score(labels, predictions)

    print("\n" + "=" * 70)
    print("RESULTADOS DEL ENSEMBLE")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Reporte por clase
    report = classification_report(labels, predictions, target_names=classes, digits=4)
    print("\n" + report)

    # Comparar con modelos individuales
    print("\n" + "=" * 70)
    print("COMPARACIÓN CON MODELOS INDIVIDUALES")
    print("=" * 70)

    individual_accs = []
    for i, model in enumerate(models):
        model_preds = []
        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                model_preds.extend(preds.cpu().numpy())

        model_acc = accuracy_score(labels, model_preds)
        individual_accs.append(model_acc)
        print(f"Modelo {i+1}: {model_acc:.4f} ({model_acc*100:.2f}%)")

    avg_individual = np.mean(individual_accs)
    improvement = accuracy - avg_individual

    print(f"\nPromedio individual: {avg_individual:.4f} ({avg_individual*100:.2f}%)")
    print(f"Ensemble: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Mejora: +{improvement:.4f} (+{improvement*100:.2f}%)")

    # Guardar resultados
    results = {
        "ensemble_accuracy": float(accuracy),
        "individual_accuracies": [float(acc) for acc in individual_accs],
        "average_individual": float(avg_individual),
        "improvement": float(improvement),
        "method": args.method,
        "num_models": len(models),
        "classes": classes,
        "classification_report": report,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Resultados guardados en: {args.output}")

    # Guardar predicciones
    predictions_file = Path(args.output).stem + "_predictions.npy"
    np.save(
        predictions_file,
        {"predictions": predictions, "labels": labels, "probabilities": probabilities},
    )
    print(f"✅ Predicciones guardadas en: {predictions_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Predicción interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
