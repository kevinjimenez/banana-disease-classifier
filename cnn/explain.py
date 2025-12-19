"""
Script de interpretabilidad para entender qué ve el modelo.

Ejemplo de uso:
    python scripts/interpret.py \
        --checkpoint checkpoints/best_model.pth \
        --image path/to/image.jpg \
        --methods integrated_gradients gradcam occlusion
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import argparse
from PIL import Image
import numpy as np

from cnn.config.config import Config
from cnn.core.inference import BananaClassifier
from cnn.core.transforms import get_transforms

# interpretation.explainer necesita verificarse si existe


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Carga modelo desde checkpoint.

    Args:
        checkpoint_path: Ruta al checkpoint
        device: Device

    Returns:
        Tupla (model, config, classes)
    """
    print(f"[INFO] Cargando checkpoint desde {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Cargar config
    if "config" in checkpoint and checkpoint["config"]:
        config_dict = checkpoint["config"]
        classes = config_dict["data"]["classes"]
        model_name = config_dict["model"]["name"]
        num_classes = config_dict["model"]["num_classes"]
    else:
        # Default
        classes = ["cordana", "fusarium_r4t", "healthy", "moko", "pestalotiopsis", "sigatoka"]
        model_name = "convnext_base"
        num_classes = 6

    # Crear modelo
    model = BananaClassifier(
        model_name=model_name, num_classes=num_classes, pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Modelo cargado: {model_name}")
    print(f"[INFO] Clases: {classes}")

    return model, classes


def load_and_preprocess_image(image_path: str, transform) -> torch.Tensor:
    """
    Carga y preprocesa imagen.

    Args:
        image_path: Ruta a la imagen
        transform: Transformación a aplicar

    Returns:
        Tensor de imagen preprocesada
    """
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Aplicar transform
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]

    return image_tensor


def predict_single_image(
    model: torch.nn.Module, image: torch.Tensor, classes: list, device: str = "cuda"
) -> dict:
    """
    Realiza predicción en una imagen.

    Args:
        model: Modelo
        image: Tensor de imagen
        classes: Lista de clases
        device: Device

    Returns:
        Diccionario con predicción
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()

    # Top-3 predicciones
    top3_probs, top3_indices = torch.topk(probs, min(3, len(classes)))

    result = {
        "predicted_class": classes[pred_idx],
        "predicted_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "top3": [
            {"class": classes[idx], "confidence": float(prob)}
            for idx, prob in zip(top3_indices, top3_probs)
        ],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Interpretabilidad de predicciones")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Ruta al checkpoint del modelo"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Ruta a la imagen a analizar"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["integrated_gradients", "gradcam", "occlusion"],
        help="Métodos de interpretabilidad a usar",
    )
    parser.add_argument(
        "--true_class",
        type=int,
        default=None,
        help="Índice de clase verdadera (opcional)",
    )
    parser.add_argument("--output_dir", type=str, default="./interpretations")
    parser.add_argument(
        "--compare_classes",
        action="store_true",
        help="Comparar interpretaciones para diferentes clases",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelo
    model, classes = load_model_from_checkpoint(args.checkpoint, args.device)

    # Cargar y preprocesar imagen
    transform = get_transforms("val", image_size=(224, 224))
    image_tensor = load_and_preprocess_image(args.image, transform)

    # Realizar predicción
    prediction = predict_single_image(model, image_tensor, classes, args.device)

    print("\n" + "=" * 70)
    print("PREDICCIÓN")
    print("=" * 70)
    print(f"Imagen: {args.image}")
    print(f"Predicción: {prediction['predicted_class']}")
    print(f"Confianza: {prediction['confidence']:.4f}")
    print("\nTop-3 predicciones:")
    for i, pred in enumerate(prediction["top3"], 1):
        print(f"  {i}. {pred['class']:<20} - {pred['confidence']:.4f}")
    print("=" * 70 + "\n")

    # Crear explainer
    explainer = DiseaseExplainer(model=model, classes=classes, device=args.device)

    # Explicar predicción
    print("[INFO] Generando explicaciones...")

    image_name = Path(args.image).stem
    save_path = output_dir / f"{image_name}_explanation.png"

    explainer.explain_prediction(
        image=image_tensor,
        true_class=(
            args.true_class
            if args.true_class is not None
            else prediction["predicted_idx"]
        ),
        pred_class=prediction["predicted_idx"],
        methods=args.methods,
        visualize=True,
        save_path=str(save_path),
    )

    print(f"[INFO] Explicación guardada en {save_path}")

    # Comparar clases si se solicita
    if args.compare_classes:
        print("[INFO] Generando comparación entre clases...")
        compare_save_path = output_dir / f"{image_name}_class_comparison.png"

        explainer.compare_classes(
            image=image_tensor,
            method="integrated_gradients",
            save_path=str(compare_save_path),
        )

        print(f"[INFO] Comparación guardada en {compare_save_path}")

    print(f"\n Interpretabilidad completada!")
    print(f" Resultados en: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
