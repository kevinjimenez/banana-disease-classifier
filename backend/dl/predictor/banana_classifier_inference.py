"""
Módulo de inferencia para el modelo de clasificación de enfermedades de banano.
Carga el modelo entrenado y realiza predicciones.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Union
import json

from dl.architectures.banana_classifier import BananaClassifier


class BananaClassifierInference:
    """
    Clase para realizar inferencia con el modelo entrenado.

    Uso:
        predictor = BananaClassifierInference('checkpoints/exp1_convnext_tiny_mps_100acc/best_model.pth')
        result = predictor.predict('imagen.jpg')
        print(result['class'], result['confidence'])
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Inicializa el predictor cargando el modelo.

        Args:
            checkpoint_path: Ruta al checkpoint (.pth)
            device: 'auto', 'cuda', 'mps', o 'cpu'
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Detectar device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[INFO] Usando device: {self.device}")

        # Cargar modelo
        self.model, self.classes = self._load_model()
        self.model.eval()

        # Cargar transformaciones (usar PyTorch transforms en lugar de Albumentations)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print(f"[INFO] Modelo cargado exitosamente")
        print(f"[INFO] Clases: {self.classes}")

    def _load_model(self) -> Tuple[BananaClassifier, List[str]]:
        """Carga el modelo desde el checkpoint."""

        # Cargar checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extraer configuración
        if "config" in checkpoint and checkpoint["config"]:
            config = checkpoint["config"]
            classes = config["data"]["classes"]
            model_config = config.get("model", {})
            model_name = model_config.get("name", "convnext_tiny")
            use_custom_head = model_config.get("use_custom_head", True)
            hidden_dims = model_config.get("hidden_dims", None)
            drop_rate = model_config.get("drop_rate", 0.2)
            drop_path_rate = model_config.get("drop_path_rate", 0.1)
        else:
            # Default
            classes = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]
            model_name = "convnext_tiny"
            use_custom_head = True
            hidden_dims = None
            drop_rate = 0.2
            drop_path_rate = 0.1

        # Crear modelo
        model = BananaClassifier(
            model_name=model_name,
            num_classes=len(classes),
            pretrained=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_custom_head=use_custom_head,
            hidden_dims=hidden_dims,
        )

        # Cargar pesos (puede ser 'model_state_dict' o 'state_dict')
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError(
                "No se encontró 'model_state_dict' o 'state_dict' en el checkpoint"
            )

        model.load_state_dict(state_dict, strict=False)

        # Mover a device
        model = model.to(self.device)

        return model, classes

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_all_probs: bool = False,
    ) -> Dict:
        """
        Realiza predicción sobre una imagen.

        Args:
            image: Ruta a imagen, PIL Image, o numpy array
            return_all_probs: Si True, retorna probabilidades de todas las clases

        Returns:
            Dict con:
                - class: Clase predicha
                - confidence: Confianza (0-1)
                - class_id: ID de la clase
                - all_probabilities: (opcional) Probabilidades de todas las clases
        """

        # Cargar imagen
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")

        # Aplicar transformaciones
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Resultado
        class_id = predicted.item()
        class_name = self.classes[class_id]
        conf_value = confidence.item()

        result = {
            "class": class_name,
            "confidence": float(conf_value),
            "class_id": int(class_id),
        }

        if return_all_probs:
            all_probs = probabilities[0].cpu().numpy()
            result["all_probabilities"] = {
                class_name: float(prob)
                for class_name, prob in zip(self.classes, all_probs)
            }

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        return_all_probs: bool = False,
    ) -> List[Dict]:
        """
        Realiza predicción sobre un batch de imágenes.

        Args:
            images: Lista de imágenes
            return_all_probs: Si True, retorna probabilidades de todas las clases

        Returns:
            Lista de resultados
        """
        results = []
        for img in images:
            result = self.predict(img, return_all_probs=return_all_probs)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Retorna información del modelo."""
        return {
            "model_name": self.model.model_name,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "device": self.device,
            "checkpoint_path": str(self.checkpoint_path),
        }


if __name__ == "__main__":
    # Ejemplo de uso
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia con modelo de banano")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Ruta al checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )

    args = parser.parse_args()

    # Crear predictor
    predictor = BananaClassifierInference(args.checkpoint, device=args.device)

    # Realizar predicción
    result = predictor.predict(args.image, return_all_probs=True)

    # Mostrar resultado
    print("\n" + "=" * 70)
    print("RESULTADO DE PREDICCIÓN")
    print("=" * 70)
    print(f"Imagen: {args.image}")
    print(f"Clase predicha: {result['class']}")
    print(f"Confianza: {result['confidence']*100:.2f}%")

    if "all_probabilities" in result:
        print("\nProbabilidades por clase:")
        for class_name, prob in result["all_probabilities"].items():
            print(f"  {class_name:20s}: {prob*100:5.2f}%")
    print("=" * 70)
