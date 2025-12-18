"""
Modelos de clasificación para enfermedades de banano.
Usa timm (PyTorch Image Models) para acceder a arquitecturas SOTA.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import timm
from timm.models import create_model


class BananaClassifier(nn.Module):
    """
    Clasificador de enfermedades de banano basado en arquitecturas modernas.

    Soporta múltiples backbones:
    - ConvNeXt: convnext_base, convnext_large
    - Swin Transformer: swin_base_patch4_window7_224
    - EfficientNetV2: efficientnetv2_m, efficientnetv2_l
    - Vision Transformer: vit_base_patch16_224

    Args:
        model_name: Nombre del modelo en timm
        num_classes: Número de clases
        pretrained: Usar pesos preentrenados de ImageNet
        drop_rate: Dropout rate
        drop_path_rate: DropPath rate (regularización para transformers)
        use_custom_head: Si True, usa cabeza de clasificación personalizada
        hidden_dims: Dimensiones de capas ocultas en la cabeza custom
    """

    def __init__(
        self,
        model_name: str = "convnext_base",
        num_classes: int = 4,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        use_custom_head: bool = True,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Crear modelo base con timm
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Sin cabeza de clasificación (features only)
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool="avg",  # Global average pooling
        )

        # Obtener dimensión de features
        self.feature_dim = self.backbone.num_features

        # Cabeza de clasificación
        if use_custom_head and hidden_dims:
            self.classifier = self._build_custom_head(hidden_dims)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _build_custom_head(self, hidden_dims: List[int]) -> nn.Module:
        """
        Construye cabeza de clasificación personalizada con:
        - Capas fully-connected
        - BatchNorm
        - Dropout
        - GELU activation

        Args:
            hidden_dims: Lista con dimensiones de capas ocultas

        Returns:
            Sequential module con la cabeza
        """
        layers = []
        input_dim = self.feature_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                ]
            )
            input_dim = hidden_dim

        # Capa final de clasificación
        layers.append(nn.Linear(input_dim, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de entrada (B, C, H, W)

        Returns:
            Logits de clasificación (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae features sin clasificación (útil para visualización).

        Args:
            x: Tensor de entrada (B, C, H, W)

        Returns:
            Features (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def freeze_backbone(self):
        """Congela backbone para fine-tuning rápido."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[INFO] Backbone congelado")

    def unfreeze_backbone(self):
        """Descongela backbone para fine-tuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"[INFO] Backbone descongelado")

    def unfreeze_last_n_layers(self, n: int):
        """
        Descongela las últimas n capas del backbone.
        Útil para fine-tuning gradual.

        Args:
            n: Número de capas a descongelar
        """
        # Congelar todo primero
        self.freeze_backbone()

        # Descongelar últimas n capas
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"[INFO] Últimas {n} capas descongeladas")

    def count_parameters(self) -> Dict[str, int]:
        """
        Cuenta parámetros del modelo.

        Returns:
            Diccionario con conteo de parámetros
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def print_model_info(self):
        """Imprime información del modelo."""
        params = self.count_parameters()

        print("\n" + "=" * 60)
        print("INFORMACIÓN DEL MODELO")
        print("=" * 60)
        print(f"Arquitectura: {self.model_name}")
        print(f"Número de clases: {self.num_classes}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"\nParámetros:")
        print(f"  Total:      {params['total']:,}")
        print(f"  Trainable:  {params['trainable']:,}")
        print(f"  Frozen:     {params['frozen']:,}")
        print("=" * 60 + "\n")


class ModelRegistry:
    """
    Registro de modelos recomendados para clasificación de enfermedades.
    """

    # Modelos por categoría
    MODELS = {
        # Convolution-based (rápidos, precisos)
        "convnext": {
            "convnext_tiny": "Ligero y rápido (28M params)",
            "convnext_small": "Balance velocidad/precisión (50M params)",
            "convnext_base": "Recomendado - excelente precisión (89M params)",
            "convnext_large": "Máxima precisión, más lento (197M params)",
        },
        # EfficientNet (eficientes)
        "efficientnet": {
            "efficientnetv2_s": "Muy eficiente (24M params)",
            "efficientnetv2_m": "Balance óptimo (54M params)",
            "efficientnetv2_l": "Alta precisión (119M params)",
        },
        # Transformers (SOTA pero requieren más datos)
        "vision_transformer": {
            "vit_base_patch16_224": "ViT base (86M params)",
            "vit_large_patch16_224": "ViT large (307M params)",
        },
        # Swin Transformer (mejor que ViT para imágenes)
        "swin": {
            "swin_tiny_patch4_window7_224": "Swin tiny (28M params)",
            "swin_base_patch4_window7_224": "Swin base (88M params)",
        },
        # ResNet (clásico, confiable)
        "resnet": {
            "resnet50": "ResNet50 (25M params)",
            "resnet101": "ResNet101 (45M params)",
        },
    }

    @classmethod
    def list_models(cls, category: Optional[str] = None):
        """
        Lista modelos disponibles.

        Args:
            category: Categoría específica o None para todas
        """
        print("\n" + "=" * 70)
        print("MODELOS DISPONIBLES")
        print("=" * 70)

        if category and category in cls.MODELS:
            models = {category: cls.MODELS[category]}
        else:
            models = cls.MODELS

        for cat, model_dict in models.items():
            print(f"\n{cat.upper().replace('_', ' ')}:")
            for model_name, description in model_dict.items():
                print(f"  • {model_name:40s} - {description}")

        print("\n" + "=" * 70 + "\n")

    @classmethod
    def get_recommended_model(cls, dataset_size: str = "medium") -> str:
        """
        Retorna modelo recomendado según tamaño del dataset.

        Args:
            dataset_size: 'small' (<1000 imgs), 'medium' (1k-5k), 'large' (>5k)

        Returns:
            Nombre del modelo recomendado
        """
        recommendations = {
            "small": "convnext_tiny",  # Menos overfitting
            "medium": "convnext_base",  # Balance óptimo
            "large": "convnext_large",  # Máxima capacidad
        }

        return recommendations.get(dataset_size, "convnext_base")


def create_model_from_config(config) -> BananaClassifier:
    """
    Crea modelo desde objeto Config.

    Args:
        config: Objeto Config con model_config

    Returns:
        BananaClassifier inicializado
    """
    model = BananaClassifier(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        drop_rate=config.model.drop_rate,
        drop_path_rate=config.model.drop_path_rate,
        use_custom_head=config.model.use_custom_head,
        hidden_dims=config.model.hidden_dims,
    )

    return model


# Ejemplo de uso
if __name__ == "__main__":
    # Listar modelos disponibles
    ModelRegistry.list_models()

    # Crear modelo
    model = BananaClassifier(
        model_name="convnext_base",
        num_classes=4,
        pretrained=True,
        use_custom_head=True,
        hidden_dims=[512, 256],
    )

    # Info del modelo
    model.print_model_info()

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Extraer features
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")

    # Congelar backbone
    model.freeze_backbone()
    params = model.count_parameters()
    print(f"\nDespués de congelar - Trainable: {params['trainable']:,}")
