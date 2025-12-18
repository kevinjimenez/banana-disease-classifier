"""
Sistema de configuración para el clasificador de enfermedades de banano.
Usa dataclasses para tipado estático y validación.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """Configuración de datos y preprocesamiento."""

    # Rutas
    data_dir: str = "./data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"

    # Clases de enfermedades (orden específico)
    # IMPORTANTE: Este orden determina los índices de clase
    # Clase 0 = cordana, Clase 1 = healthy, etc.
    classes: List[str] = field(
        default_factory=lambda: [
            "cordana",  # Clase 0
            "healthy",  # Clase 1
            "pestalotiopsis",  # Clase 2
            "sigatoka",  # Clase 3
        ]
    )

    # Tamaño de imagen
    image_size: Tuple[int, int] = (224, 224)

    # Splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_level: str = "medium"  # light, medium, heavy

    def __post_init__(self):
        """Validación post-inicialización."""
        assert len(self.classes) > 0, "Debe haber al menos una clase"
        assert self.train_split + self.val_split + self.test_split == 1.0
        assert self.augmentation_level in ["light", "medium", "heavy"]


@dataclass
class ModelConfig:
    """Configuración del modelo."""

    # Arquitectura
    name: str = "convnext_base"  # timm model name
    pretrained: bool = True
    num_classes: int = 4

    # Regularización
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    # Cabeza de clasificación
    use_custom_head: bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])

    def __post_init__(self):
        """Validación del modelo."""
        assert self.num_classes > 1, "Debe haber al menos 2 clases"
        assert 0.0 <= self.drop_rate <= 0.8
        assert 0.0 <= self.drop_path_rate <= 0.5


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""

    # Épocas y batch
    epochs: int = 50
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd, adam
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9  # Solo para SGD

    # Learning rate scheduler
    scheduler: str = "cosine"  # cosine, step, plateau, onecycle
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Loss function
    loss_fn: str = "cross_entropy"  # cross_entropy, focal_loss, label_smoothing
    label_smoothing: float = 0.1

    # Regularización
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    save_best_only: bool = True
    save_frequency: int = 5  # Guardar cada N epochs

    def __post_init__(self):
        """Validación de entrenamiento."""
        assert self.epochs > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.optimizer in ["adamw", "sgd", "adam"]
        assert self.scheduler in ["cosine", "step", "plateau", "onecycle"]
        assert self.loss_fn in ["cross_entropy", "focal_loss", "label_smoothing"]


@dataclass
class LoggingConfig:
    """Configuración de logging y tracking."""

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "banana-disease-classification"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["baseline"])

    # Tensorboard
    use_tensorboard: bool = True

    # Logging
    log_interval: int = 10  # Log cada N batches
    eval_interval: int = 1  # Evaluar cada N epochs

    # Directorios de salida
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Verbosidad
    verbose: bool = True


@dataclass
class InterpretabilityConfig:
    """Configuración para interpretabilidad."""

    # Métodos de interpretabilidad
    methods: List[str] = field(
        default_factory=lambda: ["integrated_gradients", "gradcam", "occlusion"]
    )

    # Configuración por método
    ig_steps: int = 50  # Integrated Gradients steps
    occlusion_size: int = 15  # Tamaño de parche para occlusion

    # Visualización
    save_visualizations: bool = True
    visualization_dir: str = "./interpretations"

    # Cantidad de ejemplos a interpretar
    num_examples_per_class: int = 5


@dataclass
class Config:
    """Configuración completa del sistema."""

    # Sub-configuraciones
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    interpretability: InterpretabilityConfig = field(
        default_factory=InterpretabilityConfig
    )

    # Configuración general
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name: str = "baseline_experiment"

    def __post_init__(self):
        """Setup post-inicialización."""
        # Crear directorios si no existen
        Path(self.logging.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.interpretability.visualization_dir).mkdir(parents=True, exist_ok=True)

        # Ajustar num_classes según data.classes
        self.model.num_classes = len(self.data.classes)

    def to_dict(self) -> dict:
        """Convierte configuración a diccionario."""
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: str):
        """Guarda configuración a archivo YAML."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Carga configuración desde archivo YAML."""
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Reconstruir dataclasses anidados
        data_config = DataConfig(**config_dict.pop("data", {}))
        model_config = ModelConfig(**config_dict.pop("model", {}))
        training_config = TrainingConfig(**config_dict.pop("training", {}))
        logging_config = LoggingConfig(**config_dict.pop("logging", {}))
        interpretability_config = InterpretabilityConfig(
            **config_dict.pop("interpretability", {})
        )

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            interpretability=interpretability_config,
            **config_dict,
        )


# Configuraciones predefinidas para diferentes escenarios
def get_quick_test_config() -> Config:
    """Configuración para pruebas rápidas."""
    config = Config()
    config.training.epochs = 2
    config.training.batch_size = 8
    config.logging.use_wandb = False
    config.data.num_workers = 2
    return config


def get_production_config() -> Config:
    """Configuración optimizada para producción."""
    config = Config()
    config.model.name = "convnext_base"
    config.training.epochs = 100
    config.training.batch_size = 64
    config.training.use_amp = True
    config.training.gradient_accumulation_steps = 2
    return config


def get_interpretability_config() -> Config:
    """Configuración enfocada en interpretabilidad."""
    config = Config()
    config.model.name = "vit_base_patch16_224"  # ViT para attention maps
    config.interpretability.methods = [
        "integrated_gradients",
        "gradcam",
        "attention_rollout",
        "occlusion",
    ]
    config.interpretability.num_examples_per_class = 10
    return config


# Ejemplo de uso
if __name__ == "__main__":
    # Crear configuración default
    config = Config()

    # Modificar parámetros
    config.model.name = "convnext_large"
    config.training.learning_rate = 5e-5
    config.data.classes = [
        "cordana",
        "healthy",
        "pestalotiopsis",
        "sigatoka",
        "fusarium_t4",
    ]

    # Guardar
    config.save("config_example.yaml")

    # Cargar
    loaded_config = Config.load("config_example.yaml")

    print(f"Config cargada: {loaded_config.model.name}")
    print(f"Clases: {loaded_config.data.classes}")
    print(f"Device: {loaded_config.device}")
