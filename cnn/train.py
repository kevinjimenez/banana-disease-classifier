"""
Script principal de entrenamiento para clasificador de enfermedades de banano.
Integra todos los componentes del sistema.

Ejemplo de uso:
    python cnn/train.py --data_dir ./data/train --model convnext_base --epochs 50
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Cargar variables de entorno
load_dotenv(ROOT_DIR / '.env')

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import argparse
import random
import numpy as np
import warnings

# Suprimir warning de pin_memory en MPS
warnings.filterwarnings('ignore', message='.*pin_memory.*MPS.*')

from cnn.config.config import Config, get_quick_test_config, get_production_config
from cnn.core.dataset import BananaDiseaseDataset, create_dataloaders
from cnn.core.transforms import get_transforms
from cnn.core.inference import BananaClassifier, ModelRegistry
from cnn.core.trainer import Trainer


def set_seed(seed: int):
    """
    Fija semillas para reproducibilidad.

    Args:
        seed: Semilla aleatoria
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config: Config):
    """
    Crea optimizer según configuración.

    Args:
        model: Modelo a optimizar
        config: Configuración

    Returns:
        Optimizer configurado
    """
    optimizer_name = config.training.optimizer.lower()

    if optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer no soportado: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config: Config, steps_per_epoch: int):
    """
    Crea learning rate scheduler.

    Args:
        optimizer: Optimizer
        config: Configuración
        steps_per_epoch: Steps por época

    Returns:
        Scheduler o None
    """
    scheduler_name = config.training.scheduler.lower()

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=config.training.min_lr
        )
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=config.training.min_lr
        )
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            epochs=config.training.epochs,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        scheduler = None

    return scheduler


def create_criterion(config: Config, class_weights: torch.Tensor = None):
    """
    Crea loss function según configuración.

    Args:
        config: Configuración
        class_weights: Pesos de clase para balanceo

    Returns:
        Loss function
    """
    loss_name = config.training.loss_fn.lower()

    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "label_smoothing":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=config.training.label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion


def main():
    # Parsear argumentos
    parser = argparse.ArgumentParser(
        description="Entrenamiento de clasificador de enfermedades de banano"
    )

    # Datos
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con datos de entrenamiento",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Directorio con datos de validación (opcional)",
    )

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default="convnext_base",
        help="Nombre del modelo (ver ModelRegistry)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Usar pesos preentrenados",
    )

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--augmentation",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Nivel de data augmentation",
    )

    # Configuración
    parser.add_argument(
        "--config", type=str, default=None, help="Ruta a archivo de configuración YAML"
    )
    parser.add_argument(
        "--quick_test", action="store_true", help="Modo de prueba rápida (2 épocas)"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directorio de salida"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="banana_experiment",
        help="Nombre del experimento",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directorio para guardar checkpoints (default: ./checkpoints/EXPERIMENT_NAME)",
    )

    # Wandb
    parser.add_argument(
        "--no_wandb", action="store_true", help="Desactivar Wandb logging"
    )

    # GPU
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device para entrenamiento (cuda, mps para Mac, cpu)",
    )

    args = parser.parse_args()

    # Crear o cargar configuración
    if args.config:
        config = Config.load(args.config)
    elif args.quick_test:
        config = get_quick_test_config()
    else:
        config = Config()

    # Sobrescribir con argumentos de línea de comando
    config.data.data_dir = args.data_dir
    config.model.name = args.model
    config.model.pretrained = args.pretrained
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.augmentation_level = args.augmentation
    config.experiment_name = args.experiment_name

    # Configurar todos los directorios dentro del experimento
    # Usar variables de entorno si están definidas, sino valores por defecto
    experiments_dir = os.getenv('EXPERIMENTS_DIR', './experiments')
    experiment_name = os.getenv('EXPERIMENT_NAME', args.experiment_name)
    
    # Si se proporciona checkpoint_dir, usar su directorio padre como base
    if args.checkpoint_dir:
        experiment_base = str(Path(args.checkpoint_dir).parent)
        config.logging.checkpoint_dir = args.checkpoint_dir
        config.logging.output_dir = os.getenv('OUTPUT_DIR') or f"{experiment_base}/outputs"
        config.logging.log_dir = os.getenv('LOG_DIR') or f"{experiment_base}/logs"
        config.interpretability.visualization_dir = os.getenv('INTERPRETATION_DIR') or f"{experiment_base}/interpretations"
    else:
        # Usar estructura basada en variables de entorno
        experiment_base = f"{experiments_dir}/{experiment_name}"
        config.logging.checkpoint_dir = os.getenv('CHECKPOINT_DIR') or f"{experiment_base}/checkpoints"
        config.logging.output_dir = os.getenv('OUTPUT_DIR') or f"{experiment_base}/outputs"
        config.logging.log_dir = os.getenv('LOG_DIR') or f"{experiment_base}/logs"
        config.interpretability.visualization_dir = os.getenv('INTERPRETATION_DIR') or f"{experiment_base}/interpretations"

    # Detectar device disponible
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA no disponible, usando CPU")
        config.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS no disponible, usando CPU")
        config.device = "cpu"
    else:
        config.device = args.device

    if args.no_wandb:
        config.logging.use_wandb = False

    # Crear directorios necesarios
    Path(config.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.interpretability.visualization_dir).mkdir(parents=True, exist_ok=True)

    # Fijar semilla
    set_seed(config.seed)

    # Imprimir configuración
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN DEL EXPERIMENTO")
    print("=" * 70)
    print(f"Experimento: {config.experiment_name}")
    print(f"Modelo: {config.model.name}")
    print(f"Épocas: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Augmentation: {config.data.augmentation_level}")
    print(f"Wandb: {config.logging.use_wandb}")
    print("=" * 70 + "\n")

    # Crear transformaciones
    train_transform = get_transforms(
        mode="train",
        image_size=config.data.image_size,
        augmentation_level=config.data.augmentation_level,
    )

    val_transform = get_transforms(mode="val", image_size=config.data.image_size)

    # Crear datasets
    print("[INFO] Cargando datasets...")
    train_dataset = BananaDiseaseDataset(
        root_dir=config.data.data_dir,
        transform=train_transform,
        classes=config.data.classes,
        cache_images=False,  # Cambiar a True si tienes suficiente RAM
    )

    # Dataset de validación
    if args.val_dir:
        val_dataset = BananaDiseaseDataset(
            root_dir=args.val_dir,
            transform=val_transform,
            classes=config.data.classes,
            cache_images=False,
        )
    else:
        # Usar mismo directorio con split
        val_dataset = BananaDiseaseDataset(
            root_dir=config.data.data_dir,
            transform=val_transform,
            classes=config.data.classes,
            cache_images=False,
        )

    # Imprimir estadísticas
    train_dataset.print_statistics()

    # Crear dataloaders
    dataloaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Crear modelo
    print("[INFO] Creando modelo...")
    model = BananaClassifier(
        model_name=config.model.name,
        num_classes=len(config.data.classes),
        pretrained=config.model.pretrained,
        drop_rate=config.model.drop_rate,
        drop_path_rate=config.model.drop_path_rate,
        use_custom_head=config.model.use_custom_head,
        hidden_dims=config.model.hidden_dims,
    )

    model.print_model_info()

    # Crear optimizer
    optimizer = create_optimizer(model, config)

    # Crear scheduler
    scheduler = create_scheduler(
        optimizer, config, steps_per_epoch=len(dataloaders["train"])
    )

    # Crear criterion con class weights opcional
    class_weights = None
    if hasattr(train_dataset, "get_class_weights_tensor"):
        class_weights = train_dataset.get_class_weights_tensor().to(config.device)
        print(f"[INFO] Usando class weights: {class_weights}")

    criterion = create_criterion(config, class_weights)

    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=config.device,
    )

    # Entrenar
    trainer.train(epochs=config.training.epochs)

    # Guardar configuración final
    config.save(f"{config.logging.output_dir}/{config.experiment_name}_config.yaml")

    print("\n✅ Entrenamiento completado!")
    print(f"📁 Resultados en: {config.logging.checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
