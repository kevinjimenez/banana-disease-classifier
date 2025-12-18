"""
Configuración optimizada para RTX 4060 (8GB VRAM).
Usar este config para máximo rendimiento en tu GPU.

Uso:
    from config.config_rtx4060 import get_rtx4060_config
    config = get_rtx4060_config()
"""

from dataclasses import dataclass, field
from typing import List
from config.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    InterpretabilityConfig,
)


def get_rtx4060_config() -> Config:
    """
    Configuración optimizada para NVIDIA RTX 4060 (8GB VRAM).

    Optimizaciones incluidas:
    - Batch size: 32 (óptimo para 8GB)
    - Mixed precision: Habilitado (crítico)
    - Gradient accumulation: 1 (puede aumentar a 2 si OOM)
    - Workers: 4 (balance CPU-GPU)
    - Pin memory: True (acelera transferencias)

    Returns:
        Config optimizado para RTX 4060
    """

    # Data config
    data_config = DataConfig(
        data_dir="./data",
        train_dir="train",
        val_dir="val",
        test_dir="test",
        classes=["cordana", "healthy", "pestalotiopsis", "sigatoka"],
        image_size=(224, 224),
        # Optimizado para RTX 4060
        num_workers=4,  # ✅ Balance óptimo CPU-GPU
        pin_memory=True,  # ✅ Acelera transferencias a GPU
        use_augmentation=True,
        augmentation_level="medium",  # ✅ Balance calidad/velocidad
    )

    # Model config
    model_config = ModelConfig(
        name="convnext_base",  # ✅ Perfecto para 8GB
        pretrained=True,
        num_classes=4,
        # Regularización
        drop_rate=0.2,
        drop_path_rate=0.1,
        # Custom head
        use_custom_head=True,
        hidden_dims=[512, 256],
    )

    # Training config - OPTIMIZADO PARA RTX 4060
    training_config = TrainingConfig(
        epochs=50,
        # 🎯 BATCH SIZE ÓPTIMO PARA 8GB
        batch_size=32,  # ✅ Usa ~6.5GB VRAM
        gradient_accumulation_steps=1,  # ✅ Cambiar a 2 si OOM
        # Optimizer
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=1e-5,
        # LR Scheduler
        scheduler="cosine",
        warmup_epochs=5,
        min_lr=1e-7,
        # Loss
        loss_fn="cross_entropy",
        label_smoothing=0.1,
        # Regularización
        use_mixup=True,
        mixup_alpha=0.2,
        use_cutmix=True,
        cutmix_alpha=1.0,
        # 🚀 MIXED PRECISION - CRÍTICO PARA RTX 4060
        use_amp=True,  # ✅ OBLIGATORIO - 2x velocidad + ahorra memoria
        # Gradient clipping
        max_grad_norm=1.0,
        # Early stopping
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        # Checkpointing
        save_best_only=True,
        save_frequency=5,
    )

    # Logging config
    logging_config = LoggingConfig(
        use_wandb=True,
        wandb_project="banana-disease-rtx4060",
        wandb_entity=None,
        wandb_tags=["rtx4060", "convnext_base", "8gb"],
        use_tensorboard=True,
        log_interval=10,
        eval_interval=1,
        output_dir="./outputs",
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
        verbose=True,
    )

    # Interpretability config
    interpretability_config = InterpretabilityConfig(
        methods=["integrated_gradients", "gradcam", "occlusion"],
        ig_steps=50,
        occlusion_size=15,
        save_visualizations=True,
        visualization_dir="./interpretations",
        num_examples_per_class=5,
    )

    # Config completo
    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
        logging=logging_config,
        interpretability=interpretability_config,
        seed=42,
        device="cuda",
        experiment_name="banana_rtx4060_experiment",
    )

    return config


def get_rtx4060_light_config() -> Config:
    """
    Configuración ligera para RTX 4060 (más rápida, menos precisa).

    Usar para:
    - Pruebas rápidas
    - Datasets pequeños
    - Cuando necesitas velocidad sobre precisión

    Returns:
        Config ligero para RTX 4060
    """
    config = get_rtx4060_config()

    # Modelo más pequeño
    config.model.name = "convnext_tiny"

    # Batch size más grande (modelo pequeño)
    config.training.batch_size = 64

    # Menos épocas
    config.training.epochs = 30

    # Augmentation más ligero
    config.data.augmentation_level = "light"

    config.experiment_name = "banana_rtx4060_light"

    return config


def get_rtx4060_heavy_config() -> Config:
    """
    Configuración pesada para RTX 4060 (máxima precisión).

    Usar para:
    - Producción
    - Cuando necesitas máxima precisión
    - Dataset grande y balanceado

    Returns:
        Config pesado para RTX 4060
    """
    config = get_rtx4060_config()

    # Modelo más grande (usa más VRAM)
    config.model.name = "convnext_large"

    # Batch size más pequeño (modelo grande)
    config.training.batch_size = 16
    config.training.gradient_accumulation_steps = 2  # Efectivo = 32

    # Más épocas
    config.training.epochs = 100

    # Augmentation más agresivo
    config.data.augmentation_level = "heavy"

    # Early stopping más paciente
    config.training.early_stopping_patience = 15

    config.experiment_name = "banana_rtx4060_heavy"

    return config


# Tabla de referencia de modelos vs VRAM
MODEL_VRAM_REQUIREMENTS = {
    "convnext_tiny": {
        "vram_gb": 3.5,
        "batch_size_8gb": 64,
        "speed": "Muy rápido",
        "accuracy": "92-95%",
    },
    "convnext_small": {
        "vram_gb": 4.5,
        "batch_size_8gb": 48,
        "speed": "Rápido",
        "accuracy": "94-96%",
    },
    "convnext_base": {
        "vram_gb": 5.5,
        "batch_size_8gb": 32,
        "speed": "Medio",
        "accuracy": "95-97%",
    },
    "convnext_large": {
        "vram_gb": 7.5,
        "batch_size_8gb": 16,
        "speed": "Lento",
        "accuracy": "96-98%",
    },
    "vit_base_patch16_224": {
        "vram_gb": 6.0,
        "batch_size_8gb": 32,
        "speed": "Medio",
        "accuracy": "94-97%",
    },
    "efficientnetv2_m": {
        "vram_gb": 5.0,
        "batch_size_8gb": 40,
        "speed": "Rápido",
        "accuracy": "94-96%",
    },
}


def print_model_recommendations():
    """Imprime recomendaciones de modelos para RTX 4060."""
    print("\n" + "=" * 70)
    print("RECOMENDACIONES DE MODELOS PARA RTX 4060 (8GB)")
    print("=" * 70)
    print(f"{'Modelo':<25} {'VRAM':>8} {'Batch':>8} {'Velocidad':>12} {'Accuracy':>12}")
    print("-" * 70)

    for model, specs in MODEL_VRAM_REQUIREMENTS.items():
        print(
            f"{model:<25} "
            f"{specs['vram_gb']:>6.1f}GB "
            f"{specs['batch_size_8gb']:>8d} "
            f"{specs['speed']:>12s} "
            f"{specs['accuracy']:>12s}"
        )

    print("-" * 70)
    print("\n💡 RECOMENDACIONES:")
    print("  • Para pruebas rápidas: convnext_tiny (batch_size=64)")
    print("  • Para balance óptimo: convnext_base (batch_size=32) ✅")
    print("  • Para máxima precisión: convnext_large (batch_size=16)")
    print("\n⚠️  IMPORTANTE:")
    print("  • Mixed precision SIEMPRE activado (use_amp=True)")
    print("  • Si OOM: reducir batch_size o usar gradient_accumulation")
    print("=" * 70 + "\n")


# Ejemplo de uso
if __name__ == "__main__":
    # Mostrar recomendaciones
    print_model_recommendations()

    # Config estándar
    config = get_rtx4060_config()
    print("\n📋 Configuración Estándar RTX 4060:")
    print(f"  Modelo: {config.model.name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Mixed precision: {config.training.use_amp}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")

    # Config ligera
    config_light = get_rtx4060_light_config()
    print("\n⚡ Configuración Ligera RTX 4060:")
    print(f"  Modelo: {config_light.model.name}")
    print(f"  Batch size: {config_light.training.batch_size}")

    # Config pesada
    config_heavy = get_rtx4060_heavy_config()
    print("\n🔥 Configuración Pesada RTX 4060:")
    print(f"  Modelo: {config_heavy.model.name}")
    print(f"  Batch size: {config_heavy.training.batch_size}")
    print(
        f"  Gradient accumulation: {config_heavy.training.gradient_accumulation_steps}"
    )
