"""
Trainer profesional para clasificación de enfermedades de banano.
Incluye todas las optimizaciones modernas de PyTorch.
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

# Wandb para tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb no disponible. Instalar con: pip install wandb")


class Trainer:
    """
    Trainer profesional con todas las features modernas.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Experiment tracking (Wandb)
    - Progress bars (tqdm)
    - Metrics tracking

    Args:
        model: Modelo PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (opcional)
        config: Objeto Config con configuraciones
        device: Device para entrenamiento
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        config=None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Mixed precision training
        self.use_amp = config.training.use_amp if config else True

        # Configurar GradScaler según el dispositivo
        if self.use_amp:
            if "cuda" in device:
                self.scaler = GradScaler("cuda")
            elif "mps" in device:
                # MPS no soporta GradScaler, desactivar AMP
                self.use_amp = False
                self.scaler = None
                print("[INFO] AMP desactivado en MPS (no soportado)")
            else:
                self.scaler = None
        else:
            self.scaler = None

        # Gradient accumulation
        self.grad_accum_steps = (
            config.training.gradient_accumulation_steps if config else 1
        )

        # Gradient clipping
        self.max_grad_norm = config.training.max_grad_norm if config else 1.0

        # Early stopping
        self.early_stopping_patience = (
            config.training.early_stopping_patience if config else 10
        )
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # Checkpointing
        self.checkpoint_dir = (
            Path(config.logging.checkpoint_dir) if config else Path("./checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = config.training.save_best_only if config else True

        # History tracking
        self.history = defaultdict(list)
        self.current_epoch = 0
        self.global_step = 0

        # Wandb tracking
        self.use_wandb = (
            config.logging.use_wandb if config and WANDB_AVAILABLE else False
        )
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Inicializa Wandb para experiment tracking."""
        wandb.init(
            project=self.config.logging.wandb_project,
            entity=self.config.logging.wandb_entity,
            name=self.config.experiment_name,
            tags=self.config.logging.wandb_tags,
            config=self.config.to_dict(),
        )
        wandb.watch(self.model, log="all", log_freq=100)

    def train_epoch(self) -> Dict[str, float]:
        """
        Entrena una época.

        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1} [Train]",
        )

        for batch_idx, (images, labels) in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass con mixed precision
            device_type = "cuda" if "cuda" in self.device else "cpu"
            with autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Normalizar loss si usamos gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation: solo actualizar cada N steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Métricas
            running_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Actualizar progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix(
                {"loss": running_loss / (batch_idx + 1), "acc": f"{current_acc:.2f}%"}
            )

            # Log a wandb cada N batches
            if (
                self.use_wandb
                and (batch_idx + 1) % self.config.logging.log_interval == 0
            ):
                wandb.log(
                    {
                        "train/batch_loss": loss.item() * self.grad_accum_steps,
                        "train/batch_acc": current_acc,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    }
                )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return {"train_loss": epoch_loss, "train_acc": epoch_acc}

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Valida una época.

        Returns:
            Diccionario con métricas de validación
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Para métricas por clase
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (sin gradient)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Métricas por clase
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

            # Actualizar progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix(
                {
                    "loss": running_loss / len(self.val_loader),
                    "acc": f"{current_acc:.2f}%",
                }
            )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        # Accuracy por clase
        per_class_acc = {
            cls: 100.0 * class_correct[cls] / class_total[cls]
            for cls in class_total.keys()
        }

        return {
            "val_loss": epoch_loss,
            "val_acc": epoch_acc,
            "per_class_acc": per_class_acc,
        }

    def train(self, epochs: int):
        """
        Loop principal de entrenamiento.

        Args:
            epochs: Número de épocas a entrenar
        """
        print("\n" + "=" * 70)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 70)
        print(f"Épocas: {epochs}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.grad_accum_steps} steps")
        print("=" * 70 + "\n")

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Entrenar
            train_metrics = self.train_epoch()

            # Validar
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Guardar métricas
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_acc"].append(train_metrics["train_acc"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Log a wandb
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["train_loss"],
                    "train/acc": train_metrics["train_acc"],
                    "val/loss": val_metrics["val_loss"],
                    "val/acc": val_metrics["val_acc"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }

                # Agregar accuracy por clase
                for cls_idx, acc in val_metrics["per_class_acc"].items():
                    class_name = self.config.data.classes[cls_idx]
                    log_dict[f"val/acc_{class_name}"] = acc

                wandb.log(log_dict)

            # Print resumen con formato mejorado
            print(f"\n{'='*70}")
            print(f"📊 Época {epoch + 1}/{epochs}")
            print(f"{'='*70}")
            print(f"🔹 Train → Loss: {train_metrics['train_loss']:.4f} | Acc: {train_metrics['train_acc']:.2f}%")
            print(f"🔸 Val   → Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_acc']:.2f}%")
            print(f"⚙️  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Checkpointing
            is_best = val_metrics["val_acc"] > self.best_val_acc

            if is_best:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_val_loss = val_metrics["val_loss"]
                self.early_stopping_counter = 0

                if self.save_best_only:
                    self.save_checkpoint("best_model.pth", is_best=True)
                    print(f"✅ Mejor modelo guardado (Acc: {self.best_val_acc:.2f}%)")
            else:
                self.early_stopping_counter += 1
                print(f"⏳ Sin mejora ({self.early_stopping_counter}/{self.early_stopping_patience})")

            # Guardar checkpoint regular
            if (
                not self.save_best_only
                and (epoch + 1) % self.config.training.save_frequency == 0
            ):
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping activado después de {epoch + 1} épocas")
                print(f"  Mejor Val Acc: {self.best_val_acc:.2f}%")
                break

        # Finalizar
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time/60:.2f} minutos")
        print(f"Mejor Val Acc: {self.best_val_acc:.2f}%")
        print(f"Mejor Val Loss: {self.best_val_loss:.4f}")
        print("=" * 70 + "\n")

        # Guardar history
        self.save_history()

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Guarda checkpoint del modelo.

        Args:
            filename: Nombre del archivo
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "history": dict(self.history),
            "config": self.config.to_dict() if self.config else None,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            # También guardar como 'best_model.pth'
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Carga checkpoint para continuar entrenamiento.

        Args:
            checkpoint_path: Ruta al checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = defaultdict(list, checkpoint.get("history", {}))

        print(f"[INFO] Checkpoint cargado desde época {self.current_epoch}")

    def save_history(self):
        """Guarda history a JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"[INFO] History guardado en {history_path}")


# Ejemplo de uso
if __name__ == "__main__":
    from models.classifier import BananaClassifier
    from torch.optim import AdamW

    # Crear modelo dummy
    model = BananaClassifier(num_classes=4)

    # Crear data loaders dummy
    train_loader = [
        (torch.randn(8, 3, 224, 224), torch.randint(0, 4, (8,))) for _ in range(10)
    ]
    val_loader = [
        (torch.randn(8, 3, 224, 224), torch.randint(0, 4, (8,))) for _ in range(5)
    ]

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Entrenar
    # trainer.train(epochs=2)
