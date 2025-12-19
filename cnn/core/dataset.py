"""
Dataset classes para clasificación de enfermedades de banano.
Compatible con PyTorch DataLoader y Albumentations.
"""

import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import warnings

# Suprimir warnings de PIL para imágenes con problemas menores de formato
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


class BananaDiseaseDataset(Dataset):
    """
    Dataset para clasificación de enfermedades de banano.

    Estructura esperada:
        data_dir/
            cordana/
                img1.jpg
                img2.jpg
            healthy/
                img1.jpg
            pestalotiopsis/
                ...
            sigatoka/
                ...

    Args:
        root_dir: Directorio raíz con subdirectorios por clase
        transform: Transformaciones Albumentations
        classes: Lista de nombres de clases (orden importa para labels)
        cache_images: Si True, carga todas las imágenes en RAM (rápido pero requiere memoria)
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
        cache_images: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_images = cache_images

        # Detectar clases automáticamente o usar las proporcionadas
        if classes is None:
            self.classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.classes = classes

        # Crear mapeo clase -> índice
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Cargar rutas de imágenes y labels
        self.image_paths = []
        self.labels = []
        self._load_image_paths()

        # Cache opcional de imágenes
        self.image_cache = {}
        if self.cache_images:
            self._cache_all_images()

        # Estadísticas del dataset
        self._compute_class_weights()

    def _load_image_paths(self):
        """Carga rutas de todas las imágenes y sus labels."""
        for class_name in self.classes:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                print(
                    f"[WARNING] Clase '{class_name}' no encontrada en {self.root_dir}"
                )
                continue

            # Buscar todas las imágenes válidas
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.image_paths.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {self.root_dir}")

        print(f"[INFO] Dataset cargado: {len(self.image_paths)} imágenes")
        print(f"[INFO] Clases ({len(self.classes)}): {self.classes}")

    def _cache_all_images(self):
        """Carga todas las imágenes en memoria (usa mucha RAM)."""
        print("[INFO] Cacheando imágenes en memoria...")
        for idx, img_path in enumerate(self.image_paths):
            self.image_cache[idx] = self._load_image(img_path)
            if (idx + 1) % 500 == 0:
                print(f"  Cacheadas {idx + 1}/{len(self.image_paths)} imágenes")
        print("[INFO] Cache completado")

    def _load_image(self, path: str) -> np.ndarray:
        """
        Carga imagen desde disco.

        Args:
            path: Ruta a la imagen

        Returns:
            Array numpy en formato RGB
        """
        try:
            # Usar PIL para mejor compatibilidad
            img = Image.open(path).convert("RGB")
            return np.array(img)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {path}: {e}")
            # Retornar imagen negra como fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def _compute_class_weights(self):
        """Calcula pesos de clase para balanceo (útil para loss weighted)."""
        from collections import Counter

        label_counts = Counter(self.labels)
        total_samples = len(self.labels)

        # Peso inversamente proporcional a frecuencia
        self.class_weights = {
            cls_idx: total_samples / (len(self.classes) * count)
            for cls_idx, count in label_counts.items()
        }

        # Estadísticas
        self.class_distribution = {
            self.classes[cls_idx]: count for cls_idx, count in label_counts.items()
        }

    def get_class_weights_tensor(self) -> torch.Tensor:
        """Retorna pesos de clase como tensor para loss function."""
        weights = [self.class_weights[i] for i in range(len(self.classes))]
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retorna un sample del dataset.

        Args:
            idx: Índice del sample

        Returns:
            Tupla (imagen_transformada, label)
        """
        # Obtener imagen (desde cache o disco)
        if self.cache_images:
            image = self.image_cache[idx].copy()
        else:
            image = self._load_image(self.image_paths[idx])

        label = self.labels[idx]

        # Aplicar transformaciones (Albumentations)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Si no hay transform, convertir a tensor manualmente
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_sample_by_class(
        self, class_name: str, n: int = 1
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Obtiene n samples aleatorios de una clase específica.

        Args:
            class_name: Nombre de la clase
            n: Número de samples a retornar

        Returns:
            Lista de tuplas (imagen, label)
        """
        if class_name not in self.classes:
            raise ValueError(f"Clase '{class_name}' no encontrada")

        class_idx = self.class_to_idx[class_name]
        indices = [i for i, label in enumerate(self.labels) if label == class_idx]

        if len(indices) < n:
            print(f"[WARNING] Solo hay {len(indices)} samples de '{class_name}'")
            n = len(indices)

        selected_indices = np.random.choice(indices, size=n, replace=False)
        return [self[idx] for idx in selected_indices]

    def get_statistics(self) -> Dict:
        """Retorna estadísticas del dataset."""
        return {
            "total_samples": len(self),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "class_distribution": self.class_distribution,
            "class_weights": self.class_weights,
            "imbalance_ratio": max(self.class_distribution.values())
            / min(self.class_distribution.values()),
        }

    def print_statistics(self):
        """Imprime estadísticas del dataset."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DEL DATASET")
        print("=" * 60)
        print(f"Total de imágenes: {stats['total_samples']}")
        print(f"Número de clases: {stats['num_classes']}")
        print(f"\nDistribución por clase:")
        for cls, count in stats["class_distribution"].items():
            percentage = (count / stats["total_samples"]) * 100
            print(f"  {cls:20s}: {count:5d} ({percentage:5.2f}%)")
        print(f"\nRatio de desbalanceo: {stats['imbalance_ratio']:.2f}x")
        print("=" * 60 + "\n")


def create_data_splits(
    data_dir: str,
    classes: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Crea splits estratificados train/val/test desde un directorio.

    Args:
        data_dir: Directorio con subdirectorios por clase
        classes: Lista de nombres de clases
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        random_state: Semilla para reproducibilidad

    Returns:
        Tupla con (train_paths, val_paths, test_paths)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    data_dir = Path(data_dir)
    all_paths = []
    all_labels = []

    # Recolectar todas las rutas
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in BananaDiseaseDataset.VALID_EXTENSIONS:
                all_paths.append(str(img_path))
                all_labels.append(class_name)

    # Split estratificado train/temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=random_state,
    )

    # Split estratificado val/test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state,
    )

    print(f"[INFO] Splits creados:")
    print(f"  Train: {len(train_paths)} imágenes")
    print(f"  Val:   {len(val_paths)} imágenes")
    print(f"  Test:  {len(test_paths)} imágenes")

    return train_paths, val_paths, test_paths


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders para train, val y opcionalmente test.

    Args:
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validación
        test_dataset: Dataset de test (opcional)
        batch_size: Tamaño de batch
        num_workers: Número de workers para carga paralela
        pin_memory: Pin memory para GPU (acelera transferencia)

    Returns:
        Diccionario con dataloaders
    """
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Evita batch incompleto al final
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return dataloaders


# Ejemplo de uso
if __name__ == "__main__":
    from cnn.core.transforms import get_transforms

    # Crear dataset de prueba
    data_dir = "./data/train"
    classes = ["cordana", "fusarium_r4t", "healthy", "moko", "pestalotiopsis", "sigatoka"]

    # Transformaciones
    train_transform = get_transforms("train", augmentation_level="medium")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = BananaDiseaseDataset(
        data_dir, transform=train_transform, classes=classes
    )

    # Estadísticas
    train_dataset.print_statistics()

    # Obtener sample
    img, label = train_dataset[0]
    print(f"Sample shape: {img.shape}, Label: {label} ({classes[label]})")

    # Pesos de clase
    class_weights = train_dataset.get_class_weights_tensor()
    print(f"Class weights: {class_weights}")
