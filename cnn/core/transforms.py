"""
Data augmentation para agricultura usando Albumentations.
Compatible con todas las versiones de Albumentations.

IMPORTANTE: Este módulo usa augmentation on-the-fly (no guarda imágenes).
Las transformaciones se aplican en tiempo real durante el entrenamiento.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class AugmentationPolicy:
    """
    Políticas de augmentation para imágenes de enfermedades de plantas.
    Tres niveles: light, medium, heavy
    """

    @staticmethod
    def get_train_transforms_light(image_size: tuple = (224, 224)):
        """
        Augmentation LIGERO - Para datasets grandes (>3000 imágenes).

        Transformaciones mínimas que preservan características críticas.
        """
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        return A.Compose(
            [
                # Resize y crop - COMPATIBLE CON TODAS LAS VERSIONES
                A.Resize(height=h, width=w, p=1.0),
                # Flips
                A.HorizontalFlip(p=0.5),
                # Transformaciones geométricas suaves
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Normalización ImageNet (crítico para modelos preentrenados)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_train_transforms_medium(image_size: tuple = (224, 224)):
        """
        Augmentation MEDIO - RECOMENDADO para datasets medianos (1k-3k imágenes).

        Balance entre preservar características y variedad.
        Optimizado para enfermedades de plantas en campo.
        """
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        return A.Compose(
            [
                # Resize - COMPATIBLE
                A.Resize(height=h, width=w, p=1.0),
                # Flips (plantas pueden verse desde cualquier ángulo)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                # Transformaciones geométricas
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.6,
                ),
                # Distorsiones (perspectiva de cámara)
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=20, p=1.0),
                    ],
                    p=0.3,
                ),
                # Variaciones de color (importantes para agricultura)
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=20,
                            val_shift_limit=15,
                            p=1.0,
                        ),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                        ),
                    ],
                    p=0.5,
                ),
                # Noise y blur (simula condiciones reales)
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=5, p=1.0),
                    ],
                    p=0.3,
                ),
                # Sombras (crítico para agricultura)
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.3,
                ),
                # Dropout (oclusiones parciales)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(h * 0.1),
                    max_width=int(w * 0.1),
                    min_holes=1,
                    min_height=int(h * 0.05),
                    min_width=int(w * 0.05),
                    fill_value=0,
                    p=0.3,
                ),
                # Normalización ImageNet
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_train_transforms_heavy(image_size: tuple = (224, 224)):
        """
        Augmentation FUERTE - Para datasets pequeños (<1000 imágenes).

        Máxima variedad para evitar overfitting.
        """
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        return A.Compose(
            [
                # Resize - COMPATIBLE
                A.Resize(height=h, width=w, p=1.0),
                # Flips agresivos
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Transformaciones geométricas agresivas
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.25,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7,
                ),
                # Perspectiva (simula ángulos de cámara)
                A.Perspective(scale=(0.05, 0.15), p=0.4),
                # Distorsiones múltiples
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.2, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=30, p=1.0),
                    ],
                    p=0.5,
                ),
                # Variaciones de color agresivas
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=25,
                            p=1.0,
                        ),
                        A.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=1.0
                        ),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.6,
                ),
                # Noise y blur variados
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                    ],
                    p=0.4,
                ),
                # Efectos climáticos
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=0.4,
                ),
                A.RandomRain(
                    slant_range=(-10, 10),
                    drop_length=15,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3,
                    brightness_coefficient=0.9,
                    rain_type="drizzle",
                    p=0.2,
                ),
                # Compresión (simula fotos de celular)
                A.ImageCompression(quality_range=(60, 100), p=0.3),
                # Dropout agresivo
                A.CoarseDropout(
                    num_holes_range=(3, 12),
                    hole_height_range=(int(h * 0.05), int(h * 0.15)),
                    hole_width_range=(int(w * 0.05), int(w * 0.15)),
                    fill_value=0,
                    p=0.4,
                ),
                # Normalización ImageNet
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: tuple = (224, 224)):
        """
        Transformaciones para validación/test.

        Solo preprocessing necesario:
        - Resize center crop
        - Normalización
        """
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        return A.Compose(
            [
                A.Resize(
                    height=int(h * 1.14), width=int(w * 1.14), p=1.0  # 256 para 224
                ),
                A.CenterCrop(height=h, width=w, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_test_time_augmentation() -> list:
        """
        Test-Time Augmentation (TTA) - mejora predicciones finales.

        Genera múltiples versiones de la imagen y promedia predicciones.
        Mejora típica: +1-2% accuracy.
        """
        image_size = (224, 224)
        h, w = image_size

        return [
            # Original
            A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            ),
            # Flip horizontal
            A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            ),
            # Crop izquierda
            A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            ),
            # Crop derecha
            A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            ),
        ]


def get_transforms(
    mode: str = "train",
    image_size: tuple = (224, 224),
    augmentation_level: str = "medium",
):
    """
    Factory function para obtener transformaciones.

    Args:
        mode: 'train' o 'val'
        image_size: Tuple (height, width)
        augmentation_level: 'light', 'medium', 'heavy'

    Returns:
        Albumentations Compose object
    """
    policy = AugmentationPolicy()

    if mode == "train":
        if augmentation_level == "light":
            return policy.get_train_transforms_light(image_size)
        elif augmentation_level == "medium":
            return policy.get_train_transforms_medium(image_size)
        elif augmentation_level == "heavy":
            return policy.get_train_transforms_heavy(image_size)
        else:
            raise ValueError(
                f"augmentation_level debe ser 'light', 'medium' o 'heavy', got {augmentation_level}"
            )

    elif mode in ["val", "test"]:
        return policy.get_val_transforms(image_size)

    else:
        raise ValueError(f"mode debe ser 'train', 'val' o 'test', got {mode}")


# Exportar para uso externo
__all__ = ["AugmentationPolicy", "get_transforms", "ToTensorV2"]
