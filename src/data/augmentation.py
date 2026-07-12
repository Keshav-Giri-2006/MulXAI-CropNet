"""
Advanced augmentation strategies for disease classification.

Implements multiple augmentation pipelines for different training scenarios.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Callable


class AugmentationFactory:
    """Factory for creating different augmentation pipelines."""

    @staticmethod
    def get_light_augmentation() -> A.Compose:
        """Minimal augmentation for validation/test."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_moderate_augmentation() -> A.Compose:
        """Moderate augmentation for standard training (approved pipeline only)."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomResizedCrop(224, 224, scale=(0.9, 1.0), p=0.2),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_strong_augmentation() -> A.Compose:
        """Strong augmentation for robust training (approved pipeline only)."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.3),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_custom_augmentation(
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.3,
        rotation: int = 20,
        brightness_contrast: float = 0.3,
    ) -> A.Compose:
        """
        Create custom augmentation pipeline.

        Args:
            horizontal_flip: Probability of horizontal flip
            vertical_flip: Probability of vertical flip
            rotation: Maximum rotation angle
            brightness_contrast: Probability of brightness/contrast adjustment

        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.HorizontalFlip(p=horizontal_flip),
            A.VerticalFlip(p=vertical_flip),
            A.Rotate(limit=rotation, p=0.5),
            A.RandomBrightnessContrast(p=brightness_contrast),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])


def get_augmentation_by_name(name: str) -> A.Compose:
    """
    Get augmentation pipeline by name.

    Args:
        name: Name of augmentation ('light', 'moderate', 'strong')

    Returns:
        Albumentations Compose object

    Raises:
        ValueError: If name is not recognized
    """
    augmentations = {
        'light': AugmentationFactory.get_light_augmentation,
        'moderate': AugmentationFactory.get_moderate_augmentation,
        'strong': AugmentationFactory.get_strong_augmentation,
    }

    if name not in augmentations:
        raise ValueError(f"Unknown augmentation: {name}")

    return augmentations[name]()
