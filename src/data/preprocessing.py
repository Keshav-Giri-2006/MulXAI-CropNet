"""
Data preprocessing utilities for PlantVillage dataset.

Handles image normalization, conversion, and standardization.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from pathlib import Path


class ImagePreprocessor:
    """Handles image preprocessing operations."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Initialize ImagePreprocessor.

        Args:
            target_size: Target image size (H, W). Default: (224, 224)
            normalize: Whether to apply normalization. Default: True
            mean: ImageNet mean values. Default: (0.485, 0.456, 0.406)
            std: ImageNet std values. Default: (0.229, 0.224, 0.225)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean)
        self.std = np.array(std)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W, 3)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]))

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.

        Args:
            image: Input image in range [0, 255]

        Returns:
            Normalized image in range [-1, 1]
        """
        if not self.normalize:
            return image.astype(np.float32) / 255.0

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image
        """
        image = self.load_image(image_path)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image


def validate_image(image_path: str) -> bool:
    """
    Validate if image file is readable and valid.

    Args:
        image_path: Path to image file

    Returns:
        True if valid, False otherwise
    """
    try:
        image = cv2.imread(image_path)
        return image is not None and image.size > 0
    except Exception:
        return False


def get_image_statistics(
    image_dir: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate mean and std of images in directory.

    Args:
        image_dir: Path to directory containing images

    Returns:
        Tuple of (mean, std) with 3 channels each
    """
    images = []
    image_path = Path(image_dir)

    for img_file in image_path.glob("**/*.jpg"):
        img = cv2.imread(str(img_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32) / 255.0)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    images = np.array(images)
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))

    return tuple(mean), tuple(std)
