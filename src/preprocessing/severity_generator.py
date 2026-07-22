"""
Continuous severity percentage generation for a single leaf image.

Implements the formula given in MEMBER2_SEVERITY_GUIDE.md:

    Severity = (Infected Pixels / Total Leaf Pixels) x 100

Output is always a continuous float in [0, 100] -- categorical severity
buckets (Mild/Moderate/Severe) are explicitly forbidden by DATASET_USAGE.md
and are never produced anywhere in this module.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.leaf_segmentation import segment_leaf_and_lesion


def load_rgb_image(image_path: str) -> np.ndarray:
    """
    Load an image file as an RGB uint8 array.

    Uses the same PIL-based loading convention (Image.open(...).convert
    ("RGB")) as src/training/dataset_loader.py's PlantVillageDataset, so
    pixel values are read identically to how Member 1's pipeline reads them
    -- this module only differs in what it does with the pixels afterward
    (HSV masking instead of Albumentations/normalization).

    Args:
        image_path: Path to an image file.

    Returns:
        RGB image array of shape (H, W, 3), dtype uint8.
    """
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def compute_severity(rgb_image: np.ndarray, is_healthy: bool = False) -> float:
    """
    Compute a continuous severity percentage (0-100) for a single leaf image.

    Args:
        rgb_image: RGB image array of shape (H, W, 3), dtype uint8.
        is_healthy: True if the image's ground-truth disease label is the
            "healthy" class. Healthy-class images are assigned severity 0.0
            directly rather than run through lesion detection: HSV threshold
            noise (e.g. leaf-vein shadow, minor color variation) can produce
            a small spurious lesion count even on a genuinely healthy leaf,
            and the guide's own worked example defines Healthy -> 0%
            unconditionally. Default: False.

    Returns:
        Severity percentage in [0, 100].
    """
    if is_healthy:
        return 0.0

    result = segment_leaf_and_lesion(rgb_image)

    if result.leaf_pixel_count == 0:
        # Segmentation found no leaf pixels at all (e.g. a corrupt or
        # entirely-background image). There is no meaningful severity to
        # report, so this is treated the same as "no visible disease"
        # rather than raising, so a single bad image doesn't halt a batch
        # run over thousands of files.
        return 0.0

    severity = (result.lesion_pixel_count / result.leaf_pixel_count) * 100.0
    return float(np.clip(severity, 0.0, 100.0))


def compute_severity_for_file(image_path: str, is_healthy: bool = False) -> float:
    """
    Load an image from disk and compute its severity percentage.

    Args:
        image_path: Path to an image file.
        is_healthy: See compute_severity(). Default: False.

    Returns:
        Severity percentage in [0, 100].
    """
    rgb_image = load_rgb_image(image_path)
    return compute_severity(rgb_image, is_healthy=is_healthy)
