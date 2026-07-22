"""
Leaf and lesion segmentation orchestration.

Combines the low-level HSV mask primitives in hsv_mask.py into a single
per-image segmentation step, and provides the pixel-count utilities that
severity_generator.py turns into a severity percentage. Kept separate from
hsv_mask.py so the hue/saturation/value threshold logic (which is what would
need retuning if a different dataset were used) stays isolated from the
orchestration and counting logic (which would not).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.hsv_mask import compute_leaf_mask, compute_lesion_mask


@dataclass
class SegmentationResult:
    """
    Result of segmenting a single RGB leaf image.

    Attributes:
        leaf_mask: Boolean mask (H, W); True where the pixel is leaf tissue
            (healthy or diseased).
        lesion_mask: Boolean mask (H, W); True where the pixel is diseased
            leaf tissue.
        leaf_pixel_count: Total number of leaf-tissue pixels.
        lesion_pixel_count: Total number of diseased-tissue pixels.
    """

    leaf_mask: np.ndarray
    lesion_mask: np.ndarray
    leaf_pixel_count: int
    lesion_pixel_count: int


def segment_leaf_and_lesion(rgb_image: np.ndarray) -> SegmentationResult:
    """
    Segment a single RGB leaf image into leaf and lesion masks, and count
    the pixels in each.

    Args:
        rgb_image: RGB image array of shape (H, W, 3), dtype uint8.

    Returns:
        SegmentationResult with both masks and their pixel counts.
    """
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    leaf_mask = compute_leaf_mask(hsv_image)
    lesion_mask = compute_lesion_mask(hsv_image, leaf_mask)

    return SegmentationResult(
        leaf_mask=leaf_mask,
        lesion_mask=lesion_mask,
        leaf_pixel_count=int(leaf_mask.sum()),
        lesion_pixel_count=int(lesion_mask.sum()),
    )
