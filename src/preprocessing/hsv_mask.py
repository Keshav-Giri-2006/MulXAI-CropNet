"""
HSV-based mask primitives for severity pseudo-label generation.

Implements the classical (non-learned) leaf/lesion segmentation described in
MEMBER2_SEVERITY_GUIDE.md ("PSEUDO LABEL GENERATION"): RGB -> HSV -> leaf
mask -> lesion mask -> infected-pixel ratio. No trained segmentation network
is used anywhere in this module, per the guide's explicit "DO NOT use U-Net /
YOLO / Mask R-CNN / segmentation networks" constraint.

This module only computes boolean pixel masks from an already-loaded HSV
image array. It has no file I/O and no dependency on any Member 1 module,
so it can be unit-tested in isolation with synthetic arrays.
"""

import cv2
import numpy as np

# Hue range for healthy tomato leaf tissue in OpenCV HSV space (H: 0-179).
# Tuned to cover typical leaf greens without over-including yellow chlorosis,
# which should be counted as diseased tissue.
GREEN_HUE_LOW = 25
GREEN_HUE_HIGH = 95

# Minimum saturation/value for a pixel to be considered "green tissue" at
# all (very low saturation/value pixels are likely background or shadow,
# not confidently healthy leaf, so they are left for the leaf mask step to
# decide via the Otsu threshold rather than being force-classified green).
GREEN_MIN_SATURATION = 40
GREEN_MIN_VALUE = 40

# Morphological kernel sizes. Leaf mask uses a larger kernel to close gaps
# from leaf veins/glare and remove small background speckle; lesion mask
# uses a smaller kernel since lesions can be legitimately small.
_LEAF_KERNEL = np.ones((5, 5), np.uint8)
_LESION_KERNEL = np.ones((3, 3), np.uint8)


def compute_leaf_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Segment leaf tissue (healthy + diseased) from background.

    PlantVillage images use a plain, low-saturation background behind the
    leaf. Otsu thresholding on the saturation channel separates the two
    without a fixed, hand-tuned saturation cutoff, so it adapts per-image
    to lighting variation. Morphological closing then opening removes small
    holes (e.g. leaf-vein glare) and small background speckle.

    Args:
        hsv_image: HSV image array of shape (H, W, 3), dtype uint8
            (OpenCV convention: H in [0,179], S and V in [0,255]).

    Returns:
        Boolean mask of shape (H, W); True where the pixel is leaf tissue.
    """
    saturation = hsv_image[:, :, 1]

    # Use OpenCV's own binarized output (applies src > thresh) rather than
    # recomputing the comparison manually with `saturation >= otsu_thresh`.
    # The two are not equivalent at the boundary: when the background is
    # uniformly zero-saturation, Otsu can select a threshold of exactly 0,
    # and `>=` would then include every zero-saturation background pixel
    # as well, since 0 >= 0. `>` (what THRESH_BINARY actually applies)
    # correctly excludes them.
    _, raw_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, _LEAF_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _LEAF_KERNEL)

    return cleaned > 0


def compute_green_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Identify healthy (green) leaf tissue within an HSV image.

    Args:
        hsv_image: HSV image array of shape (H, W, 3), dtype uint8.

    Returns:
        Boolean mask of shape (H, W); True where the pixel is healthy green
        tissue.
    """
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]

    green = (
        (hue >= GREEN_HUE_LOW)
        & (hue <= GREEN_HUE_HIGH)
        & (saturation >= GREEN_MIN_SATURATION)
        & (value >= GREEN_MIN_VALUE)
    )
    return green


def compute_lesion_mask(hsv_image: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Identify diseased (lesion) tissue within the leaf area.

    Healthy tomato leaf tissue is reliably green in HSV space. Lesions
    (bacterial spot, blight, leaf mold, mosaic virus discoloration, necrotic
    spots, etc.) manifest as brown, yellow, or near-black discoloration --
    i.e. leaf-mask pixels that fall outside the green hue band. This keeps
    the rule simple and directly testable, per the "avoid unnecessary
    abstraction" preference for this research codebase, rather than
    maintaining several overlapping brown/yellow/dark hue windows that would
    need independent tuning and justification.

    Args:
        hsv_image: HSV image array of shape (H, W, 3), dtype uint8.
        leaf_mask: Boolean leaf mask from compute_leaf_mask(), same (H, W).

    Returns:
        Boolean mask of shape (H, W); True where the pixel is diseased leaf
        tissue.
    """
    green_mask = compute_green_mask(hsv_image)
    diseased = leaf_mask & (~green_mask)

    raw_mask = diseased.astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, _LESION_KERNEL)

    return cleaned > 0
