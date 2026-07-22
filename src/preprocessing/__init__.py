"""Severity pseudo-label generation package (Member 2).

Classical HSV-threshold based leaf/lesion segmentation -- no learned
segmentation network is used, per MEMBER2_SEVERITY_GUIDE.md.
"""

from src.preprocessing.hsv_mask import (
    compute_green_mask,
    compute_leaf_mask,
    compute_lesion_mask,
)
from src.preprocessing.leaf_segmentation import SegmentationResult, segment_leaf_and_lesion
from src.preprocessing.pseudo_labels import generate_severity_labels, is_healthy_class
from src.preprocessing.severity_generator import (
    compute_severity,
    compute_severity_for_file,
    load_rgb_image,
)

__all__ = [
    "compute_green_mask",
    "compute_leaf_mask",
    "compute_lesion_mask",
    "SegmentationResult",
    "segment_leaf_and_lesion",
    "generate_severity_labels",
    "is_healthy_class",
    "compute_severity",
    "compute_severity_for_file",
    "load_rgb_image",
]
