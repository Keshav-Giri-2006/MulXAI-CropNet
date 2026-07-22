"""
Tests for the Milestone 1 severity pseudo-label pipeline
(src/preprocessing/hsv_mask.py, leaf_segmentation.py, severity_generator.py,
pseudo_labels.py).

Uses small synthetic, HSV-colored images rather than real PlantVillage
photos (which are git-ignored per PROJECT_STRUCTURE.md and not available in
this environment), in the same style as tests/test_cross_validation.py's
synthetic-dataset fixture.
"""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.hsv_mask import (
    compute_green_mask,
    compute_leaf_mask,
    compute_lesion_mask,
)
from src.preprocessing.leaf_segmentation import segment_leaf_and_lesion
from src.preprocessing.pseudo_labels import generate_severity_labels, is_healthy_class
from src.preprocessing.severity_generator import compute_severity, load_rgb_image


def _make_synthetic_leaf(
    size: int = 100,
    background_gray: int = 200,
    leaf_color=(34, 139, 34),  # forest green
    lesion_color=None,
    lesion_fraction: float = 0.0,
) -> np.ndarray:
    """
    Build a synthetic RGB image: a solid low-saturation gray background with
    a centered square "leaf" of a solid color, optionally with a smaller
    solid-color "lesion" patch inside it covering roughly lesion_fraction of
    the leaf square's area.
    """
    image = np.full((size, size, 3), background_gray, dtype=np.uint8)

    leaf_margin = size // 5
    leaf_slice = slice(leaf_margin, size - leaf_margin)
    image[leaf_slice, leaf_slice] = leaf_color

    if lesion_color is not None and lesion_fraction > 0:
        leaf_side = size - 2 * leaf_margin
        lesion_side = int(leaf_side * (lesion_fraction ** 0.5))
        lesion_start = leaf_margin + (leaf_side - lesion_side) // 2
        lesion_slice = slice(lesion_start, lesion_start + lesion_side)
        image[lesion_slice, lesion_slice] = lesion_color

    return image


class TestHsvMaskPrimitives:
    """Tests for hsv_mask.py's low-level mask functions."""

    def test_leaf_mask_isolates_leaf_from_background(self):
        image = _make_synthetic_leaf()
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        leaf_mask = compute_leaf_mask(hsv)

        # The background is large low-saturation gray; the leaf square is a
        # smaller, saturated green region. Leaf pixels should be a clear
        # minority of the image and roughly match the leaf square's area.
        leaf_margin = 100 // 5
        expected_leaf_pixels = (100 - 2 * leaf_margin) ** 2

        assert leaf_mask.sum() > 0
        assert abs(int(leaf_mask.sum()) - expected_leaf_pixels) < expected_leaf_pixels * 0.25

    def test_green_mask_flags_green_pixels(self):
        image = _make_synthetic_leaf()
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        green_mask = compute_green_mask(hsv)

        assert green_mask.sum() > 0

    def test_lesion_mask_empty_for_uniformly_green_leaf(self):
        image = _make_synthetic_leaf(lesion_color=None, lesion_fraction=0.0)
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        leaf_mask = compute_leaf_mask(hsv)
        lesion_mask = compute_lesion_mask(hsv, leaf_mask)

        # A uniformly healthy green leaf should have ~zero lesion pixels
        # (allowing a small tolerance for morphological edge effects).
        assert lesion_mask.sum() < leaf_mask.sum() * 0.05

    def test_lesion_mask_detects_brown_patch(self):
        image = _make_synthetic_leaf(
            lesion_color=(101, 67, 33), lesion_fraction=0.3  # brown
        )
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        leaf_mask = compute_leaf_mask(hsv)
        lesion_mask = compute_lesion_mask(hsv, leaf_mask)

        assert lesion_mask.sum() > 0


class TestSegmentLeafAndLesion:
    """Tests for leaf_segmentation.py's orchestration."""

    def test_returns_consistent_pixel_counts(self):
        image = _make_synthetic_leaf(lesion_color=(101, 67, 33), lesion_fraction=0.3)
        result = segment_leaf_and_lesion(image)

        assert result.leaf_pixel_count == int(result.leaf_mask.sum())
        assert result.lesion_pixel_count == int(result.lesion_mask.sum())
        assert result.leaf_pixel_count >= result.lesion_pixel_count


class TestComputeSeverity:
    """Tests for severity_generator.py's severity computation."""

    def test_healthy_flag_forces_zero_regardless_of_pixels(self):
        # Even an image that would otherwise score high on lesion coverage
        # must return exactly 0.0 when is_healthy=True.
        image = _make_synthetic_leaf(lesion_color=(101, 67, 33), lesion_fraction=0.9)
        severity = compute_severity(image, is_healthy=True)

        assert severity == 0.0

    def test_uniformly_green_leaf_gives_low_severity(self):
        image = _make_synthetic_leaf(lesion_color=None, lesion_fraction=0.0)
        severity = compute_severity(image, is_healthy=False)

        assert 0.0 <= severity < 5.0

    def test_severity_increases_with_lesion_fraction(self):
        small_lesion = _make_synthetic_leaf(
            lesion_color=(101, 67, 33), lesion_fraction=0.1
        )
        large_lesion = _make_synthetic_leaf(
            lesion_color=(101, 67, 33), lesion_fraction=0.6
        )

        severity_small = compute_severity(small_lesion, is_healthy=False)
        severity_large = compute_severity(large_lesion, is_healthy=False)

        assert severity_large > severity_small

    def test_severity_bounded_in_valid_range(self):
        image = _make_synthetic_leaf(lesion_color=(20, 20, 20), lesion_fraction=0.95)
        severity = compute_severity(image, is_healthy=False)

        assert 0.0 <= severity <= 100.0

    def test_severity_is_continuous_not_categorical(self):
        # Regression guard for the DATASET_USAGE.md requirement that
        # severity values are continuous floats, never bucketed strings.
        image = _make_synthetic_leaf(lesion_color=(101, 67, 33), lesion_fraction=0.4)
        severity = compute_severity(image, is_healthy=False)

        assert isinstance(severity, float)


class TestIsHealthyClass:
    """Tests for pseudo_labels.py's healthy-class detection."""

    @pytest.mark.parametrize(
        "class_name,expected",
        [
            ("Tomato_healthy", True),
            ("Tomato_Healthy", True),
            ("healthy", True),
            ("Tomato_Early_blight", False),
            ("Tomato_Late_blight", False),
        ],
    )
    def test_is_healthy_class(self, class_name, expected):
        assert is_healthy_class(class_name) is expected


class TestGenerateSeverityLabels:
    """End-to-end test of the batch CSV generation over a synthetic dataset."""

    @pytest.fixture
    def synthetic_dataset_root(self):
        tmp_dir = tempfile.mkdtemp(prefix="mulxai_severity_test_")
        dataset_root = Path(tmp_dir) / "PlantVillage"

        healthy_dir = dataset_root / "Tomato_healthy"
        blight_dir = dataset_root / "Tomato_Early_blight"
        healthy_dir.mkdir(parents=True)
        blight_dir.mkdir(parents=True)

        for i in range(3):
            healthy_image = _make_synthetic_leaf(lesion_color=None, lesion_fraction=0.0)
            Image.fromarray(healthy_image).save(healthy_dir / f"healthy_{i}.jpg")

            diseased_image = _make_synthetic_leaf(
                lesion_color=(101, 67, 33), lesion_fraction=0.3
            )
            Image.fromarray(diseased_image).save(blight_dir / f"blight_{i}.jpg")

        yield str(dataset_root)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_generates_csv_with_expected_columns(self, synthetic_dataset_root, tmp_path):
        output_csv = tmp_path / "severity_labels.csv"
        generate_severity_labels(synthetic_dataset_root, output_csv=str(output_csv))

        assert output_csv.exists()

        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["filename", "disease", "severity"]
            rows = list(reader)

        assert len(rows) == 6  # 3 healthy + 3 diseased

    def test_healthy_rows_have_zero_severity(self, synthetic_dataset_root, tmp_path):
        output_csv = tmp_path / "severity_labels.csv"
        generate_severity_labels(synthetic_dataset_root, output_csv=str(output_csv))

        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        healthy_rows = [r for r in rows if r["disease"] == "Tomato_healthy"]
        assert len(healthy_rows) == 3
        assert all(float(r["severity"]) == 0.0 for r in healthy_rows)

    def test_severity_values_stay_within_bounds(self, synthetic_dataset_root, tmp_path):
        output_csv = tmp_path / "severity_labels.csv"
        generate_severity_labels(synthetic_dataset_root, output_csv=str(output_csv))

        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            severity = float(row["severity"])
            assert 0.0 <= severity <= 100.0

    def test_no_categorical_labels_ever_written(self, synthetic_dataset_root, tmp_path):
        output_csv = tmp_path / "severity_labels.csv"
        generate_severity_labels(synthetic_dataset_root, output_csv=str(output_csv))

        with open(output_csv, newline="") as f:
            content = f.read()

        for forbidden in ("Mild", "Moderate", "Severe"):
            assert forbidden not in content


class TestLoadRgbImage:
    """Tests for severity_generator.py's load_rgb_image()."""

    def test_loads_image_as_rgb_uint8_array(self, tmp_path):
        image = _make_synthetic_leaf()
        image_path = tmp_path / "sample.jpg"
        Image.fromarray(image).save(image_path)

        loaded = load_rgb_image(str(image_path))

        assert loaded.dtype == np.uint8
        assert loaded.shape[-1] == 3


if __name__ == "__main__":
    pytest.main([__file__])
