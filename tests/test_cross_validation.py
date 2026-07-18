"""
Tests for Cross Validation implementation (ADR-007).

Covers:
    - get_cv_pool_and_holdout() correctness and CVSplit integrity
    - Disjointness of the CV pool and the held-out test partition
    - Regression check that train_val_test_split() is unchanged after the
      _stratified_holdout_split() refactor
    - generate_stratified_folds() coverage and stratification properties
    - build_fold_dataloaders() construction, where practical without a real
      dataset on disk
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.training.dataset_loader import (
    CVSplit,
    get_cv_pool_and_holdout,
    train_val_test_split,
    get_class_names,
    load_dataset_paths,
)
from src.training.cross_validation import (
    generate_stratified_folds,
    build_fold_dataloaders,
)


@pytest.fixture(scope="module")
def synthetic_dataset_root():
    """
    Create a small synthetic PlantVillage-style dataset directory with three
    classes, enough images per class for 10-fold stratification to be
    meaningful, and clean it up after the test module completes.
    """
    tmp_dir = tempfile.mkdtemp(prefix="mulxai_cv_test_")
    dataset_root = Path(tmp_dir) / "PlantVillage"

    class_names = ["Tomato_Healthy", "Tomato_Early_Blight", "Tomato_Late_Blight"]
    images_per_class = 40

    for class_name in class_names:
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (32, 32), color=(i % 256, (i * 3) % 256, (i * 7) % 256))
            img.save(class_dir / f"img_{i:03d}.jpg")

    yield str(dataset_root)

    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestStratifiedHoldoutRefactorRegression:
    """
    Regression coverage: train_val_test_split() must produce identical
    output before and after the _stratified_holdout_split() extraction.
    """

    def test_train_val_test_split_deterministic_with_seed(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, labels = load_dataset_paths(synthetic_dataset_root, class_names)

        result_a = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )
        result_b = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )

        # Identical seed and inputs must yield identical splits — this is
        # the behavioral contract the refactor is required to preserve.
        assert result_a == result_b

    def test_train_val_test_split_return_shape_unchanged(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, labels = load_dataset_paths(synthetic_dataset_root, class_names)

        train_split, val_split, test_split = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )

        # Public contract: three 2-tuples of (paths, labels)
        for split in (train_split, val_split, test_split):
            assert isinstance(split, tuple)
            assert len(split) == 2
            paths, split_labels = split
            assert len(paths) == len(split_labels)

    def test_train_val_test_split_ratios_approximately_correct(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, labels = load_dataset_paths(synthetic_dataset_root, class_names)
        total = len(image_paths)

        train_split, val_split, test_split = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )

        train_frac = len(train_split[0]) / total
        val_frac = len(val_split[0]) / total
        test_frac = len(test_split[0]) / total

        assert abs(train_frac - 0.7) < 0.05
        assert abs(val_frac - 0.15) < 0.05
        assert abs(test_frac - 0.15) < 0.05

    def test_train_val_test_split_all_samples_accounted_for(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, labels = load_dataset_paths(synthetic_dataset_root, class_names)

        train_split, val_split, test_split = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )

        all_paths = set(train_split[0]) | set(val_split[0]) | set(test_split[0])
        assert all_paths == set(image_paths)

        total_count = len(train_split[0]) + len(val_split[0]) + len(test_split[0])
        assert total_count == len(image_paths)


class TestGetCvPoolAndHoldout:
    """Tests for get_cv_pool_and_holdout() and CVSplit integrity."""

    def test_returns_cvsplit_instance(self, synthetic_dataset_root):
        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)
        assert isinstance(result, CVSplit)

    def test_cvsplit_fields_populated(self, synthetic_dataset_root):
        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        assert len(result.cv_pool_paths) > 0
        assert len(result.cv_pool_labels) > 0
        assert len(result.test_paths) > 0
        assert len(result.test_labels) > 0
        assert len(result.class_names) > 0

        assert len(result.cv_pool_paths) == len(result.cv_pool_labels)
        assert len(result.test_paths) == len(result.test_labels)

    def test_cv_pool_holdout_disjoint(self, synthetic_dataset_root):
        """
        The CV pool and held-out test partition must never overlap — this is
        the core guarantee required by ADR-007.
        """
        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        pool_set = set(result.cv_pool_paths)
        holdout_set = set(result.test_paths)

        assert pool_set.isdisjoint(holdout_set)

    def test_cv_pool_holdout_union_covers_dataset(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, _ = load_dataset_paths(synthetic_dataset_root, class_names)

        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        union = set(result.cv_pool_paths) | set(result.test_paths)
        assert union == set(image_paths)

    def test_holdout_matches_train_val_test_split_test_partition(self, synthetic_dataset_root):
        """
        Critical ADR-007 guarantee: the test partition returned by
        get_cv_pool_and_holdout() must be identical to the test partition
        that train_val_test_split() would produce given the same test_ratio
        and random_seed, since both delegate to _stratified_holdout_split().
        """
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, labels = load_dataset_paths(synthetic_dataset_root, class_names)

        _, _, (tvt_test_paths, tvt_test_labels) = train_val_test_split(
            image_paths, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42,
        )

        cv_split = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        assert set(cv_split.test_paths) == set(tvt_test_paths)
        assert sorted(cv_split.test_labels) == sorted(tvt_test_labels)

    def test_holdout_ratio_approximately_correct(self, synthetic_dataset_root):
        class_names = get_class_names(synthetic_dataset_root)
        image_paths, _ = load_dataset_paths(synthetic_dataset_root, class_names)
        total = len(image_paths)

        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        holdout_frac = len(result.test_paths) / total
        pool_frac = len(result.cv_pool_paths) / total

        assert abs(holdout_frac - 0.15) < 0.05
        assert abs(pool_frac - 0.85) < 0.05

    def test_deterministic_with_fixed_seed(self, synthetic_dataset_root):
        result_a = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)
        result_b = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        assert set(result_a.test_paths) == set(result_b.test_paths)
        assert set(result_a.cv_pool_paths) == set(result_b.cv_pool_paths)

    def test_class_names_match_dataset(self, synthetic_dataset_root):
        expected_class_names = get_class_names(synthetic_dataset_root)
        result = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)

        assert result.class_names == expected_class_names


class TestGenerateStratifiedFolds:
    """Tests for generate_stratified_folds()."""

    def test_returns_expected_number_of_folds(self):
        labels = [0, 1, 2] * 20  # 60 samples, balanced across 3 classes
        folds = generate_stratified_folds(labels, n_splits=10, random_seed=42)
        assert len(folds) == 10

    def test_returns_expected_number_of_folds_custom_n_splits(self):
        labels = [0, 1, 2] * 20
        folds = generate_stratified_folds(labels, n_splits=5, random_seed=42)
        assert len(folds) == 5

    def test_each_fold_returns_index_arrays(self):
        labels = [0, 1, 2] * 20
        folds = generate_stratified_folds(labels, n_splits=10, random_seed=42)

        for train_idx, val_idx in folds:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)

    def test_fold_train_val_disjoint(self):
        labels = [0, 1, 2] * 20
        folds = generate_stratified_folds(labels, n_splits=10, random_seed=42)

        for train_idx, val_idx in folds:
            assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))

    def test_fold_val_indices_cover_all_samples_exactly_once(self):
        """
        Standard k-fold property: across all folds, each sample index must
        appear in exactly one fold's validation set.
        """
        labels = [0, 1, 2] * 20
        n_samples = len(labels)
        folds = generate_stratified_folds(labels, n_splits=10, random_seed=42)

        val_index_counts = {}
        for _, val_idx in folds:
            for idx in val_idx.tolist():
                val_index_counts[idx] = val_index_counts.get(idx, 0) + 1

        assert set(val_index_counts.keys()) == set(range(n_samples))
        assert all(count == 1 for count in val_index_counts.values())

    def test_stratification_preserves_class_balance(self):
        """
        Each fold's validation split should contain a proportional
        representation of each class, per the stratification requirement.
        """
        labels = [0] * 30 + [1] * 30 + [2] * 30
        folds = generate_stratified_folds(labels, n_splits=10, random_seed=42)
        labels_array = np.array(labels)

        for train_idx, val_idx in folds:
            val_labels = labels_array[val_idx]
            unique_classes = set(val_labels.tolist())
            # With 90 samples / 10 folds = 9 per fold, 3 balanced classes
            # should all be represented in every fold's validation split.
            assert unique_classes == {0, 1, 2}

    def test_deterministic_with_fixed_seed(self):
        labels = [0, 1, 2] * 20
        folds_a = generate_stratified_folds(labels, n_splits=10, random_seed=42)
        folds_b = generate_stratified_folds(labels, n_splits=10, random_seed=42)

        for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b):
            assert np.array_equal(train_a, train_b)
            assert np.array_equal(val_a, val_b)


class TestBuildFoldDataloaders:
    """Tests for build_fold_dataloaders(), using a small synthetic dataset."""

    def test_returns_train_and_val_keys(self, synthetic_dataset_root):
        cv_split = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)
        folds = generate_stratified_folds(cv_split.cv_pool_labels, n_splits=5, random_seed=42)
        train_idx, val_idx = folds[0]

        loaders = build_fold_dataloaders(cv_split, train_idx, val_idx, batch_size=4, num_workers=0)

        assert set(loaders.keys()) == {"train", "val"}

    def test_dataloader_sizes_match_fold_indices(self, synthetic_dataset_root):
        cv_split = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)
        folds = generate_stratified_folds(cv_split.cv_pool_labels, n_splits=5, random_seed=42)
        train_idx, val_idx = folds[0]

        loaders = build_fold_dataloaders(cv_split, train_idx, val_idx, batch_size=4, num_workers=0)

        assert len(loaders["train"].dataset) == len(train_idx)
        assert len(loaders["val"].dataset) == len(val_idx)

    def test_dataloader_batches_are_correctly_shaped(self, synthetic_dataset_root):
        cv_split = get_cv_pool_and_holdout(synthetic_dataset_root, test_ratio=0.15, random_seed=42)
        folds = generate_stratified_folds(cv_split.cv_pool_labels, n_splits=5, random_seed=42)
        train_idx, val_idx = folds[0]

        loaders = build_fold_dataloaders(cv_split, train_idx, val_idx, batch_size=4, num_workers=0)

        images, labels = next(iter(loaders["val"]))
        assert images.ndim == 4  # (batch, channels, height, width)
        assert images.shape[0] == labels.shape[0]