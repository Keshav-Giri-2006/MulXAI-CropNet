"""
Dataset loader for PlantVillage Tomato disease classification.

This module provides a PyTorch Dataset class and utility functions to load
and preprocess the PlantVillage Tomato dataset with Albumentations augmentation.

Classes:
    PlantVillageDataset: Custom PyTorch Dataset for image classification
    CVSplit: Structured result for CV pool / holdout partitioning

Functions:
    get_class_names: Automatically discovers disease class names from dataset directory
    get_dataloaders: Creates and returns DataLoader instances for train/val/test splits
    get_cv_pool_and_holdout: Returns the 85% CV pool and the permanently held-out
        15% test partition (ADR-007), using the same split implementation as
        train_val_test_split()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.data.augmentation import AugmentationFactory

class PlantVillageDataset(Dataset):
    """
    Custom PyTorch Dataset for PlantVillage Tomato disease classification.

    Loads images from organized folder structure where each subfolder represents
    a disease class. Applies different augmentations based on the split (train/val/test).

    Attributes:
        image_paths (List[str]): List of absolute paths to image files
        labels (List[int]): List of class indices corresponding to images
        class_names (List[str]): Ordered list of disease class names
        transform: Albumentations composition of augmentation/preprocessing steps
        split (str): Dataset split type ('train', 'val', or 'test')
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: A.Compose,
        split: str = "train",
    ) -> None:
        """
        Initialize PlantVillageDataset.

        Args:
            image_paths: List of absolute paths to image files
            labels: List of class indices corresponding to images
            class_names: Ordered list of disease class names
            transform: Albumentations Compose object for preprocessing/augmentation
            split: Dataset split type ('train', 'val', or 'test'). Default: 'train'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.split = split

        assert len(image_paths) == len(labels), "Number of images and labels must match"
        assert split in ["train", "val", "test"], f"Split must be 'train', 'val', or 'test', got {split}"

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess a single sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple containing:
                - image (torch.Tensor): Preprocessed image tensor (3, 224, 224)
                - label (int): Class index of the disease
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        augmented = self.transform(image=image)
        image_tensor = augmented["image"]
        label = self.labels[idx]

        return image_tensor, label


def get_class_names(dataset_root: str) -> List[str]:
    """
    Automatically discover disease class names from dataset directory structure.

    Args:
        dataset_root: Path to the root dataset directory containing class folders

    Returns:
        Sorted list of class names (disease types)

    Raises:
        FileNotFoundError: If dataset_root does not exist
        ValueError: If no class folders are found in dataset_root
    """
    dataset_path = Path(dataset_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")

    class_folders = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_root}")

    return class_folders


def load_dataset_paths(
    dataset_root: str, class_names: List[str]
) -> Tuple[List[str], List[int]]:
    """Load all image paths and corresponding labels from the dataset directory."""
    image_paths = []
    labels = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for class_idx, class_name in enumerate(class_names):
        class_path = Path(dataset_root) / class_name

        if not class_path.exists():
            raise FileNotFoundError(f"Class folder not found: {class_path}")

        for image_file in class_path.iterdir():
            if image_file.suffix.lower() in supported_extensions:
                image_paths.append(str(image_file.absolute()))
                labels.append(class_idx)

    if not image_paths:
        raise ValueError(f"No images found in {dataset_root}")

    return image_paths, labels


def create_train_transform() -> A.Compose:
    """Create training augmentation pipeline.

    Delegates to AugmentationFactory.get_moderate_augmentation(), the single
    source of truth for the approved training augmentation policy
    (src/data/augmentation.py).
    """
    return AugmentationFactory.get_moderate_augmentation()


def create_val_transform() -> A.Compose:
    """Create validation/test preprocessing pipeline without augmentation.

    Delegates to AugmentationFactory.get_light_augmentation(), the single
    source of truth for the approved validation/test preprocessing pipeline
    (src/data/augmentation.py).
    """
    return AugmentationFactory.get_light_augmentation()


def _stratified_holdout_split(
    image_paths: List[str],
    labels: List[int],
    test_ratio: float,
    random_seed: int,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Single implementation of the stratified test-holdout split.

    This is the sole implementation of the "first split" stage used to carve
    out a stratified holdout partition from the full dataset. It is called by
    both train_val_test_split() (which further subdivides the remaining pool
    into train/val) and get_cv_pool_and_holdout() (which returns the remaining
    pool untouched for Cross Validation, per ADR-007). Keeping this logic in
    one place guarantees both callers produce an identical holdout partition
    given the same test_ratio and random_seed.

    Args:
        image_paths: Full list of image paths to split
        labels: Full list of class indices corresponding to image_paths
        test_ratio: Proportion of data to hold out
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (pool_paths, pool_labels, holdout_paths, holdout_labels)
    """
    from sklearn.model_selection import train_test_split

    pool_paths, holdout_paths, pool_labels, holdout_labels = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    return pool_paths, pool_labels, holdout_paths, holdout_labels


def train_val_test_split(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Stratified split dataset into train, validation, and test subsets.
    Maintains class distribution - compatible with 10-fold stratified cross-validation.

    Internally delegates its first split (separating the test holdout from the
    remaining pool) to _stratified_holdout_split(), the single implementation
    of that split logic shared with get_cv_pool_and_holdout(). Public signature
    and return values are unchanged from prior versions of this function for
    identical inputs.
    """
    from sklearn.model_selection import train_test_split

    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 < total_ratio < 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # First split: separate test set with stratification (shared implementation)
    train_val_paths, train_val_labels, test_paths, test_labels = _stratified_holdout_split(
        image_paths, labels,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # Second split: separate validation from training with stratification
    val_size = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_val_labels
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create and return PyTorch DataLoaders for train, validation, and test sets.

    Args:
        dataset_root: Path to dataset root directory
        batch_size: Number of samples per batch. Default: 32
        num_workers: Number of workers for data loading. Default: 4
        train_ratio: Proportion of data for training. Default: 0.7
        val_ratio: Proportion of data for validation. Default: 0.15
        test_ratio: Proportion of data for testing. Default: 0.15
        random_seed: Random seed for reproducibility. Default: 42

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing corresponding DataLoaders
    """
    class_names = get_class_names(dataset_root)
    print(f"Found {len(class_names)} classes: {class_names}")

    image_paths, labels = load_dataset_paths(dataset_root, class_names)
    print(f"Loaded {len(image_paths)} images")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        train_val_test_split(
            image_paths, labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
    )

    print(f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_transform = create_train_transform()
    val_transform = create_val_transform()

    train_dataset = PlantVillageDataset(train_paths, train_labels, class_names, train_transform, split="train")
    val_dataset = PlantVillageDataset(val_paths, val_labels, class_names, val_transform, split="val")
    test_dataset = PlantVillageDataset(test_paths, test_labels, class_names, val_transform, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader, "class_names": class_names}


@dataclass
class CVSplit:
    """
    Structured result for Cross Validation pool / holdout partitioning.

    Returned by get_cv_pool_and_holdout(). Carries the 85% Cross Validation
    pool (to be further divided into stratified folds by
    src/training/cross_validation.py) and the permanently held-out 15% test
    partition (per ADR-007), which must never be passed into fold generation
    and is not consumed by the Cross Validation pipeline itself.

    Attributes:
        cv_pool_paths: Image paths comprising the 85% CV pool
        cv_pool_labels: Class indices corresponding to cv_pool_paths
        test_paths: Image paths comprising the permanently held-out 15% test set
        test_labels: Class indices corresponding to test_paths
        class_names: Ordered list of disease class names
    """
    cv_pool_paths: List[str] = field(default_factory=list)
    cv_pool_labels: List[int] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    test_labels: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)


def get_cv_pool_and_holdout(
    dataset_root: str,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> CVSplit:
    """
    Return the 85% Cross Validation pool and the permanently held-out 15% test
    partition, per ADR-007.

    The (test_paths, test_labels) partition is guaranteed identical to the
    test partition produced by train_val_test_split() given the same
    test_ratio and random_seed, since both call _stratified_holdout_split(),
    the single shared implementation of the stratified holdout split.

    The returned cv_pool_paths/cv_pool_labels represent the full remaining 85%
    and are not yet divided into train/validation folds; fold generation is
    the responsibility of src/training/cross_validation.py.

    Args:
        dataset_root: Path to dataset root directory
        test_ratio: Proportion of data to permanently hold out as the test
            partition. Default: 0.15
        random_seed: Random seed for reproducibility. Default: 42

    Returns:
        CVSplit containing the CV pool, the held-out test partition, and the
        discovered class names.
    """
    class_names = get_class_names(dataset_root)
    image_paths, labels = load_dataset_paths(dataset_root, class_names)

    cv_pool_paths, cv_pool_labels, test_paths, test_labels = _stratified_holdout_split(
        image_paths, labels,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    return CVSplit(
        cv_pool_paths=cv_pool_paths,
        cv_pool_labels=cv_pool_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
    )