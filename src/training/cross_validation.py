"""
Cross Validation orchestration for disease classification.

Implements 10-Fold Stratified Cross Validation scoped to the 85% Cross
Validation pool, per ADR-007. The 15% test partition produced by
get_cv_pool_and_holdout() is never passed into any function in this module.

Reuses existing project components (PlantVillageDataset, create_train_transform,
create_val_transform, create_model, Trainer, ModelEvaluator, Metrics) rather
than duplicating logic, per the approved Member 1 Phase 2 architecture.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.training.dataset_loader import (
    CVSplit,
    PlantVillageDataset,
    create_train_transform,
    create_val_transform,
)
from src.models.model_utils import create_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator


def generate_stratified_folds(
    labels: List[int],
    n_splits: int = 10,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified K-fold train/validation index splits over the given
    labels, per the 10-Fold Stratified Cross Validation requirement in
    EVALUATION_PROTOCOL.md.

    Args:
        labels: Class indices for the Cross Validation pool (must be the
            85% pool produced by get_cv_pool_and_holdout(), never the full
            dataset or the held-out test partition, per ADR-007)
        n_splits: Number of folds. Default: 10
        random_seed: Random seed for reproducibility. Default: 42

    Returns:
        List of (train_idx, val_idx) index arrays, each indexing into the
        CV pool's paths/labels lists.
    """
    labels_array = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels_array)), labels_array):
        folds.append((train_idx, val_idx))

    return folds


def build_fold_dataloaders(
    cv_split: CVSplit,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Build train/validation DataLoaders for a single Cross Validation fold by
    indexing into the CV pool's paths/labels using the provided fold indices.

    Reuses PlantVillageDataset and the existing approved augmentation
    pipelines (create_train_transform for the fold's training split,
    create_val_transform for the fold's validation split) rather than
    introducing new transform logic.

    Args:
        cv_split: CVSplit containing the full 85% CV pool
        train_idx: Indices into cv_split.cv_pool_paths/labels for this fold's
            training subset
        val_idx: Indices into cv_split.cv_pool_paths/labels for this fold's
            validation subset
        batch_size: Number of samples per batch. Default: 32
        num_workers: Number of workers for data loading. Default: 4

    Returns:
        Dictionary with keys 'train' and 'val' containing the corresponding
        DataLoaders for this fold.
    """
    pool_paths = cv_split.cv_pool_paths
    pool_labels = cv_split.cv_pool_labels
    class_names = cv_split.class_names

    fold_train_paths = [pool_paths[i] for i in train_idx]
    fold_train_labels = [pool_labels[i] for i in train_idx]
    fold_val_paths = [pool_paths[i] for i in val_idx]
    fold_val_labels = [pool_labels[i] for i in val_idx]

    train_transform = create_train_transform()
    val_transform = create_val_transform()

    train_dataset = PlantVillageDataset(
        fold_train_paths, fold_train_labels, class_names, train_transform, split="train"
    )
    val_dataset = PlantVillageDataset(
        fold_val_paths, fold_val_labels, class_names, val_transform, split="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


def train_single_fold(
    fold_idx: int,
    fold_loaders: Dict[str, DataLoader],
    model_name: str,
    num_classes: int,
    num_epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Train one fresh model instance for a single Cross Validation fold and
    return its validation metrics.

    Reuses create_model() for model instantiation (never a new model
    definition), the existing Trainer class for the training loop (AdamW
    optimizer, per the frozen training configuration), and ModelEvaluator /
    Metrics.compute_metrics() for evaluation, so no training or evaluation
    logic is duplicated outside these existing components.

    Args:
        fold_idx: Index of this fold (for logging purposes)
        fold_loaders: Dictionary with 'train' and 'val' DataLoaders, as
            returned by build_fold_dataloaders()
        model_name: Name of the model to train (e.g. 'efficientnetb0_se')
        num_classes: Number of disease classes
        num_epochs: Number of training epochs. Default: 30 (frozen spec value)
        lr: Learning rate. Default: 0.001 (frozen spec value)
        device: Device to train on. Default: 'cpu'

    Returns:
        Dictionary containing this fold's validation metrics:
        {'accuracy', 'precision', 'recall', 'f1', 'loss'}
    """
    print(f"\n{'=' * 20} Fold {fold_idx + 1} {'=' * 20}")

    model = create_model(model_name, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        train_loader=fold_loaders['train'],
        val_loader=fold_loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=f'./checkpoints/cv_fold_{fold_idx + 1}',
    )

    # save_best=False: fold checkpoints are not part of the frozen backbone
    # selection process (ADR-007 selects the backbone via CV statistics, not
    # via any individual fold's checkpoint), so we avoid writing unnecessary
    # per-fold checkpoint artifacts to disk during Cross Validation.
    trainer.train(num_epochs=num_epochs, save_best=False)

    evaluator = ModelEvaluator(model, device=device)
    val_metrics = evaluator.evaluate(fold_loaders['val'], criterion=criterion)

    print(
        f"Fold {fold_idx + 1} — "
        f"Accuracy: {val_metrics['accuracy']:.4f}, "
        f"F1: {val_metrics['f1']:.4f}"
    )

    return val_metrics


def run_cross_validation(
    dataset_root: str,
    model_name: str = 'efficientnetb0_se',
    n_splits: int = 10,
    batch_size: int = 32,
    num_epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cpu',
    random_seed: int = 42,
) -> List[Dict[str, float]]:
    """
    Top-level orchestrator for 10-Fold Stratified Cross Validation, scoped to
    the 85% Cross Validation pool per ADR-007.

    Obtains the CV pool (and permanently held-out test partition, which is
    never used here) via get_cv_pool_and_holdout(), generates stratified
    folds over the pool via generate_stratified_folds(), and trains/evaluates
    one fresh model per fold via train_single_fold().

    Args:
        dataset_root: Path to dataset root directory
        model_name: Name of the model to train per fold. Default:
            'efficientnetb0_se' (the frozen/selected backbone, per
            INTEGRATION_PROTOCOL.md)
        n_splits: Number of folds. Default: 10 (frozen spec value)
        batch_size: Number of samples per batch. Default: 32
        num_epochs: Number of training epochs per fold. Default: 30 (frozen
            spec value)
        lr: Learning rate. Default: 0.001 (frozen spec value)
        device: Device to train on. Default: 'cpu'
        random_seed: Random seed for reproducibility. Default: 42

    Returns:
        List of n_splits per-fold metric dictionaries, each containing
        'accuracy', 'precision', 'recall', 'f1', 'loss'.
    """
    from src.training.dataset_loader import get_cv_pool_and_holdout

    cv_split = get_cv_pool_and_holdout(
        dataset_root, test_ratio=0.15, random_seed=random_seed
    )
    num_classes = len(cv_split.class_names)

    print(
        f"Cross Validation pool: {len(cv_split.cv_pool_paths)} images "
        f"(held-out test partition: {len(cv_split.test_paths)} images, "
        f"excluded from all folds per ADR-007)"
    )

    folds = generate_stratified_folds(
        cv_split.cv_pool_labels, n_splits=n_splits, random_seed=random_seed
    )

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_loaders = build_fold_dataloaders(
            cv_split, train_idx, val_idx, batch_size=batch_size,
        )
        metrics = train_single_fold(
            fold_idx=fold_idx,
            fold_loaders=fold_loaders,
            model_name=model_name,
            num_classes=num_classes,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
        )
        fold_metrics.append(metrics)

    return fold_metrics