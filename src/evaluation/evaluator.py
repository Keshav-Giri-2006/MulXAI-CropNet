"""
Model evaluator for disease classification.

Handles evaluation on validation and test sets with comprehensive metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.evaluation.metrics import Metrics


class ModelEvaluator:
    """Evaluates model performance on datasets."""

    def __init__(self, model: nn.Module, device: str = 'cpu') -> None:
        """
        Initialize ModelEvaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on. Default: 'cpu'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            criterion: Loss function. Default: None

        Returns:
            Dictionary containing evaluation metrics
        """
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Compute loss if criterion provided
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Compute metrics
        metrics = Metrics.compute_metrics(all_targets, all_preds)

        if criterion is not None:
            metrics['loss'] = total_loss / len(dataloader)

        return metrics

    def evaluate_per_class(
        self,
        dataloader: DataLoader,
        class_names: list,
    ) -> Dict:
        """
        Evaluate model performance per disease class.

        Args:
            dataloader: DataLoader for evaluation
            class_names: List of class names

        Returns:
            Dictionary with per-class metrics
        """
        per_class_metrics = {}
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Compute metrics per class
        for class_idx, class_name in enumerate(class_names):
            class_mask = all_targets == class_idx

            if class_mask.sum() == 0:
                continue

            class_preds = all_preds[class_mask]
            class_targets = all_targets[class_mask]

            per_class_metrics[class_name] = {
                'accuracy': accuracy_score(class_targets, class_preds),
                'precision': precision_score(all_targets, all_preds, labels=[class_idx], average=None, zero_division=0)[0],
                'recall': recall_score(all_targets, all_preds, labels=[class_idx], average=None, zero_division=0)[0],
                'f1': f1_score(all_targets, all_preds, labels=[class_idx], average=None, zero_division=0)[0],
                'samples': class_mask.sum(),
            }

        return per_class_metrics

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions on dataset.

        Args:
            dataloader: DataLoader for prediction

        Returns:
            Tuple of (predictions, targets)
        """
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())

        return np.array(all_preds), np.array(all_targets)
