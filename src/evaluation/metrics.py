"""
Metrics computation for disease classification.

Calculates accuracy, precision, recall, F1, and other evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from typing import Dict, Tuple


class Metrics:
    """Utility class for computing evaluation metrics."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing computed metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list,
    ) -> Dict:
        """
        Compute per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names

        Returns:
            Dictionary with per-class metrics
        """
        per_class_metrics = {}

        for class_idx, class_name in enumerate(class_names):
            class_mask = y_true == class_idx

            if class_mask.sum() == 0:
                continue

            class_preds = y_pred[class_mask]
            class_targets = y_true[class_mask]

            per_class_metrics[class_name] = {
                'accuracy': accuracy_score(class_targets, class_preds),
                'precision': precision_score(
                    class_targets, class_preds, average='weighted', zero_division=0
                ),
                'recall': recall_score(
                    class_targets, class_preds, average='weighted', zero_division=0
                ),
                'f1': f1_score(class_targets, class_preds, average='weighted', zero_division=0),
                'samples': class_mask.sum(),
            }

        return per_class_metrics

    @staticmethod
    def get_metrics_summary(metrics: Dict) -> str:
        """
        Format metrics as string summary.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted string
        """
        summary = "\n" + "=" * 50 + "\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                summary += f"{key:.<30} {value:.4f}\n"
            else:
                summary += f"{key:.<30} {value}\n"
        summary += "=" * 50 + "\n"
        return summary
