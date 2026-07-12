"""
Visualization utilities for model evaluation and analysis.

Creates plots for confusion matrices, metrics, and prediction results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
from typing import Optional, List


class Visualizer:
    """Visualization utilities for model evaluation."""

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save figure. Default: None
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100)
            print(f"Figure saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_metrics(
        train_metrics: dict,
        val_metrics: dict,
        metric_name: str = 'accuracy',
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training and validation metrics over epochs.

        Args:
            train_metrics: Dictionary of training metrics per epoch
            val_metrics: Dictionary of validation metrics per epoch
            metric_name: Metric to plot. Default: 'accuracy'
            save_path: Path to save figure. Default: None
        """
        epochs = range(1, len(train_metrics) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100)
            print(f"Figure saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_class_distribution(
        labels: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot class distribution in dataset.

        Args:
            labels: Array of labels
            class_names: List of class names
            save_path: Path to save figure. Default: None
        """
        unique, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_names)), counts)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.xlabel('Disease Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Dataset')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100)
            print(f"Figure saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_sample_predictions(
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: List[str],
        num_samples: int = 9,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sample predictions from model.

        Args:
            images: Array of images (N, H, W, C)
            true_labels: True labels
            pred_labels: Predicted labels
            class_names: List of class names
            num_samples: Number of samples to plot. Default: 9
            save_path: Path to save figure. Default: None
        """
        num_samples = min(num_samples, len(images))
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        plt.figure(figsize=(15, 15))

        for idx in range(num_samples):
            plt.subplot(grid_size, grid_size, idx + 1)

            # Handle different image formats
            image = images[idx]
            if image.shape[-1] == 3:
                plt.imshow(image)
            else:
                plt.imshow(image, cmap='gray')

            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            match = "✓" if true_labels[idx] == pred_labels[idx] else "✗"

            plt.title(f"True: {true_class}\nPred: {pred_class} {match}")
            plt.axis('off')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100)
            print(f"Figure saved to {save_path}")

        plt.show()
