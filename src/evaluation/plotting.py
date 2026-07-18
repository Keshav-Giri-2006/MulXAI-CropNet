"""
Plotting utilities for Member 1 evaluation and reporting.

Generates the graphs required by MEMBER1_CLASSIFICATION_GUIDE.md
("GRAPHS TO GENERATE"): Training Loss Curve, Validation Loss Curve,
Accuracy Curve, Confusion Matrix, and Class Distribution Plot.

This module is deliberately separate from metrics.py (metric computation
and CSV export) so that plotting dependencies (matplotlib, seaborn) stay
isolated to one place, per the requested architecture.

Design notes:
- All functions save to disk and never call plt.show(), unlike
  src/evaluation/visualizer.py (which is interactive/notebook-oriented).
  This module is meant to run unattended inside scripts/train.py and
  scripts/evaluate.py, where a blocking plt.show() would stall or fail
  in a headless environment.
- Figures use a white background, 300 dpi, and a restrained color
  palette, per the "publication-quality" requirement. This matches the
  quality bar set by FIGURE_SPECIFICATION.md without adopting that
  document's SVG export requirement, which governs a different set of
  figures (Figures 1-7, architecture/workflow diagrams) not the Member 1
  evaluation curves defined here.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: safe for headless script runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Restrained, consistent palette (no flashy colors, per FIGURE_SPECIFICATION.md style)
_COLOR_TRAIN = '#2C5F8A'   # muted steel blue
_COLOR_VAL = '#B0413E'     # muted brick red
_COLOR_BAR = '#4B7C9E'     # muted blue for bar/heatmap accents
_SAVE_DPI = 300


def _apply_publication_style() -> None:
    """Apply a consistent, restrained matplotlib style for all figures."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })


def _save_and_close(fig: 'plt.Figure', save_path: str) -> None:
    """Save a figure at publication quality and close it to free memory."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_SAVE_DPI, facecolor='white')
    plt.close(fig)
    print(f"Figure saved to {save_path}")


class Plotter:
    """Reusable, headless-safe plotting utilities for Member 1 evaluation."""

    @staticmethod
    def plot_training_loss_curve(
        train_loss: List[float],
        save_path: str,
    ) -> None:
        """
        Plot the Training Loss Curve (loss vs. epoch, training set only).

        Args:
            train_loss: Per-epoch training loss values
            save_path: Path to save the figure (PNG)
        """
        _apply_publication_style()
        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, color=_COLOR_TRAIN, linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        _save_and_close(fig, save_path)

    @staticmethod
    def plot_validation_loss_curve(
        val_loss: List[float],
        save_path: str,
    ) -> None:
        """
        Plot the Validation Loss Curve (loss vs. epoch, validation set only).

        Args:
            val_loss: Per-epoch validation loss values
            save_path: Path to save the figure (PNG)
        """
        _apply_publication_style()
        epochs = range(1, len(val_loss) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, val_loss, color=_COLOR_VAL, linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Curve')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        _save_and_close(fig, save_path)

    @staticmethod
    def plot_accuracy_curve(
        train_accuracy: List[float],
        val_accuracy: List[float],
        save_path: str,
    ) -> None:
        """
        Plot the Accuracy Curve: training and validation accuracy vs. epoch,
        on a single set of axes for direct comparison.

        Args:
            train_accuracy: Per-epoch training accuracy values
            val_accuracy: Per-epoch validation accuracy values
            save_path: Path to save the figure (PNG)
        """
        _apply_publication_style()
        epochs = range(1, len(train_accuracy) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_accuracy, color=_COLOR_TRAIN, linewidth=1.8,
                marker='o', markersize=3, label='Training')
        ax.plot(epochs, val_accuracy, color=_COLOR_VAL, linewidth=1.8,
                marker='o', markersize=3, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curve')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc='lower right')

        _save_and_close(fig, save_path)

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str,
        normalize: bool = False,
    ) -> None:
        """
        Plot the Confusion Matrix for test-set predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names, in label-index order
            save_path: Path to save the figure (PNG)
            normalize: If True, show row-normalized proportions instead of
                raw counts. Default: False (raw counts, matching the
                mandatory Confusion Matrix metric in EVALUATION_PROTOCOL.md)
        """
        _apply_publication_style()
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'

        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8),
                                         max(6, len(class_names) * 0.7)))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            linewidths=0.5,
            linecolor='white',
            ax=ax,
        )
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        _save_and_close(fig, save_path)

    @staticmethod
    def plot_class_distribution(
        labels: np.ndarray,
        class_names: List[str],
        save_path: str,
    ) -> None:
        """
        Plot the Class Distribution: number of samples per disease class.

        Args:
            labels: Array of integer class labels
            class_names: List of class names, in label-index order
            save_path: Path to save the figure (PNG)
        """
        _apply_publication_style()
        unique, counts = np.unique(labels, return_counts=True)

        # Ensure every class is represented, even if a class has zero samples
        full_counts = np.zeros(len(class_names), dtype=int)
        full_counts[unique] = counts

        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), 6))
        ax.bar(range(len(class_names)), full_counts, color=_COLOR_BAR, edgecolor='#1f3f54', linewidth=0.5)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_xlabel('Disease Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution')

        _save_and_close(fig, save_path)