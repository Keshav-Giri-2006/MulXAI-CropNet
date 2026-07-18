"""
Metrics computation for disease classification.

Calculates accuracy, precision, recall, F1, and other evaluation metrics.
Also provides Cross Validation statistical aggregation (Mean, Standard
Deviation, 95% Confidence Interval across folds), per ADR-007 and the
approved Member 1 Phase 2 Cross Validation architecture.
"""

import csv
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from typing import Dict, List, Optional, Tuple


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
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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
                'precision': precision_score(y_true, y_pred, labels=[class_idx], average=None, zero_division=0)[0],
                'recall': recall_score(y_true, y_pred, labels=[class_idx], average=None, zero_division=0)[0],
                'f1': f1_score(y_true, y_pred, labels=[class_idx], average=None, zero_division=0)[0],
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

    @staticmethod
    def compute_confidence_interval(
        values: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute a confidence interval for a list of values (e.g. per-fold
        metric values from Cross Validation) using the t-distribution, which
        is appropriate for the small sample sizes (n=10 folds) produced by
        10-Fold Stratified Cross Validation.

        Args:
            values: List of numeric values (one per fold)
            confidence: Confidence level. Default: 0.95 (95% CI, per
                EVALUATION_PROTOCOL.md)

        Returns:
            Tuple of (ci_low, ci_high)

        Raises:
            ValueError: If values is empty
        """
        if not values:
            raise ValueError("Cannot compute confidence interval on empty values list")

        n = len(values)
        mean = float(np.mean(values))

        if n == 1:
            return mean, mean

        std_err = float(np.std(values, ddof=1)) / math.sqrt(n)

        try:
            from scipy import stats
            t_critical = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
        except ImportError:
            # Fallback: normal approximation z-critical value for 95% CI
            # if scipy is unavailable. Approved dependencies (scikit-learn,
            # numpy) do not guarantee scipy is installed; this fallback
            # avoids introducing a new hard dependency.
            z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            t_critical = z_table.get(confidence, 1.96)

        margin = t_critical * std_err

        return mean - margin, mean + margin

    @staticmethod
    def aggregate_cv_metrics(
        fold_metrics: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate per-fold Cross Validation metrics into summary statistics.

        Computes Mean, Standard Deviation, and 95% Confidence Interval across
        folds for accuracy, precision, recall, and f1, per the mandatory
        reporting requirements in EVALUATION_PROTOCOL.md.

        Args:
            fold_metrics: List of per-fold metric dictionaries, each
                containing at minimum 'accuracy', 'precision', 'recall', 'f1'
                keys (as returned by train_single_fold())

        Returns:
            Dictionary of the form:
                {metric_name: {'mean': float, 'std': float,
                                'ci_low': float, 'ci_high': float}}
            for each of accuracy, precision, recall, f1.

        Raises:
            ValueError: If fold_metrics is empty
        """
        if not fold_metrics:
            raise ValueError("Cannot aggregate an empty list of fold metrics")

        tracked_metrics = ['accuracy', 'precision', 'recall', 'f1']
        aggregated: Dict[str, Dict[str, float]] = {}

        for metric_name in tracked_metrics:
            values = [fold[metric_name] for fold in fold_metrics if metric_name in fold]

            if not values:
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_low, ci_high = Metrics.compute_confidence_interval(values, confidence=0.95)

            aggregated[metric_name] = {
                'mean': mean,
                'std': std,
                'ci_low': ci_low,
                'ci_high': ci_high,
            }

        return aggregated

    @staticmethod
    def format_cv_summary(
        aggregated: Dict[str, Dict[str, float]],
        n_splits: Optional[int] = None,
    ) -> str:
        """
        Format aggregated Cross Validation statistics as a human-readable
        string summary, consistent in style with get_metrics_summary().

        Args:
            aggregated: Dictionary as returned by aggregate_cv_metrics()
            n_splits: Optional number of folds used, for display in the
                summary title (e.g. "10-Fold Stratified Cross Validation
                Summary"). This parameter is optional and additive: existing
                callers that omit it continue to work exactly as before,
                receiving the fold-count-agnostic title "Stratified Cross
                Validation Summary" rather than an incorrect hardcoded fold
                count. Default: None.

        Returns:
            Formatted string
        """
        summary = "\n" + "=" * 60 + "\n"
        if n_splits is not None:
            summary += f"{n_splits}-Fold Stratified Cross Validation Summary\n"
        else:
            summary += "Stratified Cross Validation Summary\n"
        summary += "=" * 60 + "\n"

        for metric_name, stats in aggregated.items():
            summary += f"\n{metric_name.capitalize()}\n"
            summary += f"{'  Mean:':.<30} {stats['mean']:.4f}\n"
            summary += f"{'  Std Dev:':.<30} {stats['std']:.4f}\n"
            summary += f"{'  95% CI:':.<30} [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]\n"

        summary += "\n" + "=" * 60 + "\n"
        return summary

    @staticmethod
    def save_cv_results(
        fold_metrics: List[Dict[str, float]],
        aggregated: Dict[str, Dict[str, float]],
        output_path: str,
    ) -> None:
        """
        Write per-fold Cross Validation metrics and aggregated summary
        statistics to a CSV file.

        The CSV contains one row per fold followed by summary rows for mean,
        std, ci_low, and ci_high across all tracked metrics, so the full
        record (per-fold results plus aggregation) is preserved in a single
        file under outputs/metrics/, per PROJECT_STRUCTURE.md.

        Args:
            fold_metrics: List of per-fold metric dictionaries
            aggregated: Dictionary as returned by aggregate_cv_metrics()
            output_path: Path to write the CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tracked_metrics = ['accuracy', 'precision', 'recall', 'f1']
        fieldnames = ['fold'] + tracked_metrics

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for fold_idx, fold in enumerate(fold_metrics):
                row = {'fold': fold_idx + 1}
                for metric_name in tracked_metrics:
                    if metric_name in fold:
                        row[metric_name] = fold[metric_name]
                writer.writerow(row)

            for stat_key in ('mean', 'std', 'ci_low', 'ci_high'):
                row = {'fold': stat_key}
                for metric_name in tracked_metrics:
                    if metric_name in aggregated:
                        row[metric_name] = aggregated[metric_name][stat_key]
                writer.writerow(row)

        print(f"Cross Validation results saved to {output_path}")

    @staticmethod
    def save_classification_metrics(
        metrics: Dict[str, float],
        output_path: str,
    ) -> None:
        """
        Write overall classification metrics (accuracy, precision, recall,
        f1, and any additional scalar keys such as loss) to a CSV file, per
        the "classification_metrics.csv" deliverable in
        MEMBER1_CLASSIFICATION_GUIDE.md and EVALUATION_PROTOCOL.md.

        The CSV uses a simple metric,value layout (one row per metric) so
        that it remains readable and trivially extensible if additional
        scalar metrics are added later, without changing the file format.

        Args:
            metrics: Dictionary of scalar metrics, as returned by
                Metrics.compute_metrics() (optionally with a 'loss' key
                added by the caller)
            output_path: Path to write the CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for metric_name, value in metrics.items():
                writer.writerow([metric_name, value])

        print(f"Classification metrics saved to {output_path}")

    @staticmethod
    def save_per_class_metrics(
        per_class_metrics: Dict[str, Dict[str, float]],
        output_path: str,
    ) -> None:
        """
        Write per-class metrics to a CSV file, per the
        "per_class_metrics.csv" deliverable in MEMBER1_CLASSIFICATION_GUIDE.md.

        Args:
            per_class_metrics: Dictionary as returned by
                Metrics.compute_per_class_metrics() or
                ModelEvaluator.evaluate_per_class(), i.e.
                {class_name: {'accuracy', 'precision', 'recall', 'f1', 'samples'}}
            output_path: Path to write the CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ['class', 'accuracy', 'precision', 'recall', 'f1', 'samples']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for class_name, class_metrics in per_class_metrics.items():
                row = {'class': class_name}
                row.update(class_metrics)
                writer.writerow(row)

        print(f"Per-class metrics saved to {output_path}")

    @staticmethod
    def save_training_history(
        history: Dict[str, Dict[str, List[float]]],
        output_path: str,
    ) -> None:
        """
        Write per-epoch training and validation history to a CSV file, per
        the "training_log.csv" deliverable in MEMBER1_CLASSIFICATION_GUIDE.md
        (outputs/logs/training_log.csv). This is the persisted record that
        the Training Loss Curve, Validation Loss Curve, and Accuracy Curve
        are generated from, and lets those plots be regenerated later
        without re-running training.

        Args:
            history: Dictionary of the form {'train': {...}, 'val': {...}}
                as returned by Trainer.get_training_history() / Trainer.train(),
                where each sub-dictionary has 'loss', 'accuracy', 'f1' keys,
                each a list of one value per epoch
            output_path: Path to write the CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        train_history = history['train']
        val_history = history['val']
        num_epochs = len(train_history['loss'])

        fieldnames = [
            'epoch',
            'train_loss', 'train_accuracy', 'train_f1',
            'val_loss', 'val_accuracy', 'val_f1',
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(num_epochs):
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_history['loss'][epoch],
                    'train_accuracy': train_history['accuracy'][epoch],
                    'train_f1': train_history['f1'][epoch],
                    'val_loss': val_history['loss'][epoch],
                    'val_accuracy': val_history['accuracy'][epoch],
                    'val_f1': val_history['f1'][epoch],
                })

        print(f"Training history saved to {output_path}")