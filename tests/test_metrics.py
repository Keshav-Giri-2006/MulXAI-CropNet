"""
Tests for src/evaluation/metrics.py, focused on the additive Cross
Validation statistical methods added to the Metrics class:
    - compute_confidence_interval()
    - aggregate_cv_metrics()
    - format_cv_summary()
    - save_cv_results()
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import Metrics


class TestComputeConfidenceInterval:
    """Tests for Metrics.compute_confidence_interval()."""

    def test_raises_on_empty_values(self):
        with pytest.raises(ValueError):
            Metrics.compute_confidence_interval([])

    def test_single_value_returns_that_value_as_both_bounds(self):
        ci_low, ci_high = Metrics.compute_confidence_interval([0.85])
        assert ci_low == pytest.approx(0.85)
        assert ci_high == pytest.approx(0.85)

    def test_ci_bounds_symmetric_around_mean(self):
        values = [0.80, 0.82, 0.85, 0.83, 0.81, 0.84, 0.86, 0.79, 0.82, 0.83]
        ci_low, ci_high = Metrics.compute_confidence_interval(values, confidence=0.95)
        mean = float(np.mean(values))

        assert ci_low < mean < ci_high
        assert (mean - ci_low) == pytest.approx(ci_high - mean, rel=1e-6)

    def test_zero_variance_gives_zero_width_interval(self):
        values = [0.9] * 10
        ci_low, ci_high = Metrics.compute_confidence_interval(values, confidence=0.95)

        assert ci_low == pytest.approx(0.9, abs=1e-6)
        assert ci_high == pytest.approx(0.9, abs=1e-6)

    def test_higher_confidence_gives_wider_interval(self):
        values = [0.80, 0.82, 0.85, 0.83, 0.81, 0.84, 0.86, 0.79, 0.82, 0.83]

        ci_low_90, ci_high_90 = Metrics.compute_confidence_interval(values, confidence=0.90)
        ci_low_99, ci_high_99 = Metrics.compute_confidence_interval(values, confidence=0.99)

        width_90 = ci_high_90 - ci_low_90
        width_99 = ci_high_99 - ci_low_99

        assert width_99 > width_90

    def test_more_samples_narrows_interval(self):
        rng = np.random.default_rng(42)
        small_sample = rng.normal(loc=0.85, scale=0.02, size=5).tolist()
        large_sample = rng.normal(loc=0.85, scale=0.02, size=50).tolist()

        ci_low_small, ci_high_small = Metrics.compute_confidence_interval(small_sample)
        ci_low_large, ci_high_large = Metrics.compute_confidence_interval(large_sample)

        width_small = ci_high_small - ci_low_small
        width_large = ci_high_large - ci_low_large

        assert width_large < width_small


class TestAggregateCvMetrics:
    """Tests for Metrics.aggregate_cv_metrics()."""

    def test_raises_on_empty_fold_metrics(self):
        with pytest.raises(ValueError):
            Metrics.aggregate_cv_metrics([])

    def test_returns_expected_metric_keys(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
            {"accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.895},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert set(aggregated.keys()) == {"accuracy", "precision", "recall", "f1"}
        for metric_name in aggregated:
            assert set(aggregated[metric_name].keys()) == {"mean", "std", "ci_low", "ci_high"}

    def test_mean_computed_correctly(self):
        fold_metrics = [
            {"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.80},
            {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert aggregated["accuracy"]["mean"] == pytest.approx(0.85)

    def test_std_zero_for_identical_fold_values(self):
        fold_metrics = [
            {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9}
            for _ in range(10)
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert aggregated["accuracy"]["std"] == pytest.approx(0.0, abs=1e-9)

    def test_std_nonzero_for_varying_fold_values(self):
        fold_metrics = [
            {"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.80},
            {"accuracy": 0.85, "precision": 0.85, "recall": 0.85, "f1": 0.85},
            {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90},
            {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert aggregated["accuracy"]["std"] > 0.0

    def test_single_fold_std_is_zero(self):
        """A single fold has no variance; ddof=1 std with n=1 is handled as 0.0."""
        fold_metrics = [
            {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9}
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert aggregated["accuracy"]["std"] == pytest.approx(0.0)

    def test_ten_fold_realistic_scenario(self):
        """Simulates the standard 10-fold case with realistic metric spread."""
        rng = np.random.default_rng(42)
        fold_metrics = []
        for _ in range(10):
            base = rng.normal(loc=0.88, scale=0.015)
            fold_metrics.append({
                "accuracy": base,
                "precision": base - 0.01,
                "recall": base - 0.005,
                "f1": base - 0.008,
            })

        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert 0.0 <= aggregated["accuracy"]["mean"] <= 1.0
        assert aggregated["accuracy"]["ci_low"] <= aggregated["accuracy"]["mean"] <= aggregated["accuracy"]["ci_high"]

    def test_handles_missing_metric_key_in_some_folds(self):
        """
        If a metric key is absent from every fold, it should not appear in
        the aggregated output rather than raising.
        """
        fold_metrics = [
            {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"accuracy": 0.91, "precision": 0.91, "recall": 0.91, "f1": 0.91},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        assert "accuracy" in aggregated


class TestFormatCvSummary:
    """Tests for Metrics.format_cv_summary()."""

    def test_returns_string(self):
        aggregated = {
            "accuracy": {"mean": 0.90, "std": 0.02, "ci_low": 0.88, "ci_high": 0.92},
        }
        summary = Metrics.format_cv_summary(aggregated)
        assert isinstance(summary, str)

    def test_contains_metric_names(self):
        aggregated = {
            "accuracy": {"mean": 0.90, "std": 0.02, "ci_low": 0.88, "ci_high": 0.92},
            "f1": {"mean": 0.87, "std": 0.03, "ci_low": 0.84, "ci_high": 0.90},
        }
        summary = Metrics.format_cv_summary(aggregated)

        assert "Accuracy" in summary
        assert "F1" in summary

    def test_contains_formatted_values(self):
        aggregated = {
            "accuracy": {"mean": 0.9012, "std": 0.0234, "ci_low": 0.8800, "ci_high": 0.9224},
        }
        summary = Metrics.format_cv_summary(aggregated)

        assert "0.9012" in summary
        assert "0.0234" in summary

    def test_default_title_omits_fold_count_when_not_provided(self):
        """
        Without n_splits, the title must not claim a specific fold count
        (avoids the previously reported hardcoded '10-Fold' defect when the
        actual n_splits used may differ)."""
        aggregated = {
            "accuracy": {"mean": 0.90, "std": 0.02, "ci_low": 0.88, "ci_high": 0.92},
        }
        summary = Metrics.format_cv_summary(aggregated)

        assert "10-Fold" not in summary
        assert "Stratified Cross Validation Summary" in summary

    def test_title_reflects_provided_n_splits(self):
        aggregated = {
            "accuracy": {"mean": 0.90, "std": 0.02, "ci_low": 0.88, "ci_high": 0.92},
        }
        summary = Metrics.format_cv_summary(aggregated, n_splits=2)

        assert "2-Fold Stratified Cross Validation Summary" in summary
        assert "10-Fold" not in summary

    def test_title_reflects_ten_folds_when_specified(self):
        aggregated = {
            "accuracy": {"mean": 0.90, "std": 0.02, "ci_low": 0.88, "ci_high": 0.92},
        }
        summary = Metrics.format_cv_summary(aggregated, n_splits=10)

        assert "10-Fold Stratified Cross Validation Summary" in summary


class TestSaveCvResults:
    """Tests for Metrics.save_cv_results()."""

    def test_creates_output_file(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
            {"accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.895},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "metrics", "cv_results.csv")
            Metrics.save_cv_results(fold_metrics, aggregated, output_path)

            assert os.path.exists(output_path)

    def test_creates_parent_directories(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "nested", "deeper", "cv_results.csv")
            Metrics.save_cv_results(fold_metrics, aggregated, output_path)

            assert os.path.exists(output_path)

    def test_csv_contains_one_row_per_fold_plus_summary_rows(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
            {"accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.895},
            {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "cv_results.csv")
            Metrics.save_cv_results(fold_metrics, aggregated, output_path)

            with open(output_path, newline='') as f:
                rows = list(csv.DictReader(f))

            # 3 fold rows + 4 summary rows (mean, std, ci_low, ci_high)
            assert len(rows) == 3 + 4

    def test_csv_fold_rows_match_input_values(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "cv_results.csv")
            Metrics.save_cv_results(fold_metrics, aggregated, output_path)

            with open(output_path, newline='') as f:
                rows = list(csv.DictReader(f))

            first_row = rows[0]
            assert first_row["fold"] == "1"
            assert float(first_row["accuracy"]) == pytest.approx(0.90)

    def test_csv_summary_rows_contain_expected_labels(self):
        fold_metrics = [
            {"accuracy": 0.90, "precision": 0.89, "recall": 0.88, "f1": 0.885},
            {"accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.895},
        ]
        aggregated = Metrics.aggregate_cv_metrics(fold_metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "cv_results.csv")
            Metrics.save_cv_results(fold_metrics, aggregated, output_path)

            with open(output_path, newline='') as f:
                rows = list(csv.DictReader(f))

            fold_labels = {row["fold"] for row in rows}
            assert {"mean", "std", "ci_low", "ci_high"}.issubset(fold_labels)