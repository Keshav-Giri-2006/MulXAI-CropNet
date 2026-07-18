"""Cross Validation script for MulXAI-CropNet.

Runs 10-Fold Stratified Cross Validation scoped to the 85% train+validation
pool, per ADR-007. This script is independent of scripts/train.py,
scripts/evaluate.py, and scripts/inference.py, and does not modify or replace
their behavior. The permanently held-out 15% test partition is never used by
this script; the single final evaluation on that partition remains the
responsibility of the existing train/evaluate pipeline.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.cross_validation import run_cross_validation
from src.evaluation.metrics import Metrics
from src.utils.logger import setup_logger


def main():
    """Main Cross Validation function."""
    parser = argparse.ArgumentParser(description='Run 10-Fold Stratified Cross Validation')
    parser.add_argument('--model', type=str, default='efficientnetb0_se',
                       help='Model name (efficientnetb0, mobilenetv3, efficientnetb0_se)')
    parser.add_argument('--dataset-root', type=str, default='datasets/PlantVillage/',
                       help='Dataset root path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs per fold. Defaults to the '
                            'frozen spec value (30); override for faster development '
                            'iteration. Final reported experiments should use the default.')
    parser.add_argument('--n-splits', type=int, default=10,
                       help='Number of Cross Validation folds. Defaults to the '
                            'frozen spec value (10, per EVALUATION_PROTOCOL.md).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='outputs/metrics/cv_results.csv',
                       help='Path to save Cross Validation results CSV')

    args = parser.parse_args()
    logger = setup_logger('cross_validate')

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'

    logger.info(
        f"Starting {args.n_splits}-Fold Stratified Cross Validation "
        f"for model '{args.model}' ({args.epochs} epochs/fold)"
    )

    fold_metrics = run_cross_validation(
        dataset_root=args.dataset_root,
        model_name=args.model,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        random_seed=args.random_seed,
    )

    aggregated = Metrics.aggregate_cv_metrics(fold_metrics)
    summary = Metrics.format_cv_summary(aggregated, n_splits=args.n_splits)

    print(summary)
    logger.info(summary)

    Metrics.save_cv_results(fold_metrics, aggregated, args.output)


if __name__ == '__main__':
    main()