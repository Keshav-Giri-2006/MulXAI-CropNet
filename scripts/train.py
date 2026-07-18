"""Main training script for MulXAI-CropNet."""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_loader import get_dataloaders
from src.models.model_utils import create_model
from src.training.trainer import Trainer
from src.evaluation.metrics import Metrics
from src.evaluation.plotting import Plotter
from src.utils.logger import setup_logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train disease classifier')
    parser.add_argument('--model', type=str, default='efficientnetb0', help='Model name (efficientnetb0, mobilenetv3, efficientnetb0_se)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--dataset-root', type=str, default='datasets/PlantVillage/',
                       help='Dataset root path')
    parser.add_argument('--log-dir', type=str, default='outputs/logs/',
                       help='Directory to save training_log.csv')
    parser.add_argument('--metrics-dir', type=str, default='outputs/metrics/',
                       help='Directory to save training loss/accuracy curve plots')

    args = parser.parse_args()
    logger = setup_logger('train')
    device = args.device if torch.cuda.is_available() else 'cpu'
    log_dir = Path(args.log_dir)
    metrics_dir = Path(args.metrics_dir)

    dataloaders = get_dataloaders(args.dataset_root, batch_size=args.batch_size)
    num_classes = len(dataloaders['class_names'])

    model = create_model(args.model, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    trainer = Trainer(model, dataloaders['train'], dataloaders['val'],
                     criterion, optimizer, device=device)
    history = trainer.train(args.epochs, scheduler=scheduler)

    torch.save(model.state_dict(), 'final_model.pth')

    # --- Automatic history export (training_log.csv) ---
    Metrics.save_training_history(history, str(log_dir / 'training_log.csv'))

    # --- Automatic graph generation (Training Loss Curve, Validation Loss Curve, Accuracy Curve) ---
    Plotter.plot_training_loss_curve(
        history['train']['loss'], str(metrics_dir / 'training_loss_curve.png')
    )
    Plotter.plot_validation_loss_curve(
        history['val']['loss'], str(metrics_dir / 'validation_loss_curve.png')
    )
    Plotter.plot_accuracy_curve(
        history['train']['accuracy'], history['val']['accuracy'],
        str(metrics_dir / 'accuracy_curve.png')
    )

    logger.info(f"Training history saved to {log_dir}")
    logger.info(f"Training curves saved to {metrics_dir}")


if __name__ == '__main__':
    main()