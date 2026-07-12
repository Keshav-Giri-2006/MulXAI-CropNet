"""Evaluation script for MulXAI-CropNet."""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_loader import get_dataloaders
from src.models.model_utils import create_model
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import Visualizer
from src.utils.logger import setup_logger


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate disease classifier')
    parser.add_argument('--model', type=str, default='efficientnetb0', help='Model name (efficientnetb0, mobilenetv3, efficientnetb0_se)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--dataset-root', type=str, default='datasets/PlantVillage/',
                       help='Dataset root path')
    
    args = parser.parse_args()
    logger = setup_logger('evaluate')
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    dataloaders = get_dataloaders(args.dataset_root)
    class_names = dataloaders['class_names']
    
    model = create_model(args.model, num_classes=len(class_names), device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = ModelEvaluator(model, device)
    test_metrics = evaluator.evaluate(dataloaders['test'])
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
    per_class_metrics = evaluator.evaluate_per_class(dataloaders['test'], class_names)
    for class_name, metrics in per_class_metrics.items():
        logger.info(f"{class_name}: {metrics}")


if __name__ == '__main__':
    main()
