"""Inference script for MulXAI-CropNet."""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_loader import get_dataloaders
from src.models.model_utils import create_model
from src.data.augmentation import AugmentationFactory
from src.utils.logger import setup_logger


def predict_single_image(image_path, model, transform, class_names, device):
    """Predict disease class for a single image."""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1)[0].item()
        confidence = probs[0, pred_idx].item()
    
    return class_names[pred_idx], confidence


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Inference with disease classifier')
    parser.add_argument('--model', type=str, default='efficientnetb0', help='Model name (efficientnetb0, mobilenetv3, efficientnetb0_se)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--dataset-root', type=str, default='datasets/PlantVillage/',
                       help='Dataset root path')
    
    args = parser.parse_args()
    logger = setup_logger('inference')
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    dataloaders = get_dataloaders(args.dataset_root)
    class_names = dataloaders['class_names']
    
    model = create_model(args.model, num_classes=len(class_names), device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = AugmentationFactory.get_light_augmentation()
    
    pred_class, confidence = predict_single_image(
        args.image, model, transform, class_names, device
    )
    
    logger.info(f"Predicted: {pred_class} (Confidence: {confidence:.4f})")


if __name__ == '__main__':
    main()
