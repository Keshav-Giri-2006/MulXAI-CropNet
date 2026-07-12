"""
Model utility functions for creation, loading, and saving.

APPROVED MODELS ONLY:
- EfficientNet-B0 (default)
- MobileNetV3
- EfficientNet-B0 + SE
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Type
from src.models.base_model import BaseClassifier
from src.models.efficientnet_model import EfficientNetB0, EfficientNetB0SE
from src.models.mobilenetv3_model import MobileNetV3


# APPROVED MODEL REGISTRY - Research specification compliant
MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    'efficientnetb0': EfficientNetB0,
    'efficientnetb0_se': EfficientNetB0SE,
    'mobilenetv3': MobileNetV3,
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    device: str = 'cpu',
) -> BaseClassifier:
    """
    Create a model by name (approved models only).

    Args:
        model_name: Name of the model (efficientnetb0, mobilenetv3, efficientnetb0_se)
        num_classes: Number of disease classes
        pretrained: Use pretrained weights. Default: True
        device: Device to move model to. Default: 'cpu'

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not in approved registry
    """
    model_name_lower = model_name.lower().replace('_', '')

    normalized_registry = {key.replace('_', ''): key for key in MODEL_REGISTRY}

    if model_name_lower not in normalized_registry:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model not approved: {model_name}. Approved models: {available}")

    registry_key = normalized_registry[model_name_lower]
    model_class = MODEL_REGISTRY[registry_key]
    model = model_class(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    metrics: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        filepath: Path to save checkpoint
        metrics: Dictionary of metrics to save. Default: None
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if metrics:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filepath: str = None,
    device: str = 'cpu',
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into. Default: None
        filepath: Path to checkpoint file
        device: Device to load checkpoint to. Default: 'cpu'

    Returns:
        Dictionary containing checkpoint metadata

    Raises:
        FileNotFoundError: If checkpoint file not found
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    metadata = {
        'epoch': checkpoint.get('epoch'),
        'metrics': checkpoint.get('metrics'),
    }

    print(f"Checkpoint loaded from {filepath}")
    return metadata


def get_model_info(model: nn.Module) -> Dict:
    """
    Get model information.

    Args:
        model: Model to analyze

    Returns:
        Dictionary containing model info
    """
    if isinstance(model, BaseClassifier):
        return {
            'class_name': model.__class__.__name__,
            'num_classes': model.num_classes,
            'total_params': model.get_num_parameters(),
            'trainable_params': model.get_trainable_parameters(),
        }
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'class_name': model.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
