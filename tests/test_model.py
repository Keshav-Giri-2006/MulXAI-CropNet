"""Tests for model architectures."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.resnet_model import ResNet50, ResNet101
from src.models.efficientnet_model import EfficientNetB0
from src.models.model_utils import create_model


@pytest.fixture
def device():
    """Get device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_resnet50_creation(device):
    """Test ResNet50 model creation."""
    model = ResNet50(num_classes=10)
    assert model.num_classes == 10
    assert model.get_trainable_parameters() > 0


def test_resnet50_forward(device):
    """Test ResNet50 forward pass."""
    model = ResNet50(num_classes=10).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 10)


def test_efficientnet_creation(device):
    """Test EfficientNet-B0 creation."""
    model = EfficientNetB0(num_classes=10)
    assert model.num_classes == 10


def test_create_model(device):
    """Test model factory."""
    model = create_model('resnet50', num_classes=10, device=device)
    assert model.num_classes == 10
    
    with pytest.raises(ValueError):
        create_model('unknown_model', num_classes=10, device=device)


def test_model_freeze_backbone(device):
    """Test backbone freezing."""
    model = ResNet50(num_classes=10)
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model.freeze_backbone()
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert frozen_params < initial_params


if __name__ == '__main__':
    pytest.main([__file__])
