"""Tests for model architectures."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.efficientnet_model import EfficientNetB0, EfficientNetB0SE
from src.models.mobilenetv3_model import MobileNetV3
from src.models.model_utils import create_model


@pytest.fixture
def device():
    """Get device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def test_efficientnetb0_creation(device):
    """Test EfficientNet-B0 model creation."""
    model = EfficientNetB0(num_classes=10)
    assert model.num_classes == 10
    assert model.get_trainable_parameters() > 0


def test_efficientnetb0_forward(device):
    """Test EfficientNet-B0 forward pass."""
    model = EfficientNetB0(num_classes=10).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10)


def test_mobilenetv3_creation(device):
    """Test MobileNetV3 creation."""
    model = MobileNetV3(num_classes=10)
    assert model.num_classes == 10


def test_mobilenetv3_forward(device):
    """Test MobileNetV3 forward pass."""
    model = MobileNetV3(num_classes=10).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10)


def test_efficientnetb0_se_creation(device):
    """Test EfficientNet-B0 (efficientnetb0_se identifier, ADR-001: standard
    architecture with native SE, no external SE module) creation."""
    model = EfficientNetB0SE(num_classes=10)
    assert model.num_classes == 10
    assert model.get_trainable_parameters() > 0


def test_create_model_approved(device):
    """Test model factory with an approved model name."""
    model = create_model('efficientnetb0', num_classes=10, device=device)
    assert model.num_classes == 10


def test_create_model_rejects_unapproved(device):
    """Test model factory rejects unapproved/unknown model names."""
    with pytest.raises(ValueError):
        create_model('resnet50', num_classes=10, device=device)

    with pytest.raises(ValueError):
        create_model('unknown_model', num_classes=10, device=device)


def test_model_freeze_backbone(device):
    """Test backbone freezing."""
    model = MobileNetV3(num_classes=10)
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.freeze_backbone()
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert frozen_params < initial_params


if __name__ == '__main__':
    pytest.main([__file__])