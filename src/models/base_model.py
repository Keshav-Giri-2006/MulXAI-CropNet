"""
Base model class for tomato disease classification.

Provides abstract base class for implementing disease classification models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseClassifier(ABC, nn.Module):
    """Abstract base class for disease classifiers."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """
        Initialize BaseClassifier.

        Args:
            num_classes: Number of disease classes
            pretrained: Whether to use pretrained weights. Default: True
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Output logits of shape (B, num_classes)
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_except_head(self) -> None:
        """Freeze all parameters except classification head."""
        for name, param in self.named_parameters():
            if 'head' not in name and 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

    def summary(self) -> None:
        """Print model summary."""
        print(f"Model: {self.__class__.__name__}")
        print(f"Total parameters: {self.get_num_parameters():,}")
        print(f"Trainable parameters: {self.get_trainable_parameters():,}")
        print(f"Number of classes: {self.num_classes}")
