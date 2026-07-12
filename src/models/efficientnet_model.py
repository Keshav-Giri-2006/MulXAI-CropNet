"""
Approved EfficientNet model implementations.

Only EfficientNet-B0 variants are approved per research specification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.models.base_model import BaseClassifier


class EfficientNetB0Classifier(BaseClassifier):
    """EfficientNet-B0 classifier (approved model)."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Initialize EfficientNet-B0 classifier.

        Args:
            num_classes: Number of disease classes
            pretrained: Use pretrained ImageNet weights. Default: True
            dropout_rate: Dropout rate for regularization. Default: 0.5
        """
        super().__init__(num_classes, pretrained)
        self.dropout_rate = dropout_rate

        # Load pretrained EfficientNet-B0
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Get feature dimension
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Custom head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class EfficientNetB0SEClassifier(BaseClassifier):
    """
    EfficientNet-B0 with Squeeze-Excitation blocks (approved model).
    
    SE blocks enhance channel-wise feature recalibration.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Initialize EfficientNet-B0 with SE.

        Args:
            num_classes: Number of disease classes
            pretrained: Use pretrained ImageNet weights. Default: True
            dropout_rate: Dropout rate for regularization. Default: 0.5
        """
        super().__init__(num_classes, pretrained)
        self.dropout_rate = dropout_rate

        # Load base EfficientNet-B0
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # SE blocks already built into EfficientNet
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Custom head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits


# Aliases for convenience
EfficientNetB0 = EfficientNetB0Classifier
EfficientNetB0SE = EfficientNetB0SEClassifier
