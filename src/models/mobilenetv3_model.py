"""
Approved MobileNetV3 model implementation.

MobileNetV3 is an efficient architecture approved for this research.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.models.base_model import BaseClassifier


class MobileNetV3Classifier(BaseClassifier):
    """MobileNetV3-Large classifier (approved model)."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Initialize MobileNetV3 classifier.

        Args:
            num_classes: Number of disease classes
            pretrained: Use pretrained ImageNet weights. Default: True
            dropout_rate: Dropout rate for regularization. Default: 0.5
        """
        super().__init__(num_classes, pretrained)
        self.dropout_rate = dropout_rate

        # Load pretrained MobileNetV3
        try:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
        except:
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)

        # Get feature dimension
        feature_dim = self.backbone.classifier[0].in_features
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


# Alias for convenience
MobileNetV3 = MobileNetV3Classifier
