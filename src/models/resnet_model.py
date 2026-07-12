"""
DEPRECATED: ResNet models are NOT approved per research specification.

This file is preserved for reference only.
ResNet50 and ResNet101 have been removed from the approved model registry.

APPROVED MODELS:
- EfficientNet-B0 (default)
- MobileNetV3
- EfficientNet-B0 + SE

See: src/models/efficientnet_model.py
See: src/models/mobilenetv3_model.py
"""

import warnings

warnings.warn(
    "ResNet models are not approved per MulXAI-CropNet research specification. "
    "Use approved models: efficientnetb0, mobilenetv3, or efficientnetb0_se",
    DeprecationWarning,
    stacklevel=2
)


class ResNetDeprecated:
    """Placeholder for deprecated ResNet models."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ResNet models are not approved. "
            "Use: efficientnetb0, mobilenetv3, or efficientnetb0_se"
        )
