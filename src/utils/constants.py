"""Constants for MulXAI-CropNet project."""

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Image settings
DEFAULT_IMAGE_SIZE = 224
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Model settings
DEFAULT_MODEL = 'efficientnetb0'
SUPPORTED_MODELS = {
    'efficientnetb0',           # Approved: lightweight
    'mobilenetv3',              # Approved: efficient
    'efficientnetb0_se'         # Approved: with squeeze-excitation
}

# Training settings
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001

# Data split
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Augmentation settings
AUGMENTATION_TYPES = {'light', 'moderate', 'strong'}
DEFAULT_AUGMENTATION = 'moderate'

# Disease classes (PlantVillage Tomato dataset)
TOMATO_DISEASES = [
    'Bacterial spot',
    'Early blight',
    'Late blight',
    'Leaf mold',
    'Septoria leaf spot',
    'Spider mites',
    'Target spot',
    'Yellow leaf curl virus',
    'Mosaic virus',
    'Healthy'
]
