# MulXAI-CropNet Usage Guide

## Setup & Installation

```bash
# 1. Navigate to project directory
cd documents-ccps

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Dataset Preparation

Organize your PlantVillage Tomato dataset as:
```
datasets/
└── PlantVillage/
    ├── Bacterial spot/
    │   ├── image_001.jpg
    │   └── ...
    ├── Early blight/
    │   ├── image_001.jpg
    │   └── ...
    └── ... (other disease classes)
```

## Training

### Basic Training
```bash
python scripts/train.py --dataset-root datasets/PlantVillage/
```

### Advanced Training with Custom Parameters
```bash
python scripts/train.py \
    --model efficientnetb0 \
    --dataset-root datasets/PlantVillage/ \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.001 \
    --device cuda  # or 'cpu'
```

### Approved Models (Research Specification)
- `efficientnetb0` (default, lightweight)
- `mobilenetv3` (efficient)
- `efficientnetb0_se` (with squeeze-excitation)

## Evaluation

### Evaluate on Test Set
```bash
python scripts/evaluate.py \
    --model efficientnetb0 \
    --checkpoint checkpoints/best_model.pth \
    --dataset-root datasets/PlantVillage/
```

### View Per-Class Metrics
The evaluation script automatically computes and displays:
- Overall accuracy, precision, recall, F1
- Per-class metrics
- Confusion matrix visualization

## Inference

### Single Image Prediction
```bash
python scripts/inference.py \
    --model efficientnetb0 \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/leaf_image.jpg
```

Output:
```
Predicted: Early blight (Confidence: 0.9821)
```

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test
```bash
pytest tests/test_dataset.py -v
```

### Coverage Report
```bash
pytest tests/ --cov=src
```

## Configuration

### Using Config File
```python
from src.utils.config import Config

config = Config()
print(config.data.batch_size)  # 32
print(config.model.model_name)  # resnet50
print(config.training.num_epochs)  # 100
```

### Creating Custom Config
```python
from src.utils.config import Config, DataConfig, ModelConfig, TrainingConfig

config = Config(
    data=DataConfig(batch_size=64, num_workers=8),
    model=ModelConfig(model_name='efficientnetb4'),
    training=TrainingConfig(num_epochs=50, learning_rate=0.0001)
)
```

## Logging

Logs are automatically saved to `logs/` directory with timestamps:
```
logs/
├── train_20240101_120000.log
├── evaluate_20240101_121000.log
└── inference_20240101_122000.log
```

## Model Checkpoints

Checkpoints are saved to `checkpoints/` directory:
```
checkpoints/
├── best_model.pth          # Best model based on validation loss
└── epoch_010_model.pth     # Periodic checkpoints (optional)
```

## Advanced Usage

### Custom Training Loop
```python
from src.training.trainer import Trainer
from src.training.dataset_loader import get_dataloaders
from src.models.model_utils import create_model
import torch.nn as nn
import torch.optim as optim

# Load data
dataloaders = get_dataloaders('datasets/PlantVillage/')

# Create approved model
model = create_model('efficientnetb0', num_classes=10)

# Setup training (AdamW optimizer - approved)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train
trainer = Trainer(model, dataloaders['train'], dataloaders['val'],
                 criterion, optimizer)
history = trainer.train(num_epochs=30)
```

### Custom Data Augmentation
```python
from src.data.augmentation import AugmentationFactory

# Light augmentation (validation)
light = AugmentationFactory.get_light_augmentation()

# Moderate augmentation (training)
moderate = AugmentationFactory.get_moderate_augmentation()

# Strong augmentation (robust training)
strong = AugmentationFactory.get_strong_augmentation()

# Custom augmentation
custom = AugmentationFactory.get_custom_augmentation(
    horizontal_flip=0.7,
    vertical_flip=0.5,
    rotation=45,
    brightness_contrast=0.5
)
```

### Model Evaluation & Visualization
```python
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import Visualizer
import numpy as np

# Create evaluator
evaluator = ModelEvaluator(model, device='cuda')

# Get predictions
preds, targets = evaluator.predict(test_loader)

# Plot confusion matrix
Visualizer.plot_confusion_matrix(targets, preds, class_names)

# Plot metrics over epochs
Visualizer.plot_metrics(train_metrics, val_metrics, metric_name='f1')

# Plot class distribution
Visualizer.plot_class_distribution(targets, class_names)
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 16

# Use CPU
python scripts/train.py --device cpu
```

### Dataset Not Found
```bash
# Check dataset structure
python -c "from src.training.dataset_loader import get_class_names; print(get_class_names('datasets/PlantVillage/'))"
```

### Import Errors
```bash
# Ensure you're in project root directory
cd documents-ccps

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Performance Tips

1. **Use GPU**: Set `--device cuda` for 10-50x speedup
2. **Batch Size**: Larger batches (64, 128) for faster training
3. **Data Workers**: Increase `--num-workers` (e.g., 8)
4. **Model Selection**:
   - ResNet50: Good baseline
   - EfficientNetB4: Best accuracy/speed balance
   - EfficientNetB0: Lightweight, edge deployment

## Project Structure for Customization

```
src/
├── training/        # Modify training pipeline
├── models/          # Add new architectures
├── evaluation/      # Customize metrics/visualization
├── data/            # Add new augmentations
└── utils/           # Configuration and utilities
```

For detailed file documentation, see `FILE_STRUCTURE.md`
