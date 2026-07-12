# PyTorch Checkpoint (.pth) Files Guide

## What is a .pth File?

A `.pth` file is a **PyTorch model checkpoint** that contains:
- Model architecture weights
- Optimizer state
- Training epoch number
- Metrics/performance data

## .pth File Structure

```python
{
    'epoch': 50,                              # Current epoch
    'model_state_dict': {...},                # Model weights
    'optimizer_state_dict': {...},            # Optimizer state (for resuming)
    'metrics': {                              # Training metrics
        'accuracy': 0.9521,
        'loss': 0.1234,
        'f1': 0.9487,
    }
}
```

## Where .pth Files Are Stored

```
project_root/
└── checkpoints/
    ├── best_model.pth           # Best model (lowest val loss)
    ├── epoch_010_model.pth      # Checkpoint at epoch 10
    ├── epoch_020_model.pth      # Checkpoint at epoch 20
    └── ...
```

## Creating .pth Files

### Method 1: Automatic (During Training)
```bash
python scripts/train.py --model efficientnetb0 --epochs 30
# Automatically saves to checkpoints/best_model.pth
```

### Method 2: Create Dummy Checkpoint (For Testing)
```bash
python scripts/create_dummy_checkpoint.py \
    --model efficientnetb0 \
    --num-classes 10 \
    --output checkpoints/test_model.pth
```

### Method 3: Manual Checkpoint Creation
```python
from src.models.model_utils import create_model, save_checkpoint
import torch.optim as optim
import torch.nn as nn

# Use approved model
model = create_model('efficientnetb0', num_classes=10)
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Approved optimizer
criterion = nn.CrossEntropyLoss()

# After training...
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=30,
    filepath='checkpoints/my_model.pth',
    metrics={'accuracy': 0.95, 'loss': 0.05}
)
```

## Loading .pth Files

### Load for Inference
```python
from src.models.model_utils import create_model, load_checkpoint

model = create_model('efficientnetb0', num_classes=10, device='cuda')
metadata = load_checkpoint(model, filepath='checkpoints/best_model.pth', device='cuda')

print(f"Epoch: {metadata['epoch']}")
print(f"Metrics: {metadata['metrics']}")
```

### Load for Resuming Training
```python
import torch.optim as optim

model = create_model('efficientnetb0', num_classes=10, device='cuda')
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Approved optimizer

metadata = load_checkpoint(
    model=model,
    optimizer=optimizer,
    filepath='checkpoints/best_model.pth',
    device='cuda'
)

# Resume training from loaded epoch
start_epoch = metadata['epoch'] + 1
for epoch in range(start_epoch, num_epochs):
    # Continue training...
    pass
```

## .pth File Sizes

Typical sizes for approved models:

| Model | Parameters | .pth Size |
|-------|-----------|-----------|
| EfficientNet-B0 | 5.3M | ~22 MB |
| EfficientNet-B0 + SE | 5.3M | ~22 MB |
| MobileNetV3 | 5.4M | ~22 MB |

## Checkpoint Management

### Save Best Model Only
```python
trainer = Trainer(model, train_loader, val_loader, 
                  criterion, optimizer, save_best=True)
```

### Save Periodic Checkpoints
```python
# Save every N epochs
if (epoch + 1) % 10 == 0:
    save_checkpoint(
        model, optimizer, epoch,
        f'checkpoints/epoch_{epoch:03d}_model.pth'
    )
```

### Model Comparison

```python
# Load multiple approved models and compare
models = {}
for model_name in ['efficientnetb0', 'mobilenetv3', 'efficientnetb0_se']:
    model = create_model(model_name, num_classes=10, device='cuda')
    load_checkpoint(model, f'checkpoints/{model_name}.pth', device='cuda')
    models[model_name] = model
```

## Using Checkpoints in Scripts

### Training Script
```bash
# Automatically saves best_model.pth
python scripts/train.py --model efficientnetb0 --epochs 30
```

### Evaluation Script
```bash
# Use saved checkpoint for evaluation
python scripts/evaluate.py \
    --model efficientnetb0 \
    --checkpoint checkpoints/best_model.pth
```

### Inference Script
```bash
# Load checkpoint and predict on image
python scripts/inference.py \
    --model efficientnetb0 \
    --checkpoint checkpoints/best_model.pth \
    --image test_image.jpg
```

## Creating Custom Checkpoint Utilities

### Save Custom Metadata
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'augmentation': 'moderate',
    },
    'dataset_info': {
        'num_classes': 10,
        'train_samples': 5000,
        'val_samples': 1000,
    }
}

torch.save(checkpoint, 'checkpoints/detailed_model.pth')
```

### Load and Inspect Checkpoint
```python
checkpoint = torch.load('checkpoints/best_model.pth')

print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Metrics: {checkpoint['metrics']}")
```

## Best Practices

1. **Always save checkpoints during training** to avoid losing progress
2. **Save best model** based on validation metric
3. **Include metadata** for reproducibility
4. **Version your checkpoints** with meaningful names
5. **Store in organized directory** (e.g., `checkpoints/`)
6. **Document model details** in checkpoint
7. **Clean up old checkpoints** to save disk space
8. **Back up important checkpoints** to remote storage

## Troubleshooting

### Checkpoint Mismatch Error
```
RuntimeError: Error(s) in loading state_dict for ...
```
**Solution**: Ensure model architecture matches checkpoint
```python
# Correct
model = create_model('resnet50', ...)  # Load ResNet50 checkpoint
load_checkpoint(model, 'resnet50_checkpoint.pth')

# Wrong
model = create_model('efficientnetb0', ...)  # Mismatch!
load_checkpoint(model, 'resnet50_checkpoint.pth')  # Error
```

### Checkpoint Too Large
```python
# Keep only best model, delete intermediate checkpoints
import os
for file in os.listdir('checkpoints/'):
    if file.startswith('epoch_'):
        os.remove(os.path.join('checkpoints/', file))
```

### CUDA/Device Issues
```python
# Always specify device when loading
checkpoint = torch.load(path, map_location='cpu')  # Load to CPU first
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')  # Move to GPU
```

## Example Workflow

```bash
# 1. Train and automatically save best checkpoint
python scripts/train.py --model resnet50 --epochs 100

# 2. Create test checkpoint (for rapid testing)
python scripts/create_dummy_checkpoint.py --model resnet50

# 3. Evaluate with best checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# 4. Inference on new images
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image image.jpg
```

---

**Note**: .pth files are generated automatically during training. You only need to create them manually for testing or if you need to manually save a model.
