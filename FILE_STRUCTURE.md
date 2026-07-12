# MulXAI-CropNet File Structure & Overview

## Complete Project Structure (23 Files)

### 1. **Core Training Module** (4 files)
- `src/training/dataset_loader.py` - PyTorch Dataset and DataLoader implementation
- `src/training/trainer.py` - Main training class with validation loop
- `src/training/training_loop.py` - Flexible training utilities and EarlyStopping
- `src/training/callbacks.py` - Training callbacks (checkpointing, logging)

### 2. **Model Architectures** (4 files)
- `src/models/base_model.py` - Abstract base class for all models
- `src/models/resnet_model.py` - ResNet50 and ResNet101 implementations
- `src/models/efficientnet_model.py` - EfficientNet-B0, B4, B7 implementations
- `src/models/model_utils.py` - Model creation, saving, and loading utilities

### 3. **Data Processing** (2 files)
- `src/data/preprocessing.py` - Image preprocessing and normalization
- `src/data/augmentation.py` - Albumentations augmentation pipelines (light, moderate, strong)

### 4. **Evaluation Module** (3 files)
- `src/evaluation/evaluator.py` - Model evaluation on datasets
- `src/evaluation/metrics.py` - Metric computation (accuracy, F1, precision, recall)
- `src/evaluation/visualizer.py` - Visualization (confusion matrix, metrics plots)

### 5. **Utilities** (3 files)
- `src/utils/config.py` - Configuration management (DataConfig, ModelConfig, TrainingConfig)
- `src/utils/logger.py` - Logging setup with file and console handlers
- `src/utils/constants.py` - Project constants (ImageNet stats, default values)

### 6. **Main Scripts** (3 files)
- `scripts/train.py` - Training script with CLI arguments
- `scripts/evaluate.py` - Evaluation script for test sets
- `scripts/inference.py` - Single image prediction script

### 7. **Tests** (2 files)
- `tests/test_dataset.py` - Dataset loader tests
- `tests/test_model.py` - Model architecture tests

### 8. **Configuration Files** (3 files)
- `requirements.txt` - Python dependencies
- `setup.py` - Package setup file
- `README.md` - Project documentation

### 9. **Package Init Files** (8 files)
- `src/__init__.py`
- `src/training/__init__.py`
- `src/models/__init__.py`
- `src/evaluation/__init__.py`
- `src/data/__init__.py`
- `src/utils/__init__.py`
- `scripts/__init__.py`
- `tests/__init__.py`

---

## File Summary by Function

### Data Loading & Processing
- **dataset_loader.py** - Loads PlantVillage dataset, handles train/val/test splits
- **preprocessing.py** - Normalizes images, validates image files
- **augmentation.py** - Multiple augmentation strategies for training

### Model Development
- **base_model.py** - Abstract interface for all models
- **resnet_model.py** - State-of-the-art ResNet models
- **efficientnet_model.py** - Efficient models for edge deployment
- **model_utils.py** - Create, save, load models; count parameters

### Training & Evaluation
- **trainer.py** - Complete training pipeline with checkpointing
- **training_loop.py** - Flexible training steps, early stopping
- **callbacks.py** - Training callbacks and monitoring
- **evaluator.py** - Comprehensive model evaluation
- **metrics.py** - Metric calculation and reporting
- **visualizer.py** - Results visualization and plotting

### Execution
- **train.py** - Full training pipeline executable
- **evaluate.py** - Model evaluation on test set
- **inference.py** - Single image prediction

### Project Management
- **config.py** - Centralized configuration
- **logger.py** - Logging across the project
- **constants.py** - Shared constants and defaults
- **requirements.txt** - Dependencies
- **setup.py** - Package installation

### Testing
- **test_dataset.py** - Dataset functionality tests
- **test_model.py** - Model functionality tests

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --model resnet50 --epochs 100 --lr 0.001
```

### Evaluation
```bash
python scripts/evaluate.py --model resnet50 --checkpoint checkpoints/best_model.pth
```

### Inference
```bash
python scripts/inference.py --model resnet50 --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

### Testing
```bash
pytest tests/
```

---

## Key Features Across Files

✅ **Modularity** - Each file has a single responsibility
✅ **Type Hints** - All functions have type annotations
✅ **Documentation** - Comprehensive docstrings
✅ **Error Handling** - Proper exception raising and handling
✅ **Configuration** - Centralized config management
✅ **Logging** - Detailed logging throughout
✅ **Testing** - Unit tests for critical components
✅ **Reproducibility** - Random seed management
✅ **Flexibility** - Easy to extend and customize

---

**Total Files: 23**
**Total Python Modules: 19**
**Configuration Files: 3**
**Documentation: 1**
