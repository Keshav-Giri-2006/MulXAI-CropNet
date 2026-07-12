# MulXAI-CropNet: Tomato Disease Classification

An undergraduate research project for tomato disease classification using deep learning.

## Project Structure

```
MulXAI-CropNet/
├── src/
│   ├── training/           # Training utilities
│   │   ├── dataset_loader.py
│   │   ├── trainer.py
│   │   ├── training_loop.py
│   │   └── callbacks.py
│   ├── models/             # Model architectures
│   │   ├── base_model.py
│   │   ├── resnet_model.py
│   │   ├── efficientnet_model.py
│   │   └── model_utils.py
│   ├── evaluation/         # Evaluation utilities
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── visualizer.py
│   ├── data/               # Data utilities
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   └── utils/              # Utilities
│       ├── config.py
│       ├── logger.py
│       └── constants.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Assumes PlantVillage Tomato dataset is located at: `datasets/PlantVillage/`

## Training

```bash
python scripts/train.py --model efficientnetb0 --epochs 30 --lr 0.001
```

## Evaluation

```bash
python scripts/evaluate.py --model efficientnetb0 --checkpoint checkpoints/best_model.pth
```

## Inference

```bash
python scripts/inference.py --model efficientnetb0 --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

## Features

- ✅ Automatic dataset discovery (no hardcoded class names)
- ✅ Multiple model architectures (ResNet, EfficientNet)
- ✅ Comprehensive evaluation metrics
- ✅ Data augmentation strategies
- ✅ Modular and extensible design
- ✅ Detailed logging and checkpointing

## Approved Models (Research Specification)

- EfficientNet-B0 (default, lightweight)
- MobileNetV3 (efficient)
- EfficientNet-B0 — selected/shared backbone, model identifier `efficientnetb0_se`.
+   Uses the standard EfficientNet-B0 architecture; Squeeze-and-Excitation is
+   native to its MBConv blocks. The `_se` suffix is retained from earlier
+   project planning for backward compatibility and is not a separate architecture.

## Authors

Member 1 - MulXAI-CropNet Research Team
