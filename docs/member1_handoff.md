# Member 1 Engineering Handoff
**Project:** MulXAI-CropNet
**Status:** Engineering Complete
**Date:** July 2026

---

# Purpose

This document summarizes all deliverables produced by Member 1 (Disease Classification Backbone) and serves as the official engineering handoff before Member 2 implementation begins.

---

# Final Backbone Selection

Selected Backbone:

**EfficientNet-B0**

Reason for selection:

- Highest mean cross-validation accuracy
- Lowest variance across folds
- Highest held-out test accuracy
- Best overall generalization

---

# Candidate Backbone Results

| Backbone | Validation Accuracy | CV Accuracy | CV Std Dev |
|-----------|--------------------:|------------:|-----------:|
| EfficientNet-B0 | 99.63% | **99.43%** | **0.30%** |
| EfficientNetB0_SE | 99.63% | 99.09% | 0.35% |
| MobileNetV3 | 99.42% | 96.88% | 2.30% |

Final selection:

**EfficientNet-B0**

---

# Held-Out Test Results

Checkpoint evaluated:

outputs/experiments/efficientnetb0/best_model.pth

Results:

- Accuracy: **99.75%**
- Precision: **99.75%**
- Recall: **99.75%**
- F1 Score: **99.69%**

Evaluation protocol:

- Independent held-out test set
- Never used during training
- Never used during cross-validation
- ADR-007 compliant

---

# Dataset

Dataset:

PlantVillage Tomato

Classes:

10

Images:

16011

Split:

- Train: 70%
- Validation: 15%
- Test: 15%

Cross-validation performed only on the Train + Validation partition.

---

# Training Configuration

Model:

EfficientNet-B0

Input Size:

224 × 224

Loss:

CrossEntropyLoss

Optimizer:

AdamW

Learning Rate:

0.001

Scheduler:

ReduceLROnPlateau

Epochs:

30

Batch Size:

32

Random Seed:

42

---

# Runtime Note

Cross-validation required a Windows-specific runtime adjustment.

DataLoader:

num_workers:

4 → 0

Reason:

Windows shared-memory mapping exhaustion
(Error 1455)

This modification affects only runtime stability and does not alter the experimental methodology or results.

---

# Canonical Checkpoint

Primary checkpoint:

outputs/experiments/efficientnetb0/best_model.pth

Retained checkpoint:

outputs/experiments/efficientnetb0/final_model.pth

The canonical checkpoint for all downstream work is:

best_model.pth

---

# Experiment Artifacts

Archived under:

outputs/experiments/

Includes:

- Training logs
- Learning curves
- Cross-validation summaries
- Held-out evaluation
- Classification metrics
- Per-class metrics
- Confusion matrix
- Archived checkpoints

---

# Frozen Interfaces

The following interfaces are considered frozen for downstream development:

- EfficientNet-B0 architecture
- 10-class output ordering
- Input resolution (224×224)
- Preprocessing pipeline
- Checkpoint format
- Dataset interface
- Inference interface

Member 2 should not modify these interfaces.

---

# Member 2 Inputs

Backbone:

EfficientNet-B0

Checkpoint:

outputs/experiments/efficientnetb0/best_model.pth

Input:

224×224 RGB

Output:

10-class logits

Transforms:

Reuse the existing validation/inference preprocessing.

---

# Remaining Member 1 Work

The following tasks are publication-oriented and do not block engineering:

- Runtime benchmarking
- Grad-CAM / SG-GradCAM visualizations
- Publication figures
- Comparison tables
- Manuscript preparation

---

# Engineering Status

Implementation:

✅ Complete

Training:

✅ Complete

Cross-validation:

✅ Complete

Backbone selection:

✅ Complete

Held-out evaluation:

✅ Complete

Repository handoff:

✅ Complete

Member 2 may begin implementation.