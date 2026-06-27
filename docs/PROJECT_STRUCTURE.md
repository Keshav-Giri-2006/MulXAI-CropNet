# PROJECT_STRUCTURE.md

# MulXAI-CropNet

## Official Repository Structure

**Version:** 1.1

**Status:** Frozen

**Last Updated:** June 2026

---

# Purpose

This document defines the official repository structure of the MulXAI-CropNet research project.

It serves as the single source of truth for:

* Repository organization
* Folder responsibilities
* Dataset locations
* Member ownership
* Implementation locations
* Integration rules

This document should be used by:

* Project members
* ChatGPT supervisors
* Claude App
* Future contributors

---

# Repository Root

```text
MulXAI-CropNet/

├── datasets/
├── docs/
├── notebooks/
├── experiments/
├── outputs/
├── src/

├── requirements.txt
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
└── train.py
```

---

# Complete Repository Tree

```text
MulXAI-CropNet/

│
├── datasets/
│
│   ├── PlantVillage/
│   │
│   │   ├── Tomato_Bacterial_spot/
│   │   ├── Tomato_Early_blight/
│   │   ├── Tomato_healthy/
│   │   ├── Tomato_Late_blight/
│   │   ├── Tomato_Leaf_Mold/
│   │   ├── Tomato_Septoria_leaf_spot/
│   │   ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   │   ├── Tomato__Target_Spot/
│   │   ├── Tomato__Tomato_mosaic_virus/
│   │   └── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   │
│   ├── TomatoVillage/
│   │
│   │   ├── Variant-a-Multiclass_Classification/
│   │   │
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   │
│   │   ├── Variant-b-MultiLabel_Classification/
│   │   │
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   │
│   │   └── Variant-c-Object_Detection/
│   │       │
│   │       ├── train/
│   │       │   ├── images/
│   │       │   ├── pascal_voc/
│   │       │   └── yolo/
│   │       │
│   │       └── val/
│   │           ├── images/
│   │           ├── pascal_voc/
│   │           └── yolo/
│   │
│   ├── CCMT/
│   │
│   │   └── Tomato/
│   │       ├── healthy/
│   │       ├── leaf blight/
│   │       ├── leaf curl/
│   │       ├── septoria leaf spot/
│   │       └── verticulium wilt/
│   │
│   └── severity_labels/
│       └── severity_labels.csv
│
├── docs/
│
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── Project_Specification_v1.1.md
│   ├── PROJECT_STRUCTURE.md
│   ├── DATASET_USAGE.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── COMPARISON_BASELINES.md
│   ├── INTEGRATION_PROTOCOL.md
│   ├── FIGURE_SPECIFICATION.md
│   ├── IMPLEMENTATION_PROGRESS.md
│   ├── MEMBER1_CLASSIFICATION_GUIDE.md
│   ├── MEMBER2_SEVERITY_GUIDE.md
│   ├── MEMBER3_XAI_CPS_GUIDE.md
│   ├── CPS-Literature-Review.docx
│   ├── CPS Citations.docx
│   ├── CPS Reference Papers Summary.xlsx
│   ├── Research Audit Report.md
│   └── CPS-Proposed-Workflow.png
│
├── notebooks/
│
│   ├── member1/
│   ├── member2/
│   └── member3/
│
├── experiments/
│
│   ├── member1/
│   ├── member2/
│   └── member3/
│
├── outputs/
│
│   ├── checkpoints/
│   ├── metrics/
│   ├── heatmaps/
│   └── logs/
│
├── src/
│
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── xai/
│   └── cps/
│
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
└── train.py
```

---

# Folder Responsibilities

## datasets/

Contains all datasets used throughout the project.

This folder is intentionally excluded from GitHub because of its size.

Only datasets belong here.

No Python source code.

No notebooks.

No generated results.

---

## docs/

Contains all documentation governing the project.

Includes:

* Literature review
* Specifications
* Evaluation protocols
* Dataset policies
* Member guides
* Research planning
* Project governance

No implementation code belongs here.

---

## notebooks/

Contains exploratory Jupyter notebooks.

Purpose:

* Initial experiments
* Visualization
* Debugging
* Small-scale testing

Each member maintains notebooks only inside their own folder.

Production code should eventually move into `src/`.

---

## experiments/

Contains temporary implementation prototypes.

Examples:

* Baseline experiments
* Hyperparameter testing
* Alternative preprocessing methods
* Experimental evaluation

Code that becomes stable should be migrated into `src/`.

---

## outputs/

Stores generated outputs.

This directory is produced during implementation.

### checkpoints/

Model weights.

Examples:

* `.pth`
* `.pt`

Ignored by Git.

---

### metrics/

Stores:

* CSV results
* Evaluation tables
* Confusion matrices
* Performance graphs
* Cross-validation summaries

Tracked in Git.

---

### heatmaps/

Stores:

* GradCAM
* GradCAM++
* SG-GradCAM++

visualizations.

Ignored by Git.

---

### logs/

Training logs.

Execution logs.

TensorBoard logs.

Ignored by Git.

---

## src/

Primary implementation directory.

All stable implementation code belongs here.

---

# Source Directory Responsibilities

## src/models/

Contains model architectures.

Examples:

* EfficientNetB0
* EfficientNetB0 + SE
* MobileNetV3
* Severity Regression Head

---

## src/preprocessing/

Contains preprocessing utilities.

Examples:

* Image loading
* Dataset transforms
* HSV conversion
* Leaf segmentation
* Lesion extraction
* Severity pseudo-label generation

---

## src/training/

Contains:

* Dataset loaders
* Training loops
* Loss functions
* Optimizers
* Learning-rate schedulers
* Cross-validation pipeline

---

## src/evaluation/

Contains:

* Accuracy evaluation
* Severity evaluation
* Cross-domain evaluation
* Statistical testing
* Performance reporting

---

## src/xai/

Contains explainability modules.

Examples:

* GradCAM
* GradCAM++
* SG-GradCAM++
* IoU computation
* Pointing Game
* Heatmap generation

---

## src/cps/

Contains Cyber-Physical System modules.

Examples:

* Treatment recommendation
* Rule engine
* Decision layer
* ONNX export
* Quantization
* Raspberry Pi deployment

---

# Member Ownership

## Member 1

Primary responsibility:

Disease Classification

Owns:

* src/models/
* src/training/
* src/evaluation/

Working directories:

* notebooks/member1/
* experiments/member1/

Produces:

* Classification checkpoints
* Accuracy metrics
* Cross-validation results

---

## Member 2

Primary responsibility:

Severity Estimation

Owns:

* src/preprocessing/
* Severity-related code inside src/models/
* Severity-related training
* Severity evaluation

Working directories:

* notebooks/member2/
* experiments/member2/

Produces:

* severity_labels.csv
* Regression metrics
* Cross-domain severity evaluation

---

## Member 3

Primary responsibility:

Explainable AI

Cyber Physical System

Edge Deployment

Owns:

* src/xai/
* src/cps/

Working directories:

* notebooks/member3/
* experiments/member3/

Produces:

* Heatmaps
* IoU
* Pointing Game
* Recommendation outputs
* ONNX model
* Quantized model

---

# Repository Rules

1.

No implementation code inside `docs/`.

---

2.

No permanent implementation inside `notebooks/`.

---

3.

Experimental code belongs inside `experiments/`.

---

4.

Stable implementation belongs inside `src/`.

---

5.

Members should modify only their assigned implementation folders.

---

6.

Direct commits to `main` are prohibited.

---

7.

Integration follows:

Member 1

↓

Member 2

↓

Member 3

↓

Final Merge

---

8.

All generated outputs must be stored inside the appropriate subdirectory of `outputs/`.

---

# Claude App Context

When providing this project to Claude App:

DO NOT upload datasets.

Instead, provide:

* Project_Specification_v1.1.md
* PROJECT_STRUCTURE.md
* Relevant MEMBER guide
* Relevant protocol documents

Claude should assume that the datasets already exist locally according to this folder structure.

---

# Final Statement

This document defines the official repository blueprint for MulXAI-CropNet.

All implementation, experimentation, integration, and future development shall conform to this structure unless a future project specification formally revises it.

This repository structure is considered frozen as of Version 1.1.
