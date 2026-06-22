# README.md

# MulXAI-CropNet

### A Lightweight Explainable Multi-task Deep Learning Framework for Tomato Disease Classification, Continuous Severity Regression, Severity-aware CPS Recommendation and Edge Deployment

---

## Project Overview

MulXAI-CropNet is a research-oriented lightweight Cyber Physical System (CPS) framework designed for intelligent tomato disease diagnosis and management.

The proposed framework combines:

* Tomato Disease Classification
* Continuous Severity Regression (0–100%)
* Severity-aware Explainable AI (SG-GradCAM++)
* Severity-aware Treatment Recommendation
* CPS Decision Layer
* Raspberry Pi Edge Deployment

within a unified multi-task architecture.

---

## Research Motivation

Most existing agricultural AI systems focus on:

* Disease Classification only
* Categorical Severity Levels
* Standalone Explainable AI
* Independent Treatment Recommendations

Very few provide:

* Continuous Severity Estimation
* Severity-aware Explainability
* Severity-aware CPS Recommendations
* Edge Deployment

within a single lightweight framework.

MulXAI-CropNet addresses these gaps.

---

## Research Objectives

### Objective 1

Develop a lightweight multi-task framework using:

EfficientNetB0 + SE

for:

* Disease Classification
* Continuous Severity Regression

---

### Objective 2

Generate pseudo severity labels:

0–100%

using:

HSV based lesion estimation.

---

### Objective 3

Develop:

Severity Guided GradCAM++

(SG-GradCAM++)

for severity-aware explainability.

---

### Objective 4

Develop:

Severity-aware Treatment Recommendation

using:

Disease + Severity

↓

Treatment + Urgency

---

### Objective 5

Deploy:

ONNX

INT8 Quantization

Raspberry Pi 5

---

### Objective 6

Perform:

Cross Domain Evaluation

Train:

PlantVillage

Test:

CCMT

---

# Project Structure

```text
MulXAI-CropNet/

datasets/

notebooks/

docs/

experiments/

outputs/

├── checkpoints/

├── metrics/

├── heatmaps/

└── logs/

src/

├── models/

├── preprocessing/

├── training/

├── evaluation/

├── xai/

└── cps/

README.md

requirements.txt
```

---

# Dataset Structure

```text
datasets/

PlantVillage/

TomatoVillage/

├── Variant-B

└── Variant-c-Object_Detection

CCMT/

severity_labels/
```

---

# Team Responsibilities

## Member 1

Disease Classification

Lightweight Backbone

EfficientNetB0 + SE

---

## Member 2

Pseudo Label Generation

Continuous Severity Regression

Kendall Uncertainty Loss

Cross Domain Severity

---

## Member 3

SG-GradCAM++

Treatment Recommendation

CPS Decision Layer

Edge Deployment

---

# Official Workflow

Data Collection

↓

Preprocessing

↓

Severity Label Generation

↓

Dataset Split

↓

EfficientNetB0 + SE

↙             ↘

Classification

Severity

↘             ↙

SG-GradCAM++

↓

Evaluation

↓

Treatment Recommendation

↓

CPS

↓

Edge Deployment

---

# Official Datasets

PlantVillage

Purpose:

Training

---

TomatoVillage Variant B

Purpose:

Pseudo Label Verification

---

TomatoVillage Variant C

Purpose:

XAI Evaluation

---

CCMT

Purpose:

Cross Domain Testing ONLY

---

# Frozen Evaluation Protocol

PlantVillage:

Train

70%

Validation

15%

Test

15%

---

10 Fold Stratified Cross Validation

Mandatory

---

Cross Domain:

Train:

PlantVillage

Test:

CCMT

---

# Expected Results

Classification Accuracy:

98.5–99.3%

---

Severity MAE:

3–6%

---

R²:

0.90–0.95

---

IoU:

0.55–0.75

---

Pointing Game:

70–85%

---

Latency:

80–150 ms

---

# Technologies

Python

PyTorch

EfficientNetB0

Albumentations

GradCAM

OpenCV

scikit-image

ONNX

NumPy

Pandas

---

# Forbidden Technologies

TensorFlow

YOLO

Vision Transformers

MaskRCNN

SAM

Docker

Kubernetes

Cloud Deployment

Large Ensembles

Enterprise MLOps

---

# Important Documents

Project_Specification_v1.1.md

DATASET_USAGE.md

EVALUATION_PROTOCOL.md

COMPARISON_BASELINES.md

INTEGRATION_PROTOCOL.md

FIGURE_SPECIFICATION.md

---

# Project Status

Research Planning

COMPLETE

---

Literature Review

COMPLETE

---

Dataset Collection

COMPLETE

---

Implementation

READY TO START

---

Version

v1.1

Status:

Frozen Research Specification
