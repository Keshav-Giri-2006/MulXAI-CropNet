# Project Specification v1.1

## Project Title

**MulXAI-CropNet: A Lightweight Explainable Multi-task Deep Learning Framework for Tomato Disease Classification, Continuous Severity Regression, Severity-aware CPS Recommendation and Edge Deployment**

---

# Version Information

Version: 1.1

Status: Frozen

Date: June 2026

Institution:

Center for Cyber Physical Systems (CCPS)

Research Domain:

Agriculture 4.0

Cyber Physical Systems

Explainable Artificial Intelligence

Deep Learning

---

# Team Structure

### Member 1

Disease Classification

Lightweight Backbone

EfficientNetB0 (selected backbone, identifier: `efficientnetb0_se`)

---

### Member 2

Severity Pseudo-label Generation

Continuous Severity Regression

Cross-domain Severity Evaluation

---

### Member 3

SG-GradCAM++

Severity-aware Treatment Recommendation

CPS Decision Layer

Edge Deployment

---

# Problem Statement

Existing tomato disease diagnosis systems primarily focus on disease classification alone.

A few systems incorporate severity estimation.

Some use explainable AI.

Others provide treatment recommendations.

However, no existing lightweight framework simultaneously provides:

* Disease Classification
* Continuous Severity Regression
* Severity-aware Explainable AI
* Severity-aware Treatment Recommendation
* CPS Decision Making
* Raspberry Pi Edge Deployment

within a unified multi-task architecture.

This fragmentation limits practical deployment in real agricultural environments.

Therefore, this project proposes:

MulXAI-CropNet,

a lightweight explainable multi-task CPS framework capable of jointly performing:

Disease Classification,

Continuous Severity Estimation,

Severity-aware Explainability,

Treatment Recommendation,

and Edge Deployment.

---

# Research Gaps (Frozen)

## Gap 1

Lack of lightweight multi-task frameworks combining:

Disease Classification

*

Continuous Severity Regression

*

Edge Deployment

---

## Gap 2

Lack of severity-aware explainability.

Existing methods:

GradCAM

GradCAM++

ignore disease severity.

---

## Gap 3

Lack of severity-aware treatment recommendation.

Most systems:

Disease

↓

Recommendation

without considering:

Severity

Urgency

Intervention Priority

---

## Gap 4

Cross-domain disease severity estimation remains underexplored.

Most studies:

Train:

PlantVillage

Test:

PlantVillage

Only.

---

## Gap 5

No unified framework integrates:

Multi-task Learning

Continuous Severity

Quantitative XAI

CPS Recommendation

and

Edge Deployment

in a single tomato disease framework.

---

# Research Objectives (Frozen)

## Objective 1

Develop a lightweight multi-task CNN architecture capable of:

Tomato Disease Classification

and

Continuous Severity Regression

using a shared EfficientNetB0 backbone (project identifier `efficientnetb0_se`).

This is the standard EfficientNet-B0 architecture; Squeeze-and-Excitation is

native to its MBConv blocks — no additional external SE module is implemented.

---

## Objective 2

Develop a pseudo-label generation pipeline capable of estimating continuous severity scores:

0–100%

using HSV based lesion extraction.

---

## Objective 3

Develop:

Severity Guided GradCAM++

(SG-GradCAM++)

where disease severity influences heatmap intensity and lesion localization.

---

## Objective 4

Develop a severity-aware treatment recommendation module

that maps:

Disease

*

Severity

↓

Treatment

Urgency

Recommendation.

---

## Objective 5

Deploy the model on an edge environment

through:

ONNX Export

INT8 Quantization

Raspberry Pi 5 Compatibility.

---

## Objective 6

Evaluate model robustness through:

Cross-domain validation

between:

PlantVillage

and

CCMT.

---

# Dataset Specification

## Dataset 1

PlantVillage Tomato Subset

Purpose:

Primary Training Dataset

Tasks:

Disease Classification

Severity Pseudo Labels

Model Training

---

Classes:

Tomato_Bacterial_spot

Tomato_Early_blight

Tomato_healthy

Tomato_Late_blight

Tomato_Leaf_Mold

Tomato_Septoria_leaf_spot

Tomato_Spider_mites_Two_spotted_spider_mite

Tomato__Target_Spot

Tomato__Tomato_mosaic_virus

Tomato__Tomato_YellowLeaf__Curl_Virus

---

## Dataset 2

TomatoVillage

Variant B:

Pseudo Label Verification

Auxiliary Experiments

---

Variant C:

Object Detection Dataset

Bounding Boxes

Used for:

XAI Quantitative Evaluation

IoU

Pointing Game

---

## Dataset 3

CCMT Tomato Dataset

Purpose:

Cross-domain Evaluation

---

Classes:

Healthy

Leaf Blight

Leaf Curl

Septoria Leaf Spot

Verticillium Wilt

---

Training on CCMT is NOT allowed.

Testing ONLY.

---

# Proposed Workflow

Data Collection

↓

Preprocessing

↓

Severity Pseudo Label Generation

↓

Dataset Split

↓

EfficientNetB0 Shared Backbone (`efficientnetb0_se`)

↙                    ↘

Disease Classification

Severity Regression

↘                    ↙

SG-GradCAM++

↓

Evaluation

↓

Treatment Recommendation

↓

CPS Decision Layer

↓

Edge Deployment

---

# Frozen Architecture

Shared Backbone:

EfficientNetB0 — standard architecture, native SE

(project identifier: `efficientnetb0_se`, retained for compatibility)

---

Classification Head:

Disease Prediction

---

Severity Head:

Continuous Regression

0–100%

---

XAI:

GradCAM

GradCAM++

SG-GradCAM++

---

Treatment Module:

Rule Based

Severity Aware

---

Deployment:

ONNX

INT8 Quantization

Raspberry Pi 5

---

# Frozen Training Configuration

Optimizer:

AdamW

---

Learning Rate:

0.001

---

Batch Size:

32

---

Epochs:

30

---

Scheduler:

ReduceLROnPlateau

---

Input Size:

224 × 224

---

Augmentations:

Horizontal Flip

Vertical Flip

Rotation

Brightness Contrast

Random Resized Crop

---

Library:

Albumentations ONLY.

---

# Frozen Loss Functions

Classification:

Cross Entropy

---

Severity:

MSE

---

Multi-task:

Kendall Uncertainty Weighted Loss

CVPR 2018

---

Manual:

α CE + β MSE

NOT ALLOWED.

---

# Evaluation Protocol

## PlantVillage

Training:

70%

Validation:

15%

Testing:

15%

---

## Cross Domain

Train:

PlantVillage

Test:

CCMT

---

## Cross Validation

10 Fold Stratified Cross Validation

Mandatory.

Scope (see ADR-007): performed only within the combined 85% Training + Validation pool. The 15% Testing partition remains permanently held out and is never included in any fold. After model selection via Cross Validation, the selected backbone is evaluated exactly once on the untouched 15% Testing partition; this single evaluation is the final reported test result.

---

# Expected Results (Frozen)

## Classification

Accuracy:

98.5 – 99.3 %

---

Macro F1:

0.98 – 0.99

---

## Severity

MAE:

3 – 6 %

---

RMSE:

5 – 8 %

---

R²:

0.90 – 0.95

---

Cross Domain MAE Increase:

+2%

to

+4.5%

---

## XAI

IoU:

0.55 – 0.75

---

Pointing Game:

70 – 85 %

---

## Edge Deployment

Model Size:

2.5 – 5 MB

---

Latency:

80 – 150 ms

---

RAM:

<500 MB

---

FPS:

5 – 10

---

# Publication Figures (Frozen)

Figure 1

Overall Workflow

---

Figure 2

MulXAI-CropNet Architecture

---

Figure 3

HSV Severity Pseudo Label Generation

---

Figure 4

GradCAM vs GradCAM++ vs SG-GradCAM++

---

Figure 5

Severity-aware Treatment Recommendation

---

Figure 6

Cross Domain Evaluation

PlantVillage → CCMT

---

Figure 7

Raspberry Pi Edge Deployment

---

# Git Repository Rules

Repository:

MulXAI-CropNet

---

Branches:

main

member1-classification

member2-severity

member3-xai-cps

---

main:

Protected.

No direct commits.

---

Members MUST commit only to their own branches.

---

# Frozen Technologies

Python

PyTorch

EfficientNetB0

(native SE — no separate technology entry required)

Albumentations

GradCAM

GradCAM++

ONNX

OpenCV

scikit-image

NumPy

Pandas

---

# Strictly Forbidden

TensorFlow

Vision Transformers

YOLO

MaskRCNN

SAM

Docker

Kubernetes

AWS

Cloud Deployment

Enterprise MLOps

Custom Backbones

Large Ensembles

---

# Final Statement

MulXAI-CropNet is a lightweight explainable multi-task CPS framework designed to jointly perform tomato disease classification, continuous severity regression, severity-aware explainability, treatment recommendation and edge deployment within a unified architecture.

The architecture, objectives, datasets, evaluation metrics and workflow defined in this document are frozen and shall remain unchanged throughout implementation unless explicitly approved through a future specification revision.
