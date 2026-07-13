# DATASET_USAGE.md

# MulXAI-CropNet

## Official Dataset Usage Specification

Version: 1.0

Status: Frozen

Date: June 2026

---

# Purpose of this Document

This document defines:

1.

Which datasets are officially used.

2.

The purpose of each dataset.

3.

Training and testing permissions.

4.

Cross-domain evaluation policy.

5.

Dataset ownership across team members.

6.

Which figures and metrics are generated from each dataset.

---

# Dataset Directory

Official location:

```text
F:\CPS-Research\MulXAI-CropNet\datasets\
```

Structure:

```text
datasets/

├── PlantVillage/

├── TomatoVillage/

│   ├── Variant-B-Multilabel/

│   └── Variant-c-Object_Detection/

├── CCMT/

└── severity_labels/
```

---

# DATASET 1

# PlantVillage Tomato Dataset

Official Role:

PRIMARY DATASET

---

Location

```text
datasets/PlantVillage/
```

---

Classes

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

Purpose

Primary Disease Classification

Pseudo Label Generation

Severity Regression

Shared Backbone Training

SG-GradCAM Visualization

---

Used By

Member 1

Disease Classification

---

Member 2

Severity Labels

Severity Regression

---

Member 3

GradCAM

GradCAM++

SG-GradCAM++

---

Official Split

Training:

70%

Validation:

15%

Testing:

15%

This split is created once, using stratified sampling with a fixed random seed, and is not reshuffled.

---

Cross Validation

10 Fold Stratified

Mandatory

Scope (see ADR-007): performed only within the combined 85% Training + Validation pool. The 15% Testing partition is set aside once, remains permanently held out from every Cross Validation fold, and is used only for the single final evaluation of the selected backbone after model selection.

---

Training Allowed

YES

---

Testing Allowed

YES

---

This is the ONLY dataset allowed for initial model training.

---

# DATASET 2

# TomatoVillage Variant B

Official Role

Auxiliary Dataset

---

Location

```text
datasets/TomatoVillage/Variant-B/
```

---

Contains

Images

CSV Metadata

Multi-label Information

---

Purpose

Pseudo Label Verification

Auxiliary Experiments

Severity Consistency Checks

---

Used By

Member 2 ONLY

---

Training Allowed

NO

---

Benchmarking Allowed

YES

---

Cross Validation Allowed

YES

---

This dataset is NOT part of the primary training pipeline.

---

# DATASET 3

# TomatoVillage Variant C

Official Role

Object Detection Dataset

---

Location

```text
datasets/TomatoVillage/Variant-c-Object_Detection/
```

---

Structure

train/

images/

pascal_voc/

yolo/

---

val/

images/

pascal_voc/

yolo/

---

Contains

Images

Bounding Boxes

YOLO Labels

Pascal VOC Labels

---

Purpose

Quantitative XAI Evaluation

---

Used By

Member 3 ONLY

---

Metrics

IoU

Pointing Game

Localization Accuracy

---

Training Allowed

NO

---

Classification Training

FORBIDDEN

---

Severity Training

FORBIDDEN

---

Used ONLY for:

XAI Evaluation.

---

# DATASET 4

# CCMT Tomato Dataset

Official Role

Cross Domain Dataset

---

Location

```text
datasets/CCMT/Tomato/
```

---

Classes

Healthy

Leaf Blight

Leaf Curl

Septoria Leaf Spot

Verticillium Wilt

---

Purpose

Cross Domain Testing

Generalization Evaluation

Real World Robustness

---

Used By

Member 1

Cross Domain Classification

---

Member 2

Cross Domain Severity

---

Member 3

Cross Domain Heatmaps

---

Training Allowed

ABSOLUTELY NOT

---

Validation Allowed

NO

---

Testing Allowed

YES

---

Official Protocol

Train:

PlantVillage

↓

Test:

CCMT

---

This rule MUST NEVER be violated.

---

# DATASET 5

# Severity Labels

Official Role

Generated Dataset

---

Location

```text
datasets/severity_labels/
```

---

Created By

Member 2

---

Generated Using

HSV Conversion

↓

Leaf Segmentation

↓

Lesion Extraction

↓

Infected Area Calculation

↓

Severity Percentage

---

Format

severity_labels.csv

---

Columns

filename

disease

severity

---

Example

img001.jpg

Early Blight

34.5

---

Range

0

to

100

Continuous

---

Categorical Labels:

FORBIDDEN

---

# Dataset Ownership

Member 1

Read:

PlantVillage

CCMT

---

Write:

NONE

---

Member 2

Read:

PlantVillage

TomatoVillage

CCMT

---

Write:

severity_labels/

ONLY

---

Member 3

Read:

PlantVillage

TomatoVillage Variant C

CCMT

severity_labels

---

Write:

NONE

---

# Official Dataset Usage Matrix

PlantVillage

Classification

YES

---

Severity

YES

---

XAI

YES

---

Cross Domain

NO

---

TomatoVillage B

Classification

NO

---

Severity Validation

YES

---

XAI

NO

---

TomatoVillage C

Classification

NO

---

Severity

NO

---

XAI

YES

---

Bounding Boxes

YES

---

CCMT

Classification Testing

YES

---

Severity Testing

YES

---

XAI Testing

YES

---

Training

NO

---

# Figures Supported

Figure 1

Workflow

No Dataset

---

Figure 2

Architecture

No Dataset

---

Figure 3

HSV Severity Generation

PlantVillage

---

Figure 4

SG-GradCAM++

PlantVillage

TomatoVillage C

---

Figure 5

Treatment Recommendation

PlantVillage

Severity Labels

---

Figure 6

Cross Domain Evaluation

PlantVillage

CCMT

---

Figure 7

Raspberry Pi Deployment

PlantVillage

---

# Important Restrictions

CCMT

MUST NEVER be used for training.

---

TomatoVillage Variant C

MUST NEVER be used for classification training.

---

TomatoVillage Variant B

MUST NEVER replace PlantVillage.

---

Severity Labels

MUST remain continuous:

0–100%

---

Categorical Severity:

Mild

Moderate

Severe

is NOT allowed.

---

# Final Dataset Policy

PlantVillage

↓

Train

Validate

Test

Pseudo Labels

---

CCMT

↓

Cross Domain Testing ONLY

---

TomatoVillage B

↓

Auxiliary Severity Validation

---

TomatoVillage C

↓

XAI Evaluation ONLY

---

This dataset policy is frozen and remains valid for all future MulXAI-CropNet implementations unless explicitly revised in a future specification version.

The Cross Validation scope clarification above reflects ADR-007 and does not alter any other frozen requirement in this document.
