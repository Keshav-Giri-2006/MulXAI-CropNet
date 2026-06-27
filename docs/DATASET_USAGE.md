# DATASET_USAGE.md

# MulXAI-CropNet

## Official Dataset Usage Specification

**Version:** 1.1

**Status:** Frozen

**Last Updated:** June 2026

---

# Purpose

This document defines the official usage policy for every dataset included in the MulXAI-CropNet repository.

It specifies:

* Dataset purpose
* Allowed usage
* Forbidden usage
* Training policy
* Testing policy
* Cross-domain evaluation policy
* Member ownership
* Figures and evaluation metrics supported by each dataset

This document is authoritative for all future implementation.

---

# Official Dataset Directory

```text
datasets/

├── PlantVillage/
├── TomatoVillage/
│   ├── Variant-a-Multiclass_Classification/
│   ├── Variant-b-MultiLabel_Classification/
│   └── Variant-c-Object_Detection/
├── CCMT/
└── severity_labels/
```

---

# Dataset Summary

| Dataset                 | Primary Purpose                     | Official Status                 |
| ----------------------- | ----------------------------------- | ------------------------------- |
| PlantVillage            | Main training dataset               | Primary                         |
| TomatoVillage Variant A | Auxiliary multiclass classification | Optional                        |
| TomatoVillage Variant B | Auxiliary multi-label validation    | Optional                        |
| TomatoVillage Variant C | XAI quantitative evaluation         | Required                        |
| CCMT                    | Cross-domain evaluation             | Required                        |
| severity_labels         | Generated pseudo-labels             | Generated during implementation |

---

# DATASET 1

# PlantVillage Tomato Dataset

## Official Status

Primary Dataset

---

## Location

```text
datasets/PlantVillage/
```

---

## Disease Classes

* Tomato_Bacterial_spot
* Tomato_Early_blight
* Tomato_healthy
* Tomato_Late_blight
* Tomato_Leaf_Mold
* Tomato_Septoria_leaf_spot
* Tomato_Spider_mites_Two_spotted_spider_mite
* Tomato__Target_Spot
* Tomato__Tomato_mosaic_virus
* Tomato__Tomato_YellowLeaf__Curl_Virus

---

## Official Purpose

Primary disease classification.

Primary severity regression.

Pseudo-label generation.

Training of the shared backbone.

Model evaluation.

SG-GradCAM++ visualization.

---

## Used By

### Member 1

Disease classification

Baseline comparison

Backbone training

---

### Member 2

Severity label generation

Severity regression

Cross-validation

---

### Member 3

GradCAM

GradCAM++

SG-GradCAM++

Treatment recommendation demonstrations

---

## Official Dataset Split

Training

70%

Validation

15%

Testing

15%

---

## Cross Validation

10-Fold Stratified Cross Validation

Mandatory.

---

## Allowed Usage

* Training
* Validation
* Testing
* Cross-validation
* Visualization

---

## Forbidden Usage

None.

This is the primary benchmark dataset.

---

# DATASET 2

# TomatoVillage Variant A

## Official Status

Auxiliary Dataset

---

## Location

```text
datasets/TomatoVillage/Variant-a-Multiclass_Classification/
```

---

## Contents

* train
* validation
* test

Classes include:

* Early Blight
* Healthy
* Late Blight
* Leaf Miner
* Magnesium Deficiency
* Nitrogen Deficiency
* Potassium Deficiency
* Spotted Wilt Virus

---

## Purpose

This dataset is **not** part of the primary MulXAI-CropNet benchmark.

It is retained for:

* Supplementary multiclass classification experiments
* External robustness analysis
* Additional benchmarking
* Future project extensions

---

## Used By

Primarily:

Member 1

Only for optional supplementary experiments.

---

## Allowed Usage

* Exploratory experiments
* Robustness analysis
* Future work

---

## Forbidden Usage

Do NOT replace PlantVillage with Variant A for the official experiments.

Do NOT use Variant A as the primary benchmark dataset for publication results.

---

# DATASET 3

# TomatoVillage Variant B

## Official Status

Auxiliary Dataset

---

## Location

```text
datasets/TomatoVillage/Variant-b-MultiLabel_Classification/
```

---

## Purpose

Provides multi-label annotations for supplementary experiments.

Used for:

* Multi-label benchmarking
* Auxiliary validation
* Severity-related exploratory studies
* Future extensions

---

## Used By

Primarily:

Member 2

---

## Allowed Usage

* Supplementary experiments
* Validation
* Exploratory research

---

## Forbidden Usage

Not part of the official MulXAI-CropNet benchmark.

Do not report Variant B as the primary experimental dataset.

---

# DATASET 4

# TomatoVillage Variant C

## Official Status

Required

---

## Location

```text
datasets/TomatoVillage/Variant-c-Object_Detection/
```

---

## Structure

Contains:

* Images
* Pascal VOC annotations
* YOLO annotations

---

## Purpose

Quantitative evaluation of explainable AI.

Used to compare model attention against lesion annotations.

---

## Used By

Member 3

---

## Metrics Supported

* IoU
* Pointing Game
* Localization Accuracy

---

## Allowed Usage

* XAI evaluation
* Heatmap validation
* Lesion localization

---

## Forbidden Usage

Not permitted for:

* Disease classifier training
* Severity regression training
* Official benchmark training

---

# DATASET 5

# CCMT Tomato Dataset

## Official Status

Required

---

## Location

```text
datasets/CCMT/Tomato/
```

---

## Classes

* Healthy
* Leaf Blight
* Leaf Curl
* Septoria Leaf Spot
* Verticillium Wilt

---

## Purpose

Cross-domain evaluation.

Real-world robustness assessment.

Generalization testing.

---

## Used By

Member 1

Cross-domain classification.

---

Member 2

Cross-domain severity estimation.

---

Member 3

Cross-domain SG-GradCAM++ visualization.

---

## Allowed Usage

Testing ONLY.

---

## Forbidden Usage

Training

Validation

Hyperparameter tuning

Model selection

CCMT must never influence training.

---

# DATASET 6

# Severity Labels

## Official Status

Generated Dataset

---

## Location

```text
datasets/severity_labels/
```

---

## Generated By

Member 2

---

## Pipeline

RGB Image

↓

HSV Conversion

↓

Leaf Segmentation

↓

Lesion Extraction

↓

Infected Area Calculation

↓

Continuous Severity Percentage

---

## Output File

```text
severity_labels.csv
```

---

## Required Columns

* filename
* disease_class
* severity_percentage

---

## Severity Range

Continuous values

0–100%

---

Categorical labels such as:

* Mild
* Moderate
* Severe

are NOT permitted.

---

# Official Member Dataset Ownership

## Member 1

Reads:

* PlantVillage
* CCMT
* Variant A (optional)

Writes:

None

---

## Member 2

Reads:

* PlantVillage
* Variant B
* CCMT

Writes:

severity_labels/

---

## Member 3

Reads:

* PlantVillage
* Variant C
* CCMT
* severity_labels/

Writes:

None

---

# Official Dataset Usage Matrix

| Dataset      | Classification | Severity | XAI     | Cross-Domain | Official Paper     |
| ------------ | -------------- | -------- | ------- | ------------ | ------------------ |
| PlantVillage | ✓              | ✓        | ✓       | ✗            | ✓                  |
| Variant A    | Optional       | ✗        | ✗       | ✗            | Supplementary Only |
| Variant B    | Optional       | Optional | ✗       | ✗            | Supplementary Only |
| Variant C    | ✗              | ✗        | ✓       | ✗            | ✓                  |
| CCMT         | Testing        | Testing  | Testing | ✓            | ✓                  |

---

# Figures Supported

## Figure 1

Workflow

No dataset required.

---

## Figure 2

Architecture

No dataset required.

---

## Figure 3

HSV Severity Generation

PlantVillage

---

## Figure 4

GradCAM

GradCAM++

SG-GradCAM++

PlantVillage

Variant C

---

## Figure 5

Treatment Recommendation

PlantVillage

Severity Labels

---

## Figure 6

Cross-Domain Evaluation

PlantVillage

↓

CCMT

---

## Figure 7

Edge Deployment

PlantVillage

---

# Critical Rules

1. PlantVillage is the official training dataset.

2. CCMT is reserved exclusively for cross-domain testing.

3. Variant C is reserved for XAI evaluation.

4. Variant A and Variant B are auxiliary datasets only.

5. Continuous severity regression (0–100%) is mandatory.

6. Categorical severity labels are prohibited.

7. Training on CCMT is strictly forbidden.

8. No dataset substitutions are permitted without a formal revision of the project specification.

---

# Final Statement

This document defines the official dataset governance policy for MulXAI-CropNet.

All experiments, benchmarks, comparisons, figures, and publication results shall comply with the dataset usage rules defined herein unless superseded by a future version of the Project Specification.

Version 1.1 is considered frozen for the implementation phase.
