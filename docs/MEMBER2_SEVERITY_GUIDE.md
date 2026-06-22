# MEMBER 2 GUIDE

# MulXAI-CropNet

## Continuous Severity Regression & Pseudo-Label Generation Module

---

# ROLE

You are responsible for:

1. Severity Pseudo-Label Generation

2. Continuous Severity Regression

3. Cross-Domain Severity Evaluation

---

Your work is NOT a secondary feature.

It is one of the MAIN NOVELTIES of the paper.

Without your work:

MulXAI-CropNet becomes:

only a disease classifier.

With your work:

MulXAI-CropNet becomes:

A Multi-task Disease Classification + Continuous Severity Regression Framework.

---

# RESEARCH GAP YOU ARE ADDRESSING

Existing papers:

Mostly perform:

Disease Classification

OR

Categorical Severity

Example:

Low

Medium

High

---

Very few papers perform:

Continuous Severity Regression

Example:

12%

37%

68%

92%

---

Even fewer perform:

Continuous Severity Estimation

*

Lightweight CNN

*

Cross Domain Testing

*

Edge Deployment

---

THIS is the gap you are solving.

---

# FINAL OBJECTIVE

Generate:

```text
Image

↓

Severity %

↓

Continuous Regression

↓

0 – 100 %
```

instead of:

```text
Mild

Moderate

Severe
```

---

# DATASETS

## MAIN DATASET

datasets/

PlantVillage/

This dataset has:

Disease Labels

BUT

NO severity labels.

---

Therefore:

YOU MUST CREATE THEM.

---

# SECONDARY DATASET

datasets/

TomatoVillage/

Variant B

Contains:

images

csv metadata

---

Purpose:

Cross validation

Pseudo-label verification

---

# CROSS DOMAIN DATASET

datasets/

CCMT/

Purpose:

Real-world evaluation.

DO NOT use for training initially.

---

# MOST IMPORTANT IDEA

PlantVillage DOES NOT provide:

```text
Severity %
```

You must create:

```text
Image

↓

Leaf Segmentation

↓

Lesion Segmentation

↓

Infected Area

↓

Severity Percentage
```

---

# PSEUDO LABEL GENERATION

This is one of the paper novelties.

---

Step 1

Read image

---

Step 2

Convert:

RGB

↓

HSV

---

Step 3

Mask green regions

Leaf extraction

---

Step 4

Detect diseased regions

Brown

Yellow

Dark spots

Necrotic areas

---

Step 5

Calculate:

```text
Severity

=

Infected Pixels

/

Total Leaf Pixels

×

100
```

---

Output:

```text
0–100 %
```

---

Example:

Healthy

↓

0%

---

Small lesions

↓

8%

---

Moderate

↓

42%

---

Severe

↓

85%

---

# FOLDER TO STORE LABELS

Create:

datasets/

severity_labels/

---

Store:

severity_labels.csv

---

Format:

filename

disease

severity

---

Example:

```text
img001.jpg

Early Blight

34.5
```

---

# PREPROCESSING

Image:

224 × 224

---

Normalization:

ImageNet

---

Augmentation:

Horizontal Flip

Vertical Flip

Rotation

Brightness

Contrast

Zoom

---

Library:

Albumentations

ONLY.

---

# MODEL ARCHITECTURE

DO NOT create a separate CNN.

You MUST use:

Member 1 Backbone.

---

Shared Backbone:

EfficientNetB0 + SE

---

Freeze initially:

Train only:

Severity Head.

---

Later:

Fine tune entire network.

---

# REGRESSION HEAD

Input:

Backbone Features

---

Layers:

Linear

↓

ReLU

↓

Dropout

↓

Linear

↓

Severity

---

Output:

```text
0–100
```

Single neuron.

---

# LOSS FUNCTION

VERY IMPORTANT

DO NOT USE:

```text
α CE

+

β MSE
```

This was removed.

---

USE:

Kendall Uncertainty Weighted Loss

Reference:

CVPR 2018

---

Combined:

Classification Loss

*

Severity Loss

Automatically weighted.

---

This was recommended by:

Research Audit June 2026.

---

# TRAINING SETTINGS

Optimizer:

AdamW

---

Learning Rate:

0.001

---

Batch:

32

---

Epochs:

30

---

Scheduler:

ReduceLROnPlateau

---

# FILES TO CREATE

src/preprocessing/

severity_generator.py

leaf_segmentation.py

hsv_mask.py

pseudo_labels.py

---

src/models/

severity_head.py

multitask_model.py

---

src/training/

train_severity.py

losses.py

---

src/evaluation/

evaluate_severity.py

---

# FOLDER PERMISSIONS

You MAY EDIT:

src/preprocessing/

src/models/

src/training/

src/evaluation/

outputs/metrics/

notebooks/

experiments/

---

You MAY READ:

src/models/

Member1 outputs

---

You MUST NOT TOUCH:

src/xai/

src/cps/

outputs/heatmaps/

---

# METRICS TO REPORT

VERY IMPORTANT

Report:

MAE

RMSE

R² Score

---

Expected:

MAE:

3–6 %

---

RMSE:

5–8 %

---

R²:

0.90 – 0.95

---

# CROSS DOMAIN TESTING

Train:

PlantVillage

---

Test:

CCMT

---

Report:

Cross Domain MAE

---

Expected:

Increase:

+2%

to

+4.5%

---

This was explicitly added after:

Research Audit June 2026.

---

# GRAPHS TO PRODUCE

Save:

Severity Distribution

---

Predicted vs Actual

---

MAE Curve

---

RMSE Curve

---

Residual Plot

---

Cross Domain Error Plot

---

Save inside:

outputs/metrics/

---

# OUTPUT FILES

outputs/metrics/

severity_results.csv

---

cross_domain_results.csv

---

outputs/checkpoints/

severity_model.pth

---

multitask_model.pth

---

# COMPARISON PAPERS

Compare against:

MTDLF

MTDL

TSTC

HBDS

LLRL

---

Columns:

Severity Type

Continuous

Categorical

MAE

RMSE

R²

Dataset

Cross Domain

---

# IMPORTANT RESEARCH TARGET

Your model should:

Predict:

```text
37.5 %
```

NOT

```text
Moderate
```

---

Continuous regression

is one of the strongest novelties of MulXAI-CropNet.

---

# DO NOT DO

Do NOT:

Use U-Net

Use YOLO

Use MaskRCNN

Use Vision Transformers

Use object detection

Use segmentation networks

Create a new backbone

Replace EfficientNet

Change Member1 architecture

Touch XAI

Touch CPS

Touch Raspberry Pi

---

# CLAUDE APP RULES

Whenever prompting Claude:

Always say:

This is undergraduate research.

Prioritize:

simple

modular

lightweight

reproducible

---

DO NOT ask:

"Create SOTA architecture"

DO NOT ask:

"Create transformer hybrid"

DO NOT ask:

"Build production AI"

---

Ask:

"Implement a minimal research-grade severity regression pipeline."

---

# FINAL SUCCESS CONDITION

Your work is COMPLETE only when:

Pseudo labels generated

↓

Severity labels saved

↓

Regression model trained

↓

MAE < 6 %

↓

R² > 0.90

↓

Cross Domain MAE reported

↓

Results reproducible

↓

Shared backbone compatible with:

Member 1

and

Member 3

↓

Ready for final MulXAI-CropNet integration.
