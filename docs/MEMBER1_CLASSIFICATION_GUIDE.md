# MEMBER 1 GUIDE

# MulXAI-CropNet

## Disease Classification & Lightweight Backbone Module

---

# ROLE

You are responsible for:

**Disease Classification using Lightweight CNNs**

Your work forms the backbone of the entire research.

Your output will later be used by:

* Member 2 → Severity Regression
* Member 3 → XAI and CPS

If your classifier is poor, the rest of the project suffers.

Therefore:

**Your first responsibility is building a stable, reproducible, lightweight classifier.**

---

# RESEARCH CONTEXT

We are NOT building:

* a generic CNN classifier
* a Kaggle competition model
* a large transformer
* a huge ensemble

We ARE building:

A lightweight edge-deployable multi-task architecture for:

1. Tomato Disease Classification
2. Severity Estimation
3. Explainable AI
4. CPS Treatment Recommendation
5. Raspberry Pi Deployment

Your responsibility:

ONLY:

Disease Classification.

---

# RESEARCH OBJECTIVE

Build:

A lightweight CNN backbone capable of:

* high classification accuracy
* low latency
* low model size
* edge deployment compatibility

while serving as:

the shared feature extractor for the final multi-task architecture.

---

# DATASETS

## PRIMARY DATASET

datasets/

PlantVillage/

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

## CROSS DOMAIN DATASET

datasets/

CCMT/

Tomato/

healthy

leaf blight

leaf curl

septoria leaf spot

verticulium wilt

Purpose:

Generalization testing.

DO NOT train on this initially.

---

# ALGORITHMS TO IMPLEMENT

You are NOT free to experiment randomly.

You MUST remain consistent with our literature review.

Implement EXACTLY:

---

## Baseline 1

EfficientNetB0

Purpose:

Strong benchmark.

---

## Baseline 2

MobileNetV3 Small

Purpose:

Lightweight comparison.

---

## Proposed Backbone

EfficientNetB0

(Standard architecture — Squeeze-and-Excitation is native to its MBConv

blocks; no additional external SE module is implemented.)

Code identifier: `efficientnetb0_se` (retained for backward compatibility

with existing configuration and checkpoints).

This becomes:

MulXAI Shared Backbone.

This is the backbone used by:

Member 2

and

Member 3.

Therefore:

DO NOT CHANGE IT.

---

# INPUT PREPROCESSING

Image size:

224 × 224

Normalization:

ImageNet mean/std

---

Augmentations:

Horizontal Flip

Vertical Flip

Rotation

Brightness Contrast

Random Zoom

---

Library:

Albumentations ONLY.

---

# TRAINING SETTINGS

Optimizer:

AdamW

Learning Rate:

0.001

Batch Size:

32

Epochs:

30 initially

---

Loss:

CrossEntropyLoss

---

Scheduler:

ReduceLROnPlateau

---

# DATA SPLIT

PlantVillage:

70%

Training

15%

Validation

15%

Testing

---

Later:

Train:

PlantVillage

Test:

CCMT

Cross Domain Evaluation

---

# FOLDER ACCESS RULES

You MAY EDIT:

src/models/

src/training/

src/evaluation/

outputs/checkpoints/

outputs/metrics/

outputs/logs/

notebooks/

experiments/

---

You MAY READ:

src/preprocessing/

---

You MUST NOT TOUCH:

src/xai/

src/cps/

outputs/heatmaps/

Member2 files

Member3 files

---

# FILES TO CREATE

src/models/

efficientnet_baseline.py

mobilenet_baseline.py

mulxai_backbone.py

---

src/training/

train_classifier.py

dataset_loader.py

trainer.py

---

src/evaluation/

evaluate_classifier.py

metrics.py

---

# OUTPUTS TO SAVE

outputs/checkpoints/

best_model.pth

last_model.pth

---

outputs/metrics/

classification_metrics.csv

---

outputs/logs/

training_log.csv

---

# METRICS TO REPORT

You MUST produce:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

---

Expected:

Accuracy:

98–99%

F1:

> 98%

---

# GRAPHS TO GENERATE

Save:

Training Loss Curve

Validation Loss Curve

Accuracy Curve

Confusion Matrix

Class Distribution Plot

---

Save inside:

outputs/metrics/

---

# CROSS VALIDATION

VERY IMPORTANT.

You MUST perform:

10 Fold Stratified Cross Validation

Report:

Mean Accuracy

Mean F1

Standard Deviation

95% Confidence Interval

---

# COMPARISON TABLE

You will compare against:

EfficientNetB0

MobileNetV3

ShuffleNetV2

RTR Lite MobileNetV2

LLRL

XSE TomatoNet

---

Table columns:

Accuracy

Precision

Recall

F1

Parameters

Latency

Model Size

---

# IMPORTANT RESEARCH TARGETS

Your model should:

Remain below:

25 MB

---

Inference:

Less than:

100 ms/image

---

Parameters:

Preferably:

<8 million

---

# DO NOT DO

Do NOT:

Use TensorFlow

Use Vision Transformers

Use YOLO

Use ResNet101

Use DenseNet201

Create ensembles

Use object detection

Use segmentation

Touch severity estimation

Touch XAI

Touch CPS

Touch Raspberry Pi deployment

---

# CLAUDE APP WORKFLOW

Whenever asking Claude:

Always say:

This is undergraduate research.

Prioritize:

simple

modular

readable

research oriented

reproducible

---

Never ask Claude:

"Create complete production architecture"

Never ask:

"Build enterprise AI pipeline"

Never ask:

"Create microservices"

---

Always ask:

"Create minimal research implementation"

---

# FINAL GOAL

At the end of your work:

You should produce:

1.

A trained EfficientNetB0 baseline

2.

A trained MobileNetV3 baseline

3.

The final EfficientNetB0 backbone (`efficientnetb0_se`)

4.

10 Fold Cross Validation Results

5.

Training Graphs

6.

Confusion Matrix

7.

Classification Metrics Table

8.

A stable backbone

ready for:

Member 2 Severity Regression

and

Member 3 XAI + CPS integration.

---

# SUCCESS CONDITION

Your work is COMPLETE only when:

PlantVillage Accuracy:

≥ 98%

Cross-domain accuracy on CCMT:

stable

Model Size:

small

Inference:

fast

Code:

modular

Results:

reproducible

and

the backbone is frozen for integration into MulXAI-CropNet.
