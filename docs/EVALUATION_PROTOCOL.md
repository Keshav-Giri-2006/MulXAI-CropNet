# EVALUATION_PROTOCOL.md

# MulXAI-CropNet

## Official Evaluation Protocol

Version: 1.0

Status: Frozen

Date: June 2026

---

# Purpose

This document defines:

* Dataset splits
* Cross validation procedure
* Cross-domain evaluation
* Performance metrics
* Statistical validation
* Reporting rules

These protocols are mandatory for all experiments.

---

# Dataset Split

## PlantVillage

Training Set:

70%

Validation Set:

15%

Testing Set:

15%

---

The split must:

* Preserve class distribution
* Use stratified sampling
* Be reproducible using fixed random seed

---

Random reshuffling after experiments:

NOT ALLOWED.

---

# Cross Validation

Official Protocol:

10 Fold Stratified Cross Validation

Mandatory.

---

Scope (see ADR-007):

Cross Validation is performed only within the combined 85% Training + Validation pool defined above.

The 15% Testing partition is set aside once and remains permanently held out; it is never included in any Cross Validation fold.

---

Procedure:

Split the 85% Training + Validation pool into:

10 folds

For each fold:

Train:

9 folds

Test:

1 fold

Repeat:

10 times

---

Report:

Mean

Standard Deviation

95% Confidence Interval

for all metrics, computed across the 10 folds.

---

Final Reported Test Metrics:

After model selection via Cross Validation, evaluate the selected backbone exactly once on the untouched 15% Testing partition.

This single evaluation is the final reported test result and is not repeated or reshuffled.

---

# Cross Domain Evaluation

Official Protocol:

Train:

PlantVillage

↓

Test:

CCMT

---

Training on CCMT:

FORBIDDEN

---

Validation on CCMT:

FORBIDDEN

---

CCMT is reserved exclusively for:

Generalization Evaluation.

---

# Classification Metrics

Mandatory:

Accuracy

Precision

Recall

Macro F1 Score

Confusion Matrix

---

Expected:

Accuracy:

98.5–99.3 %

---

Macro F1:

0.98–0.99

---

# Severity Metrics

Mandatory:

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

0.90–0.95

---

Cross Domain MAE increase:

+2 %

to

+4.5 %

---

# XAI Metrics

Mandatory:

IoU

Pointing Game

Localization Accuracy

---

Expected:

IoU:

0.55–0.75

---

Pointing Game:

70–85 %

---

# Edge Deployment Metrics

Mandatory:

Model Size

Inference Latency

RAM Usage

Frames Per Second

---

Expected:

Model Size:

2.5–5 MB

---

Latency:

80–150 ms

---

RAM:

<500 MB

---

FPS:

5–10

---

# Statistical Validation

Recommended:

McNemar Test

Wilcoxon Signed Rank Test

Bootstrap Confidence Interval

---

All reported improvements should be statistically justified whenever applicable.

---

# Final Rule

These evaluation protocols are frozen and must remain unchanged throughout the MulXAI-CropNet project.

The Cross Validation scope clarification in this document reflects ADR-007 and does not alter any other frozen requirement.
