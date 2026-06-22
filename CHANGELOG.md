# CHANGELOG.md

# MulXAI-CropNet Changelog

All important modifications to the project are documented here.

---

## Version 1.0

Date:

March 2026

Status:

Initial Research Design

Added:

* Initial Literature Review
* Research Gap Identification
* CPS Workflow
* Proposed Architecture

---

## Version 1.1

Date:

June 2026

Status:

Research Audit Applied

Major Additions:

* Kendall Uncertainty Weighted Loss
* LLRL Baseline
* Pointing Game Metric
* Cross Domain Severity MAE
* SG-GradCAM++
* Severity-aware CPS Recommendation

---

Research Gaps Frozen:

1.

Lightweight Multi-task Learning

---

2.

Continuous Severity Regression

---

3.

Severity-aware Explainability

---

4.

Severity-aware CPS Recommendation

---

5.

Cross Domain Generalization

---

## Datasets Frozen

PlantVillage

Primary Training

---

TomatoVillage Variant B

Pseudo Label Validation

---

TomatoVillage Variant C

XAI Evaluation

---

CCMT

Cross Domain Testing ONLY

---

## Evaluation Protocol Frozen

10 Fold Stratified Cross Validation

---

PlantVillage

70

15

15

---

Cross Domain:

Train:

PlantVillage

Test:

CCMT

---

## Architecture Frozen

Shared Backbone:

EfficientNetB0 + SE

---

Classification Head

---

Severity Head

---

SG-GradCAM++

---

Treatment Recommendation

---

CPS Decision Layer

---

ONNX

INT8 Quantization

Raspberry Pi Deployment

---

## Official Member Ownership

Member 1

Classification

Backbone

---

Member 2

Pseudo Labels

Severity

---

Member 3

XAI

CPS

Deployment

---

## Repository Governance

Protected Branch:

main

---

Working Branches:

member1-classification

member2-severity

member3-xai-cps

---

Direct commits to main:

FORBIDDEN

---

## Version 1.1 Final Statement

Research Direction:

Frozen

---

Datasets:

Frozen

---

Architecture:

Frozen

---

Evaluation Protocol:

Frozen

---

Publication Figures:

Frozen

---

Only implementation changes are permitted beyond this point.

---

Future versions:

v1.2

Implementation Updates

---

v2.0

Integrated MulXAI-CropNet

---

Current Status:

Research Planning Complete

Implementation Ready
