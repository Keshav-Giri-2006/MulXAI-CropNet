# FIGURE_SPECIFICATION.md

# MulXAI-CropNet

## Official Figure Specification

Version: 1.0

Status: Frozen

---

# General Guidelines

All figures must:

* White background
* Publication quality
* High resolution
* 300 dpi minimum
* Export in PNG and SVG
* Uniform font style
* Consistent arrows
* Professional appearance
* No flashy colors

---

# Figure 1

Complete Workflow

Data Collection

↓

Preprocessing

↓

Severity Label Generation

↓

Dataset Split

↓

Shared Backbone

↓

Classification

*

Severity

↓

SG-GradCAM++

↓

Treatment Recommendation

↓

CPS

↓

Edge Deployment

---

# Figure 2

MulXAI-CropNet Architecture

Shared EfficientNetB0 + SE

↓

Classification Head

↓

Severity Head

↓

SG-GradCAM++

↓

Treatment Recommendation

↓

Deployment

---

# Figure 3

Severity Pseudo Label Generation

RGB

↓

HSV

↓

Leaf Mask

↓

Lesion Mask

↓

Infected Area

↓

Severity %

---

# Figure 4

XAI Comparison

GradCAM

vs

GradCAM++

vs

SG-GradCAM++

---

Display:

Original Image

Heatmap

Overlay

---

# Figure 5

Treatment Recommendation Workflow

Disease

*

Severity

↓

Recommendation

↓

Urgency

↓

Treatment

---

# Figure 6

Cross Domain Evaluation

Train:

PlantVillage

↓

Test:

CCMT

↓

Classification Metrics

↓

Severity Metrics

---

# Figure 7

Edge Deployment Pipeline

PyTorch

↓

ONNX

↓

INT8 Quantization

↓

Raspberry Pi 5

↓

Inference

---

# Figure Saving Policy

Save:

PNG

SVG

---

Folders:

outputs/figures/

outputs/heatmaps/

---

# Final Rule

These figures are part of the official publication plan and shall remain consistent throughout implementation and paper writing.
