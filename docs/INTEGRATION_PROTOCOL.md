# INTEGRATION_PROTOCOL.md

# MulXAI-CropNet

## Official Integration Order

Version: 1.0

Status: Frozen

---

# Philosophy

MulXAI-CropNet is implemented incrementally.

Every module must stabilize before the next module begins integration.

---

# STAGE 1

MEMBER 1

Disease Classification

↓

EfficientNetB0

↓

MobileNetV3

↓

EfficientNetB0 (Final Selected Backbone — `efficientnetb0_se`)

↓

Cross Validation

↓

Freeze Backbone

---

Ownership:

Member 1

---

Modification Rights:

Member 1 ONLY

---

# STAGE 2

MEMBER 2

Pseudo Labels

↓

HSV Masking

↓

Lesion Extraction

↓

Severity Labels

↓

Severity Head

↓

Kendall Loss

↓

Cross Domain Severity

↓

Freeze Severity Module

---

Ownership:

Member 2

---

Modification Rights:

Member 2 ONLY

---

# STAGE 3

MEMBER 3

GradCAM

↓

GradCAM++

↓

SG-GradCAM++

↓

IoU

↓

Pointing Game

↓

Treatment Rules

↓

Recommendation Engine

↓

ONNX Export

↓

INT8 Quantization

↓

Freeze CPS Module

---

Ownership:

Member 3

---

Modification Rights:

Member 3 ONLY

---

# FINAL INTEGRATION

Frozen Backbone

*

Frozen Severity Module

*

Frozen SG-GradCAM++

*

Frozen CPS

↓

MulXAI-CropNet v1

---

# Ownership Rules

Member 2 MUST NOT modify:

Backbone

---

Member 3 MUST NOT modify:

Backbone

Severity Head

Loss Functions

---

Member 1 MUST NOT modify:

XAI

CPS

---

# Final Rule

No integration shall occur unless the preceding stage is frozen and verified.
