# MEMBER 3 GUIDE

# MulXAI-CropNet

## SG-GradCAM++, CPS Recommendation and Edge Deployment

---

# ROLE

You are responsible for:

1.

Explainable AI

(SG-GradCAM++)

2.

Severity-aware Treatment Recommendation

3.

CPS Decision Layer

4.

Edge Deployment

(Raspberry Pi 5)

---

This is NOT an optional module.

This is one of the PRIMARY NOVELTIES of the paper.

---

# RESEARCH GAPS YOU ARE ADDRESSING

Existing papers:

Use:

GradCAM

or

GradCAM++

---

But:

They explain:

"Which regions influenced prediction"

ONLY.

---

They DO NOT explain:

"How disease severity influences model attention."

---

Similarly:

Most CPS systems:

Disease

↓

Recommendation

---

But:

Ignore:

Severity

Urgency

Intervention Priority

---

This is the gap you are solving.

---

# FINAL OBJECTIVE

Build:

```text id="0r1pgx"
Disease

+

Severity

↓

SG-GradCAM++

↓

Treatment Recommendation

↓

CPS Decision

↓

Raspberry Pi Deployment
```

---

# INPUTS

You DO NOT train a CNN.

You consume:

Member 1 outputs:

Disease Class

---

Member 2 outputs:

Severity Score

---

Example:

```text id="bjhqk0"
Disease:

Early Blight

Severity:

42 %
```

---

Input:

```text id="rzhz0a"
Early Blight

42 %
```

---

Output:

```text id="7m86ep"
GradCAM Heatmap

Treatment Recommendation

Moderate Urgency

CPS Alert
```

---

# DATASETS

You will use:

PlantVillage

for:

GradCAM visualization

---

TomatoVillage

Variant C

for:

Ground Truth Bounding Boxes

---

Location:

datasets/

TomatoVillage/

Variant-c-Object_Detection

---

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

This dataset is EXTREMELY IMPORTANT.

Because:

It allows:

Quantitative XAI evaluation.

---

# XAI MODULE

Implement:

---

1

GradCAM

---

2

GradCAM++

---

3

SG-GradCAM++

(Proposed Method)

---

# SG-GradCAM++ IDEA

Standard:

```text id="pv36wb"
GradCAM

↓

Heatmap
```

---

Our method:

```text id="y5s4za"
Heatmap

×

Severity Weight

↓

SG-GradCAM++
```

---

Example:

Severity:

10 %

↓

Weak lesion highlighting

---

Severity:

85 %

↓

Strong lesion highlighting

---

Therefore:

Attention becomes:

Severity Aware.

---

# IMPLEMENTATION

Create:

src/xai/

---

Files:

gradcam_baseline.py

gradcamplusplus.py

sg_gradcampp.py

heatmap_utils.py

visualize.py

---

# OUTPUTS

Save:

outputs/

heatmaps/

---

Structure:

```text id="0y9gzy"
heatmaps/

gradcam/

gradcampp/

sg_gradcampp/
```

---

Save:

Original Image

Heatmap

Overlay

---

# XAI METRICS

VERY IMPORTANT

Report:

---

IoU

Intersection over Union

---

Pointing Game Score

---

Localization Accuracy

---

Expected:

IoU:

0.55

to

0.75

---

Pointing Game:

70%

to

85%

---

These numbers come directly from:

Research Audit June 2026.

---

# TREATMENT RECOMMENDATION

This is the CPS contribution.

---

Input:

Disease

*

Severity

---

Output:

Treatment

Urgency

Recommendation

---

Example:

```text id="u4hdzb"
Severity

0–20 %

↓

Monitor
```

---

```text id="d0rdik"
21–50 %

↓

Mild Treatment
```

---

```text id="o6l9tv"
51–80 %

↓

Immediate Intervention
```

---

```text id="y4ewyq"
>80 %

↓

Urgent Action
```

---

# TREATMENT TABLE

Create:

src/cps/

treatment_rules.py

---

Store:

Dictionary:

Disease

↓

Severity Range

↓

Recommendation

---

Example:

```text id="36skga"
Early Blight

0-20

Monitor
```

---

```text id="pqjzze"
Early Blight

21-50

Copper Fungicide
```

---

```text id="mhltl6"
Early Blight

>50

Immediate Spray
```

---

# IMPORTANT

Treatment recommendations MUST:

Be literature supported.

---

Use:

FAO

EPPO

Plant pathology references

---

DO NOT invent pesticides.

---

# CPS MODULE

Create:

src/cps/

---

Files:

recommendation_engine.py

decision_layer.py

alert_system.py

---

Workflow:

```text id="r35eha"
Disease

+

Severity

↓

Recommendation

↓

Urgency

↓

Alert

↓

GPIO Trigger
```

---

GPIO trigger:

Conceptual only.

---

You DO NOT need:

Actual hardware implementation.

---

# EDGE DEPLOYMENT

You are responsible for:

Raspberry Pi 5 Deployment.

---

Convert:

PyTorch

↓

ONNX

↓

INT8 Quantization

↓

Raspberry Pi

---

Files:

deploy_model.py

onnx_export.py

quantize_model.py

---

# EXPECTED RESULTS

Model Size:

2.5 MB

to

5 MB

---

Latency:

80

to

150 ms

---

Peak RAM:

< 500 MB

---

FPS:

5

to

10

---

# FOLDER ACCESS

You MAY EDIT:

src/xai/

src/cps/

outputs/heatmaps/

outputs/logs/

experiments/

notebooks/

---

You MAY READ:

outputs/checkpoints/

Member1 models

Member2 models

---

You MUST NOT TOUCH:

src/models/

src/preprocessing/

src/training/

---

You are NOT allowed to:

Modify:

EfficientNetB0

SE Blocks

Severity Head

Loss Functions

---

# FILES TO CREATE

src/xai/

gradcam_baseline.py

gradcamplusplus.py

sg_gradcampp.py

visualization.py

metrics.py

---

src/cps/

treatment_rules.py

recommendation_engine.py

decision_layer.py

deploy_model.py

quantize_model.py

---

# GRAPHS TO PRODUCE

Generate:

Heatmap Comparisons

---

GradCAM

vs

GradCAM++

vs

SG-GradCAM++

---

IoU Comparison

---

Pointing Game Score

---

Latency Plot

---

Model Size Comparison

---

Save:

outputs/metrics/

---

# COMPARISON PAPERS

Compare against:

LeafAI

XSE TomatoNet

GradCAM

GradCAM++

AI-IoT Pivot

Ayid Framework

Tiny LiteNet

---

Columns:

XAI

Severity Aware

IoU

Pointing Game

CPS

Recommendation

Edge Deployment

---

# DO NOT DO

Do NOT:

Train a CNN

Create a backbone

Change EfficientNet

Change Member1 outputs

Change Member2 outputs

Use Vision Transformers

Use YOLO

Use U-Net

Create segmentation models

Create cloud deployment

Use AWS

Use Docker

Use Kubernetes

---

# CLAUDE APP RULES

Whenever prompting Claude:

Always say:

This is undergraduate research.

Prioritize:

simple

lightweight

research oriented

modular

reproducible

---

Never ask:

"Create enterprise CPS"

Never ask:

"Build production IoT"

Never ask:

"Create cloud deployment"

---

Always ask:

"Implement a minimal research-grade CPS and XAI pipeline."

---

# FINAL SUCCESS CONDITION

Your work is COMPLETE only when:

Disease

*

Severity

↓

SG-GradCAM++ generated

↓

IoU computed

↓

Pointing Game computed

↓

Treatment Recommendation generated

↓

CPS decision created

↓

ONNX exported

↓

Quantized model generated

↓

Latency benchmarked

↓

Raspberry Pi deployment ready

↓

Integrated with:

Member 1

and

Member 2

↓

MulXAI-CropNet Complete.
