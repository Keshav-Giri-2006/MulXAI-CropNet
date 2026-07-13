# ARCHITECTURAL_DECISIONS.md

# MulXAI-CropNet
## Architecture Decision Record (ADR)

---

# Purpose

This document records major architectural and research decisions made during the development of MulXAI-CropNet.

Unlike `CHANGELOG.md`, which records chronological repository changes, this document explains **why important technical decisions were made**.

Every decision recorded here is considered **frozen** unless a future ADR explicitly supersedes it.

---

# Decision Status

Each decision contains:

- ID
- Date
- Status
- Context
- Decision
- Rationale
- Impact

Status values:

- Accepted
- Superseded
- Deprecated

Current Version:

**ADR Version 1.0**

---

# ADR-001

## Title

EfficientNet-B0 Backbone Terminology Clarification

### Date

July 2026

### Status

Accepted

---

## Context

Early project planning referred to the proposed backbone as:

> EfficientNet-B0 + SE

During implementation review, it was identified that the standard EfficientNet-B0 architecture already contains native Squeeze-and-Excitation (SE) blocks within every MBConv block.

Implementing an additional external SE module would create an artificial architecture that was never part of the intended research contribution.

---

## Decision

MulXAI-CropNet will use the standard torchvision EfficientNet-B0 implementation without adding any external SE module.

The repository identifier:

`efficientnetb0_se`

is retained solely for project compatibility and historical continuity.

It does **not** represent a different neural network architecture.

---

## Rationale

This decision:

- preserves compatibility with existing code
- maintains existing model identifiers
- avoids unnecessary architectural complexity
- improves technical correctness
- prevents reviewer confusion
- aligns implementation with EfficientNet literature

---

## Impact

No implementation changes required.

Documentation updated to clarify terminology.

Model identifiers remain unchanged.

---

# ADR-002

## Title

Research Novelty Scope

### Date

July 2026

### Status

Accepted

---

## Context

The project originally focused on multiple lightweight backbone experiments.

During research refinement, it became clear that the scientific novelty should not depend on creating another CNN architecture.

---

## Decision

The research novelty of MulXAI-CropNet comes from the complete integrated pipeline rather than modifications to the backbone CNN.

Novel contributions include:

- disease classification
- continuous severity regression
- HSV-based pseudo-label generation
- uncertainty-aware multi-task learning
- SG-GradCAM++
- CPS treatment recommendation
- Raspberry Pi edge deployment
- cross-domain evaluation
- explainable agricultural AI workflow

---

## Rationale

This creates a stronger and more defensible research contribution than proposing a minimally modified backbone.

---

## Impact

The backbone remains lightweight and reproducible.

Future work should focus on system-level improvements rather than classifier redesign.

---

# ADR-003

## Title

Dataset Responsibilities

### Date

July 2026

### Status

Accepted

---

## Context

Three tomato datasets are available:

- PlantVillage
- TomatoVillage
- CCMT

Each dataset serves a different purpose.

---

## Decision

PlantVillage

- primary training dataset
- Member 1 classification
- Member 2 severity regression

CCMT

- cross-domain evaluation only
- never used for model training

TomatoVillage

Variant A

- optional future benchmarking

Variant B

- reserved for future multi-label experiments

Variant C

- reserved for future object detection work

---

## Rationale

This prevents dataset leakage while maintaining clear experimental boundaries.

---

## Impact

Evaluation protocol remains reproducible.

Cross-domain experiments remain scientifically valid.

---

# ADR-004

## Title

Member Responsibility Isolation

### Date

July 2026

### Status

Accepted

---

## Context

The project is developed by three independent contributors working on separate Git branches.

Uncontrolled modification across components increases merge conflicts and integration risk.

---

## Decision

Member responsibilities are frozen.

Member 1

- disease classification
- lightweight backbone
- training
- evaluation

Member 2

- HSV preprocessing
- pseudo-label generation
- severity regression
- uncertainty-aware learning

Member 3

- GradCAM
- SG-GradCAM++
- CPS decision layer
- ONNX export
- quantization
- Raspberry Pi deployment
- final repository integration

No member may modify another member's implementation without explicit approval.

---

## Rationale

This minimizes merge conflicts and preserves clear ownership.

---

## Impact

Git branches remain independent until planned integration.

---

# ADR-005

## Title

Implementation Philosophy

### Date

July 2026

### Status

Accepted

---

## Context

Large language models frequently introduce unnecessary abstractions and architectural complexity.

---

## Decision

MulXAI-CropNet follows the following engineering philosophy:

Preserve

↓

Repair

↓

Verify

↓

Stop

Every implementation should:

- minimize code changes
- preserve working modules
- avoid unnecessary abstractions
- prioritize reproducibility
- remain undergraduate research friendly

---

## Rationale

Simple, reproducible research software is easier to validate, maintain, and publish.

---

## Impact

Future contributors should favor incremental improvements over large refactors.

---

# ADR-006

## Title

Frozen Implementation Order

### Date

July 2026

### Status

Accepted

---

## Decision

The implementation order is fixed.

Member 1

↓

Freeze Backbone

↓

Member 2

↓

Freeze Severity Model

↓

Member 3

↓

Freeze XAI

↓

Freeze CPS

↓

Freeze Edge Deployment

↓

Final Integration

This sequence shall not change without an approved ADR.

---

## Rationale

Dependencies flow naturally from classification to severity estimation, then explainability and deployment.

---

## Impact

Prevents downstream work from depending on unstable upstream components.

---

# ADR-007

## Title

Interaction Between the Fixed 70/15/15 Split and 10-Fold Stratified Cross Validation

### Date

July 2026

### Status

Accepted

---

## Context

`EVALUATION_PROTOCOL.md`, `Project_Specification_v1.1.md`, `MEMBER1_CLASSIFICATION_GUIDE.md`, and `DATASET_USAGE.md` each independently mandate two requirements for the PlantVillage dataset:

1. A fixed 70% / 15% / 15% train / validation / test split, with random reshuffling after experiments explicitly forbidden.
2. 10-Fold Stratified Cross Validation, reported as Mean, Standard Deviation, and 95% Confidence Interval across folds.

None of these documents specified how the two requirements combine procedurally. Read literally, "split data into 10 folds" could be interpreted as operating over the entire dataset, which would conflict with maintaining a single, permanently untouched test partition — since a dataset-wide 10-fold procedure would, across its 10 repetitions, use every sample as test data at some point, including the samples otherwise designated as the fixed 15% test set.

This ambiguity was identified during the Member 1 implementation audit and confirmed, by exhaustive search of all governance documents, to be a genuine specification gap rather than an oversight resolvable by inspection.

---

## Decision

The following procedure is the authoritative interpretation and shall govern all Member 1 evaluation going forward:

1. PlantVillage is split exactly once into a fixed 70% / 15% / 15% train / validation / test partition, using stratified sampling with a fixed random seed, as already implemented in `train_val_test_split()`.
2. The 15% test partition is set aside and remains permanently held out. It is not used in any capacity during Cross Validation.
3. 10-Fold Stratified Cross Validation is performed only within the remaining 85% (the combined train + validation pool), not across the full dataset.
4. Cross-validation statistics — Mean, Standard Deviation, and 95% Confidence Interval — are computed across the 10 folds drawn from this 85% pool, for all mandated metrics (Accuracy, Precision, Recall, Macro F1).
5. After model selection via Cross Validation, the finally selected backbone is evaluated exactly once on the untouched 15% test partition. This single evaluation produces the final reported test metrics; it is not repeated, reshuffled, or averaged with the CV folds.

---

## Rationale

This interpretation:

- satisfies "Random reshuffling after experiments: NOT ALLOWED" literally, since the test partition is fixed once and never re-entered into any subsequent split or fold;
- satisfies the Cross Validation requirement's statistical intent (Mean/Std/95% CI across folds) without contaminating the final test evaluation with data the model selection process has already seen;
- follows standard nested-validation methodology (CV for model selection and robustness estimation, a single untouched holdout for the final reported number), which is the conventional resolution to this exact ambiguity in machine learning research practice;
- avoids retroactively reinterpreting the already-implemented `train_val_test_split()` function, minimizing implementation churn per ADR-005.

---

## Impact

- `train_val_test_split()` in `src/training/dataset_loader.py` is unchanged and continues to produce the fixed 70/15/15 partition.
- A new stratified 10-fold split utility, operating only on the 85% train+validation pool (not the full dataset and not including the 15% test partition), is required for Cross Validation implementation.
- The 15% test partition must never be passed into the Cross Validation fold-generation logic.
- `EVALUATION_PROTOCOL.md`, `Project_Specification_v1.1.md`, `MEMBER1_CLASSIFICATION_GUIDE.md`, and `DATASET_USAGE.md` are updated to reflect this resolved protocol (see corresponding Changelog/document updates).

---

# Future ADRs

Future architectural decisions should follow this template.

---

## ADR-XXX

Title

Date

Status

Context

Decision

Rationale

Impact

---

# End of Document
