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