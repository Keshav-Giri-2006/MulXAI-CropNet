# Research Audit Report: State-of-the-Art Review & Publication Readiness Assessment
## Project: Lightweight Explainable Multi-Task Deep Learning Framework for Tomato Disease Detection, Continuous Severity Regression, and CPS-Based Treatment Recommendation
**Audit Date:** 9 June 2026 | **Auditor Role:** Senior Journal Reviewer & Agricultural AI Research Strategist

***

## Executive Summary

This audit evaluates the current 30-paper literature foundation (file: CPS-Reference-Papers-Summary.xlsx) and the finalized six-objective methodology document (file: CPS-Proposed-System-Objectives-and-Expected-Results.docx) against the state of the art as of June 2026. The overall assessment is that the project carries **Strong** publication potential in its current form, and can be elevated to **Very Strong** through targeted modifications identified in this report. The core novelty — a lightweight end-to-end multi-task pipeline combining continuous image-based severity regression, a novel SG-GradCAM++ explainability layer, and severity-conditioned CPS treatment recommendation on a Raspberry Pi 5 — remains unmatched by any single published work as of June 2026.

***

## Phase 1 — Literature Coverage Assessment

### Is 30 papers sufficient?

Yes — and marginally exceeds what is necessary. For a focused empirical paper at *Computers and Electronics in Agriculture* (CEA), *IEEE Transactions on AgriFood Electronics*, or *Scientific Reports*, 25–35 well-chosen references is the expected norm. The current 30-paper set is appropriately scoped. Expanding beyond 35 risks diffusing the narrative; the priority is depth of engagement with each paper, not raw count.

### Coverage Evaluation by Subdomain

| Subdomain | Papers in Set | Assessment |
|---|---|---|
| Tomato disease classification | 6 (XSE-TomatoNet, MTDL, LDI-NET, ResNet-FPN Islam, LDAMNet, Hoang 2026) | **Well covered** |
| Severity estimation | 7 (MTDLF, MTDL, Nasir, TSTC, Citrus Seg, Pomegranate HBDS, Pome Lite-YOLACT) | **Well covered** |
| Multi-task learning | 6 (MTDLF, MTDL, LDI-NET, ResNet-FPN, DH-CNN, Rose ViT) | **Well covered** |
| Explainable AI | 5 (XSE-TomatoNet, Balanageshwara, XAI Quant, LeafAI, Nasir) | **Adequate** |
| Quantitative XAI | 3 (Balanageshwara, XAI Quant Sci Rep, LeafAI Grad-CAM) | **UNDERREPRESENTED** — no IoU-vs-lesion benchmark exists yet |
| Edge AI / Raspberry Pi | 5 (Tiny-LiteNet, RTR_Lite, Hoang, L-Net, BMCNN) | **Well covered** |
| Model compression / quantization | 3 (Hasan KD+INT8, Thermal Edge IEEE GRSL, STRM-KD) | **Needs one more INT8 specifically for multi-task** |
| Domain adaptation | 5 (MGA, MSUN, Jeon BG-UDA, DG-PLDR, PlantCLR) | **Well covered** |
| Cross-domain robustness | 4 | **Well covered** |
| Agricultural CPS | 3 (AI-IoT Pivot, Ayid Framework, Tiny-LiteNet) | **Underrepresented for 2026 reviewers** |
| Treatment recommendation | 4 (Frontiers biocontrol, Pomegranate RAG-LLM, Ayid, AI-IoT) | **Adequate but weak on severity-conditioned** |

### Which Papers Are Now Outdated or Add Low Value?

The following three papers should be considered for replacement or demotion to a footnote citation, as newer papers make them redundant or their contribution is subsumed:

1. **LDI-NET (Yang et al., 2024, Sci. Rep.)** — The tri-task CNN+Transformer with multi-label severity categories is now superseded by more recent multi-task papers (Rose ViT 2026, ResNet-FPN 2026) that provide cleaner architectural comparisons. LDI-NET still has citation value for the AI Challenger dataset but is weak as a primary baseline.
2. **Subbarayudu et al. (PLOS ONE, 2025) — Maize DCNN** — Maize-only; the pseudo-label method is partially analogous but crop and architecture choices are so different that reviewers may question why it is included over a tomato-specific paper.
3. **Islam ResNet-FPN (IJAIMLDS, 2026)** — Published in a non-indexed journal (IJAIMLDS) with unreported results. This is the weakest paper in the set and is a liability in a reviewer's eyes. Should be either replaced or relegated to a brief comparison mention.

### Missing Papers and Missing Areas

**Critical gap — Quantitative XAI benchmarking with IoU:**
The LLRL paper (Plant Phenomics, 2025, DOI: 10.1016/j.plaphe.2025.100058) by [Pubmed 41415163] proposes location-guided lesion representation learning specifically for *severity assessment* of apple, potato, and tomato with 12,098 annotated images and demonstrates at least 1% accuracy gain in severity assessment. This is the closest published precedent for IoU-based XAI-severity linkage and is currently absent from the literature set.

**Critical gap — Uncertainty-weighted multi-task loss:**
The Kendall, Gal & Cipolla (CVPR 2018) paper on homoscedastic uncertainty loss weighting for multi-task learning (2,610+ citations) is the canonical reference for replacing fixed α/β weights in the composite loss \(\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{MSE}\) with learned task uncertainty weights. Its absence from the bibliography is a notable weakness that reviewers of multi-task learning papers will flag.

**Moderate gap — Recent tomato ensemble + EfficientNetB0 (2026):**
The hybrid DenseNet121+EfficientNetB0 ensemble with SMOTE (Frontiers in Horticulture, 2026, DOI: 10.3389/fhort.2026.1762580) achieves 98.72% ± 0.18 mean accuracy on tomato and uses SMOTE for class imbalance — directly comparable to the classification head in Objective 1.

**Moderate gap — EfficientNetV2 + MobileNetV2 edge hybrid (2025):**
The edge-hybrid EfficientNetV2 + MobileNetV2 + SE + ViT comparison paper (Journal of Edge Computing, 2025, DOI: 10.55056/jec.905) achieves 99.5% accuracy with 0.15 s/image and 97.97% field accuracy, providing a direct edge-deployment comparison for Objective 5.

**Moderate gap — XAI systematic review for agriculture (Computers and Electronics in Agriculture, 2025):**
A systematic review of XAI for spectroscopy/precision agriculture appears in CEA vol. 235 (2025, art. 110354). Citing a review from the target journal strengthens manuscript positioning.

***

## Phase 2 — Novelty Assessment

### Contribution A: Multi-Task Lightweight Architecture (EfficientNetB0 + SE + dual head)

**A. Still novel?** Yes, partially. The exact combination of EfficientNetB0 + SE + simultaneous classification + continuous regression heads has not been published for tomato as of June 2026.
**B. Similar approaches published?** MTDL (2024, Front. Plant Sci.) uses multi-task EfficientNet with KD but focuses on classification+ordinal severity. Rose ViT (BMC, 2026) uses transformer multi-task with severity but is not lightweight. DH-CNN (Smart Agric. Tech., 2025) uses dual heads but no severity regression. None combine all four: EfficientNetB0 + SE + continuous regression + edge deployment on a single model.
**C. Novelty weakened by recent literature?** Weakened slightly by the Rose ViT paper (2026) for multi-task severity and by MTDLF (2026) for continuous regression. However, neither is lightweight nor edge-deployable.
**D. How to strengthen?** Replace the fixed α/β grid search loss with **learnable uncertainty-weighted loss** (Kendall et al. CVPR 2018). This is a well-established technique (2,610 citations) that would: (1) eliminate the manual grid search, (2) allow the model to dynamically balance the classification/regression trade-off during training, (3) add a formally justifiable methodological contribution beyond architecture design, and (4) is directly citable in a multi-task paper without adding implementation complexity.

### Contribution B: Continuous Severity Regression (0–100%)

**A. Still novel?** Yes — for tomato specifically. The MTDLF paper achieves continuous regression (MAE=7.5, R²=0.92) but uses ResNet-50 and includes segmentation, making it a heavier system. MTDL uses continuous severity but only achieves a categorical quality output in practice.
**B. Similar approaches published?** LLRL (Plant Phenomics, 2025, DOI: 10.1016/j.plaphe.2025.100058) uses image-generation-based location-guided representation for severity on apple, potato, and tomato — the most direct competitor for continuous severity on tomato. The Pomegranate HBDS paper uses a continuous deviation score via Grad-CAM++ lesion area, which is a second competitor.
**C. Novelty weakened?** Moderately weakened by LLRL. However, LLRL does not involve multi-task learning, does not use pseudo-labels, and does not integrate with CPS treatment. The combination remains novel.
**D. How to strengthen?** (1) Explicitly cite LLRL as the strongest single-task comparator and demonstrate that the proposed multi-task continuous regression achieves competitive or superior severity MAE with a 5× smaller model; (2) add the LLRL approach as Baseline 7 in the comparison table.

### Contribution C: Severity Pseudo-Label Generation (HSV pixel-ratio)

**A. Still novel?** Moderately novel. HSV-space pseudo-labelling is used in Subbarayudu et al. (maize) and alluded to in the Citrus segmentation paper. However, the specific application to PlantVillage tomato with expert-validated correlation r ≥ 0.85 remains original.
**B. Similar approaches published?** Subbarayudu et al. (PLOS ONE, 2025) applies pixel-area-ratio pseudo-labels for maize — the closest analogue. The 2023 CNN severity review specifically calls pixel-ratio the most reproducible method.
**C. Novelty weakened?** Weakened because HSV pixel-ratio is now established methodology. The novelty lies not in the method itself but in: (a) its systematic application to PlantVillage tomato with cross-validation against expert labels, and (b) its use as the first such validated pseudo-label pipeline enabling multi-task EfficientNetB0 training on PlantVillage without external annotation.
**D. How to strengthen?** Explicitly report the inter-rater agreement statistics (Cohen's κ or Pearson r between pseudo-labels and expert ratings on the 200-image validation set). This transforms pseudo-labelling from a utility step into a reportable contribution. Targeting r ≥ 0.85 is appropriate and achievable based on prior work.

### Contribution D: Severity-Guided GradCAM++ (SG-GradCAM++)

**A. Still novel?** **Yes — this is the strongest novelty claim in the project and remains unmatched as of June 2026.** No published paper applies severity-score-weighted activation map scaling to Grad-CAM++ for plant disease. The HBDS pomegranate paper uses Grad-CAM++ to *measure* severity but does not use severity to *modulate* the heatmap.
**B. Similar approaches published?** LLRL (Plant Phenomics, 2025) uses location guidance for severity representation, but the modulation mechanism is entirely different (image generation vs. activation weighting). The Balanageshwara comparative XAI study (ETASR, 2026) establishes Grad-CAM++ as the qualitative benchmark but provides no severity-guided variant.
**C. Novelty weakened?** Not weakened — SG-GradCAM++ remains a genuine technical contribution with no direct prior art.
**D. How to strengthen?** (1) Formally define the SG-GradCAM++ weighting operation as a clearly stated mathematical expression to make it reviewer-citeable; (2) measure both the Pointing Game score and IoU vs. lesion masks — reviewers increasingly expect both metrics for XAI evaluation in 2026; (3) add a qualitative user study with 3–5 agricultural domain experts rating heatmap utility on a Likert scale — this is now standard in agricultural XAI papers.

### Contribution E: CPS Treatment Recommendation Module

**A. Still novel?** **Yes — the severity-graduated, offline, rule-based treatment recommendation tied to a CPS actuator pipeline on Raspberry Pi remains unmatched.** The Pomegranate RAG-LLM paper is the closest competitor but requires cloud LLM access. The AI-IoT Pivot (Ibrahim 2025) triggers binary treatment with no severity grading.
**B. Similar approaches published?** The Frontiers biocontrol paper (Gunasekaran, 2026) provides class-conditional biocontrol recommendations without severity conditioning. No published paper implements the four-tier quantile severity (0–25%, 25–50%, 50–75%, >75%) dosage graduation linked to a Raspberry Pi GPIO output.
**C. Novelty weakened?** Not weakened, but reviewers may question the rule-based nature. In 2026, reviewers increasingly expect either: (a) a learned recommendation model, or (b) clinical/agronomic validation of the rule table against real-world treatment outcomes.
**D. How to strengthen?** (1) Ground each treatment rule entry in a specific cited plant pathology guideline (e.g., FAO treatment protocols, national pest management guides); (2) add a simple expert validation survey where 3 plant pathologists rate rule correctness and completeness; (3) consider adding a "Rule confidence score" column to the treatment table to communicate uncertainty — this adds a CPS-safety dimension that reviewers value.

### Contribution F: Edge Deployment on Raspberry Pi 5

**A. Still novel?** Moderately novel. RTR_Lite_MobileNetV2 (Duhan 2025), Tiny-LiteNet (Nyakuri 2025), and L-Net (2026) all demonstrate Raspberry Pi deployment. However, multi-task model deployment with both classification and severity regression heads on RPi5 has not been reported for tomato.
**B. Similar approaches published?** Hoang et al. (Sci. Rep., 2026) benchmarks 7 lightweight models including EfficientNetB0 on Raspberry Pi and Jetson Nano for tomato — directly overlapping Objective 5.
**C. Novelty weakened?** Weakened for classification-only deployment. The novelty is preserved specifically for the multi-task (classification + regression + XAI generation) deployment profile, which is unmatched.
**D. How to strengthen?** Report a three-way latency breakdown: (1) backbone inference, (2) dual-head output, (3) SG-GradCAM++ generation — separately. Also report throughput (images/minute) as a practical metric for field use. These disaggregated numbers are not reported by any competitor paper.

### Contribution G: Cross-Domain Evaluation

**A. Still novel?** Moderate novelty. Hasan et al. (2025) already establishes the PlantVillage→TomatoVillage cross-domain benchmark for single-task classification. The proposed paper extends this with severity MAE measurement under domain shift, which is new.
**B. Similar approaches published?** All domain adaptation papers (MGA, MSUN, DG-PLDR) measure classification accuracy only; none measure severity metric degradation under domain shift. This is a clear gap the proposed paper fills.
**C. Novelty weakened?** Not weakened — the severity MAE under domain shift measurement is unique.
**D. How to strengthen?** Report separately: (1) zero-shot cross-domain classification accuracy drop; (2) zero-shot severity MAE increase; (3) fine-tuned (20% target data) performance recovery; (4) fine-tuned severity MAE recovery. This four-value matrix is more informative than single accuracy numbers and has no published precedent for multi-task tomato disease models.

***

## Phase 3 — Objective-by-Objective Audit

### Objective 1: Multi-Task Lightweight Architecture Design

**Classification: Needs Minor Revision**

The architecture design (EfficientNetB0 + SE + multi-scale fusion + dual heads) is well-specified and feasible. The composite loss \(\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{MSE}\) with grid-searched fixed weights is the primary weakness. In 2026, multi-task learning papers at journals above IF 5 are expected to justify their loss weighting strategy formally. The Kendall et al. CVPR 2018 uncertainty-weighting approach is the recommended replacement for the grid search component. The objective statement also refers to "quality detection" in its opening sentence ("disease detection, severity estimation, and quality detection") — this third task is never defined or evaluated anywhere in the methodology document. This inconsistency must be removed or defined to avoid reviewer rejection on grounds of unsupported claims.

**Recommended revision:** (a) Replace "α and β determined by grid search" with "α and β treated as learnable log-variance parameters following Kendall et al. [CVPR 2018] to enable automatic task-uncertainty-weighted loss balancing"; (b) remove "quality detection" from the objective statement or define it explicitly.

### Objective 2: Severity Pseudo-Label Generation

**Classification: Strong — should remain with one addition**

The two-step HSV pseudo-labelling pipeline (HSV masking → infected pixel ratio) with Pearson r ≥ 0.85 validation target is well-defined and achievable. The only gap is the absence of an inter-rater reliability metric. The 200-image expert validation set is well chosen.

**Recommended addition:** Report Cohen's κ between the HSV pseudo-label and two independent human rater annotations (in addition to Pearson r). This elevates the pseudo-labelling contribution from a utility step to a formally validated annotation protocol — reviewable as a standalone methodological contribution.

### Objective 3: Severity-Guided GradCAM++ (SG-GradCAM++) Implementation

**Classification: Strong — should remain with two additions**

The SG-GradCAM++ implementation and IoU evaluation against 200 annotated masks is the strongest single contribution in the project. The objective is well-defined, feasible, and novel. Two additions would substantially strengthen it:

**Recommended additions:** (a) Add the Pointing Game score as a second XAI metric alongside IoU — it is a widely accepted complementary metric; (b) add a brief qualitative expert rating (3–5 plant pathologists rate heatmap utility on a 5-point Likert scale) to provide a human-in-the-loop validation dimension that reviewers in 2026 increasingly request.

### Objective 4: Severity-Aware CPS Treatment Recommendation Module

**Classification: Needs Minor Revision**

The rule-based module is correctly designed and the four-tier severity quantile → treatment type+dosage+repetition logic is sensible. The current validation approach (consulting "research gaps finalized consultation") is vague and not scientifically rigorous enough for a 2026 journal submission.

**Recommended revision:** (a) Ground each rule in a specifically cited peer-reviewed plant pathology source (fungicide type from Agrios *Plant Pathology* textbook or EPPO guidelines); (b) add "Expert Validation Protocol" as a named sub-step — have 3 plant pathologists evaluate the 44-rule table (11 diseases × 4 severity tiers) on correctness and completeness, reporting the mean expert agreement score; (c) add a worked example demonstrating the full CPS pipeline from image capture → model inference → recommendation output → GPIO signal on Raspberry Pi 5 — this demonstrates CPS integration concretely.

### Objective 5: Edge Deployment and Quantization

**Classification: Needs Minor Revision**

The ONNX INT8 dynamic quantization pipeline targeting ≤5 MB and ≤200 ms on Raspberry Pi 5 is well-designed and feasible based on Tiny-LiteNet's 80 ms baseline. Two concerns: (1) the latency target of ≤200 ms is conservative — Hasan et al. achieves 0.29 ms with ShuffleNetV2-INT8, so reviewers may question why EfficientNetB0 multi-task achieves 200 ms. The answer lies in the multi-task overhead and SG-GradCAM++ computation, but this must be explicitly justified; (2) the comparison set (RTR_Lite vs ShuffleNetV2-INT8) is currently two models — adding Tiny-LiteNet (80 ms) as a third edge comparison strengthens the benchmarking.

**Recommended revision:** (a) Add Tiny-LiteNet as the third edge comparison baseline; (b) change the latency target to "≤200 ms per image including SG-GradCAM++ generation, targeting ≤150 ms for inference only" to properly separate model inference from XAI overhead; (c) report FLOPs alongside parameters and model size.

### Objective 6: Cross-Domain Generalization Evaluation

**Classification: Strong — should remain with one enhancement**

The three-dataset evaluation (PlantVillage → TomatoVillage → CCMT) with zero-shot + fine-tuning protocols is well-designed and aligns with the best published benchmarks (Hasan et al. 3.45% drop, MSUN 50.58% on tomato cross-domain).

**Recommended enhancement:** Explicitly report severity MAE on cross-domain sets alongside classification accuracy — this is currently mentioned in the objective but not in the expected results table. Adding expected severity MAE under domain shift (see Phase 4) makes Objective 6 the most comprehensive cross-domain evaluation of any multi-task tomato disease paper published to date.

### Should Additional Objectives Be Added?

**No new objectives are needed.** The six objectives form a complete, well-scoped pipeline. Adding a seventh objective (e.g., federated learning or online learning) would make the project unrealistic within a 5-week timeline and reduce publication focus.

***

## Phase 4 — Expected Results Audit

### Metric-by-Metric Assessment

| Metric | Current Range | Assessment | Updated Target |
|---|---|---|---|
| **Classification accuracy (PlantVillage test)** | 98.5%–99.3% | Realistic and competitive. Top single-task models now reach 99.66% (PlantaNet), 99.92% (RTR_Lite), and 99.9% (Frontiers biocontrol). Multi-task slightly reduces classification. Range is well-calibrated. | **Keep 98.5%–99.3%; flag that multi-task overhead may reduce by 0.2–0.5% vs single-task baseline** |
| **Macro F1 (PlantVillage tomato)** | 0.97–0.99 | Competitive. PlantaNetLite achieves 0.99. Lower bound of 0.97 acceptable for multi-task. | **Keep 0.97–0.99** |
| **Severity MAE (PlantVillage test)** | 5.5–8.5 | Realistic. MTDLF baseline: MAE=7.5. A tighter lower bound of 5.5 is achievable but optimistic for pseudo-labelled data. The LLRL paper reports "at least 1% improvement" in severity accuracy. | **Revise to 5.0–8.0; add footnote that MAE depends on pseudo-label quality** |
| **Severity R² (PlantVillage test)** | 0.87–0.93 | Realistic. MTDLF: R²=0.92. Upper bound is competitive. | **Keep 0.87–0.93** |
| **SG-GradCAM++ IoU vs. lesion masks** | 0.60–0.75 | Speculative — no direct prior published on plant disease XAI with lesion mask IoU. The medical XAI literature suggests 0.60–0.75 for comparable architectures. This is the highest-risk metric in the expected results table. | **Revise to 0.55–0.72 with explicit statement: "IoU target calibrated from medical XAI analogues as no plant disease XAI IoU baseline currently exists in the literature"** |
| **Cross-domain accuracy drop (→TomatoVillage)** | 3%–7% | Well-calibrated against Hasan et al. 3.45%. Multi-task may slightly increase degradation for classification but reduce it for severity. | **Keep 3%–7% for classification; add "severity MAE increase under domain shift: expected +2.0 to +4.5 percentage points"** |
| **Model size (INT8 post-quantization)** | 2.5–5.0 MB | Achievable. Hasan et al. achieves 1.46 MB for single-task ShuffleNetV2; EfficientNetB0 multi-task will be larger. Upper bound of 5 MB is appropriate. | **Keep 2.5–5.0 MB; target ≤4.5 MB for the two-head model** |
| **Latency (RPi 5, per image)** | 80–200 ms | The 80 ms lower bound is aggressive — that is Tiny-LiteNet's inference-only time with no XAI. The EfficientNetB0 multi-task + SG-GradCAM++ will realistically be 120–200 ms. | **Revise lower bound to 100 ms: target range 100–200 ms total (inference + XAI), 80–140 ms inference only** |
| **Peak RAM (RPi 5)** | 300–600 MB | Reasonable for an EfficientNetB0-class ONNX model. The RPi 5 has 4–8 GB RAM so this is feasible. | **Keep 300–600 MB; add throughput: expected 5–10 images/minute** |
| **XAI generation overhead** | 30–80 ms additional | Reasonable. Grad-CAM++ on EfficientNetB0 adds minimal overhead on GPU; on RPi 5 ARM, 30–80 ms is realistic. | **Keep 30–80 ms; report separately from inference latency** |

### Missing Metric: Cross-Domain Severity MAE

The expected results table contains no expected severity MAE under domain shift. This is a significant omission given that cross-domain severity measurement is proposed as a novel contribution in Objective 6. A target range of +2.0 to +5.0 percentage points increase in MAE from PlantVillage to TomatoVillage should be added, derived from the general finding that domain shift increases regression error proportionally to classification accuracy drop.

***

## Phase 5 — Evaluation Plan Audit

### Baselines

The six selected baselines are well-chosen. However, two additions would make the comparison table reviewer-proof:

| Missing Baseline | Why Needed |
|---|---|
| **Tiny-LiteNet (Nyakuri et al., 2025)** | Only direct published multi-task RPi5 benchmark at 80 ms; absent from current baseline list despite being in literature set |
| **LLRL (Plant Phenomics, 2025)** | The strongest published continuous severity model on tomato; if not included as a baseline, reviewers will ask why |

The existing six baselines (EfficientNetB0 single-task classification, EfficientNetB0+SE single-task severity, RTR_Lite_MobileNetV2, ShuffleNetV2-INT8, MTDLF, MTDL) are all appropriate and justified.

### Cross-Validation Plan

The 10-fold stratified cross-validation (Objective C.1) is appropriate, rigorous, and consistent with XSE-TomatoNet. The use of five independent random seed initializations with mean ± std reporting is excellent practice. No revision needed.

### Cross-Domain Evaluation

The zero-shot + 20/80 fine-tune split design is well-structured. One addition: include a **per-class** accuracy analysis on the cross-domain test set (not just aggregate accuracy). Per-class performance on rare disease classes under domain shift is a frequently highlighted gap in reviews.

### Ablation Studies

Seven ablation variants (A–G) are defined. This is thorough and would satisfy any reviewer. One suggested addition:

| Suggested Ablation H | Component | Purpose |
|---|---|---|
| **Variant H** | Replace learnable uncertainty loss (proposed) with fixed α/β (original) | Demonstrates benefit of uncertainty weighting over grid search — only relevant if uncertainty weighting is adopted in Objective 1 |

### Statistical Validation

The statistical validation plan is **exceptionally strong** and goes beyond what most 2026 agricultural AI papers provide:
- McNemar test for classification accuracy comparisons ✓
- Wilcoxon signed-rank test for severity MAE ✓
- Bootstrap confidence intervals (1,000 resamples) for XAI IoU ✓
- Bonferroni correction for multiple comparisons ✓
- Fairness protocols (identical splits, augmentation, preprocessing across all models) ✓

This statistical rigor is a significant manuscript strength. No revisions required.

### What Additional Reviewer Expectations Are Likely in 2026?

Based on current review standards at CEA, IEEE Transactions, and Scientific Reports in 2026, the following are likely review requests that are *not currently addressed*:

1. **Confusion matrix at class level** for both PlantVillage and cross-domain test sets — standard in all disease classification papers
2. **FLOPs (GFLOPs)** reported alongside parameters and model size — expected by IEEE reviewers in 2026
3. **Ablation on pseudo-label quality** — what happens to severity MAE if pseudo-labels have different noise levels (test with 3 threshold variants for the HSV mask)
4. **Energy consumption** on RPi 5 — emerging as a metric in edge AI papers (Watts or mWh per inference)
5. **Qualitative failure analysis** — show 3–5 examples where the model mis-classifies or produces poor severity estimates, with SG-GradCAM++ maps highlighting why — this is now considered standard practice for XAI papers

***

## Phase 6 — Publication Potential Analysis

### A. Current Plan: **Strong**

**Justification:** The current project as described in the methodology document contains a novel architecture (EfficientNetB0 + SE + dual-head for tomato), a validated pseudo-labelling pipeline, a genuinely novel XAI contribution (SG-GradCAM++), a severity-graduated CPS recommendation module, a Raspberry Pi 5 deployment benchmark, and a two-dataset cross-domain evaluation — all in a single integrated paper. No single published paper combines all six of these contributions simultaneously. The 30-paper literature foundation is well-constructed and justifies each gap. This profile matches acceptance criteria for *IEEE Transactions on AgriFood Electronics*, *Computers and Electronics in Agriculture* (IF 8.3), *Frontiers in Plant Science* (IF 5.6), and *Scientific Reports* (IF 3.8).

The **downside risk** in the current plan: (a) the fixed α/β grid-search loss may draw reviewer criticism in a multi-task journal paper; (b) the absence of a formal expert validation for treatment recommendations is a weakness at higher-IF venues; (c) no FLOPs or energy consumption data.

### B. Publication Potential After Recommended Changes: **Very Strong**

**Justification:** Adding uncertainty-weighted loss (Kendall CVPR 2018 method), expert validation for the treatment module, the Pointing Game XAI metric, LLRL as Baseline 7, cross-domain severity MAE, and the three missing expected-results entries would collectively: (1) remove all principal grounds for rejection at IF ≥ 5 venues; (2) position the paper as the **defining benchmark paper** for lightweight multi-task disease+severity+XAI on edge devices; (3) make the statistical analysis the strongest in the subfield.

After changes, the paper meets the bar for:
- *Computers and Electronics in Agriculture* (IF 8.3) — **primary target**
- *IEEE Transactions on AgriFood Electronics* — **secondary target**
- *Expert Systems with Applications* (IF 8.5) — **alternative target**
- *Frontiers in Plant Science* (IF 5.6) — **safety target**

***

## Phase 7 — High-Impact Recommendations

### Priority 1 — Changes Most Critical for Publication Probability

1. **Replace fixed α/β loss weighting with Kendall et al. (CVPR 2018) learnable uncertainty-weighted multi-task loss.** Add the paper to the literature set. This resolves the most likely methodology-level rejection at multi-task-aware venues. Implementation complexity is low (two learnable log-σ parameters).

2. **Add LLRL (Plant Phenomics, 2025, DOI: 10.1016/j.plaphe.2025.100058) as Baseline 7** in the severity regression comparison. This is the strongest single-task continuous severity model on tomato and its absence will be flagged by reviewers at Plant Phenomics and CEA.

3. **Add cross-domain severity MAE to the expected results table.** Zero-shot severity MAE on TomatoVillage and CCMT is currently not in the expected results despite being claimed as a novel contribution in Objective 6.

4. **Add Pointing Game score as second XAI evaluation metric** alongside IoU. The combination of IoU + Pointing Game provides a complete spatial explanation quality profile that reviewers at top venues now expect.

5. **Remove or replace the ResNet-FPN Islam (IJAIMLDS, 2026) paper** — it is published in a non-indexed journal with unreported results and is a liability in the reference list.

### Priority 2 — Changes That Moderately Improve Publication Probability

6. **Add the Kendall CVPR 2018 paper and EfficientNetV2+MobileNetV2 hybrid (JEC 2025)** to the literature set to strengthen the multi-task loss and edge deployment sections.

7. **Add a 3-pathologist expert validation protocol for the treatment recommendation rule table** — at minimum, report that 3 domain experts reviewed the rules and agreed with X/44 entries (correctness ≥ 90% expected).

8. **Add Tiny-LiteNet (Nyakuri 2025) to the edge deployment comparison baselines.**

9. **Add a confusion matrix and per-class F1 table** for both PlantVillage and the best cross-domain result — this is expected in any 2026 classification paper.

10. **Report FLOPs (GFLOPs) for the proposed model and all lightweight baselines** — this is standard in IEEE papers on edge AI.

### Priority 3 — Nice-to-Have Improvements

11. **Add energy consumption estimate (mWh per inference) on RPi 5** — emerging metric that differentiates from competitors.

12. **Add a qualitative failure analysis section** (3–5 failure examples with SG-GradCAM++ maps) — transforms the XAI section from a black-box evaluation into an interpretable diagnostic tool demonstration.

13. **Add a worked end-to-end CPS pipeline demonstration** (image → model → severity score → treatment recommendation → GPIO signal → actuator output) as a figure in the paper — this is the CPS contribution made visually concrete.

14. **Cite the CEA XAI systematic review (2025, art. 110354)** in the XAI background section — signals to reviewers that the paper is current with its target journal's own published reviews.

15. **Add ablation Variant H** (fixed loss weights vs. learnable uncertainty weights) if Priority 1 item 1 is adopted.

***

## Recommended Additional Papers (5 papers to add/replace)

| # | Paper | Year | Venue | DOI | Why Add |
|---|---|---|---|---|---|
| 1 | Kendall, Gal, Cipolla — Multi-Task Learning Using Uncertainty to Weigh Losses | 2018 | CVPR | [10.48550/arXiv.1705.07115](https://doi.org/10.48550/arXiv.1705.07115) | Foundational citation for uncertainty-weighted multi-task loss (2,610 citations); required to justify Objective 1 loss revision |
| 2 | LLRL — Location-Guided Lesion Representation Learning | 2025 | Plant Phenomics (IF 7.4) | [10.1016/j.plaphe.2025.100058](https://doi.org/10.1016/j.plaphe.2025.100058) | Strongest continuous severity competitor on tomato; must be included as Baseline 7 |
| 3 | EfficientNetV2 + MobileNetV2 Edge Hybrid | 2025 | Journal of Edge Computing | [10.55056/jec.905](https://doi.org/10.55056/jec.905) | 99.5% accuracy; 97.97% field accuracy; SE blocks + ViT hybrid on Android — direct edge competitor |
| 4 | DenseNet121+EfficientNetB0+SMOTE Ensemble Tomato | 2026 | Frontiers in Horticulture | [10.3389/fhort.2026.1762580](https://doi.org/10.3389/fhort.2026.1762580) | 98.72% ± 0.18 mean accuracy tomato; class imbalance handling comparison |
| 5 | XAI Systematic Review for Agriculture | 2025 | CEA | [10.1016/j.compag.2025.110354](https://doi.org/10.1016/j.compag.2025.110354) | Target journal's own XAI review; citing it signals current literature awareness |

**Paper to remove:** Islam ResNet-FPN (IJAIMLDS, 2026) — non-indexed journal, unreported results.

***

## Recommended Objective Changes (Summary)

| Objective | Change Type | Key Change |
|---|---|---|
| Obj. 1 | Minor revision | Replace fixed α/β grid search with learnable uncertainty-weighted loss (Kendall CVPR 2018); remove "quality detection" reference |
| Obj. 2 | Add inter-rater metric | Report Cohen's κ alongside Pearson r for pseudo-label validation |
| Obj. 3 | Add two metrics | Add Pointing Game score alongside IoU; add 3–5 expert Likert rating of heatmap quality |
| Obj. 4 | Minor revision | Cite specific plant pathology guidelines for each treatment rule; add named expert validation protocol |
| Obj. 5 | Minor revision | Add Tiny-LiteNet as third edge baseline; add FLOPs; separate inference latency from XAI overhead |
| Obj. 6 | Enhancement | Add expected severity MAE increase under domain shift to results table |

***

## Final Verdict

### Acceptance Reasons (if submitted late 2026)

1. **Unique combination of contributions** — no published paper simultaneously combines lightweight multi-task disease+severity, quantitatively validated novel XAI, severity-graduated offline CPS recommendation, and edge deployment with cross-domain evaluation for tomato. This makes the paper genuinely novel rather than incrementally improving one aspect.

2. **SG-GradCAM++ is the only novel XAI architecture in the project** and has no direct prior art. In an era where reviewers demand explainability, a formally defined and IoU-validated XAI method is highly publishable on its own merits.

3. **End-to-end CPS pipeline on Raspberry Pi 5** — there are virtually no published papers demonstrating multi-task disease+severity inference followed by a hardware actuator recommendation on RPi 5 for tomato. This closes the lab-to-field gap that reviewers frequently cite as missing in agricultural AI papers.

4. **Rigorous statistical validation** — the McNemar + Wilcoxon + Bootstrap CI + Bonferroni plan exceeds the statistical rigor of nearly all competing papers in the literature set. This is a strong acceptance signal for any top-quartile journal.

5. **Severity pseudo-label pipeline** with r ≥ 0.85 expert validation enables the entire continuous regression contribution on a dataset (PlantVillage) that no prior paper has used for continuous severity — directly addressing a gap identified in the literature by five separate papers in the current set.

### Rejection Reasons (if submitted late 2026)

1. **Fixed α/β loss weighting** — In 2026, multi-task learning papers at IF ≥ 5 venues are expected to justify their loss weighting method formally. A grid-searched fixed weight is the least principled option and will draw a "why not use uncertainty weighting?" reviewer comment.

2. **Treatment recommendation module lacks formal validation** — "research gaps finalized consultation" is insufficient for a peer-reviewed contribution. Without named expert validators or cited agronomic guidelines for each rule, reviewers will question the clinical validity of the CPS recommendations.

3. **No LLRL comparison** — If the paper claims continuous severity regression as a contribution without comparing to the 2025 Plant Phenomics LLRL paper (which also performs severity on tomato), reviewers familiar with that journal will flag it as an incomplete comparison.

4. **XAI IoU metric is self-referential** — The 200 annotated masks are generated by the project team. Without inter-annotator agreement statistics (κ ≥ 0.7 typically required), reviewers may question whether the IoU evaluation is biased toward the proposed method.

5. **Latency comparison gap** — Claiming ≤200 ms as an achievement while Hasan et al. achieves 0.29 ms for single-task ShuffleNetV2-INT8 requires explicit justification that the multi-task + XAI pipeline overhead is the reason for the 600× difference. Without this, a reviewer may incorrectly conclude the model is slow.