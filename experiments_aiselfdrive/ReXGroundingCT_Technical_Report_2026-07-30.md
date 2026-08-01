---
title: "ReXGroundingCT / VoxTell Technical Experiment Report"
subtitle: "Comprehensive synthesis of methods, diagnostics, results, failure modes, and unresolved questions"
status_through: "2026-07-30"
document_type: "Internal technical memory and AI-agent handoff"
primary_project_root: "/common/lidxxlab/chushu/RexGroundingCT"
---

# ReXGroundingCT / VoxTell Technical Experiment Report

**Status through 30 July 2026**

**Purpose.** This document consolidates the ReXGroundingCT / VoxTell work discussed and completed to date. It is designed for two readers: a human collaborator who needs a coherent technical history, and an AI research agent that must understand what has already been tried before proposing new experiments or diagnosing hidden errors.

**Scope and provenance.** The report synthesizes completed experiment summaries, project audits, internal presentation values, job records, and prior technical discussions. It is not a substitute for reopening every raw output. Each result is labeled by cohort and evidence status. Where an exact row, implementation detail, or metric name was not recoverable, the limitation is stated explicitly rather than reconstructed from guesswork.

**Central conclusion.** The project has not found a large, general, end-to-end improvement over its strongest matched baseline. Several interventions produced small or subgroup-specific gains, and one S3 prompt-adapter result was statistically positive, but no method solved the dominant full-volume localization and false-positive problem. The evidence also rules out a simplistic diagnosis that the model ignores text: prompt location changes strongly affect embeddings, logits, masks, and laterality. The remaining failure is better described as insufficient spatial selectivity, unsafe coupling, proposal/selection error, partial-label ambiguity, and training-inference mismatch.

<!-- PAGEBREAK -->

# Executive summary

| Direction | What was learned | Bottom-line interpretation |
| --- | --- | --- |
| Baseline reconstruction | Rebuilt fine-tuning and improved the provided checkpoint from 0.229 to 0.263 Dice. | Necessary success, but not a solution. |
| Loss / sampling | Category-adaptive Tversky gave a reproducible +0.00145 Dice and reduced FP; more complex focal/boundary losses did not help. | Error-specific weighting helps slightly; loss complexity is not the main bottleneck. |
| Text sensitivity | Location-word swaps strongly changed embeddings, logits, masks, and centroids; laterality often transferred to the opposite lung. | The model is not simply ignoring text. |
| Prompt adaptation | A frozen rank-16 adapter improved S3 by +0.00697 Dice with positive finding- and case-level CIs, but failed on ordinary VoxTell. | Small, credible synergy with explicit spatial text-image attention. |
| Attention | S1/S2/S3 progressively improved prompt-specific coarse localization, but attention precision stayed low and Dice barely moved. | Attention quality and segmentation quality are not interchangeable. |
| Multi-resolution | Oracle native-96 refinement strongly helped tiny lesions; automatic proposals/fusion reduced overall Dice. | Refinement is viable; deployable localization/selection is not. |
| Component selection | GT component oracles improved Dice by about +0.085 to +0.088; learned linear/CNN/ranking selectors recovered almost none of this gap. | Candidate selection is the largest demonstrated headroom and the largest unsolved problem. |
| Anatomy / lobe priors | Hard lobe filtering improved Dice but deleted true components; semantic whole/side/lobe post-processing safely improved a separate lineage by +0.0153. | Anatomy is useful as a conservative output constraint, not as a strong hard gate or unbounded query injection. |
| Anatomy representation | ACA P0 learned lobe/laterality structure; P1 additive coupling collapsed all foreground while ZERO reproduced baseline. | Representation learning succeeded; the coupling mechanism failed. |
| Image windows | A direct 2-channel HU-window input was statistically indistinguishable from a strict 1-channel control after 500 updates. | NO-GO for the direct-input design. |

## Most important takeaways

1. **The text pathway is active.** Qwen location-word edits produced large representation and logit changes, low overlap between correct- and swapped-prompt masks, and large centroid shifts. A laterality swap often moved the largest component to the opposite lung. The core problem is not simply frozen-text insensitivity.
2. **The model can refine a lesion when location is supplied.** Oracle-localized native-96 refinement improved tiny-lesion Dice from 0.4392 to 0.5713. The same high-resolution idea failed in full-volume or automatically proposed settings, which isolates localization and proposal precision as the key missing capabilities.
3. **There is large, measurable component-selection headroom.** GT-assisted component oracles improved full-validation Dice from roughly 0.274 to 0.359-0.363. Simple geometry/probability/attention classifiers, local CNNs, crop rankers, and utility selectors captured little of that gap.
4. **Attention became more anatomically meaningful without becoming sufficiently selective.** S1 and S2 distinguished correct from shuffled prompts but did not point into GT. S3 improved pointing to 28.3%, AUROC to 0.927, and peak distance to about 41.6 voxels, yet AUPRC remained only 0.0125 and segmentation Dice was almost unchanged. Broad support is useful for heatmaps but poor for component rejection.
5. **Anatomy is useful only when coupled conservatively.** Whole-lung, laterality, and semantic lobe/side post-processing produced the largest practical gain in one lineage with 99.1% GT retention. Hard lobe filtering deleted true components. ACA representation learning succeeded, but direct additive anatomy-query coupling collapsed all foreground.
6. **Most apparent improvements are modest or non-comparable.** The strongest learned full-validation result in the current record is the S3 prompt residual adapter (+0.006971 Dice with positive bootstrap CIs). The semantic post-processing gain (+0.015283) is larger but is deterministic inference filtering in a different checkpoint/evaluation lineage. Neither constitutes a large architecture breakthrough.

## Modest positive results that should not be lost

| Method | Observed gain | Evidence | Interpretation |
| --- | --- | --- | --- |
| Category-adaptive Tversky | +0.001452 Dice | Matched full-validation loss ablation | Small, reliable FP suppression. |
| Prompt cleanup / canonical_visual | +0.0061 to +0.0064 Dice | Full validation at threshold 0.5 | AP decreased; likely calibration/mask shrinkage rather than improved ranking. |
| Prompt residual adapter on S3 | +0.006971 Dice; CI above zero | Full validation | Best learned improvement in this record; still modest. |
| Semantic anatomy post-processing v1 | +0.015283 vs raw | Separate 381-finding lineage | Best practical inference gain, but it is deterministic post-processing. |
| Oracle native-96 | +0.1321 on 20 tiny findings | GT-localized | Strong proof that local high-resolution refinement can work. |
| Component oracle | +0.085 to +0.088 overall | GT component selection | Largest demonstrated headroom; not deployable. |

## Clear negative or non-scaling results

| Method | Result | Primary diagnosis |
| --- | --- | --- |
| Focal/Unified Focal/boundary losses | No Dice gain | Loss engineering does not solve localization or partial-label FP. |
| Category prefix | -0.001647 Dice | More metadata in text did not improve overlap. |
| CoLiPri semantic residual | +0.000024; rho 0.000775 | Model learned to turn the alternative representation off. |
| Seven-offset ensemble | -0.01387 Dice | Highly correlated offsets; averaging harmed small findings. |
| Automatic native-96 zoom | Large overall regression | Proposal precision and fusion were poor. |
| Zoom-out native-256 | -0.016687 direct inference | Larger FOV diluted spatial detail. |
| Learned component/crop selectors | Near-zero or negative Dice | Simple tabular/CNN/utility models did not model set-valued truth and FP cost. |
| Hard lobe filter | +0.01268 Dice but 87.6% true-component retention | Unsafe gain. |
| Scalar lobe channel / soft prior | Approximately zero | The network ignored or weakly used the side information. |
| ACA P1 direct query coupling | Dice approximately zero; 0/49 hits | Unbounded additive coupling caused foreground collapse. |
| Direct 2-channel HU windows | +0.002308 vs strict control; CI crosses zero | No evidence to scale the design. |

# 1. Task, data, evaluation, and comparability

## 1.1 Task definition

ReXGroundingCT is prompt-conditioned three-dimensional chest CT segmentation. The input is a CT volume and a free-form radiology finding such as "3 mm nodule in the left lower lobe." The output is a prompt-specific binary 3D mask. Unlike category-only segmentation, the system must resolve disease type, laterality, lobe, superior-inferior location, focality, multiplicity, and descriptive morphology from a sentence and then localize the corresponding abnormality in the scan.

The released dataset contains 3,142 non-contrast chest CTs, 8,028 finding-to-segmentation pairs, and 16,301 annotated entities across 14 lung and pleural categories. Approximately 79% of findings are focal and 21% are non-focal. The official dataset includes a 50-case public validation set and a 100-case private test set. This project additionally uses internal development splits, especially a 200-case / 381-finding full-volume validation set.

A critical annotation property is that training annotators were allowed to mark no more than three representative instances for a finding. Therefore, a prediction can be clinically plausible yet counted as false positive because an additional true instance is unlabeled. This is especially important for multifocal nodules, diffuse disease, bronchiectasis, emphysema, ground-glass opacity, and consolidation.

## 1.2 Working cohorts

| Name | Size | Main uses | Caution |
| --- | --- | --- | --- |
| Full validation | 200 cases / 381 findings | Most baseline, loss, attention, zoom, and semantic-postprocessing studies. | Use case-level clustered uncertainty because multiple findings can share a case. |
| fixed30 | 30 cases / 49 findings | Rapid causal and architecture pilots: prompt swaps, CoLiPri, multi-window, ACA P1, lobe encoding. | Fast but high-variance; absolute Dice is checkpoint-specific. |
| Fold-0 crop studies | 78 findings | P0-B2 proposal coverage, top-8 oracle, pre/post-inference utility selectors. | Do not compare directly with 381-finding full validation. |
| P0-F component cohort | 162 eligible findings / 112 cases | Same-anatomy component separability and ranking. | Large exclusion set; candidate coverage itself limits conclusions. |

## 1.3 Metrics

- **Dice per finding** is the main internal metric. Several reports also include Dice per case.
- **HIT** is the project overlap-detection metric reported by count and rate. Because historical scripts and paper reporting may use slightly different hit definitions or threshold conventions, this report preserves the recorded values rather than silently harmonizing them.
- **Precision, recall, predicted/GT volume ratio, FP volume, and mean FP voxels** are used to distinguish detection from over-segmentation.
- **Attention AUROC/AUPRC, pointing accuracy, peak-to-GT distance, enrichment, and correct-vs-shuffled correlation** characterize spatial text-image alignment.
- **Proposal/candidate recall, safe coverage, component recall, utility AUROC, Spearman correlation, and top-k selection** characterize multi-resolution and component-selection experiments.

## 1.4 Non-comparability warning

Absolute Dice values in this project range from about 0.20 to 0.35 because they come from different checkpoints, thresholds, crops, prompt variants, post-processing rules, and cohorts. A fixed30 baseline of 0.205, an anatomy-coupling ZERO control of 0.276, a semantic-postprocessing raw baseline of 0.303, and a seven-offset baseline of 0.348 are not contradictory; they are different experimental lineages. The only defensible causal statements are paired comparisons that share the same cohort, checkpoint initialization, inference path, threshold, and post-processing.

For any future paper or AI-generated proposal, the first step must be to define one canonical evaluation manifest and one checkpoint lineage. Results should be grouped by cohort and never ranked in a single leaderboard without a matched-control flag.

# 2. Baseline architecture and pipeline audit

## 2.1 VoxTell baseline

VoxTell was selected because it was the most directly aligned available model: free-form text-conditioned 3D medical image segmentation. The model uses a frozen Qwen3-Embedding-4B text encoder, with a 2048-dimensional prompt representation in the project implementation. Image features and prompt features are fused at multiple decoder stages, and segmentation outputs receive deep supervision at progressively finer scales.

The published system was pretrained on more than 62,000 volumes and 1,087 concepts with 9,682 rewritten labels. For ReXGroundingCT, the project reconstructed the missing fine-tuning pipeline. Direct inference from the provided checkpoint achieved Dice 0.229 and HIT 0.549; the reconstructed fine-tuning reached Dice 0.263 and HIT 0.646. The paper reported 0.282 and 0.678, so reproduction narrowed but did not eliminate the gap.

## 2.2 Default inference path

The audited inference sequence is:

```text
full CT -> RAS-oriented read -> crop_to_nonzero/body crop -> per-volume z-score
        -> pad if needed -> overlapping 192^3 sliding windows
        -> Gaussian-weighted logit accumulation -> threshold
        -> remove padding -> restore prediction to original CT shape
```

Default inference does **not** use the GT mask, finding bounding box, GT lesion center, lobe mask, anatomy crop, predicted ROI, or lesion-centered crop. Every sliding-window tile receives the same positive finding prompt and must decide both whether the finding is present in that tile and how to segment it.

## 2.3 Training crop path and the central mismatch

Training selects non-empty finding masks, samples foreground coordinates, and constructs 192^3 crops that contain selected GT foreground. With `--require-positive-crop`, the sampled positive prompt must have foreground in the crop. Training-time validation calls the same crop sampler, so it evaluates local refinement in an oracle-like region, not full-volume search.

This creates a severe mismatch:

```text
training:  prompt + GT-assisted lesion-containing crop -> refine/segment
inference: prompt + every body tile                   -> search + reject + segment
```

The mismatch explains several recurring observations: good patch validation with weaker full-volume Dice; large background FP accumulation; especially poor tiny-lesion search; and diffuse-pattern expansion when the prompt is applied repeatedly across the lungs.

## 2.4 Sampler audit

The current `anchor85_hardneg70` sampler is an infinite with-replacement stream rather than an epoch-based traversal. Cases are sampled uniformly from 1,936 eligible anchor cases. Two distinct same-case findings are chosen; the crop is nominally 85% anchor-positive and 15% same-case spatial-empty. A third prompt comes from another case, and 70% of these cross-case prompts are category- or keyword-matched hard negatives. Prompt order is shuffled.

Important consequences:

- Uniform case sampling is not uniform finding sampling. Findings in cases with many findings receive much less exposure.
- The 1,056 single-finding cases are excluded as anchors, potentially changing the training distribution.
- In a 20,000-patch run, the expected same-case exposures were estimated at 10.33 for a finding in a two-finding case but only 1.59 for a finding in a 13-finding case; about 126 eligible findings were expected to receive zero same-case selection.
- DDP ranks sample independently with seeds `2027 + rank`; coincidental duplicates remain possible, and there is no DistributedSampler.
- "Two positive prompts" means two same-case finding identities were selected, not that both masks are non-empty in every crop. In one audited run, rank 0 saw 4,450 patches with two non-empty targets and 5,010 with only one non-empty target.
- There is no explicit category balancing, which leaves rare categories rare.

## 2.5 Current loss and optimization

For a non-empty prompt at each deep-supervision scale:

```text
L_seg = mean voxel BCE + (1 - category-adaptive Tversky)
```

For an empty target:

```text
L_seg_empty = 0.5 * mean voxel BCE
```

Deep-supervision weights are `[8/15, 4/15, 2/15, 1/15, 0]`. FP-heavy categories 1b, 1c, 1e, 2b, 2c, and 2d use `alpha_FP=0.7, beta_FN=0.3`; other categories use 0.5/0.5. Current runs do not activate ordinary Dice, focal loss, boundary loss, presence loss, voxel weighting, or prompt-group weighting, even though some historical arguments remain in configuration files.

The validated Dual-S3 continuation used SGD with Nesterov momentum 0.99, weight decay 3e-5, BF16, 2 x H100 80 GB, one crop/GPU, gradient accumulation 8, and effective batch 16. The optimizer and schedule were restored when resuming; actual learning rates were much lower than nominal maxima. This is important when interpreting later continuation experiments.

## 2.6 Preprocessing and HU audit

Training cache volumes are stored in float16 and loaded as float32 after a nonzero crop and per-volume z-score normalization. Recomputing the z-score from cached inputs matched the training NPZ exactly. Raw NIfTI volumes were int16 with slope/intercept 1/0. The audited range was -8192 to 12096 HU, with selected quantiles around -1044, -1024, -1013, -834, 92, 440, and 1091 HU.

No common physical-spacing resampling was found in the current preprocessing path. This creates a potential hidden source of inconsistency: a fixed 192-voxel crop represents different physical fields of view across scans. Physical-mm dilation was therefore required in all lobe-mask safety analyses.

# 3. Baseline error anatomy

## 3.1 Dominant errors

Across early fine-tuning and later S3 analyses, the dominant error is over-segmentation or background false positive volume. Category confusion exists, but it is not the main explanation for most failures. The model often identifies visually related abnormal tissue while activating too broadly or on detached structures elsewhere in the lungs.

Small and tiny findings are hardest because the training crop already reveals the lesion location, whereas full-volume inference must search. Diffuse and subtle categories are also difficult because the target can be widespread, the annotations may be representative rather than exhaustive, and similar texture can occur in multiple unlabeled regions.

## 3.2 Category patterns

The better-performing categories in the internal presentation were pleural effusion (2e), consolidation/atelectasis (2b), ground-glass opacity (2c), and pneumothorax/pleural air (2g), although 2g recall was low. Weaker categories included miscellaneous subtle airway/vascular findings (1f), bronchiectasis (1b), emphysema (1c), bronchial wall thickening (1a), and other focal lung/airway/pleural findings (2h).

Observed confusions included 1b -> 1a/1d, 1d -> 2c, 2b -> 2c, 2d -> 2c, and 2h -> 1c. These patterns matter for hard-negative sampling, but the broader finding remained that detached and connected FP, rather than wrong category alone, drive Dice loss.

## 3.3 Partial-label ambiguity

Because the training masks can contain at most three representative instances, ordinary BCE/Tversky treats some plausible unannotated lesions as background. This can penalize the model for learning clinically reasonable multifocal patterns and can also corrupt a component selector trained from GT-overlap labels. Any future selector or weak-label loss must distinguish "definite background" from "unlabeled but plausible same-finding tissue."

# 4. Loss, sampler, and optimization experiments

## 4.1 Loss experiment table

| Strategy | Dice/finding | HIT | Main effect |
| --- | --- | --- | --- |
| Weighted-BCE baseline | 0.2657 | 0.6457 | Reference |
| Strong background BCE, w_background=1.5 | 0.2668 | 0.6588 | Modest FP suppression / more hits |
| Weighted BCE + Anchor85 | 0.2671 | 0.6509 | Improved sampler baseline |
| Focal Tversky | 0.2658 | 0.6483 | FN-oriented weighting did not help |
| Asymmetric Unified Focal | 0.2661 | 0.6483 | No gain |
| Asymmetric Unified Focal, LR x4 | 0.2647 | 0.6614 | More hits but worse overlap |
| Unified Focal + 0.05 SDF boundary | 0.2662 | 0.6457 | Boundary term did not solve localization/FP |
| Category-adaptive Tversky | 0.2689 | 0.6562 | Small consistent precision/FP improvement |

## 4.2 Interpretation

The loss experiments rule out a simple answer based on adding more sophisticated overlap or boundary terms. Boundary and focal formulations operate on the mask once the correct region has been found; they do not teach full-volume search, patch rejection, or set-valued component selection. Category-adaptive Tversky helped because it matched the observed FP-heavy error pattern, but the effect size was only +0.00145 Dice.

The sampler experiments also show that "hard negative" is not a single concept. The old hard-negative-triplet sampler used mismatched abnormal patches. The later `anchor85_hardneg70` sampler uses a same-case crop-empty branch plus a cross-case prompt that is often category matched. The most task-faithful negative remains a correct patient-level prompt applied to a local tile where the finding is absent. That negative is present only 15% of the time and is not balanced by category, lesion size, or anatomical region.

## 4.3 Presence heads

Presence V1 added a global fused-query scalar classifier. It was intended to answer "is this prompt present in this patch?" and used approximately `L_total = L_seg + 0.1 L_presence` with `pos_weight` around 2. Inference-time gating reduced performance. The training result was also confounded by continued training: Anchor85 initially beat the hard-triplet sampler, but after presence-head continuation the hard-triplet lineage appeared better while Anchor85 regressed.

Presence V2 replaced the opaque scalar with a spatial evidence map followed by top-k pooling. This was a more interpretable compatibility head but still not a segmentation mask. No complete matched full-volume gain was recorded. Both versions were superseded by explicit multiscale S3 attention.

# 5. Text-side processing and prompt adaptation

## 5.1 Category prefix

A controlled 200-optimizer-update experiment prepended metadata using:

```text
Category: {category_code}, {category_name}.
Finding: {original_finding_text}
```

Original prompts achieved Dice 0.268182 and 250 hits. Category-prefix prompts achieved Dice 0.266535 and 252 hits. The extra category information slightly increased detection while reducing overlap. It did not solve spatial ambiguity and was not retained as a standalone improvement.

## 5.2 Prompt cleanup and canonical_visual

The `canonical_visual` pipeline is primarily deterministic removal of temporal, comparison, speculation, recommendation, and other non-visual spans, followed by whitespace/punctuation cleanup. It is not a full semantic parser and does not create a structured finding graph.

At threshold 0.5 on 381 findings: original Dice was 0.267003, `cleanup_only` was 0.273079, and `canonical_visual` was 0.273445. HIT was unchanged at 0.650919. AP decreased from 0.803206 to 0.796268/0.798727. Therefore, almost all of the Dice gain came from cleanup rather than semantic span removal, and the lower AP argues against improved voxel ranking. The most plausible explanation is a global logit/calibration or mask-shrinkage effect: precision increased and recall decreased.

A rigorous follow-up was designed but not documented as completed. It would save paired logits or deterministic voxel samples, evaluate thresholds 0.05-0.95, match predicted volume/precision/recall, cross-fit calibration, and ablate each removed phrase category. Until that audit is complete, canonicalization should not be described as better language understanding.

## 5.3 Causal location swaps

The minimal location-swap audit is one of the most important mechanistic results. Swapping only laterality or upper/lower words changed the token sequence, pooled embedding, final logits, mask shape, and centroid substantially. Correct and swapped masks had Dice only 0.0636, and the mean centroid displacement was 211.6 voxels. This shows that the frozen Qwen and downstream fusion preserve location information strongly enough to alter segmentation.

The directional audit then asked whether the change followed the requested anatomy rather than merely destabilizing the output. Nine of 16 edits transferred toward the new target, four suppressed the original target, two produced off-target changes, and one barely changed. Laterality transfer was much more reliable than upper/lower transfer. The correct conclusion is: the model uses spatial words, but its spatial control is coarse and sometimes unsafe.

This evidence led to a NO-GO for Qwen LoRA or a token-level Prompt Composer motivated by the belief that location words were being ignored. It does not rule out all text adaptation; it changes the goal from "make text matter" to "make existing text sensitivity more selective and stable."

## 5.4 Prompt residual adapter

A frozen rank-16 residual adapter was applied to the prompt representation. On ordinary VoxTell, it increased hits from 248 to 261 but reduced Dice by 0.001357, implying more detection without better spatial overlap. On S3, the same concept produced a small but statistically reliable improvement: 0.274406 to 0.281376, with hits 261 to 266. The finding-level bootstrap CI was [+0.001382, +0.012474], and the case-clustered CI was [+0.003740, +0.015595].

The interaction is informative. S3 creates an explicit spatial text-image matching pathway; the adapter can reshape the prompt in a way that improves those spatial gates. In ordinary VoxTell, adapting a global prompt query seems to change sensitivity without controlling spatial precision. The S3 adapter should be treated as a valid small positive result and replicated under the final canonical evaluation, not as a solved model.

## 5.5 CoLiPri alternative clinical text representation

Semantic Residual Hybrid A tested whether a clinical text representation could improve the frozen Qwen query without replacing it:

```text
q_final = q_Qwen + rho * (q_CoLiPri_adapted - q_Qwen)
rho = 0.5 * tanh(raw_rho)
```

The CoLiPri encoder and VoxTell were frozen; only the adapter and scalar gate were trained. The identity condition `rho=0` reproduced Anchor85 exactly. After 400 steps on one H200 (about 97 minutes), fixed30 Dice changed from 0.205104 to 0.205128, HIT was unchanged at 0.693878, mean FP increased from 155,394 to 155,674, and the Dice delta CI included zero. Final `rho=0.000775`, so the model learned to disable the residual. This is a clear NO-GO for the frozen semantic-residual design.

# 6. Attention mechanisms and what the maps revealed

## 6.1 Attention lineage

| Generation | Mechanism | Supervision | Observed behavior | Status |
| --- | --- | --- | --- | --- |
| Presence V1 | Global fused query -> MLP scalar | Patch/prompt presence | No robust gain; inference gating hurt. | Obsolete. |
| Presence V2 | Spatial evidence map + top-k pooling | Coarse prompt compatibility | No validated full-volume gain. | Superseded. |
| S1 | 1/4 skip gate | Residual suppressive spatial gate | Correct > shuffled, but 0% pointing and 344-voxel median peak distance. | Prompt-specific, not local. |
| S2 | 1/4 + 1/2 gates | Category-routed 2b/2c | AUROC 0.787, AUPRC 0.000365, 0% pointing; Dice 0.268876. | Broad low-precision support. |
| S3 | 1/4 + 1/2 + 1/1 | Projected cosine voxel-text attention + direct attention loss | AUROC 0.927, AUPRC 0.0125, pointing 28.3%; Dice unchanged around 0.2686. | Mechanistic improvement, little task gain. |
| Dual-S3 | 192 context + 96 query view | Shared weights + view consistency | Validated 0.274238 / 261 hits; automatic zoom still failed. | Strong baseline, not a solved multi-FOV system. |

## 6.2 S3 architecture

S3 applies voxel-text attention to encoder skip features at 1/4, 1/2, and 1/1 resolution. At each scale, skip features and decoder context are projected to 128 dimensions; the 2048-dimensional text embedding is projected to 128; image and text vectors are L2-normalized; cosine similarity is multiplied by a learned exponential logit scale, clamped to 50, plus a learned bias; sigmoid converts the score to an attention map.

The skip gate is:

```text
gated_skip = skip * ((1 - rho) + rho * attention)
```

The validated configuration used base `rho=0.25`, category multiplier 0.25 for 1a-1d (effective 0.0625), multiplier 1.0 for 2b/2c (effective 0.25), and 0 for other categories. Final learned exp(logit-scale) values were about 10.01 at all scales, with biases close to zero.

For eligible positive 2b/2c prompts, direct attention supervision was:

```text
L_direct = BCE(attention, pooled/dilated target) + 1 - softDice(attention, target)
L_hardBG = ReLU(0.05 - mean_fg_attention + mean_top1pct_bg_attention)
L_view = L_seg + 0.05 * L_direct + 0.02 * L_hardBG
```

Several nearby ideas were disabled in the validated run: presence V1/V2, old hard-negative-triplet loss, shuffled/wrong-prompt contrastive loss, native ranking, coverage/TV loss, full-resolution sampled alignment, and prompt-variant consistency. This matters when interpreting requests to add "attention supervision" that may already refer to a disabled prototype rather than the current model.

## 6.3 Pattern discovered in attention

The progression from S1 to S3 shows a consistent pattern:

- Correct prompts produce stronger lesion-region enrichment than shuffled prompts.
- High voxel-level AUROC can coexist with extremely low AUPRC because lesion voxels are sparse and attention support is broad.
- The highest-attention voxel can remain outside GT even when average attention is enriched near the lesion.
- Attention maps can be prompt-specific yet too diffuse to decide which connected component is true.
- Aggregate attention features did not improve a component selector and reduced the reported rescoring metric by about 0.0046.
- Better attention maps did not directly improve segmentation Dice, which implies that attention is not the only or final bottleneck; decoder calibration, thresholding, partial labels, and output component accumulation remain important.

The correct research interpretation is that S3 learned a coarse spatial compatibility field, not a reliable lesion detector. It should be used as one signal in a better structured selector or bounded spatial gate, not treated as a faithful explanation map.

## 6.4 Dual-S3

Dual-S3 uses a native 192^3 context crop and, for eligible category-2 positive prompts, a centered native 96^3 crop upsampled to 192^3. Both views use the same network and all weights are shared. Features are not concatenated. The views interact only through weighted segmentation/attention losses and a context-zoom consistency term. Full-volume evaluation at update 2000 used only the context branch and achieved Dice 0.274238 with 261 hits.

The dual-view training did not itself solve automatic zoom. Its value was mainly to prove that the shared model can exploit a high-resolution crop when the crop is correct.

## 6.5 Seven-offset ensemble and Grad-CAM discussion

Seven spatial offsets were ensembled to test whether sliding-window phase instability could be averaged away. Predictions were highly correlated, yet ensemble Dice fell by 0.01387, with the largest harm in small findings. The errors were not independent; averaging softened small positive regions and did not eliminate systematic FP.

Grad-CAM was discussed as a post-hoc alternative. No completed Grad-CAM experiment is recorded. Conceptually, Grad-CAM would provide class-gradient saliency from a chosen layer, but it would be lower-resolution, threshold- and target-dependent, and not guaranteed to be a better spatial gate than the trained S3 map. It remains a visualization diagnostic, not an established replacement.

# 7. Multi-resolution, zoom, proposal, and crop-selection experiments

## 7.1 Oracle high-resolution result

The oracle native-96 result cleanly separates refinement from search. For 20 tiny findings, a correctly localized native-96 crop improved Dice from 0.4392 to 0.5713, with a positive bootstrap CI. This is one of the strongest mechanistic results in the project: the shared network has enough capacity to segment tiny lesions better at higher native resolution when the region is known.

Dense full-volume native-96 inference moved in the opposite direction, from 0.2081 to 0.1711 for tiny findings. More high-resolution tiles meant more opportunities for FP, and the prompt was repeatedly applied to local anatomy that resembled the target. The bottleneck is therefore not only image resolution; it is deciding where to spend that resolution and when to abstain.

## 7.2 Automatic zoom proposals

Unbounded proposals caused severe FP inflation. Bounding the system to top-3 proposals with threshold 0.2, peak ranking, 96-voxel NMS, and replacement/max fusion improved a tiny subgroup but reduced overall Dice dramatically. Top-3 ROI recall was 94.2% overall and 100% for large findings, yet fusion still failed. This demonstrates that proposal recall alone is an inadequate objective. A proposal system must optimize the expected *incremental utility after local segmentation*, including the FP cost of every extra crop.

## 7.3 Component oracle

At low threshold, the baseline produces many connected components. A GT oracle that keeps every component with any GT overlap and removes detached zero-overlap components raises Dice by about +0.085 to +0.088. Small findings benefit most. The oracle does not choose a single component; it keeps a variable number, which is important because a prompt can describe multifocal disease.

This oracle cannot remove false-positive voxels connected to a true component, so its gain is a lower bound on the broader benefit of better shape refinement. It is nevertheless the clearest evidence that detached FP rejection is a central problem.

## 7.4 Crop coverage ranker P0-B2-lite

The learned ranker improved top-1 safe crop coverage substantially, from 41.03% to 56.41%, but top-3 coverage improved only from 61.54% to 66.67%. Mean top-3 union coverage and GT-component recall improved by about one percentage point. This means the ranker learned some ordering signal but did not generate enough new coverage. The candidate set and training target remained limiting, especially for tiny findings.

A top-8 choose-0-to-3 oracle reached 0.278672 and 57 hits compared with a P0 router at 0.264542 and 55 hits. The modest oracle gap shows that even perfect routing among those candidates could not transform performance. Thirteen findings needed heuristic ranks 4-8, and tiny findings had no oracle rescue, again implicating candidate generation.

The pre-inference utility selector over-selected crops and introduced FP; the post-inference selector had reasonable AUROC but abstained on almost everything. This is the characteristic precision-recall tension of crop selection: aggressive selection adds FP, while conservative selection misses the few useful refinements.

## 7.5 Zoom-out

Mixed native-192/native-256 training was effectively neutral under native-192 inference (+0.000048). Direct native-256 inference reduced Dice by 0.016687 and lost 19 net hits. A two-stage native-256 context followed by native-192 local refinement still failed globally, although replacement helped tiny findings and max fusion helped large findings. An oracle size-based router reached about 0.2845, but that routing rule was not deployable and may exploit the GT-size stratification itself.

Together, zoom-in and zoom-out experiments support a mixture-of-field-of-view hypothesis but not a working implementation: tiny findings want localized high resolution, some large findings can benefit from broader context, and a single fixed fusion policy harms other groups.

# 8. Learned component and crop selection

## 8.1 Why binary component classification is structurally difficult

A ReX finding can have zero, one, or many true components. The correct number depends on category, prompt wording, and partial annotation. A component-level binary classifier treats candidates independently, but the deployment decision is set-valued: selecting one component changes the marginal value and FP cost of selecting another. Diffuse findings need a different policy from a solitary nodule.

Basic tabular features, dual-view local CNNs, and aggregate attention statistics did not solve this. The P0-C tabular model gained only +0.000475 Dice; the dual-view CNN lost 0.006323; AUPRC was 0.0722. Attention features made the reported linear rescoring metric about 0.0046 worse. Broad attention support and severe class imbalance explain part of the failure.

## 8.2 P0-F same-anatomy ranking

P0-F restricted the negative set to components in the same anatomical region, which is a harder and more realistic problem than rejecting contralateral or outside-lung components. The experiment extracted 2,167 frozen candidate features: 391 GT-overlapping and 1,776 same-anatomy FP components. It used case-separated train/validation/test folds.

The result was Decision D - NO-GO. A top-16 cap could not retain at least 95% of positive components on train and test. The candidate recall ceiling was also limited: mean GT voxel coverage on held-out findings was only 0.5667, although every eligible test finding had at least one overlapping candidate. This means ranking cannot solve missing coverage, and the frozen features were not sufficiently separable within the same anatomy.

The reporting pipeline encountered a stale-DataFrame error after GPU feature extraction, but the final report was recovered through CPU jobs 5314568, 5314587, and 5314622. That operational detail matters because the final decision should be tied to the recovered report, not the intermediate failed notebook state.

# 9. Anatomy and lobe-mask directions

## 9.1 Hard and soft lobe filtering

A conservative parser recognized explicit RUL/RML/RLL/LUL/LLL phrases, lingula, bilateral upper/lower lobes, and unions of multiple explicit lobes. It deliberately did not hard-filter laterality-only, apical, basal, hilar, central, peripheral, pleural, or vague location language. TotalSegmentator provided five lobe masks, and physical dilation was applied in millimeters.

The hard 10-mm filter improved the reported Dice by 0.01268 and removed 20.5% of detached FP components, but retained only 87.6% of true components and reduced HIT. This is not safe enough. The soft prior was essentially neutral and had only 11.73% useful-component precision.

The result is important even though it was a NO-GO: exact lobe information is predictive, but segmentation/annotation boundaries and parser uncertainty make complete deletion brittle. Anatomy should modify probability or late selection conservatively, with an identity-preserving fallback.

## 9.2 Scalar lobe input

Adding a scalar lobe-label channel changed Dice by only +0.000125 in an inference diagnostic and remained neutral after 500 updates. The model could ignore the extra channel, and no explicit objective forced it to align text, lobe label, and prediction. This ruled out the simplest "append a lobe map as another input" approach.

## 9.3 Semantic-aware inference v1

The semantic-aware constraint is the strongest practical anatomy result. It uses a rule-based parser to assign findings to groups such as exact-lobe, ambiguous-lobe, laterality-only, bilateral/multifocal, and nonlocalizable. High-confidence exact-lobe prompts use an exact-lobe mask dilated 15 mm; side-level prompts use the corresponding lung side dilated 15 mm; broad or nonlocalizable findings fall back to a whole-lung mask dilated 10 mm.

On 381 findings, raw Dice 0.303199 increased to 0.313767 with whole-lung10, 0.317296 with previous laterality10, and 0.318482 with semantic v1. The semantic filter gained seven hits relative to raw and three relative to whole-lung10, while retaining 99.0561% of GT and 90.5765% of prediction volume. Compared with whole-lung10, 58 findings improved, 320 tied, and only 3 worsened.

This should be reported as deterministic inference post-processing, not learned grounding. It proves that prompt-derived anatomy can safely remove some FP when uncertainty is handled hierarchically.

## 9.4 Output-space alignment versus representation-space alignment

The lobe-mask encoding/output-alignment experiments used containment-style losses or gates on the segmentation output. The streamlined fixed30 program ended in NO-GO; the recovered baseline was raw 0.276443 and whole10 0.284751, but the full per-checkpoint table was not available in the current synthesis.

This is conceptually different from ACA. Output-space alignment says "prediction should stay inside this anatomy." ACA attempts to learn anatomy-level visual representations and match them to text. Combining these without distinguishing them can lead to incorrect conclusions about what failed.

## 9.5 ACA anatomy representation and P1 collapse

The ACA-inspired adaptation pooled frozen visual features inside TotalSegmentator anatomy masks to form lobe/global tokens, added anatomy identity/context, and trained anatomy-text contrastive and within-scan routing objectives. P0 was successful: explicit-lobe accuracy reached 70.4%, laterality 85.7%, correct matching beat lobe-permuted controls, and frozen VoxTell weights were unchanged.

P1 directly coupled the learned anatomy context into the segmentation query and used causal controls: CORRECT anatomy, LEFT_RIGHT_SWAP, LOBE_PERMUTED, and ZERO coupling. ZERO reproduced the baseline at Dice 0.27637 with 33/49 hits. CORRECT, swap, and permuted conditions all produced approximately zero Dice and 0/49 hits. Because every nonzero anatomy condition collapsed, the failure was not incorrect anatomy semantics; it was the coupling itself, most likely an additive distribution shift or scale that suppressed all foreground.

The correct conclusion is not that anatomy representation failed. P0 succeeded. The safe next design would need bounded, identity-initialized, late spatial or logit coupling with a causal scale sweep. The completed direct-additive P1 is a clear NO-GO.

## 9.6 Mentor-requested spatial-modifier FP audit

The requested metric has not yet been computed in the exact required form. Existing analyses answer related but different questions: about 5.3% median FP outside the whole lung, and about 82.1% of FP within the GT-positive lobes. Neither measures FP outside the **prompt-specified** location.

The required analysis should restrict to findings with a parseable spatial modifier and report:

- number and percentage of FP voxels outside the specified target lobe or side;
- number and percentage of detached FP components outside the target;
- breakdown into ipsilateral non-target lobe, contralateral lung, and outside lung;
- finding-level mean, median, distribution, and count of findings with any off-location foreground;
- examples such as a right-lower-lobe basal finding producing foreground in other lobes.

This should be kept as a primary unresolved analysis rather than implied from the lobe-filter results.

# 10. Multi-window and intensity-channel experiments

The HU audit confirmed that the cached training input is per-volume z-score normalized and that raw HU can be reconstructed from the original NIfTI. A direct-input pilot compared a strict one-channel control with a two-channel window representation for 500 updates on 2 x H200.

The original checkpoint reference was 0.296870 Dice, HIT 0.734694, and mean FP 19,502.8. Training itself regressed the strict one-channel control to 0.262326, HIT 0.673469, and mean FP 27,817.6. The two-channel model reached 0.264634, HIT 0.693878, and mean FP 27,903.6. The correct causal comparison is therefore two-channel versus strict one-channel: +0.002308 Dice with a CI spanning -0.012476 to +0.014576, HIT +0.020408, and essentially unchanged FP.

The decision was NO-GO for continuing this direct-input design to 1,000 updates, three channels, or 200-case validation. The experiment did not show that CT windows are generally useless; it showed that simple channel expansion under this training setup did not overcome optimization regression or improve overlap reliably.

# 11. Training-scale, checkpoint, and workflow audits

The training scripts count microsteps, optimizer updates, GPU forward/backward calls, patch exposures, and prompt-mask pairs differently. There is no natural epoch because sampling is infinite with replacement. For accumulation 8 on two GPUs, 10,000 microsteps correspond to 1,250 optimizer updates and 20,000 patch exposures, not 10,000 optimizer steps.

An audited all-category / full-resolution lineage was cancelled at microstep 9,460 after 1,182 optimizer updates. The latest durable scheduled checkpoint was update 1,000 at microstep 8,000. Because no final checkpoint/evaluation was produced, this run cannot support a performance conclusion. Similarly, Dual-S3 update-2500 existed as a raw checkpoint after an unexpected EOF but had no validated export or full-volume evaluation; update-2000 remains the latest completed artifact in that lineage.

Future experiment reports should always state: microsteps, optimizer updates, number of GPUs, per-GPU batch, accumulation, effective global patches, prompt-mask pairs, exact resumed learning rates, checkpoint selection rule, and whether the full-volume evaluation was completed.

# 12. Consolidated result ledger by cohort

## 12.1 Full validation: 200 cases / 381 findings

| ID | Experiment | Matched baseline | Result | Decision |
| --- | --- | --- | --- | --- |
| L01 | Strong-background weighted BCE | weighted-BCE baseline 0.2657 / HIT 0.6457 | Dice 0.2668; HIT 0.6588 | small sensitivity/background-rejection trade-off; not a breakthrough |
| L02 | Anchor85 sampler baseline | weighted-BCE control | step 3000 Dice 0.266750; step 3800 Dice 0.267057; HIT 0.648294 / 0.650919 | retained as a core baseline lineage |
| L03 | Focal Tversky | Anchor85 approximately 0.2671 | Dice 0.265799; HIT approximately 0.6483 | NO-GO as a general loss replacement |
| L04 | Asymmetric Unified Focal and boundary variants | Anchor85 approximately 0.2671 | Unified Focal 0.2661 / HIT 0.6483; LR x4 0.2647 / HIT 0.6614; +0.05 SDF boundary 0.2662 / HIT 0.6457 | NO-GO |
| L05 | Category-adaptive Tversky | Dice+BCE control 0.267446; Dice/case 0.278167; HIT 0.656168 | Dice/finding 0.268898; Dice/case 0.279923; HIT 0.656168; precision 0.339116; recall 0.421294; median FP 2737 mm3 | small but consistent GO as the default loss; not a step change |
| T01 | Category-prefix prompt | original prompt 0.268182 / HIT 0.656168 | category prefix 0.266535 / HIT 0.661417 | NO-GO as a standalone prompt format |
| T02 | Prompt cleanup and canonical_visual | original Dice 0.267003; HIT 0.650919; AP 0.803206 | cleanup_only Dice 0.273079, AP 0.796268; canonical_visual Dice 0.273445, AP 0.798727; HIT unchanged | useful formatting/calibration clue, but not evidence of better ranking or semantic understanding |
| T06 | Frozen rank-16 Prompt Residual Adapter on S3 | S3 0.274406; 261 hits | 0.281376; 266 hits; finding bootstrap CI [+0.001382, +0.012474]; case-clustered CI [+0.003740, +0.015595] | small validated GO; strongest learned improvement in this record, but still modest |
| A04 | S2 two-scale, category-routed attention | contemporary control | Dice 0.268876; 254 hits. Attention AUROC 0.787; AUPRC 0.000365; pointing 0%; peak distance about 247 voxels; correct/shuffled corr 0.883. | NO-GO; superseded by S3 |
| A05 | S3 multiscale voxel-text attention, early controlled run | HardNeg70 Dice 0.268496 | step 400 Dice 0.268616; step 800 Dice 0.268602. Correct enrichment 3.23 vs shuffled 1.68; pointing 28.3%; AUROC 0.927; AUPRC 0.0125; peak-to-GT about 41.6 voxels. | mechanistic GO, performance NO-GO |
| A06 | Dual-S3 context/query multi-FOV | S3 lineage | checkpoint_update_2000: Dice/finding 0.274238; Dice/case 0.286391; 261/381 hits; HIT 0.685039 | retained as a strong attention baseline; automatic zoom remained negative |
| A07 | Seven-offset ensemble | single-offset baseline Dice 0.34751 | ensemble Dice 0.33364; improved/same/worse 138/60/183; pairwise corr 0.8317; soft Dice 0.8598 | NO-GO |
| Z03 | Unbounded zoom proposals and fusion | native-192 baseline | replacement Dice delta -0.0520; max fusion delta -0.0334; tiny replacement +0.0242 | NO-GO |
| Z04 | Bounded top-3 native-96 proposals | native-192 0.2742 / 261 hits | replacement 0.2222 / 244; max 0.2408 / 244. Tiny replacement 0.2324 / 15 vs 0.2081 / 12. | NO-GO as global fusion |
| Z05 | Low-threshold component oracle | native-192 approximately 0.2742 | oracle 0.3627; about 36 candidates/finding and 4.7 retained | major non-deployable headroom; selector is the bottleneck |
| Z08 | Zoom-out mixed training and direct native-256 inference | matched native-192 control 0.274528 | zoom-out mix at native-192 0.274576; direct native-256 0.257889 | NO-GO as a default scale |
| Z09 | Native-256 Stage-1 plus native-192 Stage-2 fusion | native-192 reference 0.2746 / 261 hits | Stage-1 native-256 0.2579 / 242; replacement 0.2730 / 252; max 0.2696 / 248; safe-OR 0.2683 / 246 | NO-GO globally; subgroup routing signal only |
| C01 | Threshold-0.4 component oracle | baseline approximately 0.2740 | oracle approximately 0.3590; 77.7% improved; 8.9% worsened; mean 29.1 candidates, 4.3 GT-overlap retained, 24.9 detached removed | strongest evidence of component-selection headroom |
| N01 | Hard prompt-to-lobe component filter | reported baseline 0.30184 | 0.31452 at 10-mm dilation; 20.5% detached FP removed; 87.6% true-component retention; HIT decreased by 0.87 percentage points | NO-GO as a hard deployable filter |
| N02 | Soft lobe prior | nonlinear base 0.302003 | soft-lobe 0.302004; useful-component recall 93.75%; precision 11.73% | NO-GO |
| N03 | Scalar lobe-label input diagnostic and training | zero-map 0.30185191; 274 hits | correct map 0.30197673; 274 hits. Explicit-lobe subset: 0.33076476 to 0.33095381. After 100/250/500 updates: approximately 0.303199 to 0.303212. | NO-GO |
| N04 | Semantic-aware inference constraint v1 | raw 0.303199; 274 hits | whole-lung10 0.313767 / 278; previous laterality10 0.317296 / 280; semantic exact15-side15-whole10 0.318482 / 281 | practical post-processing GO; not an end-to-end model gain |

## 12.2 fixed30: 30 cases / 49 findings

| ID | Experiment | Matched baseline | Result | Decision |
| --- | --- | --- | --- | --- |
| T03 | Minimal location-swap causal audit | correct prompt | changed-token rel-L2 0.4440; pooled rel-L2 0.3229; logit corr 0.5837; mean abs logit delta 8.9582; correct-swap Dice difference +0.1969; mask Dice 0.0636; centroid shift 211.6 voxels | NO-GO for the hypothesis that Qwen/text is ignored; do not prioritize Qwen LoRA or token-level Prompt Composer for this reason |
| T04 | Directional transfer audit (P0b-lite) | original location | 9 transfer; 4 suppression; 2 off-target; 1 little change. Laterality: 7/12 transfer and 8/12 largest component contralateral. Upper/lower: 2/4 transfer, 2/4 off-target. | mechanistic GO: text pathway contains directional control; deployment still unreliable |
| T07 | CoLiPri Semantic Residual Hybrid A | Anchor85 Dice 0.205104; HIT 0.693878; mean FP 155394 | Dice 0.205128; HIT 0.693878; mean FP 155674; final rho 0.000775 | NO-GO; learned gate shut the residual off |
| N05 | Output-space lobe-alignment / lobe-mask encoding streamlined runs | baseline raw 0.276443; whole10 0.284751 | No checkpoint clearly exceeded the baseline after raw/whole10/semantic evaluation | final NO-GO |
| N06 | ACA-style anatomy representation P0 | random/permuted anatomy matching | explicit-lobe accuracy 70.4%; laterality accuracy 85.7%; correct anatomy matching better than permuted; frozen VoxTell weights unchanged | P0 GO; proceed only with safe bounded coupling |
| N07 | ACA P1 direct anatomy-query coupling | ZERO control Dice 0.27637; 33/49 hits | CORRECT, LEFT_RIGHT_SWAP, and LOBE_PERMUTED all Dice approximately 0 and 0/49 hits | NO-GO |
| I01 | HU audit and direct multi-window input pilot | strict 1-channel control 0.262326 / HIT 0.673469 / mean FP 27817.6 | 2-channel 0.264634 / HIT 0.693878 / mean FP 27903.6 | NO-GO; do not continue to 1000 updates, 3 channels, or 200-case validation under this direct-input design |

## 12.3 Fold-0 and component-specific cohorts

| ID | Experiment | Cohort | Result | Decision |
| --- | --- | --- | --- | --- |
| Z06 | P0-B2-lite fixed-96 crop-coverage ranker | fold 0; 78 findings | Top-1 safe coverage >=0.90: 41.03% to 56.41%; top-3: 61.54% to 66.67%; mean top-3 union 52.00% to 53.06%; GT-component recall 42.24% to 43.62% | useful diagnostic, insufficient for deployment |
| Z07 | Top-8 choose-0-to-3 crop oracle and learned selectors | fold 0; 78 findings | top-8 oracle 0.278672 / 57 hits; pre-inference selector 0.259313 / 55; post-inference selector 0.260111 / 55 | NO-GO for current utility selectors |
| C02 | Basic tabular component selector and dual-view CNN | component diagnostic | P0-C tabular Dice delta +0.000475; dual-view CNN delta -0.006323; AUPRC 0.0722 | NO-GO |
| C03 | Attention-feature logistic rescoring | component diagnostic | with aggregate attention features 0.3463 | NO-GO; simple attention statistics were not useful selector features |
| C04 | P0-F-lite same-anatomy component separability/ranking | 162 eligible findings / 112 cases; test fold 0 has 29 findings and 378 candidates | 2,167 frozen candidate features; 391 GT-overlap and 1,776 same-anatomy FP components. Mean held-out GT voxel coverage upper bound 0.5667; every test finding had at least one overlapping candidate. | Decision D - NO-GO |

# 13. Cross-experiment synthesis: what is probably wrong

## 13.1 The model is trained more as a local refiner than a global grounder

GT-assisted crops teach the model what to segment after the target is nearby. Full-volume inference asks a harder question: find the target and reject every other tile. Oracle native-96 success and full-volume native-96 failure are exactly what this mismatch predicts.

## 13.2 Text semantics survive, but spatial control is low precision

Location swaps and S3 diagnostics show that text affects the model. The problem is that the response field is broad, often off-target, and difficult to translate into a discrete set of correct components. Text sensitivity is necessary but not sufficient for grounding.

## 13.3 The output is a set, not one mask blob

Many findings are multifocal or diffuse. A selector must choose a variable-size set of components and account for interactions between them. Independent binary component classification and top-k crop selection are mismatched to this structure. The oracle keeps all GT-overlap components, not only the top one.

## 13.4 Candidate precision dominates proposal recall

Top-3 proposal recall was already high, especially for medium/large findings, yet fusion failed. Max fusion cannot remove baseline-positive voxels, but it still reduced Dice, proving that added zoom predictions themselves were often false positive. Better ranking alone is insufficient when local refinement is not calibrated to abstain.

## 13.5 Partial labels corrupt both segmentation and selector targets

Unannotated true instances are treated as negatives by BCE/Tversky and by component labels. This can produce two apparently contradictory behaviors: the model over-segments relative to GT, while the dataset may also penalize correct additional lesions. A component selector trained on GT overlap may learn to discard clinically plausible components.

## 13.6 Strong auxiliary context can collapse the base model

ACA P1 is the clearest example: a representation with good lobe/laterality probe accuracy caused zero foreground under every nonzero coupling condition. The base query distribution and decoder calibration are fragile. Any new context injection should have an exact identity at initialization, a bounded scale, and causal zero/correct/swap controls before training.

## 13.7 Threshold and calibration effects can masquerade as semantic gains

Prompt cleanup increased Dice while AP decreased and recall fell. Lobe constraints also remove prediction volume directly. These are useful operationally, but they should not be interpreted as improved representation learning unless ranking metrics, matched-volume comparisons, and cross-fitted thresholds also improve.

## 13.8 Evaluation lineage drift obscures progress

Different baselines, thresholds, post-processing modes, and cohorts create apparent jumps that are not causal. The project needs one locked evaluation harness with saved logits, stable case-level bootstrapping, and explicit checkpoint ancestry. Without that, an AI agent may incorrectly combine 0.205, 0.274, 0.303, and 0.348 results as if they came from the same system.

# 14. Unresolved checks, possible bugs, and evidence gaps

1. **Canonical evaluation manifest.** Lock one 381-finding manifest, threshold policy, prompt variant, preprocessing version, and post-processing mode. Re-run the best baseline and all surviving candidates through the same path.
2. **Saved logits.** Existing outputs often contain only thresholded summaries at 0.40/0.45/0.50. Save compact per-finding threshold curves and a stratified full-logit subset so calibration and component persistence can be audited without new inference.
3. **Physical spacing.** There is no common-spacing resampling in the current preprocessing. Quantify whether lesion size, crop FOV, and lobe dilation errors correlate with spacing.
4. **Patch-validation checkpoint selection.** Best checkpoints can be selected from only 50 stochastic oracle-like patches and even mid-accumulation. Compare full-volume curves for scheduled checkpoints before trusting patch-best models.
5. **Partial labels.** Audit likely unlabeled same-category components with radiology text, bilateral symmetry, and component persistence. Estimate how much "FP" may be plausible unannotated disease.
6. **Orientation and mask alignment.** Lobe-mask smoke tests must verify RAS orientation, spacing-aware dilation, component seed coordinates, and exact reconstruction of the raw thresholded mask.
7. **Spatial-modifier metric.** The mentor-requested prompt-location FP analysis remains open and should not be replaced by GT-lobe statistics.
8. **Prompt-adapter replication.** The +0.006971 S3 adapter result is important. Reproduce it from a clean frozen baseline, record parameter count/runtime/formula, and test on a second fold or hidden split.
9. **Component metric labeling.** The recovered 0.3509 versus 0.3463 attention-rescoring comparison lacks a confirmed metric name. Reopen the exact report before using it in a manuscript.
10. **Incomplete runs.** Do not treat update-2500 Dual-S3 or the cancelled 9,460-microstep full-resolution run as completed evidence.

# 15. Questions an AI research agent should answer next

The next agent should not begin by adding another unconstrained attention block or another generic loss. It should first answer these concrete questions:

1. Can the current best S3 + prompt-adapter model reproduce its gain under a locked evaluation and case-level cross-validation?
2. How much full-volume error disappears when training includes a much higher rate of correct-prompt, same-case, lesion-absent local tiles?
3. Can candidate components be represented with local native-resolution image evidence, prompt-conditioned differences, anatomy, multi-threshold persistence, and set-level context rather than aggregate geometry alone?
4. Can a selector optimize direct Dice utility or expected FP/TP change while choosing a variable number of components and allowing abstention?
5. Can anatomy be injected only as a bounded late logit prior or seed-dependent gate, with exact identity at scale zero and no risk of foreground collapse?
6. How much of the component-oracle gap is unattainable because candidates cover only 56.7% of GT voxels or because GT is incomplete?
7. Does semantic post-processing remain beneficial after matching predicted volume and threshold calibration, and which parser groups drive the gain?
8. For explicit spatial modifiers, what percentage of FP lies in non-target lobes, and does that error correlate with prompt-swap transfer behavior?
9. Would focal, diffuse, and multifocal findings require separate search/selection policies rather than one universal decoder and fusion rule?
10. Are physical-spacing variation, stochastic checkpoint selection, or pipeline lineage differences masking a real gain or creating false gains?

# 16. Recommended evidence standard for future experiments

Every future experiment should include the following minimum record:

- exact parent checkpoint hash/path and whether optimizer/scheduler were restored;
- all trainable and frozen parameter counts, plus frozen-weight equality check for adapter studies;
- cohort manifest hash, case/finding count, and exclusion reasons;
- microsteps, optimizer updates, patch exposures, prompt-mask pairs, GPU count, accumulation, and wall time;
- full-volume paired baseline and intervention metrics at the same threshold/post-processing;
- finding-level and case-clustered bootstrap CIs;
- per-size, focal/diffuse, category, and parser-group effects;
- predicted volume, FP, recall, HIT transitions, and empty-prediction count;
- identity/zero control and correct/swap/permuted causal controls for any new conditioning path;
- a clear GO/NO-GO rule defined before looking at the final result;
- output directory and exact report/JSON/CSV paths.

<!-- PAGEBREAK -->

# Appendix A. Complete experiment inventory

The following table is generated from the companion JSON ledger. "Evidence level" distinguishes exact recovered values from partial reconstructions.

| ID | Direction | Experiment | Status | Cohort | Delta / key effect | Decision | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | Baseline | Provided VoxTell checkpoint, direct ReX inference | completed | internal ReX evaluation; early pipeline | reference point | baseline only | exact from internal presentation |
| B02 | Baseline | Reconstructed VoxTell fine-tuning | completed | internal ReX evaluation; early pipeline | +0.034 Dice; +0.097 HIT | successful reproduction/adaptation, but still below desired performance | exact from internal presentation |
| L01 | Loss and sampling | Strong-background weighted BCE | completed | 200 cases / 381 findings | +0.0011 Dice; +0.0131 HIT | small sensitivity/background-rejection trade-off; not a breakthrough | exact from internal presentation |
| L02 | Loss and sampling | Anchor85 sampler baseline | completed | 200 cases / 381 findings | small gain over earlier weighted-BCE baseline | retained as a core baseline lineage | exact from prior project discussion |
| L03 | Loss and sampling | Focal Tversky | completed | 200 cases / 381 findings | negative | NO-GO as a general loss replacement | exact from prior project discussion and presentation |
| L04 | Loss and sampling | Asymmetric Unified Focal and boundary variants | completed | 200 cases / 381 findings | no Dice improvement | NO-GO | exact from internal presentation |
| L05 | Loss and sampling | Category-adaptive Tversky | completed | 200 cases / 381 findings | +0.001452 Dice/finding; +0.012609 precision; -0.013142 recall; -338 mm3 median FP | small but consistent GO as the default loss; not a step change | exact from internal presentation |
| T01 | Text and prompt | Category-prefix prompt | completed controlled comparison | 200 cases / 381 findings; 200 optimizer updates | -0.001647 Dice; +2 hits | NO-GO as a standalone prompt format | exact from audit report |
| T02 | Text and prompt | Prompt cleanup and canonical_visual | completed initial full-validation comparison; extended audit pending | 200 cases / 381 findings; threshold 0.5 | +0.006076 cleanup; +0.006442 canonical; AP decreased | useful formatting/calibration clue, but not evidence of better ranking or semantic understanding | exact from prior discussion |
| T03 | Text and prompt | Minimal location-swap causal audit | completed | fixed30; 49 findings screened; 12 eligible; 16 pairs | large causal response to location-word changes | NO-GO for the hypothesis that Qwen/text is ignored; do not prioritize Qwen LoRA or token-level Prompt Composer for this reason | exact from completed job 5307357 |
| T04 | Text and prompt | Directional transfer audit (P0b-lite) | completed | fixed30; 16/16 prompt pairs | laterality more reliable than superior/inferior location | mechanistic GO: text pathway contains directional control; deployment still unreliable | exact from completed job 5313162 |
| T05 | Text and prompt | Frozen rank-16 Prompt Residual Adapter on ordinary VoxTell | completed | full validation, ordinary VoxTell lineage | higher detection, lower overlap | NO-GO on ordinary VoxTell | exact result recovered from prior discussion; implementation details partially recovered |
| T06 | Text and prompt | Frozen rank-16 Prompt Residual Adapter on S3 | completed full validation | 200 cases / 381 findings | +0.006971 Dice; +5 hits | small validated GO; strongest learned improvement in this record, but still modest | exact from prior discussion |
| T07 | Text and prompt | CoLiPri Semantic Residual Hybrid A | completed | fixed30 / 49 findings; 400 steps | +0.000024 Dice; 95% CI [-0.000223, +0.000252] | NO-GO; learned gate shut the residual off | exact from report and paired JSON |
| A01 | Attention | Presence V1 scalar head | completed/obsolete | patch-level training lineage | negative/uncertain | NO-GO as implemented | design and conclusion exact; final controlled metric not recovered |
| A02 | Attention | Presence V2 spatial evidence map | diagnostic/obsolete | patch-level training lineage | not established | NO-GO / superseded by spatial attention | design exact; result incomplete |
| A03 | Attention | S1 single-scale spatial attention | completed diagnostic | attention audit subset | prompt-specific but not spatially precise | NO-GO as sufficient localization | exact from prior discussion |
| A04 | Attention | S2 two-scale, category-routed attention | completed | 200 cases / 381 findings | no meaningful segmentation gain | NO-GO; superseded by S3 | exact from prior discussion |
| A05 | Attention | S3 multiscale voxel-text attention, early controlled run | completed | 200 cases / 381 findings | attention localization improved; Dice approximately unchanged | mechanistic GO, performance NO-GO | exact from prior discussion |
| A06 | Attention | Dual-S3 context/query multi-FOV | completed validated checkpoint | 200 cases / 381 findings | lineage improvement, but no clean large matched gain attributable only to dual view | retained as a strong attention baseline; automatic zoom remained negative | exact from audit report |
| A07 | Attention | Seven-offset ensemble | completed | 381 findings; separate evaluation lineage | -0.01387 overall; focal -0.01415; diffuse -0.01211; small -0.02747; medium -0.01321; large -0.00093 | NO-GO | exact from completed experiment memory |
| Z01 | Multi-resolution | Oracle-localized native-96 refinement | completed | tiny findings; n=20 | +0.1321; bootstrap CI [+0.0608, +0.2149] | strong mechanistic GO: high-resolution refinement works when location is known | exact from completed zoom experiment |
| Z02 | Multi-resolution | Dense full-volume native-96 inference | completed | tiny findings; n=23 in full-volume analysis | -0.0370 Dice despite one extra hit | NO-GO | exact from completed zoom experiment |
| Z03 | Multi-resolution | Unbounded zoom proposals and fusion | completed | 381 findings | large overall regression | NO-GO | exact from experiment memory |
| Z04 | Multi-resolution | Bounded top-3 native-96 proposals | completed | 381 findings | negative overall; tiny subgroup gain | NO-GO as global fusion | exact from completed experiment |
| Z05 | Multi-resolution | Low-threshold component oracle | completed oracle | 381 findings | +0.0884 overall; tiny +0.0982; small +0.1639; medium +0.0734; large +0.0378 | major non-deployable headroom; selector is the bottleneck | exact from completed oracle |
| Z06 | Multi-resolution | P0-B2-lite fixed-96 crop-coverage ranker | completed | fold 0; 78 findings | +15.38 pp top-1; +5.13 pp top-3; small safe recall 84% to 92%; tiny unchanged at 66.67% | useful diagnostic, insufficient for deployment | exact from completed experiment memory |
| Z07 | Multi-resolution | Top-8 choose-0-to-3 crop oracle and learned selectors | completed | fold 0; 78 findings | oracle +0.014130; both learned selectors negative | NO-GO for current utility selectors | exact from prior discussion |
| Z08 | Multi-resolution | Zoom-out mixed training and direct native-256 inference | completed | 381 findings | +0.000048 under standard inference; -0.016687 at native-256 | NO-GO as a default scale | exact from report |
| Z09 | Multi-resolution | Native-256 Stage-1 plus native-192 Stage-2 fusion | completed | 381 findings | all global fusions below reference | NO-GO globally; subgroup routing signal only | exact from completed experiment memory |
| C01 | Component selection | Threshold-0.4 component oracle | completed oracle | 381 findings; S3 step-6000 lineage | approximately +0.0850 Dice | strongest evidence of component-selection headroom | exact from report |
| C02 | Component selection | Basic tabular component selector and dual-view CNN | completed | component diagnostic | near zero or negative | NO-GO | exact summary recovered; detailed model table not recovered |
| C03 | Component selection | Attention-feature logistic rescoring | completed diagnostic | component diagnostic | -0.0046 on the reported rescoring metric | NO-GO; simple attention statistics were not useful selector features | exact summary; metric label not fully recovered |
| C04 | Component selection | P0-F-lite same-anatomy component separability/ranking | completed | 162 eligible findings / 112 cases; test fold 0 has 29 findings and 378 candidates | learned ranking did not satisfy required recall/separability | Decision D - NO-GO | exact from completed experiment memory |
| N01 | Anatomy and lobe priors | Hard prompt-to-lobe component filter | completed diagnostic | 381 findings; 229 with explicit lobe text | +0.01268 Dice with safety cost | NO-GO as a hard deployable filter | exact from experiment memory |
| N02 | Anatomy and lobe priors | Soft lobe prior | completed | 381 findings | approximately +0.000001 Dice | NO-GO | exact from experiment memory |
| N03 | Anatomy and lobe priors | Scalar lobe-label input diagnostic and training | completed | 381 findings | +0.00012481 diagnostic; training effectively neutral | NO-GO | exact from experiment memory |
| N04 | Anatomy and lobe priors | Semantic-aware inference constraint v1 | completed | 200 cases / 381 findings | +0.015283 raw to semantic; +0.004715 vs whole10; +0.001187 vs laterality10 | practical post-processing GO; not an end-to-end model gain | exact from job 5306186 summary |
| N05 | Anatomy and lobe priors | Output-space lobe-alignment / lobe-mask encoding streamlined runs | completed; details partially recovered | fixed30 / 49 findings | no robust improvement | final NO-GO | verdict exact; full row table not recovered |
| N06 | Anatomy and lobe priors | ACA-style anatomy representation P0 | completed | anatomy probe plus fixed30 causal framework | representation learning succeeded | P0 GO; proceed only with safe bounded coupling | exact from final ACA report |
| N07 | Anatomy and lobe priors | ACA P1 direct anatomy-query coupling | completed | fixed30 / 49 findings | catastrophic foreground collapse | NO-GO | exact from outputs/aca_anatomy_final_20260730T225800Z/report.md |
| I01 | Image input | HU audit and direct multi-window input pilot | completed | fixed30 / 49 findings; 500 updates | +0.002308 Dice; bootstrap CI [-0.012476, +0.014576]; HIT +0.020408; FP ratio 1.0031 | NO-GO; do not continue to 1000 updates, 3 channels, or 200-case validation under this direct-input design | exact from completed multi-window experiment memory |
| O01 | Incomplete/ongoing | All-category full-resolution 1/1 attention versus ordinary VoxTell | planned/partially run; no valid final metric | training lineage | not evaluable | open/incomplete | exact from training audit |
| O02 | Incomplete/ongoing | Direct spatial-modifier false-positive audit requested by mentor | planned at report cutoff | findings with explicit spatial modifiers | missing requested metric | high-priority open analysis | exact status from latest discussion |

# Appendix B. Key equations and implementation details

## B.1 Category-adaptive Tversky

```text
Tversky = TP / (TP + alpha_FP * FP + beta_FN * FN)
L_nonempty = mean_BCE + (1 - Tversky)
L_empty = 0.5 * mean_BCE
```

## B.2 S3 voxel-text attention

```text
v = normalize(W_v * visual_feature)
t = normalize(W_t * text_embedding)
a = sigmoid(exp(s) * cosine(v, t) + b)
gated_skip = skip * ((1 - rho) + rho * a)
```

## B.3 S3 auxiliary loss

```text
L_direct = BCE(a, target) + 1 - softDice(a, target)
L_hardBG = ReLU(0.05 - mean_fg(a) + mean_top1pct_bg(a))
L_view = L_seg + 0.05 * L_direct + 0.02 * L_hardBG
```

## B.4 Dual-S3 view coupling

```text
L_total = (L_context + 0.5 * L_zoom) / 1.5 + 0.05 * L_consistency
```

The zoom teacher is detached. Context and zoom share the same network; there is no explicit feature concatenation.

## B.5 CoLiPri semantic residual

```text
q_final = q_Qwen + rho * (q_CoLiPri_adapted - q_Qwen)
rho = 0.5 * tanh(raw_rho)
```

## B.6 Semantic anatomy post-processing

```text
exact-lobe prompt  -> exact lobe union, dilated 15 mm
side-level prompt  -> left/right lung union, dilated 15 mm
broad/unknown      -> whole lung union, dilated 10 mm
filtered mask      -> prediction AND allowed-anatomy region
```

# Appendix C. Important artifact and report locations

- Project root: `/common/lidxxlab/chushu/RexGroundingCT`
- Dual-S3 validated checkpoint: `outputs/dual_s3_multifov/.../checkpoint_update_2000.pth`
- Dual-S3 compatible export: `.../voxtell_compatible_models/update2000_dual_s3_context_only_update2000/fold_0/checkpoint_final.pth`
- Category-prefix experiment: `outputs/category_prompt_experiment/`
- S3 component oracle: `outputs/s3_attention_candidate_diagnostic_all_categories_step6000/report.md`
- S3 component table: `outputs/s3_attention_candidate_diagnostic_all_categories_step6000/candidate_component_summary.csv`
- Zoom-in Stage-1/two-stage report: `outputs/dual_s3_two_stage_zoom96/REPORT.md`
- Zoom-out comparison: `outputs/zoom_out_256_matched/fullvol_evaluation/zoomout_mix_native256_to192_update2625/paired_vs_same_model_native192/report.md`
- CoLiPri report: `reports/colipri_semantic_residual_hybrid_v1.md`
- CoLiPri paired metrics: `outputs/colipri_semantic_residual_hybrid_v1/pilot400_5281554/fixed30/semantic_residual_fixed30.paired.json`
- ACA final report: `outputs/aca_anatomy_final_20260730T225800Z/report.md`
- Prompt location swap: Slurm job `5307357`
- Directional transfer audit: Slurm job `5313162`
- Semantic-aware anatomy post-processing: Slurm job `5306186`
- P0-F feature extraction: Slurm job `5314465`; report recovery jobs `5314568`, `5314587`, `5314622`

# Appendix D. Evidence caveats

This synthesis intentionally preserves several unresolved discrepancies instead of forcing a single number:

- Zoom-out mixed-training delta appears as +0.000048 in the exact report and approximately +0.000338/+0.000343 in other summaries under slightly different matched comparisons. All versions support the same conclusion: effectively neutral.
- The attention-rescoring values 0.3509 and 0.3463 were recovered without a confirmed metric label. They should not be called Dice in a manuscript until the exact report is reopened.
- Exact formulas, parameter counts, and runtime for the rank-16 prompt residual adapter were not recovered, although the full-validation paired metrics and CIs were recovered.
- The fixed30 lobe-alignment streamlined report was summarized as NO-GO, but only the baseline raw/whole10 values were recovered here.
- The exact two HU-window definitions and first-convolution initialization for the multi-window pilot were not recovered. The paired outcome and decision are exact.

These caveats are part of the technical result. An AI agent should treat them as retrieval tasks before citing the affected experiments or designing close variants.
