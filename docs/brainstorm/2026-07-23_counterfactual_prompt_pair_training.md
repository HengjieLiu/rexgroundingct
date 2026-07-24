---
doc_type: research_brainstorm
created: 2026-07-23
updated: 2026-07-23
status: active
topic: Counterfactual prompt-pair supervision for ReXGroundingCT
scope: Minimal-change VoxTell training design
depends_on:
  - 2026-07-23_voxtell_anatomy_language_capability_gap.md
related_files:
  - scripts/rexgroundingct/train_text_conditioned_voxtell.py
  - scripts/rexgroundingct/precompute_text_embeddings.py
  - docs/prompt_illustration/
---

# Counterfactual Prompt-Pair Training

## Research Motivation

A factual training prompt can be converted into minimally different prompts
that should either select another known lesion, reject the original lesion, or
preserve the same prediction.

Example factual prompt:

> A nodule in the left upper lobe.

Candidate modifications:

- A nodule in the right upper lobe.
- A nodule in the left lower lobe.
- Septal thickening in the left upper lobe.
- A pulmonary nodule in the left upper lobe.
- A nodule in the left lung.

The intention is to enrich supervision without collecting new masks and
without replacing the VoxTell architecture. The model should learn that small,
meaningful language changes have predictable effects on its segmentation.

## Practical Verdict

This is a practical and high-value direction. It can initially be implemented
through:

- an offline prompt parser and generator,
- a counterfactual manifest,
- precomputed text embeddings,
- structured sampling,
- paired losses,
- new validation metrics.

The base segmentation network can remain unchanged for the first experiment.
However, reliable left/right and lobe counterfactual behavior may eventually
require a small global-coordinate input because current sliding-window patches
do not explicitly carry their position in the full CT.

The most important design rule is:

> Do not assign an all-zero target merely because the counterfactual prompt is
> absent from the case annotations.

ReX training annotations are not necessarily exhaustive for every instance of
an annotated finding: the protocol limits training masks to three
representative instances. Treating all unlisted prompts as globally absent
could therefore teach false negatives, especially for same-pathology spatial
edits. The protocol does not directly prove that an entirely different
unlisted pathology is present. See:

[`2026-07-23_rex_training_partial_instance_annotation_evidence.md`](2026-07-23_rex_training_partial_instance_annotation_evidence.md)

## Three Counterfactual Label Types

### Type A: Known positive counterfactual

Use this when the same image contains a real finding and mask matching the
edited prompt.

Example:

| Prompt | Target |
| --- | --- |
| Nodule in the left upper lobe | Left-upper-lobe nodule mask |
| Nodule in the right lower lobe | Right-lower-lobe nodule mask |

This is the strongest supervision because it teaches true spatial relocation.
Both prompts have positive segmentation targets, and neither requires an
assumption about absence.

The pair can also receive a separation loss so that each prompt prefers its own
mask over the other finding's mask.

### Type B: Partial negative counterfactual

Use this when the edited prompt does not have a known matching mask, but global
absence is uncertain.

For:

> Nodule in the right upper lobe

paired with a known left-upper-lobe nodule, it is safe to require that the
counterfactual prompt does not select the known left-sided lesion. It is not
always safe to require an empty prediction everywhere else.

Training should therefore:

- penalize counterfactual probability inside the factual GT mask,
- rank the factual prompt above the counterfactual on that mask,
- leave uncertain voxels outside the factual mask unlabeled.

This is a partial-label or positive-unlabeled treatment rather than an ordinary
empty-mask target.

### Type C: Confirmed absent counterfactual

Use an all-zero target only when there is strong evidence that the requested
finding is absent. Evidence may include:

- all normalized findings for the case,
- report-derived concepts,
- anatomical CoT,
- compatible organ or lobe masks,
- explicit negative statements when available,
- manual review for a small high-confidence benchmark.

Even these examples should initially receive less loss weight than factual
positive masks.

## Why "Septal Thickening" Is Not Automatically Empty

Changing:

> nodule in the left upper lobe

to:

> septal thickening in the left upper lobe

is a valuable pathology counterfactual. It tests whether the model distinguishes
a focal nodule from an interstitial pattern.

However, septal thickening could genuinely coexist in the same lung or could be
unannotated. The safe objective is:

- the septal-thickening prompt should not activate on the known nodule mask;
- global voxels should remain ignored unless absence is independently
  confirmed.

This still gives a strong pathology-discrimination signal without claiming
knowledge that the annotation does not provide.

## Prompt Representation

Each prompt should be parsed into a versioned structured record:

```json
{
  "prompt": "A nodule in the left upper lobe",
  "pathology": "nodule",
  "laterality": "left",
  "organ": "lung",
  "lobe": "upper lobe",
  "segment": null,
  "spatial_relation": null,
  "morphology": null,
  "size": null,
  "quantity": "single",
  "severity": null,
  "temporal": null,
  "uncertainty": null,
  "parse_confidence": 1.0
}
```

Every concept should retain:

- normalized concept ID,
- original text span,
- English canonical term,
- Chinese translation for review,
- category code where available,
- parse confidence,
- source rule or reviewer decision.

The existing bilingual prompt and vocabulary documentation can support manual
review, but training requires machine-readable structured records.

## Counterfactual Operation Vocabulary

### Invariant operations

These should preserve the target:

- faithful paraphrase,
- recognized synonym replacement,
- minor article or word-order normalization,
- removal of a weak qualifier that does not select a different instance.

### Spatial operations

These should reject the factual lesion, relocate to a known target, or abstain:

- left to right and right to left,
- upper to lower and lower to upper,
- lobe replacement,
- segment replacement,
- central to peripheral,
- subpleural to peribronchial,
- apical to basal.

### Pathology operations

These should reject the factual lesion unless another known mask matches:

- nodule to ground-glass opacity,
- nodule to septal thickening,
- consolidation to atelectasis,
- bronchiectasis to bronchial wall thickening,
- scarring to active infiltration.

Pathology substitutions must be curated. Some concepts overlap or commonly
coexist, and certain category distinctions may not be visually exclusive.

### Specificity operations

These do not necessarily imply an empty result:

- remove laterality,
- remove lobe or segment,
- remove morphology,
- remove size,
- remove quantity.

The expected output may become a superset, a broader region, or a less certain
prediction. These should not use ordinary equality or empty-mask losses.

### Contradiction operations

Anatomically impossible phrases can test abstention, but they should be a
separate class:

- left middle lobe,
- right lingula,
- an incompatible segment and lobe combination.

Impossible synthetic phrases are easy for a model to detect as text artifacts.
They should not dominate training or replace plausible hard negatives.

## Anatomical Validity Rules

The generator must encode a basic lung ontology:

- the right lung has upper, middle, and lower lobes;
- the left lung has upper and lower lobes;
- the lingula belongs to the left upper lobe;
- right-middle-lobe medial and lateral segments have no literal left-middle-lobe
  counterpart;
- segment substitutions must remain compatible with the selected lobe;
- bilateral and diffuse prompts require different logic from unilateral focal
  findings;
- airway, pleural, fissural, and parenchymal locations require different
  spatial priors.

The first experiment should exclude ambiguous, diffuse, bilateral, comparison,
and multi-location prompts. Begin with high-confidence unilateral prompts that
contain one disease and one explicit lobe or segment.

## Candidate Generation Pipeline

1. Extract every training finding in numeric finding-ID order.
2. Normalize prompt text without discarding the exact original string.
3. Parse each prompt into semantic slots.
4. Assign a confidence score to each parsed slot.
5. Generate only one atomic edit per candidate.
6. Reject anatomically invalid edits unless creating an explicit contradiction
   example.
7. Search all findings in the same case for a matching real counterfactual.
8. Label the pair as known positive, partial negative, or confirmed absent.
9. Record the evidence supporting that label.
10. Deduplicate exact and semantically equivalent candidates.
11. Balance candidates by pathology, side, location, and operation.
12. Precompute text embeddings using the same frozen Qwen pipeline.
13. Freeze the generator version before validation.

An example manifest row should include:

```json
{
  "case_name": "example_case.nii.gz",
  "finding_id": "1",
  "source_prompt": "A nodule in the left upper lobe",
  "counterfactual_prompt": "A nodule in the right upper lobe",
  "operation": "swap_laterality",
  "changed_slot": "laterality",
  "source_mask_channel": 0,
  "counterfactual_label_type": "partial_negative",
  "counterfactual_mask_channel": null,
  "parse_confidence": 1.0,
  "absence_confidence": 0.4,
  "generator_version": "v1"
}
```

## Training Objective

Let:

- `P_pos` be the prediction for the factual prompt,
- `P_cf` be the prediction for the counterfactual prompt,
- `M_pos` be the factual GT mask,
- `M_cf` be a known counterfactual GT mask when available.

A practical objective is:

```text
L_total =
    L_seg(P_pos, M_pos)
  + w_pair * L_pair
  + w_anti * L_anti
  + w_para * L_paraphrase
  + w_empty * L_confirmed_empty
```

### Factual segmentation loss

Use the existing Dice and BCE objective on `P_pos` and `M_pos`.

### Pairwise ranking loss

Require the factual prompt to score its lesion more strongly than the edited
prompt:

```text
L_pair = max(
    0,
    margin - mean(P_pos inside M_pos) + mean(P_cf inside M_pos)
)
```

### Anti-overlap loss

Suppress the counterfactual response on the known factual lesion:

```text
L_anti = mean(P_cf inside M_pos)
```

This loss is valid even when another unannotated counterfactual finding could
exist elsewhere.

### Known-positive counterfactual loss

When `M_cf` exists, also apply:

```text
L_seg(P_cf, M_cf)
```

Optionally require each prompt to prefer its own mask:

```text
score(P_pos, M_pos) > score(P_pos, M_cf)
score(P_cf, M_cf) > score(P_cf, M_pos)
```

### Paraphrase consistency

For a faithful paraphrase, supervise both prompts with the same GT mask.
Prediction consistency can be added with a stop-gradient or teacher target to
avoid trivial collapse.

### Confirmed-empty loss

Only Type C examples receive a global empty-mask BCE or probability-mass loss.
Use a lower weight initially and monitor empty-prediction frequency.

## Interaction with Patch Sampling

Counterfactual pairs should share the same image encoding whenever possible.
The current model can process multiple text queries for one image, which avoids
duplicating the expensive image encoder pass.

Patch sampling requires care:

- A crop centered on the factual lesion is ideal for pairwise rejection because
  the counterfactual must not select that lesion.
- It does not prove that the counterfactual is globally absent.
- A crop in the edited anatomical region is useful when testing relocation.
- Whole-volume global coordinates are needed if the model must distinguish
  visually similar left and right patches without reliable landmarks.

## Minimal Model Change for Spatial Counterfactuals

The first experiment can leave VoxTell unchanged. If spatial swaps remain weak,
the smallest useful model change is one of:

1. Add full-volume normalized `x`, `y`, and `z` coordinate channels.
2. Embed patch center, patch extent, spacing, and orientation and add that
   embedding to the image or text query.
3. Use a parsed anatomy term to select an external lobe or airway ROI before
   segmentation.

Coordinate channels provide voxel-level position and are the most direct test.
The pretrained first-layer CT weights can be retained, with new coordinate
weights initialized to zero.

## Sampling and Balance

Counterfactual supervision can create shortcuts if edits are unbalanced.
Recommended safeguards:

- generate equal numbers of left-to-right and right-to-left edits;
- balance upper-to-lower and lower-to-upper edits;
- ensure every substituted term also appears as a factual positive;
- stratify by MICCAI pathology category;
- cap frequent nodule and opacity examples;
- oversample rare categories conservatively;
- generate several candidates offline but sample only one per finding per
  epoch;
- prevent synthetic prompt wording from revealing the label;
- warm up ordinary segmentation before increasing counterfactual loss;
- monitor how often the model returns an empty mask for factual prompts.

A reasonable initial query schedule is:

| Query type | Initial share |
| --- | ---: |
| Factual positive | 60-70% |
| Spatial counterfactual | 15-20% |
| Same-case known-positive reassignment | 5-10% |
| Pathology counterfactual | 5-10% |
| Paraphrase invariant | 5-10% |

These are starting values, not fixed conclusions. Actual weights should be
chosen after counting high-confidence candidates.

## Split and Leakage Policy

Understanding the vocabulary across train, validation, and test is useful for
documentation and error analysis. It must not turn into validation or test
supervision.

Recommended policy:

- define the medical ontology from established anatomy and training prompts;
- fit synonym lists, frequencies, and balancing weights on training only;
- freeze parsing and generation rules before validation evaluation;
- create a separate validation counterfactual benchmark without adding those
  examples to training;
- never use test masks or test outcomes to tune generation rules;
- record generator version and ontology version in every experiment.

## Evaluation Beyond Dice

### Standard task metrics

- mean global Dice per finding,
- hit rate,
- category-stratified Dice,
- empty-prediction frequency.

### Counterfactual comprehension metrics

- factual score minus counterfactual score on the factual GT,
- counterfactual activation inside factual GT,
- known-positive relocation Dice,
- laterality compliance,
- lobe compliance,
- centroid displacement after a location edit,
- paraphrase mask consistency,
- predicted-volume response to reduced specificity,
- confirmed-negative false-positive volume.

### Important slices

Report results separately for:

- focal versus diffuse findings,
- left/right versus bilateral prompts,
- lobe-level versus segment-level prompts,
- pathology substitutions versus location substitutions,
- common versus rare categories,
- known-positive, partial-negative, and confirmed-absent pairs.

## First Ablation Matrix

| Arm | Training change | Model change | Purpose |
| --- | --- | --- | --- |
| A | Existing random negatives | None | Current baseline |
| B | Structured counterfactuals with global empty targets | None | Measure effect and false-negative risk |
| C | Partial-label ranking and anti-overlap | None | Test the safer paired objective |
| D | Arm C plus known-positive relocation pairs | None | Test positive compositional grounding |
| E | Arm D plus global coordinates | Small input change | Test whether patch position limits anatomy |

Arm B is scientifically useful as a comparison but should not become the
default recipe unless its labels are proven reliable.

## Compute and Implementation Feasibility

The expected work is primarily in the data and loss pipeline:

- prompt parser and ontology,
- manifest generator and validator,
- embedding precomputation,
- paired sampler,
- partial-label losses,
- counterfactual metrics and reports.

Text embeddings can be precomputed once. Factual and counterfactual prompts can
share image features, so paired training should add much less compute than two
fully independent image passes. Four RTX 6000 Ada GPUs are sufficient for the
proposed ablations.

This approach should be understood as compositional grounding rather than
general-purpose reasoning. It can make the model accountable to controlled
language changes while remaining close to the existing VoxTell pipeline.

## Main Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Unannotated pathology becomes a false negative | Use partial-label anti-overlap instead of global empty masks |
| Model learns that synthetic wording means "empty" | Make minimal fluent edits and match factual wording |
| Model learns side or lobe frequency priors | Bidirectional balancing and factual-positive coverage |
| Spatial counterfactuals fail because patches lack position | Add global coordinates or anatomy-guided ROI selection |
| Too many negatives cause empty-mask collapse | Warm-up, low negative weight, and empty-rate monitoring |
| Pathology substitutions are not mutually exclusive | Curate compatibility rules and prefer ranking on known GT |
| Test vocabulary influences training design | Freeze ontology and generator using training data and medical knowledge |
| Diffuse findings violate focal assumptions | Exclude initially and create a separate diffuse policy |

## Recommended First Milestone

Before training a full model:

1. Parse all training prompts into semantic slots.
2. Measure parser coverage and manually review a stratified sample.
3. Count valid unilateral location swaps by category.
4. Count same-case known-positive counterfactual pairs.
5. Build a small validation benchmark from fixed cases.
6. Run the current checkpoint on factual, paraphrase, location-swap, and
   pathology-swap prompts.
7. Determine whether failures primarily reflect language insensitivity or
   missing global coordinates.
8. Finalize the loss and sampling ratios from observed candidate counts.

This milestone produces useful scientific evidence even before any new
fine-tuning run.

## Related Work

- [ReXGroundingCT](https://arxiv.org/abs/2507.22030)
- [VoxTell](https://arxiv.org/abs/2511.11450)
- [Mask Grounding](https://arxiv.org/abs/2312.12198)
- [Compositional hard negatives for vision-language learning](https://arxiv.org/abs/2306.08832)
- [Counterfactual Segmentation Reasoning](https://arxiv.org/abs/2506.21546)
- [Meta Compositional Referring Expression Segmentation](https://arxiv.org/abs/2304.04415)
