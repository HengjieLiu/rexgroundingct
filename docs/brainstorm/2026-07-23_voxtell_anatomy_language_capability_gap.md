---
doc_type: research_brainstorm
created: 2026-07-23
updated: 2026-07-23
status: active
topic: VoxTell anatomy, pathology, language, and reasoning capability gaps
scope: ReXGroundingCT validation observations and model redesign directions
related_files:
  - external/VoxTell/voxtell/inference/predictor.py
  - external/VoxTell/voxtell/model/voxtell_model.py
  - external/VoxTell/voxtell/model/transformer.py
  - external/VoxTell/voxtell/utils/text_embedding.py
  - scripts/rexgroundingct/train_text_conditioned_voxtell.py
  - docs/voxtell/preprocessing.md
  - docs/voxtell/normalization.md
---

# VoxTell Anatomy and Language Capability Gap

## Research Motivation

Visual inspection of ReXGroundingCT validation predictions suggests that the
model frequently recognizes a broad image pattern without reliably satisfying
the full anatomical and linguistic instruction. A representative failure is a
prompt describing bronchial wall thickening while the model activates around
cardiac or other central structures instead of following the relevant airway
anatomy.

The central research questions are:

1. Does the model understand anatomy, pathology, and language composition, or
   is it primarily matching local CT textures to a sentence-level embedding?
2. Which failures come from limited ReXGroundingCT supervision, and which are
   imposed by the current architecture?
3. Can reasoning or iterative editing be added with four RTX 6000 Ada GPUs and
   ReXGroundingCT-scale training data?
4. Can controlled prompt changes expose and eventually correct the model's
   language-comprehension failures?

## Main Conclusion

The observed errors are probably not explained by insufficient training epochs
alone. They reveal a structural gap between local pattern recognition and
compositional grounding.

The current pipeline can learn associations such as "ground-glass opacity
looks hazy" or "bronchial abnormalities occur near tubular structures."
However, it has no explicit mechanism requiring the final mask to satisfy all
parts of a prompt:

- the requested pathology,
- the correct side,
- the correct lobe or segment,
- the stated spatial relationship,
- the requested morphology, size, or quantity.

The most important immediate hypothesis is that sliding-window inference hides
global anatomical position from the model. The second is that reducing the
whole prompt to one sentence vector makes it difficult to bind individual
language concepts to different parts of the image.

## Current Pipeline and Its Implications

| Stage | Current behavior | Likely implication |
| --- | --- | --- |
| CT input | A standardized volume is normalized and processed through sliding windows | The model sees high-resolution local detail |
| Image crop | VoxTell uses `192 x 192 x 192` patches | Many patches do not contain enough global landmarks |
| Position | Patches receive the same internal positional encoding; full-volume patch origin is not supplied | Left/right, upper/lower, and lobe identity can be ambiguous |
| Text preprocessing | Prompt is lowercased and wrapped in an embedding instruction | Surface formatting is standardized |
| Text representation | Frozen Qwen3-Embedding-4B output is reduced by last-token pooling to one vector | Disease, anatomy, and modifiers are not separately grounded |
| Image-text fusion | The sentence vector becomes a segmentation query | The model can condition on the sentence globally but has weak token-level accountability |
| Training loss | Dice plus weighted BCE | Wrong-location predictions are penalized only through final mask overlap |
| Negative prompts | Prompts absent from the current case can be treated as empty targets | Incomplete annotations can create false-negative supervision |

Relevant implementation points:

- Sliding-window prediction:
  [`predictor.py`](../../external/VoxTell/voxtell/inference/predictor.py)
- Fixed positional encoding and sentence projection:
  [`voxtell_model.py`](../../external/VoxTell/voxtell/model/voxtell_model.py)
- Last-token text pooling:
  [`text_embedding.py`](../../external/VoxTell/voxtell/utils/text_embedding.py)
- Current prompt sampling and loss:
  [`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py)

The active decoder path does not appear to use query-to-query self-attention.
Therefore, interference between multiple prompts in one case is not the leading
explanation. The more important issue is what each individual query can infer
from a local patch without global coordinates.

## Why Central-Structure Errors Are Plausible

Bronchial wall thickening is not simply a generic lung-texture target. Correct
localization requires a sequence resembling:

1. Identify the trachea and carina.
2. Follow the appropriate main bronchus.
3. Follow lobar and segmental bronchi.
4. Evaluate wall thickness around airway lumina.
5. Enforce the requested side, lobe, and segment.

Central airways, vessels, mediastinal boundaries, and structures adjacent to
the heart can share strong edges and tubular or curvilinear appearances.
A local patch can therefore provide plausible texture cues while still lacking
the context needed to identify the correct anatomical tree. Prediction around
the myocardium or pericardial border is an error, even though central
bronchovascular anatomy can legitimately lie near the heart.

For this category, a whole-lung mask alone is not a sufficient prior. A useful
prior should be derived from the airway tree, dilated airway walls, and the
prompted lobe or side.

## Data and Supervision Limits

The ReXGroundingCT training split contains:

- 2,992 cases,
- 7,687 finding instances,
- 6,148 unique prompt strings.

The high ratio of unique prompts to findings makes it difficult to learn every
combination of disease, anatomy, morphology, and quantity from repetition
alone.

The VoxTell paper's instance-focused experiment should not be interpreted as
ReX-only fine-tuning. Its training pool also included public lesion datasets
and anatomy-derived localization examples. Base VoxTell was pretrained on more
than 62,000 volumes. The current ReX-only run has much less positive variation
and fewer anatomical anchors.

Training annotations can also be incomplete at the instance level. ReX
training masks include at most three representative entities for an annotated
finding, whereas validation and test use exhaustive instance labeling.
Therefore, particularly for the same pathology at another location:

> Absence from the list of annotated findings is not proof that a pathology is
> absent from the CT.

The three-instance rule directly establishes possible missing instances of an
annotated finding. It does not prove that an entirely different unlisted
pathology is present. That broader concern is a conservative modeling
assumption. The evidence and its limits are documented in:

[`2026-07-23_rex_training_partial_instance_annotation_evidence.md`](2026-07-23_rex_training_partial_instance_annotation_evidence.md)

## What "Reasoning" Would Mean Here

It is useful to distinguish several capabilities:

1. **Appearance recognition**
   - Detect image patterns resembling a nodule, opacity, thickening, or scar.
2. **Anatomical localization**
   - Identify side, lobe, segment, pleura, fissure, airway, or mediastinum.
3. **Compositional grounding**
   - Require one prediction to satisfy pathology and location simultaneously.
4. **Verification**
   - Check whether the predicted mask violates part of the prompt.
5. **Iterative correction**
   - Accept a previous mask plus a correction and produce a revised mask.
6. **Abstention**
   - Return an empty or low-confidence result for an impossible or absent
     combination.

Current VoxTell mainly addresses the first three in a single forward pass, with
limited explicit supervision for the second and third. It has no dedicated
verifier, uncertainty head, previous-mask input, or correction interface.

## Multi-Round Editing Feasibility

### Feasible with current hardware

Four 48 GiB RTX 6000 Ada GPUs are sufficient for:

- lightweight anatomy-aware adapters,
- a low-resolution whole-volume locator,
- global coordinate channels,
- role-specific language projections,
- a mask-refinement network,
- LoRA experiments,
- synthetic click or scribble correction training,
- parallel ablations or multi-GPU fine-tuning.

### Not realistic from ReX alone

It is not realistic to train a new general-purpose medical reasoning
vision-language model from scratch using only 7,687 ReX training findings.
Fine-tuning a large language encoder also risks memorizing rare phrases without
fixing the missing global visual context.

### Interactive versus automatic use

An interactive model could be trained without manual interaction logs by
simulating corrections:

- select the largest false-negative region for a positive click,
- select the largest false-positive region for a negative click,
- provide the previous mask as an additional input,
- train the refiner to produce the corrected GT mask.

This is practical for a separate annotation or clinical-review tool. It would
not directly improve an automatic challenge submission unless the refinement
step can generate its own reliable corrections. Self-refinement without a
verifier may sharpen boundaries while preserving the wrong anatomical target.

## Prompt Perturbation as a Diagnostic

Prompt masking and controlled rewriting can determine whether the model uses
specific language concepts. Different edits require different expected
behaviors:

| Prompt change | Expected behavior |
| --- | --- |
| Synonym or faithful paraphrase | Prediction should remain stable |
| Remove a weak qualifier such as "mild" | Location should usually remain stable |
| Remove laterality or lobe | Prediction may become broader or less certain |
| Swap left and right | Prediction should relocate or abstain |
| Swap upper and lower lobe | Prediction should relocate or abstain |
| Remove the pathology noun | Original lesion prediction should lose specificity |
| Replace the pathology | Original lesion should no longer be selected |
| Contradict anatomical terms | Model should abstain or report low confidence |
| Shuffle words | Diagnostic for bag-of-words behavior, not a training target |

These relationships can be organized into four behaviors:

- **Invariance:** equivalent language should preserve the mask.
- **Equivariance:** changing location should relocate the mask.
- **Monotonicity:** removing specificity may broaden the possible region.
- **Contradiction:** an impossible or absent request should suppress output.

Useful diagnostic measurements include:

- Dice and hit rate for the original prompt,
- prediction overlap between original and modified prompts,
- probability change inside the GT lesion,
- predicted volume ratio,
- centroid displacement,
- laterality and lobe compliance,
- probability outside the allowed anatomy,
- false-positive volume for a confirmed-negative prompt.

## Available Anatomical Supervision

The local dataset includes:

[`anatomical_cot.json`](/data/hengjie/datasets/rexgroundingct/anatomical_cot.json)

It covers nearly all training findings and commonly decomposes localization as:

```text
whole scan -> lung -> lobe -> segment or structure -> lesion
```

This is especially relevant for airway findings because the descriptions
explicitly reference the trachea, carina, main bronchi, lobar bronchi, and
segmental bronchi. The generated chains and boxes should be treated as noisy
supervision rather than clinical ground truth, but they could supervise a
coarse anatomy locator without new manual labels.

## Candidate Redesigns

### 1. Anatomy-aware postprocessing baseline

Parse the prompt and softly suppress prediction outside a compatible anatomy
region:

- parenchymal disease: lung or prompted lobe,
- bronchial disease: airway-centered region,
- pleural disease: dilated pleural shell,
- fissural disease: lobe-boundary region,
- apical or basal disease: corresponding normalized lobe region.

Use soft penalties or dilated masks instead of hard clipping. Pleural, hilar,
and subpleural abnormalities can cross imperfect organ boundaries.

### 2. Add global coordinates

Provide normalized full-volume `x`, `y`, and `z` coordinates to each patch, or
embed the crop origin and spacing. New coordinate-channel weights can be
zero-initialized so the pretrained CT channel is preserved.

This is a small architectural change and directly tests whether missing global
position causes side and lobe errors.

### 3. Coarse-to-fine localization

Use a low-resolution whole-volume stage to predict:

- relevant anatomy,
- coarse heatmap,
- bounding box,
- centroid or lobe-level region.

Then run high-resolution VoxTell only in the selected region. GT masks provide
box and centroid targets, while anatomical CoT can provide hierarchy labels.
Diffuse findings should use region heatmaps rather than a single centroid.

### 4. Structured language roles

Parse the prompt into:

- pathology,
- laterality,
- lobe or segment,
- spatial relationship,
- morphology,
- size and quantity,
- severity or temporal qualifier.

Keep the original full-sentence embedding, but add role-specific embeddings.
Location roles can guide coarse localization while pathology and morphology
roles guide fine segmentation.

### 5. Counterfactual supervision

Generate minimally edited prompts and train the model to preserve, relocate, or
suppress masks according to the semantic change. The detailed proposal is in:

[`2026-07-23_counterfactual_prompt_pair_training.md`](2026-07-23_counterfactual_prompt_pair_training.md)

### 6. Interactive refinement

Train a separate model using the CT, original prompt, previous mask, and
synthetic corrective interactions. Treat this as an annotation and model-QC
track rather than the first challenge-performance intervention.

## Recommended Research Order

1. Build a no-training perturbation benchmark on the fixed validation subset.
2. Audit laterality, lobe, pathology, paraphrase, and contradiction behavior.
3. Replace noisy random negatives with structured counterfactual pairs.
4. Test a category-specific soft-anatomy gating baseline.
5. Add global coordinates and repeat the same counterfactual benchmark.
6. Train a low-resolution locator using GT boxes and anatomical CoT.
7. Add structured role embeddings if the language audit still shows weak
   composition.
8. Explore interactive refinement as a separate capability.

## Core Ablation Questions

| Hypothesis | Minimal experiment |
| --- | --- |
| The model ignores location words | Swap laterality/lobe and measure activation on the original GT |
| Missing global position causes anatomy errors | Compare baseline against global coordinate channels |
| Sentence pooling loses compositional detail | Compare whole-sentence embedding against role embeddings |
| Random negatives are noisy | Replace them with controlled paired negatives |
| Anatomy priors can prevent gross leakage | Apply category-specific soft gating without retraining |
| ReX-only supervision is insufficient | Add anatomical CoT/location supervision while holding the backbone fixed |

## Success Criteria

A successful redesign should improve more than aggregate Dice. It should:

- reduce prediction outside compatible anatomy,
- respond directionally to left/right and lobe edits,
- preserve predictions under faithful paraphrases,
- suppress the original lesion after pathology replacement,
- maintain or improve challenge Dice per finding,
- avoid learning an empty-mask shortcut,
- remain robust for diffuse and bilateral findings.

## Related Work

- [ReXGroundingCT](https://arxiv.org/abs/2507.22030)
- [VoxTell](https://arxiv.org/abs/2511.11450)
- [Visual Grounding of Whole Radiology Reports for 3D CT Images](https://arxiv.org/abs/2312.04794)
- [Mask Grounding](https://arxiv.org/abs/2312.12198)
- [Compositional hard negatives for vision-language learning](https://arxiv.org/abs/2306.08832)
- [Counterfactual Segmentation Reasoning](https://arxiv.org/abs/2506.21546)
- [VISTA3D](https://arxiv.org/abs/2406.05285)
- [SegVol](https://arxiv.org/abs/2311.13385)
