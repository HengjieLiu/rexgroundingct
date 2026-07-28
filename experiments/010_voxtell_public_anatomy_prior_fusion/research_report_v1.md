---
created: 2026-07-26
updated: 2026-07-26
status: research_design_v1
experiment_id: 010_voxtell_public_anatomy_prior_fusion
baseline: exp006_v123_cached_e5_d4
initialization: public_voxtell_v1_1
preprocessing_classification: intentionally_changed_by_public_anatomy_prior
---

# Experiment 010 Research Report v1

## Executive Decision

Experiment 010 should test a public model's anatomical segmentations as a
soft, learned conditioning signal for VoxTell. It should not use anatomy as a
hard crop, hard output gate, or replacement for the free-text prompt.

The recommended first implementation is a three-part **AnatomyBridge**:

1. A lightweight mask pyramid injects anatomy into multi-scale image features.
2. Anatomy tokens let the existing prompt representation attend to labeled
   regions, connecting free text to anatomy.
3. Prompt-conditioned anatomy maps guide all five existing VoxTell decoder
   scales.

Every new path should end in a zero-initialized residual projection. With all
gates at zero, the model must reproduce the public VoxTell v1.1 logits before
the first optimizer update. This gives a clean and testable starting point for
fine-tuning from the public checkpoint.

TotalSegmentator is the recommended first anatomy producer because it is
public, mature, CT-native, and already supplies lung lobes, trachea, thoracic
organs, vessels, and a separate airway task. The first experiment should use
only anatomical tasks. Disease-predicting subtasks such as nodule or pleural
effusion segmentation would turn the method into an anatomy-plus-lesion
ensemble and confound the scientific question.

The first four-arm comparison should be:

| Arm | Image prior | Text-anatomy bridge | Decoder anatomy guide |
| --- | --- | --- | --- |
| `a0_pt_v123_e5d4_control` | No | No | No |
| `a1_image_pyramid` | Yes | No | No |
| `a2_image_pyramid_anatomy_tokens` | Yes | Yes | No |
| `a3_full_anatomybridge` | Yes | Yes | Yes |

All four arms should start from public VoxTell v1.1 and otherwise copy the
experiment 006 `v123_cached_e5_d4` training contract.

## Research Question

The challenge is not ordinary categorical segmentation. A model must distinguish
different free-text findings in the same scan, including findings with the same
appearance but different side, lobe, airway, pleural, or central/peripheral
location.

The public anatomy masks can contribute three kinds of information:

- **visual context:** which voxel belongs to which anatomical region;
- **semantic context:** which named anatomical regions are compatible with the
  prompt;
- **spatial control:** where the decoder should pay more attention without
  forbidding recovery outside an imperfect prior.

The core hypothesis is:

> A predicted anatomical layout will improve free-text lesion grounding only
> when the model can bind the prompt to the relevant labeled regions. Supplying
> masks as prompt-agnostic image channels may help, but it does not by itself
> solve the language-to-anatomy binding problem.

## Exact Starting Baseline

The baseline is experiment 006 `v123_cached_e5_d4`, not an experiment 006
fine-tuned checkpoint. Experiment 010 should copy its training logic but
initialize every arm from the public VoxTell v1.1 checkpoint.

| Setting | Exp010 baseline value |
| --- | --- |
| Initialization | public VoxTell v1.1 |
| CT tensor | native-resolution `192 x 192 x 192` patch |
| CT preprocessing | orientation fix, crop-to-nonzero, full cropped-volume z-score once |
| CT cache | `crop_zscore_native_v1` |
| Text | precomputed frozen Qwen3-Embedding-4B embeddings |
| Prompt slots | 3 |
| Multi-finding case | 2 positive + 1 negative |
| One-finding case | 1 positive + 2 negative |
| Positive sampling | selected positive must be nonempty in its patch |
| Batch / DDP | batch 1, no DDP |
| Updates | 10,000, organized as 100 epochs x 100 updates |
| Optimizer | SGD Nesterov, momentum 0.99, weight decay `3e-5` |
| Encoder LR | `1e-5` |
| Decoder and new-module LR | `1e-4` |
| Schedule | 100-update warmup, polynomial decay power 0.9 |
| Gradient clipping | global norm 12 |
| Nonempty loss | Dice + weighted BCE |
| Empty loss | weighted BCE only, multiplied by 0.5 |
| BCE weights | foreground 1.0, boundary 1.5, background 0.5 |
| Boundary radius | 10 voxels |
| Deep supervision | `[1, 1/2, 1/4, 1/8, 1/16]`, normalized |

The fixed val200 result to beat is Dice `0.3241`, hit rate `0.7612`, and
`290/381` hits. The fixed val20 epoch-100 result is Dice `0.3907`, hit rate
`0.7419`, and `23/31` hits.

Local baseline evidence:

- [experiment 006 config](../../configs/experiments/006_voxtell_cached_native_v123_lr_ablation.json)
- [experiment 006 training spec](../006_voxtell_cached_native_v123_lr_ablation/training_finetuning_spec.md)
- [experiment 006 results](../006_voxtell_cached_native_v123_lr_ablation/report.md)

## What VoxTell Already Does

The local public model is a six-stage residual 3D U-Net with channels
`[32, 64, 128, 256, 320, 320]` and InstanceNorm3d. A `192^3` patch produces
feature grids at `192, 96, 48, 24, 12, and 6` voxels per axis.

The active public predictor constructs:

- one CT input channel;
- a 2,048-dimensional prompt/image query space;
- a six-layer, eight-head transformer prompt decoder;
- a 2,560-dimensional Qwen text input;
- 32 prompt-fusion channels at the decoder stages;
- image memory from the `12^3` encoder feature map;
- fixed 3D positional encoding for those `12^3 = 1,728` image tokens.

The prompt transformer first lets each text query attend to the `12^3` image
memory. Its output is projected to the channel dimensions of five decoder
scales. At each scale, VoxTell computes a channel-wise text-image dot product
and concatenates the resulting prompt-specific features into the U-Net decoder.
Deep supervision forces those interactions to matter at multiple resolutions.

Relevant local code:

- [public model](../../external/VoxTell/voxtell/model/voxtell_model.py)
- [public predictor construction](../../external/VoxTell/voxtell/inference/predictor.py)
- [current ReX training loop](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py)

This architecture is already suitable for anatomy fusion. The safest extension
is to add residual conditioning around the existing image memory and decoder
features, without replacing the pretrained CT encoder or the established
text-image dot products.

## Relation to Experiment 008

Experiment 008 is the closest local implementation precedent, but it answers a
different question.

| Property | Experiment 008 | Experiment 010 |
| --- | --- | --- |
| Source | exp006 epoch-100 fine-tuned model | public VoxTell v1.1 |
| Guide | learned prompt-specific proposal mask | fixed public anatomical masks |
| Guide semantics | possible target foreground | organs, lobes, airways, vessels, and spatial regions |
| Architecture | proposal and refinement decoders | one original decoder plus lightweight anatomy adapters |
| Goal | high-recall proposal then refinement | bind free text to stable anatomical context |

Experiment 008 already established useful engineering patterns:

- keep upstream VoxTell unchanged and use a local wrapper;
- copy all source weights strictly;
- zero-initialize new guide outputs;
- require source-equivalent logits at every scale before training;
- record an explicit model specification for checkpoint materialization and
  inference;
- log guide parameter norms and gradient norms.

Experiment 010 should reuse those patterns and relevant helper ideas from
[voxtell_dual_branch.py](../../scripts/rexgroundingct/voxtell_dual_branch.py),
but it should not add the proposal/refinement branch in v1. Combining learned
proposal masks with public anatomy masks before either mechanism is understood
would make attribution difficult. A later experiment can feed AnatomyBridge
features into experiment 008's proposal and refinement paths.

## Important Finding From the VoxTell Paper

The proposed direction is paper-consistent, but it is not already implemented
as an inference input.

The VoxTell paper used TotalSegmentator when constructing part of its
instance-focused training set. It extracted lung lobes, liver Couinaud
segments, and left/right kidney masks, then used them as contextual anchors to
convert semantic lesion masks into location-specific text prompts. In other
words, TotalSegmentator supplied anatomy for **data and prompt construction**.
The released VoxTell forward pass still receives only CT plus text; it does not
receive TotalSegmentator masks.

Experiment 010 extends that idea:

```text
VoxTell paper:
public anatomy mask -> localized training text

Experiment 010:
public anatomy mask -> image features
                    -> prompt/anatomy interaction
                    -> multiscale decoder guidance
```

This is a meaningful next step rather than a duplicate of the published
pipeline.

## Challenge Compliance

The current challenge rules allow public or private external data and
pretrained models, provided they are disclosed. They also require the exact
free-text prompt to condition segmentation and disallow a categorical method
that ignores or rewrites the prompt into a fixed class.

Experiment 010 can remain eligible for the Main track if:

- the raw finding text is passed unchanged through the existing Qwen/VoxTell
  path;
- anatomy masks are auxiliary image-derived context;
- prompt-to-anatomy attention is learned from the unchanged text embedding;
- the final prediction remains finding-specific and can separate two findings
  in different locations;
- external model name, version, weights, tasks, and data provenance are
  disclosed.

A rule-based parser that rewrites the prompt into a fixed lobe or disease class
would instead risk moving the method to the Overall track or being treated as
categorical segmentation. The first design should therefore use learned
attention from the original prompt embedding, not a prompt rewrite.

Primary source: [ReXGroundingCT challenge rules](https://rexrank.ai/ReXGroundingCT/challenge.html).

## Public Anatomy Model Choice

### Recommended: TotalSegmentator

The original TotalSegmentator paper trained on 1,204 heterogeneous CT exams and
reported 104 anatomical structures. The current public `total` task contains
117 main classes. The official repository also exposes an open
`lung_vessels` task with lung arteries, lung veins, lung airways, and airway
wall masks. The default high-resolution model operates at 1.5 mm; `--fast`
uses 3 mm. The public `total` and `lung_vessels` tasks are listed as
Apache-2.0/open-use tasks.

Advantages for ReXGroundingCT:

- native 3D CT model;
- five separate lung lobes;
- trachea, esophagus, heart, pulmonary artery, aorta, ribs, and vertebrae;
- optional airway and pulmonary-vessel masks;
- mature command line and Docker workflows;
- multilabel output and ROI-subset execution;
- already used in the VoxTell paper's instance-data construction.

Limitations:

- it segments anatomy, not the requested free-text lesion instance;
- predictions may degrade in severely abnormal lungs;
- the `total` task is assembled from multiple models, and the official
  documentation warns that saved probability output does not work well for
  that task;
- high-resolution inference over more than 3,000 cases is a material one-time
  compute cost;
- hard masks do not express model uncertainty.

For v1, use hard anatomical masks and preserve them as soft guidance through
the fusion network. Do not treat hard mask boundaries as truth.

Sources:

- [TotalSegmentator paper](https://doi.org/10.1148/ryai.230024)
- [official TotalSegmentator repository](https://github.com/wasserth/TotalSegmentator)

### Alternatives

| Model | Strength | Reason not to use first |
| --- | --- | --- |
| VISTA3D | 127 automatic classes plus interactive correction; public code and weights | broader foundation-model dependency and less direct alignment with the VoxTell paper |
| SegVol | more than 200 anatomy categories with semantic and spatial prompts | another promptable lesion/anatomy model would blur anatomy-prior fusion with model ensembling |
| SAT | 497 text-prompted classes and an anatomy knowledge tree | text-prompted output is closer to a competing grounding model than a neutral anatomy producer |
| custom lung model | could specialize in lobes, airways, pleura, or vessels | adds training and validation work before testing the central fusion hypothesis |

VISTA3D is the best second anatomy source for a model-source ablation after the
TotalSegmentator design works.

Sources:

- [VISTA3D paper](https://arxiv.org/abs/2406.05285)
- [SegVol paper](https://arxiv.org/abs/2311.13385)
- [SAT paper](https://arxiv.org/abs/2312.17183)

## Anatomy Inputs for the First Version

Do not feed all 117 TotalSegmentator classes as independent float32 volumes.
Most are irrelevant to chest findings, and one-hot expansion would waste
memory and I/O.

Use a compact thoracic ontology:

### Direct predicted masks

- left upper and lower lung lobes;
- right upper, middle, and lower lung lobes;
- trachea;
- lung airways;
- airway wall;
- lung arteries and lung veins;
- pulmonary artery;
- heart;
- aorta;
- esophagus;
- ribs grouped into left/right rib cage;
- thoracic vertebrae grouped into a spine anchor.

### Deterministic derived regions

- whole left lung and whole right lung;
- bilateral whole lung;
- lobe boundaries/fissure neighborhoods;
- inner and outer lung-boundary shells;
- subpleural band;
- central/hilar region around proximal airways and vessels;
- apical, middle, and basal normalized lung zones;
- distance-to-airway and distance-to-pleural-boundary maps.

These derived regions are useful because several challenge targets are defined
relative to anatomy rather than inside a single organ mask:

- bronchial wall thickening and bronchiectasis relate to airways;
- pleural effusion, pleural thickening, and pneumothorax relate to a pleural
  or thoracic boundary region;
- nodules and ground-glass opacities may be peripheral, central, or
  lobe-specific;
- diffuse abnormalities may span multiple lobes or both lungs.

Distance maps should be clipped to a fixed physical range and normalized.
Distances must use voxel spacing in millimeters, not a fixed number of native
voxels.

### Explicit exclusions

Do not include public-model outputs for:

- lung nodules;
- pleural effusion;
- pericardial effusion;
- pneumonia or opacity;
- any other target-like pathology.

Those could be studied later as a disclosed teacher/ensemble experiment, but
they must not be mixed into the first anatomy-prior result.

## Fusion Alternatives

| Method | Benefit | Main weakness | Recommendation |
| --- | --- | --- | --- |
| Hard crop or hard mask intersection | almost no model change | irreversible prior misses; poor for pleural and diffuse disease | reject for v1 |
| Anatomy-aware postprocessing | quick no-training diagnostic | prompt parsing and hard heuristics can violate the scientific question | diagnostic only |
| Concatenate masks to CT input | simple and common | changes first convolution; prompt-agnostic; expensive with many channels | useful minimal baseline |
| Separate prior encoder into U-Net skips | preserves CT encoder and provides multi-scale spatial context | still prompt-agnostic | recommended image path |
| Anatomy tokens after prompt fusion | directly binds prompt semantics to named regions | token construction and missing-mask handling must be careful | recommended text path |
| Prompt-conditioned decoder guide | strongest direct spatial-language interaction | easiest place to overtrust noisy masks | recommended with zero residuals |
| Full new cross-attention at every voxel and scale | highly expressive | large memory and implementation burden | defer |
| Auxiliary anatomy segmentation task | may regularize image features | TotalSegmentator predictions are pseudo labels, not ground truth | later ablation |

Adding previous segmentation maps as inputs is a standard coarse-to-fine
strategy in medical segmentation. nnU-Net's 3D cascade refines low-resolution
predictions at full resolution. MA-SAM similarly uses pseudo-mask prompts,
learns anatomy features in a prompt encoder, and concatenates multi-scale prompt
features into decoder upsampling blocks. These support using a mask pyramid,
but neither solves this project's free-text-to-anatomy binding by itself.

Sources:

- [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z)
- [MA-SAM](https://pubmed.ncbi.nlm.nih.gov/40030770/)

## Recommended AnatomyBridge Architecture

### Overview

```text
raw CT
  |-- TotalSegmentator on raw HU CT -> predicted anatomy in source geometry
  |
  `-- VoxTell preprocessing -> normalized CT patch --------------------.
                                                                       |
predicted anatomy -> same orientation/crop/patch -> compact masks      |
                         |                                             |
                         +-> anatomy pyramid -> zero residuals -> image skips
                         |
prompt -> frozen Qwen -> base VoxTell prompt/image transformer -> m0
   |                     anatomy tokens -> gated cross-attention -> m
   |                                                                   |
   `-> prompt/anatomy compatibility -> spatial guide maps -------------+
                                                                       |
                            existing five-scale VoxTell decoder
                            + zero residual anatomy guides
                                                                       |
                            prompt-specific lesion mask
```

The CT and anatomy branches must use the exact same source affine,
orientation transform, crop bounding box, and patch start.

### 1. Image-path anatomy pyramid

Let `M` be the compact anatomy tensor for one patch. A small 3D convolutional
encoder creates a pyramid `P_s` aligned to VoxTell's six image scales.

At scale `s`:

```text
z_s_conditioned = z_s + ZeroConv_s(P_s)
```

`ZeroConv_s` is a `1 x 1 x 1` convolution with zero weights and zero bias.
Therefore, before training:

```text
z_s_conditioned == z_s
```

This path gives both the U-Net decoder and the prompt transformer's selected
`12^3` memory access to anatomical layout. It is preferable to changing the
first CT convolution because the pretrained CT pathway remains structurally
unchanged.

### 2. Text-path anatomy tokens

For each available anatomy label `k`, construct a token from:

- a masked average of the selected `12^3` image feature;
- the same Qwen embedding family applied once to the canonical anatomy name;
- region geometry such as centroid, volume, side, and validity.

Conceptually:

```text
t_k = LayerNorm(
    W_visual * MaskedPool(z_4, M_k)
  + W_label  * Qwen(anatomy_name_k)
  + W_geom   * geometry_k
)
```

The existing VoxTell prompt/image transformer produces `m0`. A small
cross-attention adapter then lets `m0` attend to the anatomy tokens:

```text
delta_m = CrossAttention(query=m0, keys=t_k, values=t_k)
m = m0 + ZeroLinear(delta_m)
```

This should be a residual adapter after the existing transformer, not anatomy
tokens appended directly to the pretrained transformer memory. Direct
appending would change attention normalization and therefore change the public
checkpoint output at initialization. `ZeroLinear` preserves exact initial
behavior.

This design is supported by evidence that semantic label embeddings can encode
relationships among organs and lesions. The CLIP-Driven Universal Model found
that text-derived label embeddings produced more anatomically structured
features than one-hot labels. Experiment 010 uses Qwen instead of CLIP so its
anatomy labels and finding prompts live in the same embedding family.

Source: [CLIP-Driven Universal Model](https://arxiv.org/abs/2301.00785).

### 3. Prompt-conditioned multiscale decoder guide

For prompt `q` and anatomy label embedding `e_k`, predict independent
compatibility weights:

```text
c_k = sigmoid(similarity(W_q q, W_e e_k) / temperature + bias_k)
A_q(x) = sum_k c_k * SoftRegion_k(x)
```

Use sigmoid rather than softmax because a finding may be compatible with
several nested or bilateral regions. Include a learned null/unknown option.

At each decoder scale:

```text
G_qs = PriorAdapter_s(P_s, downsample(A_q))
h_s_conditioned = h_s + ZeroConvGuide_s(G_qs)
```

The existing VoxTell text-image dot product then operates on
`h_s_conditioned`. The guide must never multiply or clip the final logit.
VoxTell can learn to use, ignore, or correct it.

This is related to:

- FiLM, which conditions features through learned affine modulation;
- SPADE, which preserves spatial semantic layouts through spatially varying
  modulation;
- ControlNet, which uses zero-initialized residual connections to add spatial
  control to a pretrained model without perturbing its initial output;
- recent localization-infused text segmentation, which combines localization
  features, gated attention, and multi-scale supervision.

Sources:

- [FiLM](https://arxiv.org/abs/1709.07871)
- [SPADE](https://arxiv.org/abs/1903.07291)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Localization-Infused Vision-Language Semantic Fusion](https://arxiv.org/abs/2607.16327)

The last source is a July 2026 arXiv v1 tested on 2D-oriented benchmarks, so it
is directional evidence, not direct validation for 3D ReXGroundingCT.

## Why Zero Initialization Is Essential

Experiment 010 starts from the public model, but its new modules have no
pretrained counterpart. Randomly injecting mask features would cause an
immediate output shift and make a failed five-epoch result ambiguous.

The required epoch-0 invariant is:

```text
VoxTellWithAnatomy(CT, text, any_valid_mask)
    == PublicVoxTell(CT, text)
```

within a strict numerical tolerance, at all five deep-supervision scales.

Achieve this with:

- zero output convolutions on image-pyramid residuals;
- zero output projection on anatomy-token cross-attention;
- zero output convolutions on decoder guides;
- copied public weights for every original VoxTell parameter;
- no replacement of the original transformer attention or decoder blocks.

New modules may have ordinary internal initialization because their final
projection is zero. Log their parameter and gradient norms to confirm they
begin learning after the first updates.

## Data and Cache Contract

TotalSegmentator must run on the original HU-valued CT, before VoxTell z-score
normalization. Running it on normalized cached arrays would violate its input
contract.

### Val200 inventory amendment

Before building the final high-resolution thorax cache, exp010 now runs the
3 mm `total_fast` task over all 200 fixed validation cases. This first cache is
an ontology and visual-QC audit, not the final AnatomyBridge training input:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/
  totalsegmentator_total_fast_3mm_v2_16_0/
```

Persistent package weights live separately at:

```text
/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0/
```

The all-label audit establishes which of the 117 configured structures
actually survive the chest CT field of view. It also measures direct overlap
with all 381 validation findings. The selected thoracic subset is then checked
at 1.5 mm on 20 fixed cases before train/validation-wide high-resolution cache
generation. This ordering avoids generating dozens of high-resolution pelvic,
head, and abdominal channels that are not useful for ReX grounding.

Recommended external cache:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/
  totalsegmentator_thorax_v1/
    manifest.json
    .complete
    cases/<case_key>/
      total_labels.npy
      thorax_aux_bits.npy
      metadata.json
      .complete
```

Storage principles:

- keep TotalSegmentator's official multilabel `total` output in packed `uint8`;
- keep overlapping airway/vessel/derived masks in a packed bit field, not
  separate float32 volumes;
- create one-hot mask channels only when loading a sampled patch;
- compute distance features once over the complete aligned cropped volume,
  clip them to a documented physical range, quantize them compactly, and then
  crop them with the CT patch;
- never calculate a distance transform independently inside a sampled patch,
  because the artificial patch boundary would produce incorrect distances;
- never store heavy masks in Git or in the exp010 repo-local folder.

Required manifest fields:

- TotalSegmentator package/container version;
- exact task and weight identifiers;
- package and weight hashes when available;
- full class map and selected class list;
- source CT path and hash;
- source and output shapes, spacings, affines, and axis codes;
- interpolation and resampling policy;
- source-to-VoxTell orientation and crop metadata;
- per-class voxel counts;
- empty/missing classes and inference failures;
- case-level content hashes;
- elapsed time and peak GPU memory;
- train/val/test case counts and completion markers.

Geometry rules:

1. Run public anatomy inference in source CT geometry.
2. Verify output shape and affine against the source CT.
3. Apply the same validated LPS/RAS orientation handling used by exp001 and the
   native VoxTell cache.
4. Apply the exact VoxTell crop-to-nonzero bounding box.
5. Use nearest-neighbor interpolation for categorical masks.
6. Use physical spacing for distance transforms.
7. Sample CT, target, and anatomy with the same patch start.
8. Restore only the lesion prediction to evaluator space; anatomy masks remain
   auxiliary inputs.

Before full generation, benchmark standard 1.5 mm and `--fast` 3 mm inference
on 20 representative scans. The scientific default should be 1.5 mm with a
thoracic ROI subset unless the runtime audit shows that 3 mm has equivalent
lobe/airway containment for this task.

## Robustness to Imperfect Public Masks

Predicted anatomy is a noisy modality. Its errors may correlate with severe
pathology, which is exactly where lesion grounding is difficult.

The model must not assume every mask is present or correct:

- add one validity indicator per mask family;
- use whole-prior dropout on approximately 10% of training patches;
- use class-group dropout on another small fraction;
- optionally perturb boundaries by small physical dilations/erosions;
- keep the original CT path fully available;
- use soft residual guidance rather than hard gating;
- permit a null/unknown anatomy token;
- log missing-mask and prior-dropout behavior.

The first controlled run should include whole-prior dropout but avoid many
simultaneous perturbations. Morphological noise can be a later ablation.

## Training Plan

### Shared settings

All four arms:

- start from the identical public VoxTell v1.1 checkpoint;
- use the same `crop_zscore_native_v1` CT cache;
- use the same predicted TotalSegmentator cache, even if the control ignores it;
- consume the same 10,000-event experiment 006 schedule;
- use the experiment 006 `v123_cached_e5_d4` optimizer, loss, and LR schedule;
- use batch 1, no DDP, and `100` optimizer updates per epoch;
- evaluate fixed val20 at epochs `5/20/40/60/80/100`;
- evaluate fixed val200 at epoch 100;
- save probability maps at val200 for future ensembles.

New anatomy modules belong to the non-encoder LR group at `1e-4`. The original
image encoder remains at `1e-5`. Do not freeze the original model in the main
comparison, because that would no longer match the strongest exp006 recipe.

### Four-arm matrix

| Arm | Purpose | New trainable paths |
| --- | --- | --- |
| `a0_pt_v123_e5d4_control` | exact no-anatomy reference under the exp010 loader | none |
| `a1_image_pyramid` | test whether spatial anatomy helps visual features without text binding | prior encoder and zero skip adapters |
| `a2_image_pyramid_anatomy_tokens` | test prompt-to-named-region interaction | a1 plus anatomy token and gated cross-attention adapter |
| `a3_full_anatomybridge` | test direct prompt-specific multiscale spatial guidance | a2 plus compatibility router and five decoder guide adapters |

This cumulative design answers:

1. Do anatomy masks help at all?
2. Does explicitly connecting named anatomy to the prompt add value?
3. Does spatial prompt-conditioned guidance at every decoder scale add value?

The existing exp006 result remains an external reference, but the exp010
control should still be run. It catches changes caused by the new cache loader,
model wrapper, data transfer, or dependency environment.

### Loss

Keep the v123 lesion loss unchanged in v1. Do not add anatomy pseudo-label loss
or compatibility loss to the first four-arm comparison.

The anatomy router can begin with frozen Qwen cosine similarities and learn a
zero-initialized residual correction from lesion loss. A later experiment can
add soft compatibility supervision derived from GT/anatomy overlap:

```text
r_k = |GT intersect dilate(M_k)| / |GT|
```

That auxiliary is promising but would make the first result harder to
attribute.

## Smoke and Acceptance Gates

### Cache gate

- raw CT, TotalSegmentator output, native cache, and target have consistent
  case identity;
- shape, affine, orientation, crop, and patch alignment pass;
- all five lobes and major thoracic structures have plausible voxel counts;
- missing masks are represented explicitly;
- no lesion ground truth enters the public-anatomy cache;
- no lesion-predicting TotalSegmentator task is present.

### Model gate

- all four models load public VoxTell v1.1 strictly;
- a0 matches public VoxTell at all scales;
- a1/a2/a3 match a0 before training for both valid masks and all-zero masks;
- loss is finite for positive and negative prompts;
- new adapter gradients become nonzero after one update;
- selected positive targets remain nonempty;
- peak memory fits batch 1 with margin;
- one-case evaluator-space export matches the established orientation path.

### Early scientific gate

At epoch 5, report rather than automatically terminate:

- val20 Dice and hit rate;
- anatomy-router weights for representative prompts;
- prediction volume inside/outside lungs, lobes, airway neighborhoods, and
  pleural shells;
- correct-prior, zero-prior, and shuffled-prior inference on a small probe.

Shuffled-prior inference is important. If a model performs identically with
another patient's anatomy, it is probably ignoring the new input. If it
collapses completely, it may be overdependent on the prior.

## Evaluation Beyond Aggregate Dice

Primary challenge metrics remain mean Dice per finding and hit rate.
Experiment 010 also needs mechanism-specific metrics.

### Prompt and anatomy strata

- laterality present versus absent;
- lobe/segment named versus not named;
- airway-related findings;
- pleural-related findings;
- diffuse/bilateral findings;
- focal nodules and masses;
- small/sparse masks;
- categories that lost performance in exp006, including pleural findings and
  pneumothorax.

### Spatial compliance

- GT fraction contained in each candidate anatomy region;
- prediction fraction outside whole lung or a relevant soft region;
- lobe/laterality compliance when explicitly stated;
- distance of predicted voxels to airway or pleural surfaces;
- false-positive volume around heart/mediastinum for lung prompts;
- centroid change relative to the no-prior control.

### Prior dependence

Evaluate the same checkpoint with:

- correct prior;
- all-zero prior plus invalid flag;
- same-case prior with one class group removed;
- shuffled prior from another case.

### Interpretation requirement

An improvement in val20 alone is not sufficient. The method is promising only
if val200 improves and at least one anatomy-sensitive stratum improves without
material collapse in diffuse, pleural, or small-lesion strata.

## Expected Outcomes

### Most likely

The image-only pyramid produces a modest improvement in localization and fewer
gross extra-pulmonary false positives. Text-anatomy tokens and decoder guides
produce additional gains for prompts with explicit side/lobe/airway language.

### Plausible failure

The model overfits to predicted lobes and suppresses pleural or boundary
findings. This would appear as better nodule localization but worse pleural
effusion, pleural thickening, pneumothorax, or diffuse disease.

### Another plausible failure

The public model's lobe and airway errors are correlated with severe disease,
so the anatomy branch is least trustworthy on the hardest cases. Prior dropout,
soft residuals, and zero/incorrect-prior tests are designed to expose this.

### What anatomy will not solve

Anatomy masks do not identify a 3 mm nodule, distinguish ground glass from
consolidation, or resolve incomplete ReX training labels. They mainly provide
global location, named-region identity, and structural context. Appearance
recognition and instance separation still depend on VoxTell and ReX
fine-tuning.

## Implementation Touchpoints

No code is implemented in this report. A later execution spec should prefer
new exp010 modules rather than modifying the upstream VoxTell submodule.

Likely additions:

```text
scripts/rexgroundingct/prepare_totalsegmentator_thorax_cache.py
scripts/rexgroundingct/voxtell_anatomy_bridge.py
scripts/rexgroundingct/test_voxtell_anatomy_bridge.py
scripts/rexgroundingct/run_010_anatomy_prior_fusion.sh
scripts/rexgroundingct/summarize_010_results.py
configs/experiments/010_voxtell_public_anatomy_prior_fusion.json
```

Likely minimal edits:

- training loader: return aligned anatomy patches;
- model creation: wrap public VoxTell with AnatomyBridge;
- optimizer grouping: classify original encoder, original non-encoder, and new
  anatomy parameters explicitly;
- checkpoint/model spec: record architecture variant and anatomy ontology hash;
- inference: load the same anatomy cache and support prior-ablation modes;
- reporting: add anatomy compliance and prior-dependence metrics.

Do not alter `external/VoxTell` unless an upstream bug is found. The wrapper
should preserve strict loading of the public checkpoint and materialize an
explicit model specification for inference.

## Recommended Work Order

1. Pin TotalSegmentator version, weights, open tasks, and class ontology.
2. Run a 20-case raw-HU anatomy inference audit.
3. Measure GT containment and baseline prediction leakage against the proposed
   anatomy regions before training.
4. Compare 1.5 mm and 3 mm public anatomy output on the same 20 cases.
5. Build the packed anatomy cache with geometry and hash audits.
6. Implement a1 and prove exact epoch-0 equivalence.
7. Implement a2 and a3 with the same equivalence gate.
8. Run one-update and one-case inference smokes.
9. Launch the four controlled 100-epoch arms from public VoxTell v1.1.
10. Evaluate val20 milestones and final val200 with prior-dependence tests.

The no-training containment audit is important. If TotalSegmentator's proposed
regions fail to include many ReX targets, architecture work should pause until
the ontology and derived regions are corrected.

## Final Recommendation

Proceed with TotalSegmentator as a **soft anatomical coordinate system**, not
as a lesion detector and not as a hard segmentation constraint.

The highest-value design is the full AnatomyBridge because it answers the
user's central question at all three interfaces:

- **image features** receive a multi-scale anatomical pyramid;
- **text features** attend to named, image-grounded anatomy tokens;
- **decoder features** receive prompt-specific spatial anatomy guidance at
  every existing VoxTell fusion scale.

The architecture should be introduced cumulatively and initialized to the exact
public VoxTell function. That combination gives the method enough expressive
power to be interesting while preserving a clean causal comparison with
experiment 006.

## Primary References

- Rokuss et al., [VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation](https://arxiv.org/abs/2511.11450).
- Wasserthal et al., [TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images](https://doi.org/10.1148/ryai.230024).
- Isensee et al., [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z).
- Liu et al., [CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection](https://arxiv.org/abs/2301.00785).
- He et al., [VISTA3D: A Unified Segmentation Foundation Model for 3D Medical Imaging](https://arxiv.org/abs/2406.05285).
- Fan et al., [MA-SAM: A Multi-Atlas Guided SAM Using Pseudo Mask Prompts Without Manual Annotation](https://pubmed.ncbi.nlm.nih.gov/40030770/).
- Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871).
- Park et al., [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291).
- Zhang et al., [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543).
- Han et al., [Localization-Infused Vision-Language Semantic Fusion for Text-Guided Medical Image Segmentation](https://arxiv.org/abs/2607.16327).
