---
created: 2026-07-26
updated: 2026-07-26
status: active
doc_type: experiment architecture and training reference
experiment_id: 008_voxtell_dual_branch_proposal_refinement_ablation
run_group: exp008_dual_branch_20260726T182647Z
---

# Experiment 008 Variant Reference

## Purpose

This document is the exact side-by-side reference for the four
proposal/refinement variants in experiment 008:

- `v1_sharedfusion_softguide`;
- `v1_dualfusion_softguide`;
- `v2_dualfusion_precision`;
- `v3_dualfusion_softguide_joint`.

It records the implemented model and data flow, equations, gradient paths,
initialization, loss, training settings, validation contract, and current
interim evidence. It is intended both as an experiment record and as a stable
starting point for future architecture and loss brainstorming.

Use the companion
[`asymmetric_proposal_refinement.md`](asymmetric_proposal_refinement.md) for the
motivation, partial-annotation discussion, literature, and broader future
directions. The machine-readable config and execution spec remain authoritative
for paths and launch behavior.

## Executive Comparison

All four variants have:

- one shared VoxTell image encoder and base text-image transformer;
- one proposal decoder and one refinement decoder;
- full-image refinement features;
- soft, additive proposal guidance at every refinement scale;
- a recall-biased proposal objective;
- the original v123 segmentation objective at full weight for the final mask.

They differ in only three intended method axes. The Exp006 source is included
as the reference architecture:

| Model | Fusion after shared transformer | Proposal-to-final gradient | Extra final precision loss | Parameters |
| --- | --- | --- | --- | ---: |
| Exp006 source single branch | One projector/decoder path | Not applicable | No | 429,797,381 |
| `v1_sharedfusion_softguide` | Same tokens and projectors for both branches | Detached | No | 452,206,570 |
| `v1_dualfusion_softguide` | Branch residual adapters and separate projectors | Detached | No | 527,821,578 |
| `v2_dualfusion_precision` | Branch residual adapters and separate projectors | Detached | Yes, weight `0.1` | 527,821,578 |
| `v3_dualfusion_softguide_joint` | Branch residual adapters and separate projectors | Joint | No | 527,821,578 |

The experiment asks:

1. Does branch-specific fusion adaptation help over shared fusion?
2. Does a modest explicit final precision term help or hurt?
3. Should final-loss gradients be allowed to reshape the proposal path?

The phrase "dual fusion" in experiment 008 means lightweight adaptation after
one shared transformer. It does not mean two complete transformer trunks.

## Common Data Flow

### Training preprocessing and sampling

```text
Raw CT NIfTI + ReX target masks + finding text
                  |
                  v
Validated image/target orientation handling
                  |
                  v
Crop complete image and masks to the nonzero image region
                  |
                  v
Z-score the complete cropped CT volume once
                  |
                  v
Standard native cache: crop_zscore_native_v1
                  |
                  v
Deterministic schedule event
  - case
  - prompt slots
  - positive/negative choices
  - foreground-positive patch seed/start
                  |
                  v
Native-resolution 192 x 192 x 192 image patch
+ three prompt embeddings
+ three target-mask channels
                  |
                  v
One experiment 008 model variant
                  |
                  v
Proposal logits at five scales + final logits at five scales
                  |
                  v
Composite proposal/final loss -> one optimizer update
```

Normalization is not performed independently on each patch. The complete
cropped volume is normalized once, and patches are sampled from that normalized
volume. This preserves the established VoxTell input dynamics.

The training prompt policy is:

- multi-finding case: two positive prompts and one negative prompt;
- one-finding case: one positive prompt and two negative prompts;
- every selected positive target must be nonempty inside its sampled patch;
- negative targets are empty masks.

Three channels are a training sampling policy, not a fixed inference
requirement. At validation, a model receives the prompts defined for the case
and returns one output channel per requested finding.

### Validation and export

```text
Cached, fully preprocessed validation volume + case prompts
                         |
                         v
Overlapping 192^3 sliding windows, inference window batch size 1
                         |
                         v
Proposal and final probabilities stitched in preprocessed space
                         |
                         v
Inverse crop/orientation geometry
                         |
                         v
Native evaluator-space 4D NIfTI
  - final mask at threshold 0.5
  - proposal masks at thresholds 0.1, 0.3, and 0.5
                         |
                         v
ReX mean global Dice per finding, hit rate, and proposal diagnostics
```

Inference window batch sizes 2 and 4 were tested but were slower than batch 1,
so all milestone evaluations use batch 1.

## Common Model Symbols

The diagrams below use:

```text
S_s       encoder skip feature at spatial scale s
M         projected bottleneck image-memory volume
m0        base prompt token after shared text-image fusion
A_prop    proposal residual fusion adapter
A_ref     refinement residual fusion adapter
Q_prop_s  proposal projector for decoder scale s
Q_ref_s   refinement projector for decoder scale s
P_s       proposal logit map at scale s
p_s       sigmoid(P_s), the proposal probability map
F_s       final/refinement logit map at scale s
G_s       learned 1x1x1 spatial guide adapter at scale s
SG(x)     stop-gradient/detach operation
c_prop    proposal-weighted bottleneck image context
```

The shared base fusion is:

```text
S = Encoder(image)
M = ProjectBottleneck(S_bottleneck)
t = ProjectText(prompt_embedding)
m0 = TextImageTransformer(t, M, positional_encoding)
```

The expensive components shared by every branch are:

- image encoder;
- bottleneck projection;
- text projection;
- six-layer text-image transformer;
- positional encoding.

The source segmentation decoder is copied into a proposal decoder and a
refinement decoder. The refinement copy is extended with one soft-guide adapter
per scale.

## Shared Soft Spatial Guide

Every variant provides proposal guidance at each refinement scale:

```text
p_s = sigmoid(P_s)
guide_input_s = SG(p_s)       for detached variants
guide_input_s = p_s           for the joint variant
guide_s = G_s(guide_input_s)
x_ref_s = x_ref_s + guide_s
```

Each `G_s` is a `1x1x1` convolution initialized with zero weights and zero
bias. The proposal is never used as a hard mask, image crop, multiplication
gate, or irreversible region restriction. The refinement decoder retains all
encoder skips and can recover proposal misses.

## Reference: Exp006 Single-Branch Source

Name: `v123_cached_e5_d4`, epoch 100

Role: initialization checkpoint and epoch-0 performance reference

### Structure

```text
Image patch -> image encoder --------------------------> skips S_s
                    |
                    +-> bottleneck projection -> memory M
                                                   |
Prompt embedding -> text projection                 |
                    |                                |
                    +---- text-image transformer <---+
                                  |
                                  v
                                 m0
                                  |
                         projector set Q_s
                                  |
                                  v
                     one segmentation decoder
                                  |
                                  v
                         final logits F_s
```

The source has no proposal branch, proposal loss, guide adapters, proposal
context pooling, or branch-specific residual fusion adapters.

Its loss is:

```text
L_source = L_v123(F, Y)
```

The source checkpoint was produced in experiment 006 by fine-tuning public
VoxTell v1.1 for 10,000 batch-1 optimizer updates on the standard cached native
data with:

```text
encoder LR     = 1e-5
non-encoder LR = 1e-4
optimizer      = SGD Nesterov, momentum 0.99
schedule       = 100-update warmup + poly decay, power 0.9
loss           = v123
schedule       = deterministic events 0-9999
```

Experiment 008 begins from these trained weights, but resets all optimizer and
schedule state and consumes continuation events `10000-19999`.

This source checkpoint is not the missing continuation control. A proper
control would continue this single-branch architecture through the same
experiment 008 events and optimization horizon.

## Variant 1A: Shared Fusion, Detached Soft Guide

Name: `v1_sharedfusion_softguide`

GPU assignment: GPU 0

### Structure

```text
Image patch -> shared encoder -----------------------> skips S_s
                    |
                    +-> bottleneck memory M
                                  |
Prompt embedding -> text projection                 |
                    |                                |
                    +---- shared transformer <-------+
                                  |
                                  v
                                 m0
                                  |
                    shared projector set Q_s
                       /                    \
                      v                      v
             proposal decoder       refinement decoder
                      |               ^       |
                      v               |       v
                  P_s -> sigmoid -> SG -> G_s F_s
```

Both branches receive the same `m0` and the same per-scale projected prompt
embeddings:

```text
m_prop = m_ref = m0
e_prop_s = e_ref_s = Q_s(m0)
```

There is no proposal-weighted bottleneck context and there are no branch fusion
adapters. The only direct proposal-to-refinement connection is the detached
multiscale spatial guide.

### Gradient behavior

```text
proposal loss -> proposal decoder
proposal loss -> shared projectors, transformer, and encoder

final loss -> refinement decoder and guide adapters
final loss -> shared projectors, transformer, and encoder
final loss -X-> proposal decoder through the detached guide
```

This is the lowest-capacity dual-branch arm, but the proposal and final tasks
can still interfere through the shared projectors and shared backbone.

### Question answered

Does adding a separately supervised proposal decoder and detached soft spatial
guidance help when text-image fusion remains completely shared?

## Variant 1B: Dual Fusion, Detached Soft Guide

Name: `v1_dualfusion_softguide`

GPU assignment: GPU 1

### Structure

```text
Image patch -> shared encoder ----------------------------> skips S_s
                    |
                    +-> bottleneck memory M
                                  |
Prompt embedding -> text projection                       |
                    |                                      |
                    +------ shared transformer <------------+
                                  |
                                  v
                                 m0
                         /                  \
                        v                    v
       m_prop = m0 + A_prop(m0)       proposal decoded first
                        |                    |
                 proposal Q_prop_s          v
                        |                   P_s
                        v                    |
               proposal decoder             +-> sigmoid(P_low)
                        |                              |
                        +-------- P_s                  v
                                  |          weighted pool over M
                                  |                    |
                                  |                   SG
                                  |                    |
                                  |                 c_prop
                                  |                    |
                                  |       concat(m0, c_prop)
                                  |                    |
                                  |   m_ref = m0 + A_ref(...)
                                  |                    |
                                  |              Q_ref_s
                                  |                    |
                                  +-> SG -> G_s -> refinement decoder
                                                       |
                                                       v
                                                      F_s
```

The proposal token path is:

```text
m_prop = m0 + A_prop(m0)
e_prop_s = Q_prop_s(m_prop)
P_s = ProposalDecoder(S_s, e_prop_s)
```

The lowest-resolution proposal probability performs prompt-specific weighted
pooling over image memory:

```text
p_low = sigmoid(P_low)
c_prop = sum(SG(p_low) * M) / (sum(SG(p_low)) + epsilon)
```

The refinement token path is:

```text
m_ref = m0 + A_ref(concat(m0, SG(c_prop)))
e_ref_s = Q_ref_s(m_ref)
F_s = RefinementDecoder(S_s, e_ref_s, G_s(SG(p_s)))
```

`A_prop` and `A_ref` are residual MLP adapters. The proposal and refinement
projectors are separate trainable copies. The base transformer remains shared.

### Gradient behavior

```text
proposal loss -> A_prop, Q_prop_s, and proposal decoder
proposal loss -> shared transformer and encoder

final loss -> A_ref, Q_ref_s, refinement decoder, and G_s
final loss -> shared transformer and encoder
final loss -X-> A_prop, Q_prop_s, or proposal decoder through guidance/context
```

The detach is applied to both proposal-dependent routes:

- multiscale spatial probability guides;
- proposal-weighted bottleneck context.

### Question answered

Does allowing proposal and refinement tokens/projectors to specialize improve
upon the fully shared fusion arm while preserving a recall-oriented proposal
that is insulated from final-loss gradients?

## Variant 2: Dual Fusion, Detached Guide, Precision Auxiliary

Name: `v2_dualfusion_precision`

GPU assignment: GPU 2

### Structure

The model and gradient flow are identical to
`v1_dualfusion_softguide`:

```text
shared encoder/transformer -> m0
         |
         +-> A_prop -> Q_prop_s -> proposal decoder -> P_s
         |
         +-> [m0, SG(proposal context)] -> A_ref -> Q_ref_s
                                                    |
SG(P_s) -> multiscale guide adapters --------------+
                                                    |
                                            refinement decoder
                                                    |
                                                    v
                                                   F_s
```

The only intended difference is an additional modest precision-oriented
Tversky loss on the full-resolution final prediction.

### Gradient behavior

The proposal guide and context remain detached. The final precision auxiliary
updates the final path and shared backbone, but it does not pass through the
detached proposal outputs.

### Question answered

Once the proposal is recall-biased, does a small explicit false-positive
penalty improve final refinement, or does it suppress useful predictions under
ReXGroundingCT's partial-instance training annotations?

## Variant 3: Dual Fusion, Joint Soft Guide

Name: `v3_dualfusion_softguide_joint`

GPU assignment: GPU 3

### Structure

The modules are the same as the detached dual-fusion variants, but `SG` is
removed from both proposal-dependent paths:

```text
Image/text -> shared encoder and transformer -> m0
                         |
                         +-> A_prop -> Q_prop_s -> proposal decoder -> P_s
                         |                                      |       |
                         |                                      |       v
                         |                                      |  sigmoid(P_s)
                         |                                      |       |
                         |                       weighted pool M |       |
                         |                                      v       v
                         +-> concat(m0, c_prop) -> A_ref     G_s(p_s)
                                                       \       /
                                                        v     v
                                                  refinement decoder
                                                          |
                                                          v
                                                         F_s
```

Equations:

```text
m_prop = m0 + A_prop(m0)
P_s = ProposalDecoder(S_s, Q_prop_s(m_prop))

c_prop = sum(sigmoid(P_low) * M) / (sum(sigmoid(P_low)) + epsilon)
m_ref = m0 + A_ref(concat(m0, c_prop))
F_s = RefinementDecoder(S_s, Q_ref_s(m_ref), G_s(sigmoid(P_s)))
```

### Gradient behavior

```text
proposal loss -> proposal path and shared backbone
final loss -> refinement path and shared backbone
final loss -> proposal path through spatial guides
final loss -> proposal path through proposal-weighted context
```

At exact initialization, zero-initialized guide and residual outputs can block
some final-to-proposal gradients temporarily. Those paths become active as the
new adapters learn nonzero weights.

### Question answered

Does end-to-end cooperation between proposal and refinement improve the final
mask, or does final supervision pull the proposal away from its intended
high-recall role?

## Loss Definitions

### Base v123 segmentation loss

Let `L_v123(Z, Y)` denote the established experiment 003/006 v123 loss across
five prediction scales.

For a nonempty target:

```text
L_seg(Z, Y) = DiceLoss(Z, Y) + WeightedBCE(Z, Y)
```

For an empty target:

```text
L_seg(Z, empty) = 0.5 * WeightedBCE(Z, empty)
```

The BCE voxel weights are:

```text
foreground = 1.0
boundary   = 1.5
background = 0.5
boundary radius = 10 voxels
```

Dice is not voxel-weighted.

The raw deep-supervision weights are:

```text
[1, 1/2, 1/4, 1/8, 1/16]
```

After normalization, they are:

```text
[16/31, 8/31, 4/31, 2/31, 1/31]
```

Therefore:

```text
L_v123(Z, Y) = sum_s w_s * L_seg(Z_s, resize_nearest(Y, shape(Z_s)))
```

Both proposal and final branches receive this complete five-scale v123 loss.

### Asymmetric Tversky auxiliaries

For soft probabilities and one target:

```text
Tversky(alpha, beta)
    = (TP + epsilon)
      / (TP + alpha * FP + beta * FN + epsilon)

L_Tversky(alpha, beta) = 1 - Tversky(alpha, beta)
```

The proposal recall auxiliary is:

```text
L_recall = L_Tversky(alpha=0.3, beta=0.7)
```

The final precision auxiliary is:

```text
L_precision = L_Tversky(alpha=0.7, beta=0.3)
```

These auxiliaries:

- apply only to nonempty prompt targets;
- apply only to the full-resolution output;
- return zero for a batch containing only empty targets.

The proposal loss is common to all four variants:

```text
L_proposal = L_v123(P, Y) + 0.5 * L_recall(P_full, Y)
```

### Exact per-variant total losses

For `v1_sharedfusion_softguide`:

```text
L_total
    = L_v123(F, Y) + 0.5 * L_proposal
    = L_v123(F, Y)
      + 0.5 * L_v123(P, Y)
      + 0.25 * L_recall(P_full, Y)
```

For `v1_dualfusion_softguide`:

```text
L_total
    = L_v123(F, Y)
      + 0.5 * L_v123(P, Y)
      + 0.25 * L_recall(P_full, Y)
```

For `v2_dualfusion_precision`:

```text
L_total
    = L_v123(F, Y)
      + 0.5 * L_v123(P, Y)
      + 0.25 * L_recall(P_full, Y)
      + 0.1 * L_precision(F_full, Y)
```

For `v3_dualfusion_softguide_joint`:

```text
L_total
    = L_v123(F, Y)
      + 0.5 * L_v123(P, Y)
      + 0.25 * L_recall(P_full, Y)
```

Thus v1 dual and v3 have the same scalar objective. Their difference is
gradient connectivity, not loss coefficients.

## Gradient Ownership Summary

| Component | Shared fusion detached | Dual fusion detached | Dual fusion joint |
| --- | --- | --- | --- |
| Shared encoder and transformer | Proposal + final losses | Proposal + final losses | Proposal + final losses |
| Shared projectors | Proposal + final losses | Not present | Not present |
| `A_prop` and proposal projectors | Not present | Proposal loss only | Proposal loss + final loss through guides/context |
| Proposal decoder | Proposal loss only | Proposal loss only | Proposal loss + final loss through guides/context |
| `A_ref` and refinement projectors | Not present/separate | Final loss | Final loss |
| Refinement decoder | Final loss | Final loss | Final loss |
| Spatial guide adapters `G_s` | Final loss | Final loss | Final loss |
| Proposal values used by refinement | Detached | Detached | Differentiable |

Detached guidance does not make the two tasks fully independent because both
losses still update the shared encoder and transformer.

## Initialization And Stability Contract

All variants initialize from experiment 006 `v123_cached_e5_d4` epoch 100:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/
  006_voxtell_cached_native_v123_lr_ablation/
  runs/exp006_cached_native_lr_20260725T050001Z/
  v123_cached_e5_d4/model_epoch100
```

Source validation references:

```text
val20:  Dice 0.390702, hit rate 0.741935, 23/31 hits
val200: Dice 0.324134, hit rate 0.761155, 290/381 hits
```

Initialization behavior:

- load model weights only;
- copy the source decoder into proposal and refinement decoders;
- copy or share projectors according to the variant;
- zero-initialize the output layers of `A_prop` and `A_ref`;
- zero-initialize all spatial guide convolutions;
- reset optimizer, momentum, AMP scaler, LR schedule, and update counter.

At update zero:

```text
A_prop(m0) = 0
A_ref(...) = 0
G_s(p_s) = 0
proposal logits = source logits
final logits = source logits
```

The fixed-patch audit confirmed bitwise identity between source, proposal, and
final logits at all five scales for all variants. Independent full-volume
mixed-precision sliding-window evaluations can differ by approximately `3e-5`
in aggregate Dice despite that identity, so the epoch-0 aggregate gate uses
Dice tolerance `1e-4` while requiring exact hit counts.

## Shared Training Specification

### Data and reproducibility

| Setting | Value |
| --- | --- |
| Preprocessing class | Cached-equivalent to native VoxTell |
| Cache | `crop_zscore_native_v1` |
| Cache cases/targets | 3,192 cases, 8,068 targets, 0 empty stored targets |
| Patch tensor | Native-resolution `192 x 192 x 192` |
| Seed | `20260723` |
| Schedule events | Source events `10000-19999`, reindexed locally |
| Schedule length | 10,000 events |
| Schedule SHA256 | `757b00cf58a3e43e661413679ac913bda2eed14522b7db94be37eea1792eb57f` |
| Events shared across arms | Yes, identical event stream |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Distributed training | No |

### Duration and optimizer

| Setting | Value |
| --- | --- |
| Epoch definition | 100 optimizer updates |
| Duration | 100 epochs, 10,000 optimizer updates |
| Optimizer | SGD with Nesterov |
| Momentum | `0.99` |
| Weight decay | `3e-5` |
| Encoder base LR | `1e-5` |
| Non-encoder base LR | `1e-4` |
| Warmup | 100 optimizer updates |
| Post-warmup schedule | Polynomial decay |
| Poly power | `0.9` |
| Gradient clipping | Global norm maximum `12` |
| AMP | Enabled |
| Trainable parameters | All |

The optimizer and LR scheduler retain a 10,000-update horizon. The active run is
currently configured to pause after update 8000 and completed epoch-80 val20,
then resume the final 2,000 updates only after an explicit release. This is a
runtime pause, not a shortened 8,000-update poly schedule.

The warmup multiplier at optimizer step `s`, for `1 <= s <= 100`, is:

```text
lr_factor(s) = s / 100
```

Thus the first update uses `0.01` of each parameter group's base LR, and update
100 reaches the full base LR.

For `100 < s <= 10000`:

```text
progress(s) = (s - 100) / (10000 - 100)
lr_factor(s) = (1 - progress(s))^0.9
```

The same factor multiplies both LR groups. Encoder and non-encoder LRs retain
their 10x ratio throughout the run.

AMP gradients are unscaled before measurement and clipping. The global L2 norm
across all trainable parameters is clipped to 12 before the optimizer step.

### Checkpoint and evaluation cadence

```text
immutable checkpoints: updates 2000, 4000, 6000, 8000, 10000
rolling recovery checkpoint: every 500 updates
val20: epochs 0, 20, 40, 60, 80, 100
val200: epoch 100
```

Each arm follows:

```text
train segment -> save stable checkpoint -> exit trainer/release CUDA
-> same-GPU val20 -> restore optimizer/scaler/schedule state -> next segment
```

There is no evaluation sidecar and no training/evaluation overlap on one arm's
GPU. At epoch 100, val20 completes before val200 begins.

Runtime amendment:

```text
epoch 80 checkpoint -> epoch 80 val20 -> resumable pause at update 8000
explicit resume      -> epochs 81-100 -> epoch 100 val20 -> val200
```

The pause preserves model weights, optimizer momentum, AMP scaler, LR horizon,
schedule cursor, and deterministic event order.

Final-mask reporting uses threshold `0.5`. Proposal diagnostics use thresholds
`0.1`, `0.3`, and `0.5` and report:

- mean global Dice per finding;
- challenge hit rate and hits/findings;
- GT voxel coverage;
- proposal-to-GT volume ratio;
- any-overlap rate.

## Interim Evidence Through Epoch 60

This section is a dated interim snapshot, not the final experiment conclusion.
The fixed val20 probe contains 20 cases and 31 findings.

### Final branch trajectory

| Variant | Epoch 0 Dice/hits | Epoch 20 Dice/hits | Epoch 40 Dice/hits | Epoch 60 Dice/hits | Epoch 60 Dice vs source |
| --- | --- | --- | --- | --- | ---: |
| `v1_sharedfusion_softguide` | 0.3907, 23/31 | 0.3135, 21/31 | 0.3246, 22/31 | 0.3295, 21/31 | -0.0612 |
| `v1_dualfusion_softguide` | 0.3907, 23/31 | 0.3419, 23/31 | 0.3612, 24/31 | 0.3124, 22/31 | -0.0783 |
| `v2_dualfusion_precision` | 0.3907, 23/31 | 0.3217, 21/31 | 0.3357, 21/31 | 0.3024, 23/31 | -0.0883 |
| `v3_dualfusion_softguide_joint` | 0.3907, 23/31 | 0.2846, 21/31 | 0.3572, 23/31 | 0.3585, 23/31 | -0.0322 |

### Epoch-60 proposal-to-final comparison at threshold 0.5

| Variant | Proposal Dice/hits | Final Dice/hits | Final minus proposal Dice |
| --- | --- | --- | ---: |
| `v1_sharedfusion_softguide` | 0.3375, 20/31 | 0.3295, 21/31 | -0.0079 |
| `v1_dualfusion_softguide` | 0.3239, 21/31 | 0.3124, 22/31 | -0.0115 |
| `v2_dualfusion_precision` | 0.3302, 21/31 | 0.3024, 23/31 | -0.0279 |
| `v3_dualfusion_softguide_joint` | 0.3553, 22/31 | 0.3585, 23/31 | +0.0032 |

Interim interpretation:

- no arm has exceeded the source Dice;
- joint guidance is the strongest epoch-60 model, held its epoch-40 recovery,
  and remains the only arm whose final output improves Dice over its proposal;
- the detached dual-fusion arm's epoch-40 lead was not stable and its Dice fell
  by approximately `0.049` at epoch 60;
- the precision arm recovered the source hit count but has the weakest overlap
  Dice and its final mask is substantially worse than its proposal;
- shared fusion currently trails branch-specific adaptation.

Because one hit changes val20 hit rate by approximately 3.23 percentage points,
the final val200 evaluation is required before selecting a method.

## What The Four-Way Comparison Does Not Control

Experiment 008 has no same-source single-branch continuation arm. Therefore, a
later improvement over the experiment 006 source could reflect:

- the dual-branch architecture;
- another 10,000 optimizer updates;
- the continuation schedule and LR reset;
- or a combination of these.

Model capacity is also not identical: the shared-fusion arm has approximately
452.2M parameters, while each dual-adapter arm has approximately 527.8M.

The clean missing control is:

```text
Exp006 source checkpoint
+ original single-branch architecture
+ identical events 10000-19999
+ identical optimizer/LR horizon
+ identical synchronous validation cadence
```

## Brainstorming Axes Preserved By This Design

Future experiments can change one axis at a time:

| Axis | Current choice | Controlled follow-up |
| --- | --- | --- |
| Fusion separation | Shared transformer plus residual adapters | Branch LoRA, duplicated transformer, or later branch split |
| Spatial guide | Additive multiscale probabilities | Logits, uncertainty, learned attention, or gated residual |
| Context guide | Proposal-weighted bottleneck mean | Multi-region tokens, top-k pooling, or attention pooling |
| Gradient coupling | Fully detached or fully joint | Detach only spatial or only context; scheduled unfreezing |
| Proposal objective | v123 plus recall Tversky weight 0.5 inside branch | Sweep recall weight and alpha/beta |
| Final objective | v123, optional precision weight 0.1 | Partial-label-aware precision or boundary refinement |
| Proposal use | Soft full-image guidance | Proposal-guided sampling, multiple ROIs, or hard cascade |
| Scale | Native 192^3 context | Native/2 mm proposal ensemble or multiscale routing |
| Control | No single-branch continuation | Add exact continuation control before attribution |

The most informative immediate follow-ups are:

1. complete the current 100-epoch and val200 evaluations;
2. add the exact single-branch continuation control;
3. test whether the refinement branch improves over the proposal consistently,
   rather than only whether final Dice improves over the source;
4. separate the spatial-guide and context-guide contributions;
5. revisit precision pressure only with partial-label-aware supervision.

## Sources Of Truth

- Concept and literature:
  `docs/voxtell/asymmetric_proposal_refinement.md`
- This exact comparison:
  `docs/voxtell/exp008_variant_reference.md`
- Canonical config:
  `configs/experiments/008_voxtell_dual_branch_proposal_refinement_ablation.json`
- Execution contract:
  `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/codex_execution_spec.md`
- Evolving result snapshot:
  `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/report.md`
- Model implementation:
  `scripts/rexgroundingct/voxtell_dual_branch.py`
- Loss and optimizer implementation:
  `scripts/rexgroundingct/train_text_conditioned_voxtell.py`
- Synchronous launcher:
  `scripts/rexgroundingct/run_008_dual_branch_ablation.sh`
