---
created: 2026-07-26
updated: 2026-07-26
status: active
topic: VoxTell asymmetric dual-branch proposal and refinement
related_experiment: 008_voxtell_dual_branch_proposal_refinement_ablation
---

# Asymmetric Proposal And Refinement For VoxTell

For an exact side-by-side record of all four implemented experiment 008
variants, including one data-flow chart per arm, gradient ownership, complete
loss formulas, training settings, and interim metrics, see
[`exp008_variant_reference.md`](exp008_variant_reference.md).

## Purpose

This note records the design discussion that led to experiment 008. It is a
technical reference for a two-branch VoxTell model in which:

1. a proposal branch is biased toward sensitivity and target inclusion; and
2. a refinement branch uses the proposal softly while preserving the original
   segmentation objective.

The immediate experiment compares shared versus branch-adapted text-image
fusion, a modest precision auxiliary, and detached versus joint proposal
guidance. This note also records the alternatives that are deliberately not
part of the first experiment.

## Starting Observation

The current VoxTell model has one text-conditioned decoder path and one final
segmentation output per prompt. It must solve two difficult behaviors with the
same representation and loss:

- find all voxels that plausibly belong to a small or subtle requested lesion;
- reject visually similar anatomy and unrelated findings precisely.

Those goals are asymmetric. A proposal mechanism should prefer a false positive
over a false negative because a later stage can remove excess foreground. A
refinement mechanism should preserve the proposal's useful candidates while
recovering a challenge-quality final mask.

A second decoder is not useful merely because it exists. If two decoders receive
the same features and neither output affects the other, the second branch is
only another prediction head. The proposal branch helps the final segmentation
only when its output changes the computation used by the refinement branch.

Experiment 008 therefore uses the proposal probability maps as explicit,
learned, soft guidance at every refinement scale.

## Why Soft Guidance

A hard crop, binary gate, or intersection would make the proposal an
irreversible bottleneck:

```text
proposal miss -> anatomy removed -> refinement cannot recover it
```

That is undesirable while the proposal branch is experimental. Instead, the
refinement branch receives:

- the complete encoder skip features;
- its own text-conditioned fusion representation; and
- additive features derived from proposal probabilities.

The final prediction is not multiplied by the proposal and is not restricted
to a proposal bounding box. The refiner can use, ignore, or correct the guide.

At decoder scale `s`, the guide is:

```text
p_s = sigmoid(z_proposal_s)
g_s = Conv1x1x1_s(p_s)
x_refine_s = x_refine_s + g_s
```

Every guide convolution is initialized with zero weights and zero bias. The
guide therefore has no effect at initialization. This is an important
stability property: the copied refinement decoder begins as the selected
single-branch checkpoint, not as a randomly perturbed model.

## VoxTell Components And Memory Constraint

Parameter counts must be de-duplicated because the upstream segmentation
decoder retains a reference to the shared encoder. Summing module-local counts
therefore overcounts the model. The experiment 008 materialization audit
records these de-duplicated trainable totals:

| Model | Parameters |
| --- | ---: |
| Exp006 source model | 429,797,381 |
| Shared-fusion dual branch | 452,206,570 |
| Dual-adapter dual branch | 527,821,578 |

The shared-fusion design adds a second decoder path and guide adapters. The
dual-fusion design additionally copies the per-scale projectors and adds the
two residual fusion adapters. A second complete transformer fusion trunk would
be substantially more expensive before optimizer state or activations. The
existing batch-1 source model already peaked near 25.9 GiB in experiment 006.
DDP does not combine the memory of multiple GPUs; every process must hold one
complete model.

The first dual-fusion implementation therefore keeps the expensive base
transformer shared and adds lightweight branch-specific adaptation after the
shared fused representation. A fully duplicated transformer or LoRA inside
separate transformer paths remains a later experiment.

## Shared Backbone And Two Decoders

All experiment 008 variants share:

- the image encoder;
- the bottleneck-to-query projection;
- the text-to-query projection;
- the six-layer text-image transformer;
- the fixed positional encoding.

The source segmentation decoder is copied twice:

- `proposal_decoder`;
- `refinement_decoder`.

Both copies are initialized from experiment 006
`v123_cached_e5_d4` epoch 100. All new residual adapters and spatial guide
adapters are zero-initialized.

The proposal and final outputs must both match the source model before any
training update.

## Shared Fusion

The shared-fusion variant uses the same fused prompt tokens and the same
per-scale projectors for both decoders:

```text
image, text -> shared transformer -> m0
                                 -> shared projectors -> proposal decoder
                                                      -> refinement decoder
```

The proposal probability maps still guide the refinement decoder spatially.
The proposal guide is detached before entering the refiner, so final-loss
gradients do not flow through the proposal-specific decoder computation.

The proposal loss and final loss both update the shared encoder, transformer,
and projector. "Detached" therefore means detached at the guide boundary; it
does not mean that the two tasks have completely independent shared weights.

## Lightweight Dual Fusion

The dual-fusion variants begin with the same shared transformer output `m0` but
adapt it separately:

```text
m_prop = m0 + A_prop(m0)
```

`A_prop` is a residual multilayer perceptron with a zero-initialized final
linear layer. Proposal projectors are copied from the source model.

After proposal decoding, the lowest-resolution proposal probability is used to
pool prompt-specific image context from the projected bottleneck memory:

```text
c_prop = sum(p_low * bottleneck_memory) / (sum(p_low) + epsilon)
m_ref = m0 + A_ref(concat(m0, c_prop))
```

The flattening order must match VoxTell's bottleneck memory order. VoxTell
rearranges encoder features from `B,C,D,H,W` to `B,H,W,D,C` before flattening;
the proposal map must use the same `H,W,D` order for weighted pooling.

`A_ref` also has a zero-initialized final layer. The refinement branch has its
own copy of the per-scale projectors. At initialization:

```text
A_prop(...) = 0
A_ref(...) = 0
m_prop = m_ref = m0
```

Together with copied projectors, copied decoders, and zero spatial guides, this
preserves the source prediction.

This design is called "dual fusion" for experiment naming, but it is not two
complete transformer fusion trunks. It is a shared base fusion followed by
branch-specific residual adaptation and projection.

## Detached And Joint Guidance

For detached variants, both proposal-dependent paths are detached:

- multiscale proposal probabilities used by spatial guide adapters;
- proposal-weighted context used by the refinement fusion adapter.

The proposal branch still receives its direct proposal loss. Shared modules
still receive gradients from both branch losses.

For the joint variant, proposal maps and proposal context are not detached.
The final segmentation loss can therefore update proposal-specific modules
through the guide path. At exact initialization the zero guide adapters can
temporarily block part of that gradient, but the path becomes active as the
adapters learn nonzero weights.

Joint training may improve cooperation, but it can also cause the proposal to
stop behaving as a high-recall proposal and instead optimize whatever is most
convenient for the final decoder. Direct proposal supervision is retained to
counter that drift.

## Loss Design

Let:

```text
Tversky(alpha, beta)
    = TP / (TP + alpha * FP + beta * FN + epsilon)
```

The proposed auxiliaries are:

```text
proposal_recall = 1 - Tversky(alpha=0.3, beta=0.7)
final_precision = 1 - Tversky(alpha=0.7, beta=0.3)
```

The proposal objective is:

```text
proposal_loss
    = v123(proposal)
    + 0.5 * proposal_recall
```

The common total objective is:

```text
total_loss
    = v123(final)
    + 0.5 * proposal_loss
```

Only `v2_dualfusion_precision` adds:

```text
total_loss += 0.1 * final_precision
```

This keeps final v123 segmentation as the dominant objective. Expanding the
formula shows the effective additions:

```text
total_loss
    = 1.0 * v123(final)
    + 0.5 * v123(proposal)
    + 0.25 * proposal_recall
    + optional 0.1 * final_precision
```

The asymmetric Tversky auxiliaries are calculated only for nonempty prompts
and only at full resolution. This avoids undefined empty-mask behavior and
avoids making small-lesion survival at coarse deep-supervision scales a new
method variable.

Both proposal and final v123 losses retain:

- nonempty target: Dice plus weighted BCE;
- empty target: weighted BCE only, multiplied by `0.5`;
- BCE foreground/boundary/background weights `1.0/1.5/0.5`;
- boundary radius `10`;
- deep-supervision weights `[1, 1/2, 1/4, 1/8, 1/16]`, normalized;
- no voxel-weighted Dice.

## Why The Precision Auxiliary Is Modest

ReXGroundingCT training labels are partial at the instance level. For an
annotated finding, annotators segmented at most three representative instances
in training even when more were visible. Validation and test instead request
all visible instances.

As a result, a predicted fourth instance can be a clinically correct region but
appear as a false positive against the training mask. A strong global precision
loss could teach the model to suppress valid additional lesions.

Experiment 008 therefore:

- uses a low final precision coefficient of `0.1`;
- applies it to only one arm;
- keeps the standard final v123 loss dominant;
- reports the partial-label caveat whenever interpreting precision behavior.

See
`docs/brainstorm/2026-07-23_rex_training_partial_instance_annotation_evidence.md`
for the source evidence and exact limits of this claim.

## Four Experiment Variants

| Variant | Fusion after shared transformer | Spatial/context guide | Final precision auxiliary |
| --- | --- | --- | --- |
| `v1_sharedfusion_softguide` | Shared tokens and projectors | Detached | No |
| `v1_dualfusion_softguide` | Branch adapters and projector copies | Detached | No |
| `v2_dualfusion_precision` | Branch adapters and projector copies | Detached | Yes, weight 0.1 |
| `v3_dualfusion_softguide_joint` | Branch adapters and projector copies | Joint | No |

The comparisons answer different questions:

- shared versus dual fusion: does branch-specific adaptation help?
- dual soft-guide versus dual precision: does an explicit modest final
  precision bias help?
- detached versus joint: should final-loss gradients shape the proposal through
  the guide?

These are cumulative architectural comparisons, not four independently tuned
models.

## Training Stability

Stability is protected by:

- source weights copied into both decoders and both projector paths;
- zero-initialized residual and guide outputs;
- final v123 loss retained at full weight;
- proposal loss downweighted in the total objective;
- the same optimizer family and learning rates as the selected source recipe;
- 100-update warmup;
- global gradient clipping at norm 12;
- epoch-0 equivalence evaluation before training;
- finite-loss and memory smokes before full launch.

Training logs must separate:

- final v123 loss;
- proposal v123 loss;
- proposal recall Tversky loss;
- optional final precision Tversky loss;
- total weighted loss;
- global gradient norm before clipping;
- guide-adapter parameter norms;
- proposal and refinement branch gradient norms;
- encoder and non-encoder learning rates;
- update time and peak GPU memory.

Without component logging, a decreasing total loss would not reveal whether the
proposal collapsed, the final task degraded, or the precision auxiliary
dominated.

## Proposal Evaluation

The challenge metric remains final mean global Dice per finding at threshold
0.5. Proposal quality needs additional diagnostics because a useful inclusive
proposal may have poor Dice by design.

For proposal thresholds `0.1`, `0.3`, and `0.5`, report:

- fraction of GT voxels included;
- number and fraction of findings with any overlap;
- proposal volume divided by GT volume;
- Dice;
- challenge hit rate at Dice at least 0.1.

Also report the refinement transition:

- proposal false-positive volume removed by the final branch;
- proposal GT voxels retained by the final branch;
- final Dice minus proposal Dice.

These metrics distinguish "high recall but controllable" from "nearly
everything is foreground."

## Literature Context

### Tversky Loss

Salehi et al., [Tversky loss function for image segmentation using 3D fully
convolutional deep networks](https://arxiv.org/abs/1706.05721), formalized
different false-positive and false-negative costs for imbalanced medical
segmentation. It directly motivates the proposal and precision auxiliaries.
Experiment 008 retains Dice and BCE rather than replacing the complete
segmentation objective with Tversky.

### Focal Tversky

Abraham and Khan, [A Novel Focal Tversky loss function with improved Attention
U-Net for lesion segmentation](https://arxiv.org/abs/1810.07842), focuses
optimization on difficult examples and small lesions. Focal Tversky is a future
option, not part of experiment 008, because adding a focal exponent would make
the first architectural test harder to interpret.

### Sensitive Candidate Then False-Positive Reduction

Valverde et al., [Improving automated multiple sclerosis lesion segmentation
with a cascaded 3D convolutional neural network
approach](https://arxiv.org/abs/1702.04869), use a first lesion-sensitive
network followed by a second network that reduces misclassified candidate
voxels. This is the closest high-level motivation for the proposed asymmetric
roles. Experiment 008 differs by sharing a large text-conditioned backbone and
using differentiable soft guidance instead of a separate patch classifier.

### Coarse-To-Fine Cascades

Roth et al., [An application of cascaded 3D fully convolutional networks for
medical image segmentation](https://arxiv.org/abs/1803.05431), first define a
candidate region and then refine it with a second 3D FCN. nnU-Net's broader
automated design work is described in [Automated Design of Deep Learning
Methods for Biomedical Image Segmentation](https://arxiv.org/abs/1904.08128).
These works establish coarse-to-fine processing as a standard segmentation
strategy. Experiment 008 remains full-FOV within each 192-cubed window and does
not yet use a hard region-of-interest crop.

### Conservative And Radical Predictions

Shi et al., [Inconsistency-aware Uncertainty Estimation for Semi-supervised
Medical Image Segmentation](https://arxiv.org/abs/2110.08762), introduce
CoraNet with conservative and radical predictions induced by different
misclassification costs. Their disagreement estimates uncertain regions for
semi-supervised learning. Experiment 008 instead uses the inclusive branch as
direct guidance for a final supervised refinement branch.

### High-Recall Ensembles

Ma et al., [Ensembling Low Precision Models for Binary Biomedical Image
Segmentation](https://openaccess.thecvf.com/content/WACV2021/html/Ma_Ensembling_Low_Precision_Models_for_Binary_Biomedical_Image_Segmentation_WACV_2021_paper.html),
show that diverse high-recall, low-precision models can have inconsistent false
positives that cancel under ensembling while retaining consistent positives.
This supports a future ensemble of proposal branches, but experiment 008 first
tests whether one inclusive proposal improves its paired refinement output.

## What Experiment 008 Can Establish

If the four arms are controlled and complete, the experiment can show:

- whether branch-specific fusion adaptation outperforms shared fusion under the
  same continuation schedule;
- whether a modest precision auxiliary improves or damages final segmentation;
- whether joint final-to-proposal gradients help;
- whether the proposal becomes more inclusive without becoming unusably large;
- the memory and runtime cost of the dual architecture.

## What Experiment 008 Cannot Establish

There is no fifth, same-source, single-branch continuation control. Therefore:

- an improvement over the experiment 006 starting checkpoint may partly come
  from another 10,000 updates;
- `v1_sharedfusion_softguide` is the closest architectural control but still
  adds a proposal decoder, proposal loss, and guidance;
- the experiment cannot attribute every gain over epoch 0 exclusively to the
  dual-branch design.

A true control should later continue the same experiment 006 checkpoint with
the original single-branch architecture, identical schedule events, optimizer,
learning-rate horizon, and validation cadence.

## Future Experiments

Prioritized follow-ups are:

1. Run the true single-branch continuation control.
2. Tune proposal recall and final precision coefficients one at a time.
3. Test proposal threshold calibration and probability-average proposal
   ensembles.
4. Replace lightweight adapters with branch-specific LoRA in the transformer.
5. Duplicate the complete fusion transformer only if memory evidence justifies
   it.
6. Add partial-label-aware precision supervision that ignores uncertain
   background regions.
7. Use proposal uncertainty or branch disagreement as an additional guide.
8. Test proposal-guided sampling during training.
9. Introduce hard proposal crops only after proposal inclusion is reliably
   near exhaustive.
10. Combine proposal/refinement behavior with native, resampled multiscale, or
    mixture-of-experts patch routing.

## Experiment 008 Contract

The concrete experiment uses:

- source: experiment 006 `v123_cached_e5_d4` epoch 100;
- preprocessing: standard cached-equivalent native crop and full-volume
  z-score;
- patch: `192 x 192 x 192`;
- batch size: 1, no DDP;
- schedule: deterministic source events `10000-19999`;
- duration: 100 epochs, 100 optimizer updates per epoch;
- synchronous val20 at epochs `0/20/40/60/80/100`;
- final val200 after epoch 100;
- four independent GPUs, one variant per GPU.

The canonical machine-readable config and execution spec are stored under the
experiment 008 config and repo-local experiment directories.
