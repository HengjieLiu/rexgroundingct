---
created: 2026-09-10
updated: 2026-09-10
status: design_proposal_pending_user_review
---

# Explicit keep/remove/add editing: proposed follow-up

Follow-up: the user subsequently authorized the bounded deletion diagnostic.
See [its completed results](deletion_diagnostic/results.md). The text below
preserves the original proposal; the diagnostic does not authorize larger runs.

Recommendation: start with a voxelwise false-positive-removal editor, while
designing the interface to support a separate false-negative-recovery head.
Target error correction and preservation explicitly. This is a proposal, not
an implemented or launched experiment; Exp027 remains stopped after epoch 50.

The key change is the output meaning, eligible edit domain and loss reduction.
A categorical editor can still use convolutional features. Replacing a
convolutional encoder with another architecture is a separate choice. Nor is
BCE intrinsically inappropriate: binary cross entropy is a natural loss for a
remove/keep action. The problematic prior setup averaged final-segmentation
losses over huge mostly-background patches without direct control of harmful
edits.

## Fixed-base action targets

Let B be the frozen base mask, `base_logit >= 0`, and Y the provided GT for the
same CT–finding pair. Define all four groups relative to B, not a moving refined
prediction. Include only valid, unpadded voxels.

| B | Y | Base status | Correct action |
| --- | --- | --- | --- |
| 1 | 0 | FP | Remove foreground |
| 1 | 1 | TP | Keep foreground |
| 0 | 1 | FN | Add foreground, if inside the allowed search domain |
| 0 | 0 | TN | Keep background |

Protecting TN is as necessary as protecting TP once an addition branch exists.
Without TN supervision, predicting additions everywhere could satisfy FN
recovery while producing overwhelming new false positives.

## Recommended generalized interface

Inputs: CT, frozen base logit and optionally its explicit binary mask. The mask
is derivable from the logit, but is useful for making routing explicit. Preserve
native normalization/geometry, original finding identity and the FP32 default.

A shared feature encoder produces two action logits:

- d_logit: remove versus keep, used only where B=1.
- a_logit: add versus keep, used only where B=0 and inside an explicit search
  support C defined without GT at inference.

This represents keep/remove/add with two conditional binary decisions. Remove
is never applied to background; add is never applied to foreground. A single
three-class head is possible, but it must mask invalid actions and handle the
dominant keep class. Separate heads make error costs and operating thresholds
easier to inspect.

At inference, blend continuous action scores consistently across overlapping
tiles, then threshold once in volume coordinates:

```text
R = B & (remove_score > remove_threshold)
A = (~B) & C & (add_score > add_threshold)
final_mask = (B & ~R) | A
```

For the initial FP-only experiment, force A=0 and omit the add-head loss.
Inactive tiles vote for keep; define overlap handling explicitly. Keep the
prior recommendation of including their geometric weights as zero-edit
contributions if using the base-positive tile gate. At base-positive voxels
all covering tiles are active, so this gate does not itself restrict deletion
decisions there.

Use an initial bias toward keep, such as low initial action probabilities, so
thresholded output equals the base mask before training. This guarantees initial
**mask** identity, not equality between learned soft scores and cached logits.
If downstream consumers require unchanged base logits for kept voxels, retain
those values explicitly in the keep branch. Do not silently claim the previous
zero-residual logit identity for this different output representation.

## Targeted loss

Let d and a be sigmoid action scores. The following describes the objective;
implementation should use stable BCEWithLogits rather than explicit log(sigmoid).
Angle brackets mean a mean over the named valid group, not the whole tile.

```text
L_remove = lambda_FP * mean_FP[-log(d)]
         + lambda_TP * mean_TP[-log(1-d)]

L_add    = lambda_FN * mean_FN_in_C[-log(a)]
         + lambda_TN * mean_TN_in_C[-log(1-a)]

L_total  = L_remove + L_add
```

Absent groups contribute zero; never divide by an empty group. FP-only training
uses only L_remove, so millions of TN voxels have no role in its loss. Separate
means prevent one group from hiding another merely through its size.

Give TP preservation an explicit, substantial cost, and select its strength
alongside the removal threshold using the desired FP-removal/TP-retention
tradeoff. For the add branch, combine sampled hard TN with representative
background rather than letting billions of easy TN dominate. Neither the exact
weights nor thresholds are adopted yet. Class balancing changes score
calibration, so an action score of 0.9 should not automatically be interpreted
as a calibrated 90% error probability.

A global Dice or Tversky term can be evaluated later as an auxiliary measure of
final shape, but it should not replace edit-specific supervision. Tversky
provides asymmetric FP/FN weighting, whereas the proposed edit losses also
distinguish correcting a base error from damaging a base-correct voxel.
[Original Tversky paper](https://arxiv.org/abs/1706.05721).

There is no new information in the four targets: they are derived from B and Y.
With unconstrained edits, the two heads can express ordinary segmentation.
The intended benefit comes from conditional routing, preservation costs,
separate normalization, restricted search and a keep decision. Merely naming
outputs FP/FN while using the same unbalanced global loss would not establish
a better method.

## Why start with FP removal

FP-only editing guarantees `final_mask` is a subset of B. It cannot create a
new foreground voxel, and the absolute FP count cannot increase. However, it
can delete TP, so recall can decrease and Dice or precision can worsen. Strong
TP supervision is an objective, not a guarantee on unseen data.

There is substantial theoretical headroom on the frozen base. Reconstructing
its verified per-finding counts gives 382,990 TP, 933,611 FP and 664,495 FN
voxels across 69 validation findings. Perfectly deleting every FP while keeping
all TP would give:

| Subset | Cached-base mean Dice | Perfect FP-removal oracle mean Dice |
| --- | ---: | ---: |
| A | 0.339416 | 0.587559 |
| B | 0.334950 | 0.593843 |
| Full | 0.337215 | 0.590656 |

For each finding this is `(2*TP + epsilon) / (2*TP + FN + epsilon)`, epsilon
1e-6. It uses GT to make perfect decisions and is **not a model result or
forecast**. Two findings have no base TP and cannot acquire a true positive
under deletion-only editing.

The challenge is selective deletion, not making the output smaller. Keeping
everything is a trivial high-retention policy; deleting everything removes
all FP but also all TP. Both extremes must be exposed by the metrics.

## Architecture and sampling

Start with voxelwise edit maps. A whole-connected-component keep/remove
classifier is another possible module, but a component can contain both TP and
FP. Whole-component deletion cannot trim an incorrect branch attached to a
correct structure. An oracle component analysis would be needed before making
that the only permitted action.

A lightweight multiscale 3D encoder-decoder is a reasonable architecture for
the proposed editor because the prior network has only a 35³ convolutional
receptive field. A CNN is compatible with categorical outputs; a transformer
is not required. For attribution, first compare the edit head/loss with the
existing backbone on fixed real patches, then consider the multiscale variant
as a separate change. Do not start by changing architecture, sampling and many
loss terms without such a diagnostic.

For FP removal, sample base-positive patches and supervise only their base-
positive voxels, including tiles with no GT. Preserve uniform findings within
the existing source mixtures. Retain all findings in evaluation, including
ones with no eligible edits. Every label is prompt/finding-specific; other 2a
findings in the same CT are not automatically foreground for the current pair.

Addition needs a clearly defined background search domain. A base-positive
tile gate or a neighborhood around B provides bounded search, but cannot
recover FN outside that domain. The previous audit identified one entirely
unreachable 116-voxel validation finding under the tile gate. Recovering such
remote misses requires expanding the search or a separate proposal mechanism.

## First diagnostic sequence and acceptance evidence

1. Measure simple base-threshold suppression as a no-learning FP-removal
   reference. Compare learned editing at similar TP retention; otherwise an
   apparent gain might be obtainable by raising the base probability threshold.
2. Fit a small fixed set of real FP/TP patches with the removal head and its
   two group-normalized loss terms. Check whether it can distinguish errors
   while retaining correct voxels. Then test different windows and full cases.
3. Report FP removed fraction, TP retained fraction, precision of deletion
   decisions, base-to-final Dice, and per-finding harms. All 69 validation
   findings remain in the A/B/full summaries. Threshold/weight decisions use
   training/A evidence, and B remains held out from refiner gradients.
4. Add the FN branch only after establishing useful removal behavior. Report
   FN recovered and new FP created as separate quantities, with TN protection
   in the loss. Joint training can alter a shared deletion encoder; it must
   continue to satisfy the deletion and preservation checks.

Incomplete GT remains a central concern. A voxel labeled FP can represent
unannotated disease or another prompt's finding. Explicit FP supervision does
not solve this and can strengthen inappropriate deletion. Review representative
FP/TP examples with CT, the original prompt and annotations before increasing
preservation/edit weights or drawing anatomical conclusions.

Error prediction from image-plus-segmentation inputs is an established related
direction; for example, published 3D MRI work uses a CNN to locate segmentation
errors. That supports the feasibility of learning error maps, but does not
demonstrate automatic correction performance for our 2a task.
[Primary error-prediction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10563140/).

## Artifacts and decision status

- [Oracle arithmetic and per-finding counts](runtime/full_fp32_100ep/analysis/categorical_editing_proposal/fp_removal_oracle.json)
- [Base-positive tile gate audit](positive_tile_gate_audit.md)
- [Stopped Exp027 results](user_stop_after_val50.md)

No new model, loss, training duration, architecture or thresholds are adopted
by this discussion. No GPU jobs were started. The numerical oracle is a
read-only calculation from saved, verified records.
