---
created: 2026-09-10
updated: 2026-09-10
status: proposal_audit_pending_user_review
---

# Audit: enable refinement only on base-positive 192³ tiles

This is a meaningful change and a reasonable next ablation. It imposes an exact
identity outside the union of active tiles and avoids spending training updates
on branches that would never run. However, it does not make edits within an
active tile conservative, does not fix the weak negative-voxel loss, and can
prevent recovery of a finding far from any base prediction. Improvement is
plausible, not established by this geometry audit.

No refiner training or inference was run. Exp027 remains stopped after epoch 50.
The proposed gate and other recommendations below have not been implemented
or adopted as a new experiment.

## What differs from the current scheme

| Aspect | Exp027 as run | Proposed gate |
| --- | --- | --- |
| Inference windows | Refine every native 192³ tile | Refine only tiles containing a base-positive voxel |
| Training windows | 50% GT, 25% prediction, 25% random anchors | Only base-positive tiles contribute refiner updates |
| Where logits can change | Entire preprocessed extent | Union of active tiles |
| Entire finding with no base prediction | Refiner may create foreground | Return unchanged base; no eligible training tile |
| Base-positive tile with no annotated GT | Can occur | Must still be retained for false-positive-removal training |

Define a base-positive voxel as `base_logit >= 0`, equivalent to probability
at least 0.5, on valid unpadded voxels. Test the **same CT–finding channel** that
the refiner will correct, not another prompt or the union of all 2a channels.
Use only the frozen base to decide eligibility; do not let newly refined
predictions activate more tiles.

## Direct measurements on the existing caches

The audit uses the exact production `tile_starts` grid, 192³ tiles and 50%
overlap. It scans all 69 validation finding channels on 63 CTs, computes exact
box-union coverage and GT reachability, and reads prediction-availability
metadata for all 1,120 training findings. It took 96.5 seconds on CPU, with
GPUs disabled. Volumes are counted per CT–finding pair, as required by the model.

| Geometry measurement | A | B | Full |
| --- | ---: | ---: | ---: |
| Total inference tiles | 3,098 | 2,624 | 5,722 |
| Active tiles | 1,177 | 1,121 | 2,298 |
| Active fraction | 38.0% | 42.7% | 40.2% |
| Active tiles with no GT | 475 | 502 | 977 |
| Editable voxel fraction, pooled | 52.3% | 58.5% | 55.2% |
| GT voxels outside every active tile | 0 | 116 | 116 |
| Findings with all GT geometrically reachable | 35/35 | 33/34 | 68/69 |

The gate skips **3,424/5,722 tiles (59.8%)**. That is a reduction in refiner
forward calls, not a measured wall-time speedup. Cache reading, gate checks,
blending, restoration and metric work still cost time. Peak memory for an
active 192³ forward need not decrease.

Because tiles overlap, **55.2% of pooled voxels remain editable**. The mean
per-finding editable fraction is 58.4%, with a range of 17.5%–100%. This is a
broad spatial permission, even though fewer than half of windows run.

The union retains 99.9889% of annotated voxels, but that aggregate hides a
whole missed finding: `train_13417_d_1.nii.gz::1`, B, prompt "Fibrotic
subcentimeter densities in the lower lobe of the right lung". Its 116 GT voxels
lie outside every active tile. Nine base-positive voxels elsewhere activate
twelve tiles, all GT-empty. The gated method cannot recover this finding.
The ungated epoch-50 run 3 recovered 105 of its 116 GT voxels, although its
2,438 FP voxels kept Dice at only 0.079. This is a real recall tradeoff.

All 69 validation findings have some base-positive prediction. One original
training finding, `train_7777_a_2.nii.gz::4`, has none and would have no eligible
training patch under this rule. Track this exclusion explicitly; do not drop
ineligible findings from evaluation or headline denominators.

Of the 128 historical training patches sampled in the prior audit, **110
(85.9%) already contain base-positive voxels**. Filtering that sample rejects
only 18 patches: 15 random and 3 GT-mode patches. Thus this is a large change
to inference coverage, but simply filtering the existing training sampler may
be a much smaller change to its training distribution. This historical sample
is descriptive, not a census of all training patches.

## Critical distinctions

### Base-positive is not the same as GT-positive

**977/2,298 active validation tiles (42.5%) contain no GT for their finding.**
They contain only annotated false-positive base predictions and surrounding
background. These are essential examples for learning to remove false positives.
Excluding every GT-empty patch would directly undermine the intended behavior.

Even within an active tile, foreground is sparse. Mean BCE over seven million
voxels still dilutes the penalty for hundreds of wrong voxels, and empty-target
soft Dice remains almost constant. The gate removes inactive examples; it does
not repair the loss on the many active, GT-empty examples.

### A skipped window does not make all its voxels immutable

An active neighboring window may cover the same voxel and contribute a
residual there. Indeed, 115 skipped windows intersect GT, but almost all of
that GT is covered by other active windows. The exact guarantee is:

> The gate guarantees unchanged logits where every covering window is inactive.

One predicted voxel enables its whole 192³ tile. The method can still add
foreground far from that prediction within the enabled tile.

### An inference-only gate cannot change corrections at base-positive voxels

For a fixed checkpoint and unchanged tiling/blending, every tile covering a
base-positive voxel necessarily passes the gate. Consequently the residual
and final prediction at that voxel are identical with or without gating.
This preserves both removal of existing base false positives and harmful
removal of existing true positives. The latter failure needs better training;
gating alone cannot repair it.

This invariance concerns adding the gate to fixed weights. Retraining can
change those weights and therefore learn different behavior on base positives.

Nor is false-positive count guaranteed to decrease in overlapping background.
For example, at a base-logit -1 voxel shared by two equally weighted tiles,
residuals +4 from an active tile and -3 from an inactive tile yield final logit
-0.5. Zeroing the inactive tile changes the final logit to +1. The skipped tile
had supplied helpful suppression. This explains why the gate must be tested
with actual paired predictions, not judged from skipped-tile counts alone.

## Recommended overlap rule

For tile t, let g_t be its frozen-base gate and w_t its current Gaussian weight.
Use the following conservative interpretation of a disabled branch:

```text
effective_tile_residual = gate * network_output
blended_residual(v) = sum_t(weight_t(v) * effective_tile_residual_t(v))
                      / sum_t(weight_t(v))
final_logit(v) = base_logit(v) + blended_residual(v)
```

Skip the network computation on an inactive tile, but retain its geometric
weight in the denominator as a zero-residual contribution. Averaging only over
active tiles is a different method: it increases the contribution of remaining
tiles in mixed overlaps. Both choices require explicit behavior where no tile
is active; the recommended rule returns exactly the base there.

Preserve FP32 blending and refiner precision, zero-head initialization, original
geometry and the frozen cache contract. No VoxTell logit regeneration is needed.

## Training changes that matter most

1. **Match the eligible training windows to the inference rule.** As a clean
   reference, sample a finding uniformly within the existing source mixture,
   then sample one of its eligible inference-grid tiles. Do not sample globally
   over all tiles, which weights large or widely predicted findings more.
   Keep original run-2 source mixing at 50/50 if retaining the four data arms.
   Exclude no-prediction findings from optimizer sampling with explicit counts.
   Joint flips preserve eligibility. If using random shifted windows instead,
   apply the same gate and quantify that distribution change.
2. **Retain and strengthen supervision on active GT-empty and hard-background
   regions.** Log GT-present versus GT-empty active tiles separately. Compare
   the current loss with a controlled alternative using separately normalized
   foreground/hard-background BCE or hard-negative mining. Retain a foreground
   overlap term where GT exists. Merely deleting empty-target Dice's near-
   constant term does not solve background-gradient dilution. Avoid stacking
   several new losses before testing their individual effects.
3. **Measure harm relative to the base, not only absolute training loss.** On
   fixed probe tiles and full volumes, record paired Dice, new FP, FP removed,
   TP removed and FN recovered, separated by active GT-present/GT-empty tiles.
   A model that reduces FP by deleting correct base foreground is not meeting
   the intended conservative objective.
4. **If spatial conservatism is still insufficient, test an explicit voxel
   support mask separately.** Restrict applied residuals to a physical-distance
   neighborhood of the base mask, with unchanged logits outside it. Select and
   audit the margin using training/A evidence and per-finding GT coverage, not
   an arbitrary voxel radius. Do not restrict edits to the exact base mask:
   that would prohibit recovery of every base false negative. A support mask
   can still allow deletion of base true positives, so it is not a substitute
   for harmful-edit control.

Sampling matching matters particularly because GroupNorm uses statistics from
each input during both training and evaluation. It does not freeze spatial
statistics in `eval()` mode.
[PyTorch GroupNorm documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html).

## Efficient test sequence before repeating a long four-arm run

1. **Inference-only ablation on the existing epoch-50 checkpoints:** compare
   the saved ungated results with gated inference at the same weights. This
   isolates the gate's effect without spending more training time. Verify
   unchanged outputs on base-positive voxels and outside the active union.
2. **Short real-patch fitting diagnostic:** fixed active GT-present and active
   GT-empty tiles, then different eligible tiles from the same cases and their
   full volumes. Check foreground fitting, FP suppression and TP preservation.
3. **Controlled retraining:** compare the current loss with one negative-aware
   alternative, using identical active-tile schedules, initialization and update
   budgets. Keep the network, LR 1e-4, clipping 1.0 and FP32 unchanged initially.
   Repeat the four data-source arms after the mechanics and fitting check pass.

Keep all 69 validation findings in A/B/full evaluation, including unchanged or
unreachable cases. A and B retain their existing exposure definitions; repeated
diagnostic use means B is held out from gradients, not a fresh unseen final test.
Ranking and checkpoint selection remain the user's decisions. No GPU evaluation
or retraining in this test sequence has been authorized or started by the audit.

## Verification and artifacts

Three synthetic box-union checks, including small/padded volumes, and two slab
coordinate extraction checks passed. All 69 base-positive voxel counts and GT
counts match the existing epoch-50 baseline records. The production source
fingerprint is unchanged. The initial audit attempt stopped before scanning
data because the no-resampling flag is `null`, not `false`; the audit check was
corrected to the actual schema before the successful run.

- [Audit script](../../scripts/rexgroundingct/audit_027_positive_tile_gate.py)
- [Counts and per-finding geometry](runtime/full_fp32_100ep/analysis/positive_tile_gate_audit/)
- [Previous learning diagnosis](learning_diagnosis_e040.md)
- [Stopped experiment and epoch-50 results](user_stop_after_val50.md)
