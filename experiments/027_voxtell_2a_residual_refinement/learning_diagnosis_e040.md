---
created: 2026-09-10
updated: 2026-09-10
status: interim_diagnosis_pending_user_review
---

# Exp027 learning diagnosis through epoch 40

Subsequent user decision: [stop after completing epoch-50 validation](user_stop_after_val50.md).
This diagnostic snapshot remains fixed at epoch 40.

The refiner is learning and changing predictions. The main observed failure is
that improvements on sampled foreground patches do not translate into better
whole-volume Dice. Harmful background additions and removal of true foreground
offset useful edits. The evidence supports investigating the patch objective
and sampling distribution before increasing capacity, learning rate, or residual
magnitude. It does not establish one exclusive cause.

This audit freezes the sixteen completed evaluations at updates 1,000–4,000 and
the first 4,000 training records per arm. During the audit, the authorized run
reached update 5,000 and was evaluating epoch 50. Partial epoch-50 records are
excluded. Training continues unchanged; no checkpoint is selected or ranked.

## Evidence and reproducibility

- Recomputed every one of 1,104 per-finding evaluation records (4 arms × 4
  milestones × 69 findings), including baseline Dice from reconstructed base
  confusion counts, exact finding IDs/prompts, exposure and A/B/full aggregation.
- Replayed 128 exact historical 192³ patch inputs, 32 per arm. Each arm has 16
  samples from updates 1–1,000 and 16 from 3,001–4,000, each stratified as 8 GT,
  4 prediction and 4 random patches. Selection seed is `270040 + arm_index`;
  update 1 is included to verify zero-residual baseline identity.
- Compared each replayed base patch to its logged refinement at that historical
  update. These comparisons share inputs exactly, but the historical weights
  vary across samples. This is a small descriptive sample, not a frozen-model
  evaluation or a statistically representative estimate of all training patches.
- Inspected all four initial and epoch-40 checkpoints on CPU. Checked production
  loss against an independent NumPy calculation for both empty and nonempty
  targets. All checks passed.
- Ran four CPU forward passes: two shifted 192³ windows each for runs 3 and 4
  on one case selected because run 3 regressed strongly. Used PyTorch 2.8.0 in
  the existing Docker image, GPUs disabled, two CPU threads and a 12 GiB memory
  limit. Each forward took about 15–16 seconds. No GPU diagnostics or new
  training jobs were launched.
- Verified that all ten production source hashes still match the full-run
  launch snapshot. The repository workflow and whitespace checks passed;
  consistency emitted 19 pre-existing warnings for other experiments.

Scripts:
[paired data audit](../../scripts/rexgroundingct/audit_027_refiner_learning.py),
[checkpoint/context probe](../../scripts/rexgroundingct/audit_027_refiner_context.py).
Detailed outputs, hashes and provenance:
[analysis directory](runtime/full_fp32_100ep/analysis/learning_diagnosis_e040/).

![Interim learning curves](runtime/full_fp32_100ep/analysis/learning_diagnosis_e040/learning_diagnosis.png)

## Completed full-volume results

Dice values below are fractions, not percentages. A contains 35 findings; B
contains 34. The base model is the exact cache's own baseline.

| Model, fixed order | A Dice, epoch 40 | B Dice, epoch 40 | Full Dice, epoch 40 | Full change | Hits / 69 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached base | 0.339416 | 0.334950 | 0.337215 | — | 59 |
| Run 1: train | 0.336684 | 0.330105 | 0.333442 | -0.003773 | 59 |
| Run 2: train+A | 0.338600 | 0.326161 | 0.332471 | -0.004745 | 59 |
| Run 3: A | 0.338266 | 0.303986 | 0.321375 | -0.015841 | 60 |
| Run 4: A+B | 0.337568 | 0.332108 | 0.334878 | -0.002338 | 60 |

A is training data for runs 2–4; B is training data for run 4. Full scores for
runs 2–3 mix exposure. Held-out here means held out from refiner training; the
frozen base checkpoint was already selected using validation evidence.

| Full Dice change from base | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 40 |
| --- | ---: | ---: | ---: | ---: |
| Run 1 | -0.006832 | -0.004282 | +0.000002 | -0.003773 |
| Run 2 | -0.001689 | -0.004899 | -0.000074 | -0.004745 |
| Run 3 | -0.007257 | -0.012307 | -0.006458 | -0.015841 |
| Run 4 | -0.009773 | -0.007448 | -0.007195 | -0.002338 |

There is no consistent full-volume gain so far. Run 4's fitting diagnostic is
disappointing, but epoch 40 is not convergence and this arm is not a mathematical
upper bound on every residual-refinement method.

Descriptive patient-cluster bootstrap intervals, 10,000 resamples with seed
270040, put run 3's B change at -0.03096 with a 95% interval
[-0.05165, -0.01206]. Run 4's full change is -0.00234 with interval
[-0.01121, +0.00743]. These are exploratory paired intervals at an interim
checkpoint, with no multiple-comparison adjustment. They do not support a
claim that every small difference is a reliable effect.

## 1. The patch objective rewards some useful edits but weakly penalizes others

For the sampled **nonempty** patches from updates 3,001–4,000:

| Run | Nonempty patches sampled | Base hard patch Dice | Refined hard patch Dice | Paired gain |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.305626 | 0.308421 | +0.002795 |
| 2 | 12 | 0.212020 | 0.227368 | +0.015349 |
| 3 | 12 | 0.393577 | 0.414510 | +0.020933 |
| 4 | 11 | 0.450425 | 0.507508 | +0.057083 |

In run 4's eight sampled GT-mode patches, the paired gain is +0.077521.
Thus the existing dashboard's noisy, unpaired training curve understates some
foreground learning. Its patch mix changes every update and it has no paired
base curve.

However, each late sample contains 4–5 empty-target patches out of 16. In run
3, two previously empty, correctly predicted random patches acquire foreground.
In run 4, one such patch does. Their hard patch Dice changes from 1 to nearly
zero, while their total losses increase by only 0.00048–0.00081.

Concrete example: run 3, update 3,525,
`train_3027_a_2.nii.gz::0`, random patch, no GT or base foreground:

- Base hard Dice: 1.0; base total loss: 1.000053.
- Refined hard Dice: approximately 1.54e-9; total loss: 1.000862.
- The logged hard Dice implies about 648 false-positive voxels.
- Valid patch size is 7,077,888 voxels; BCE is averaged across all of them.
- Base sigmoid probability mass is about 375 despite zero thresholded positives.
  With an empty target, soft Dice is `epsilon / (probability_mass + epsilon)`.
  With epsilon 1e-6, the Dice loss is already almost 1 and provides very little
  gradient for distinguishing these empty predictions.

The loss therefore gives large incentives to improve nonempty patch overlap,
while several hundred wrong voxels in an empty context can have a tiny cost.
Full-volume inference covers all those contexts. This is the strongest measured
lead for why training-patch learning does not yield a cohort gain. The hard-Dice
collapse on an empty patch is also an extreme property of that metric, so the
confusion counts and small loss change are more informative than the patch Dice
alone.

Simply omitting empty-target Dice would remove a near-constant from the plotted
loss; it would not by itself fix weak background gradients. A follow-up needs
an effective negative-voxel signal, for example separately normalized foreground
and hard-background BCE or controlled hard-negative mining, while retaining
whole-volume evaluation. Such changes are proposed ablations, not adopted edits.

## 2. Small average residuals hide substantial and often harmful local edits

Epoch-40 counts over all 69 findings, relative to the frozen base. Helpful and
harmful here are defined strictly against the provided annotation.

| Run | FP removed | FN recovered | TP removed | New FP added | Maximum absolute residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 11,460 | 3 | 79,848 | 5.91 |
| 2 | 24,188 | 12,863 | 4,694 | 93,269 | 13.73 |
| 3 | 236,230 | 25,209 | 48,164 | 217,871 | 42.01 |
| 4 | 216,486 | 8,305 | 42,563 | 75,964 | 32.19 |

Run 1 mostly expands foreground; only 12.55% of its newly positive voxels are
annotated GT. Run 4 removes many false positives but also removes over five
times as many true positives as it recovers missed positives. Macro recall
falls from base 0.471855 to 0.456451, while macro precision rises only from
0.352823 to 0.356572. These sums explain edit direction, but do not replace the
per-finding Dice calculation: removing one FP and one TP have different effects.

The mean fraction of the base/refined/GT union edited is 7.25%, 10.87%, 27.60%,
and 19.30% in run order. Mean absolute residual across the entire native extent
is only 0.00382, 0.03550, 0.05984 and 0.04016. Most voxels are background, so
that average is a poor measure of behavior near a finding. Raising residual
magnitude is not the obvious missing ingredient.

An illustrative run-3 failure is `train_3006_a_2.nii.gz::1`, B, prompt
"Scattered linear atelectasis in both lungs": Dice falls 0.484414 → 0.301625;
7,020 new FP voxels appear while only 205 FN voxels are recovered. Other large
B regressions include `train_2675_a_1.nii.gz::0` (-0.159402 Dice) and
`train_19456_a_1.nii.gz::1` (-0.128608). These are diagnostic inspection targets,
not checkpoint rankings. No clinical adjudication of their annotations has
been performed in this audit.

## 3. Run 3 has a generalization problem; run 4 argues against a labels-only explanation

At epoch 30, run 3 improves A by 0.003639 but worsens B by 0.016852. At epoch
40, its A change is -0.001150 while B is -0.030964. The growing B deficit and
stronger local corrections are consistent with overfitting to A's patch
distribution. They are not evidence of no learning.

Run 4 receives only the released validation annotations and still fails to
improve full-volume Dice so far. Therefore incomplete labels in the original
training set cannot explain the whole result. Validation annotations may also
be incomplete or prompt-specific, and a CT-plus-logit refiner can lack enough
information to resolve ambiguous unlabeled structures. This requires visual
review with the original prompts; metrics alone cannot establish label quality.

By update 4,000, run 1 sampled 1,098/1,120 original training findings, but only
944 had a GT-mode patch. Run 2 sampled 932 training findings, only 675 with a
GT-mode patch. All 35 A findings and all 69 run-4 findings received GT-mode
patches. Thus "40 epochs" is 4,000 updates, not 40 exhaustive passes through
the original training findings; underexposure can matter for runs 1–2, but
does not explain run 4's behavior by itself.

## 4. Tile context matters, but the measured effect is smaller than the principal regression

The convolutional spatial receptive field is 35³ voxels, despite a 192³ input
tile: `3 + 4 × (1 + 2 + 4 + 1) = 35` along each axis. GroupNorm adds global
tile statistics, not structured anatomical context or the finding's text.
PyTorch GroupNorm uses input statistics in both train and evaluation modes;
calling `eval()` does not freeze them.
[Official documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html).

On `train_3006_a_2.nii.gz::1`, shifting a 192³ window by 32 voxels changed
predictions within a shared interior that excludes the 17-voxel convolutional
radius at both windows' boundaries:

| Epoch-40 model | Interior voxels changing label | On GT | Maximum residual difference | Local Dice, window 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Run 3 | 38 | 21 | 0.352 | 0.6322 / 0.6348 |
| Run 4 | 77 | 41 | 0.767 | 0.6662 / 0.6712 |

Base Dice on this same interior is 0.6693. This verifies tile-context
dependence, consistent with GroupNorm's behavior. It does not prove that
Gaussian blending or normalization explains the large whole-volume drop:
this is one selected case, these are unblended local windows, and the measured
change between windows is modest. A controlled normalization/tiling comparison
is a secondary test, not a justified production fix yet.

The small local receptive field and missing explicit prompt/anatomical context
are plausible capacity limitations for long or bilateral 2a findings. A larger
input tile alone does not supply that spatial context. Establish real-patch
fitting and loss behavior before replacing the architecture.

## 5. Confident base misses are harder to correct than uncertain boundaries

Across the late sampled patches, the fractions of base false-negative voxels
with logits below -10 are 27.6%, 36.6%, 55.1% and 27.1% in run order.
These are patch-sample voxel fractions, not whole-cohort estimates; repeated
voxels/findings are possible. Fractions of base FP voxels above +10 are much
smaller: 0.17%, 2.90%, 1.68% and 0.94%.

A voxel starting below -10 requires a positive residual greater than 10 to
cross the 0.5 probability threshold. At logit -10, sigmoid is about 4.54e-5,
so the soft-Dice gradient through sigmoid is weak. BCE still has a useful
logit derivative for a wrong confident voxel, but the current mean divides
each voxel's contribution by roughly seven million on an unpadded patch.
This supports examining error-conditioned gradients and weighted/mined BCE.
It does not justify changing the frozen cache clipping contract, and the
observed residual maxima show that a hard output cap is not the bottleneck.

## What the audit did not find

- All initial tensors match exactly across arms. At epoch 40, stem, residual
  blocks and head weights have changed in every arm. The backbone is learning.
- All inspected parameters are finite FP32. Every optimizer state reports
  4,000 updates; the scaler state is empty and LR is 1e-4. No AMP overflow
  retries occurred in the first 4,000 updates of any arm.
- Gradient clipping remains 1.0. In updates 3,001–4,000, preclip norms exceed
  1 on 20.2%, 28.7%, 53.2%, and 54.3% of steps; mean norms are 0.650, 0.799,
  1.211 and 1.585. Clipping is active, but these facts do not show that it
  prevents learning or that LR is the primary problem.
- L1 residual penalties average 0.000037, 0.000100, 0.000327 and 0.000237 in
  that block, much smaller numerically than Dice losses. Scalar sizes alone
  cannot rule out gradient effects; a component-gradient or zero-L1 ablation
  is more informative than inferring dominance from the total loss.
- Baseline provenance, per-finding identities and metric recomposition agree.
  The existing implementation's orientation and zero-residual tests also
  passed. This is evidence against gross metric/cache mismatch, not a new
  exhaustive visual proof of CT/GT alignment.

## Suggested next work, in priority order

1. **Run a short, fixed real-patch fitting diagnostic.** Use a few reviewed
   findings, fixed positive and empty/hard-negative 192³ patches, common
   pristine initialization, and paired base/refined metrics. Track the exact
   training patches, different windows from the same cases, and full volumes
   separately. This separates ability to fit from context/generalization loss.
   Keep this diagnostic outside the ongoing four-arm experiment.
2. **Compare the current objective with one explicit negative/error-aware
   objective.** Log nonempty/empty losses, FP/FN edits, foreground and hard-
   background BCE components and their gradient norms. Correct the negative
   signal rather than merely deleting the near-constant empty Dice term.
3. **Review several helped and harmed cases with CT, original prompt, GT and
   base/refined overlays.** Determine whether new positives are missed labels,
   wrong structures, boundary expansion, or prompt-inappropriate findings.
   This is necessary before attributing failure to annotation completeness.
4. **If the diagnostic fits patches but loses gains on the same cases' full
   volumes, test window consistency and sampling.** Compare a normalization
   without whole-tile spatial statistics and/or matched training-window
   distribution, one change at a time, with the same full-volume evaluation.
5. **Only then test stronger context or optimizer/regularizer changes.** A
   multiscale refiner or explicit prompt/anatomical input addresses different
   limitations from changing LR, L1, or logit calibration. Preserve separate
   ablations and avoid deciding from A-trained or full mixed-exposure scores.

The present data do not yet justify abandoning residual refinement as a method.
They do justify treating this particular objective and patch-to-volume behavior
as unresolved. Proposed follow-ups await the user's choice; the authorized
10,000-update run and its evaluation schedule remain unchanged.
