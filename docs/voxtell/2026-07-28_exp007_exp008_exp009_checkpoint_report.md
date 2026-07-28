---
created: 2026-07-28
updated: 2026-07-28
status: active
report_type: checkpoint_model_comparison
related_experiments:
  - "006_voxtell_cached_native_v123_lr_ablation"
  - "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
  - "008_voxtell_dual_branch_proposal_refinement_ablation"
  - "009_voxtell_s3_attention_coupling_ablation"
---

# Exp007/Exp008/Exp009 Checkpoint Report

This report compares the current VoxTell continuation family around checkpoint
A. Checkpoint A is exp006 `v123_cached_e5_d4` at epoch 100: VoxTell v1.1
pretrained weights fine-tuned with the v123 cached-native recipe for 100
epochs.

Exp008 and exp009 both start from checkpoint A, reset optimizer/scheduler state,
and train another 100 epochs / 10,000 updates with poly LR decay. Exp009 also
contains `baseline_cont100`, the plain unmodified continuation from checkpoint
A. That row is the clean continuation control for exp009 and the best matched
external continuation control for exp008, with the caveat that exp008 did not
include its own concurrent plain-control arm.

All metric cells are:

```text
Dice / hit rate (hits/total)
```

Dice is `mean_global_dice_per_finding` from
`val_quick_global_eval_summary.json`.

## Model Lineage

| Row | Model | Experiment | Starting point | Change | Role |
| --- | --- | --- | --- | --- | --- |
| 1 | `v123_cached_e5_d4` epoch 100 | exp006 | VoxTell v1.1 pretrained | v123 cached-native single-GPU fine-tune | checkpoint A |
| 2 | `baseline_cont100` | exp009 | checkpoint A | plain continuation, no architecture change | shared continuation control |
| 3 | `v1_sharedfusion_softguide` | exp008 | checkpoint A | shared-fusion proposal/refinement | dual-head candidate |
| 4 | `v1_dualfusion_softguide` | exp008 | checkpoint A | dual-fusion proposal/refinement | dual-head candidate |
| 5 | `v2_dualfusion_precision` | exp008 | checkpoint A | dual-fusion plus precision-biased refinement | dual-head candidate |
| 6 | `v3_dualfusion_softguide_joint` | exp008 | checkpoint A | dual-fusion soft guidance with joint gradients | dual-head candidate |
| 7 | `s3v1_fixedrho_suppress_half_quarter` | exp009 | checkpoint A | `1/2 + 1/4` suppress-only S3 feature gate | attention candidate |
| 8 | `s3v2_balanced_feature_half_quarter` | exp009 | checkpoint A | `1/2 + 1/4` balanced boost/suppress S3 gate | attention candidate |
| 9 | `s3v3_logit_residual_half_quarter` | exp009 | checkpoint A | `1/2 + 1/4` S3 logit residual | attention candidate |
| ref | DDP4 epoch 50 | exp007 | VoxTell v1.1 pretrained | v123 cached-native DDP global batch 4 | val200 reference |
| ref | DDP4 epoch 100 | exp007 | VoxTell v1.1 pretrained | v123 cached-native DDP global batch 4 | val200 reference |

## Final Val200: 9 Primary Models

| Model | Family | Val200 epoch 100 |
| --- | --- | ---: |
| exp006 `v123_cached_e5_d4` checkpoint A | source | 0.3241 / 0.7612 (290/381) |
| exp009 `baseline_cont100` | plain continuation | 0.3398 / 0.7559 (288/381) |
| exp008 `v1_sharedfusion_softguide` | dual-head | 0.3366 / 0.7533 (287/381) |
| exp008 `v1_dualfusion_softguide` | dual-head | 0.3360 / 0.7507 (286/381) |
| exp008 `v2_dualfusion_precision` | dual-head | 0.3333 / 0.7533 (287/381) |
| exp008 `v3_dualfusion_softguide_joint` | dual-head | 0.3345 / 0.7559 (288/381) |
| exp009 `s3v1_fixedrho_suppress_half_quarter` | S3 attention | 0.3347 / 0.7480 (285/381) |
| exp009 `s3v2_balanced_feature_half_quarter` | S3 attention | 0.3367 / 0.7664 (292/381) |
| exp009 `s3v3_logit_residual_half_quarter` | S3 attention | 0.3327 / 0.7612 (290/381) |

Among the nine primary rows, exp009 plain continuation has the highest val200
Dice. Exp009 `s3v2_balanced_feature_half_quarter` has the best val200 hit rate
and hit count.

## Table 1: Dual-Head Val200 Comparison

| Model | Role | Val200 |
| --- | --- | ---: |
| exp006 `v123_cached_e5_d4` checkpoint A | source checkpoint | 0.3241 / 0.7612 (290/381) |
| exp009 `baseline_cont100` | plain continuation control | 0.3398 / 0.7559 (288/381) |
| exp008 `v1_sharedfusion_softguide` | dual-head | 0.3366 / 0.7533 (287/381) |
| exp008 `v1_dualfusion_softguide` | dual-head | 0.3360 / 0.7507 (286/381) |
| exp008 `v2_dualfusion_precision` | dual-head | 0.3333 / 0.7533 (287/381) |
| exp008 `v3_dualfusion_softguide_joint` | dual-head | 0.3345 / 0.7559 (288/381) |
| exp007 DDP4 epoch 50 | reference | 0.3333 / 0.7559 (288/381) |
| exp007 DDP4 epoch 100 | reference | 0.3310 / 0.7585 (289/381) |

All four dual-head arms improve Dice over checkpoint A, but none beat the plain
exp009 continuation on val200 Dice. Their hit counts stay below checkpoint A;
`v3_dualfusion_softguide_joint` matches the plain continuation hit count at
`288/381`.

## Table 2: S3 Attention Val200 Comparison

| Model | Role | Val200 |
| --- | --- | ---: |
| exp006 `v123_cached_e5_d4` checkpoint A | source checkpoint | 0.3241 / 0.7612 (290/381) |
| exp009 `baseline_cont100` | plain continuation control | 0.3398 / 0.7559 (288/381) |
| exp009 `s3v1_fixedrho_suppress_half_quarter` | S3 suppress-only gate | 0.3347 / 0.7480 (285/381) |
| exp009 `s3v2_balanced_feature_half_quarter` | S3 balanced feature modulation | 0.3367 / 0.7664 (292/381) |
| exp009 `s3v3_logit_residual_half_quarter` | S3 logit residual | 0.3327 / 0.7612 (290/381) |
| exp007 DDP4 epoch 50 | reference | 0.3333 / 0.7559 (288/381) |
| exp007 DDP4 epoch 100 | reference | 0.3310 / 0.7585 (289/381) |

The S3 arms do not beat the plain continuation on val200 Dice. The useful S3
signal is `s3v2`: it keeps Dice close to the best dual-head rows and raises the
hit count to `292/381`, above both checkpoint A and the plain continuation.

## Table 3: Dual-Head Val20 Evolution

The first row is the original exp006 path from VoxTell pretrained weights to
checkpoint A. The other rows start from checkpoint A at E0 and then continue.
Exp008 did not run E5 val20, so those cells are `--`.

| Model | E0 | E5 | E20 | E40 | E60 | E80 | E100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp006 `v123_cached_e5_d4` trajectory | 0.2986 / 0.5806 (18/31) | 0.2881 / 0.6452 (20/31) | 0.3332 / 0.6452 (20/31) | 0.3462 / 0.7097 (22/31) | 0.3316 / 0.6774 (21/31) | 0.3531 / 0.7097 (22/31) | 0.3907 / 0.7419 (23/31) |
| exp009 `baseline_cont100` | 0.3907 / 0.7419 (23/31) | 0.3188 / 0.6452 (20/31) | 0.3300 / 0.6774 (21/31) | 0.3636 / 0.7419 (23/31) | 0.3522 / 0.7419 (23/31) | 0.3836 / 0.6774 (21/31) | 0.3791 / 0.7097 (22/31) |
| exp008 `v1_sharedfusion_softguide` | 0.3907 / 0.7419 (23/31) | -- | 0.3135 / 0.6774 (21/31) | 0.3246 / 0.7097 (22/31) | 0.3295 / 0.6774 (21/31) | 0.3689 / 0.6774 (21/31) | 0.3696 / 0.7097 (22/31) |
| exp008 `v1_dualfusion_softguide` | 0.3907 / 0.7419 (23/31) | -- | 0.3419 / 0.7419 (23/31) | 0.3612 / 0.7742 (24/31) | 0.3124 / 0.7097 (22/31) | 0.3791 / 0.7097 (22/31) | 0.3722 / 0.7097 (22/31) |
| exp008 `v2_dualfusion_precision` | 0.3907 / 0.7419 (23/31) | -- | 0.3217 / 0.6774 (21/31) | 0.3357 / 0.6774 (21/31) | 0.3024 / 0.7419 (23/31) | 0.3769 / 0.7097 (22/31) | 0.3710 / 0.7097 (22/31) |
| exp008 `v3_dualfusion_softguide_joint` | 0.3907 / 0.7419 (23/31) | -- | 0.2846 / 0.6774 (21/31) | 0.3572 / 0.7419 (23/31) | 0.3585 / 0.7419 (23/31) | 0.3780 / 0.7419 (23/31) | 0.3813 / 0.7419 (23/31) |

Val20 suggests `v3_dualfusion_softguide_joint` is the strongest dual-head row
at E100 on the small probe, but final val200 does not preserve a Dice advantage
over the plain continuation.

## Table 4: S3 Attention Val20 Evolution

| Model | E0 | E5 | E20 | E40 | E60 | E80 | E100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp006 `v123_cached_e5_d4` trajectory | 0.2986 / 0.5806 (18/31) | 0.2881 / 0.6452 (20/31) | 0.3332 / 0.6452 (20/31) | 0.3462 / 0.7097 (22/31) | 0.3316 / 0.6774 (21/31) | 0.3531 / 0.7097 (22/31) | 0.3907 / 0.7419 (23/31) |
| exp009 `baseline_cont100` | 0.3907 / 0.7419 (23/31) | 0.3188 / 0.6452 (20/31) | 0.3300 / 0.6774 (21/31) | 0.3636 / 0.7419 (23/31) | 0.3522 / 0.7419 (23/31) | 0.3836 / 0.6774 (21/31) | 0.3791 / 0.7097 (22/31) |
| exp009 `s3v1_fixedrho_suppress_half_quarter` | 0.3907 / 0.7419 (23/31) | 0.2844 / 0.6452 (20/31) | 0.3287 / 0.7097 (22/31) | 0.3587 / 0.7419 (23/31) | 0.3541 / 0.7419 (23/31) | 0.3881 / 0.6774 (21/31) | 0.4061 / 0.7419 (23/31) |
| exp009 `s3v2_balanced_feature_half_quarter` | 0.3907 / 0.7419 (23/31) | 0.3147 / 0.6452 (20/31) | 0.3087 / 0.6129 (19/31) | 0.3509 / 0.7419 (23/31) | 0.3552 / 0.7097 (22/31) | 0.4052 / 0.7097 (22/31) | 0.4028 / 0.7419 (23/31) |
| exp009 `s3v3_logit_residual_half_quarter` | 0.3907 / 0.7419 (23/31) | 0.3244 / 0.6774 (21/31) | 0.2899 / 0.6452 (20/31) | 0.3404 / 0.6774 (21/31) | 0.3334 / 0.6452 (20/31) | 0.3793 / 0.6774 (21/31) | 0.3822 / 0.7097 (22/31) |

Val20 favors the S3 feature-coupling arms at E100: `s3v1` and `s3v2` both
exceed the plain continuation on Dice and recover checkpoint A's `23/31` hit
count. On val200, only `s3v2` carries a hit-count advantage.

## Interpretation

Final val200 is the primary comparison signal. On val200, the best Dice among
these models is the unmodified exp009 plain continuation, not a dual-head or S3
arm. That means some of the gains from exp008 and exp009 must be interpreted
against the simple fact that continuing checkpoint A for another 10,000 updates
already improves Dice.

The dual-head variants are directionally positive for Dice versus checkpoint A
but do not beat the plain continuation. Their hit rates are slightly lower than
checkpoint A and do not show a clear final advantage.

The S3 variants are mixed. `s3v1` and `s3v2` look strong on val20 at E100, but
only `s3v2` has a final val200 advantage in hit count. This suggests that
balanced S3 modulation may help recall/localization, while fixed-strength S3 is
not yet a clear Dice improvement over plain continuation.

The exp007 DDP4 rows are reference points for the same broader v123 e5/d4
family. They do not change the main conclusion: exp009 plain continuation and
exp009 `s3v2` are the strongest rows here, depending on whether the priority is
val200 Dice or val200 hit count.

Val20 remains a small 20-case / 31-finding probe. It is useful for trajectory
and debugging behavior, but it should not override final val200.
