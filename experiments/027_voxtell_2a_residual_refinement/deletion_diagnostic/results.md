# Deletion diagnostic results

Status: completed, pending user review. No larger training runs launched.

The authorized diagnostic completed 200 FP32 optimizer updates on eight
distinct half-A patients and 24 fixed 192³ patches, then evaluated the same
eight complete volumes. B was excluded from selection, fitting and evaluation.
All eight FP-focus patches are GT-empty. These are deliberately selected
in-sample fitting results, not an estimate of performance on unseen findings.

## Assessment

The editor learned useful selective deletion at a conservative operating point,
with a full-volume benefit beyond simple base-threshold suppression. However,
the default action threshold 0.5 deletes too many TP. A pooled99% TP-retention
target also hides substantial harm to individual small findings. This supports
planning a controlled follow-up with explicit per-finding preservation checks;
it does not justify scaling the unchanged threshold 0.5 recipe.

## Complete-volume results

The conservative removal threshold 0.89594984 and simple base-logit threshold
0.19104004 were each calibrated to retain 99% of TP on the update200 fitting
patches, then held fixed for complete-volume inference. Removal uses score>t;
base suppression removes base-positive voxels with base_logit<t.

| Policy | Mean Dice | TP retained | FP removed | Improved / unchanged / worsened |
| --- | ---: | ---: | ---: | --- |
| Frozen base | 0.327060 | 100% | 0% | — |
| Editor, threshold 0.5 | 0.322993 | 71.145% | 70.018% | 5 / 0 / 3 |
| Editor, patch-calibrated threshold 0.89595 | 0.371671 | 98.813% | 23.240% | 6 / 0 / 2 |
| Base-logit threshold 0.19104 | 0.328792 | 98.980% | 3.150% | 5 / 0 / 3 |

Dice is the macro mean across these eight findings. TP retention and FP removal
pool voxel counts. The conservative editor removes 12,020 annotated FP and 354 TP;
97.139% of deleted voxels are annotated FP. It cannot introduce new foreground.
The eight-finding baseline0.327060 is different from the 69-finding baseline.

| Finding | Base Dice | Conservative editor Dice | TP retained | TP removed |
| --- | ---: | ---: | ---: | ---: |
| train_13155_a_2.nii.gz::2 | 0.033377 | 0.029919 | 88.205% | 69 |
| train_13497_a_1.nii.gz::0 | 0.121240 | 0.076706 | 35.985% | 169 |
| train_2657_a_2.nii.gz::1 | 0.186068 | 0.209777 | 98.363% | 26 |
| train_3020_a_2.nii.gz::0 | 0.321908 | 0.404506 | 93.838% | 79 |
| train_18641_a_2.nii.gz::0 | 0.357795 | 0.578586 | 98.980% | 10 |
| train_19325_a_2.nii.gz::2 | 0.419622 | 0.431928 | 100.000% | 0 |
| train_3015_a_1.nii.gz::0 | 0.515891 | 0.556911 | 100.000% | 0 |
| train_13288_a_2.nii.gz::0 | 0.660580 | 0.685034 | 99.991% | 1 |

The second finding loses169/264 base TP (64.0%), despite the high pooled
retention. This is observed failure evidence, not a hypothetical concern.

## Matched-retention comparison

As an explicitly retrospective diagnostic, calibrating both methods on the
same complete-volume GT to99% pooled TP retention gives exactly99.0006%
retention for both. The editor removes 21.954% of FP with Dice 0.370066; simple
threshold suppression removes 3.101% with Dice 0.328755. This supports useful
error discrimination beyond merely accepting more TP loss. These thresholds
use full training-volume GT and are not held-out selection or adopted settings.
The complete90/95/99% tradeoffs are preserved in the linked JSON.

## Patch fitting and tile context

| Update | Mean loss | All-patch Dice | TP retained | FP removed |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.0641 | 0.287201 | 100.000% | 0.000% |
| 50 | 1.5270 | 0.264909 | 97.053% | 11.678% |
| 100 | 1.2041 | 0.397236 | 95.534% | 25.227% |
| 150 | 1.2574 | 0.546164 | 81.843% | 53.783% |
| 200 | 1.3906 | 0.542600 | 75.254% | 63.423% |

These rows use the fixed0.5 threshold; no checkpoint is selected automatically.
At update200, six of eight GT-empty patches become completely empty, which
contributes strongly to mean patch Dice. On the 16 GT-containing patches alone,
Dice is0.438900 versus base 0.430801. At the conservative patch-calibrated
threshold, GT-containing patch Dice is0.472519 and four empty patches clear.
Overall conservative patch Dice is0.481679, with 99.0003% TP retention and
20.459% FP removal, versus 2.697% FP removal for the simple threshold at exactly
the same patch retention.

For the same base-positive voxels, isolated fitting-patch removal scores and
full-volume blended scores differ by 0.07336 on average, weighted by patch
voxel count. At threshold 0.5,10.881% of these decisions differ. This establishes
tile-context/blending sensitivity; it does not isolate GroupNorm as the cause.
Training on more varied eligible windows and checking per-finding damage should
precede claims of reliable whole-volume editing.

## Verification and timing

CPU verification: 11 tests passed in the VoxTell Docker image with GPUs disabled.
The initial test run caught a threshold dtype edge case; it was fixed and the
complete diagnostic test suite passed before launch. Canonical repository
workflow checks passed with 19 existing missing-artifact warnings; shell syntax,
explicit compilation of the new entrypoint and whitespace checks passed.

Post-run checks pass: exact200-update schedule, all five checkpoints and optimizer
cursors, finite FP32 parameters/losses/gradients, unchanged source/cache files,
eight unique A patients, complete eight-finding coverage, original-geometry
baseline reproduction and deletion-only outputs. Figures were visually inspected
for readable labels and overlays; this is not a clinical annotation review.

- Cache/patch preparation:51.112 seconds.
- Measured optimizer-update loops:202.103 seconds; mean 1.011 s/update,
  p50 1.017 s, p95 1.042 s; mean data loading0.0195 s/update.
- Complete-volume phase:172.557 seconds, including111.293 seconds of inference.
- Total training plus all patch/full-volume checks and reporting:429.537 seconds
  (7 min10 s), plus the separate51-second preparation and Docker startup.
- Peak allocated GPU memory:11.933 GiB. No OOM or non-finite failure.
- Complete-volume grid:675 tiles,270 active; all original CT extents covered.
- Diagnostic runtime files:approximately 1.72 GB, excluding reused source caches.
- Completed2026-09-10 at 16:31:03 UTC. Worker exited; GPUs 0–3 verified idle.

GPU launch command:

```bash
START_GPU_WORK=1 DIAGNOSTIC_GPU=0 bash scripts/rexgroundingct/run_027_deletion_host.sh run
```

- [Live report](../runtime/deletion_diagnostic_v1/report.md)
- [Fixed inputs and provenance](../runtime/deletion_diagnostic_v1/manifest.json)
- [Launch log](../runtime/deletion_diagnostic_v1/launch.log)
- [Complete-volume per-finding results](../runtime/deletion_diagnostic_v1/full_volume/per_finding.csv)
- [Verification](../runtime/deletion_diagnostic_v1/verification.json)
- [Matched-retention comparison](../runtime/deletion_diagnostic_v1/matched_retention_comparison.json)
- [Patch subgroup history](../runtime/deletion_diagnostic_v1/patch_subgroup_history.json)
- [Tile-context comparison](../runtime/deletion_diagnostic_v1/tile_context_comparison.json)

## Next decision

Plan further deletion experiments around per-finding TP preservation, a declared
conservative operating-point procedure, broader eligible-window sampling and
the existing base-threshold comparator. Keep FN addition deferred. Any future
generalization claim needs unseen patients. Do not initialize full runs from
this disposable diagnostic or automatically resume the stopped residual arms.
Large training runs remain for user planning and review.
