---
created: 2026-09-11
updated: 2026-09-11
status: implementation_verified
---

# Verification and handoff

The 2a coordinator was held while its epoch-60 evaluators finished, preventing
update 6001. All four 69-finding summaries, checkpoint hashes and 201-threshold
analyses passed verification. The coordinator exited after TERM then CONT; every
history ends at update 6000 (24,000 updates total). Stop receipt:
[epoch-60 evidence](../runtime/deletion_loss_ablation_train_100ep/reports/user_stop_epoch60.json).
The old canonical config, schedules, checkpoints and all 26 bound source hashes
remain unchanged. Status is `stopped_by_user`, with results pending user review.

All 15 category CPU tests passed in 22.577 seconds in the existing
`rexgroundingct-voxtell:cu126` image, without GPU devices. They cover category
membership and the confirmed patient exclusion, deterministic 5000-event
schedules and per-epoch patch mixtures, common historical pristine weights,
FP32 settings, seven four-model barriers with different cohort sizes, input
corruption, GPU opt-in and Docker-free dry runs, exact loss values/gradients,
empty targets and groups, full-finding Dice including FN, threshold ties and
threshold-one identity, deletion-only overlapping/small-volume inference,
interrupted evaluation without repeated inference, dense threshold analysis,
training interruption at update 2107 and bitwise-equivalent resume through 2200,
and independently published completed-only category plots.

All 29 shared-cache CPU regression tests also passed (3.309 seconds), including
original geometry, prompt/channel ordering, exact historical tile indexing,
empty predictions, cache-only loading, storage-mask preservation and integrity
checks. The repository workflow check passed with 19 pre-existing missing
runtime/sync warnings. Python compilation and Git whitespace checks passed.

The labeled synthetic 3x3 dashboard was inspected: one category has completed
results and three are pending; no partial Dice values enter validation curves.
The final table additions for elapsed time and tile progress passed a focused
CPU dashboard regression. Baselines, thresholds 0.50/0.90, source exposure and
A/B/full record links remain separate from incomplete evaluation progress.

Real preparation verified all 2730 CTs/5106 cached findings and reproduced
full-validation baselines 0.357030/0.415254/0.394980/0.410408. A/B counts remain
25/24, 31/29, 61/71 and 8/3. The eligible training pools contain
1365/1506/1727/236 findings. After the confirmed 2d exclusion, each model's
training patients are disjoint from all its validation patients.

All four immutable schedules contain 5000 events. Uniform sampling with
replacement visits 1341/1450/1629/236 unique findings over those events;
an epoch is 100 updates, not a complete pass through the pool. Common pristine
weights passed the pinned hash check. Execution context:
`9e07ee08369c03000bdcac3114a0ed8a81693efcef938debc2cf6d98acbeffa5`.

The authorized detached launch started container
`rex027_deletion_categories_bcde_50ep` / `0b3640cdc9a4`, assigning
2b/2c/2d/2e to GPUs 0/1/2/3. Source snapshots, preparation counts, CPU verification
records and the synthetic dashboard are preserved under runtime `verification/`.
The job will stop after 5000 updates per category and seven complete evaluation
barriers with status `pending_user_review`; no automatic continuation or ranking.


At 2026-09-11 16:26:30 UTC (09:26 Pacific), live startup verification found
15/14/13/15 completed updates in 2b/2c/2d/2e order, all finite. All four initial
checkpoints exactly match the historical pristine weight hash, record FP32 with
TF32 disabled and use `dice_bce_tp2`. Peak allocated memory was about 12 GiB per
worker. The dashboard was actively refreshing training histories and displaying
pending validation with each category's correct cached baseline. Live evidence:
`verification/initial_live_check.json`. First evaluation is update 100; subsequent
evaluations remain at 500/1000/2000/3000/4000/5000.
