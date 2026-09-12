---
created: 2026-09-11
updated: 2026-09-11
status: authorized
experiment_id: "027_voxtell_2a_residual_refinement"
---

# Four category-specific deletion runs after the 2a epoch-60 stop

## Intent and decisions

The user requests stopping the four 2a loss arms after all epoch-60 evaluations,
then starting four category-specific models concurrently. Interpret the repeated
`2d/c/d/e` as the four just-cached categories 2b/2c/2d/2e, in that fixed order.
Every model uses original training data only, the same Dice + BCE TP2 recipe,
50 epochs of 100 updates (5000 total), and the existing validation protocol.
The user confirmed 2b/2c/2d/2e and exclusion of the single overlapping 2d
training finding `train_2936_a_1.nii.gz::3`. All validation findings remain.
Other categories have no such overlap. Reuse the existing A/B patient split unchanged.

## Stop and resource boundary

2a was evaluating update 6000 when the instruction arrived. Hold only its
coordinator process so evaluator and CPU report/analysis processes can finish
without launching update 6001. Verify all four 69-finding summaries, checkpoint
hashes and saved score analyses, preserve the completed barrier, then terminate
the coordinator gracefully and record `stopped_by_user` at update 6000. Do not
change its canonical 100-epoch config, source files, schedules or checkpoints.
The new GPU launch requires a verified stop receipt and no active 2a workers.

## Inputs, preprocessing and schedule

Reuse the completed `cache_2bcde_v1` inventory (2730 CTs/5106 findings), its
array/source proofs, original finding IDs, prompts and immutable category views.
Check all proof file identities and tile-index hashes before training; missing
or altered arrays block all trainers. Never invoke VoxTell in this experiment.
Cache metadata contains exact whole-finding TP/FP/FN/GT and eligible tile counts.
Use every eligible original training finding for the selected category, except
the explicitly recorded overlap exclusion. Validation includes all findings,
including those with no base-positive voxels.

Native cropped-volume z-score normalization, geometry and absence of resampling
or augmentation remain unchanged. Use 192-cubed patches, batch 1, valid-voxel
padding exclusion. Per epoch, uniformly sample findings and shuffle the existing
50 TP-containing / 25 FP-containing / 25 eligible-tile branches, with recorded
same-finding fallbacks. Save one immutable 5000-event schedule per category,
using independent reproducible seeds 20260911–20260914 in fixed category order.
Reuse the common historical pristine weights (SHA256 of weights
`ff8d94ec3648a9ef24a0a2d4865c0590782da60a522b7f4b701cf57748cbba73`), zero
head weights, initial removal probability 0.05, fresh AdamW and RNG seed 20261612.

## Model, loss and evaluation

Keep the existing deletion editor architecture and inference helpers. Training
and evaluation use FP32, TF32 disabled, constant LR 1e-4, weight decay 1e-4,
gradient norm clipping 1.0 and no accumulation. All models optimize
`D + 0.25 * (F + 2*K) / 3`, with the existing finding-aware patch Dice surrogate,
full-finding GT including false negatives, independent FP/TP BCE means,
differentiable zero for missing groups and Dice epsilon 1e-6. Thresholds do not
enter gradients. Log total loss, F, K, D, pre-clipping gradient norm and timings.

Evaluate at updates 100, 500, 1000, 2000, 3000, 4000 and 5000 (epochs 1, 5, 10,
20, 30, 40 and 50). Use synchronous four-model barriers. Each model processes
its full category validation set once: 49/60/132/11 findings respectively.
Use the existing base-positive tile gate, 192-cubed tiles, 50% overlap, FP32
Gaussian blending of removal probabilities and original-geometry restoration.
Final foreground remains a subset of the frozen base mask.

Retain six live thresholds (0.5, 0.8, 0.9, 0.95, 0.99, 1), both display
thresholds 0.50/0.90, frozen-base controls and CPU-only 201-point threshold
sweeps from saved FP32 scores. Derive A/B/full metrics from the same finding
records. Keep per-finding Dice, hits, damage, TP loss, FP removal, precision,
recall and timings. Full baseline Dice is 0.357030/0.415254/0.394980/0.410408.

## Live reporting and recovery

One CPU writer polls every second and updates training curves within about five
seconds, using immediate append-only histories. The 3x3 dashboard has total
loss/F/K on row 1, finding-aware D/2b full Dice/2c full Dice on row 2, and
gradient norm/2d full Dice/2e full Dice on row 3. Training overlays use fixed
category colors. Each category Dice panel shows completed 0.50 and 0.90 curves
plus its own frozen-base reference. A running evaluation is labeled pending
with finding progress; no provisional Dice points or A/B plots. Publish each
category's completed result immediately, even while another category evaluates.
Raw and smoothed training curves remain distinguishable. Save A/B/full JSON/CSV
and full-cohort threshold figures, with no automatic ranking or selection.

Save full-state checkpoints every 100 updates and on graceful interruption,
including category/loss identity, optimizer, RNG, schedule cursor and provenance.
Resume without duplicate histories or missing evaluations. Stop the group on
OOM, non-finite data/results or integrity failures; preserve evidence. No silent
precision, patch-size, threshold or loss changes. Stop at `pending_user_review`
after 20000 total updates, 28 complete evaluations and 28 dense analyses.

## Verification and launch

Use separate category adapters so prior runs and cache producer code remain
unchanged. Add prepare/train/evaluate/report/analyze/audit/dry-run/resume
commands. CPU tests in the existing GPU-disabled Docker image cover category
membership and exclusion, frozen A/B views, exact loss formulas, reproducible
schedules and common initialization, cache gating, deletion-only inference,
threshold semantics, resume, all seven barriers and completed-only plot points.
Inspect a labeled synthetic dashboard with delayed and completed categories.
Run repository and whitespace checks before the authorized launch:

```bash
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_categories_host.sh orchestrate --gpus 0 1 2 3
```

Config: `configs/experiments/027_deletion_categories_bcde_50ep.json`.
Runtime: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/deletion_categories_bcde_50ep`.
All heavy artifacts are external; stable README links and a small verification
record remain in the repository.
