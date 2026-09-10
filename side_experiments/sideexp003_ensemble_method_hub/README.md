---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# SideExp003 Ensemble Method Hub

SideExp003 is the single tracked control plane for future RexGroundingCT
ensembles. It owns the fixed-val200 checkpoint catalog, reproducible method/run
contracts, and the provenance for the shared logit cache. Large arrays, logs,
and progress files live only under:

```text
/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/
  sideexp003_ensemble_method_hub/
```

This is intentionally a side experiment. Do not add it to
`experiments/registry.yaml` or `configs/experiments/`.

## Current milestone

Foundation steps 1–4 are implemented. The user has additionally authorized a
fresh-only val200 export for the frozen top 20 leaderboard checkpoints. This
job deliberately ignores all SideExp002 logits, runs five consecutive waves of
four checkpoints, and publishes only new content-addressed SideExp003 caches.

Ranks 1–8 completed under `j001_top20_val200_fresh`. Repository source changed
before Wave 3 could launch, so ranks 9–20 continue under the separately hashed
`j002_top20_val200_fresh_r09_r20` job. Its source-drift audit verifies the eight
parent strict caches, retains original ranks and wave numbers 3–5, and assigns
new cache keys bound to the continuation source bundle. It never relabels or
overwrites a `j001` cache.

The cache job alone does not authorize greedy search. Later explicit user
decisions authorized metrics-only top-4 and top-8 full-val diagnostics. They
are not OOF results, do not set `Mmax`, and cannot become formal recipes.

## Catalog workflow

`checkpoint_catalog.json` is the only ranking data source. Both Markdown files
are atomically rendered from it and must not be edited by hand.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  catalog add --experiment 027 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  catalog add --experiment 027 --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py check
```

The scanner accepts only single-checkpoint fixed-val200 evaluators with exactly
200 cases and 381 findings. It recomposes all metrics after joining to the
official category map by `(case name, finding index)`. Checkpoints are unique by
SHA-256. Conflicting duplicate evaluators stop the update and print an override
stub; the tool never guesses which source is canonical. Merge a reviewed choice
into `catalog_overrides.json`, including a non-empty rationale, then rerun the
dry-run. Empty or stale override paths remain hard failures.

Rows with missing checkpoint or threshold provenance remain visible as
`needs_review`, but roster construction excludes them by default.

## Runtime layout

```text
sideexp003_ensemble_method_hub/
├── cache/logits/
│   ├── legacy_sideexp002/
│   └── by_cache_key/
├── cache/jobs/<job-id>/
└── methods/<method>/runs/rNNN_<roster-slug>/
```

Tracked run folders mirror the method/run relative path and contain immutable
specifications, result JSON, reports, and the final decision. Runtime folders
contain only large caches, progress, logs, and materialized logits.

Cache keys bind dataset, checkpoint, config/inference contract,
preprocessing-manifest, source bundle, container image, and storage contract.
Fresh exports stage evaluator-layout `(F,X,Y,Z)` float32 pre-sigmoid logits
clipped to `[-30,30]`. They publish float16 only when an offline cast preserves
the threshold-zero mask at every voxel; otherwise the same staged inference is
published as float32. Dtype fallback never invokes the model twice.

## Fresh top-20 cache workflow

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  roster freeze --top 20 --metric overall_dice \
  --roster-id top20_val200_dice_20260907T211605Z

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache plan --roster top20_val200_dice_20260907T211605Z \
  --split val200 --wave-size 4 --reuse-policy fresh-only --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache run --job j001_top20_val200_fresh --auto-continue

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache watch --job j001_top20_val200_fresh --interval 60
```

When a frozen source bundle changes between waves, create a separate audited
continuation rather than changing the original job specification:

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache continue --parent-job j001_top20_val200_fresh \
  --job-id j002_top20_val200_fresh_r09_r20 --start-rank 9 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache continue --parent-job j001_top20_val200_fresh \
  --job-id j002_top20_val200_fresh_r09_r20 --start-rank 9 --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache run --job j002_top20_val200_fresh_r09_r20 --auto-continue
```

Each worker receives one physical GPU. Every wave retains its largest-case
prediction as a smoke result, requires 20,000 MiB free before launch and 4,096
MiB after the smoke, and does not release the remaining cases until all four
workers pass. Any candidate failure prevents the next wave from starting.

## Top-4 preliminary comparison

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble compare --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --run-id r001_top4_fresh_val200 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble compare --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --run-id r001_top4_fresh_val200 --apply
```

The worker is CPU-only and single-process. It streams sigmoid probabilities,
uses threshold `>= 0.5`, records generation and evaluation timing separately,
and writes metrics only. Its full-val Caruana path starts at rank 1 and adds
three basket positions with replacement.

For an independently frozen top-k uniform diagnostic:

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble uniform --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --top 8 \
  --run-id r002_top8_fresh_val200 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble uniform --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --top 8 \
  --run-id r002_top8_fresh_val200 --apply
```

Test300 uses the same split-aware artifact schema. The separate GPU8 job below
prepares 300-case/582-prompt caches with CT geometry and exact storage checks.
Its labels are withheld, so it cannot report Dice or hits.

## Wave-4-gated top-16 seeded diagnostic

The authorized top-16 diagnostic combines strict ranks 1–8 from `j001` with
strict ranks 9–16 from `j002`. It polls every ten minutes, starts while Wave 5
continues, computes the top-16 uniform result, and then runs independent
Caruana-with-replacement curves for `all/2a/2b/2c/2d` from frozen per-scope
K=4 seeds. Five CPU workers shard cases and share every model read across
scopes. No derived logits are written.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble seeded-plan \
  --roster top20_val200_dice_20260907T211605Z \
  --source-job j001_top20_val200_fresh \
  --source-job j002_top20_val200_fresh_r09_r20 \
  --top 16 --workers 5 \
  --job a001_top16_after_wave4_fullval --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble seeded-run --job a001_top16_after_wave4_fullval \
  --poll-interval 600 --auto-start
```

Use `ensemble seeded-watch --job a001_top16_after_wave4_fullval --once` for a
one-shot status snapshot. These results are optimistic same-val diagnostics,
not OOF estimates or final recipes.

See [codex_execution_spec.md](codex_execution_spec.md) for gates and
[prior_ensemble_summary.md](prior_ensemble_summary.md) for historical evidence.

## GPU8 resume

The authorized CPU benchmark and K=16 continuation use
`resume_seeded_caruana.py`, outside the frozen scoring modules. Read
[gpu8_resume_execution_spec.md](gpu8_resume_execution_spec.md) before launch.
The helper holds the parent shared launch lock for the complete session and
requires the GPU9 worker and supervisor to have relinquished ownership.

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp003_ensemble_method_hub/resume_seeded_caruana.py \
  run --session-id gpu8_YYYYMMDDTHHMMSSZ
```

Run this host-side under a detached process. Session state, benchmark reports,
logs, archived parent evidence, and closeout live under the parent analysis
runtime's `resume_sessions/<session-id>/`. Higher concurrency creates the
separate `a002_gpu8_top16_seed4_resume` continuation and a tracked
`resume_spec.json`; that specification is not an input to `ensemble seeded-run`.
An interrupted helper does not automatically retry. Preserve completed partials
and inspect its session state before arranging another takeover.

See [gpu8_resume_launch_report.md](gpu8_resume_launch_report.md) for the measured
benchmark decision and the session's live-state and closeout locations.

See [val200_top16_comparison.md](val200_top16_comparison.md) for the completed
K=4–16 curves, the verified uniform top-16 baseline, checkpoint additions, and
final normalized weights across all five scopes.

## GPU8 top-20 test300 caches

**2026-09-10 update:** profiling is complete and the user authorized Waves 2–5
with four CPU workers. The separate [j004 continuation](cache_jobs/j004_test300_r05_r20_cpu4_gpu8/README.md)
now owns that work. Original j003 remains closed after Wave 1; use
`continue_test300.py watch`. The historical hold/recovery notes below do not
authorize restarting j003. d1/d2/d3 are complete.


Current user decision: finish Wave 1 publication/validation, then hold ranks
5–20 pending a test-versus-val performance investigation. The separate
`finish_test300_wave1.py` guard has stopped only the scheduler; the four
existing GPU workers continue. See [the wave hold specification](finish_test300_wave1_spec.md)
and runtime `finish_wave1_state.json`. Do not restart or manually resume the
coordinator while this hold is in force.

The 2026-09-09 recovery uses `resume_test300_no_pause.py` under
[the no-pause recovery policy](test300_no_pause_execution_spec.md). The user
accepts contention and requires uninterrupted val200 CPU search. Use that
launcher for restarts; the original `hub.py cache run` command below retains
its frozen automatic CPU-pausing behavior. Inference code, job/cache identities
and staging validation are unchanged. Read the job's `recovery_report.md` for
the latest launch evidence.

Read [test300_execution_spec.md](test300_execution_spec.md). Job
`j003_top20_test300_fresh_gpu8` preserves the frozen val200 roster and original
ranks. `test300_cache.py` implements image-only inference and publication;
`test300_runner.py` coordinates five four-GPU waves after the CPU benchmark.
Its `v3_` keys bind the test split, CT geometry, preprocessing case manifests,
checkpoint/config hashes, source bundle, image ID, and storage policy.

Build the separate native test cache with the existing preprocessing script,
`--preprocess-id crop_zscore_native_v1 --splits test`, and cache root
`/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_test300_v1`.
The runtime records the complete preprocessing command. The original train/val
native cache and complete iso07 cache remain immutable.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py cache plan \
  --roster top20_val200_dice_20260907T211605Z \
  --job-id j003_top20_test300_fresh_gpu8 --split test300 \
  --wave-size 4 --reuse-policy fresh-only --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py cache run \
  --job j003_top20_test300_fresh_gpu8 --auto-continue

python side_experiments/sideexp003_ensemble_method_hub/hub.py cache watch \
  --job j003_top20_test300_fresh_gpu8 --once
```

Run the coordinator detached on gpu8. It stages float32 under
`/data/hengjie/sideexp003_staging/j003_top20_test300_fresh_gpu8/`, then atomically
publishes shared float16 arrays only if every threshold-zero mask voxel is
preserved; otherwise it publishes float32. It removes only its own local stage
after all published cases pass validation. GPU containers hide segmentation
files. Test status updates affect only the catalog's `test300` artifacts.

Runtime state, commands, worker logs, pause ownership, per-wave timings, and
the paired val/test inventory live under
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j003_top20_test300_fresh_gpu8/`.
The coordinator monitors GPU, disk, and CPU-search progress every minute.
Insufficient projected wave capacity waits without deleting caches; the shared
reserve is 20 TiB. SIGTERM or a runtime `control/ABORT` file stops its workers
and restores any CPU container pause it owns. Remove an intentional ABORT
sentinel before restarting. A live owner, changed provenance, or corrupt
completed case is rejected. Keep the frozen source files unchanged while this
job runs. Restarting the same command validates and reuses this job's outputs.

Tests: run `python -m unittest discover -s
side_experiments/sideexp003_ensemble_method_hub -p 'test_*.py'` inside the frozen
VoxTell image. The geometry tests require that image's dependencies. Final
inventory/report files are copied into the tracked job folder on completion;
ensemble choices and submission are separate, later work.

## Automatic top-four test submission packages

The separate frozen [s001 job](submission_jobs/s001_top4_test300_d123/README.md)
is armed in detached CPU-only container `sideexp003_s001_top4_test300_d123`.
It waits for the Wave 1 finish guard and all four strict caches, then generates
d1 (equal probability average), d2 (eligible whole-lung 20 mm support) and d3
(eligible prompt-selected 20 mm support), plus one verified ZIP per variant.
The largest CT and largest finding-array case are retained smoke outputs.
Ranks 5–20 stay held; frozen exporter and scoring sources remain unchanged.
See [the launch report](submission_jobs/s001_top4_test300_d123/launch_report.md)
for the exact container command and one-shot status command. Live state and
final manifests belong to the separate shared `submission_jobs/s001_top4_test300_d123`
runtime; Exp024 a/b outputs and original manifests are preserved.

## Test300 pipeline profiling

The isolated [p001 benchmark](profiling_jobs/p001_test300_pipeline16_gpu8/README.md)
measures 16 model–case jobs on four GPUs and CPU replay at 4/8/16 workers.
It never resumes production Waves 2–5 or changes the frozen exporter.
