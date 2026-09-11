---
created: 2026-09-09
updated: 2026-09-09
status: training
---

# Full FP32 run: verification and launch

The user authorized complete training-2a caching followed by four concurrent
FP32 refiners. The schedule is 100 updates per epoch for 100 epochs (10,000
optimizer updates per arm), with synchronous complete 2a evaluation at epochs
10, 20, …, 100. The run starts from common pristine weights, using AdamW LR
1e-4, gradient clipping 1.0 and 192³ tiles. FP32 is now the repository default
for refinement training/evaluation; the frozen base export remains unchanged.

The repository workflow check and Git whitespace checks passed. Experiment
consistency reports 19 existing warnings in other experiments. No files were
staged or committed.

The authorized orchestration launched at 2026-09-10 06:50 UTC, host PID
`2754829`. All four cache workers started with 216 disjoint CTs each, and the
independent board is refreshing. Current progress is in the live dashboard;
training waits for the complete-cache gate.

Complete-cache verification passed at 2026-09-10 08:32 UTC for all 864 CTs /
1,189 findings. Preparation took 6,132.993 seconds (102.2 minutes), generating
668 CT caches and reusing 196. All four FP32 trainers started automatically;
the first full-run evaluation is scheduled at update 1,000. No failures were
reported at the 08:42 UTC status check.

Interim [results through epoch 40](full_run_progress.md) have been checked for
complete finding coverage and exact metric recomposition. Training continues
toward epoch 50 as of 2026-09-10 14:28 UTC; no failures are reported.

## Launch and monitoring

```bash
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_host.sh orchestrate --gpus 0 1 2 3
```

Without `START_GPU_WORK=1`, this is a non-mutating dry run. The authorized
command first completes and verifies 801 training CTs / 1,120 findings and
63 validation CTs / 69 findings. Missing or invalid caches block all trainers.
Run membership remains train; train+A (50/50 per epoch); A; A+B.

- [Full-run live dashboard](runtime/full_fp32_100ep/reports/live_dashboard.md)
- [Live subplot figure](runtime/full_fp32_100ep/reports/live_dashboard.png)
- [Launch provenance](runtime/full_fp32_100ep/run_manifest.json)
- [Verified cache inventory](runtime/full_fp32_100ep/cache_inventory.json)
- [Completed evaluation barriers](runtime/full_fp32_100ep/barriers.json)
- [Labeled synthetic dashboard fixture](runtime/full_fp32_100ep/verification/synthetic_live_dashboard/live_dashboard.md)
- [Benchmark comparison, preserved](precision_benchmark_results.md)

The runtime is the separate `full_fp32_100ep` child of Exp027's external root.
Its `cache` symlink references the existing shared arrays; no benchmark data,
checkpoints or reports are overwritten. All old source files named by the
benchmark's code manifest are preserved under the full run's
`config/benchmark_source_before_full`, with their original SHA256 hashes.

## Verification evidence

- 36 CPU contract/controller tests pass, including ten complete orchestration
  barriers, full-pool cache selection, missing/corrupt/changed cache rejection,
  immutable full schedules, source mixtures, GPU gates, append-only journal
  recovery, partial validation and final per-arm publication.
- 16 CPU PyTorch tests pass in the existing VoxTell Docker image with GPUs
  disabled. These include finite FP32 gradients, synthetic fitting, orientation,
  tiling, checkpoint/resume and failure recovery, and a mocked frozen-model
  export preserving all prompt positions while saving only required channels.
- The cache producer's exported computation has an AST-identical body to the
  benchmark implementation. Orientation, clipping/storage and point-sampling
  helpers are unchanged, as are upstream inference/preprocessing source hashes.
  The [input-bound compatibility proof](cache_compatibility_full.json) accepts
  exactly the verified legacy contract alongside the new producer-only contract.
  It does not rewrite existing metadata or bypass array/source verification.
- Representative train/A/B caches pass full CT, logit, target, geometry and
  provenance verification. Full verification of all 864 cases is a mandatory
  launch stage, not inferred from these examples.
- A real independent CPU report process published training updates after
  1.639 and 4.114 seconds and a provisional finding update after 2.170 seconds.
  The synthetic 3×3 figure rendered in 1.161 seconds and was visually inspected.
  It clearly distinguishes raw/smoothed training curves, provisional points,
  matched-subset baselines and completed per-arm evaluations.

The writer checks for updates every second, coalesces training renders to a
five-second budget, and refreshes on finding completion and phase changes.
Training journals flush every actual optimizer update; JSON/CSV snapshots and
full checkpoints are saved every 100 updates and on graceful interruption.
Recovery preserves orphaned journal observations in an archive and restores the
checkpoint cursor without duplicated updates. Complete schedules stay immutable.

Refiner subprocesses disable TF32 before importing PyTorch. Cache subprocesses
retain the original base-export environment. The model, loss, optimizer and
sampling methods remain as accepted. No data-loader optimization is introduced.

After all 10 evaluation barriers, status becomes `pending_user_review`.
Results, ranking and checkpoint selection remain the user's decisions.
