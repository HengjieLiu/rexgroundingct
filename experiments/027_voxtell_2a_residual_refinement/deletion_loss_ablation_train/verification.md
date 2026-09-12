# Train-only loss comparison: verification and launch

Implementation is complete and the authorized detached job launched at
2026-09-11 04:20:37 UTC (September 10, 21:20 Pacific), after CPU tests and
repository checks. Historical source files and runtime artifacts are preserved.

Container: `rex027_deletion_loss_ablation_train_100ep`, ID prefix `16d2a53f86cc`.
Image: `sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f`.
Context: `bbea3f1bda42d25ae35f8654ce811b57a44c934648059d98753d17bf1d724a6f`.
The first observed phase has four CPU cache-verification workers and a live
dashboard. GPU trainers are gated on all 864 CTs passing verification.
Verification logs, source/config snapshots and the inspected synthetic dashboard
are preserved in the new runtime's `verification/` directory.

## CPU verification

78 distinct CPU tests passed in the existing `rexgroundingct-voxtell:cu126`
Docker image with GPUs disabled and dataset mounts read-only. The initial
41-test run covered new and historical loss/training/evaluation loops. Follow-up
checks covered the complete train+val gate, historical initialization, all 12
barriers, strict reference failure, progressive full threshold tables, original
geometry, cache contracts and deletion-only inference.

The real tiny-model training loop was interrupted at update 2,107 and resumed
to 2,200 with identical final weights/loss history and no duplicated updates
or schedule changes. Evaluation recovery reused one completed finding and
processed only the remaining 68. Initial weights match the historical digest.

Two legacy CPU-import assertions initially failed when combined into a process
that had already loaded PyTorch. Rerunning that 17-test contract suite in a fresh
process passed; no implementation or contract was weakened. The separate
PyTorch follow-up run passed 21 tests. One dashboard test was intentionally
repeated after adding train-only label assertions (79 passing executions,
78 distinct tests overall).

Inspected the labeled synthetic 3×3 dashboard with provisional A, absent B,
matching baselines, raw/smoothed losses, fixed run order and a synthetic failure.
Its title explicitly says it is not an experiment result. CPU device count is
zero. Full tables publish all 201 rows for A/B/full with missing arms pending,
then replace them when each model's analysis becomes available.

The host launcher dry run starts no Docker/GPU work even with START_GPU_WORK=1.
Shell syntax and Python compilation passed. The canonical repository workflow
and whitespace checks are recorded in the external verification logs.

## Input and schedule evidence

Read-only exploration verified all 864 historical tile-index hashes and
training/validation patient separation. Required arrays will be hash-verified
by the launched CPU preparation workers before any GPU trainer starts.

- Original training pool: 801 CTs / 1,120 findings / 741 patients.
- Eligible training pool: 1,119 findings.
- Excluded empty-base finding: `train_7777_a_2.nii.gz::4`.
- Validation: 63 CTs / 69 findings, split A=35 and B=34.
- Extended 10,000-event digest:
  `6975fc98840f2b589d88294fff628923ad01989ea14ea32528368ef074ecfeda`.
- Historical 2,000-event prefix digest:
  `cfe5c1bc3a450b4a2e4355c97e4ebeac605d9e8dab5a2c5f1e5b56856177e70f`.
- All 1,119 eligible findings appear; first 2,000 events cover 933.
- 501 recorded same-finding fallbacks across the extended schedule.
- Pristine model digest:
  `ff8d94ec3648a9ef24a0a2d4865c0590782da60a522b7f4b701cf57748cbba73`.
- Initial training RNG seed: 20261612 for every arm.

## Commands

Dry run, without GPU work:

```bash
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_train_host.sh orchestrate
```

Authorized detached launch:

```bash
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_train_host.sh orchestrate --gpus 0 1 2 3
```

Explicit resume with the previous stopped container retained:

```bash
START_GPU_WORK=1 DETACH=1 DELETION_TRAIN_ABLATION_CONTAINER_NAME=rex027_deletion_loss_ablation_train_100ep_resume bash scripts/rexgroundingct/run_027_deletion_loss_ablation_train_host.sh orchestrate --gpus 0 1 2 3 --resume
```

Other supported commands are `prepare`, `report`, `analyze`, `audit`, and
`train`/`evaluate --arm ARM --update MILESTONE --gpus GPU` with explicit GPU
opt-in. Standalone train/evaluate require existing complete preparation.
Reporting and analysis use no GPU. Never start a second report writer while
the orchestrator owns the dashboard lock.

The new runtime is `deletion_loss_ablation_train_100ep`. Completion remains
pending until 40,000 updates, 48 evaluations, 48 dense analyses and all four
historical reference checks pass; no threshold or model is automatically chosen.

## Recovery after epoch 5

All four update-500 evaluations finished at approximately 2026-09-11 06:38 UTC.
Historical BCE weights and all per-finding threshold metrics matched exactly
at updates 100 and 500. The next training workers failed before update 501:
NVML could not initialize and PyTorch could not see the assigned GPU in the
original container. Host GPUs were healthy and idle; a fresh container verified
four CUDA devices. The external trigger for the lost visibility is unconfirmed.

Preserved the failed container, logs, pre-recovery status/launch/preparation
records and checkpoint/history/evaluation hashes under runtime
`verification/recovery_after_epoch5_20260911/`. Resumed with `--resume` in
`rex027_deletion_loss_ablation_train_100ep_resume_e005` (ID `fa2ced55be54`),
without code, method, schedule or precision changes. Existing verified cache
receipts were reused and completed evaluation barriers were retained.
At 06:41:28 UTC all four models were training at update 507, with contiguous
histories and no duplicated updates. The target remains update 10,000.
