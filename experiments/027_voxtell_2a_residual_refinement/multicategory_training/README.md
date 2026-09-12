# Four category-specific Dice + BCE TP2 deletion models

The user authorized four concurrent runs for 2b, 2c, 2d and 2e, after the 2a
epoch-60 evaluation finishes and its training stops. Each category fits its
eligible original-training findings for 50 epochs of 100 updates.

- [Execution specification](codex_execution_spec.md)
- [2a A-only evidence and proposed category follow-up](a_only_followup_audit.md)
- [Live full-validation dashboard](../runtime/deletion_categories_bcde_50ep/reports/live_dashboard.md)
- [Full threshold records, including A/B](../runtime/deletion_categories_bcde_50ep/reports/threshold_sweep/index.md)
- [Verification and launch](verification.md)

Config: `configs/experiments/027_deletion_categories_bcde_50ep.json`.
Runtime: Exp027 external root, `deletion_categories_bcde_50ep`.
Full-validation plot points appear only when the individual category evaluation
is complete. A/B metrics are retained in records, without separate A/B plots.

## Commands

The coordinator verifies the complete shared cache and the 2a epoch-60 stop
receipt before launching the four GPU workers. It never exports base logits.

```bash
bash scripts/rexgroundingct/run_027_deletion_categories_host.sh orchestrate --dry-run
bash scripts/rexgroundingct/run_027_deletion_categories_host.sh prepare
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_categories_host.sh orchestrate --gpus 0 1 2 3
# Explicit recovery; choose a new container name if the old exited container remains.
START_GPU_WORK=1 DETACH=1 DELETION_CATEGORIES_CONTAINER_NAME=rex027_categories_resume_1 bash scripts/rexgroundingct/run_027_deletion_categories_host.sh orchestrate --gpus 0 1 2 3 --resume
bash scripts/rexgroundingct/run_027_deletion_categories_host.sh report
bash scripts/rexgroundingct/run_027_deletion_categories_host.sh analyze
bash scripts/rexgroundingct/run_027_deletion_categories_host.sh audit
```

Standalone workers use `train --arm 2b --update 100 --gpus 0` or
`evaluate --arm 2b --update 100 --gpus 0`, with `START_GPU_WORK=1`.
Use the coordinator for the synchronized four-model protocol. Reporting and
analysis commands use CPU-only containers; a running writer owns its lock.
Evaluation epochs: 1, 5, 10, 20, 30, 40, 50. Each epoch is 100 updates.
All categories use original-training data only; validation A/B are development
subsets. The confirmed overlapping 2d training finding is excluded.

## User-requested stop

All four original-training arms stopped at update 2000 on 2026-09-11, without
starting that checkpoint's validation, as explicitly requested. Completed
validation remains available at 100/500/1000. Checkpoints at 2000 are preserved.
[Stop evidence](../runtime/deletion_categories_bcde_50ep/reports/user_stop_update2000.json).
The [A-only follow-up](../multicategory_val_a/README.md) has separate A/B curves.
