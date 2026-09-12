# Four category models fitted on validation A

The user approved this A-only follow-up and the separate A/B dashboard curves.
The preceding original-training models stop at exactly update 2000, without
starting their update-2000 validation. Their earlier evaluations are retained.

- [Execution specification](codex_execution_spec.md)
- [Verification and launch](verification.md)
- [Live A/B dashboard](../runtime/deletion_categories_bcde_val_a_20ep/reports/live_dashboard.md)
- [Threshold records](../runtime/deletion_categories_bcde_val_a_20ep/reports/threshold_sweep/index.md)

Each category trains a fresh Dice + BCE TP2 deletion editor on A only for
2000 updates, evaluating at 100, 500, 1000 and 2000. A is fitted, B is held-out
development, and full A+B mixes those exposures. Category 2e B has three findings.

Config: `configs/experiments/027_deletion_categories_bcde_val_a_20ep.json`.
All arrays, checkpoints and figures remain in the external Exp027 runtime.

## Commands

```bash
bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh orchestrate --dry-run
bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh prepare
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh orchestrate --gpus 0 1 2 3
# Explicit recovery with a distinct container name if the stopped container remains.
START_GPU_WORK=1 DETACH=1 DELETION_CATEGORIES_VAL_A_CONTAINER_NAME=rex027_categories_val_a_resume_1 bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh orchestrate --gpus 0 1 2 3 --resume
bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh report
bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh analyze
bash scripts/rexgroundingct/run_027_deletion_categories_val_a_host.sh audit
```

Standalone workers accept `train --arm 2b --update 100 --gpus 0` or
`evaluate --arm 2b --update 100 --gpus 0`, with `START_GPU_WORK=1`. Use the
coordinator for synchronized four-model barriers. CPU report/analysis commands
respect the running writer locks. No web service is used; reload the local
Markdown/PNG viewer when necessary.
