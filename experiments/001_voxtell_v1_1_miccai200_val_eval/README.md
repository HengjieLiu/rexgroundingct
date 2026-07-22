# VoxTell v1.1 MICCAI 200-case validation evaluation

This is a repo-local index for a runtime experiment. Edit the canonical
config in the repo, and keep large runtime artifacts under `/mnt/shengdata1`.

## Status

- Status: `evaluation_complete`
- Canonical config: `configs/experiments/001_voxtell_v1_1_miccai200_val_eval.json`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval`
- Runtime link: `runtime` is ignored by git and points to the runtime directory.

## Synced Small Artifacts

- Report snapshot: `experiments/001_voxtell_v1_1_miccai200_val_eval/report.md`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
