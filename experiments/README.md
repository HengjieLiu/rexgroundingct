---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# ReXGroundingCT Experiments

This directory is a lightweight repo-local index. It keeps configs and
small summaries easy to inspect while large outputs remain on `/mnt/shengdata1`.

## Folder Map

- `README.md`: experiment index overview and drift policy.
- `registry.yaml`: generated machine-readable experiment summary.
- `<experiment-id>/README.md`: one experiment's status and ownership rules.
- `<experiment-id>/codex_execution_spec.md`: active or historical Codex execution spec.
- `<experiment-id>/metrics_summary.json`: small synced metrics/provenance summary.
- `<experiment-id>/sync_manifest.json`: synced hash and path manifest.
- `<experiment-id>/report.md`: small copied runtime report when available.
- `<experiment-id>/runtime`: ignored symlink to heavyweight runtime outputs.

## Drift Policy

- Edit canonical configs under `configs/experiments/`.
- Write or update `codex_execution_spec.md` before substantial long-running experiment work.
- Runtime configs are hashed snapshots copied at run start.
- `experiments/*/runtime` symlinks are ignored by git.
- Use `scripts/rexgroundingct/check_experiment_consistency.py` before committing.

## Experiments

| ID | Status | Report | Runtime |
| --- | --- | --- | --- |
| `001_voxtell_v1_1_miccai200_val_eval` | `evaluation_complete` | `experiments/001_voxtell_v1_1_miccai200_val_eval/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval` |
| `002_voxtell_text_ft_miccai_train_val` | `runtime_initialized` | `` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val` |
