# ReXGroundingCT Experiments

This directory is a lightweight repo-local index. It keeps configs and
small summaries easy to inspect while large outputs remain on `/mnt/shengdata1`.

## Drift Policy

- Edit canonical configs under `configs/experiments/`.
- Runtime configs are hashed snapshots copied at run start.
- `experiments/*/runtime` symlinks are ignored by git.
- Use `scripts/rexgroundingct/check_experiment_consistency.py` before committing.

## Experiments

| ID | Status | Report | Runtime |
| --- | --- | --- | --- |
| `001_voxtell_v1_1_miccai200_val_eval` | `evaluation_complete` | `experiments/001_voxtell_v1_1_miccai200_val_eval/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval` |
| `002_voxtell_text_ft_miccai_train_val` | `runtime_initialized` | `` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val` |
