---
created: 2026-08-14
updated: 2026-08-14
status: active
---

# VoxTell 0.7 mm isotropic HU preprocessing

This repo-local folder records the preprocessing-only setup for the 0.7 mm
isotropic VoxTell cache. Large arrays, manifests, and logs stay on
`/mnt/shengdata1`.

## Status

- Status: `preprocessing_complete`
- Canonical config: `configs/experiments/016_voxtell_iso07_hu_preprocessing.json`
- Execution spec: `experiments/016_voxtell_iso07_hu_preprocessing/codex_execution_spec.md`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/016_voxtell_iso07_hu_preprocessing`
- Standard cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`
- Report snapshot:
  `experiments/016_voxtell_iso07_hu_preprocessing/report.md`

## Ownership Rule

- The cache is shared derived data, not a Git artifact.
- The experiment runtime folder should contain logs and snapshots only.
- This step builds preprocessing only; finetuning launch needs a separate spec.
