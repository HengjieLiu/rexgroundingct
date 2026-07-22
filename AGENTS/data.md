---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Data Handling

## Dataset Conventions

- Gated ReXGroundingCT files are documented in `dataset/README.md`.
- CT-RATE fixed volumes are stored outside the repo under `/mnt/shengdata1`.
- Segmentations and MICCAI metadata are stored under `/data/hengjie`.

## Git Policy

Do not commit heavyweight medical data, predictions, logs, checkpoints, or model
weights. The `.gitignore` covers runtime outputs under `experiments/`, including
NIfTI files, checkpoints, logs, predictions, and runtime symlinks.

## Repo-Local Artifacts

Small, intentional provenance artifacts may be committed:

- canonical experiment configs
- README files and reports
- metrics summaries
- sync manifests
- documentation generated from metadata
