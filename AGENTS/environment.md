---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Environment And Docker

## Submodules

- VoxTell lives at `external/VoxTell`.
- Initialize with:

```bash
git submodule update --init --recursive
```

## Docker

- Docker harness lives under `docker/voxtell/`.
- Build and usage notes are in `docker/voxtell/README.md`.
- The container expects the repo at `/workspace`.
- Launch scripts often call `/workspace/scripts/rexgroundingct/...`.

## Runtime Roots

- ReXGroundingCT metadata and masks: `/data/hengjie/datasets/rexgroundingct`
- CT-RATE subset: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Experiment runtime root: `/mnt/shengdata1/hengjie/experiments/rexgroundingct`

Keep these absolute paths centralized in scripts where possible.
