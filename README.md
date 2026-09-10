---
created: 2026-07-23
updated: 2026-07-23
status: active
---

# ReXGroundingCT

This repository keeps a reproducible VoxTell baseline and a lightweight
challenge workflow for ReXGroundingCT. Code, configs, small summaries, and
documentation live here; CT volumes, predictions, logs, checkpoints, and model
weights stay on the external data and experiment mounts.

## Start Here

- `AGENTS.md`: required agent reading instructions.
- `docs/README.md`: documentation map.
- `docs/ai_workflow.md`: AI-assisted research workflow rules adapted from prior
  projects.
- `docs/current_status.md`: current decisions, active work, blockers, and next
  actions.
- [Submission register](submissions/REGISTER.md): a/b/c/d aliases, prediction paths,
  model recipes, packaging status and recorded uploads.
- `experiments/README.md`: repo-local experiment index.
- `dataset/README.md`: gated ReXGroundingCT and CT-RATE download workflow.
- `docker/voxtell/README.md`: VoxTell Docker build and run notes.

## Common Commands

```bash
python scripts/rexgroundingct/check_repo_workflow.py
```

For the expanded manual check sequence, see `docs/ai_workflow.md`.

## Artifact Rule

Commit small intentional provenance artifacts only: configs, docs, metrics
summaries, reports, manifests, and execution specs. Keep heavyweight medical
data and runtime outputs outside Git.
