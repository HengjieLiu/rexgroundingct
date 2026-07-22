---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Literature And Code Map

## Literature

- `literature/challenge_paper.pdf`: ReXGroundingCT challenge paper.
- `literature/challenge_motivation_paper.pdf`: background motivation paper.
- `literature/voxeltell.pdf`: VoxTell paper.

## External Code

- `external/VoxTell`: upstream VoxTell implementation pinned as a submodule.

## Local Code Map

- `scripts/rexgroundingct/common.py`: shared paths, experiment definitions, and
  provenance helpers.
- `scripts/rexgroundingct/run_voxtell_val_inference.py`: VoxTell validation
  inference and orientation-aware export.
- `scripts/rexgroundingct/train_text_conditioned_voxtell.py`: experimental
  text-conditioned fine-tuning scaffold.
- `scripts/rexgroundingct/sync_experiment_index.py`: sync small runtime
  summaries into the repo-local experiment index.
- `scripts/rexgroundingct/check_experiment_consistency.py`: pre-commit and
  pre-launch consistency checks.
