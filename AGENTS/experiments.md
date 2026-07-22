---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Experiment Rules

## Current Experiment IDs

- `001_voxtell_v1_1_miccai200_val_eval`: pretrained VoxTell validation
  evaluation on the 200-case MICCAI validation split.
- `002_voxtell_text_ft_miccai_train_val`: text-conditioned VoxTell fine-tuning
  scaffold for MICCAI train and validation data.

Do not rename experiment IDs casually. Runtime paths, configs, manifests, and
reports depend on them.

## Canonical Files

- Canonical configs live in `configs/experiments/`.
- Runtime config snapshots live under the `/mnt/shengdata1` experiment root.
- Repo-local summaries live under `experiments/`.

## Sync And Consistency

Use:

```bash
python scripts/rexgroundingct/sync_experiment_index.py
python scripts/rexgroundingct/check_experiment_consistency.py
```

The repo-local `experiments/` directory is an index, not the runtime output
store.
