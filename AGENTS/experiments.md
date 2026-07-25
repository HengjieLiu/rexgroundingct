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
- `003_voxtell_rex_ft_rescue_ablation`: four-arm VoxTell rescue fine-tuning
  ablation with fixed schedules and fixed val20/val200 probes.
- `004_voxtell_v123_native_vs_2mm_global_context_ft`: paired continuation from
  exp003 v123 epoch 100 comparing native VoxTell preprocessing with 2 mm
  global-context preprocessing.
- `005_voxtell_global_proposal_local_cascade`: two-stage global proposal to
  local VoxTell segmentation cascade.
- `006_voxtell_cached_native_v123_lr_ablation`: cached-equivalent native v123
  fine-tuning sanity check and learning-rate ablation.

Do not rename experiment IDs casually. Runtime paths, configs, manifests, and
reports depend on them.

## Preprocessing Contract

Every substantial experiment spec must state whether preprocessing is:

- unchanged from the VoxTell baseline;
- cached-equivalent to the baseline; or
- intentionally changed as a method variable.

For VoxTell/ReXGroundingCT work, record normalization scope, resampling,
patch/window policy, cache ownership, orientation/export checks, and expected
data-pipeline bottlenecks. Use `docs/voxtell/preprocessing_variants.md` for the
standard cache IDs and required fields.

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
