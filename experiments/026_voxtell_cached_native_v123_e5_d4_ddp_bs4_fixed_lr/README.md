---
created: 2026-09-06
updated: 2026-09-06
status: active
---

# VoxTell cached-native v123 e5/d4 DDP batch4 fixed-LR run

This is a repo-local index for the Exp026 runtime experiment. Canonical intent
lives in the repository; checkpoints, logs, predictions, and evaluator outputs
stay under `/mnt/shengdata1`.

## Status

- Status: `implemented_delayed_launch`
- Canonical config: `configs/experiments/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr.json`
- Execution spec: `experiments/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr`

## Training Contract

- Public VoxTell v1.1 initialization; no fine-tuned source checkpoint.
- Cached-equivalent native preprocessing: `crop_zscore_native_v1`.
- Four-GPU DDP, local batch 1, effective global batch 4.
- Encoder LR `1e-5`, decoder LR `1e-4`, fixed from update 1, zero warmup.
- 100 epochs / 10,000 optimizer updates.
- Immutable checkpoint and synchronous full-val200 barrier every 5 epochs.

## Ownership Rule

- Edit the canonical config and launchers in the repository.
- Runtime configuration snapshots are immutable provenance.
- Keep logs, predictions, checkpoints, and raw evaluator outputs outside Git.
- Sync the small report and metrics summary after runtime completion.
