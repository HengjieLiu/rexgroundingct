---
created: 2026-08-02
updated: 2026-08-02
status: active
experiment_id: "013_voxtell_public_category_only_specialists"
---

# Exp013 Execution Spec

## Objective

Train four category-only VoxTell specialists from the public VoxTell v1.1
checkpoint for 100 epochs and evaluate target progress immediately after each
20-epoch training barrier.

## Experiment Matrix

| GPU | Arm | Target categories | Replay |
| ---: | --- | --- | ---: |
| 0 | `category_1all_diffuse_target100` | 1a-1f | 0% |
| 1 | `category_2all_focal_target100` | 2a-2h | 0% |
| 2 | `category_2bc_target100` | 2b, 2c | 0% |
| 3 | `category_2d_target100` | 2d | 0% |

## Training Contract

- Initialize from public VoxTell v1.1 only.
- Reset optimizer, AMP scaler, update counter, RNG stream, warmup, and
  polynomial horizon.
- Train 100 epochs, 100 updates per epoch, batch 1, no DDP.
- Use the Exp012/Exp009 v123 optimization recipe: encoder/decoder LR
  `1e-5/1e-4`, 100-update warmup, polynomial power `0.9`, SGD Nesterov
  momentum `0.99`, weight decay `3e-5`, gradient clipping `12`, and v123 loss.
- Use exactly 100 targeted events per epoch and no replay.
- Save immutable checkpoints at updates `2000/4000/6000/8000/10000` and a
  rolling recovery checkpoint every 500 updates.

## Evaluation Contract

- Epochs `0/20/40/60/80`: target census only.
- Epoch `100`: full val200 only; target, non-target, per-category, and overall
  metrics are derived from val200 outputs.
- Target census populations:
  - `1all`: 42 cases / 52 target findings.
  - `2all`: 196 cases / 329 target findings.
  - `2bc`: 76 cases / 109 target findings.
  - `2d`: 119 cases / 132 target findings.
- Selection uses target Dice, then target hit rate, then earlier epoch.

## Live Reporting

Maintain:

- `$RUN_GROUP/reports/exp013_progress.md`
- `$RUN_GROUP/reports/exp013_progress.json`
- immutable milestone snapshots under `$RUN_GROUP/milestones/epochXXX/`
- experiment-level `latest_progress` links to the active report

The report updates after epochs `0/20/40/60/80/100`.

## Verification Gates

- Exactly four visible GPUs with no silent fallback.
- Target-only schedules contain exactly 10,000 sequential events per arm.
- Every epoch has exactly 100 targeted events and 0 replay events.
- Every targeted event contains its requested positive target and crop anchor.
- Target manifests have the expected case/finding counts and no train/val
  leakage.
- One-update finite-loss smoke for every arm.
- Continuous-versus-segmented resume equivalence.
- Epoch-100 outputs contain exactly 200 cases and 381 findings per arm.
- Final status is `selection_ready`, with no subsequent evaluation launched.
