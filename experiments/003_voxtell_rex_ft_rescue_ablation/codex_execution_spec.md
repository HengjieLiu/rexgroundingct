---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "003_voxtell_rex_ft_rescue_ablation"
---

# Codex Execution Spec

## Objective

Run a four-arm rescue ablation for VoxTell v1.1 ReXGroundingCT fine-tuning.
The goal is to identify whether exp002 failed mainly because of optimization,
positive-patch sampling, empty-target loss pressure, or deep-supervision
weighting.

## Prior Evidence

- Exp001 fixed-orientation pretrained validation baseline is usable.
- Exp002 z-score 192 baseline collapsed to all-background predictions on fixed
  val20.
- External ReX/VoxTell training code uses low differential LR, warmup+poly,
  gradient clipping, required positive crops, empty-target BCE-only loss, and
  deep-supervision weights with the final scale dropped.
- Orientation handling was already verified against the external cache method.

## Scope

In scope:

- Implement exp003 configs, launchers, schedule generation, and trainer switches.
- Run four cumulative single-GPU batch1 variants.
- Use fixed seeded training schedules and fixed val20/val200 JSON probes.

Out of scope:

- Non-192 patch-size ablations.
- HU/window input normalization.
- DDP or batch-size comparisons.
- Architecture changes beyond loss/optimizer/sampling/deep-supervision switches.

## Inputs And Paths

- Canonical config: `configs/experiments/003_voxtell_rex_ft_rescue_ablation.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- Runtime experiment directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation`
- Repo-local summary directory: `experiments/003_voxtell_rex_ft_rescue_ablation`

## Method

Shared training settings:

- VoxTell v1.1 initialization.
- `192^3` patches.
- VoxTell default crop-to-nonzero plus per-volume z-score normalization.
- Correct ReX raw `FXYZ` mask orientation to VoxTell/nnU-Net `FZYX` before crop
  and patch sampling.
- True batch1: `batch_size=1`, `grad_accum=1`, no DDP.
- `100` optimizer updates per epoch.
- Seed `20260723`.
- One materialized 10000-event train schedule per variant. The 5-epoch run
  consumes the first 500 events; the fresh 100-epoch run consumes all events
  from event 0.

Variants:

- `v1_opt`: differential low LR, warmup+poly, gradient clipping.
- `v12_opt_poscrop`: `v1` plus every selected positive prompt must be non-empty
  in the patch.
- `v123_opt_poscrop_emptyloss`: `v12` plus empty-target BCE-only downweighted
  loss and weighted BCE.
- `v1234_opt_poscrop_emptyloss_ds`: `v123` plus deep-supervision weights
  `[1, 1/2, 1/4, 1/8, 0]`.

Launch command inside Docker:

```bash
bash /workspace/scripts/rexgroundingct/run_003_voxtell_rescue_ablation.sh
```

Host Docker command:

```bash
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_003_voxtell_rescue_ablation_docker.sh
```

## Smoke Gate

- Generate or verify fixed val20 and val200 JSON files.
- Generate one 10000-event train schedule per variant.
- Run `sample-test` for each variant against the first schedule events.
- For positive-crop variants, confirm sampled positive targets are non-empty.
- Run a tiny one-step train smoke if `RUN_TINY_TRAIN_SMOKE=1`.

## Success Criteria

- Four 5-epoch runs complete and evaluate fixed val20.
- Four fresh 100-epoch runs complete.
- Each 100-epoch run has immutable checkpoints at updates
  `2000/4000/6000/8000/10000`.
- Fixed val20 metrics exist for epochs `20/40/60/80/100`.
- Fixed val200 metrics exist for epoch `100`.
- Training and evaluation logs, manifests, schedules, and metrics are written
  under the exp003 runtime directory.

## Verification

```bash
python scripts/rexgroundingct/check_repo_workflow.py
```

After runtime results:

```bash
python scripts/rexgroundingct/sync_experiment_index.py
python scripts/rexgroundingct/check_experiment_consistency.py
```

## Closeout Plan

- Sync small metrics summaries into `experiments/`.
- Update `docs/current_status.md` with the best 5-epoch and 100-epoch evidence.
- Record failed or incomplete arms explicitly instead of overwriting them.
