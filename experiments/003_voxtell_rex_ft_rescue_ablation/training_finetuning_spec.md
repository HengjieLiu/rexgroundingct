---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "003_voxtell_rex_ft_rescue_ablation"
---

# Training And Fine-Tuning Spec

## Purpose

Experiment 003 tests four cumulative fixes after the exp002 all-background
collapse. It keeps the baseline input distribution and model fixed, then adds
the external successful ReX/VoxTell training ingredients one group at a time.

## Shared Settings

| Setting | Exp003 value | Same as VoxTell/pretraining-style baseline? | Difference or note |
| --- | --- | --- | --- |
| Base checkpoint | VoxTell v1.1 | Yes | Fine-tuning starts from public VoxTell weights. |
| Input crop | crop to nonzero | Yes | Same as exp001/exp002 VoxTell preprocessing path. |
| Normalization | per-volume z-score | Yes | HU/window input is deferred to a later ablation. |
| Patch size | `192^3` | Yes | No patch-size change in exp003. |
| Orientation | raw ReX `FXYZ` masks converted to VoxTell/nnU-Net `FZYX` | Same intended behavior | This is the exp001/exp002 orientation fix. |
| Prompt slots | 3 | Yes | Same model output/prompt interface. |
| Multi-finding prompt mix | 2 positive + 1 negative | Yes/paper-aligned | Negative targets are empty. |
| One-finding fallback | 1 positive + 2 negatives | Unknown | Paper fallback remains unresolved; we move forward with this documented rule. |
| Epoch definition | 100 optimizer updates | No | Project standardization; not VoxTell paper epoch length. |
| Batch | true batch1 | No | Required by memory and controlled ablation design. |
| DDP | off | No | Single-GPU arms isolate method effects. |
| Seed | `20260723` | Project policy | Data order and validation probe are fixed. |

## Reproducibility Rules

- Each variant has one materialized 10000-event schedule generated with seed
  `20260723`.
- The 5-epoch diagnostic run consumes the first 500 events from that schedule.
- The fresh 100-epoch run restarts from VoxTell v1.1 and consumes all 10000
  events from event index 0.
- Fixed val20 is
  `configs/evaluation/rexgroundingct_val20_seed20260723.json`.
- Fixed val200 is
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`.
- Evaluation uses fixed JSON overrides, not `--limit`.

This fixes data order and patch sampling. It does not promise bitwise-identical
weights because CUDA/AMP kernels may still be nondeterministic.

## Materialized Schedule And Probe Records

These are the generated records used by the launched exp003 run group
`exp003_full_20260723T075256Z`.

| Record | Events/size | Seed | SHA256 |
| --- | ---: | ---: | --- |
| `train_schedule_v1_opt_seed20260723_100ep_100steps_gb1.jsonl` | 10000 | 20260723 | `96df9c67478f38f666f686a09ff050203c77b14238d6763ac9475f3d246632a9` |
| `train_schedule_v12_opt_poscrop_seed20260723_100ep_100steps_gb1.jsonl` | 10000 | 20260723 | `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776` |
| `train_schedule_v123_opt_poscrop_emptyloss_seed20260723_100ep_100steps_gb1.jsonl` | 10000 | 20260723 | `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776` |
| `train_schedule_v1234_opt_poscrop_emptyloss_ds_seed20260723_100ep_100steps_gb1.jsonl` | 10000 | 20260723 | `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776` |
| `rexgroundingct_val20_seed20260723.json` | 20 | 20260723 | `31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d` |
| `rexgroundingct_val200_seed20260723.json` | 200 | 20260723 | `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897` |

Prompt-slot and foreground statistics:

| Variant | Positive slots | Negative slots | Multi-finding events | One-finding fallback events | Foreground-overlap requested | Positive-crop required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `v1_opt` | 16487 | 13513 | 6487 | 3513 | 9005 | no |
| `v12_opt_poscrop` | 16260 | 13740 | 6260 | 3740 | 10000 | yes |
| `v123_opt_poscrop_emptyloss` | 16260 | 13740 | 6260 | 3740 | 10000 | yes |
| `v1234_opt_poscrop_emptyloss_ds` | 16260 | 13740 | 6260 | 3740 | 10000 | yes |

Positive-crop schedule eligibility for v12/v123/v1234 is computed in VoxTell
training space after CT reorientation and `crop_to_nonzero`. 2822 of 2992 train
cases are eligible, with 1766 eligible multi-finding cases, 1056 eligible
one-finding cases, 170 ineligible multi-finding cases, and 8437 compatible
positive-finding pairs. Positive-crop schedules materialize `patch_starts` for
every event; the generated schedule has zero missing `patch_starts` and zero
missing `positive_crop_points`. The three positive-crop variants intentionally
reuse the same schedule hash so that loss/deep-supervision effects are isolated.

## Variants

| Variant | Additions | GPU | Expected diagnostic |
| --- | --- | ---: | --- |
| `v1_opt` | encoder LR `1e-7`, decoder LR `1e-6`, SGD Nesterov, momentum `0.99`, weight decay `3e-5`, warmup 100 updates, poly LR, grad clip 12 | 0 | Tests whether exp002 mainly failed from too-large fixed LR. |
| `v12_opt_poscrop` | `v1` plus require every selected positive prompt to be non-empty in the patch | 1 | Tests whether empty nominal-positive crops caused unstable supervision. |
| `v123_opt_poscrop_emptyloss` | `v12` plus empty-target BCE-only loss, empty target weight `0.5`, weighted BCE foreground/boundary/background `1.0/1.5/0.5`, boundary radius `10` | 2 | Tests whether Dice on empty targets/background pressure drove collapse. |
| `v1234_opt_poscrop_emptyloss_ds` | `v123` plus deep-supervision weights `[1, 1/2, 1/4, 1/8, 0]` | 3 | Tests whether supervising the lowest-resolution output hurt fine-tuning. |

## Checkpoints And Validation

- 5-epoch diagnostic:
  - Train 500 optimizer updates.
  - Evaluate fixed val20 at epoch 5.
- Fresh 100-epoch run:
  - Train 10000 optimizer updates from VoxTell v1.1.
  - Save immutable checkpoints at updates `2000`, `4000`, `6000`, `8000`, and
    `10000`.
  - Evaluate fixed val20 for epochs `20`, `40`, `60`, `80`, and `100`.
  - Evaluate fixed val200 for epoch `100`.

## Known Differences From Exp002

- Exp002 used single LR `1e-4`; exp003 uses differential `1e-7/1e-6`.
- Exp002 used fixed LR/no warmup/no clipping; exp003 uses warmup+poly+clip.
- Exp002 guaranteed only one anchor positive could be in the patch; v12+ require
  all selected positive targets to be non-empty.
- Exp002 applied Dice+BCE to all channels including empty targets; v123+ use
  BCE-only for empty targets.
- Exp002 used deep-supervision weights `[1, 1/2, 1/4, 1/8, 1/16]`; v1234 uses
  `[1, 1/2, 1/4, 1/8, 0]`.
