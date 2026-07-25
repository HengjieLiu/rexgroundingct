# Experiment 006 Execution Specification

## Question

Does cached-native preprocessing reproduce the exp003 v123 VoxTell fine-tuning
recipe, and do higher decoder learning rates improve adaptation or destabilize
training?

## Arms

| Arm | GPU | Encoder LR | Decoder LR | Purpose |
| --- | ---: | ---: | ---: | --- |
| `v123_cached_e7_d6` | 0 | `1e-7` | `1e-6` | cached-native sanity check |
| `v123_cached_e7_d4` | 1 | `1e-7` | `1e-4` | decoder LR effect |
| `v123_cached_e6_d4` | 2 | `1e-6` | `1e-4` | moderate encoder adaptation |
| `v123_cached_e5_d4` | 3 | `1e-5` | `1e-4` | aggressive full-model adaptation |

All arms initialize from public VoxTell v1.1, not from exp003 or exp004
fine-tuned checkpoints.

## Preprocessing Contract

This is a cached-equivalent native VoxTell experiment.

The shared cache is:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1
```

The cache contract is the VoxTell native baseline:

1. Read CTs through `NibabelIOWithReorient`.
2. Convert released masks from raw `F,X,Y,Z` to VoxTell `F,Z,Y,X`.
3. Crop CT to nonzero and apply the same crop to all targets.
4. Z-score the complete cropped CT volume once.
5. Sample native-resolution `192^3` training patches from that normalized
   cropped volume.

The cache manifest used for launch must report:

- cases: `3192`
- targets: `8068`
- empty targets: `0`
- manifest SHA256: `fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3`

## Reproducibility

- Seed: `20260723`.
- One shared schedule, 10,000 events.
- Source schedule:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/config/train_schedule_v123_opt_poscrop_emptyloss_seed20260723_100ep_100steps_gb1.jsonl`
- Required schedule SHA256:
  `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776`.
- The same event stream is consumed by all four arms.
- Fixed validation JSONs are passed by path. `--limit` must not define a probe.

Prompt sampling remains exp003 v123:

- Multi-finding case: 2 positive prompts plus 1 negative prompt.
- One-finding case: 1 positive prompt plus 2 negative prompts.
- Every selected positive target must be nonempty in the sampled patch.

## Optimization

- 100 epochs, 100 optimizer updates per epoch, batch size 1, accumulation 1.
- Single GPU per arm, no DDP.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- 100-update warmup, polynomial decay power `0.9` over 10,000 updates.
- Gradient norm clipping at `12`.
- No new augmentation.

The v123 loss is:

- Nonempty target: Dice plus weighted BCE.
- Empty target: weighted BCE only, loss weight `0.5`.
- BCE weights: foreground `1.0`, boundary `1.5`, background `0.5`.
- Boundary radius: `10`.
- Deep supervision weights: `[1, 1/2, 1/4, 1/8, 1/16]`, normalized by the
  existing trainer loss.

## Checkpoints And Evaluation

Training processes only train and save checkpoints. A sidecar coordinator owns
validation and eval locks.

Immutable checkpoints:

```text
checkpoint_update_000500.pth
checkpoint_update_002000.pth
checkpoint_update_004000.pth
checkpoint_update_006000.pth
checkpoint_update_008000.pth
checkpoint_update_010000.pth
```

A rolling recovery checkpoint is written every 500 optimizer updates.

Canonical eval directories per arm:

```text
eval_epoch005_val20
eval_epoch020_val20
eval_epoch040_val20
eval_epoch060_val20
eval_epoch080_val20
eval_epoch100_val20
eval_epoch100_val200
```

The coordinator starts evaluation as soon as each checkpoint is stable. If an
eval fails, including GPU OOM, it releases the eval lock and retries later
without touching the trainer.

## Smoke Gates

Before full launch:

- Confirm native cache manifest counts and SHA256.
- Confirm schedule SHA256.
- Confirm val20 and val200 JSON SHA256.
- Sample-test each arm for 8 scheduled cached events.
- Run one-update training smoke for `v123_cached_e7_d6` and
  `v123_cached_e5_d4`.
- Run one-case cached-native inference smoke using
  `run_voxtell_val_inference.py --preprocessed-cache-dir`.

## Acceptance

- All four arms complete 10,000 optimizer updates.
- All four arms produce val20 at epochs 5, 20, 40, 60, 80, and 100.
- All four arms produce val200 at epoch 100.
- The final report ranks LR arms by Dice and hit rate and flags instability if
  loss diverges or val20 collapses.
