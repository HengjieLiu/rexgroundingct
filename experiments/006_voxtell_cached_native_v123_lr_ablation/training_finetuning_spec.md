# Experiment 006 Training And Fine-Tuning Spec

## Baseline Being Mimicked

Experiment 006 uses the exp003 v123 fine-tuning recipe with one deliberate
engineering change: native VoxTell preprocessing is loaded from the standard
`crop_zscore_native_v1` cache instead of recomputed for every sampled patch.
The scientific preprocessing is intended to be unchanged.

## Same As Exp003 v123

- Public VoxTell v1.1 initialization.
- `192^3` model input and positional encoding.
- Fixed seed `20260723`.
- Fixed 10,000-event positive-crop schedule with SHA256
  `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776`.
- Three prompt channels per training sample.
- Multi-finding cases use `2 positive + 1 negative`.
- One-finding cases use `1 positive + 2 negatives`.
- Every selected positive target must be nonempty in the sampled patch.
- Batch size 1, no DDP.
- 100 optimizer updates per epoch.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- Warmup 100 optimizer updates, polynomial LR decay power `0.9`.
- Gradient clipping max norm `12`.
- Loss mode `empty_bce_only`:
  - nonempty target: Dice + weighted BCE;
  - empty target: weighted BCE only with weight `0.5`.
- BCE voxel weights: foreground `1.0`, boundary `1.5`, background `0.5`.
- Boundary radius `10`.
- Deep-supervision weights `[1, 0.5, 0.25, 0.125, 0.0625]`.
- Fixed val20 and val200 JSONs are passed directly.

## Different From Exp003 v123

- Preprocessing is cached-equivalent native rather than on-the-fly native.
- Epoch-5 val20 is part of the continuous 100-epoch run, not a separate
  diagnostic run.
- Three arms raise decoder LR to `1e-4`.
- The immutable checkpoint cadence includes update 500 for epoch-5 evaluation.

## Different From VoxTell Pretraining

- The dataset is ReXGroundingCT with free-text findings and released finding
  masks.
- The one-finding fallback uses `1 positive + 2 negative` prompt channels.
- Fine-tuning uses batch size 1 and standardized 100-update epochs.
- The v123 loss adds empty-target handling and weighted BCE.
- Validation uses fixed ReXGroundingCT val20/val200 probe JSONs.

## Arm Matrix

| Arm | GPU | Encoder LR | Decoder LR |
| --- | ---: | ---: | ---: |
| `v123_cached_e7_d6` | 0 | `1e-7` | `1e-6` |
| `v123_cached_e7_d4` | 1 | `1e-7` | `1e-4` |
| `v123_cached_e6_d4` | 2 | `1e-6` | `1e-4` |
| `v123_cached_e5_d4` | 3 | `1e-5` | `1e-4` |

## Evaluation Outputs

Primary metrics are mean global Dice per finding, hit rate, hits/targets,
training loss trend, and mean update seconds. Probability maps are not saved
for this LR sweep unless a later ensemble analysis explicitly requests them.
