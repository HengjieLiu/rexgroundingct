# Experiment 011 Training And Fine-Tuning Specification

## Purpose

Experiment 011 answers three controlled questions:

1. Does CT-specific clipping or fixed HU normalization improve the v123
   fine-tuning recipe?
2. Is encoder LR `1e-4` stable and useful when decoder LR remains `1e-4`?
3. Do the normalization conclusions persist with encoder/decoder LR
   `1e-5/1e-4`, and does native e5d4 reproduce experiment 006 under the same
   segmented workflow?

All arms start from public VoxTell v1.1. Training was authorized at
`2026-07-28T07:28:16Z`. Both cache audits completed at
`2026-07-28T08:48Z`; preparation and GPU-headroom gates passed, and run group
`exp011_ct_norm_e4d4_20260728T085103Z` is active.

## Arm Matrix

| Arm | GPU | Normalization | Image padding | Encoder LR | Decoder LR |
| --- | ---: | --- | ---: | ---: | ---: |
| `v123_e4d4_zscore` | 0 | Full cropped-volume z-score | `0` | `1e-4` | `1e-4` |
| `v123_e4d4_clip1024_zscore` | 1 | Clip `[-1024,1024]`, then full-crop z-score | `0` | `1e-4` | `1e-4` |
| `v123_e4d4_clip1024_linear` | 2 | `clip(HU,-1024,1024)/1024` | `-1` | `1e-4` | `1e-4` |
| `v123_e5d4_zscore` | 0 | Full cropped-volume z-score | `0` | `1e-5` | `1e-4` |
| `v123_e5d4_clip1024_zscore` | 1 | Clip `[-1024,1024]`, then full-crop z-score | `0` | `1e-5` | `1e-4` |
| `v123_e5d4_clip1024_linear` | 2 | `clip(HU,-1024,1024)/1024` | `-1` | `1e-5` | `1e-4` |

All targets use zero padding.

## Shared Training Contract

- Public VoxTell v1.1 model-weight initialization.
- Native geometry and `192^3` model tensors.
- Batch size 1, gradient accumulation 1, no DDP.
- 100 epochs and exactly 100 optimizer updates per epoch.
- One shared 10,000-event schedule, seed `20260723`.
- Multi-finding cases: two positive prompts and one negative.
- One-finding cases: one positive prompt and two negatives.
- Every selected positive target is nonempty in its sampled patch.
- No additional augmentation.

Optimizer:

- SGD with Nesterov momentum.
- Momentum `0.99`.
- Weight decay `3e-5`.
- Completed e4d4 profile: encoder/decoder LR `1e-4/1e-4`.
- Authorized e5d4 profile: encoder/decoder LR `1e-5/1e-4`.
- Warmup 100 optimizer updates.
- Polynomial decay power `0.9` through update 10,000.
- Gradient clipping max norm `12`.

Loss:

- Nonempty target: Dice plus weighted BCE.
- Empty target: weighted BCE only, multiplied by `0.5`.
- BCE foreground/boundary/background weights: `1.0/1.5/0.5`.
- Boundary radius `10`.
- Deep-supervision weights `[1, 0.5, 0.25, 0.125, 0.0625]`,
  normalized by the trainer.

## Evaluation Contract

- Immutable checkpoints at epochs `5/20/40/60/80/100`.
- Fixed val20 at every immutable checkpoint.
- Fixed val200 at epoch 100.
- Primary threshold `0.5`.
- No probability maps by default.
- Only GPUs 0-2 are used.
- Each milestone ends the current training process and releases GPU memory.
- The three val20 jobs then run concurrently on GPUs 0-2.
- No arm starts its next training segment until all three val20 jobs complete.
- The canonical JSON and Markdown progress reports refresh after every val20
  barrier.
- The report update is part of the barrier; training cannot resume if report
  validation fails.
- Resume restores weights, SGD momentum, AMP scaler, Python/NumPy/PyTorch CPU
  and CUDA RNG streams, global update, LR horizon, and schedule cursor.
- Epoch-100 val200 runs concurrently after the epoch-100 val20 barrier.
- The e5d4 rerun uses a host-side idempotent state machine with separate Docker
  containers per stage, so an orchestration interruption cannot invalidate
  completed milestones.

## Same As Exp006

- Public model initialization.
- Cached native geometry, orientation, target handling, patch size, prompt
  schedule, v123 loss, optimizer family, warmup, polynomial decay, gradient
  clipping, batch size, update count, checkpoint cadence, and validation sets.
- The z-score arm uses the same standard native cache as exp006.

## Different From Exp006

- Encoder LR increases from the previous maximum `1e-5` to `1e-4`.
- Two arms intentionally change normalization.
- Fixed-HU image padding is `-1` rather than zero.
- Training uses full-state milestone segments and synchronous val20 barriers.
- GPU 3 remains untouched.

## Different From VoxTell Pretraining

- Training data are ReXGroundingCT prompts and released finding masks.
- Fine-tuning uses the project-specific one-finding prompt fallback.
- Batch size is 1 and epochs are defined as 100 optimizer updates.
- The v123 loss has explicit empty-target and weighted-BCE behavior.
- Two arms introduce CT-specific normalization not used by public VoxTell.

## Launch Guard

The launcher performs preparation only by default. Training requires:

```bash
START_TRAINING=1 \
bash /workspace/scripts/rexgroundingct/run_011_ct_normalization_ablation.sh
```

The required user order was received at `2026-07-28T07:28:16Z`. The cache
manifests, HU audit, sampler comparison, and GPU-headroom checks passed before
the active run was launched.
