---
created: 2026-07-25
updated: 2026-07-25
status: active
experiment_id: "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
---

# Training And Fine-Tuning Spec

## Short Name

`exp007_ddp_bs4_update_matched`

## Question

Should the cached-native v123 recipe use single-GPU batch1 or four-GPU DDP
global batch4 for future challenge fine-tuning?

## Baseline

The comparison baseline is exp006 `v123_cached_e5_d4`, trained from public
VoxTell v1.1 with single-GPU batch1 for 100 epochs:

- epoch100 val200 Dice: `0.3241`;
- epoch100 val200 hit rate: `0.7612`;
- hits/targets: `290/381`.

## Unchanged From Exp006 `v123_cached_e5_d4`

- Public VoxTell v1.1 initialization.
- Native cached preprocessing: orientation fix, crop-to-nonzero, full
  cropped-volume z-score once, native-resolution `192^3` windows.
- Text prompt embeddings.
- v123 prompt policy: multi-finding `2 positive + 1 negative`; one-finding
  fallback `1 positive + 2 negatives`.
- Every selected positive target must be nonempty in the sampled patch.
- v123 empty-target/weighted-BCE loss.
- Deep supervision weights `[1, 0.5, 0.25, 0.125, 0.0625]`.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- Encoder LR `1e-5`, decoder LR `1e-4`.
- Warmup `100` updates, poly LR decay over `10,000` updates, power `0.9`.
- Gradient clipping max norm `12`.

## Changed From Exp006

- Use DDP on four GPUs.
- Per-GPU batch size stays `1`, but effective global batch size becomes `4`.
- Training schedule contains `40,000` events instead of `10,000`.
- Training pauses at epoch `25/50/75/100`; val200 uses all GPUs; training then
  resumes from full optimizer/scaler state.
- Routine val20 sidecar evaluation is omitted.

## How To Read The Checkpoints

- Epoch25 is sample-matched to exp006 epoch100: both have consumed `10,000`
  patch events. It has only `2,500` optimizer updates, so it is not
  update-matched.
- Epoch100 is update-matched to exp006 epoch100: both have `10,000` optimizer
  updates. It has consumed `40,000` patch events, so it measures whether larger
  global batch plus more samples helps.

## Schedule Rule

With local batch1 and world size4:

```text
event_index = update * 4 + rank
```

This makes the rank stream deterministic and lets resumed segments continue at
the correct global update.
