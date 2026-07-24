---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "003_voxtell_rex_ft_rescue_ablation"
---

# Exp003 Multiscale Inference Plan

## Purpose

Test whether inference-only multiscale source windows improve ReXGroundingCT
validation performance without changing the VoxTell model's internal `192^3`
input shape or positional encoding.

## Design

Use the `v123_opt_poscrop_emptyloss` epoch-100 checkpoint as the first
single-model multiscale candidate. Keep the VoxTell model patch size fixed at
`192 x 192 x 192`; the tested values are source-window sizes sampled from the
already-normalized image and resampled to the model patch size.

Scales:

- `96 -> 192`: strong zoom-in, small-lesion biased.
- `128 -> 192`: moderate zoom-in.
- `160 -> 192`: mild zoom-in.
- `192 -> 192`: native reference.
- `256 -> 192`: zoom-out, context biased.

Preprocessing order:

1. Load and reorient the full CT.
2. Crop the full CT to its nonzero region.
3. Apply one z-score normalization over that cropped full image.
4. Sample source windows from the normalized image.
5. Resize each source window to `192^3` with trilinear interpolation.
6. Run VoxTell unchanged.
7. Resize logits back to source-window size with trilinear interpolation.
8. Blend overlapping logits in source space, then apply sigmoid once.

Do not z-score individual source windows or resized model patches. That would
be a separate normalization ablation.

## Val20 Probe

Run the four non-native scales on GPUs `0..3`, then run the native `192`
probability reference. Save probability maps for every scale so later scale
ensembles can be tested without re-running inference.

Thresholds:
`0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90`.

Runtime output root:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/multiscale/v123_opt_poscrop_emptyloss_epoch100_val20`

Native hard-mask reference before this probe:

- `v123_opt_poscrop_emptyloss` epoch100 val20 Dice `0.348291`.
- Hit rate `0.677419` (`21/31`).

## Decision Rule

Run val200 if any scale or simple scale ensemble improves Dice over native
val20, improves hit rate without a large Dice loss, or shows useful
complementarity with native predictions.

## Val20 Result

Completed for `v123_opt_poscrop_emptyloss` epoch100 on the fixed val20 probe.
All source-window sizes used full-image crop-to-nonzero z-score normalization
once, then trilinear resampling of normalized source windows to the unchanged
`192^3` VoxTell model input. Antialiasing was not used.

Saved summaries:

- Runtime summary: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/multiscale/v123_opt_poscrop_emptyloss_epoch100_val20/multiscale_val20_summary.md`
- Runtime JSON: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/multiscale/v123_opt_poscrop_emptyloss_epoch100_val20/multiscale_val20_summary.json`

Best threshold per scale:

| Scale | Best Threshold | Dice/Finding | Hit Rate | Hits/Targets |
| --- | ---: | ---: | ---: | ---: |
| Native hard-mask reference | fixed | 0.348291 | 0.677419 | 21/31 |
| `192 -> 192` probability sweep | 0.40 | 0.349014 | 0.677419 | 21/31 |
| `96 -> 192` | 0.30 | 0.158897 | 0.580645 | 18/31 |
| `128 -> 192` | 0.35 | 0.282825 | 0.645161 | 20/31 |
| `160 -> 192` | 0.30 | 0.330668 | 0.677419 | 21/31 |
| `256 -> 192` | 0.20 | 0.339816 | 0.677419 | 21/31 |

Interpretation: inference-only source-window resampling did not beat native
`192^3` on val20. Smaller source windows degraded substantially, especially
`96 -> 192`. The `256 -> 192` zoom-out setting was closest to native but still
lower Dice. This does not rule out true patch-size fine-tuning, but it argues
against running val200 for these single-scale inference-only predictions unless
we first test a probability ensemble for complementarity.
