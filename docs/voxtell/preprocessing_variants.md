---
created: 2026-07-24
updated: 2026-07-24
status: active
---

# VoxTell Preprocessing Variants And Standard Caches

This note is the canonical place to record VoxTell preprocessing variants for
ReXGroundingCT. Use it before proposing a new experiment that changes spacing,
normalization, patch/window geometry, or cache ownership.

## Baseline Invariant

The VoxTell-compatible baseline is:

1. Load CT with `NibabelIOWithReorient`.
2. Convert released masks from evaluator layout `(F, X, Y, Z)` to VoxTell
   training layout `(F, Z, Y, X)` using the same affine-derived orientation
   transform.
3. Crop the full image to nonzero once.
4. Apply z-score normalization once over the complete cropped image.
5. Sample, resize, or slide windows from that already-normalized cropped image.

Do not z-score each sampled patch independently unless the experiment is an
explicit normalization ablation.

## Standard Cache Root

Heavy preprocessed arrays live outside Git:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/
```

Standard cache variants use:

```text
<preprocess-id>/
  manifest.json
  .complete
  cases/<case_key>/
    image.npy
    targets.npz
    metadata.json
    .complete
```

`targets.npz` is required for train/validation cache entries with released
labels. Test/image-only cache entries may omit it.

Experiment runtime folders should record cache paths and hashes, not duplicate
large cache contents, unless the cache is intentionally experiment-specific.

As of 2026-07-24, `crop_zscore_2mm_v1` is registered under the shared root as a
symlink to the audited exp004 cache:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/cache/crop_zscore_2mm_v1
```

## Variant Registry

| Preprocess ID | Status | Image | Targets | Comparability |
| --- | --- | --- | --- | --- |
| `crop_zscore_native_v1` | Standard native cache | `float32`, `(1,Z,Y,X)`, original cropped voxel spacing | `uint8`, `(F,Z,Y,X)`, no resampling | Equivalent to current on-the-fly native VoxTell preprocessing; changes runtime, not scientific preprocessing |
| `crop_zscore_2mm_v1` | Standard 2 mm cache | Full cropped-volume z-score first, then trilinear 2 mm isotropic resampling with `align_corners=False`, no antialiasing | Nearest-exact 2 mm masks, foreground-center fallback if a nonempty target disappears | Changes physical sampling and field of view; not comparable as only a runtime optimization |
| `full_fov_4mm_192_v1` | Proposal-stage cache | Inherits z-scored 2 mm cache, downsamples to 4 mm, center-pads to `192^3` | Native cropped targets retained for proposal-region labels | Stage-1 proposal input, not a direct VoxTell segmentation input |
| Future CT/HU variants | Planned ablations | Must state clipping/windowing/statistics before launch | Same orientation and mask policy unless explicitly changed | Normalization ablation; do not mix with spacing changes without a matrix |

## Required Fields For New Variants

Every new preprocessing variant must record:

- orientation reader and target reorientation policy;
- crop rule and whether crop-to-nonzero occurs before any resampling;
- normalization scope and statistics source;
- target spacing and image interpolation;
- mask interpolation and foreground-preserving fallback;
- image and mask padding values;
- patch or sliding-window policy;
- inverse restoration/export path;
- cache dtype, schema version, and root path;
- expected data-pipeline bottleneck and storage estimate;
- whether the variant is cached-equivalent or scientifically different from
  the VoxTell baseline.

## Exp004 Runtime Lesson

Experiment 004 showed that the same `192^3` model tensor can have very
different wall time. The native continuation repeatedly loads compressed CTs,
reorients, crops, z-scores, loads masks, and samples patches on demand. The
2 mm arm uses a prebuilt cache where orientation, crop, z-score, and resampling
are already complete.

For future native-control continuations, prefer `crop_zscore_native_v1` as the
control cache. It preserves the scientific preprocessing while removing the
repeated host I/O and CPU preprocessing bottleneck.

## Generation Command

Build the native standard cache inside the VoxTell Docker environment:

```bash
python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
  --preprocess-id crop_zscore_native_v1 \
  --cache-root /mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1 \
  --splits train val \
  --num-workers 4
```

The builder estimates storage from sample cases, requires free space with a
20 percent buffer by default, writes per-case `.complete` markers, and writes a
root `manifest.json` plus `.complete` marker only after the audit passes.
