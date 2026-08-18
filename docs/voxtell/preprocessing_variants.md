---
created: 2026-07-24
updated: 2026-07-27
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
| `crop_clip1024_zscore_native_v1` | Experiment 011 standard HU-clipped cache | Clip the complete native crop to `[-1024,1024]` HU, then full-crop z-score; image padding `0` | Identical native targets and geometry to `crop_zscore_native_v1` | Changes normalization only; tests whether robust clipping removes harmful sentinel/outlier influence |
| `crop_clip1024_linear_native_v1` | Experiment 011 standard fixed-HU cache | `clip(HU,-1024,1024)/1024`; image padding `-1` | Identical native targets and geometry to `crop_zscore_native_v1` | Changes normalization only; tests fixed HU calibration under VoxTell InstanceNorm |
| `crop_zscore_2mm_v1` | Standard 2 mm cache | Full cropped-volume z-score first, then trilinear 2 mm isotropic resampling with `align_corners=False`, no antialiasing | Nearest-exact 2 mm masks, foreground-center fallback if a nonempty target disappears | Changes physical sampling and field of view; not comparable as only a runtime optimization |
| `crop_clip1024_linear_iso07_v1` | Experiment 016 0.7 mm cache | Crop native CT to nonzero, clip HU to `[-1024,1024]`, divide by `1024`, then trilinear 0.7 mm isotropic resampling with `align_corners=False`, image padding `-1` | Native-cropped masks resampled with nearest-exact 0.7 mm spacing, foreground-center fallback if a nonempty target disappears; test entries are image-only | Changes both physical sampling and normalization; compare as a new finetuning preprocessing experiment, not as a runtime-only cache |
| `full_fov_4mm_192_v1` | Proposal-stage cache | Inherits z-scored 2 mm cache, downsamples to 4 mm, center-pads to `192^3` | Native cropped targets retained for proposal-region labels | Stage-1 proposal input, not a direct VoxTell segmentation input |
| Future CT/HU variants | Planned ablations | Must state clipping/windowing/statistics before launch | Same orientation and mask policy unless explicitly changed | Normalization ablation; do not mix with spacing changes without a matrix |

The two exp011 caches completed on `2026-07-28` with `3,192` cases, `8,068`
nonempty targets, and zero geometry, target, HU-header, or normalization audit
failures:

- `crop_clip1024_zscore_native_v1` manifest SHA256:
  `f834ec24e16c0787f02f2e0589a10fafe16a2962dfffdb3666ea9a83eb425c35`.
- `crop_clip1024_linear_native_v1` manifest SHA256:
  `0ba44e6792b32a91159a1aa9d35465c732ed6c6bf7122f63368634c59e298229`.
- Shared target-index SHA256:
  `1cafeb8920934e01d020d405b95e7b2352b5239148a4c2a20d159db25ae92581`.

The evidence and interpretation behind the two HU variants are recorded in
`ct_hu_normalization_analysis.md`.

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

Build both experiment 011 caches in one raw-CT pass:

```bash
python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
  --preprocess-id crop_clip1024_zscore_native_v1 \
  --preprocess-id crop_clip1024_linear_native_v1 \
  --reference-cache-root /mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1 \
  --splits train val \
  --num-workers 4
```

Build the experiment 016 0.7 mm isotropic HU-linear cache:

```bash
python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
  --preprocess-id crop_clip1024_linear_iso07_v1 \
  --splits train val test \
  --num-workers 16 \
  --multiprocessing-start-method spawn \
  --worker-threads 1 \
  --estimate-cases 30
```

For this 0.7 mm cache, the 2026-08-14 calibration found that fork-style
multiprocessing after torch-backed estimation could make no case-level writes.
Use `spawn` plus one torch/OpenMP thread per worker. A 45-case calibration with
`16` workers matched the serial content hashes and reduced build time from
`14:03` to `1:14`.
