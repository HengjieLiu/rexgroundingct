---
created: 2026-08-14
updated: 2026-08-14
status: draft
experiment_id: "016_voxtell_iso07_hu_preprocessing"
---

# Codex Execution Spec

## Objective

Build and audit one preprocessing cache for the next VoxTell finetuning test:
cropped CT and labels resampled to 0.7 mm isotropic resolution with fixed HU
normalization.

## Preprocessing Contract

- Cache ID: `crop_clip1024_linear_iso07_v1`.
- Cache root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`.
- Splits: `train`, `val`, and `test`.
- Labels: required for `train` and `val`; absent/omitted for `test`.
- Image orientation: `NibabelIOWithReorient`, matching the existing VoxTell
  cache helpers.
- Target orientation: released `(F,X,Y,Z)` masks converted to VoxTell
  `(F,Z,Y,X)` with the CT affine-derived transform.
- Crop: crop CT to nonzero in native geometry, and apply the same crop to each
  target.
- Normalization: clip cropped HU to `[-1024,1024]`, divide by `1024`, then
  resample.
- Image resampling: trilinear to `0.7 x 0.7 x 0.7 mm`, `align_corners=False`,
  no antialiasing.
- Mask resampling: nearest-exact to the same shape, with foreground-center
  splat fallback if a nonempty target disappears.
- Storage schema: `image.npy` as `float32 (1,Z,Y,X)`, `targets.npz` as
  `uint8 (F,Z,Y,X)` when labels exist, per-case `metadata.json`, root
  `manifest.json`, and `.complete` markers.
- Padding values for later model use: image `-1`, mask `0`.

## Runtime Plan

Run the builder inside the VoxTell Docker environment:

```bash
python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
  --preprocess-id crop_clip1024_linear_iso07_v1 \
  --splits train val test \
  --num-workers 16 \
  --multiprocessing-start-method spawn \
  --worker-threads 1 \
  --max-cases-per-split 15 \
  --estimate-cases 30
```

Recommended sequence:

1. Smoke 3-5 mixed cases with `--max-cases-per-split 1` or explicit
   `--case-name`.
2. Calibrate 30-50 mixed cases and record seconds/case plus output bytes.
3. Build all splits with `--num-workers 16 --multiprocessing-start-method spawn
   --worker-threads 1`.

Expected full-cache scale from the native-resolution audit:

- all-split CT images: about `1.73 TiB` uncompressed full-FOV;
- train+val dense labels: about `1.00 TiB` uncompressed full-FOV;
- actual cache size may differ because this build crops before resampling and
  stores labels as compressed `.npz`.

Observed calibration on 45 mixed cases completed with `--num-workers 16`,
`--multiprocessing-start-method spawn`, and `--worker-threads 1`. It matched
the serial content and target hashes while reducing build time from `14:03` to
`1:14`.

## Required Audit

- CT missing count is `0`.
- Train/val label missing count is `0`.
- Test labels are expected-missing and image-only entries have no
  `targets.npz`.
- Train/val target count stays nonempty after nearest-exact resampling and
  fallback.
- HU header checks pass for all cases.
- Resampled image stats remain within `[-1,1]`.
- Manifest records target spacing `[0.7, 0.7, 0.7]`, image padding `-1`, and
  target padding `0`.

## Out Of Scope

- No resampled prediction exports.
- No checkpoint creation.
- No finetuning launch.
- No parallel 1.0 mm cache.
