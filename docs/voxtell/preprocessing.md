---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# VoxTell Direct Inference Preprocessing Workflow

## Scope

This note documents the preprocessing workflow used by our corrected
ReXGroundingCT VoxTell direct inference pipeline for Experiment 001.

For standardized cached preprocessing variants and future experiment contracts,
see `preprocessing_variants.md`.

Primary implementation points:

- VoxTell predictor:
  `external/VoxTell/voxtell/inference/predictor.py`
- ReXGroundingCT wrapper:
  `scripts/rexgroundingct/run_voxtell_val_inference.py`
- Canonical validation experiment:
  `001_voxtell_v1_1_miccai200_val_eval`

## Inputs

For each MICCAI validation case, the wrapper uses:

- CT volume from the CT-RATE fixed layout under
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Ground-truth ReXGroundingCT segmentation from
  `/data/hengjie/datasets/rexgroundingct/segmentations`
- Free-text findings from
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`

One output NIfTI is written per case, with one channel per finding:

```text
(F, X, Y, Z)
```

where `F` is the number of text findings for that case.

## Image Loading and Orientation

The CT image is loaded with nnU-Net's `NibabelIOWithReorient`.

This reader:

1. Loads the original NIfTI with nibabel.
2. Stores the original affine.
3. Reorients the image using nibabel orientation metadata.
4. Stores the reoriented affine.
5. Reads voxel data with `get_fdata()`.
6. Transposes the image into nnU-Net axis order.
7. Returns image data as:

```text
(C, Z, Y, X)
```

For single-channel CT, `C = 1`.

Important consequence:

VoxTell predicts in the reoriented nnU-Net voxel space, not directly in the raw
CT file's original voxel axis order.

## Image Preprocessing Inside VoxTell

VoxTell receives the image in nnU-Net layout and then applies:

1. Channel handling
   - If the input is 3D, add a channel dimension.
   - Our wrapper already supplies a single-channel 4D array.

2. Type conversion
   - Convert image data to `float32`.

3. Shape recording
   - Store the original pre-crop shape.

4. Nonzero crop
   - Use nnU-Net `crop_to_nonzero(data, None)`.
   - The nonzero mask is based on voxels not equal to zero.
   - In the 200-case validation set, this crop usually did not reduce the CT
     shape, so original and cropped shapes were effectively identical in our
     preprocessing stats.

5. Z-score normalization
   - Apply `ZScoreNormalization(intensityproperties={})`.
   - Since no mask is passed, mean and standard deviation are computed over the
     whole cropped image tensor.

The normalization is:

```python
image = image.astype(target_dtype, copy=False)
mean = image.mean()
std = image.std()
image -= mean
image /= max(std, eps)
```

There is no CT HU clipping, no CT windowing, and no resampling in the direct
inference path.

## Image Scale and Patch Geometry

The public `voxtell_v1.1` plan uses:

```text
patch_size: [192, 192, 192]
normalization_schemes: ["ZScoreNormalization"]
spacing: None
resampling_fn_data: None
resampling_fn_seg: None
```

Sliding-window inference pads images smaller than the patch size and uses a
tile step size of `0.5`, meaning 50 percent patch overlap.

For the 200-case validation set, observed tile counts were:

```text
min:    25
median: 98
max:    125
mean:   84.885
```

Observed validation image shapes in nnU-Net `Z, Y, X` order:

```text
Z min / median / max: 172 / 346 / 568
Y min / median / max: 512 / 512 / 768
X min / median / max: 512 / 512 / 768
```

## Text Preprocessing

For each case, the wrapper extracts findings from the metadata and sorts them
by numeric key order. Each finding becomes one prompt.

The prompt preprocessing is:

1. Convert prompt strings to lowercase.
2. Check the optional precomputed embedding bank.
3. If a prompt is present in the bank, load its cached embedding.
4. If a prompt is missing, embed it on the fly with `Qwen/Qwen3-Embedding-4B`.

The public embedding bank for `voxtell_v1.1` contains:

```text
labels:     14194
embeddings: 14194 x 2560, float16
```

For our MICCAI validation split:

```text
total prompts: 381
unique prompts: 365
embedding-bank hits: 0
```

So all validation findings were embedded on the fly.

## Qwen Embedding Path

For prompts that are not in the bank, VoxTell:

1. Wraps each lowercase finding with its embedding instruction.
2. Tokenizes with the Qwen tokenizer.
3. Uses left padding.
4. Truncates at `max_text_length = 8192`.
5. Runs `Qwen/Qwen3-Embedding-4B`.
6. Applies last-token pooling.
7. Converts embeddings to `float32`.
8. Caches computed embeddings in memory as `float16` for reuse within the same
   session.

The resulting tensor shape passed to the segmentation model is:

```text
(1, F, 2560)
```

where `F` is the number of findings/prompts for that case.

## Model Inference

For each sliding-window patch:

1. Copy the patch to the selected CUDA device.
2. Run VoxTell with the image patch and text embeddings.
3. Produce one logit volume per prompt.
4. Apply Gaussian blending to merge overlapping patch predictions.
5. Divide by the accumulated Gaussian weights.
6. Return logits in nnU-Net layout:

```text
(F, Z, Y, X)
```

The wrapper converts logits to binary masks by using VoxTell's standard
postprocessing path, which applies a sigmoid threshold.

## Export to ReXGroundingCT Layout

The corrected exporter is necessary because VoxTell output and ReXGroundingCT
ground truth use different axis conventions.

VoxTell output:

```text
(F, Z, Y, X)
```

ReXGroundingCT evaluator expected layout:

```text
(F, X, Y, Z)
```

The corrected export does:

1. Confirm the raw prediction shape matches expected nnU-Net layout:

```text
(F, gt_Z, gt_Y, gt_X)
```

2. Transpose:

```text
(F, Z, Y, X) -> (F, X, Y, Z)
```

3. Compute the nibabel orientation transform from the reoriented CT affine back
   to the original CT affine.

4. Apply that transform independently to each finding mask.

5. Save the result as `uint8` NIfTI using the ground-truth affine and header.

6. Verify final prediction shape exactly equals the ground-truth shape.

The status JSON records:

- raw prediction shape
- GT shape
- CT original axis codes
- CT reoriented axis codes
- orientation transform
- final prediction shape
- GPU/runtime metadata

## Why the Orientation Fix Mattered

The first validation pass saved predictions in a shape-compatible but
orientation-wrong layout. Because some cases can have dimensions that appear
compatible after a naive transpose, shape checks alone were not sufficient.

The corrected pipeline treats VoxTell output as nnU-Net layout first, then uses
affine-derived orientation metadata to return predictions to the evaluator's
original CT voxel space.

## Current Standard Evaluation

Experiment 001 is standardized on the corrected 200-case quick/global
evaluation.

Primary challenge metric:

```text
Mean global Dice per finding: 0.22522803991156426
```

Secondary metric:

```text
Hit rate: 0.5354330708661418
```

This quick/global evaluation does not compute connected-component instance
precision, recall, or F1.
