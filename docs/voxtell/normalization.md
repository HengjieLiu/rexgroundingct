---
created: 2026-07-22
updated: 2026-07-27
status: active
---

# VoxTell Normalization Findings and Plan

## Context

This note summarizes the current understanding of VoxTell image normalization for
ReXGroundingCT and proposes how to handle normalization in direct inference and
future fine-tuning experiments.

For standard cache IDs and preprocessing-variant requirements, see
`preprocessing_variants.md`.

For the complete 200-case HU audit, target-intensity distributions, primary
source references, InstanceNorm analysis, and experiment 011 interpretation,
see `ct_hu_normalization_analysis.md`.

The immediate question was whether VoxTell's z-score normalization is mainly a
multi-modality choice for CT, MR, and PET, and whether raw HU values might be
better for this lung CT challenge.

## Findings

### VoxTell uses z-score normalization in direct inference

The public VoxTell predictor initializes:

```python
self.normalization = ZScoreNormalization(intensityproperties={})
```

During inference, VoxTell preprocessing does:

1. Load the NIfTI volume.
2. Reorient the image into VoxTell/nnU-Net RAS-compatible layout.
3. Add a channel dimension if needed.
4. Convert the image to `float32`.
5. Crop to the nonzero region.
6. Apply z-score normalization over the cropped volume.
7. Run sliding-window inference with patch size `192 x 192 x 192`.

The z-score operation is:

```python
image = image.astype(target_dtype, copy=False)
mean = image.mean()
std = image.std()
image -= mean
image /= max(std, eps)
```

There is no CT-specific clipping, HU windowing, or dataset-level CT
normalization in the public `voxtell_v1.1` direct inference path.

### Side note: normalization scope for patches and multiscale inference

The public inference path normalizes the whole crop-to-nonzero image before
sliding-window patch inference. It does **not** independently z-score each
`192 x 192 x 192` patch.

This matters for future patch-size and multiscale experiments. To stay aligned
with VoxTell pretraining and current fine-tuning, the safe order is:

1. Load and reorient the full CT.
2. Crop the full CT to its nonzero region.
3. Compute one z-score normalization over that cropped full image.
4. Sample, resize, or slide patches from this already-normalized image.

Do not z-score each sampled or resampled patch independently unless the goal is
an explicit normalization ablation. Patch-local normalization would change the
input distribution by removing case-level intensity context and would confound
patch-size or multiscale conclusions.

### VoxTell is not CT-only

The upstream VoxTell documentation describes the model as a universal 3D
medical image segmentation model trained across CT, PET, and MRI. In that
setting, z-score normalization is a reasonable common input transform because
MRI and PET do not have a stable absolute intensity scale comparable to CT HU.

For CT, z-score is still relevant because the public checkpoint was trained and
released with that preprocessing. Changing the normalization at inference time
would create a distribution shift.

### ReXGroundingCT CT files retain HU-like values

I inspected local validation CT NIfTI files under:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct
```

The files are stored as `int16` and retain HU-like intensity ranges. Examples
from sampled validation cases:

| Case | Shape | Percentile pattern / range |
| --- | --- | --- |
| `train_13082_a_1.nii.gz` | `(512, 512, 205)` | min `-1024`, median `-838`, 95th `99`, max `31743` |
| `train_13591_a_1.nii.gz` | `(512, 512, 253)` | min `-2200`, median `-113`, 95th `97`, max `16663` |
| `train_13583_d_2.nii.gz` | `(512, 512, 514)` | min `-1024`, median `-951`, 95th `70`, max `1678` |
| `train_13155_a_2.nii.gz` | `(512, 512, 223)` | min `-8192`, median `-702`, 95th `98`, max `1865` |

Interpretation:

- Air/background is commonly near `-1024`.
- Soft tissue appears around `0` to `100`.
- Bone/high-density material appears as positive values.
- Some cases contain padding/artifact/outlier values such as `-8192` or very
  high maxima.

Therefore, the local CT data does retain HU-like scale. It is not pre-normalized
to `[0, 1]` or zero mean/unit variance on disk.

## Practical Interpretation

Z-score normalization does not mean the CT data lacks HU. It means the public
VoxTell model does not consume raw HU directly. The image's relative contrast is
preserved, but absolute HU thresholds are not directly available to the network
after preprocessing.

For lung CT grounding, raw or windowed HU could plausibly help a CT-specific
model, especially for findings whose visual definition depends on attenuation.
However, switching the public VoxTell checkpoint from z-score to raw HU during
direct inference is not recommended because the checkpoint expects z-scored
inputs.

## Normalization Plan

### Baseline direct inference

Keep VoxTell's default z-score normalization.

Reason:

- This matches the released `voxtell_v1.1` checkpoint.
- It keeps evaluation comparable to the upstream model behavior.
- The corrected Experiment 001 result should remain the canonical direct
  inference baseline.

### First fine-tuning baseline

Keep the same VoxTell z-score normalization for the first ReXGroundingCT
fine-tuning experiment.

Reason:

- It preserves compatibility with pretrained weights.
- It isolates the effect of challenge-specific text-conditioned fine-tuning.
- It avoids mixing two changes at once: adaptation to ReXGroundingCT and a new
  intensity preprocessing scheme.

Recommended experiment role:

```text
002_voxtell_text_ft_miccai_train_val
```

Normalization:

```text
VoxTell default crop-to-nonzero + per-volume z-score
```

### CT-specific ablation

After the z-score fine-tuning baseline is working, run a CT-specific
normalization ablation.

Candidate variants:

1. Clip HU to a broad CT range, then z-score.
   - Example: clip to `[-1024, 1000]`, then per-volume z-score.
   - Lower risk because output range remains standardized.

2. Use dataset-level CT normalization.
   - Compute train-set intensity statistics after clipping.
   - Use fixed mean/std from the training set.
   - This is closer to nnU-Net `CTNormalization`.

3. Use lung/mediastinal window channels.
   - Example channels: lung window and mediastinal window.
   - More invasive because it changes model input channels.
   - Requires adapting or retraining the first convolution/input stem.

4. Raw HU direct input.
   - Not recommended as a first ablation.
   - High risk because of large dynamic range and outliers.
   - Should only be tested with explicit clipping or robust scaling.

The controlled implementation is experiment
`011_voxtell_v123_e4d4_ct_normalization_ablation`. It compares the native
z-score baseline, clipped z-score, and fixed linear HU while holding model,
schedule, loss, geometry, and optimization fixed.

## Recommended Decision

Use this order:

1. Keep z-score for public VoxTell direct inference.
2. Keep z-score for the first challenge-valid fine-tuning baseline.
3. Add CT-specific normalization only as a later controlled ablation.

The key reason is methodological: first establish whether ReXGroundingCT
fine-tuning improves the task under the exact pretrained preprocessing. Then
test whether CT-specific HU handling provides additional gain.
