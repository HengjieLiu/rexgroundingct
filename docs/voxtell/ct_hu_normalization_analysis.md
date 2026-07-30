---
created: 2026-07-27
updated: 2026-07-27
status: active
experiment_id: 011_voxtell_v123_e4d4_ct_normalization_ablation
---

# CT HU Normalization Analysis For VoxTell

## Bottom Line

HU-based preprocessing is worth testing, but the strongest immediate
opportunity is robust clipping and eventually nonlinear or multi-window HU
input, not merely replacing z-score with another affine mapping.

The first controlled test therefore keeps model geometry and training fixed
while comparing:

1. Current full cropped-volume z-score.
2. Clip to `[-1024, 1024]` HU, then full cropped-volume z-score.
3. Fixed linear HU: `clip(HU, -1024, 1024) / 1024`.

Experiment 011 initializes all three arms from public VoxTell v1.1 and uses the
same v123 e4/d4 training recipe. It is not a continuation from a fine-tuned
checkpoint.

## Audit Provenance

The read-only analysis covered:

- fixed validation JSON:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`;
- dataset JSON SHA256:
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`;
- 200 validation CTs;
- 381 released finding masks;
- 32,527,800 foreground target voxels.

The reproducible implementation is:

```text
scripts/rexgroundingct/audit_voxtell_ct_hu_normalization.py
```

The final audit completed at `2026-07-28T07:53:44Z` with script version `2`
and script SHA256
`354660bb493851addc8cfe1e37c1f334c2b7d6565af13c6799c368d0003df93a`.
It read CTs from
`/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct` and targets from
`/data/hengjie/datasets/rexgroundingct/segmentations`.

Machine-readable output belongs under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/
  011_voxtell_v123_e4d4_ct_normalization_ablation/analysis/
```

The JSON records the command, script hash, source roots, dataset hash, scan
time, per-case statistics, per-finding statistics, and category summaries.

## Can We Trust The HU Values?

The evidence is strong that the local `*_fixed` NIfTIs already contain
materialized HU values:

- All 200 validation images are stored as `int16`.
- All have effective NIfTI slope `1.0` and intercept `0.0`.
- All have LPS orientation and exactly match their released GT dimensions.
- Air-space findings center near `-1000`, pleural fluid near `0`, soft tissue
  around `0` to `100`, and calcification is positive.

The source DICOM relationship is:

```text
HU = stored_value * RescaleSlope + RescaleIntercept
```

The [pydicom rescale documentation](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_rescale.html)
describes this modality rescaling and cites the DICOM standard. The official
[CT-CLIP NIfTI loader](https://github.com/ibrahimethemhamamci/CT-CLIP/blob/main/scripts/data_inference_nii.py)
applies the CT-RATE metadata slope/intercept, clips to `[-1000, 1000]`, divides
by `1000`, and uses `-1` for padded image regions. The
[CT-RATE fixed-header discussion](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/discussions/58)
records why corrected spacing, scaling, orientation, and datatype metadata were
needed.

The original source DICOM series are not available locally, so voxelwise
DICOM-to-NIfTI identity cannot be proven. The header, geometry, and tissue-value
checks are nevertheless sufficient for this experiment to treat the loaded
fixed-NIfTI values directly as HU.

**Do not apply DICOM slope/intercept again.** Doing so would double-transform
already materialized values.

## Problems With Unclipped Z-Score

VoxTell first crops the complete image to nonzero and then computes one
full-crop z-score. It does not z-score each `192^3` patch independently.

Across the 200 validation CTs:

| Cropped statistic | Median | 5th-95th percentile |
| --- | ---: | ---: |
| Mean | `-558 HU` | `-740` to `-433 HU` |
| Standard deviation | `484 HU` | `456` to `1313 HU` |

Representative cases show how the same physical value can move substantially:

| Case | Mean / SD | `-1000 HU` after z-score | `0 HU` after z-score | Observation |
| --- | ---: | ---: | ---: | --- |
| `train_13082_a_1` | `-517 / 502` | `-0.96` | `1.03` | Typical despite sparse high outliers |
| `train_13591_a_1` | `-309 / 445` | `-1.55` | `0.69` | Different field and intensity composition |
| `train_2560_d_2` | `-790 / 407` | `-0.51` | `1.94` | Same HU maps very differently |
| `train_18416_a_1` | `-1553 / 2621` | `+0.21` | `0.59` | `14.2%` below `-1024`, including `-8192` padding |
| `train_13013_a_1` | `-665 / 1313` | `-0.26` | `0.51` | Approximate range `-31891` to `31919` |

Crop-to-nonzero does not solve this problem. Sentinel and artifact values such
as `-8192` are nonzero, remain in the cropped image, and influence both mean
and standard deviation. In the most extreme case, air at `-1000 HU` even
changes z-score sign.

Clipping before z-score is therefore a distinct and testable intervention:
it preserves the pretrained model's standardized input convention while
removing implausible padding, metal, and reconstruction outliers from the
per-case statistics.

## Ground-Truth HU Distribution

Across all 32,527,800 foreground voxels:

- voxel-weighted `P5/P25/P50/P75/P95`:
  `-991 / -727 / -295 / 2 / 48 HU`;
- finding-weighted median HU: median `-600 HU`;
- `0.084%` of target voxels exceed `400 HU`;
- `0.0086%` exceed `1000 HU`;
- `2.27%` fall below `-1000 HU`, predominantly in air-space findings.

The challenge category codes provide a reproducible grouping:

| Finding type | Code | Findings | Target HU P5 / median / P95 |
| --- | --- | ---: | ---: |
| Bronchial wall thickening | `1a` | 3 | `-885 / -489 / 29` |
| Bronchiectasis | `1b` | 11 | `-1003 / -794 / -7` |
| Emphysema | `1c` | 17 | `-969 / -894 / -447` |
| Septal thickening | `1d` | 6 | `-922 / -527 / 25` |
| Micronodules | `1e` | 11 | `-937 / -821 / -223` |
| Other diffuse | `1f` | 4 | `-816 / -727 / 89` |
| Linear/scarring | `2a` | 69 | `-893 / -620 / -32` |
| Consolidation/atelectasis | `2b` | 49 | `-772 / -37 / 87` |
| Ground glass | `2c` | 60 | `-831 / -487 / 15` |
| Nodules/masses | `2d` | 132 | `-896 / -633 / 34` |
| Pleural effusion/thickening | `2e` | 11 | `-42 / 12 / 53` |
| Pneumothorax | `2g` | 1 | `-1013 / -990 / -770` |
| Other focal | `2h` | 7 | `-999 / -666 / 9` |

These bands support HU as useful evidence, especially for air-space versus
fluid and soft-tissue findings. They also overlap substantially, so HU cannot
replace anatomical localization, morphology, or text-conditioned reasoning.

Voxel-weighted summaries understate very small calcified targets. One focal
calcification finding had median intensity around `375 HU`, with approximately
`46%` of its voxels above `400 HU`.

## Model Caveat: Instance Normalization

The VoxTell encoder is configured with:

```text
Conv3d -> affine InstanceNorm3d -> LeakyReLU
```

For a purely affine input transformation:

```text
InstanceNorm(Conv(a*x + b)) ~= InstanceNorm(Conv(x))
```

The approximation is not exact because of convolution bias, padding, finite
windows, learned affine parameters, and nonlinear clipping. Still, the first
InstanceNorm makes the architecture poorly positioned to retain absolute HU
calibration from a single fixed-linear channel.

What can survive more clearly is:

- clipping of sentinel, metal, and reconstruction outliers;
- nonlinear window transforms;
- relative differences among multiple HU-window channels;
- padding and image-boundary behavior;
- downstream adaptation during fine-tuning.

This is why fixed linear HU may yield less improvement than expected even if
the HU values themselves are valid.

## Experiment 011 Interpretation

All arms share public VoxTell v1.1 initialization, native geometry, identical
targets, the same 10,000-event schedule, v123 loss, and e4/d4 optimization.

| Arm | Transform | Image padding |
| --- | --- | ---: |
| `v123_e4d4_zscore` | Full cropped-volume z-score | `0.0` |
| `v123_e4d4_clip1024_zscore` | Clip `[-1024,1024]`, then full-crop z-score | `0.0` |
| `v123_e4d4_clip1024_linear` | Clip `[-1024,1024]`, divide by `1024` | `-1.0` |

Zero padding is retained for z-score arms because zero is the normalized
full-crop mean and matches VoxTell's existing patch behavior. It is numerically
neutral, not a physical representation of air.

For fixed linear HU, zero represents approximately water. Padding with zero
would create artificial water-density slabs at image boundaries. The correct
fixed-linear padding is `-1`, corresponding to clipped air at `-1024 HU`.
Target padding remains zero for every arm.

Interpret the outcomes as follows:

- Arm 2 improves and arm 3 does not: outlier clipping helped, while absolute HU
  calibration was mostly neutralized by InstanceNorm.
- Arm 3 improves: fixed HU information survived sufficiently to justify a
  multi-window or learned-window adapter.
- Neither improves: keep the pretrained z-score convention and prioritize
  spatial/context methods.
- Both improve: separate calibration from clipping in a follow-up and test
  nonlinear or multi-channel windows.

## Cache Consequence

The existing z-score cache cannot be inverted into HU because it does not store
each case's raw cropped image or original mean and standard deviation.
Both new experiment 011 caches must therefore be generated from the original
fixed NIfTIs while retaining the established orientation, crop boxes, targets,
and patch schedule.

See `normalization.md` for the broader normalization policy and
`preprocessing_variants.md` for the cache contract.

## Primary References

- CT-RATE fixed-NIfTI metadata correction:
  [dataset discussion 58](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/discussions/58).
- CT-CLIP reference NIfTI preprocessing:
  [official `data_inference_nii.py`](https://github.com/ibrahimethemhamamci/CT-CLIP/blob/main/scripts/data_inference_nii.py).
- DICOM rescale slope/intercept semantics:
  [pydicom `apply_rescale`](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_rescale.html).
