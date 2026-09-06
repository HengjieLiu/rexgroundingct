# Task: Plan and prepare download/audit of CT-RATE TotalSegmentator masks for ReXGroundingCT challenge validation set

## Goal

We are working on the ReXGroundingCT MICCAI 2026 challenge.

For now, do **NOT** download the full CT-RATE segmentation dataset and do **NOT** modify training/inference code.

The immediate goal is to:

1. identify the exact 200 ReXGroundingCT MICCAI challenge validation CT volumes;
2. map each ReX validation `VolumeName` back to the corresponding CT-RATE volume;
3. download only the corresponding CT-RATE **TotalSegmentator `ts_total` segmentation** for these 200 validation scans;
4. verify rigorously that each downloaded segmentation has the same native image geometry as the existing, non-resampled CT-RATE volume currently used by our ReXGroundingCT pipeline;
5. produce a concise audit report before we use these segmentations anywhere downstream.

Do this carefully and reproducibly.

---

# Background

## ReXGroundingCT

Official dataset:

https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT

The MICCAI 2026 challenge added:

`MICCAI_challenge_dataset.json`

The challenge version contains:

* validation: 200 scans, GT finding masks released;
* test: 300 scans, finding masks withheld.

The important field for us is:

```json
"name": "train_1741_b_2.nii.gz"
```

The `name` is the original CT-RATE `VolumeName`.

IMPORTANT:

The ReX challenge's `val` / `test` split is independent of CT-RATE's original `train` / `valid` split.

Therefore, a ReX validation case may still have a filename beginning with:

```text
train_...
```

and should then be found in CT-RATE `train_fixed`.

Similarly:

```text
valid_...
```

should map to CT-RATE `valid_fixed`.

Do NOT assume that ReX validation means CT-RATE `valid_fixed`.

---

# CT-RATE resources

Official CT-RATE Hugging Face repository:

https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

Relevant dataset tree:

```text
dataset/
├── train_fixed/
├── valid_fixed/
└── ts_seg/
```

CT-RATE contains TotalSegmentator-derived segmentation outputs.

The relevant segmentation task for the current experiment is:

```text
dataset/ts_seg/ts_total/
```

There are also other segmentation tasks such as:

```text
ts_lung_nodules
ts_pleural_pericard_effusion
```

but DO NOT download those yet unless needed for metadata inspection.

Our immediate target is **`ts_total` only**.

`ts_total` is a multi-label TotalSegmentator anatomy segmentation volume containing the normal TotalSegmentator `total` anatomy classes, including the five lung lobes.

Later we will derive:

```text
left lung =
    lung_upper_lobe_left
    UNION
    lung_lower_lobe_left

right lung =
    lung_upper_lobe_right
    UNION
    lung_middle_lobe_right
    UNION
    lung_lower_lobe_right
```

But do NOT yet change downstream preprocessing to use these masks.

---

# Important known CT-RATE issue

Three CT-RATE training volumes are known to be missing `ts_total` segmentations because of problematic/missing z-spacing metadata:

```text
train_1267_a_4.nii.gz
train_11755_a_3.nii.gz
train_11755_a_4.nii.gz
```

Reference:

https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/discussions/97

If any of these happen to appear in the ReX 200-case validation set:

* do not silently fail;
* record them explicitly as known upstream missing segmentations;
* do not generate replacements yet.

---

# Phase 0 — Inspect the current repository and data layout

Before downloading anything:

1. inspect the current RexGroundingCT repository;
2. locate the existing `MICCAI_challenge_dataset.json` if already present;
3. locate the existing native-resolution CT-RATE CT volumes currently used for the challenge;
4. determine whether those CT files correspond to CT-RATE `train_fixed` / `valid_fixed`;
5. identify the exact local paths.

Do NOT move or modify existing CT data.

Report the discovered paths.

If more than one CT-RATE copy exists, determine which one the current ReXGrounding pipeline actually uses.

---

# Phase 1 — Build the exact ReX validation manifest

Parse:

```text
MICCAI_challenge_dataset.json
```

and extract the 200 challenge validation entries.

Create an explicit manifest containing at minimum:

```text
rex_split
volume_name
ct_rate_original_split
expected_ct_path
expected_ts_total_hf_path
local_ts_total_path
```

Expected number of unique validation volumes:

```text
200
```

Assert that the number is exactly 200.

If not, stop and report why.

Determine CT-RATE split from the filename prefix:

```text
train_* -> train_fixed
valid_* -> valid_fixed
```

Do not infer it from the ReX split.

---

# Phase 2 — Derive the Hugging Face path for each TotalSegmentator mask

CT-RATE stores volumes hierarchically.

For a volume such as:

```text
train_1741_b_2.nii.gz
```

derive the patient/scan hierarchy using the same convention used by CT-RATE.

Existing CT-RATE download examples construct paths such as:

```text
dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
```

and replace the normal CT directory with:

```text
dataset/ts_seg/ts_total/train_fixed/...
```

for the corresponding segmentation.

Use the actual Hugging Face repository listing/API to verify the path convention instead of blindly assuming it.

For every ReX validation volume:

1. construct the expected remote path;
2. verify that the remote file exists;
3. record existence status.

Before downloading all files, print a table for the first ~10 cases showing:

```text
VolumeName
CT-RATE split
local CT path
remote ts_total path
remote exists?
```

---

# Phase 3 — Hugging Face authentication

CT-RATE is gated.

Do NOT attempt to bypass gating.

If Hugging Face authentication is missing or access has not been granted:

STOP at this point and give me the exact command/action needed.

I can manually:

* log into Hugging Face;
* accept the CT-RATE dataset terms;
* provide/login with a Hugging Face access token.

Prefer standard Hugging Face authentication, e.g. environment/CLI authentication.

Do NOT print, save, or commit my Hugging Face token.

Never put credentials into source-controlled files.

Once authentication works, continue.

---

# Phase 4 — Download ONLY the 200 required ts_total masks

Do NOT clone or download the entire CT-RATE repository.

Do NOT download 20+ TB of CT data.

Do NOT download every TotalSegmentator mask.

Download only the files corresponding to the 200 ReX challenge validation scans.

Use `huggingface_hub` / `hf_hub_download` or another targeted Hugging Face mechanism.

Repository:

```text
ibrahimhamamci/CT-RATE
```

Repository type:

```text
dataset
```

Target:

```text
dataset/ts_seg/ts_total/{train_fixed|valid_fixed}/...
```

Preserve a sensible directory structure.

Make downloading:

* resumable;
* idempotent;
* safe to rerun;
* logged.

If a file already exists and is valid, do not unnecessarily download it again.

Create a download manifest containing:

```text
volume_name
remote_path
local_path
download_status
file_size
error_if_any
```

At completion report:

```text
requested: 200
downloaded/present: N
missing upstream: N
failed: N
```

No missing case should be silently ignored.

---

# Phase 5 — Geometry audit against our ACTUAL native CT-RATE volumes

This is scientifically important.

The question is:

> Are the downloaded CT-RATE TotalSegmentator masks truly voxel-aligned with the exact native, non-resampled CT-RATE images currently used by our ReXGrounding pipeline?

For every available validation pair:

```text
native CT
vs
ts_total segmentation
```

inspect using NIfTI metadata.

At minimum compare:

### A. Shape

```python
ct.shape == seg.shape
```

### B. voxel spacing

Compare NIfTI zooms / pixel spacing.

### C. affine

Use a numerical tolerance, for example:

```python
np.allclose(ct.affine, seg.affine, atol=1e-5, rtol=...)
```

and record:

```text
max_absolute_affine_difference
```

### D. orientation

Use something equivalent to:

```python
nib.aff2axcodes(...)
```

and compare CT vs segmentation.

### E. qform / sform

Inspect whether qform/sform and corresponding codes are compatible.

Do not merely check array shape.

A same-shape volume with a different affine is NOT sufficient evidence of alignment.

---

# Phase 6 — Basic segmentation sanity QC

For every mask, record:

```text
unique label IDs
number of nonzero voxels
foreground fraction
```

Confirm that it looks like a valid multi-label TotalSegmentator segmentation rather than an empty/corrupted file.

Retrieve and record the TotalSegmentator label mapping corresponding to the CT-RATE `ts_total` masks.

Prefer the official TotalSegmentator class mapping/version.

Do not hard-code an unverified LUT.

Specifically identify the label IDs for:

```text
lung_upper_lobe_left
lung_lower_lobe_left

lung_upper_lobe_right
lung_middle_lobe_right
lung_lower_lobe_right
```

Record the source/version of the label mapping in the report.

---

# Phase 7 — Build left/right whole-lung masks for QC only

After confirming the LUT, create derived binary masks:

```text
left_lung_native
right_lung_native
```

by unioning the appropriate lobes.

These are for QC at this stage.

Do not yet integrate them into training/inference.

For each side calculate:

```text
voxel count
physical volume in mL
number of 3D connected components
largest-component fraction
```

Flag obvious failures such as:

* completely empty left/right lung;
* extremely small lung;
* highly fragmented segmentation;
* obviously abnormal component structure.

Do not perform aggressive morphological postprocessing yet.

---

# Phase 8 — Visual QC

Generate visual QC for a small representative subset, e.g. 10–20 validation cases.

Overlay:

```text
native CT
left lung boundary
right lung boundary
```

on representative axial/coronal/sagittal slices.

The purpose is specifically to detect:

* orientation mismatch;
* left/right reversal;
* translation/affine mismatch;
* severe TotalSegmentator failures;
* gross lobe union problems.

Do not rely purely on metadata.

---

# Phase 9 — Produce a concise audit report

Write something like:

```text
reports/ct_rate_ts_total_rex_val200_audit.md
```

and CSV/JSON manifests as appropriate.

The report should include:

## Dataset mapping

```text
ReX validation cases: 200

CT-RATE train_fixed cases: X
CT-RATE valid_fixed cases: Y
```

## Download coverage

```text
ts_total found: X/200
missing: ...
failed downloads: ...
```

## Geometry

```text
shape identical: X/N
spacing identical: X/N
affine identical within tolerance: X/N
orientation identical: X/N
```

List every exception individually.

## Lung-mask QC

Summarize:

```text
empty masks
fragmented masks
volume outliers
left/right anomalies
```

## Conclusion

Give one of:

```text
PASS:
The CT-RATE ts_total masks are native-grid aligned with the CT-RATE volumes used by our ReXGrounding pipeline and can be used directly.

PARTIAL PASS:
Most masks align, with listed exceptions requiring handling.

FAIL:
The masks are not on the same geometry and must be resampled before use.
```

Do not declare PASS merely because shapes match.

---

# Phase 10 — STOP before downstream integration

Do NOT yet:

* modify VoxTell/ReX training;
* modify inference;
* add lung priors;
* resample masks to 0.7 mm;
* run R231;
* compare R231 vs TotalSegmentator;
* download train/test masks.

Stop after the validation-set download + geometry/QC report.

We will inspect those results before deciding the next step.

---

# Important implementation principles

1. Reuse the current project environment when practical.
2. Do not modify scientific training behavior.
3. Do not commit credentials.
4. Do not download unnecessary CT-RATE data.
5. All case-level failures must be logged.
6. Make all scripts rerunnable.
7. Preserve original downloaded segmentation files unchanged.
8. Put derived masks/QC artifacts separately.
9. Prefer physical-coordinate correctness over assumptions based on array dimensions.
10. Before making any large download, show the resolved count and estimated download set so we can verify that only the intended 200 masks will be fetched.

At the beginning, first inspect the repository/data layout and present the resolved execution plan and paths. Then proceed through the phases above.
