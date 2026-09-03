---
created: 2026-08-22
updated: 2026-08-22
status: audit_complete
experiment_id: "019_voxtell_iso07_lung_bbox_coverage_audit"
---

# Codex Execution Spec: 0.7 mm Lung Bounding-Box Coverage Audit

## Objective

Build a fixed-val200 audit that merges TotalSegmentator lung lobes, maps the
mask into the existing 0.7 mm VoxTell cache geometry, and measures the minimal
single-valued isotropic bbox expansion needed to contain every GT finding mask.

## Prior Evidence

- Exp015 established CT-header spacing as physical truth.
- Exp016 completed the `crop_clip1024_linear_iso07_v1` cache with nearest-exact
  resampled train/validation targets.
- Exp010 completed the 200-case TotalSegmentator fast 3 mm anatomy cache.

## Scope

In scope:

- Fixed `rexgroundingct_val200_seed20260723` validation cases.
- TotalSegmentator labels 10–14 merged into one lung mask.
- Reuse of the existing per-case iso07 orientation, crop, and target geometry.
- Per-finding directional deficits, isotropic expansion, category histograms,
  and diffuse/local/all summaries.

Out of scope:

- New TotalSegmentator inference.
- Training, model inference, prediction scoring, or mask-overlap metrics.
- Any modification of CT, released GT, or existing anatomy/iso07 caches.

## Inputs And Paths

- Canonical config:
  `configs/experiments/019_voxtell_iso07_lung_bbox_coverage_audit.json`
- Fixed validation manifest:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Released GT root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- TotalSegmentator cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_total_fast_3mm_v2_16_0`
- Existing iso07 cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`
- Derived lung-mask cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_lung_iso07_v1`
- Runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/019_voxtell_iso07_lung_bbox_coverage_audit`
- Repo-local report:
  `experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md`

## Data And Preprocessing Contract

- Baseline: Exp016 `crop_clip1024_linear_iso07_v1` geometry.
- CT and GT orientation: reuse the per-case metadata transform to RAS and then
  `FXYZ` to `FZYX`.
- Crop: reuse `crop_bbox_zyx` from the iso07 case metadata.
- Lung labels: binary union of TotalSegmentator IDs 10, 11, 12, 13, and 14.
- Lung resampling: `torch.nn.functional.interpolate(..., mode="nearest-exact")`
  to the cached `resampled_shape_zyx`.
- GT resampling: use existing `targets.npz`; never resample released GT twice.
- Bbox semantics: inclusive lower/upper indices in iso07 `Z,Y,X` order.
- Direction map: X lower/upper = L/R, Y lower/upper = P/A, Z lower/upper = I/S.

## Method

For each finding, calculate the six nonnegative deficits between the lung and GT
bounding boxes. The requested scalar is:

```text
k = max(deficit_L, deficit_R, deficit_A, deficit_P, deficit_S, deficit_I)
```

The expanded bbox is the lung bbox expanded by `k` on every face and clipped to
the image grid. The report also gives `0.7 * k` millimeters.

The script writes per-case derived lung masks outside Git, per-finding CSV and
JSON summaries, and a Markdown report whose first section is the high-level
result and whose later sections contain provenance, formulas, histograms, and
the complete per-finding table.

## Smoke Gate

Run the focused synthetic geometry tests, then run one fixed validation case in
the documented `rexgroundingct-voxtell:cu126` container with lung-mask writing
disabled. The smoke gate must verify orientation, crop, target channel count,
nonempty masks, bbox construction, and post-expansion containment.

## Success Criteria

- All 200 validation cases and 381 findings are processed.
- No missing input, empty target, empty lung mask, shape mismatch, or geometry
  mismatch remains.
- Every reported expanded bbox contains its GT bbox.
- Category and group histograms sum exactly to their finding counts.
- The Markdown report is copied to the repo-local Experiment 019 directory.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/test_audit_iso07_lung_bbox_coverage.py
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_iso07_lung_bbox_coverage.py --preflight-only
python scripts/rexgroundingct/check_experiment_consistency.py --experiments 019_voxtell_iso07_lung_bbox_coverage_audit
python scripts/rexgroundingct/check_repo_workflow.py
```

The full audit must run inside the VoxTell container because the host shell does
not provide the NIfTI/Torch dependencies.

## Closeout Plan

After the full report passes all invariants, copy the small report and summary
into the repo-local Experiment 019 folder, preserve the external lung masks and
CSV as runtime artifacts, and record the final report hash in the experiment
index without touching unrelated dirty-tree changes.
