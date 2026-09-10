# Execution spec: val200 GT outside-lung analysis

## Contract

Implement the user-approved plan: 200 scans / 381 findings, exact downloaded
whole-lung labels 10–14, GT > 0, native voxel counts, physical volumes, outside
percentages, category M/N, and paired count/volume summaries and plots.
Preprocessing is unchanged; there is no image normalization, resampling, crop,
model execution, or alteration of existing experiments.

## Inputs and execution

- Fixed cohort: `configs/evaluation/rexgroundingct_val200_seed20260723.json`,
  SHA-256 `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.
- Metadata/GT root: `/data/hengjie/datasets/rexgroundingct`.
- Exp020 native source audit: runtime `reports/private/audit_results.json`.
- Independent counts: Exp022 runtime `cases/*.json`, `all_label_gt_voxels`.
- Runtime parent: `/mnt/shengdata1/hengjie/experiments/rexgroundingct`.
- Output: runtime parent plus `visualizations/2026-09-09_val200_gt_outside_lung`.
- Command: `python visualization/2026-09-09_val200_gt_outside_lung/analyze.py --workers 2`.
- Existing `rexgroundingct-voxtell:cu126` container, host UID/GID, CPU only,
  read-only data/repo mounts and one writable output mount.

## Outputs and gates

Write per-finding and per-case CSVs, category summary CSV/Markdown, individual
and combined PNG/PDF histograms and ECDFs, measured JSON, and hashed manifest.
Report paired voxel/mL statistics (sum/mean/median/p95/max), percentage statistics,
and empty-category status. Cases use a union; category sums are finding sums.
Persist source manual-QC status without promoting it.

Require exact cohort/category census, verified source/GT hashes, matching native
geometry and mm units, nonnegative integer GT/anatomy, nonempty lung, and exact
Exp022 count agreement. Unexpected empty GT is recorded with null percentage.
Stop on missing inputs, mismatched hashes, geometry, labels, or census.
CT headers are checked against Exp020; the CT intensity payload is not needed.

Test synthetic masks, spacing conversion, overlapping findings, empty inputs,
geometry rejection, and plot population accounting before the full run. Run
the repository workflow checker and inspect exported figures at closeout. Copy
only the compact aggregate report into the repository. No staging or commit.

## Approved +20 mm rerun

Use the exact same source masks and physical expansion as method 2 in
`run_024_test_inference_anatomy.py:physical_dilation`. Preserve the original
outputs. The new entrypoint is `analyze_20mm.py`; external output is the original
output root plus `/lung20mm`. Retain the aggregate as `report_20mm.md` locally.

Reuse Exp022's per-finding `all_lung:mask:20` GT coverage after checking all GT
and source-mask hashes against the verified zero-margin run. Recover integer
inside counts only when the floating-point product is within 1e-6 of an integer.
Require the denominator to equal the independently verified GT voxel count.
Directly recompute all scans with any remaining outside GT using the actual
method-2 EDT helper; also check a deterministic fully-contained representative
for each category where one exists. Compare every finding of every selected
scan against the stored counts. Compute case unions directly for these scans;
for other scans, all findings have zero outside voxels, so the union also has
zero outside voxels and its total size is reused from the zero-margin analysis.

Record the provenance of every reused/recomputed result. Compare zero-margin
and +20 mm M/N, voxels, volume, and percentage per category. Keep all findings
in geometric distributions and separately report the frozen method-2 eligible
subset (366 findings, 15 bypassed). Eligible GT outside the support is described
as GT at risk of clipping, not as an observed prediction Dice regression.

Tests cover count recovery, invalid coverage, monotonicity, anisotropic physical
dilation, the 20 mm boundary, case-union reasoning, and eligibility bypass.
Run focused tests and workflow checks; validate all CSV/figure hashes and
inspect the paired distribution figures. No model inference or saved prediction
modification is part of this rerun.
