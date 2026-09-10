# Val200 GT outside the downloaded lung mask

The original zero-margin analysis measures all 381 finding masks in the fixed 200-scan validation
cohort. Whole lung is the exact union of CT-RATE `ts_total` labels 10–14 from
Exp020. Positive GT instance IDs are unioned within each finding. No resampling,
dilation, inference, or source modification occurs.

## Files

- `analyze.py`: native-grid measurements, independent Exp022 checks, CSV reports,
  and paired voxel/volume distribution plots.
- `test_analyze.py`: measurement, geometry, aggregation, and plotting tests.
- `codex_execution_spec.md`: inputs, execution, and acceptance criteria.
- `report.md`: generated aggregate results without case identifiers.
- `analyze_20mm.py`: same-source +20 mm rerun, cached overlap verification,
  direct PP2 EDT checks, and comparison against zero expansion.
- `test_20mm.py`: expansion, integer-count recovery, and eligibility tests.
- `report_20mm.md`: +20 mm aggregate comparison and figure links.

## Usage

Run with Python, NumPy, nibabel, and Matplotlib (available in the existing
`rexgroundingct-voxtell:cu126` Docker image); no GPU is needed.

```bash
python visualization/2026-09-09_val200_gt_outside_lung/analyze.py --workers 2
python -m unittest discover -s visualization/2026-09-09_val200_gt_outside_lung -p 'test_*.py'
```

The default external output root is
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung`.
`--output-dir` overrides it; output inside the repository is rejected. `--plots-only`
regenerates reports/plots from measured JSON after verifying input hashes and
the archived measurement source. Measurement and rendering code hashes are
recorded separately so figure refinements do not require reading all masks again.
Mount source data read-only, the output directory writable, and run as the host
UID/GID in Docker. Set `MPLCONFIGDIR=/tmp/matplotlib` and
`PYTHONDONTWRITEBYTECODE=1`.

## Interpretation

M/N counts findings with any outside voxel / all findings in a category. Each
finding has equal weight in distributions. Empty GT is explicitly flagged and
has undefined outside percentage. Category 2f has no findings.

Physical volumes use the native CT affine determinant in mm³, with reports in
mL. CT spatial units must be mm. Released finding-first GT headers use identity
affines; their spatial index shape is checked against CT, following the existing
Exp022 convention. Source mask and GT hashes and CT headers are verified; CT
intensity payloads are not rehashed because this analysis only uses CT geometry.

Category totals sum findings and can count overlapping voxels repeatedly.
Per-case rows instead measure the union of all finding masks. Volume aggregation
converts each finding using its own scan spacing before summation. Percentage
histograms use 5-point bins; paired count/volume histograms separate zeros from
logarithmic positive bins. Cumulative plots use log1p coordinates with ticks in
actual units. Source human visual QC remains pending; this diagnostic report
does not change that status.

All detailed CSVs, plots, measurements, input hashes, and run provenance stay
external. Only the compact aggregate report is copied into this folder.

## Completed analysis

See [the aggregate report](report.md) for all category M/N values, paired
voxel/volume statistics, and direct PNG/PDF links. The full calculation found
271/381 findings (71.1%) and 159/200 scan unions with outside GT. There were no
empty GT findings. Across findings, outside GT sums to 5,064,858 voxels and
2,248.780 mL; these sums can include overlapping findings.

All 200 native grids, source/GT hashes, and 381 Exp022 count comparisons passed.
The external package contains 32 figure layouts in both PNG and PDF, 381 finding
rows, and 200 case rows. Independent artifact verification checked all 71 output
hashes, CSV arithmetic, summary statistics, union bounds, and plotted populations.
Eight focused tests and the repository workflow check passed; the workflow
checker retained 19 existing experiment-record warnings.

Five PNGs were visually inspected: category 1a and 2d paired distributions,
both paired overview pages (covering every category), and percentage overview
page 2. Labels and layout were readable, including zeros and the empty category.
The external `verification.json` records the inspected image hashes, checks,
and exact container image ID. Source human anatomical QC remains pending.

## +20 mm rerun

```bash
python visualization/2026-09-09_val200_gt_outside_lung/analyze_20mm.py --workers 2
```

Outputs go to the original external root plus `/lung20mm`; the zero-margin
outputs remain unchanged. The rerun verifies source/GT hashes for all 200 scans,
reuses Exp022's +20 mm per-finding coverage, and directly applies the original
PP2 EDT implementation to every scan with remaining outside GT and a
deterministic fully-contained representative per category where available.
Direct results must exactly match the cached integer counts. Case unions are
measured directly in those scans; all other scans have zero outside GT in every
finding and therefore in their union.

All-category plots describe geometry for all 381 findings. The comparison also
records frozen PP2 eligibility (366 eligible, 15 bypassed), with zero effective
GT-at-risk counts for bypassed findings. GT outside a support is an upper bound
on possible GT clipping; actual prediction damage is not measured in this audit.

The [completed +20 mm report](report_20mm.md) records 14/381 findings and 12/200
scan unions with remaining outside GT, compared with 271/381 and 159/200 at
zero margin. Of the 366 PP2-eligible findings, seven retain outside GT; the other
seven geometrically affected findings bypass PP2. Outside GT summed across
findings decreases from 5,064,858 to 537,073 voxels and from 2,248.780 to
252.648 mL. Eligible GT at risk sums to 48.283 mL. One remaining category-1f
finding has 98.816% of its GT outside the expanded mask.

All 200 source-mask/GT hash checks passed. Direct PP2 EDT calculations on 23
scans / 62 findings (all 12 affected scans plus 11 fully-contained representatives)
exactly match the cached coverage. Thirteen focused tests and workflow checks
passed, with the same 19 pre-existing experiment-record warnings. All 73 output
hashes, CSV arithmetic, plot populations, 32 PNGs, 32 PDFs, and report links
passed independent verification. All 71 original zero-margin output hashes
remain unchanged. Four PNGs were visually inspected: both paired overview
pages, category 2b paired plots, and category 1f percentages. The external
`lung20mm/verification.json` records the checks and inspected image hashes.
