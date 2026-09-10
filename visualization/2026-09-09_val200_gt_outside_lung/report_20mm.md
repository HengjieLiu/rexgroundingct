# Val200 GT outside the lung mask expanded by 20 mm

Outside GT decreases from **271/381 findings (71.1%) to 14/381 (3.7%)**. At scan-union level, **12/200 scans** retain outside GT.

Actual method 2 has **7/366 eligible findings** with GT outside its support; 15 findings bypass clipping. These are GT-coverage measurements, not observed prediction losses or Dice regressions.

Same downloaded Exp020 CT-RATE `ts_total` masks, labels 10–14. Expansion uses the actual PP2 physical Euclidean dilation helper with native spacing and a 20 mm radius (voxel-center distance ≤20 mm + 1e-7). No resampling or inference. Source status remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.

| Category | M/N, 0 mm | M/N, +20 mm | Mean outside %, +20 mm | Outside voxels, +20 mm | Outside mL, +20 mm | Eligible M/N, +20 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | 3/3 | 0/3 | 0.000 | 0.000 | 0.000 | 0/3 |
| 1b | 5/11 | 0/11 | 0.000 | 0.000 | 0.000 | 0/11 |
| 1c | 14/17 | 1/17 | 0.003 | 89.000 | 0.043 | 1/17 |
| 1d | 6/6 | 0/6 | 0.000 | 0.000 | 0.000 | 0/6 |
| 1e | 3/11 | 0/11 | 0.000 | 0.000 | 0.000 | 0/10 |
| 1f | 2/4 | 1/4 | 24.704 | 7,430.000 | 2.781 | 1/4 |
| 2a | 66/69 | 0/69 | 0.000 | 0.000 | 0.000 | 0/69 |
| 2b | 48/49 | 4/49 | 0.329 | 113,318.000 | 45.382 | 4/49 |
| 2c | 52/60 | 1/60 | 0.000 | 283.000 | 0.078 | 1/59 |
| 2d | 58/132 | 0/132 | 0.000 | 0.000 | 0.000 | 0/131 |
| 2e | 11/11 | 6/11 | 5.270 | 415,913.000 | 204.351 | 0/0 |
| 2f | 0/0 | 0/0 — no findings | — | 0.000 | 0.000 | 0/0 |
| 2g | 1/1 | 1/1 | 0.001 | 40.000 | 0.014 | 0/0 |
| 2h | 2/7 | 0/7 | 0.000 | 0.000 | 0.000 | 0/7 |

## Paired outside voxel and volume summaries at +20 mm

Each statistic has adjacent voxel and mL columns. Category sums count findings and may count overlapping voxels repeatedly. Per-scan rows use GT unions.

| Category | Sum voxels | Sum mL | Mean voxels | Mean mL | Median voxels | Median mL | P95 voxels | P95 mL | Max voxels | Max mL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1b | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1c | 89.000 | 0.043 | 5.235 | 0.003 | 0.000 | 0.000 | 17.800 | 0.009 | 89.000 | 0.043 |
| 1d | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1e | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1f | 7,430.000 | 2.781 | 1,857.500 | 0.695 | 0.000 | 0.000 | 6,315.500 | 2.364 | 7,430.000 | 2.781 |
| 2a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2b | 113,318.000 | 45.382 | 2,312.612 | 0.926 | 0.000 | 0.000 | 44.600 | 0.018 | 105,848.000 | 42.640 |
| 2c | 283.000 | 0.078 | 4.717 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 283.000 | 0.078 |
| 2d | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2e | 415,913.000 | 204.351 | 37,810.273 | 18.577 | 29,796.000 | 18.485 | 113,593.500 | 49.065 | 138,533.000 | 49.577 |
| 2f | 0.000 | 0.000 | — | — | — | — | — | — | — | — |
| 2g | 40.000 | 0.014 | 40.000 | 0.014 | 40.000 | 0.014 | 40.000 | 0.014 | 40.000 | 0.014 |
| 2h | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Distributions

All 381 findings enter these geometric distributions, including PP2-bypassed findings and zeros. The paired plots show voxel counts left and volume in mL right, with histograms above ECDFs.

| Category | Paired voxels and volume | Percentage |
| --- | --- | --- |
| 1a | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1a_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1a_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1a_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1a_outside_percentage.pdf) |
| 1b | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1b_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1b_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1b_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1b_outside_percentage.pdf) |
| 1c | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1c_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1c_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1c_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1c_outside_percentage.pdf) |
| 1d | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1d_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1d_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1d_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1d_outside_percentage.pdf) |
| 1e | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1e_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1e_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1e_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1e_outside_percentage.pdf) |
| 1f | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1f_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1f_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1f_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/1f_outside_percentage.pdf) |
| 2a | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2a_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2a_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2a_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2a_outside_percentage.pdf) |
| 2b | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2b_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2b_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2b_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2b_outside_percentage.pdf) |
| 2c | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2c_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2c_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2c_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2c_outside_percentage.pdf) |
| 2d | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2d_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2d_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2d_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2d_outside_percentage.pdf) |
| 2e | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2e_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2e_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2e_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2e_outside_percentage.pdf) |
| 2f | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2f_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2f_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2f_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2f_outside_percentage.pdf) |
| 2g | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2g_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2g_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2g_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2g_outside_percentage.pdf) |
| 2h | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2h_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2h_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2h_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/2h_outside_percentage.pdf) |

Combined overviews: [paired page 1](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/overview_voxels_volume_1.png), [paired page 2](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/overview_voxels_volume_2.png), [percentage page 1](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/overview_percentage_1.png), [percentage page 2](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/lung20mm/figures/overview_percentage_2.png).

## Verification and detailed outputs

All 200 source-mask/GT pairs were rehashed; CT headers, Exp022 records, and fixed cohort metadata match the zero-margin analysis. Per-finding +20 mm counts reuse Exp022’s exact whole-lung coverage with integer-recovery validation. The actual method-2 EDT helper independently recomputed **23 scans / 62 findings**, including every scan with remaining outside GT and a deterministic fully-contained sample by category. All counts match. Case unions were recomputed for these scans; elsewhere all individual findings are fully contained, so their union has zero outside voxels.

`per_finding.csv` retains 0 mm and +20 mm measurements, total GT sizes, native voxel volume, PP2 eligibility, and GT-at-risk measurements. `per_case.csv` records unions. `category_summary.csv` contains paired 0/+20 mm counts, volumes, and percentages. `remaining_outside_findings.csv` lists every finding with outside GT. Detailed records, figures, and provenance remain external.
