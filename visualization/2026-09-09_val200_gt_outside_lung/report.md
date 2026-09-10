# Val200 GT outside the downloaded whole-lung mask

**271/381 findings** have at least one GT voxel outside lung, across 200 scans.

Exact CT-RATE `ts_total` labels 10–14; native GT > 0; no dilation or resampling. M/N counts findings, not separate lesion IDs. Volumes use each CT affine determinant and are displayed in mL. The source remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.

| Category | M/N | Affected % | Outside % mean | Median | P95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1a | 3/3 | 100.000 | 7.195 | 7.761 | 12.783 | 13.341 |
| 1b | 5/11 | 45.455 | 5.666 | 0.000 | 26.313 | 37.906 |
| 1c | 14/17 | 82.353 | 6.145 | 4.427 | 13.924 | 26.496 |
| 1d | 6/6 | 100.000 | 1.433 | 0.884 | 3.345 | 3.560 |
| 1e | 3/11 | 27.273 | 4.939 | 0.000 | 27.026 | 46.601 |
| 1f | 2/4 | 50.000 | 49.672 | 49.345 | 99.803 | 100.000 |
| 2a | 66/69 | 95.652 | 16.811 | 9.986 | 49.489 | 70.295 |
| 2b | 48/49 | 97.959 | 20.041 | 12.992 | 63.886 | 78.611 |
| 2c | 52/60 | 86.667 | 5.305 | 1.986 | 22.053 | 64.060 |
| 2d | 58/132 | 43.939 | 4.495 | 0.000 | 24.296 | 93.881 |
| 2e | 11/11 | 100.000 | 46.584 | 45.730 | 78.426 | 85.073 |
| 2f | 0/0 — no findings | — | — | — | — | — |
| 2g | 1/1 | 100.000 | 5.476 | 5.476 | 5.476 | 5.476 |
| 2h | 2/7 | 28.571 | 7.396 | 0.000 | 35.361 | 49.574 |

## Outside voxel counts and volumes

Each statistic is paired as **voxels / mL**. Sums are across findings and may count overlapping voxels repeatedly. Per-case CSV measurements use the GT union instead.

| Category | Sum voxels | Sum mL | Mean voxels | Mean mL | Median voxels | Median mL | P95 voxels | P95 mL | Max voxels | Max mL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | 4,868.000 | 2.176 | 1,622.667 | 0.725 | 243.000 | 0.131 | 4,169.700 | 1.845 | 4,606.000 | 2.035 |
| 1b | 20,801.000 | 10.521 | 1,891.000 | 0.956 | 0.000 | 0.000 | 7,283.000 | 3.611 | 8,367.000 | 3.932 |
| 1c | 54,789.000 | 29.386 | 3,222.882 | 1.729 | 823.000 | 0.318 | 13,910.200 | 6.895 | 17,483.000 | 9.558 |
| 1d | 25,092.000 | 13.532 | 4,182.000 | 2.255 | 1,385.500 | 0.935 | 14,481.750 | 7.680 | 17,939.000 | 9.578 |
| 1e | 733.000 | 0.569 | 66.636 | 0.052 | 0.000 | 0.000 | 362.500 | 0.283 | 706.000 | 0.559 |
| 1f | 7,745.000 | 2.993 | 1,936.250 | 0.748 | 113.000 | 0.089 | 6,425.050 | 2.419 | 7,519.000 | 2.814 |
| 2a | 137,149.000 | 70.354 | 1,987.667 | 1.020 | 681.000 | 0.394 | 7,261.200 | 4.867 | 22,100.000 | 9.617 |
| 2b | 1,209,975.000 | 506.314 | 24,693.367 | 10.333 | 3,574.000 | 2.233 | 63,758.800 | 28.038 | 456,438.000 | 183.872 |
| 2c | 507,911.000 | 219.173 | 8,465.183 | 3.653 | 980.500 | 0.383 | 49,476.400 | 15.675 | 224,994.000 | 62.261 |
| 2d | 6,728.000 | 2.758 | 50.970 | 0.021 | 0.000 | 0.000 | 139.450 | 0.059 | 2,961.000 | 1.078 |
| 2e | 2,906,479.000 | 1,326.577 | 264,225.364 | 120.598 | 235,886.000 | 99.312 | 828,082.000 | 339.050 | 1,232,767.000 | 432.054 |
| 2f | 0.000 | 0.000 | — | — | — | — | — | — | — | — |
| 2g | 177,966.000 | 62.373 | 177,966.000 | 62.373 | 177,966.000 | 62.373 | 177,966.000 | 62.373 | 177,966.000 | 62.373 |
| 2h | 4,622.000 | 2.055 | 660.286 | 0.294 | 0.000 | 0.000 | 3,130.200 | 1.381 | 4,359.000 | 1.911 |

## Distribution figures

| Category | Outside voxels and volume, side by side | Outside percentage |
| --- | --- | --- |
| 1a | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1a_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1a_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1a_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1a_outside_percentage.pdf) |
| 1b | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1b_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1b_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1b_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1b_outside_percentage.pdf) |
| 1c | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1c_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1c_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1c_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1c_outside_percentage.pdf) |
| 1d | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1d_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1d_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1d_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1d_outside_percentage.pdf) |
| 1e | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1e_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1e_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1e_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1e_outside_percentage.pdf) |
| 1f | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1f_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1f_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1f_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/1f_outside_percentage.pdf) |
| 2a | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2a_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2a_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2a_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2a_outside_percentage.pdf) |
| 2b | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2b_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2b_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2b_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2b_outside_percentage.pdf) |
| 2c | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2c_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2c_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2c_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2c_outside_percentage.pdf) |
| 2d | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2d_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2d_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2d_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2d_outside_percentage.pdf) |
| 2e | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2e_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2e_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2e_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2e_outside_percentage.pdf) |
| 2f | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2f_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2f_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2f_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2f_outside_percentage.pdf) |
| 2g | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2g_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2g_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2g_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2g_outside_percentage.pdf) |
| 2h | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2h_outside_voxels_volume.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2h_outside_voxels_volume.pdf) | [PNG](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2h_outside_percentage.png) / [PDF](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/2h_outside_percentage.pdf) |

Combined overviews: [voxels/volume 1](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/overview_voxels_volume_1.png), [voxels/volume 2](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/overview_voxels_volume_2.png), [percentage 1](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/overview_percentage_1.png), [percentage 2](/mnt/shengdata1/hengjie/experiments/rexgroundingct/visualizations/2026-09-09_val200_gt_outside_lung/figures/overview_percentage_2.png).

## Verification and artifacts

All 200 native grids and 381 per-finding voxel counts were checked against Exp020/Exp022. CT spatial units are mm. Source-mask and GT hashes match the prior audit; CT intensity payloads were not rehashed. Empty GT findings: 0.

External outputs: `per_finding.csv`, `per_case.csv`, `category_summary.csv`, `measurements.json`, `run_manifest.json`, and `figures/` (PNG/PDF). Count/volume plots pair voxels left and mL right; percentage plots are separate. All valid findings, including zeros, enter the distributions.
