# Full Val200 Mean Coronal Projection Results

This is the committed, browsable package for the category-organized full-val200
coronal projection comparison. It contains 342 figures covering all 381
validation findings. CT panels use the mean lung-windowed anterior-posterior
projection; GT and prediction panels use binary maximum-intensity projections.
Displayed Dice values remain three-dimensional.

Radiology orientation is fixed throughout: superior is up and patient right is
on screen left. Each case/category figure contains only that category's
findings and has at most three finding rows.

## Color Definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP, so slight AP over/under-segmentation still shows overlap.
> Purple marks rays where FN and FP both occur at different AP depths but no
> voxel overlaps.

| Color | Meaning |
| --- | --- |
| Green | Ray contains any real voxelwise TP, `gt & pred` |
| Red | No TP; ray contains FP only |
| Blue | No TP; ray contains FN only |
| Purple | No TP; ray contains depth-disjoint FN+FP |


## Dataset category distribution

Counts and percentages are finding-level within each split. Case totals are
included separately because a scan can have multiple findings.

| Split | Cases | Findings |
| --- | --- | --- |
| Train | 2992 | 7687 |
| Validation | 200 | 381 |
| Test | 300 | 582 |

| Category | Finding type | Train n (%) | Validation n (%) | Test n (%) |
| --- | --- | --- | --- | --- |
| 1a | Bronchial wall thickening | 236 (3.07%) | 3 (0.79%) | 6 (1.03%) |
| 1b | Bronchiectasis | 282 (3.67%) | 11 (2.89%) | 11 (1.89%) |
| 1c | Emphysema (including Centrilobular, Paraseptal, Bullous) | 446 (5.80%) | 17 (4.46%) | 27 (4.64%) |
| 1d | Septal thickening (including Interlobular, Reticulation) | 194 (2.52%) | 6 (1.57%) | 9 (1.55%) |
| 1e | Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 314 (4.08%) | 11 (2.89%) | 16 (2.75%) |
| 1f | Other | 150 (1.95%) | 4 (1.05%) | 4 (0.69%) |
| 2a | Linear (including subsegmental atelectasis, scarring, fibrosis) | 1120 (14.57%) | 69 (18.11%) | 113 (19.42%) |
| 2b | Atelectasis, consolidation | 1367 (17.78%) | 49 (12.86%) | 89 (15.29%) |
| 2c | Groundglass opacity | 1507 (19.60%) | 60 (15.75%) | 87 (14.95%) |
| 2d | Pulmonary nodules/masses | 1743 (22.67%) | 132 (34.65%) | 190 (32.65%) |
| 2e | Pleural effusion or thickening | 237 (3.08%) | 11 (2.89%) | 27 (4.64%) |
| 2f | Honeycombing | 16 (0.21%) | 0 (0.00%) | 0 (0.00%) |
| 2g | Pneumothorax | 18 (0.23%) | 1 (0.26%) | 1 (0.17%) |
| 2h | Other | 57 (0.74%) | 7 (1.84%) | 2 (0.34%) |

Sources: [split category audit](../../../side_experiments/sideexp001_validation_probe_design/outputs/split_category_audit.csv)
and [analysis summary](../../../side_experiments/sideexp001_validation_probe_design/outputs/analysis_summary.json).
The released test metadata includes finding categories, but test masks are
hidden; therefore the tables below report measured model performance on the
200-case validation set only.

## Four-method val200 comparison

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Training/checkpoint description | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | Original pretrained model; no project epoch | 0.225228 | 204/381 | 53.54% |
| 2 | 100+100 attention | Exp009 s3v1_fixedrho_suppress_half_quarter; Exp006 e5d4 epoch 100 + continuation epoch 100 (10,000 updates) | 0.334709 | 285/381 | 74.80% |
| 3 | 100+100 plain best no-DDP | Exp009 baseline_cont100; Exp006 e5d4 epoch 100 + plain continuation epoch 100 (10,000 updates) | 0.339839 | 288/381 | 75.59% |
| 4 | Best DDP | Exp007 weights-only DDP continuation relative epoch 50 / absolute epoch 150 (5,000 continuation updates) | 0.346023 | 296/381 | 77.69% |

Machine-readable table: [four_model_overall_metrics.csv](full_val200_mean_by_category/four_model_overall_metrics.csv).

## Per-category val200 comparison

Each model cell is `mean Dice · hits/support (hit rate)`. Category 2f has no
validation findings, so measured performance is not available.

| Category | Findings | Public VoxTell v1.1 | 100+100 attention | 100+100 plain best no-DDP | Best DDP |
| --- | --- | --- | --- | --- | --- |
| 1a | 3 | 0.073088 · 1/3 (33.33%) | 0.118227 · 1/3 (33.33%) | 0.115071 · 1/3 (33.33%) | 0.096852 · 1/3 (33.33%) |
| 1b | 11 | 0.072314 · 2/11 (18.18%) | 0.152879 · 4/11 (36.36%) | 0.157151 · 4/11 (36.36%) | 0.145875 · 4/11 (36.36%) |
| 1c | 17 | 0.063847 · 4/17 (23.53%) | 0.173971 · 7/17 (41.18%) | 0.177998 · 6/17 (35.29%) | 0.156913 · 8/17 (47.06%) |
| 1d | 6 | 0.125096 · 1/6 (16.67%) | 0.126608 · 1/6 (16.67%) | 0.125360 · 1/6 (16.67%) | 0.116696 · 1/6 (16.67%) |
| 1e | 11 | 0.101601 · 3/11 (27.27%) | 0.178167 · 7/11 (63.64%) | 0.178663 · 6/11 (54.55%) | 0.194224 · 7/11 (63.64%) |
| 1f | 4 | 0.002137 · 0/4 (0.00%) | 0.158789 · 1/4 (25.00%) | 0.161189 · 1/4 (25.00%) | 0.212339 · 2/4 (50.00%) |
| 2a | 69 | 0.197082 · 38/69 (55.07%) | 0.333137 · 56/69 (81.16%) | 0.340520 · 57/69 (82.61%) | 0.337641 · 59/69 (85.51%) |
| 2b | 49 | 0.297481 · 33/49 (67.35%) | 0.362407 · 37/49 (75.51%) | 0.356849 · 37/49 (75.51%) | 0.357690 · 38/49 (77.55%) |
| 2c | 60 | 0.309013 · 42/60 (70.00%) | 0.387795 · 49/60 (81.67%) | 0.390826 · 50/60 (83.33%) | 0.414803 · 49/60 (81.67%) |
| 2d | 132 | 0.226553 · 70/132 (53.03%) | 0.367206 · 110/132 (83.33%) | 0.378417 · 113/132 (85.61%) | 0.394936 · 115/132 (87.12%) |
| 2e | 11 | 0.431012 · 8/11 (72.73%) | 0.449102 · 8/11 (72.73%) | 0.448142 · 8/11 (72.73%) | 0.412117 · 8/11 (72.73%) |
| 2f | 0 | — | — | — | — |
| 2g | 1 | 0.138794 · 1/1 (100.00%) | 0.050731 · 0/1 (0.00%) | 0.072083 · 0/1 (0.00%) | 0.048314 · 0/1 (0.00%) |
| 2h | 7 | 0.047717 · 1/7 (14.29%) | 0.243102 · 4/7 (57.14%) | 0.233286 · 4/7 (57.14%) | 0.165837 · 4/7 (57.14%) |

Machine-readable table: [four_model_category_metrics.csv](full_val200_mean_by_category/four_model_category_metrics.csv).

## Category galleries

| Category gallery | Findings | Figures |
| --- | --- | --- |
| [1a — Bronchial wall thickening](full_val200_mean_by_category/1a/README.md) | 3 | 3 |
| [1b — Bronchiectasis](full_val200_mean_by_category/1b/README.md) | 11 | 9 |
| [1c — Emphysema (including Centrilobular, Paraseptal, Bullous)](full_val200_mean_by_category/1c/README.md) | 17 | 15 |
| [1d — Septal thickening (including Interlobular, Reticulation)](full_val200_mean_by_category/1d/README.md) | 6 | 6 |
| [1e — Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic)](full_val200_mean_by_category/1e/README.md) | 11 | 11 |
| [1f — Other](full_val200_mean_by_category/1f/README.md) | 4 | 4 |
| [2a — Linear (including subsegmental atelectasis, scarring, fibrosis)](full_val200_mean_by_category/2a/README.md) | 69 | 63 |
| [2b — Atelectasis, consolidation](full_val200_mean_by_category/2b/README.md) | 49 | 40 |
| [2c — Groundglass opacity](full_val200_mean_by_category/2c/README.md) | 60 | 53 |
| [2d — Pulmonary nodules/masses](full_val200_mean_by_category/2d/README.md) | 132 | 119 |
| [2e — Pleural effusion or thickening](full_val200_mean_by_category/2e/README.md) | 11 | 11 |
| [2f — Honeycombing](full_val200_mean_by_category/2f_empty/README.md) | 0 | 0 |
| [2g — Pneumothorax](full_val200_mean_by_category/2g/README.md) | 1 | 1 |
| [2h — Other](full_val200_mean_by_category/2h/README.md) | 7 | 7 |

## Canonical data and provenance

- [Finding-level metrics](full_val200_mean_by_category/val200_finding_metrics.csv)
- [Case/category figure index](full_val200_mean_by_category/category_case_index.csv)
- [Renderer run manifest](full_val200_mean_by_category/run_manifest.json)
- [Visualization development README](../README.md)

The source gallery remains at:

- Host: `/data/hengjie/datasets/rexgroundingct/visualizations/2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category`
- Docker: `/database/datasets/rexgroundingct/visualizations/2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category`

Rebuild and verify this committed package from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/package_results.py
```
