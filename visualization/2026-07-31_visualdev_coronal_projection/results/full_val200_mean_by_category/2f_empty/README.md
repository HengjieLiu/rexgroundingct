# 2f — Honeycombing

[← Results overview](../../README.md) · [Case/category index](../category_case_index.csv) · [Finding metrics](../val200_finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 16 | 0.21% |
| Validation | 0 | 0.00% |
| Test | 0 | 0.00% |

## Val200 model summary

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | — | — | — |
| 2 | 100+100 attention | — | — | — |
| 3 | 100+100 plain best no-DDP | — | — | — |
| 4 | Best DDP | — | — | — |

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


## Figures (0)

There are no category-2f findings in validation or test, so this
gallery intentionally contains zero PNGs. The training split has 16
category-2f findings.
