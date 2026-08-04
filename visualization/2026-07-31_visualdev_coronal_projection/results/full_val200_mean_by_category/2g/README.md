# 2g — Pneumothorax

[← Results overview](../../README.md) · [Case/category index](../category_case_index.csv) · [Finding metrics](../val200_finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 18 | 0.23% |
| Validation | 1 | 0.26% |
| Test | 1 | 0.17% |

## Val200 model summary

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | 0.138794 | 1/1 | 100.00% |
| 2 | 100+100 attention | 0.050731 | 0/1 | 0.00% |
| 3 | 100+100 plain best no-DDP | 0.072083 | 0/1 | 0.00% |
| 4 | Best DDP | 0.048314 | 0/1 | 0.00% |

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


## Figures (1)

Figures are sorted by fixed validation index. Each figure shows only
category 2g findings and uses radiology coronal orientation.

<details>
<summary>Figures 1–1 of 1</summary>

### 1. val007_train_19891_a_2

![2g val007_train_19891_a_2](val007_train_19891_a_2.png)

[Open full-resolution PNG](val007_train_19891_a_2.png)

</details>
