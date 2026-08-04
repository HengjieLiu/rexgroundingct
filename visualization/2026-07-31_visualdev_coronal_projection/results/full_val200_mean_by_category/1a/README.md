# 1a — Bronchial wall thickening

[← Results overview](../../README.md) · [Case/category index](../category_case_index.csv) · [Finding metrics](../val200_finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 236 | 3.07% |
| Validation | 3 | 0.79% |
| Test | 6 | 1.03% |

## Val200 model summary

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | 0.073088 | 1/3 | 33.33% |
| 2 | 100+100 attention | 0.118227 | 1/3 | 33.33% |
| 3 | 100+100 plain best no-DDP | 0.115071 | 1/3 | 33.33% |
| 4 | Best DDP | 0.096852 | 1/3 | 33.33% |

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


## Figures (3)

Figures are sorted by fixed validation index. Each figure shows only
category 1a findings and uses radiology coronal orientation.

<details>
<summary>Figures 1–3 of 3</summary>

### 1. val098_train_2617_b_2

![1a val098_train_2617_b_2](val098_train_2617_b_2.png)

[Open full-resolution PNG](val098_train_2617_b_2.png)

### 2. val111_train_2560_d_2

![1a val111_train_2560_d_2](val111_train_2560_d_2.png)

[Open full-resolution PNG](val111_train_2560_d_2.png)

### 3. val121_train_13252_b_2

![1a val121_train_13252_b_2](val121_train_13252_b_2.png)

[Open full-resolution PNG](val121_train_13252_b_2.png)

</details>
