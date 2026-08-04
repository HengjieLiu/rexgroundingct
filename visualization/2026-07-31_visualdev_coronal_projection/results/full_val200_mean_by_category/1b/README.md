# 1b — Bronchiectasis

[← Results overview](../../README.md) · [Case/category index](../category_case_index.csv) · [Finding metrics](../val200_finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 282 | 3.67% |
| Validation | 11 | 2.89% |
| Test | 11 | 1.89% |

## Val200 model summary

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | 0.072314 | 2/11 | 18.18% |
| 2 | 100+100 attention | 0.152879 | 4/11 | 36.36% |
| 3 | 100+100 plain best no-DDP | 0.157151 | 4/11 | 36.36% |
| 4 | Best DDP | 0.145875 | 4/11 | 36.36% |

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


## Figures (9)

Figures are sorted by fixed validation index. Each figure shows only
category 1b findings and uses radiology coronal orientation.

<details>
<summary>Figures 1–9 of 9</summary>

### 1. val047_train_13624_a_2

![1b val047_train_13624_a_2](val047_train_13624_a_2.png)

[Open full-resolution PNG](val047_train_13624_a_2.png)

### 2. val061_train_3028_a_2

![1b val061_train_3028_a_2](val061_train_3028_a_2.png)

[Open full-resolution PNG](val061_train_3028_a_2.png)

### 3. val089_train_2657_a_2

![1b val089_train_2657_a_2](val089_train_2657_a_2.png)

[Open full-resolution PNG](val089_train_2657_a_2.png)

### 4. val092_train_2564_c_1

![1b val092_train_2564_c_1](val092_train_2564_c_1.png)

[Open full-resolution PNG](val092_train_2564_c_1.png)

### 5. val121_train_13252_b_2

![1b val121_train_13252_b_2](val121_train_13252_b_2.png)

[Open full-resolution PNG](val121_train_13252_b_2.png)

### 6. val123_train_3020_a_2

![1b val123_train_3020_a_2](val123_train_3020_a_2.png)

[Open full-resolution PNG](val123_train_3020_a_2.png)

### 7. val128_train_2594_a_2

![1b val128_train_2594_a_2](val128_train_2594_a_2.png)

[Open full-resolution PNG](val128_train_2594_a_2.png)

### 8. val129_train_18733_a_1

![1b val129_train_18733_a_1](val129_train_18733_a_1.png)

[Open full-resolution PNG](val129_train_18733_a_1.png)

### 9. val144_train_1841_b_1

![1b val144_train_1841_b_1](val144_train_1841_b_1.png)

[Open full-resolution PNG](val144_train_1841_b_1.png)

</details>
