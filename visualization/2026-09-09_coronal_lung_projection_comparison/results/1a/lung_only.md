# 1a — Bronchial wall thickening — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 236 | 3.07% |
| Val | 3 | 0.79% |
| Test | 6 | 1.03% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.096852 | 1/3 | 33.33% |
| Overall best | PP2 | 0.096918 | 1/3 | 33.33% |
| Overall best | PP3 (targeted corrections) | 0.096852 | 1/3 | 33.33% |
| Category best | Raw | 0.154380 | 2/3 | 66.67% |
| Category best | PP2 | 0.154391 | 2/3 | 66.67% |
| Category best | PP3 (targeted corrections) | 0.154380 | 2/3 | 66.67% |
| Iso07 best | Raw | 0.107182 | 1/3 | 33.33% |
| Iso07 best | PP2 | 0.107614 | 1/3 | 33.33% |
| Iso07 best | PP3 (targeted corrections) | 0.107182 | 1/3 | 33.33% |

## Color definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP. Purple marks depth-disjoint FP and FN without voxelwise overlap.

| Color | Meaning |
| --- | --- |
| Green | Any voxelwise TP on the ray; takes priority |
| Red | FP only, without TP or FN |
| Blue | FN only, without TP or FP |
| Purple | Depth-disjoint FP + FN, without TP |


Lung restriction affects only the CT background. All GT/prediction voxels remain visible. PP2 follows frozen whole-lung eligibility; PP3 follows frozen side/lobe routing except the 15 approved bilateral upper-lobe corrections. Ineligible policies leave predictions unchanged.

## Figures (3)

<details>
<summary>Figures 1–3 of 3</summary>

### 1. val098_train_2617_b_2 · finding 3

Bronchial wall thickening in the segmental bronchi

![1a val098_train_2617_b_2 · finding 3](lung_only/val098_train_2617_b_2_finding3_1a.png)

[Open full-resolution PNG](lung_only/val098_train_2617_b_2_finding3_1a.png)

### 2. val111_train_2560_d_2 · finding 0

Minimal peribronchial thickening bilaterally

![1a val111_train_2560_d_2 · finding 0](lung_only/val111_train_2560_d_2_finding0_1a.png)

[Open full-resolution PNG](lung_only/val111_train_2560_d_2_finding0_1a.png)

### 3. val121_train_13252_b_2 · finding 2

Minimal thickening of the bronchial walls

![1a val121_train_13252_b_2 · finding 2](lung_only/val121_train_13252_b_2_finding2_1a.png)

[Open full-resolution PNG](lung_only/val121_train_13252_b_2_finding2_1a.png)

</details>

