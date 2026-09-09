# 2e — Pleural effusion or thickening — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 237 | 3.08% |
| Val | 11 | 2.89% |
| Test | 27 | 4.64% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.412117 | 8/11 | 72.73% |
| Overall best | PP2 | 0.412117 | 8/11 | 72.73% |
| Overall best | PP3 (targeted corrections) | 0.412117 | 8/11 | 72.73% |
| Category best | Raw | 0.483070 | 9/11 | 81.82% |
| Category best | PP2 | 0.483070 | 9/11 | 81.82% |
| Category best | PP3 (targeted corrections) | 0.483070 | 9/11 | 81.82% |
| Iso07 best | Raw | 0.407828 | 8/11 | 72.73% |
| Iso07 best | PP2 | 0.407828 | 8/11 | 72.73% |
| Iso07 best | PP3 (targeted corrections) | 0.407828 | 8/11 | 72.73% |

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

## Figures (11)

<details>
<summary>Figures 1–10 of 11</summary>

### 1. val007_train_19891_a_2 · finding 1

Bilateral pleural effusions, greater on the right; right effusion measures approximately 60 mm; small left pleural effusion

![2e val007_train_19891_a_2 · finding 1](lung_only/val007_train_19891_a_2_finding1_2e.png)

[Open full-resolution PNG](lung_only/val007_train_19891_a_2_finding1_2e.png)

### 2. val029_train_2643_a_2 · finding 0

Bilateral pleural effusions, right up to 20 mm and left up to 12 mm at greatest depth

![2e val029_train_2643_a_2 · finding 0](lung_only/val029_train_2643_a_2_finding0_2e.png)

[Open full-resolution PNG](lung_only/val029_train_2643_a_2_finding0_2e.png)

### 3. val051_train_13292_a_2 · finding 0

Bilateral pleural effusion, more prominent on the left

![2e val051_train_13292_a_2 · finding 0](lung_only/val051_train_13292_a_2_finding0_2e.png)

[Open full-resolution PNG](lung_only/val051_train_13292_a_2_finding0_2e.png)

### 4. val069_train_18392_a_1 · finding 1

Focal thickening of the right major fissure

![2e val069_train_18392_a_1 · finding 1](lung_only/val069_train_18392_a_1_finding1_2e.png)

[Open full-resolution PNG](lung_only/val069_train_18392_a_1_finding1_2e.png)

### 5. val096_train_2148_a_2 · finding 0

Band-like pleural calcifications parallel to the pleural surface along the superior segment and posterobasal segment of the right lower lobe

![2e val096_train_2148_a_2 · finding 0](lung_only/val096_train_2148_a_2_finding0_2e.png)

[Open full-resolution PNG](lung_only/val096_train_2148_a_2_finding0_2e.png)

### 6. val107_train_19325_a_2 · finding 0

Left pleural effusion up to 10 mm in thickness at the lower lobe level

![2e val107_train_19325_a_2 · finding 0](lung_only/val107_train_19325_a_2_finding0_2e.png)

[Open full-resolution PNG](lung_only/val107_train_19325_a_2_finding0_2e.png)

### 7. val112_train_2555_a_1 · finding 0

Nonspecific subcentimeter pleural thickening along the posterobasal segment of the right lower lobe

![2e val112_train_2555_a_1 · finding 0](lung_only/val112_train_2555_a_1_finding0_2e.png)

[Open full-resolution PNG](lung_only/val112_train_2555_a_1_finding0_2e.png)

### 8. val116_train_13119_c_1 · finding 3

Minimal bilateral pleural effusion

![2e val116_train_13119_c_1 · finding 3](lung_only/val116_train_13119_c_1_finding3_2e.png)

[Open full-resolution PNG](lung_only/val116_train_13119_c_1_finding3_2e.png)

### 9. val151_train_13583_d_2 · finding 0

Progressive pleural effusion reaching a diameter of 3 cm between the leaves of the right pleura

![2e val151_train_13583_d_2 · finding 0](lung_only/val151_train_13583_d_2_finding0_2e.png)

[Open full-resolution PNG](lung_only/val151_train_13583_d_2_finding0_2e.png)

### 10. val168_train_18399_b_2 · finding 1

Left pleural effusion measuring up to 65 mm in maximal depth in the dependent portion and extending to the apex in the supine position

![2e val168_train_18399_b_2 · finding 1](lung_only/val168_train_18399_b_2_finding1_2e.png)

[Open full-resolution PNG](lung_only/val168_train_18399_b_2_finding1_2e.png)

</details>

<details>
<summary>Figures 11–11 of 11</summary>

### 11. val183_train_13013_a_1 · finding 0

Bilateral pleural effusion more prominent on the right, measuring 60 mm at its thickest point

![2e val183_train_13013_a_1 · finding 0](lung_only/val183_train_13013_a_1_finding0_2e.png)

[Open full-resolution PNG](lung_only/val183_train_13013_a_1_finding0_2e.png)

</details>

