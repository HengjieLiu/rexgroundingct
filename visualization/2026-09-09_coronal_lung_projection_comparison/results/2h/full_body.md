# 2h — Other — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 57 | 0.74% |
| Val | 7 | 1.84% |
| Test | 2 | 0.34% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.165837 | 4/7 | 57.14% |
| Overall best | PP2 | 0.239000 | 5/7 | 71.43% |
| Overall best | PP3 (targeted corrections) | 0.239000 | 5/7 | 71.43% |
| Category best | Raw | 0.371772 | 4/7 | 57.14% |
| Category best | PP2 | 0.416106 | 5/7 | 71.43% |
| Category best | PP3 (targeted corrections) | 0.416106 | 5/7 | 71.43% |
| Iso07 best | Raw | 0.302565 | 4/7 | 57.14% |
| Iso07 best | PP2 | 0.303976 | 4/7 | 57.14% |
| Iso07 best | PP3 (targeted corrections) | 0.303976 | 4/7 | 57.14% |

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

## Figures (7)

<details>
<summary>Figures 1–7 of 7</summary>

### 1. val019_train_13102_a_1 · finding 1

Fissural air cyst in the superior segment of the lower lobe of the left lung

![2h val019_train_13102_a_1 · finding 1](full_body/val019_train_13102_a_1_finding1_2h.png)

[Open full-resolution PNG](full_body/val019_train_13102_a_1_finding1_2h.png)

### 2. val035_train_13076_a_1 · finding 0

Pleural irregularities in the upper lobes of both lungs

![2h val035_train_13076_a_1 · finding 0](full_body/val035_train_13076_a_1_finding0_2h.png)

[Open full-resolution PNG](full_body/val035_train_13076_a_1_finding0_2h.png)

### 3. val053_train_18641_a_2 · finding 3

8 mm well-circumscribed thin-walled air cyst in the posterior segment of the right upper lobe

![2h val053_train_18641_a_2 · finding 3](full_body/val053_train_18641_a_2_finding3_2h.png)

[Open full-resolution PNG](full_body/val053_train_18641_a_2_finding3_2h.png)

### 4. val058_train_13082_a_1 · finding 0

Air cyst at the anterobasal level of the right lower lobe

![2h val058_train_13082_a_1 · finding 0](full_body/val058_train_13082_a_1_finding0_2h.png)

[Open full-resolution PNG](full_body/val058_train_13082_a_1_finding0_2h.png)

### 5. val098_train_2617_b_2 · finding 1

Air cyst in the posterobasal segment of the left lower lobe

![2h val098_train_2617_b_2 · finding 1](full_body/val098_train_2617_b_2_finding1_2h.png)

[Open full-resolution PNG](full_body/val098_train_2617_b_2_finding1_2h.png)

### 6. val105_train_19032_a_2 · finding 0

Peribronchial air cyst in the left lower lobe, 11 mm

![2h val105_train_19032_a_2 · finding 0](full_body/val105_train_19032_a_2_finding0_2h.png)

[Open full-resolution PNG](full_body/val105_train_19032_a_2_finding0_2h.png)

### 7. val198_train_3026_b_2 · finding 1

Architectural distortion in the apical segments of both lungs

![2h val198_train_3026_b_2 · finding 1](full_body/val198_train_3026_b_2_finding1_2h.png)

[Open full-resolution PNG](full_body/val198_train_3026_b_2_finding1_2h.png)

</details>

