# 1b — Bronchiectasis — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 282 | 3.67% |
| Val | 11 | 2.89% |
| Test | 11 | 1.89% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.145875 | 4/11 | 36.36% |
| Overall best | PP2 | 0.146140 | 4/11 | 36.36% |
| Overall best | PP3 (targeted corrections) | 0.149829 | 4/11 | 36.36% |
| Category best | Raw | 0.185143 | 5/11 | 45.45% |
| Category best | PP2 | 0.187937 | 5/11 | 45.45% |
| Category best | PP3 (targeted corrections) | 0.189222 | 5/11 | 45.45% |
| Iso07 best | Raw | 0.147219 | 4/11 | 36.36% |
| Iso07 best | PP2 | 0.147314 | 4/11 | 36.36% |
| Iso07 best | PP3 (targeted corrections) | 0.155867 | 4/11 | 36.36% |

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

### 1. val047_train_13624_a_2 · finding 0

Minimal bronchiectatic enlargements in the central zones and superior segments of the lower lobes

![1b val047_train_13624_a_2 · finding 0](full_body/val047_train_13624_a_2_finding0_1b.png)

[Open full-resolution PNG](full_body/val047_train_13624_a_2_finding0_1b.png)

### 2. val061_train_3028_a_2 · finding 2

Traction bronchiectasis in the apical portion of the left upper lobe

![1b val061_train_3028_a_2 · finding 2](full_body/val061_train_3028_a_2_finding2_1b.png)

[Open full-resolution PNG](full_body/val061_train_3028_a_2_finding2_1b.png)

### 3. val089_train_2657_a_2 · finding 2

Bronchiectasis in both lungs

![1b val089_train_2657_a_2 · finding 2](full_body/val089_train_2657_a_2_finding2_1b.png)

[Open full-resolution PNG](full_body/val089_train_2657_a_2_finding2_1b.png)

### 4. val092_train_2564_c_1 · finding 0

Minimal central bronchiectasis bilaterally

![1b val092_train_2564_c_1 · finding 0](full_body/val092_train_2564_c_1_finding0_1b.png)

[Open full-resolution PNG](full_body/val092_train_2564_c_1_finding0_1b.png)

### 5. val121_train_13252_b_2 · finding 0

Bronchiectasis most prominently at the central level of the right lower lobe

![1b val121_train_13252_b_2 · finding 0](full_body/val121_train_13252_b_2_finding0_1b.png)

[Open full-resolution PNG](full_body/val121_train_13252_b_2_finding0_1b.png)

### 6. val121_train_13252_b_2 · finding 1

Bronchiectasis in both lower lobes

![1b val121_train_13252_b_2 · finding 1](full_body/val121_train_13252_b_2_finding1_1b.png)

[Open full-resolution PNG](full_body/val121_train_13252_b_2_finding1_1b.png)

### 7. val123_train_3020_a_2 · finding 1

Tubular and varicose bronchiectasis in the posterior segment of the right upper lobe

![1b val123_train_3020_a_2 · finding 1](full_body/val123_train_3020_a_2_finding1_1b.png)

[Open full-resolution PNG](full_body/val123_train_3020_a_2_finding1_1b.png)

### 8. val128_train_2594_a_2 · finding 0

Bronchiectasis with minimal peribronchial thickening in the lateral segment of the right middle lobe

![1b val128_train_2594_a_2 · finding 0](full_body/val128_train_2594_a_2_finding0_1b.png)

[Open full-resolution PNG](full_body/val128_train_2594_a_2_finding0_1b.png)

### 9. val128_train_2594_a_2 · finding 2

Mild central bronchiectasis in both lungs

![1b val128_train_2594_a_2 · finding 2](full_body/val128_train_2594_a_2_finding2_1b.png)

[Open full-resolution PNG](full_body/val128_train_2594_a_2_finding2_1b.png)

### 10. val129_train_18733_a_1 · finding 2

Cylindrical bronchiectasis adjacent to the right lower lobe consolidation

![1b val129_train_18733_a_1 · finding 2](full_body/val129_train_18733_a_1_finding2_1b.png)

[Open full-resolution PNG](full_body/val129_train_18733_a_1_finding2_1b.png)

</details>

<details>
<summary>Figures 11–11 of 11</summary>

### 11. val144_train_1841_b_1 · finding 2

Focal bronchiectasis in the right upper lobe

![1b val144_train_1841_b_1 · finding 2](full_body/val144_train_1841_b_1_finding2_1b.png)

[Open full-resolution PNG](full_body/val144_train_1841_b_1_finding2_1b.png)

</details>

