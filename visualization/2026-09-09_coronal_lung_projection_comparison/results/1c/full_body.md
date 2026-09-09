# 1c — Emphysema (including Centrilobular, Paraseptal, Bullous) — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 446 | 5.80% |
| Val | 17 | 4.46% |
| Test | 27 | 4.64% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.156913 | 8/17 | 47.06% |
| Overall best | PP2 | 0.170336 | 8/17 | 47.06% |
| Overall best | PP3 (targeted corrections) | 0.169725 | 8/17 | 47.06% |
| Category best | Raw | 0.207804 | 9/17 | 52.94% |
| Category best | PP2 | 0.221946 | 9/17 | 52.94% |
| Category best | PP3 (targeted corrections) | 0.221820 | 9/17 | 52.94% |
| Iso07 best | Raw | 0.184289 | 8/17 | 47.06% |
| Iso07 best | PP2 | 0.186129 | 8/17 | 47.06% |
| Iso07 best | PP3 (targeted corrections) | 0.186000 | 8/17 | 47.06% |

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

## Figures (17)

<details>
<summary>Figures 1–10 of 17</summary>

### 1. val001_train_3029_a_1 · finding 0

Mild paraseptal emphysema at the apices of both upper lobes

![1c val001_train_3029_a_1 · finding 0](full_body/val001_train_3029_a_1_finding0_1c.png)

[Open full-resolution PNG](full_body/val001_train_3029_a_1_finding0_1c.png)

### 2. val001_train_3029_a_1 · finding 1

Mild centrilobular emphysema at the apices of both upper lobes

![1c val001_train_3029_a_1 · finding 1](full_body/val001_train_3029_a_1_finding1_1c.png)

[Open full-resolution PNG](full_body/val001_train_3029_a_1_finding1_1c.png)

### 3. val005_train_2675_a_1 · finding 1

Minimal emphysematous changes in both lungs

![1c val005_train_2675_a_1 · finding 1](full_body/val005_train_2675_a_1_finding1_1c.png)

[Open full-resolution PNG](full_body/val005_train_2675_a_1_finding1_1c.png)

### 4. val006_train_3011_a_2 · finding 1

Minimal paraseptal emphysema in the apices of both lungs

![1c val006_train_3011_a_2 · finding 1](full_body/val006_train_3011_a_2_finding1_1c.png)

[Open full-resolution PNG](full_body/val006_train_3011_a_2_finding1_1c.png)

### 5. val041_train_18969_a_2 · finding 0

Bulla in the anterior segment of the right upper lobe

![1c val041_train_18969_a_2 · finding 0](full_body/val041_train_18969_a_2_finding0_1c.png)

[Open full-resolution PNG](full_body/val041_train_18969_a_2_finding0_1c.png)

### 6. val041_train_18969_a_2 · finding 2

Bulla in the anterior right middle lobe

![1c val041_train_18969_a_2 · finding 2](full_body/val041_train_18969_a_2_finding2_1c.png)

[Open full-resolution PNG](full_body/val041_train_18969_a_2_finding2_1c.png)

### 7. val071_train_3008_a_1 · finding 2

Mild emphysematous changes in both lungs

![1c val071_train_3008_a_1 · finding 2](full_body/val071_train_3008_a_1_finding2_1c.png)

[Open full-resolution PNG](full_body/val071_train_3008_a_1_finding2_1c.png)

### 8. val079_train_2508_a_2 · finding 1

Paraseptal emphysema in the bilateral upper lobes

![1c val079_train_2508_a_2 · finding 1](full_body/val079_train_2508_a_2_finding1_1c.png)

[Open full-resolution PNG](full_body/val079_train_2508_a_2_finding1_1c.png)

### 9. val091_train_3008_b_2 · finding 1

Minimal emphysema in the upper lobes

![1c val091_train_3008_b_2 · finding 1](full_body/val091_train_3008_b_2_finding1_1c.png)

[Open full-resolution PNG](full_body/val091_train_3008_b_2_finding1_1c.png)

### 10. val111_train_2560_d_2 · finding 1

Minimal emphysematous changes in both lungs

![1c val111_train_2560_d_2 · finding 1](full_body/val111_train_2560_d_2_finding1_1c.png)

[Open full-resolution PNG](full_body/val111_train_2560_d_2_finding1_1c.png)

</details>

<details>
<summary>Figures 11–17 of 17</summary>

### 11. val114_train_13155_a_2 · finding 0

More prominent emphysematous changes in the upper lobes of both lungs

![1c val114_train_13155_a_2 · finding 0](full_body/val114_train_13155_a_2_finding0_1c.png)

[Open full-resolution PNG](full_body/val114_train_13155_a_2_finding0_1c.png)

### 12. val128_train_2594_a_2 · finding 4

Minimal emphysematous changes in both lungs

![1c val128_train_2594_a_2 · finding 4](full_body/val128_train_2594_a_2_finding4_1c.png)

[Open full-resolution PNG](full_body/val128_train_2594_a_2_finding4_1c.png)

### 13. val159_train_13631_a_1 · finding 1

Emphysema in both lungs

![1c val159_train_13631_a_1 · finding 1](full_body/val159_train_13631_a_1_finding1_1c.png)

[Open full-resolution PNG](full_body/val159_train_13631_a_1_finding1_1c.png)

### 14. val173_train_13189_a_1 · finding 0

Hyperaeration in the mediobasal and posterobasal segments of the right lower lobe

![1c val173_train_13189_a_1 · finding 0](full_body/val173_train_13189_a_1_finding0_1c.png)

[Open full-resolution PNG](full_body/val173_train_13189_a_1_finding0_1c.png)

### 15. val185_train_3006_a_2 · finding 0

Mild emphysematous changes in both lungs

![1c val185_train_3006_a_2 · finding 0](full_body/val185_train_3006_a_2_finding0_1c.png)

[Open full-resolution PNG](full_body/val185_train_3006_a_2_finding0_1c.png)

### 16. val189_train_3026_a_2 · finding 0

Emphysematous changes in the upper lobes of both lungs

![1c val189_train_3026_a_2 · finding 0](full_body/val189_train_3026_a_2_finding0_1c.png)

[Open full-resolution PNG](full_body/val189_train_3026_a_2_finding0_1c.png)

### 17. val198_train_3026_b_2 · finding 0

Emphysematous changes in both lungs, more pronounced in the upper lobes

![1c val198_train_3026_b_2 · finding 0](full_body/val198_train_3026_b_2_finding0_1c.png)

[Open full-resolution PNG](full_body/val198_train_3026_b_2_finding0_1c.png)

</details>

