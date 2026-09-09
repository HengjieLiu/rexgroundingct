# 2d — Pulmonary nodules/masses — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 1743 | 22.67% |
| Val | 132 | 34.65% |
| Test | 190 | 32.65% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.394936 | 115/132 | 87.12% |
| Overall best | PP2 | 0.397661 | 115/132 | 87.12% |
| Overall best | PP3 (targeted corrections) | 0.407206 | 115/132 | 87.12% |
| Category best | Raw | 0.394936 | 115/132 | 87.12% |
| Category best | PP2 | 0.397661 | 115/132 | 87.12% |
| Category best | PP3 (targeted corrections) | 0.407206 | 115/132 | 87.12% |
| Iso07 best | Raw | 0.381093 | 115/132 | 87.12% |
| Iso07 best | PP2 | 0.396238 | 117/132 | 88.64% |
| Iso07 best | PP3 (targeted corrections) | 0.412087 | 116/132 | 87.88% |

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

## Figures (132)

<details>
<summary>Figures 1–10 of 132</summary>

### 1. val004_train_18411_a_2 · finding 0

Subcentimeter nodule in the lateral aspect of the left lower lobe, difficult to distinguish from a vessel on end

![2d val004_train_18411_a_2 · finding 0](lung_only/val004_train_18411_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val004_train_18411_a_2_finding0_2d.png)

### 2. val009_train_2458_a_2 · finding 0

Indeterminate subcentimeter pulmonary nodules in both lungs

![2d val009_train_2458_a_2 · finding 0](lung_only/val009_train_2458_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val009_train_2458_a_2_finding0_2d.png)

### 3. val011_train_19456_b_1 · finding 0

Bilateral subcentimeter pulmonary nodules

![2d val011_train_19456_b_1 · finding 0](lung_only/val011_train_19456_b_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val011_train_19456_b_1_finding0_2d.png)

### 4. val013_train_13020_a_2 · finding 0

6x5 mm semisolid nodule in the posterobasal segment of the right lower lobe

![2d val013_train_13020_a_2 · finding 0](lung_only/val013_train_13020_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val013_train_13020_a_2_finding0_2d.png)

### 5. val014_train_2560_b_2 · finding 0

Peripheral nodular consolidations with ground-glass halo in the anterior, lateral, and posterobasal segments of the right lower lobe, largest measuring 12 x 10 mm

![2d val014_train_2560_b_2 · finding 0](lung_only/val014_train_2560_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val014_train_2560_b_2_finding0_2d.png)

### 6. val016_train_18729_a_1 · finding 0

A few subcentimeter pulmonary nodules in both lungs, some entirely calcified

![2d val016_train_18729_a_1 · finding 0](lung_only/val016_train_18729_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val016_train_18729_a_1_finding0_2d.png)

### 7. val021_train_3015_a_1 · finding 1

Subcentimeter pulmonary nodules present in both lungs

![2d val021_train_3015_a_1 · finding 1](lung_only/val021_train_3015_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val021_train_3015_a_1_finding1_2d.png)

### 8. val022_train_13440_a_1 · finding 0

Subcentimeter nodules in both lungs

![2d val022_train_13440_a_1 · finding 0](lung_only/val022_train_13440_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val022_train_13440_a_1_finding0_2d.png)

### 9. val024_train_18382_b_2 · finding 0

Stable, nonspecific 6 mm subpleural nodule in the lateral basal segment of the left lower lobe

![2d val024_train_18382_b_2 · finding 0](lung_only/val024_train_18382_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val024_train_18382_b_2_finding0_2d.png)

### 10. val025_train_1998_a_1 · finding 0

Subcentimeter indeterminate pulmonary nodules

![2d val025_train_1998_a_1 · finding 0](lung_only/val025_train_1998_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val025_train_1998_a_1_finding0_2d.png)

</details>

<details>
<summary>Figures 11–20 of 132</summary>

### 11. val026_train_18983_a_2 · finding 0

Indeterminate 2 mm pulmonary nodule in the lateral subpleural region of the anterior segment of the right upper lobe

![2d val026_train_18983_a_2 · finding 0](lung_only/val026_train_18983_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val026_train_18983_a_2_finding0_2d.png)

### 12. val026_train_18983_a_2 · finding 1

Indeterminate 4 x 2 mm pulmonary nodule in the right middle lobe

![2d val026_train_18983_a_2 · finding 1](lung_only/val026_train_18983_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val026_train_18983_a_2_finding1_2d.png)

### 13. val027_train_19456_a_1 · finding 2

A few subcentimeter indeterminate pulmonary nodules in the right lung

![2d val027_train_19456_a_1 · finding 2](lung_only/val027_train_19456_a_1_finding2_2d.png)

[Open full-resolution PNG](lung_only/val027_train_19456_a_1_finding2_2d.png)

### 14. val031_train_18635_a_2 · finding 0

Scattered pulmonary nodules in both lungs, some calcified, measuring up to 5 mm

![2d val031_train_18635_a_2 · finding 0](lung_only/val031_train_18635_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val031_train_18635_a_2_finding0_2d.png)

### 15. val032_train_13301_a_2 · finding 0

Subcentimeter calcific nodule in the upper lobe of the right lung

![2d val032_train_13301_a_2 · finding 0](lung_only/val032_train_13301_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val032_train_13301_a_2_finding0_2d.png)

### 16. val032_train_13301_a_2 · finding 1

Subcentimeter nonspecific nodules in the upper lobe of the left lung

![2d val032_train_13301_a_2 · finding 1](lung_only/val032_train_13301_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val032_train_13301_a_2_finding1_2d.png)

### 17. val032_train_13301_a_2 · finding 2

Subcentimeter nonspecific nodules in both lungs

![2d val032_train_13301_a_2 · finding 2](lung_only/val032_train_13301_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val032_train_13301_a_2_finding2_2d.png)

### 18. val033_train_19182_a_2 · finding 1

Multiple indeterminate subcentimeter pulmonary nodules in both lungs

![2d val033_train_19182_a_2 · finding 1](lung_only/val033_train_19182_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val033_train_19182_a_2_finding1_2d.png)

### 19. val035_train_13076_a_1 · finding 2

Subcentimeter nonspecific nodules in both lungs

![2d val035_train_13076_a_1 · finding 2](lung_only/val035_train_13076_a_1_finding2_2d.png)

[Open full-resolution PNG](lung_only/val035_train_13076_a_1_finding2_2d.png)

### 20. val037_train_19355_a_2 · finding 0

Right apical bleb

![2d val037_train_19355_a_2 · finding 0](lung_only/val037_train_19355_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val037_train_19355_a_2_finding0_2d.png)

</details>

<details>
<summary>Figures 21–30 of 132</summary>

### 21. val041_train_18969_a_2 · finding 1

Pulmonary nodule 5 mm along the right minor fissure with anterior pleural extension

![2d val041_train_18969_a_2 · finding 1](lung_only/val041_train_18969_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val041_train_18969_a_2_finding1_2d.png)

### 22. val042_train_13092_a_2 · finding 0

5.5 mm nonspecific nodule in the subpleural region of the right upper lobe posterior segment

![2d val042_train_13092_a_2 · finding 0](lung_only/val042_train_13092_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val042_train_13092_a_2_finding0_2d.png)

### 23. val044_train_19452_a_2 · finding 3

Scattered calcified subcentimeter parenchymal nodules in both lungs

![2d val044_train_19452_a_2 · finding 3](lung_only/val044_train_19452_a_2_finding3_2d.png)

[Open full-resolution PNG](lung_only/val044_train_19452_a_2_finding3_2d.png)

### 24. val045_train_13166_a_2 · finding 1

Numerous small pulmonary nodules in both lungs

![2d val045_train_13166_a_2 · finding 1](lung_only/val045_train_13166_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val045_train_13166_a_2_finding1_2d.png)

### 25. val045_train_13166_a_2 · finding 2

Largest pulmonary nodule measuring 8 mm in diameter in the left lung fissure

![2d val045_train_13166_a_2 · finding 2](lung_only/val045_train_13166_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val045_train_13166_a_2_finding2_2d.png)

### 26. val046_train_2477_a_2 · finding 0

One calcified pulmonary nodule

![2d val046_train_2477_a_2 · finding 0](lung_only/val046_train_2477_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val046_train_2477_a_2_finding0_2d.png)

### 27. val047_train_13624_a_2 · finding 1

Calcified pulmonary nodule measuring 2 mm in the superior segment of the right lower lobe

![2d val047_train_13624_a_2 · finding 1](lung_only/val047_train_13624_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val047_train_13624_a_2_finding1_2d.png)

### 28. val048_train_13430_a_2 · finding 0

Pleural-based nodule measuring 5x3 mm in the superior segment of the lower lobe of the right lung

![2d val048_train_13430_a_2 · finding 0](lung_only/val048_train_13430_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val048_train_13430_a_2_finding0_2d.png)

### 29. val050_train_19877_a_2 · finding 0

Subcentimeter calcified nodules in both upper lobes

![2d val050_train_19877_a_2 · finding 0](lung_only/val050_train_19877_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val050_train_19877_a_2_finding0_2d.png)

### 30. val052_train_18421_b_1 · finding 0

A few subcentimeter indeterminate nodules in the right lung

![2d val052_train_18421_b_1 · finding 0](lung_only/val052_train_18421_b_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val052_train_18421_b_1_finding0_2d.png)

</details>

<details>
<summary>Figures 31–40 of 132</summary>

### 31. val053_train_18641_a_2 · finding 2

4 mm nodule in the inferior lingular segment of the left upper lobe with a peripheral ground-glass halo

![2d val053_train_18641_a_2 · finding 2](lung_only/val053_train_18641_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val053_train_18641_a_2_finding2_2d.png)

### 32. val054_train_19443_b_2 · finding 0

Several peripheral calcified pulmonary nodules in both lungs

![2d val054_train_19443_b_2 · finding 0](lung_only/val054_train_19443_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val054_train_19443_b_2_finding0_2d.png)

### 33. val055_train_2464_a_1 · finding 0

4 mm indeterminate nodule in the medial subpleural region of the right middle lobe

![2d val055_train_2464_a_1 · finding 0](lung_only/val055_train_2464_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val055_train_2464_a_1_finding0_2d.png)

### 34. val056_train_18662_a_1 · finding 0

Subcentimeter indeterminate pulmonary nodules in both lungs, including the inferior lingular segment of the left upper lobe

![2d val056_train_18662_a_1 · finding 0](lung_only/val056_train_18662_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val056_train_18662_a_1_finding0_2d.png)

### 35. val057_train_19468_a_1 · finding 1

A few pulmonary nodules less than 5 mm in both lungs

![2d val057_train_19468_a_1 · finding 1](lung_only/val057_train_19468_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val057_train_19468_a_1_finding1_2d.png)

### 36. val057_train_19468_a_1 · finding 2

Two calcified nodules in the posterior left upper lobe

![2d val057_train_19468_a_1 · finding 2](lung_only/val057_train_19468_a_1_finding2_2d.png)

[Open full-resolution PNG](lung_only/val057_train_19468_a_1_finding2_2d.png)

### 37. val058_train_13082_a_1 · finding 3

Calcified nodule measuring 3 mm at the laterobasal level

![2d val058_train_13082_a_1 · finding 3](lung_only/val058_train_13082_a_1_finding3_2d.png)

[Open full-resolution PNG](lung_only/val058_train_13082_a_1_finding3_2d.png)

### 38. val060_train_2443_a_2 · finding 0

Multiple subcentimeter indeterminate pulmonary nodules in both lungs, largest 4 mm in the posterior portion of the apicoposterior segment of the left upper lobe

![2d val060_train_2443_a_2 · finding 0](lung_only/val060_train_2443_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val060_train_2443_a_2_finding0_2d.png)

### 39. val062_train_13492_b_2 · finding 0

Two nonspecific nodules measuring 4 mm in diameter in the lateral segment of the right middle lobe

![2d val062_train_13492_b_2 · finding 0](lung_only/val062_train_13492_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val062_train_13492_b_2_finding0_2d.png)

### 40. val063_train_18452_d_2 · finding 0

A few subcentimeter indeterminate pulmonary nodules in both lungs

![2d val063_train_18452_d_2 · finding 0](lung_only/val063_train_18452_d_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val063_train_18452_d_2_finding0_2d.png)

</details>

<details>
<summary>Figures 41–50 of 132</summary>

### 41. val063_train_18452_d_2 · finding 1

Some pulmonary nodules are calcified in both lungs

![2d val063_train_18452_d_2 · finding 1](lung_only/val063_train_18452_d_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val063_train_18452_d_2_finding1_2d.png)

### 42. val064_train_2646_a_1 · finding 0

Subpleural subcentimeter nodule in the inferior lingular segment of the left upper lobe

![2d val064_train_2646_a_1 · finding 0](lung_only/val064_train_2646_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val064_train_2646_a_1_finding0_2d.png)

### 43. val065_train_18542_a_1 · finding 1

Nonspecific subcentimeter nodules in the right upper lobe

![2d val065_train_18542_a_1 · finding 1](lung_only/val065_train_18542_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val065_train_18542_a_1_finding1_2d.png)

### 44. val066_train_19767_a_1 · finding 1

2 mm nodule in the anterior segment of the right upper lobe, unchanged

![2d val066_train_19767_a_1 · finding 1](lung_only/val066_train_19767_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val066_train_19767_a_1_finding1_2d.png)

### 45. val069_train_18392_a_1 · finding 0

Multiple pulmonary nodules in both lungs, less than 5 mm

![2d val069_train_18392_a_1 · finding 0](lung_only/val069_train_18392_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val069_train_18392_a_1_finding0_2d.png)

### 46. val070_train_19397_a_1 · finding 0

Pulmonary nodules in both lungs, including the left lower lobe posterobasal segment, measuring up to 5 mm

![2d val070_train_19397_a_1 · finding 0](lung_only/val070_train_19397_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val070_train_19397_a_1_finding0_2d.png)

### 47. val073_train_19341_a_2 · finding 0

Indeterminate pulmonary nodules in both lungs measuring up to 3 mm, largest in the superior segment of the right lower lobe

![2d val073_train_19341_a_2 · finding 0](lung_only/val073_train_19341_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val073_train_19341_a_2_finding0_2d.png)

### 48. val075_train_19545_a_2 · finding 2

Subcentimeter calcified nodule in the left lower lobe

![2d val075_train_19545_a_2 · finding 2](lung_only/val075_train_19545_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val075_train_19545_a_2_finding2_2d.png)

### 49. val076_train_2562_a_1 · finding 0

Indeterminate pulmonary nodule, 3 mm, in the superior segment of the left lower lobe

![2d val076_train_2562_a_1 · finding 0](lung_only/val076_train_2562_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val076_train_2562_a_1_finding0_2d.png)

### 50. val077_train_13035_a_1 · finding 2

4 mm nodule in the medial segment of the right middle lobe

![2d val077_train_13035_a_1 · finding 2](lung_only/val077_train_13035_a_1_finding2_2d.png)

[Open full-resolution PNG](lung_only/val077_train_13035_a_1_finding2_2d.png)

</details>

<details>
<summary>Figures 51–60 of 132</summary>

### 51. val078_train_2580_b_2 · finding 0

New indeterminate subpleural nodule in the posterior aspect of the right lower lobe

![2d val078_train_2580_b_2 · finding 0](lung_only/val078_train_2580_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val078_train_2580_b_2_finding0_2d.png)

### 52. val080_train_2724_a_1 · finding 0

A few subcentimeter nonspecific pulmonary nodules in both lungs

![2d val080_train_2724_a_1 · finding 0](lung_only/val080_train_2724_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val080_train_2724_a_1_finding0_2d.png)

### 53. val083_train_2592_a_2 · finding 0

10 mm nodule in the peripheral subpleural region of the superior segment of the left lower lobe

![2d val083_train_2592_a_2 · finding 0](lung_only/val083_train_2592_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val083_train_2592_a_2_finding0_2d.png)

### 54. val084_train_2659_a_2 · finding 0

Indeterminate subcentimeter pulmonary nodules in both lungs

![2d val084_train_2659_a_2 · finding 0](lung_only/val084_train_2659_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val084_train_2659_a_2_finding0_2d.png)

### 55. val087_train_18600_a_1 · finding 0

Multiple nonspecific subpleural nodules in both lungs, also seen on the prior examination

![2d val087_train_18600_a_1 · finding 0](lung_only/val087_train_18600_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val087_train_18600_a_1_finding0_2d.png)

### 56. val088_train_18976_a_2 · finding 0

Subpleural 4 mm indeterminate nodule in the posterobasal segment of the left lower lobe

![2d val088_train_18976_a_2 · finding 0](lung_only/val088_train_18976_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val088_train_18976_a_2_finding0_2d.png)

### 57. val092_train_2564_c_1 · finding 2

A few indeterminate subcentimeter pulmonary nodules in both lungs

![2d val092_train_2564_c_1 · finding 2](lung_only/val092_train_2564_c_1_finding2_2d.png)

[Open full-resolution PNG](lung_only/val092_train_2564_c_1_finding2_2d.png)

### 58. val093_train_25_a_2 · finding 0

Indeterminate pulmonary nodule along the fissure in the superior segment of the left lower lobe, measuring 6 mm

![2d val093_train_25_a_2 · finding 0](lung_only/val093_train_25_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val093_train_25_a_2_finding0_2d.png)

### 59. val094_train_2666_a_2 · finding 0

Indeterminate subcentimeter pulmonary nodules present

![2d val094_train_2666_a_2 · finding 0](lung_only/val094_train_2666_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val094_train_2666_a_2_finding0_2d.png)

### 60. val096_train_2148_a_2 · finding 2

Peripheral subpleural parenchymal nodule, 5 mm, in the lateral segment of the right middle lobe

![2d val096_train_2148_a_2 · finding 2](lung_only/val096_train_2148_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val096_train_2148_a_2_finding2_2d.png)

</details>

<details>
<summary>Figures 61–70 of 132</summary>

### 61. val097_train_2475_a_2 · finding 1

Several subcentimeter indeterminate pulmonary nodules in both lungs with largest measuring 4 mm

![2d val097_train_2475_a_2 · finding 1](lung_only/val097_train_2475_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val097_train_2475_a_2_finding1_2d.png)

### 62. val098_train_2617_b_2 · finding 4

3 mm pleural-based subpleural nodule in the lateral basal segment of the left lower lobe

![2d val098_train_2617_b_2 · finding 4](lung_only/val098_train_2617_b_2_finding4_2d.png)

[Open full-resolution PNG](lung_only/val098_train_2617_b_2_finding4_2d.png)

### 63. val099_train_2664_a_2 · finding 0

Subcentimeter calcified nodule in the posterior segment of the right upper lobe

![2d val099_train_2664_a_2 · finding 0](lung_only/val099_train_2664_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val099_train_2664_a_2_finding0_2d.png)

### 64. val101_train_13398_a_1 · finding 3

Subcentimeter nonspecific nodules in both lungs

![2d val101_train_13398_a_1 · finding 3](lung_only/val101_train_13398_a_1_finding3_2d.png)

[Open full-resolution PNG](lung_only/val101_train_13398_a_1_finding3_2d.png)

### 65. val102_train_13339_a_1 · finding 0

Nonspecific pulmonary nodule, 3.9 mm, in the anterobasal segment of the right lower lobe

![2d val102_train_13339_a_1 · finding 0](lung_only/val102_train_13339_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val102_train_13339_a_1_finding0_2d.png)

### 66. val102_train_13339_a_1 · finding 1

Nonspecific pulmonary nodule, 4.3 mm, in the laterobasal segment of the left lower lobe

![2d val102_train_13339_a_1 · finding 1](lung_only/val102_train_13339_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val102_train_13339_a_1_finding1_2d.png)

### 67. val103_train_2677_a_2 · finding 0

Several subcentimeter indeterminate pulmonary nodules in both lungs

![2d val103_train_2677_a_2 · finding 0](lung_only/val103_train_2677_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val103_train_2677_a_2_finding0_2d.png)

### 68. val104_train_18956_a_1 · finding 0

Indeterminate pulmonary nodule in the left upper lobe, 3 mm

![2d val104_train_18956_a_1 · finding 0](lung_only/val104_train_18956_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val104_train_18956_a_1_finding0_2d.png)

### 69. val105_train_19032_a_2 · finding 1

Multiple subcentimeter indeterminate pulmonary nodules in both lungs

![2d val105_train_19032_a_2 · finding 1](lung_only/val105_train_19032_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val105_train_19032_a_2_finding1_2d.png)

### 70. val108_train_19021_b_2 · finding 0

Multiple indeterminate pulmonary parenchymal nodules in both lungs measuring 3 mm, largest in the posterobasal segment of the left lower lobe

![2d val108_train_19021_b_2 · finding 0](lung_only/val108_train_19021_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val108_train_19021_b_2_finding0_2d.png)

</details>

<details>
<summary>Figures 71–80 of 132</summary>

### 71. val109_train_13113_a_1 · finding 0

A few subcentimeter nonspecific nodules in both lungs

![2d val109_train_13113_a_1 · finding 0](lung_only/val109_train_13113_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val109_train_13113_a_1_finding0_2d.png)

### 72. val110_train_13316_a_2 · finding 0

Multiple parenchymal nodules in both lungs

![2d val110_train_13316_a_2 · finding 0](lung_only/val110_train_13316_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val110_train_13316_a_2_finding0_2d.png)

### 73. val110_train_13316_a_2 · finding 1

Largest nodule in the anterobasal segment of the lower lobe of the left lung, measuring 18x15 mm

![2d val110_train_13316_a_2 · finding 1](lung_only/val110_train_13316_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val110_train_13316_a_2_finding1_2d.png)

### 74. val111_train_2560_d_2 · finding 2

Few subcentimeter indeterminate pulmonary nodules bilaterally, unchanged from prior

![2d val111_train_2560_d_2 · finding 2](lung_only/val111_train_2560_d_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val111_train_2560_d_2_finding2_2d.png)

### 75. val113_train_19279_c_1 · finding 0

Stable subcentimeter pulmonary parenchymal nodules in both lungs, some calcified; largest 5 mm in the right middle lobe

![2d val113_train_19279_c_1 · finding 0](lung_only/val113_train_19279_c_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val113_train_19279_c_1_finding0_2d.png)

### 76. val114_train_13155_a_2 · finding 1

Several subcentimeter nodules in both lungs, measuring less than 3 mm in short diameter

![2d val114_train_13155_a_2 · finding 1](lung_only/val114_train_13155_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val114_train_13155_a_2_finding1_2d.png)

### 77. val116_train_13119_c_1 · finding 1

Subcentimeter nodules in both lungs

![2d val116_train_13119_c_1 · finding 1](lung_only/val116_train_13119_c_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val116_train_13119_c_1_finding1_2d.png)

### 78. val119_train_2603_a_1 · finding 0

Indeterminate subcentimeter nodules in both lungs

![2d val119_train_2603_a_1 · finding 0](lung_only/val119_train_2603_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val119_train_2603_a_1_finding0_2d.png)

### 79. val121_train_13252_b_2 · finding 4

Subcentimeter nonspecific nodules in both lungs

![2d val121_train_13252_b_2 · finding 4](lung_only/val121_train_13252_b_2_finding4_2d.png)

[Open full-resolution PNG](lung_only/val121_train_13252_b_2_finding4_2d.png)

### 80. val122_train_2455_a_1 · finding 0

Indeterminate subcentimeter nodules (two) in the anterior right lower lobe adjacent to the major fissure, largest 3 mm

![2d val122_train_2455_a_1 · finding 0](lung_only/val122_train_2455_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val122_train_2455_a_1_finding0_2d.png)

</details>

<details>
<summary>Figures 81–90 of 132</summary>

### 81. val126_train_18399_a_2 · finding 0

A few intrapulmonary subcentimeter nodules in the right lung, unchanged in number and size; largest 4 mm in the lateral segment of the right middle lobe

![2d val126_train_18399_a_2 · finding 0](lung_only/val126_train_18399_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val126_train_18399_a_2_finding0_2d.png)

### 82. val127_train_13444_a_2 · finding 0

One or two nonspecific parenchymal nodules in both lungs

![2d val127_train_13444_a_2 · finding 0](lung_only/val127_train_13444_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val127_train_13444_a_2_finding0_2d.png)

### 83. val128_train_2594_a_2 · finding 3

Scattered subcentimeter indeterminate pulmonary nodules in both lungs, some calcified

![2d val128_train_2594_a_2 · finding 3](lung_only/val128_train_2594_a_2_finding3_2d.png)

[Open full-resolution PNG](lung_only/val128_train_2594_a_2_finding3_2d.png)

### 84. val130_train_12994_a_1 · finding 1

Calcified nodules in both lung parenchyma

![2d val130_train_12994_a_1 · finding 1](lung_only/val130_train_12994_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val130_train_12994_a_1_finding1_2d.png)

### 85. val132_train_18624_a_2 · finding 0

Multiple nonspecific pulmonary parenchymal nodules in both lungs, predominantly in bilateral lower lobes and middle lobe, measuring up to 6 mm; largest in the laterobasal segment of the left lower lobe

![2d val132_train_18624_a_2 · finding 0](lung_only/val132_train_18624_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val132_train_18624_a_2_finding0_2d.png)

### 86. val134_train_13278_b_2 · finding 0

2-3 nodules in the right lung, largest measuring approximately 4 mm in diameter

![2d val134_train_13278_b_2 · finding 0](lung_only/val134_train_13278_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val134_train_13278_b_2_finding0_2d.png)

### 87. val136_train_18948_a_2 · finding 0

Multiple indeterminate pulmonary nodules in both lungs, subcentimeter, largest 3 mm in the apicoposterior segment of the left upper lobe

![2d val136_train_18948_a_2 · finding 0](lung_only/val136_train_18948_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val136_train_18948_a_2_finding0_2d.png)

### 88. val137_train_18597_b_1 · finding 0

A few nonspecific pulmonary nodules present in both lungs; largest 4 mm in the lateral basal segment of the right lower lobe

![2d val137_train_18597_b_1 · finding 0](lung_only/val137_train_18597_b_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val137_train_18597_b_1_finding0_2d.png)

### 89. val138_train_18959_b_2 · finding 0

Multiple small indeterminate subcentimeter pulmonary nodules in both lungs, stable

![2d val138_train_18959_b_2 · finding 0](lung_only/val138_train_18959_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val138_train_18959_b_2_finding0_2d.png)

### 90. val139_train_13417_d_1 · finding 0

Stable 3 mm subcentimeter nodule near the subpleural area in the laterobasal segment of the left lower lobe

![2d val139_train_13417_d_1 · finding 0](lung_only/val139_train_13417_d_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val139_train_13417_d_1_finding0_2d.png)

</details>

<details>
<summary>Figures 91–100 of 132</summary>

### 91. val140_train_18525_a_2 · finding 0

A few subcentimeter indeterminate nodules in the anterior and lateral segments of the left lower lobe

![2d val140_train_18525_a_2 · finding 0](lung_only/val140_train_18525_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val140_train_18525_a_2_finding0_2d.png)

### 92. val141_train_2549_a_2 · finding 2

A few subcentimeter indeterminate pulmonary nodules in both lungs

![2d val141_train_2549_a_2 · finding 2](lung_only/val141_train_2549_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val141_train_2549_a_2_finding2_2d.png)

### 93. val142_train_2936_b_2 · finding 0

Juxtadiaphragmatic calcified nodular densities in the basal segments of both lower lobes

![2d val142_train_2936_b_2 · finding 0](lung_only/val142_train_2936_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val142_train_2936_b_2_finding0_2d.png)

### 94. val142_train_2936_b_2 · finding 1

Juxtadiaphragmatic noncalcified nodular densities in the basal segments of both lower lobes

![2d val142_train_2936_b_2 · finding 1](lung_only/val142_train_2936_b_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val142_train_2936_b_2_finding1_2d.png)

### 95. val143_train_2696_a_2 · finding 0

Single subcentimeter indeterminate pulmonary nodule in the right middle lobe

![2d val143_train_2696_a_2 · finding 0](lung_only/val143_train_2696_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val143_train_2696_a_2_finding0_2d.png)

### 96. val144_train_1841_b_1 · finding 3

Nonspecific 2 mm nodule in the right middle lobe

![2d val144_train_1841_b_1 · finding 3](lung_only/val144_train_1841_b_1_finding3_2d.png)

[Open full-resolution PNG](lung_only/val144_train_1841_b_1_finding3_2d.png)

### 97. val145_train_19743_c_2 · finding 1

Parenchymal nodules in the superior segment of the left lower lobe

![2d val145_train_19743_c_2 · finding 1](lung_only/val145_train_19743_c_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val145_train_19743_c_2_finding1_2d.png)

### 98. val146_train_13148_a_2 · finding 0

Two subpleural subcentimeter nodules measuring 4 mm at the basal level in the superior right lower lobe

![2d val146_train_13148_a_2 · finding 0](lung_only/val146_train_13148_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val146_train_13148_a_2_finding0_2d.png)

### 99. val146_train_13148_a_2 · finding 1

Two subpleural subcentimeter nodules measuring 4 mm in the left lower lobe

![2d val146_train_13148_a_2 · finding 1](lung_only/val146_train_13148_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val146_train_13148_a_2_finding1_2d.png)

### 100. val148_train_2482_a_1 · finding 0

Perifissural nodule measuring 4 mm along the right minor fissure

![2d val148_train_2482_a_1 · finding 0](lung_only/val148_train_2482_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val148_train_2482_a_1_finding0_2d.png)

</details>

<details>
<summary>Figures 101–110 of 132</summary>

### 101. val149_train_19923_a_2 · finding 0

Subcentimeter pulmonary nodules in both lungs, largest 2 mm in posterior segment of right upper lobe

![2d val149_train_19923_a_2 · finding 0](lung_only/val149_train_19923_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val149_train_19923_a_2_finding0_2d.png)

### 102. val150_train_2468_a_2 · finding 1

4 mm subpleural nodule along the diaphragmatic surface of the right lower lobe

![2d val150_train_2468_a_2 · finding 1](lung_only/val150_train_2468_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val150_train_2468_a_2_finding1_2d.png)

### 103. val152_train_13447_a_1 · finding 0

Subcentimeter non-specific nodules in both lungs

![2d val152_train_13447_a_1 · finding 0](lung_only/val152_train_13447_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val152_train_13447_a_1_finding0_2d.png)

### 104. val153_train_2530_a_2 · finding 0

A few pleural-based nodules in the posterior aspect of the right upper lobe, largest measuring 5 mm

![2d val153_train_2530_a_2 · finding 0](lung_only/val153_train_2530_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val153_train_2530_a_2_finding0_2d.png)

### 105. val156_train_18721_a_2 · finding 0

Two noncalcified solid nodules in the right middle lobe measuring 6 mm and 7 mm

![2d val156_train_18721_a_2 · finding 0](lung_only/val156_train_18721_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val156_train_18721_a_2_finding0_2d.png)

### 106. val156_train_18721_a_2 · finding 1

Fissure-based nodules in the superior segment of the right lower lobe

![2d val156_train_18721_a_2 · finding 1](lung_only/val156_train_18721_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val156_train_18721_a_2_finding1_2d.png)

### 107. val156_train_18721_a_2 · finding 2

Two noncalcified solid nodules in the anterior segment of the left upper lobe measuring 4 mm and 6 mm

![2d val156_train_18721_a_2 · finding 2](lung_only/val156_train_18721_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val156_train_18721_a_2_finding2_2d.png)

### 108. val157_train_2515_b_2 · finding 0

Indeterminate subcentimeter nodules present in both lungs

![2d val157_train_2515_b_2 · finding 0](lung_only/val157_train_2515_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val157_train_2515_b_2_finding0_2d.png)

### 109. val158_train_2493_a_1 · finding 1

Two indeterminate subcentimeter pulmonary nodules in the right lung

![2d val158_train_2493_a_1 · finding 1](lung_only/val158_train_2493_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val158_train_2493_a_1_finding1_2d.png)

### 110. val159_train_13631_a_1 · finding 0

3 mm calcified nodule in the posterior segment of the left upper lobe

![2d val159_train_13631_a_1 · finding 0](lung_only/val159_train_13631_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val159_train_13631_a_1_finding0_2d.png)

</details>

<details>
<summary>Figures 111–120 of 132</summary>

### 111. val160_train_19459_a_2 · finding 1

Indeterminate subcentimeter pulmonary parenchymal nodules in both lungs

![2d val160_train_19459_a_2 · finding 1](lung_only/val160_train_19459_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val160_train_19459_a_2_finding1_2d.png)

### 112. val164_train_13426_a_2 · finding 0

Subcentimeter air cyst in the posterobasal segment of the lower lobe of the right lung

![2d val164_train_13426_a_2 · finding 0](lung_only/val164_train_13426_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val164_train_13426_a_2_finding0_2d.png)

### 113. val164_train_13426_a_2 · finding 1

Two nodules, each measuring 2 mm in diameter, in the laterobasal segment of the lower lobe of the left lung

![2d val164_train_13426_a_2 · finding 1](lung_only/val164_train_13426_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val164_train_13426_a_2_finding1_2d.png)

### 114. val165_train_19315_a_2 · finding 1

Calcified nodule at the apex of the left upper lobe, subcentimeter

![2d val165_train_19315_a_2 · finding 1](lung_only/val165_train_19315_a_2_finding1_2d.png)

[Open full-resolution PNG](lung_only/val165_train_19315_a_2_finding1_2d.png)

### 115. val170_train_19771_a_2 · finding 2

Indeterminate subcentimeter right lung nodules, some completely calcified

![2d val170_train_19771_a_2 · finding 2](lung_only/val170_train_19771_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val170_train_19771_a_2_finding2_2d.png)

### 116. val171_train_2564_b_2 · finding 0

Several subcentimeter indeterminate nodules in both lungs

![2d val171_train_2564_b_2 · finding 0](lung_only/val171_train_2564_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val171_train_2564_b_2_finding0_2d.png)

### 117. val173_train_13189_a_1 · finding 1

Vascular structure extending to the right lower lobe from the aorta, consistent with pulmonary sequestration

![2d val173_train_13189_a_1 · finding 1](lung_only/val173_train_13189_a_1_finding1_2d.png)

[Open full-resolution PNG](lung_only/val173_train_13189_a_1_finding1_2d.png)

### 118. val174_train_2680_a_2 · finding 2

Nodular ground-glass opacities in the right upper lobe

![2d val174_train_2680_a_2 · finding 2](lung_only/val174_train_2680_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val174_train_2680_a_2_finding2_2d.png)

### 119. val176_train_205_a_1 · finding 0

A few subcentimeter indeterminate pulmonary nodules in both lungs

![2d val176_train_205_a_1 · finding 0](lung_only/val176_train_205_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val176_train_205_a_1_finding0_2d.png)

### 120. val177_train_2625_b_2 · finding 0

Stable 5 mm irregular ground-glass nodule in the mediobasal segment of the left lower lobe

![2d val177_train_2625_b_2 · finding 0](lung_only/val177_train_2625_b_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val177_train_2625_b_2_finding0_2d.png)

</details>

<details>
<summary>Figures 121–130 of 132</summary>

### 121. val180_train_19346_a_2 · finding 0

Nonspecific subcentimeter pulmonary nodules in both lungs

![2d val180_train_19346_a_2 · finding 0](lung_only/val180_train_19346_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val180_train_19346_a_2_finding0_2d.png)

### 122. val182_train_18454_a_2 · finding 0

Subpleural subcentimeter nodule in the posterior right upper lobe

![2d val182_train_18454_a_2 · finding 0](lung_only/val182_train_18454_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val182_train_18454_a_2_finding0_2d.png)

### 123. val183_train_13013_a_1 · finding 3

Largest nodule in the upper lobe of the right lung measuring approximately 5 mm in diameter

![2d val183_train_13013_a_1 · finding 3](lung_only/val183_train_13013_a_1_finding3_2d.png)

[Open full-resolution PNG](lung_only/val183_train_13013_a_1_finding3_2d.png)

### 124. val184_train_19422_a_1 · finding 0

Patchy subpleural nodular opacities in both lungs

![2d val184_train_19422_a_1 · finding 0](lung_only/val184_train_19422_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val184_train_19422_a_1_finding0_2d.png)

### 125. val185_train_3006_a_2 · finding 2

Indeterminate subcentimeter pulmonary nodules in both lungs

![2d val185_train_3006_a_2 · finding 2](lung_only/val185_train_3006_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val185_train_3006_a_2_finding2_2d.png)

### 126. val190_train_18569_a_2 · finding 0

3 mm subpleural nodule in the posterior segment of the right upper lobe

![2d val190_train_18569_a_2 · finding 0](lung_only/val190_train_18569_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val190_train_18569_a_2_finding0_2d.png)

### 127. val191_train_19209_a_2 · finding 2

A few subcentimeter pulmonary nodules in both lungs, some calcified

![2d val191_train_19209_a_2 · finding 2](lung_only/val191_train_19209_a_2_finding2_2d.png)

[Open full-resolution PNG](lung_only/val191_train_19209_a_2_finding2_2d.png)

### 128. val192_train_2995_a_2 · finding 0

Peripherally located pulmonary nodule measuring 4 mm in the superior segment of the right lower lobe (subcentimeter intrapulmonary lymph node)

![2d val192_train_2995_a_2 · finding 0](lung_only/val192_train_2995_a_2_finding0_2d.png)

[Open full-resolution PNG](lung_only/val192_train_2995_a_2_finding0_2d.png)

### 129. val193_train_19465_a_1 · finding 0

8 mm subpleural nodule in the lateral aspect of the right lower lobe

![2d val193_train_19465_a_1 · finding 0](lung_only/val193_train_19465_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val193_train_19465_a_1_finding0_2d.png)

### 130. val194_train_2568_a_5 · finding 0

A few indeterminate pulmonary nodules in both lungs, each less than 3 mm in diameter

![2d val194_train_2568_a_5 · finding 0](lung_only/val194_train_2568_a_5_finding0_2d.png)

[Open full-resolution PNG](lung_only/val194_train_2568_a_5_finding0_2d.png)

</details>

<details>
<summary>Figures 131–132 of 132</summary>

### 131. val197_train_18597_a_1 · finding 0

A few subcentimeter indeterminate pulmonary nodules in both lungs, each less than 3 mm

![2d val197_train_18597_a_1 · finding 0](lung_only/val197_train_18597_a_1_finding0_2d.png)

[Open full-resolution PNG](lung_only/val197_train_18597_a_1_finding0_2d.png)

### 132. val198_train_3026_b_2 · finding 3

Subcentimeter pulmonary nodules in both lungs, unchanged and stable

![2d val198_train_3026_b_2 · finding 3](lung_only/val198_train_3026_b_2_finding3_2d.png)

[Open full-resolution PNG](lung_only/val198_train_3026_b_2_finding3_2d.png)

</details>

