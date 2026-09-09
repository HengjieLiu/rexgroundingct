# 2a — Linear (including subsegmental atelectasis, scarring, fibrosis) — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 1120 | 14.57% |
| Val | 69 | 18.11% |
| Test | 113 | 19.42% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.337641 | 59/69 | 85.51% |
| Overall best | PP2 | 0.344155 | 59/69 | 85.51% |
| Overall best | PP3 (targeted corrections) | 0.348577 | 59/69 | 85.51% |
| Category best | Raw | 0.344451 | 56/69 | 81.16% |
| Category best | PP2 | 0.353661 | 57/69 | 82.61% |
| Category best | PP3 (targeted corrections) | 0.355015 | 57/69 | 82.61% |
| Iso07 best | Raw | 0.343519 | 57/69 | 82.61% |
| Iso07 best | PP2 | 0.352936 | 57/69 | 82.61% |
| Iso07 best | PP3 (targeted corrections) | 0.353189 | 57/69 | 82.61% |

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

## Figures (69)

<details>
<summary>Figures 1–10 of 69</summary>

### 1. val003_train_2553_a_2 · finding 0

Reticulonodular fibrotic scarring in both lung apices

![2a val003_train_2553_a_2 · finding 0](lung_only/val003_train_2553_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val003_train_2553_a_2_finding0_2a.png)

### 2. val005_train_2675_a_1 · finding 0

Linear atelectasis in the lingular segment of the left upper lobe

![2a val005_train_2675_a_1 · finding 0](lung_only/val005_train_2675_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val005_train_2675_a_1_finding0_2a.png)

### 3. val006_train_3011_a_2 · finding 0

Parenchymal scarring in the apices of both lungs, left lower lobe superior segment and medial, lateral, and posterior basal segments, left upper lobe lingular inferior segment, and right lower lobe posterior and lateral basal segments

![2a val006_train_3011_a_2 · finding 0](lung_only/val006_train_3011_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val006_train_3011_a_2_finding0_2a.png)

### 4. val015_train_19038_a_2 · finding 0

Bilateral apical pleuroparenchymal scarring

![2a val015_train_19038_a_2 · finding 0](lung_only/val015_train_19038_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val015_train_19038_a_2_finding0_2a.png)

### 5. val017_train_13288_a_2 · finding 0

Fibrotic bands in the medial segment of the right middle lobe

![2a val017_train_13288_a_2 · finding 0](lung_only/val017_train_13288_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val017_train_13288_a_2_finding0_2a.png)

### 6. val019_train_13102_a_1 · finding 0

Fibrotic changes in the middle lobe of the right lung

![2a val019_train_13102_a_1 · finding 0](lung_only/val019_train_13102_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val019_train_13102_a_1_finding0_2a.png)

### 7. val020_train_18455_a_2 · finding 0

Linear pleuroparenchymal scarring in both lungs including the right middle lobe, right lower lobe anterobasal segment, and left lower lobe anterobasal segment

![2a val020_train_18455_a_2 · finding 0](lung_only/val020_train_18455_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val020_train_18455_a_2_finding0_2a.png)

### 8. val021_train_3015_a_1 · finding 0

Thickening/density along the oblique fissure adjacent to the superior aspect of the left lower lobe, measuring 13 x 8 mm

![2a val021_train_3015_a_1 · finding 0](lung_only/val021_train_3015_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val021_train_3015_a_1_finding0_2a.png)

### 9. val027_train_19456_a_1 · finding 1

Linear scarring/atelectasis in the medial segment of the right middle lobe

![2a val027_train_19456_a_1 · finding 1](lung_only/val027_train_19456_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val027_train_19456_a_1_finding1_2a.png)

### 10. val028_train_19368_a_2 · finding 0

Pulmonary scarring present

![2a val028_train_19368_a_2 · finding 0](lung_only/val028_train_19368_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val028_train_19368_a_2_finding0_2a.png)

</details>

<details>
<summary>Figures 11–20 of 69</summary>

### 11. val033_train_19182_a_2 · finding 0

Biapical pleuroparenchymal scarring

![2a val033_train_19182_a_2 · finding 0](lung_only/val033_train_19182_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val033_train_19182_a_2_finding0_2a.png)

### 12. val034_train_2213_a_1 · finding 0

Mild pleuroparenchymal scarring in both lungs

![2a val034_train_2213_a_1 · finding 0](lung_only/val034_train_2213_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val034_train_2213_a_1_finding0_2a.png)

### 13. val035_train_13076_a_1 · finding 1

Fibrotic densities in the upper lobes of both lungs

![2a val035_train_13076_a_1 · finding 1](lung_only/val035_train_13076_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val035_train_13076_a_1_finding1_2a.png)

### 14. val037_train_19355_a_2 · finding 1

Mild scarring in the lingular segment

![2a val037_train_19355_a_2 · finding 1](lung_only/val037_train_19355_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val037_train_19355_a_2_finding1_2a.png)

### 15. val044_train_19452_a_2 · finding 2

Linear atelectasis in the anteromedial basal segment of the left lower lobe

![2a val044_train_19452_a_2 · finding 2](lung_only/val044_train_19452_a_2_finding2_2a.png)

[Open full-resolution PNG](lung_only/val044_train_19452_a_2_finding2_2a.png)

### 16. val053_train_18641_a_2 · finding 0

Linear opacities in the medial segment of the right middle lobe

![2a val053_train_18641_a_2 · finding 0](lung_only/val053_train_18641_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val053_train_18641_a_2_finding0_2a.png)

### 17. val053_train_18641_a_2 · finding 1

Linear opacities in the inferior lingular segment of the left upper lobe

![2a val053_train_18641_a_2 · finding 1](lung_only/val053_train_18641_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val053_train_18641_a_2_finding1_2a.png)

### 18. val057_train_19468_a_1 · finding 0

Mild pleuroparenchymal scarring in the apicoposterior segments of both upper lobes

![2a val057_train_19468_a_1 · finding 0](lung_only/val057_train_19468_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val057_train_19468_a_1_finding0_2a.png)

### 19. val058_train_13082_a_1 · finding 2

Focal scarring changes at the posterobasal level of the left lung

![2a val058_train_13082_a_1 · finding 2](lung_only/val058_train_13082_a_1_finding2_2a.png)

[Open full-resolution PNG](lung_only/val058_train_13082_a_1_finding2_2a.png)

### 20. val061_train_3028_a_2 · finding 0

Fibrosis in the apical portion of the left upper lobe

![2a val061_train_3028_a_2 · finding 0](lung_only/val061_train_3028_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val061_train_3028_a_2_finding0_2a.png)

</details>

<details>
<summary>Figures 21–30 of 69</summary>

### 21. val062_train_13492_b_2 · finding 1

Parenchymal changes consistent with scarring in the middle lobe of the right lung

![2a val062_train_13492_b_2 · finding 1](lung_only/val062_train_13492_b_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val062_train_13492_b_2_finding1_2a.png)

### 22. val065_train_18542_a_1 · finding 0

Linear atelectasis/scar in the medial segment of the right middle lobe and the inferior lingular segment of the left lung

![2a val065_train_18542_a_1 · finding 0](lung_only/val065_train_18542_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val065_train_18542_a_1_finding0_2a.png)

### 23. val068_train_19180_a_2 · finding 1

Linear pleuroparenchymal scarring in the basal segment of the left lower lobe

![2a val068_train_19180_a_2 · finding 1](lung_only/val068_train_19180_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val068_train_19180_a_2_finding1_2a.png)

### 24. val071_train_3008_a_1 · finding 1

Parenchymal scarring in both apices and in the posterobasal segments of the lower lobes

![2a val071_train_3008_a_1 · finding 1](lung_only/val071_train_3008_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val071_train_3008_a_1_finding1_2a.png)

### 25. val073_train_19341_a_2 · finding 1

Linear atelectasis in both lungs

![2a val073_train_19341_a_2 · finding 1](lung_only/val073_train_19341_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val073_train_19341_a_2_finding1_2a.png)

### 26. val079_train_2508_a_2 · finding 0

Scarring in the lungs

![2a val079_train_2508_a_2 · finding 0](lung_only/val079_train_2508_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val079_train_2508_a_2_finding0_2a.png)

### 27. val087_train_18600_a_1 · finding 1

Chronic fibrotic scarring in the left upper lobe

![2a val087_train_18600_a_1 · finding 1](lung_only/val087_train_18600_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val087_train_18600_a_1_finding1_2a.png)

### 28. val089_train_2657_a_2 · finding 1

Fibrotic scarring in the right middle lobe and inferior lingular segment of the left lung with minimal scarring in both lungs

![2a val089_train_2657_a_2 · finding 1](lung_only/val089_train_2657_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val089_train_2657_a_2_finding1_2a.png)

### 29. val090_train_13098_a_2 · finding 1

Subsegmental linear atelectasis in both lungs, especially in the lower lobes

![2a val090_train_13098_a_2 · finding 1](lung_only/val090_train_13098_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val090_train_13098_a_2_finding1_2a.png)

### 30. val092_train_2564_c_1 · finding 1

Minimal pleuroparenchymal scarring at both apices bilaterally

![2a val092_train_2564_c_1 · finding 1](lung_only/val092_train_2564_c_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val092_train_2564_c_1_finding1_2a.png)

</details>

<details>
<summary>Figures 31–40 of 69</summary>

### 31. val096_train_2148_a_2 · finding 1

Fibroatelectatic scarring at both lung apices

![2a val096_train_2148_a_2 · finding 1](lung_only/val096_train_2148_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val096_train_2148_a_2_finding1_2a.png)

### 32. val097_train_2475_a_2 · finding 0

Linear atelectasis in the left lower lobe

![2a val097_train_2475_a_2 · finding 0](lung_only/val097_train_2475_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val097_train_2475_a_2_finding0_2a.png)

### 33. val101_train_13398_a_1 · finding 1

Parenchymal changes with scarring in the right middle lobe medial segment

![2a val101_train_13398_a_1 · finding 1](lung_only/val101_train_13398_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val101_train_13398_a_1_finding1_2a.png)

### 34. val101_train_13398_a_1 · finding 2

Parenchymal changes with scarring in the left upper lobe inferior segment and laterobasal segment

![2a val101_train_13398_a_1 · finding 2](lung_only/val101_train_13398_a_1_finding2_2a.png)

[Open full-resolution PNG](lung_only/val101_train_13398_a_1_finding2_2a.png)

### 35. val107_train_19325_a_2 · finding 2

Scarring in the lingular segment

![2a val107_train_19325_a_2 · finding 2](lung_only/val107_train_19325_a_2_finding2_2a.png)

[Open full-resolution PNG](lung_only/val107_train_19325_a_2_finding2_2a.png)

### 36. val113_train_19279_c_1 · finding 2

Subsegmental atelectasis in the medial basal segment of the left lower lobe

![2a val113_train_19279_c_1 · finding 2](lung_only/val113_train_19279_c_1_finding2_2a.png)

[Open full-resolution PNG](lung_only/val113_train_19279_c_1_finding2_2a.png)

### 37. val114_train_13155_a_2 · finding 2

Areas of linear atelectasis in both lungs

![2a val114_train_13155_a_2 · finding 2](lung_only/val114_train_13155_a_2_finding2_2a.png)

[Open full-resolution PNG](lung_only/val114_train_13155_a_2_finding2_2a.png)

### 38. val118_train_18410_b_2 · finding 0

Mild apical lung scarring

![2a val118_train_18410_b_2 · finding 0](lung_only/val118_train_18410_b_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val118_train_18410_b_2_finding0_2a.png)

### 39. val120_train_19722_a_1 · finding 0

Thin pleuroparenchymal linear opacities in the apical segments of both upper lobes

![2a val120_train_19722_a_1 · finding 0](lung_only/val120_train_19722_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val120_train_19722_a_1_finding0_2a.png)

### 40. val123_train_3020_a_2 · finding 0

Pleuroparenchymal scarring in the apicoposterior segments of both upper lobes

![2a val123_train_3020_a_2 · finding 0](lung_only/val123_train_3020_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val123_train_3020_a_2_finding0_2a.png)

</details>

<details>
<summary>Figures 41–50 of 69</summary>

### 41. val130_train_12994_a_1 · finding 2

Increased density consistent with linear atelectasis in the inferior lingular segment of the left upper lobe

![2a val130_train_12994_a_1 · finding 2](lung_only/val130_train_12994_a_1_finding2_2a.png)

[Open full-resolution PNG](lung_only/val130_train_12994_a_1_finding2_2a.png)

### 42. val131_train_18403_a_2 · finding 1

Linear atelectasis in both lungs involving both lower lobes and right upper lobe

![2a val131_train_18403_a_2 · finding 1](lung_only/val131_train_18403_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val131_train_18403_a_2_finding1_2a.png)

### 43. val137_train_18597_b_1 · finding 1

Linear fibrotic scarring in both lungs

![2a val137_train_18597_b_1 · finding 1](lung_only/val137_train_18597_b_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val137_train_18597_b_1_finding1_2a.png)

### 44. val139_train_13417_d_1 · finding 1

Fibrotic subcentimeter densities in the lower lobe of the right lung

![2a val139_train_13417_d_1 · finding 1](lung_only/val139_train_13417_d_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val139_train_13417_d_1_finding1_2a.png)

### 45. val141_train_2549_a_2 · finding 0

Linear atelectasis in the right middle lobe

![2a val141_train_2549_a_2 · finding 0](lung_only/val141_train_2549_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val141_train_2549_a_2_finding0_2a.png)

### 46. val141_train_2549_a_2 · finding 1

Linear atelectasis in the lingula of the left upper lobe

![2a val141_train_2549_a_2 · finding 1](lung_only/val141_train_2549_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val141_train_2549_a_2_finding1_2a.png)

### 47. val145_train_19743_c_2 · finding 0

Pleuroparenchymal fibrosis in the superior segment of the left lower lobe

![2a val145_train_19743_c_2 · finding 0](lung_only/val145_train_19743_c_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val145_train_19743_c_2_finding0_2a.png)

### 48. val150_train_2468_a_2 · finding 0

Bandlike area of scarring along the medial aspect of the right middle lobe

![2a val150_train_2468_a_2 · finding 0](lung_only/val150_train_2468_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val150_train_2468_a_2_finding0_2a.png)

### 49. val156_train_18721_a_2 · finding 3

Subsegmental atelectasis in the lingula

![2a val156_train_18721_a_2 · finding 3](lung_only/val156_train_18721_a_2_finding3_2a.png)

[Open full-resolution PNG](lung_only/val156_train_18721_a_2_finding3_2a.png)

### 50. val156_train_18721_a_2 · finding 4

Subsegmental atelectasis in the basilar segments of both lungs

![2a val156_train_18721_a_2 · finding 4](lung_only/val156_train_18721_a_2_finding4_2a.png)

[Open full-resolution PNG](lung_only/val156_train_18721_a_2_finding4_2a.png)

</details>

<details>
<summary>Figures 51–60 of 69</summary>

### 51. val158_train_2493_a_1 · finding 0

Bandlike minimal scarring in the inferior lingular segment of the left lung

![2a val158_train_2493_a_1 · finding 0](lung_only/val158_train_2493_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val158_train_2493_a_1_finding0_2a.png)

### 52. val160_train_19459_a_2 · finding 0

Reticulonodular fibrotic scarring in both lung apices

![2a val160_train_19459_a_2 · finding 0](lung_only/val160_train_19459_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val160_train_19459_a_2_finding0_2a.png)

### 53. val163_train_13497_a_1 · finding 0

Linear atelectasis in the medial segment of the right middle lobe

![2a val163_train_13497_a_1 · finding 0](lung_only/val163_train_13497_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val163_train_13497_a_1_finding0_2a.png)

### 54. val165_train_19315_a_2 · finding 0

Fibrotic scarring at the apices of both upper lobes

![2a val165_train_19315_a_2 · finding 0](lung_only/val165_train_19315_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val165_train_19315_a_2_finding0_2a.png)

### 55. val166_train_3027_a_2 · finding 0

Pulmonary scarring present

![2a val166_train_3027_a_2 · finding 0](lung_only/val166_train_3027_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val166_train_3027_a_2_finding0_2a.png)

### 56. val169_train_2451_a_1 · finding 0

Bandlike atelectasis in the superior segment of the left lower lobe

![2a val169_train_2451_a_1 · finding 0](lung_only/val169_train_2451_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val169_train_2451_a_1_finding0_2a.png)

### 57. val170_train_19771_a_2 · finding 0

Subsegmental atelectasis in the inferior lingular segment of the left upper lobe

![2a val170_train_19771_a_2 · finding 0](lung_only/val170_train_19771_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val170_train_19771_a_2_finding0_2a.png)

### 58. val170_train_19771_a_2 · finding 1

Subsegmental atelectasis in the medial segment of the right middle lobe

![2a val170_train_19771_a_2 · finding 1](lung_only/val170_train_19771_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val170_train_19771_a_2_finding1_2a.png)

### 59. val172_train_13249_b_2 · finding 1

Areas of increased density consistent with linear atelectasis in the basal segments of the lower lobes

![2a val172_train_13249_b_2 · finding 1](lung_only/val172_train_13249_b_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val172_train_13249_b_2_finding1_2a.png)

### 60. val174_train_2680_a_2 · finding 0

Pleuroparenchymal scarring at the apices of both lungs

![2a val174_train_2680_a_2 · finding 0](lung_only/val174_train_2680_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val174_train_2680_a_2_finding0_2a.png)

</details>

<details>
<summary>Figures 61–69 of 69</summary>

### 61. val179_train_2564_a_1 · finding 0

Pleuroparenchymal scarring at the apices of both lungs

![2a val179_train_2564_a_1 · finding 0](lung_only/val179_train_2564_a_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val179_train_2564_a_1_finding0_2a.png)

### 62. val183_train_13013_a_1 · finding 1

Linear atelectasis in the lower lobes of both lungs

![2a val183_train_13013_a_1 · finding 1](lung_only/val183_train_13013_a_1_finding1_2a.png)

[Open full-resolution PNG](lung_only/val183_train_13013_a_1_finding1_2a.png)

### 63. val185_train_3006_a_2 · finding 1

Scattered linear atelectasis in both lungs

![2a val185_train_3006_a_2 · finding 1](lung_only/val185_train_3006_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val185_train_3006_a_2_finding1_2a.png)

### 64. val187_train_13479_j_1 · finding 0

Parenchymal scarring in the right lower lobe posterobasal segment

![2a val187_train_13479_j_1 · finding 0](lung_only/val187_train_13479_j_1_finding0_2a.png)

[Open full-resolution PNG](lung_only/val187_train_13479_j_1_finding0_2a.png)

### 65. val188_train_18408_a_2 · finding 0

Reticulonodular fibrotic scarring in both apices

![2a val188_train_18408_a_2 · finding 0](lung_only/val188_train_18408_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val188_train_18408_a_2_finding0_2a.png)

### 66. val188_train_18408_a_2 · finding 1

Linear subsegmental atelectasis in the right middle lobe, inferior lingular segment of the left upper lobe, and basal segments of the left lower lobe

![2a val188_train_18408_a_2 · finding 1](lung_only/val188_train_18408_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val188_train_18408_a_2_finding1_2a.png)

### 67. val189_train_3026_a_2 · finding 1

Fibrotic scarring in the upper lobes of both lungs

![2a val189_train_3026_a_2 · finding 1](lung_only/val189_train_3026_a_2_finding1_2a.png)

[Open full-resolution PNG](lung_only/val189_train_3026_a_2_finding1_2a.png)

### 68. val191_train_19209_a_2 · finding 0

Pleuroparenchymal reticulonodular fibrotic scarring in both lungs including both apices, inferior lingular segment of the left upper lobe, and basal segments of the left lower lobe

![2a val191_train_19209_a_2 · finding 0](lung_only/val191_train_19209_a_2_finding0_2a.png)

[Open full-resolution PNG](lung_only/val191_train_19209_a_2_finding0_2a.png)

### 69. val198_train_3026_b_2 · finding 2

Linear atelectasis in the posterobasal segments of both lower lobes

![2a val198_train_3026_b_2 · finding 2](lung_only/val198_train_3026_b_2_finding2_2a.png)

[Open full-resolution PNG](lung_only/val198_train_3026_b_2_finding2_2a.png)

</details>

