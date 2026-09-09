# 2b — Atelectasis, consolidation — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 1367 | 17.78% |
| Val | 49 | 12.86% |
| Test | 89 | 15.29% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.357690 | 38/49 | 77.55% |
| Overall best | PP2 | 0.361010 | 38/49 | 77.55% |
| Overall best | PP3 (targeted corrections) | 0.363268 | 38/49 | 77.55% |
| Category best | Raw | 0.374337 | 38/49 | 77.55% |
| Category best | PP2 | 0.377395 | 38/49 | 77.55% |
| Category best | PP3 (targeted corrections) | 0.377536 | 38/49 | 77.55% |
| Iso07 best | Raw | 0.355555 | 36/49 | 73.47% |
| Iso07 best | PP2 | 0.362223 | 36/49 | 73.47% |
| Iso07 best | PP3 (targeted corrections) | 0.364929 | 36/49 | 73.47% |

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

## Figures (49)

<details>
<summary>Figures 1–10 of 49</summary>

### 1. val008_train_13481_a_2 · finding 0

Areas of pneumonic consolidation with air bronchograms in the middle lobe

![2b val008_train_13481_a_2 · finding 0](full_body/val008_train_13481_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val008_train_13481_a_2_finding0_2b.png)

### 2. val008_train_13481_a_2 · finding 1

Areas of pneumonic consolidation with air bronchograms in the lower lobes of both lungs, more prominently on the right

![2b val008_train_13481_a_2 · finding 1](full_body/val008_train_13481_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val008_train_13481_a_2_finding1_2b.png)

### 3. val010_train_13256_b_2 · finding 1

Areas of consolidation in all segments of both lungs

![2b val010_train_13256_b_2 · finding 1](full_body/val010_train_13256_b_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val010_train_13256_b_2_finding1_2b.png)

### 4. val012_train_19348_a_2 · finding 1

Scattered bilateral subpleural consolidation-like opacities

![2b val012_train_19348_a_2 · finding 1](full_body/val012_train_19348_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val012_train_19348_a_2_finding1_2b.png)

### 5. val023_train_1957_a_1 · finding 1

Large consolidation with air bronchograms in the left upper lobe

![2b val023_train_1957_a_1 · finding 1](full_body/val023_train_1957_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val023_train_1957_a_1_finding1_2b.png)

### 6. val023_train_1957_a_1 · finding 2

Consolidation in the posterobasal segment of the left lower lobe

![2b val023_train_1957_a_1 · finding 2](full_body/val023_train_1957_a_1_finding2_2b.png)

[Open full-resolution PNG](full_body/val023_train_1957_a_1_finding2_2b.png)

### 7. val023_train_1957_a_1 · finding 3

Consolidation in the right lung posterobasal segment

![2b val023_train_1957_a_1 · finding 3](full_body/val023_train_1957_a_1_finding3_2b.png)

[Open full-resolution PNG](full_body/val023_train_1957_a_1_finding3_2b.png)

### 8. val027_train_19456_a_1 · finding 0

Minimal atelectasis-like opacities in the dependent portions of the lungs

![2b val027_train_19456_a_1 · finding 0](full_body/val027_train_19456_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val027_train_19456_a_1_finding0_2b.png)

### 9. val029_train_2643_a_2 · finding 1

Areas of consolidation with air bronchograms in the lateral segment of the right middle lobe, right lower lobe, and mediobasal and posterobasal segments of the left lower lobe

![2b val029_train_2643_a_2 · finding 1](full_body/val029_train_2643_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val029_train_2643_a_2_finding1_2b.png)

### 10. val030_train_18414_a_2 · finding 0

Multilobar peripheral subpleural nodular consolidations in both lungs

![2b val030_train_18414_a_2 · finding 0](full_body/val030_train_18414_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val030_train_18414_a_2_finding0_2b.png)

</details>

<details>
<summary>Figures 11–20 of 49</summary>

### 11. val038_train_12991_a_1 · finding 1

Focal consolidations in the peripheral subpleural areas of the lower lobes of both lungs

![2b val038_train_12991_a_1 · finding 1](full_body/val038_train_12991_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val038_train_12991_a_1_finding1_2b.png)

### 12. val038_train_12991_a_1 · finding 4

Nodular consolidations in the basal segments of both lower lobes

![2b val038_train_12991_a_1 · finding 4](full_body/val038_train_12991_a_1_finding4_2b.png)

[Open full-resolution PNG](full_body/val038_train_12991_a_1_finding4_2b.png)

### 13. val044_train_19452_a_2 · finding 0

Increased parenchymal density in the right middle lobe

![2b val044_train_19452_a_2 · finding 0](full_body/val044_train_19452_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val044_train_19452_a_2_finding0_2b.png)

### 14. val051_train_13292_a_2 · finding 1

Atelectasis in both lungs adjacent to the pleural effusion

![2b val051_train_13292_a_2 · finding 1](full_body/val051_train_13292_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val051_train_13292_a_2_finding1_2b.png)

### 15. val057_train_19468_a_1 · finding 3

Subsegmental atelectasis in the right middle lobe and the lingula of the left upper lobe

![2b val057_train_19468_a_1 · finding 3](full_body/val057_train_19468_a_1_finding3_2b.png)

[Open full-resolution PNG](full_body/val057_train_19468_a_1_finding3_2b.png)

### 16. val058_train_13082_a_1 · finding 1

Focal consolidative density at the paramediastinal level of the right middle lobe

![2b val058_train_13082_a_1 · finding 1](full_body/val058_train_13082_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val058_train_13082_a_1_finding1_2b.png)

### 17. val060_train_2443_a_2 · finding 1

Atelectasis in the medial segment of the right middle lobe

![2b val060_train_2443_a_2 · finding 1](full_body/val060_train_2443_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val060_train_2443_a_2_finding1_2b.png)

### 18. val061_train_3028_a_2 · finding 1

Atelectasis in the apical portion of the left upper lobe

![2b val061_train_3028_a_2 · finding 1](full_body/val061_train_3028_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val061_train_3028_a_2_finding1_2b.png)

### 19. val062_train_13492_b_2 · finding 2

Areas of consolidation in the lower lobe of the left lung

![2b val062_train_13492_b_2 · finding 2](full_body/val062_train_13492_b_2_finding2_2b.png)

[Open full-resolution PNG](full_body/val062_train_13492_b_2_finding2_2b.png)

### 20. val066_train_19767_a_1 · finding 0

Mild atelectatic changes in the inferior lingula of the left upper lobe

![2b val066_train_19767_a_1 · finding 0](full_body/val066_train_19767_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val066_train_19767_a_1_finding0_2b.png)

</details>

<details>
<summary>Figures 21–30 of 49</summary>

### 21. val068_train_19180_a_2 · finding 0

Passive atelectasis in the medial segment of the right middle lobe and the inferior lingular segment of the left upper lobe

![2b val068_train_19180_a_2 · finding 0](full_body/val068_train_19180_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val068_train_19180_a_2_finding0_2b.png)

### 22. val074_train_19072_a_2 · finding 1

Diffuse subsegmental atelectasis in the medial segment of the right middle lobe, lingula of the left upper lobe, and basal segments of both lower lobes

![2b val074_train_19072_a_2 · finding 1](full_body/val074_train_19072_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val074_train_19072_a_2_finding1_2b.png)

### 23. val075_train_19545_a_2 · finding 0

Atelectasis in the medial segment of the right middle lobe

![2b val075_train_19545_a_2 · finding 0](full_body/val075_train_19545_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val075_train_19545_a_2_finding0_2b.png)

### 24. val075_train_19545_a_2 · finding 1

Atelectasis in the lingula of the left upper lobe

![2b val075_train_19545_a_2 · finding 1](full_body/val075_train_19545_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val075_train_19545_a_2_finding1_2b.png)

### 25. val077_train_13035_a_1 · finding 0

Subsegmental atelectasis in the bilateral lower lobe posterobasal segments

![2b val077_train_13035_a_1 · finding 0](full_body/val077_train_13035_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val077_train_13035_a_1_finding0_2b.png)

### 26. val077_train_13035_a_1 · finding 1

Dependent density increases in the bilateral lower lobe posterobasal segments

![2b val077_train_13035_a_1 · finding 1](full_body/val077_train_13035_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val077_train_13035_a_1_finding1_2b.png)

### 27. val085_train_18964_a_1 · finding 0

Consolidation with air bronchograms in the right lower lobe

![2b val085_train_18964_a_1 · finding 0](full_body/val085_train_18964_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val085_train_18964_a_1_finding0_2b.png)

### 28. val098_train_2617_b_2 · finding 0

Subsegmental atelectasis in the posterobasal segment of the left lower lobe

![2b val098_train_2617_b_2 · finding 0](full_body/val098_train_2617_b_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val098_train_2617_b_2_finding0_2b.png)

### 29. val107_train_19325_a_2 · finding 1

Mild adjacent focal atelectasis

![2b val107_train_19325_a_2 · finding 1](full_body/val107_train_19325_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val107_train_19325_a_2_finding1_2b.png)

### 30. val107_train_19325_a_2 · finding 3

Focal area of consolidation in the lingula

![2b val107_train_19325_a_2 · finding 3](full_body/val107_train_19325_a_2_finding3_2b.png)

[Open full-resolution PNG](full_body/val107_train_19325_a_2_finding3_2b.png)

</details>

<details>
<summary>Figures 31–40 of 49</summary>

### 31. val116_train_13119_c_1 · finding 2

Consolidations in the lower lobe of the left lung

![2b val116_train_13119_c_1 · finding 2](full_body/val116_train_13119_c_1_finding2_2b.png)

[Open full-resolution PNG](full_body/val116_train_13119_c_1_finding2_2b.png)

### 32. val117_train_13075_a_2 · finding 0

Atelectatic changes at the basal levels of both lower lobes

![2b val117_train_13075_a_2 · finding 0](full_body/val117_train_13075_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val117_train_13075_a_2_finding0_2b.png)

### 33. val124_train_18442_a_1 · finding 1

Areas of consolidation in the left lower lobe posterobasal segment, posterior segment of the upper lobe, right lower lobe posterobasal segment, and right lower lobe anterobasal segment

![2b val124_train_18442_a_1 · finding 1](full_body/val124_train_18442_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val124_train_18442_a_1_finding1_2b.png)

### 34. val129_train_18733_a_1 · finding 0

Large consolidation with air bronchograms involving the posterobasal, mediobasal, and laterobasal segments of the right lower lobe

![2b val129_train_18733_a_1 · finding 0](full_body/val129_train_18733_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val129_train_18733_a_1_finding0_2b.png)

### 35. val131_train_18403_a_2 · finding 0

Right middle lobe opacification with air bronchograms

![2b val131_train_18403_a_2 · finding 0](full_body/val131_train_18403_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val131_train_18403_a_2_finding0_2b.png)

### 36. val135_train_13591_a_1 · finding 1

Atelectatic changes in the middle lobe of the right lung

![2b val135_train_13591_a_1 · finding 1](full_body/val135_train_13591_a_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val135_train_13591_a_1_finding1_2b.png)

### 37. val144_train_1841_b_1 · finding 1

Focal atelectasis with ground-glass attenuation in the mediobasal segment of the right lower lobe adjacent to osteophytes

![2b val144_train_1841_b_1 · finding 1](full_body/val144_train_1841_b_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val144_train_1841_b_1_finding1_2b.png)

### 38. val151_train_13583_d_2 · finding 1

Extensive consolidation areas in all segments of both lungs

![2b val151_train_13583_d_2 · finding 1](full_body/val151_train_13583_d_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val151_train_13583_d_2_finding1_2b.png)

### 39. val151_train_13583_d_2 · finding 2

Consolidation areas showing more confluence

![2b val151_train_13583_d_2 · finding 2](full_body/val151_train_13583_d_2_finding2_2b.png)

[Open full-resolution PNG](full_body/val151_train_13583_d_2_finding2_2b.png)

### 40. val154_train_19028_a_2 · finding 0

Segmental consolidation in the right lower lobe lateral basal segment and at the junction of the lateral basal and posterior basal segments of the left lower lobe

![2b val154_train_19028_a_2 · finding 0](full_body/val154_train_19028_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val154_train_19028_a_2_finding0_2b.png)

</details>

<details>
<summary>Figures 41–49 of 49</summary>

### 41. val154_train_19028_a_2 · finding 1

Air bronchograms in the right lower lobe lateral basal segment and at the junction of the lateral basal and posterior basal segments of the left lower lobe

![2b val154_train_19028_a_2 · finding 1](full_body/val154_train_19028_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val154_train_19028_a_2_finding1_2b.png)

### 42. val155_train_19190_a_2 · finding 1

Irregular, locally linear areas of consolidation in the left lower lobe posterior basal and lateral basal segments, including the subpleural region of the posterior segment

![2b val155_train_19190_a_2 · finding 1](full_body/val155_train_19190_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val155_train_19190_a_2_finding1_2b.png)

### 43. val167_train_2672_a_1 · finding 0

Minimal atelectasis in the medial segment of the right middle lobe and inferior lingular segment of the left upper lobe

![2b val167_train_2672_a_1 · finding 0](full_body/val167_train_2672_a_1_finding0_2b.png)

[Open full-resolution PNG](full_body/val167_train_2672_a_1_finding0_2b.png)

### 44. val168_train_18399_b_2 · finding 0

Parenchymal opacities with air bronchograms in the left lower lobe mediobasal and laterobasal segments

![2b val168_train_18399_b_2 · finding 0](full_body/val168_train_18399_b_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val168_train_18399_b_2_finding0_2b.png)

### 45. val175_train_19757_a_2 · finding 0

Atelectasis in the posterobasal segment of the left lower lobe

![2b val175_train_19757_a_2 · finding 0](full_body/val175_train_19757_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val175_train_19757_a_2_finding0_2b.png)

### 46. val191_train_19209_a_2 · finding 1

Atelectasis in the inferior lingular segment of the left upper lobe and basal segments of the left lower lobe

![2b val191_train_19209_a_2 · finding 1](full_body/val191_train_19209_a_2_finding1_2b.png)

[Open full-resolution PNG](full_body/val191_train_19209_a_2_finding1_2b.png)

### 47. val195_train_2577_a_1 · finding 2

Consolidation in the left lower lobe

![2b val195_train_2577_a_1 · finding 2](full_body/val195_train_2577_a_1_finding2_2b.png)

[Open full-resolution PNG](full_body/val195_train_2577_a_1_finding2_2b.png)

### 48. val196_train_13479_e_1 · finding 1

Newly emerged consolidation area in the mediobasal segment of the right lower lobe

![2b val196_train_13479_e_1 · finding 1](full_body/val196_train_13479_e_1_finding1_2b.png)

[Open full-resolution PNG](full_body/val196_train_13479_e_1_finding1_2b.png)

### 49. val199_train_13177_a_2 · finding 0

Atelectatic changes causing mild structural distortion in the left lung inferior lingular segment

![2b val199_train_13177_a_2 · finding 0](full_body/val199_train_13177_a_2_finding0_2b.png)

[Open full-resolution PNG](full_body/val199_train_13177_a_2_finding0_2b.png)

</details>

