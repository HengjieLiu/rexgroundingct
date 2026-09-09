# 2c — Groundglass opacity — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 1507 | 19.60% |
| Val | 60 | 15.75% |
| Test | 87 | 14.95% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.414803 | 49/60 | 81.67% |
| Overall best | PP2 | 0.415889 | 49/60 | 81.67% |
| Overall best | PP3 (targeted corrections) | 0.411119 | 49/60 | 81.67% |
| Category best | Raw | 0.414803 | 49/60 | 81.67% |
| Category best | PP2 | 0.415889 | 49/60 | 81.67% |
| Category best | PP3 (targeted corrections) | 0.411119 | 49/60 | 81.67% |
| Iso07 best | Raw | 0.374194 | 49/60 | 81.67% |
| Iso07 best | PP2 | 0.377321 | 49/60 | 81.67% |
| Iso07 best | PP3 (targeted corrections) | 0.374770 | 49/60 | 81.67% |

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

## Figures (60)

<details>
<summary>Figures 1–10 of 60</summary>

### 1. val000_train_19753_a_2 · finding 1

Mild surrounding ground-glass in the lateral basal segment of the right lower lobe

![2c val000_train_19753_a_2 · finding 1](lung_only/val000_train_19753_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val000_train_19753_a_2_finding1_2c.png)

### 2. val002_train_19711_a_1 · finding 0

Peripheral patchy ground-glass opacity in the posterior right lower lobe adjacent to the costovertebral junction

![2c val002_train_19711_a_1 · finding 0](lung_only/val002_train_19711_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val002_train_19711_a_1_finding0_2c.png)

### 3. val008_train_13481_a_2 · finding 2

Areas of ground-glass opacity in the lung parenchyma

![2c val008_train_13481_a_2 · finding 2](lung_only/val008_train_13481_a_2_finding2_2c.png)

[Open full-resolution PNG](lung_only/val008_train_13481_a_2_finding2_2c.png)

### 4. val010_train_13256_b_2 · finding 0

Diffuse ground-glass opacities in all segments of both lungs

![2c val010_train_13256_b_2 · finding 0](lung_only/val010_train_13256_b_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val010_train_13256_b_2_finding0_2c.png)

### 5. val012_train_19348_a_2 · finding 0

Scattered bilateral subpleural ground-glass opacities

![2c val012_train_19348_a_2 · finding 0](lung_only/val012_train_19348_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val012_train_19348_a_2_finding0_2c.png)

### 6. val018_train_13431_a_1 · finding 0

Patchy ground-glass opacity in the apicoposterior segment of the left upper lobe

![2c val018_train_13431_a_1 · finding 0](lung_only/val018_train_13431_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val018_train_13431_a_1_finding0_2c.png)

### 7. val024_train_18382_b_2 · finding 1

Subcentimeter, minimal, nonspecific focal ground-glass opacities in the posterobasal segment of the right lower lobe and in the right middle lobe

![2c val024_train_18382_b_2 · finding 1](lung_only/val024_train_18382_b_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val024_train_18382_b_2_finding1_2c.png)

### 8. val030_train_18414_a_2 · finding 1

Peripheral subpleural ground-glass opacities in both lungs

![2c val030_train_18414_a_2 · finding 1](lung_only/val030_train_18414_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val030_train_18414_a_2_finding1_2c.png)

### 9. val036_train_2639_a_1 · finding 0

Peripheral/subpleural crazy paving opacities in both lungs involving the left lower lobe lateral basal segment, left upper lobe anterior segment, and right upper lobe posterior segment

![2c val036_train_2639_a_1 · finding 0](lung_only/val036_train_2639_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val036_train_2639_a_1_finding0_2c.png)

### 10. val038_train_12991_a_1 · finding 0

Ground-glass opacities in the peripheral subpleural areas of the lower lobes of both lungs

![2c val038_train_12991_a_1 · finding 0](lung_only/val038_train_12991_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val038_train_12991_a_1_finding0_2c.png)

</details>

<details>
<summary>Figures 11–20 of 60</summary>

### 11. val038_train_12991_a_1 · finding 2

Nodular ground-glass opacities in the middle lobe of the right lung

![2c val038_train_12991_a_1 · finding 2](lung_only/val038_train_12991_a_1_finding2_2c.png)

[Open full-resolution PNG](lung_only/val038_train_12991_a_1_finding2_2c.png)

### 12. val038_train_12991_a_1 · finding 3

Ground-glass opacities in the basal segments of both lower lobes

![2c val038_train_12991_a_1 · finding 3](lung_only/val038_train_12991_a_1_finding3_2c.png)

[Open full-resolution PNG](lung_only/val038_train_12991_a_1_finding3_2c.png)

### 13. val040_train_2589_a_2 · finding 0

Diffuse, predominantly subpleural, faintly marginated ground-glass opacities in both lungs

![2c val040_train_2589_a_2 · finding 0](lung_only/val040_train_2589_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val040_train_2589_a_2_finding0_2c.png)

### 14. val043_train_18967_a_2 · finding 0

Subpleural nodular ground-glass opacities in the lower lobes of both lungs

![2c val043_train_18967_a_2 · finding 0](lung_only/val043_train_18967_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val043_train_18967_a_2_finding0_2c.png)

### 15. val044_train_19452_a_2 · finding 1

Patchy ground-glass opacities in the anteromedial basal segment of the left lower lobe

![2c val044_train_19452_a_2 · finding 1](lung_only/val044_train_19452_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val044_train_19452_a_2_finding1_2c.png)

### 16. val049_train_18422_a_1 · finding 0

Subcentimeter ground-glass opacity in the posterobasal segment of the left lower lobe

![2c val049_train_18422_a_1 · finding 0](lung_only/val049_train_18422_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val049_train_18422_a_1_finding0_2c.png)

### 17. val059_train_2550_a_2 · finding 0

Consolidation and ground-glass opacities in the left lower lobe and right upper lobe posterior segment; some are nodular

![2c val059_train_2550_a_2 · finding 0](lung_only/val059_train_2550_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val059_train_2550_a_2_finding0_2c.png)

### 18. val062_train_13492_b_2 · finding 3

Ground-glass opacity in the lower lobe of the left lung

![2c val062_train_13492_b_2 · finding 3](lung_only/val062_train_13492_b_2_finding3_2c.png)

[Open full-resolution PNG](lung_only/val062_train_13492_b_2_finding3_2c.png)

### 19. val067_train_2532_a_2 · finding 0

Patchy nodular ground-glass opacities in the right lung, predominantly subpleural in the lower lobe

![2c val067_train_2532_a_2 · finding 0](lung_only/val067_train_2532_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val067_train_2532_a_2_finding0_2c.png)

### 20. val071_train_3008_a_1 · finding 0

Mild ground-glass opacities in the dependent basal segments of both lower lobes

![2c val071_train_3008_a_1 · finding 0](lung_only/val071_train_3008_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val071_train_3008_a_1_finding0_2c.png)

</details>

<details>
<summary>Figures 21–30 of 60</summary>

### 21. val072_train_13572_a_2 · finding 0

Nodular ground glass opacities in the paraspinal area of the right lower lobe superior segment

![2c val072_train_13572_a_2 · finding 0](lung_only/val072_train_13572_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val072_train_13572_a_2_finding0_2c.png)

### 22. val072_train_13572_a_2 · finding 1

Nodular ground glass opacities in the right lower lobe posterobasal segment

![2c val072_train_13572_a_2 · finding 1](lung_only/val072_train_13572_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val072_train_13572_a_2_finding1_2c.png)

### 23. val074_train_19072_a_2 · finding 0

Centrally and peripherally distributed crazy-paving in both lungs

![2c val074_train_19072_a_2 · finding 0](lung_only/val074_train_19072_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val074_train_19072_a_2_finding0_2c.png)

### 24. val082_train_13284_a_2 · finding 0

Increase in ground-glass opacity with septal thickening in the peripheral subpleural area of the upper and lower lobes of both lungs

![2c val082_train_13284_a_2 · finding 0](lung_only/val082_train_13284_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val082_train_13284_a_2_finding0_2c.png)

### 25. val082_train_13284_a_2 · finding 1

Crazy paving appearances in the upper and lower lobes of both lungs

![2c val082_train_13284_a_2 · finding 1](lung_only/val082_train_13284_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val082_train_13284_a_2_finding1_2c.png)

### 26. val086_train_2625_a_2 · finding 0

Nodular ground-glass opacity in the basal segment of the left lower lobe, 5 mm, irregular margins, too small to characterize

![2c val086_train_2625_a_2 · finding 0](lung_only/val086_train_2625_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val086_train_2625_a_2_finding0_2c.png)

### 27. val089_train_2657_a_2 · finding 0

Peripheral focal nodular ground-glass opacity in the superior segment of the right lower lobe

![2c val089_train_2657_a_2 · finding 0](lung_only/val089_train_2657_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val089_train_2657_a_2_finding0_2c.png)

### 28. val090_train_13098_a_2 · finding 0

Ground-glass opacities in all segments of both lungs

![2c val090_train_13098_a_2 · finding 0](lung_only/val090_train_13098_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val090_train_13098_a_2_finding0_2c.png)

### 29. val091_train_3008_b_2 · finding 0

Minimally dependent ground-glass opacities in the posterobasal segments of both lower lobes

![2c val091_train_3008_b_2 · finding 0](lung_only/val091_train_3008_b_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val091_train_3008_b_2_finding0_2c.png)

### 30. val095_train_13141_a_1 · finding 0

Diffuse patchy ground-glass opacities in both lungs

![2c val095_train_13141_a_1 · finding 0](lung_only/val095_train_13141_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val095_train_13141_a_1_finding0_2c.png)

</details>

<details>
<summary>Figures 31–40 of 60</summary>

### 31. val098_train_2617_b_2 · finding 2

Early pneumonic infiltration with centrilobular ground-glass opacities in the superior segment of the right lower lobe

![2c val098_train_2617_b_2 · finding 2](lung_only/val098_train_2617_b_2_finding2_2c.png)

[Open full-resolution PNG](lung_only/val098_train_2617_b_2_finding2_2c.png)

### 32. val100_train_2641_a_1 · finding 0

Mild dependent ground-glass opacities in the posterior basal segments of both lower lobes

![2c val100_train_2641_a_1 · finding 0](lung_only/val100_train_2641_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val100_train_2641_a_1_finding0_2c.png)

### 33. val107_train_19325_a_2 · finding 4

Faint ground-glass opacities in the posterobasal segment of the lower lobe

![2c val107_train_19325_a_2 · finding 4](lung_only/val107_train_19325_a_2_finding4_2c.png)

[Open full-resolution PNG](lung_only/val107_train_19325_a_2_finding4_2c.png)

### 34. val113_train_19279_c_1 · finding 1

Juxtadiaphragmatic ground-glass opacity in the right lower lobe decreased in size

![2c val113_train_19279_c_1 · finding 1](lung_only/val113_train_19279_c_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val113_train_19279_c_1_finding1_2c.png)

### 35. val115_train_2694_a_2 · finding 0

Diffuse bilateral patchy ground-glass opacities

![2c val115_train_2694_a_2 · finding 0](lung_only/val115_train_2694_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val115_train_2694_a_2_finding0_2c.png)

### 36. val124_train_18442_a_1 · finding 0

Multiple predominantly peripheral subpleural subcentimeter ground-glass opacities in the left lower lobe posterobasal segment, posterior segment of the upper lobe, right lower lobe posterobasal segment, and right lower lobe anterobasal segment

![2c val124_train_18442_a_1 · finding 0](lung_only/val124_train_18442_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val124_train_18442_a_1_finding0_2c.png)

### 37. val125_train_13577_a_2 · finding 0

Focal ground glass opacity in the laterobasal segment of the right lower lobe

![2c val125_train_13577_a_2 · finding 0](lung_only/val125_train_13577_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val125_train_13577_a_2_finding0_2c.png)

### 38. val129_train_18733_a_1 · finding 1

Surrounding ground-glass opacities adjacent to the right lower lobe consolidation

![2c val129_train_18733_a_1 · finding 1](lung_only/val129_train_18733_a_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val129_train_18733_a_1_finding1_2c.png)

### 39. val133_train_18379_a_1 · finding 0

Bilateral multifocal ground-glass opacities in peripheral subpleural and peribronchovascular regions of both lungs

![2c val133_train_18379_a_1 · finding 0](lung_only/val133_train_18379_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val133_train_18379_a_1_finding0_2c.png)

### 40. val134_train_13278_b_2 · finding 1

Ground-glass opacities in the laterobasal segment of the right middle and lower lobes

![2c val134_train_13278_b_2 · finding 1](lung_only/val134_train_13278_b_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val134_train_13278_b_2_finding1_2c.png)

</details>

<details>
<summary>Figures 41–50 of 60</summary>

### 41. val135_train_13591_a_1 · finding 0

Patchy ground glass densities in the inferior lingula of the left upper lobe

![2c val135_train_13591_a_1 · finding 0](lung_only/val135_train_13591_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val135_train_13591_a_1_finding0_2c.png)

### 42. val144_train_1841_b_1 · finding 0

Dependent increased attenuation and ground-glass opacities in both lower lobes

![2c val144_train_1841_b_1 · finding 0](lung_only/val144_train_1841_b_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val144_train_1841_b_1_finding0_2c.png)

### 43. val147_train_18416_a_1 · finding 0

Patchy ground-glass opacities in both lungs

![2c val147_train_18416_a_1 · finding 0](lung_only/val147_train_18416_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val147_train_18416_a_1_finding0_2c.png)

### 44. val148_train_2482_a_1 · finding 1

Linear ground-glass opacity in the right lung base

![2c val148_train_2482_a_1 · finding 1](lung_only/val148_train_2482_a_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val148_train_2482_a_1_finding1_2c.png)

### 45. val155_train_19190_a_2 · finding 0

Focal ground-glass opacities in the left lower lobe superior, anteromedial basal, and lateral basal segments

![2c val155_train_19190_a_2 · finding 0](lung_only/val155_train_19190_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val155_train_19190_a_2_finding0_2c.png)

### 46. val155_train_19190_a_2 · finding 2

Nodular ground-glass opacity in the paraspinal region of the right lower lobe mediobasal segment

![2c val155_train_19190_a_2 · finding 2](lung_only/val155_train_19190_a_2_finding2_2c.png)

[Open full-resolution PNG](lung_only/val155_train_19190_a_2_finding2_2c.png)

### 47. val161_train_2537_a_2 · finding 0

Multifocal ground-glass opacities in subpleural regions of the lower lobes bilaterally, predominantly centrilobular

![2c val161_train_2537_a_2 · finding 0](lung_only/val161_train_2537_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val161_train_2537_a_2_finding0_2c.png)

### 48. val161_train_2537_a_2 · finding 1

Multifocal ground-glass opacities in peribronchovascular regions of the upper lobes bilaterally, predominantly centrilobular

![2c val161_train_2537_a_2 · finding 1](lung_only/val161_train_2537_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val161_train_2537_a_2_finding1_2c.png)

### 49. val162_train_2645_a_1 · finding 0

Diffuse ground-glass opacities throughout both lungs, predominantly in the lower lobes and subpleural regions

![2c val162_train_2645_a_1 · finding 0](lung_only/val162_train_2645_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val162_train_2645_a_1_finding0_2c.png)

### 50. val169_train_2451_a_1 · finding 1

Adjacent ground-glass opacities in the superior segment of the left lower lobe

![2c val169_train_2451_a_1 · finding 1](lung_only/val169_train_2451_a_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val169_train_2451_a_1_finding1_2c.png)

</details>

<details>
<summary>Figures 51–60 of 60</summary>

### 51. val173_train_13189_a_1 · finding 2

Nonspecific ground glass opacities anterior to the sequestration in the lung parenchyma

![2c val173_train_13189_a_1 · finding 2](lung_only/val173_train_13189_a_1_finding2_2c.png)

[Open full-resolution PNG](lung_only/val173_train_13189_a_1_finding2_2c.png)

### 52. val174_train_2680_a_2 · finding 1

Crazy paving in the right middle lobe and in the superior and posterobasal segments of both lower lobes

![2c val174_train_2680_a_2 · finding 1](lung_only/val174_train_2680_a_2_finding1_2c.png)

[Open full-resolution PNG](lung_only/val174_train_2680_a_2_finding1_2c.png)

### 53. val178_train_18397_a_2 · finding 0

Patchy ground-glass opacities in both lower lobes, more prominent in the superior segments

![2c val178_train_18397_a_2 · finding 0](lung_only/val178_train_18397_a_2_finding0_2c.png)

[Open full-resolution PNG](lung_only/val178_train_18397_a_2_finding0_2c.png)

### 54. val181_train_19416_a_1 · finding 0

Patchy peripheral/subpleural ground-glass opacities in both lungs

![2c val181_train_19416_a_1 · finding 0](lung_only/val181_train_19416_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val181_train_19416_a_1_finding0_2c.png)

### 55. val184_train_19422_a_1 · finding 1

Patchy subpleural ground-glass opacities in both lungs

![2c val184_train_19422_a_1 · finding 1](lung_only/val184_train_19422_a_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val184_train_19422_a_1_finding1_2c.png)

### 56. val186_train_299_a_1 · finding 0

Peripheral and central ground-glass opacities in the upper lobes of both lungs, more prominent on the right

![2c val186_train_299_a_1 · finding 0](lung_only/val186_train_299_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val186_train_299_a_1_finding0_2c.png)

### 57. val187_train_13479_j_1 · finding 1

Minimal ground-glass opacities in the right lower lobe posterobasal segment

![2c val187_train_13479_j_1 · finding 1](lung_only/val187_train_13479_j_1_finding1_2c.png)

[Open full-resolution PNG](lung_only/val187_train_13479_j_1_finding1_2c.png)

### 58. val195_train_2577_a_1 · finding 0

Ground-glass opacities in both lungs, most prominent in basal segments of the upper lobes, right middle lobe, and lower lobes bilaterally, with areas of confluence

![2c val195_train_2577_a_1 · finding 0](lung_only/val195_train_2577_a_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val195_train_2577_a_1_finding0_2c.png)

### 59. val196_train_13479_e_1 · finding 0

Ground glass opacities observed

![2c val196_train_13479_e_1 · finding 0](lung_only/val196_train_13479_e_1_finding0_2c.png)

[Open full-resolution PNG](lung_only/val196_train_13479_e_1_finding0_2c.png)

### 60. val196_train_13479_e_1 · finding 2

Increase in ground glass density in the mediobasal segment of the right lower lobe

![2c val196_train_13479_e_1 · finding 2](lung_only/val196_train_13479_e_1_finding2_2c.png)

[Open full-resolution PNG](lung_only/val196_train_13479_e_1_finding2_2c.png)

</details>

