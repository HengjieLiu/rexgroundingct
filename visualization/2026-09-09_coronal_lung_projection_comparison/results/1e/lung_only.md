# 1e — Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 314 | 4.08% |
| Val | 11 | 2.89% |
| Test | 16 | 2.75% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.194224 | 7/11 | 63.64% |
| Overall best | PP2 | 0.196488 | 7/11 | 63.64% |
| Overall best | PP3 (targeted corrections) | 0.195433 | 7/11 | 63.64% |
| Category best | Raw | 0.200413 | 7/11 | 63.64% |
| Category best | PP2 | 0.206401 | 7/11 | 63.64% |
| Category best | PP3 (targeted corrections) | 0.202517 | 7/11 | 63.64% |
| Iso07 best | Raw | 0.155368 | 6/11 | 54.55% |
| Iso07 best | PP2 | 0.161317 | 6/11 | 54.55% |
| Iso07 best | PP3 (targeted corrections) | 0.155557 | 6/11 | 54.55% |

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

### 1. val000_train_19753_a_2 · finding 0

Centrilobular nodular opacities in the lateral basal segment of the right lower lobe

![1e val000_train_19753_a_2 · finding 0](lung_only/val000_train_19753_a_2_finding0_1e.png)

[Open full-resolution PNG](lung_only/val000_train_19753_a_2_finding0_1e.png)

### 2. val013_train_13020_a_2 · finding 1

Subcentimeter pure calcific nonspecific nodules in both lung parenchyma

![1e val013_train_13020_a_2 · finding 1](lung_only/val013_train_13020_a_2_finding1_1e.png)

[Open full-resolution PNG](lung_only/val013_train_13020_a_2_finding1_1e.png)

### 3. val039_train_2509_b_1 · finding 0

Residual opacities with a tree-in-bud appearance in the right upper lobe, right middle lobe, superior segment of the right lower lobe, and superior segment of the left lower lobe

![1e val039_train_2509_b_1 · finding 0](lung_only/val039_train_2509_b_1_finding0_1e.png)

[Open full-resolution PNG](lung_only/val039_train_2509_b_1_finding0_1e.png)

### 4. val042_train_13092_a_2 · finding 1

Numerous calcified subcentimeter nodules in both lungs

![1e val042_train_13092_a_2 · finding 1](lung_only/val042_train_13092_a_2_finding1_1e.png)

[Open full-resolution PNG](lung_only/val042_train_13092_a_2_finding1_1e.png)

### 5. val045_train_13166_a_2 · finding 0

Faintly circumscribed, poorly distinguishable centriacinar nodular opacities in the lower lobes of both lungs

![1e val045_train_13166_a_2 · finding 0](lung_only/val045_train_13166_a_2_finding0_1e.png)

[Open full-resolution PNG](lung_only/val045_train_13166_a_2_finding0_1e.png)

### 6. val051_train_13292_a_2 · finding 2

Subcentimeter centriacinar nodules in the posterior segment of the right upper lobe

![1e val051_train_13292_a_2 · finding 2](lung_only/val051_train_13292_a_2_finding2_1e.png)

[Open full-resolution PNG](lung_only/val051_train_13292_a_2_finding2_1e.png)

### 7. val128_train_2594_a_2 · finding 1

Several subcentimeter centrilobular nodules in the lateral segment of the right middle lobe

![1e val128_train_2594_a_2 · finding 1](lung_only/val128_train_2594_a_2_finding1_1e.png)

[Open full-resolution PNG](lung_only/val128_train_2594_a_2_finding1_1e.png)

### 8. val130_train_12994_a_1 · finding 0

A few subcentimeter nonspecific nodules in both lung parenchyma

![1e val130_train_12994_a_1 · finding 0](lung_only/val130_train_12994_a_1_finding0_1e.png)

[Open full-resolution PNG](lung_only/val130_train_12994_a_1_finding0_1e.png)

### 9. val155_train_19190_a_2 · finding 3

Tree-in-bud appearance

![1e val155_train_19190_a_2 · finding 3](lung_only/val155_train_19190_a_2_finding3_1e.png)

[Open full-resolution PNG](lung_only/val155_train_19190_a_2_finding3_1e.png)

### 10. val172_train_13249_b_2 · finding 0

Centriacinar nodular infiltration in the peripheral subpleural areas

![1e val172_train_13249_b_2 · finding 0](lung_only/val172_train_13249_b_2_finding0_1e.png)

[Open full-resolution PNG](lung_only/val172_train_13249_b_2_finding0_1e.png)

</details>

<details>
<summary>Figures 11–11 of 11</summary>

### 11. val183_train_13013_a_1 · finding 2

Subcentimeter nodules in both lungs

![1e val183_train_13013_a_1 · finding 2](lung_only/val183_train_13013_a_1_finding2_1e.png)

[Open full-resolution PNG](lung_only/val183_train_13013_a_1_finding2_1e.png)

</details>

