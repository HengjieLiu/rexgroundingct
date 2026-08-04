# 2d — Pulmonary nodules/masses

[← Results overview](../../README.md) · [Case/category index](../category_case_index.csv) · [Finding metrics](../val200_finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 1743 | 22.67% |
| Validation | 132 | 34.65% |
| Test | 190 | 32.65% |

## Val200 model summary

A hit is a finding with 3D Dice greater than or equal to `0.1`.

| Order | Method | Mean Dice | Hits | Hit rate |
| --- | --- | --- | --- | --- |
| 1 | Public VoxTell v1.1 | 0.226553 | 70/132 | 53.03% |
| 2 | 100+100 attention | 0.367206 | 110/132 | 83.33% |
| 3 | 100+100 plain best no-DDP | 0.378417 | 113/132 | 85.61% |
| 4 | Best DDP | 0.394936 | 115/132 | 87.12% |

## Color Definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP, so slight AP over/under-segmentation still shows overlap.
> Purple marks rays where FN and FP both occur at different AP depths but no
> voxel overlaps.

| Color | Meaning |
| --- | --- |
| Green | Ray contains any real voxelwise TP, `gt & pred` |
| Red | No TP; ray contains FP only |
| Blue | No TP; ray contains FN only |
| Purple | No TP; ray contains depth-disjoint FN+FP |


## Figures (119)

Figures are sorted by fixed validation index. Each figure shows only
category 2d findings and uses radiology coronal orientation.

<details>
<summary>Figures 1–10 of 119</summary>

### 1. val004_train_18411_a_2

![2d val004_train_18411_a_2](val004_train_18411_a_2.png)

[Open full-resolution PNG](val004_train_18411_a_2.png)

### 2. val009_train_2458_a_2

![2d val009_train_2458_a_2](val009_train_2458_a_2.png)

[Open full-resolution PNG](val009_train_2458_a_2.png)

### 3. val011_train_19456_b_1

![2d val011_train_19456_b_1](val011_train_19456_b_1.png)

[Open full-resolution PNG](val011_train_19456_b_1.png)

### 4. val013_train_13020_a_2

![2d val013_train_13020_a_2](val013_train_13020_a_2.png)

[Open full-resolution PNG](val013_train_13020_a_2.png)

### 5. val014_train_2560_b_2

![2d val014_train_2560_b_2](val014_train_2560_b_2.png)

[Open full-resolution PNG](val014_train_2560_b_2.png)

### 6. val016_train_18729_a_1

![2d val016_train_18729_a_1](val016_train_18729_a_1.png)

[Open full-resolution PNG](val016_train_18729_a_1.png)

### 7. val021_train_3015_a_1

![2d val021_train_3015_a_1](val021_train_3015_a_1.png)

[Open full-resolution PNG](val021_train_3015_a_1.png)

### 8. val022_train_13440_a_1

![2d val022_train_13440_a_1](val022_train_13440_a_1.png)

[Open full-resolution PNG](val022_train_13440_a_1.png)

### 9. val024_train_18382_b_2

![2d val024_train_18382_b_2](val024_train_18382_b_2.png)

[Open full-resolution PNG](val024_train_18382_b_2.png)

### 10. val025_train_1998_a_1

![2d val025_train_1998_a_1](val025_train_1998_a_1.png)

[Open full-resolution PNG](val025_train_1998_a_1.png)

</details>

<details>
<summary>Figures 11–20 of 119</summary>

### 11. val026_train_18983_a_2

![2d val026_train_18983_a_2](val026_train_18983_a_2.png)

[Open full-resolution PNG](val026_train_18983_a_2.png)

### 12. val027_train_19456_a_1

![2d val027_train_19456_a_1](val027_train_19456_a_1.png)

[Open full-resolution PNG](val027_train_19456_a_1.png)

### 13. val031_train_18635_a_2

![2d val031_train_18635_a_2](val031_train_18635_a_2.png)

[Open full-resolution PNG](val031_train_18635_a_2.png)

### 14. val032_train_13301_a_2

![2d val032_train_13301_a_2](val032_train_13301_a_2.png)

[Open full-resolution PNG](val032_train_13301_a_2.png)

### 15. val033_train_19182_a_2

![2d val033_train_19182_a_2](val033_train_19182_a_2.png)

[Open full-resolution PNG](val033_train_19182_a_2.png)

### 16. val035_train_13076_a_1

![2d val035_train_13076_a_1](val035_train_13076_a_1.png)

[Open full-resolution PNG](val035_train_13076_a_1.png)

### 17. val037_train_19355_a_2

![2d val037_train_19355_a_2](val037_train_19355_a_2.png)

[Open full-resolution PNG](val037_train_19355_a_2.png)

### 18. val041_train_18969_a_2

![2d val041_train_18969_a_2](val041_train_18969_a_2.png)

[Open full-resolution PNG](val041_train_18969_a_2.png)

### 19. val042_train_13092_a_2

![2d val042_train_13092_a_2](val042_train_13092_a_2.png)

[Open full-resolution PNG](val042_train_13092_a_2.png)

### 20. val044_train_19452_a_2

![2d val044_train_19452_a_2](val044_train_19452_a_2.png)

[Open full-resolution PNG](val044_train_19452_a_2.png)

</details>

<details>
<summary>Figures 21–30 of 119</summary>

### 21. val045_train_13166_a_2

![2d val045_train_13166_a_2](val045_train_13166_a_2.png)

[Open full-resolution PNG](val045_train_13166_a_2.png)

### 22. val046_train_2477_a_2

![2d val046_train_2477_a_2](val046_train_2477_a_2.png)

[Open full-resolution PNG](val046_train_2477_a_2.png)

### 23. val047_train_13624_a_2

![2d val047_train_13624_a_2](val047_train_13624_a_2.png)

[Open full-resolution PNG](val047_train_13624_a_2.png)

### 24. val048_train_13430_a_2

![2d val048_train_13430_a_2](val048_train_13430_a_2.png)

[Open full-resolution PNG](val048_train_13430_a_2.png)

### 25. val050_train_19877_a_2

![2d val050_train_19877_a_2](val050_train_19877_a_2.png)

[Open full-resolution PNG](val050_train_19877_a_2.png)

### 26. val052_train_18421_b_1

![2d val052_train_18421_b_1](val052_train_18421_b_1.png)

[Open full-resolution PNG](val052_train_18421_b_1.png)

### 27. val053_train_18641_a_2

![2d val053_train_18641_a_2](val053_train_18641_a_2.png)

[Open full-resolution PNG](val053_train_18641_a_2.png)

### 28. val054_train_19443_b_2

![2d val054_train_19443_b_2](val054_train_19443_b_2.png)

[Open full-resolution PNG](val054_train_19443_b_2.png)

### 29. val055_train_2464_a_1

![2d val055_train_2464_a_1](val055_train_2464_a_1.png)

[Open full-resolution PNG](val055_train_2464_a_1.png)

### 30. val056_train_18662_a_1

![2d val056_train_18662_a_1](val056_train_18662_a_1.png)

[Open full-resolution PNG](val056_train_18662_a_1.png)

</details>

<details>
<summary>Figures 31–40 of 119</summary>

### 31. val057_train_19468_a_1

![2d val057_train_19468_a_1](val057_train_19468_a_1.png)

[Open full-resolution PNG](val057_train_19468_a_1.png)

### 32. val058_train_13082_a_1

![2d val058_train_13082_a_1](val058_train_13082_a_1.png)

[Open full-resolution PNG](val058_train_13082_a_1.png)

### 33. val060_train_2443_a_2

![2d val060_train_2443_a_2](val060_train_2443_a_2.png)

[Open full-resolution PNG](val060_train_2443_a_2.png)

### 34. val062_train_13492_b_2

![2d val062_train_13492_b_2](val062_train_13492_b_2.png)

[Open full-resolution PNG](val062_train_13492_b_2.png)

### 35. val063_train_18452_d_2

![2d val063_train_18452_d_2](val063_train_18452_d_2.png)

[Open full-resolution PNG](val063_train_18452_d_2.png)

### 36. val064_train_2646_a_1

![2d val064_train_2646_a_1](val064_train_2646_a_1.png)

[Open full-resolution PNG](val064_train_2646_a_1.png)

### 37. val065_train_18542_a_1

![2d val065_train_18542_a_1](val065_train_18542_a_1.png)

[Open full-resolution PNG](val065_train_18542_a_1.png)

### 38. val066_train_19767_a_1

![2d val066_train_19767_a_1](val066_train_19767_a_1.png)

[Open full-resolution PNG](val066_train_19767_a_1.png)

### 39. val069_train_18392_a_1

![2d val069_train_18392_a_1](val069_train_18392_a_1.png)

[Open full-resolution PNG](val069_train_18392_a_1.png)

### 40. val070_train_19397_a_1

![2d val070_train_19397_a_1](val070_train_19397_a_1.png)

[Open full-resolution PNG](val070_train_19397_a_1.png)

</details>

<details>
<summary>Figures 41–50 of 119</summary>

### 41. val073_train_19341_a_2

![2d val073_train_19341_a_2](val073_train_19341_a_2.png)

[Open full-resolution PNG](val073_train_19341_a_2.png)

### 42. val075_train_19545_a_2

![2d val075_train_19545_a_2](val075_train_19545_a_2.png)

[Open full-resolution PNG](val075_train_19545_a_2.png)

### 43. val076_train_2562_a_1

![2d val076_train_2562_a_1](val076_train_2562_a_1.png)

[Open full-resolution PNG](val076_train_2562_a_1.png)

### 44. val077_train_13035_a_1

![2d val077_train_13035_a_1](val077_train_13035_a_1.png)

[Open full-resolution PNG](val077_train_13035_a_1.png)

### 45. val078_train_2580_b_2

![2d val078_train_2580_b_2](val078_train_2580_b_2.png)

[Open full-resolution PNG](val078_train_2580_b_2.png)

### 46. val080_train_2724_a_1

![2d val080_train_2724_a_1](val080_train_2724_a_1.png)

[Open full-resolution PNG](val080_train_2724_a_1.png)

### 47. val083_train_2592_a_2

![2d val083_train_2592_a_2](val083_train_2592_a_2.png)

[Open full-resolution PNG](val083_train_2592_a_2.png)

### 48. val084_train_2659_a_2

![2d val084_train_2659_a_2](val084_train_2659_a_2.png)

[Open full-resolution PNG](val084_train_2659_a_2.png)

### 49. val087_train_18600_a_1

![2d val087_train_18600_a_1](val087_train_18600_a_1.png)

[Open full-resolution PNG](val087_train_18600_a_1.png)

### 50. val088_train_18976_a_2

![2d val088_train_18976_a_2](val088_train_18976_a_2.png)

[Open full-resolution PNG](val088_train_18976_a_2.png)

</details>

<details>
<summary>Figures 51–60 of 119</summary>

### 51. val092_train_2564_c_1

![2d val092_train_2564_c_1](val092_train_2564_c_1.png)

[Open full-resolution PNG](val092_train_2564_c_1.png)

### 52. val093_train_25_a_2

![2d val093_train_25_a_2](val093_train_25_a_2.png)

[Open full-resolution PNG](val093_train_25_a_2.png)

### 53. val094_train_2666_a_2

![2d val094_train_2666_a_2](val094_train_2666_a_2.png)

[Open full-resolution PNG](val094_train_2666_a_2.png)

### 54. val096_train_2148_a_2

![2d val096_train_2148_a_2](val096_train_2148_a_2.png)

[Open full-resolution PNG](val096_train_2148_a_2.png)

### 55. val097_train_2475_a_2

![2d val097_train_2475_a_2](val097_train_2475_a_2.png)

[Open full-resolution PNG](val097_train_2475_a_2.png)

### 56. val098_train_2617_b_2

![2d val098_train_2617_b_2](val098_train_2617_b_2.png)

[Open full-resolution PNG](val098_train_2617_b_2.png)

### 57. val099_train_2664_a_2

![2d val099_train_2664_a_2](val099_train_2664_a_2.png)

[Open full-resolution PNG](val099_train_2664_a_2.png)

### 58. val101_train_13398_a_1

![2d val101_train_13398_a_1](val101_train_13398_a_1.png)

[Open full-resolution PNG](val101_train_13398_a_1.png)

### 59. val102_train_13339_a_1

![2d val102_train_13339_a_1](val102_train_13339_a_1.png)

[Open full-resolution PNG](val102_train_13339_a_1.png)

### 60. val103_train_2677_a_2

![2d val103_train_2677_a_2](val103_train_2677_a_2.png)

[Open full-resolution PNG](val103_train_2677_a_2.png)

</details>

<details>
<summary>Figures 61–70 of 119</summary>

### 61. val104_train_18956_a_1

![2d val104_train_18956_a_1](val104_train_18956_a_1.png)

[Open full-resolution PNG](val104_train_18956_a_1.png)

### 62. val105_train_19032_a_2

![2d val105_train_19032_a_2](val105_train_19032_a_2.png)

[Open full-resolution PNG](val105_train_19032_a_2.png)

### 63. val108_train_19021_b_2

![2d val108_train_19021_b_2](val108_train_19021_b_2.png)

[Open full-resolution PNG](val108_train_19021_b_2.png)

### 64. val109_train_13113_a_1

![2d val109_train_13113_a_1](val109_train_13113_a_1.png)

[Open full-resolution PNG](val109_train_13113_a_1.png)

### 65. val110_train_13316_a_2

![2d val110_train_13316_a_2](val110_train_13316_a_2.png)

[Open full-resolution PNG](val110_train_13316_a_2.png)

### 66. val111_train_2560_d_2

![2d val111_train_2560_d_2](val111_train_2560_d_2.png)

[Open full-resolution PNG](val111_train_2560_d_2.png)

### 67. val113_train_19279_c_1

![2d val113_train_19279_c_1](val113_train_19279_c_1.png)

[Open full-resolution PNG](val113_train_19279_c_1.png)

### 68. val114_train_13155_a_2

![2d val114_train_13155_a_2](val114_train_13155_a_2.png)

[Open full-resolution PNG](val114_train_13155_a_2.png)

### 69. val116_train_13119_c_1

![2d val116_train_13119_c_1](val116_train_13119_c_1.png)

[Open full-resolution PNG](val116_train_13119_c_1.png)

### 70. val119_train_2603_a_1

![2d val119_train_2603_a_1](val119_train_2603_a_1.png)

[Open full-resolution PNG](val119_train_2603_a_1.png)

</details>

<details>
<summary>Figures 71–80 of 119</summary>

### 71. val121_train_13252_b_2

![2d val121_train_13252_b_2](val121_train_13252_b_2.png)

[Open full-resolution PNG](val121_train_13252_b_2.png)

### 72. val122_train_2455_a_1

![2d val122_train_2455_a_1](val122_train_2455_a_1.png)

[Open full-resolution PNG](val122_train_2455_a_1.png)

### 73. val126_train_18399_a_2

![2d val126_train_18399_a_2](val126_train_18399_a_2.png)

[Open full-resolution PNG](val126_train_18399_a_2.png)

### 74. val127_train_13444_a_2

![2d val127_train_13444_a_2](val127_train_13444_a_2.png)

[Open full-resolution PNG](val127_train_13444_a_2.png)

### 75. val128_train_2594_a_2

![2d val128_train_2594_a_2](val128_train_2594_a_2.png)

[Open full-resolution PNG](val128_train_2594_a_2.png)

### 76. val130_train_12994_a_1

![2d val130_train_12994_a_1](val130_train_12994_a_1.png)

[Open full-resolution PNG](val130_train_12994_a_1.png)

### 77. val132_train_18624_a_2

![2d val132_train_18624_a_2](val132_train_18624_a_2.png)

[Open full-resolution PNG](val132_train_18624_a_2.png)

### 78. val134_train_13278_b_2

![2d val134_train_13278_b_2](val134_train_13278_b_2.png)

[Open full-resolution PNG](val134_train_13278_b_2.png)

### 79. val136_train_18948_a_2

![2d val136_train_18948_a_2](val136_train_18948_a_2.png)

[Open full-resolution PNG](val136_train_18948_a_2.png)

### 80. val137_train_18597_b_1

![2d val137_train_18597_b_1](val137_train_18597_b_1.png)

[Open full-resolution PNG](val137_train_18597_b_1.png)

</details>

<details>
<summary>Figures 81–90 of 119</summary>

### 81. val138_train_18959_b_2

![2d val138_train_18959_b_2](val138_train_18959_b_2.png)

[Open full-resolution PNG](val138_train_18959_b_2.png)

### 82. val139_train_13417_d_1

![2d val139_train_13417_d_1](val139_train_13417_d_1.png)

[Open full-resolution PNG](val139_train_13417_d_1.png)

### 83. val140_train_18525_a_2

![2d val140_train_18525_a_2](val140_train_18525_a_2.png)

[Open full-resolution PNG](val140_train_18525_a_2.png)

### 84. val141_train_2549_a_2

![2d val141_train_2549_a_2](val141_train_2549_a_2.png)

[Open full-resolution PNG](val141_train_2549_a_2.png)

### 85. val142_train_2936_b_2

![2d val142_train_2936_b_2](val142_train_2936_b_2.png)

[Open full-resolution PNG](val142_train_2936_b_2.png)

### 86. val143_train_2696_a_2

![2d val143_train_2696_a_2](val143_train_2696_a_2.png)

[Open full-resolution PNG](val143_train_2696_a_2.png)

### 87. val144_train_1841_b_1

![2d val144_train_1841_b_1](val144_train_1841_b_1.png)

[Open full-resolution PNG](val144_train_1841_b_1.png)

### 88. val145_train_19743_c_2

![2d val145_train_19743_c_2](val145_train_19743_c_2.png)

[Open full-resolution PNG](val145_train_19743_c_2.png)

### 89. val146_train_13148_a_2

![2d val146_train_13148_a_2](val146_train_13148_a_2.png)

[Open full-resolution PNG](val146_train_13148_a_2.png)

### 90. val148_train_2482_a_1

![2d val148_train_2482_a_1](val148_train_2482_a_1.png)

[Open full-resolution PNG](val148_train_2482_a_1.png)

</details>

<details>
<summary>Figures 91–100 of 119</summary>

### 91. val149_train_19923_a_2

![2d val149_train_19923_a_2](val149_train_19923_a_2.png)

[Open full-resolution PNG](val149_train_19923_a_2.png)

### 92. val150_train_2468_a_2

![2d val150_train_2468_a_2](val150_train_2468_a_2.png)

[Open full-resolution PNG](val150_train_2468_a_2.png)

### 93. val152_train_13447_a_1

![2d val152_train_13447_a_1](val152_train_13447_a_1.png)

[Open full-resolution PNG](val152_train_13447_a_1.png)

### 94. val153_train_2530_a_2

![2d val153_train_2530_a_2](val153_train_2530_a_2.png)

[Open full-resolution PNG](val153_train_2530_a_2.png)

### 95. val156_train_18721_a_2

![2d val156_train_18721_a_2](val156_train_18721_a_2.png)

[Open full-resolution PNG](val156_train_18721_a_2.png)

### 96. val157_train_2515_b_2

![2d val157_train_2515_b_2](val157_train_2515_b_2.png)

[Open full-resolution PNG](val157_train_2515_b_2.png)

### 97. val158_train_2493_a_1

![2d val158_train_2493_a_1](val158_train_2493_a_1.png)

[Open full-resolution PNG](val158_train_2493_a_1.png)

### 98. val159_train_13631_a_1

![2d val159_train_13631_a_1](val159_train_13631_a_1.png)

[Open full-resolution PNG](val159_train_13631_a_1.png)

### 99. val160_train_19459_a_2

![2d val160_train_19459_a_2](val160_train_19459_a_2.png)

[Open full-resolution PNG](val160_train_19459_a_2.png)

### 100. val164_train_13426_a_2

![2d val164_train_13426_a_2](val164_train_13426_a_2.png)

[Open full-resolution PNG](val164_train_13426_a_2.png)

</details>

<details>
<summary>Figures 101–110 of 119</summary>

### 101. val165_train_19315_a_2

![2d val165_train_19315_a_2](val165_train_19315_a_2.png)

[Open full-resolution PNG](val165_train_19315_a_2.png)

### 102. val170_train_19771_a_2

![2d val170_train_19771_a_2](val170_train_19771_a_2.png)

[Open full-resolution PNG](val170_train_19771_a_2.png)

### 103. val171_train_2564_b_2

![2d val171_train_2564_b_2](val171_train_2564_b_2.png)

[Open full-resolution PNG](val171_train_2564_b_2.png)

### 104. val173_train_13189_a_1

![2d val173_train_13189_a_1](val173_train_13189_a_1.png)

[Open full-resolution PNG](val173_train_13189_a_1.png)

### 105. val174_train_2680_a_2

![2d val174_train_2680_a_2](val174_train_2680_a_2.png)

[Open full-resolution PNG](val174_train_2680_a_2.png)

### 106. val176_train_205_a_1

![2d val176_train_205_a_1](val176_train_205_a_1.png)

[Open full-resolution PNG](val176_train_205_a_1.png)

### 107. val177_train_2625_b_2

![2d val177_train_2625_b_2](val177_train_2625_b_2.png)

[Open full-resolution PNG](val177_train_2625_b_2.png)

### 108. val180_train_19346_a_2

![2d val180_train_19346_a_2](val180_train_19346_a_2.png)

[Open full-resolution PNG](val180_train_19346_a_2.png)

### 109. val182_train_18454_a_2

![2d val182_train_18454_a_2](val182_train_18454_a_2.png)

[Open full-resolution PNG](val182_train_18454_a_2.png)

### 110. val183_train_13013_a_1

![2d val183_train_13013_a_1](val183_train_13013_a_1.png)

[Open full-resolution PNG](val183_train_13013_a_1.png)

</details>

<details>
<summary>Figures 111–119 of 119</summary>

### 111. val184_train_19422_a_1

![2d val184_train_19422_a_1](val184_train_19422_a_1.png)

[Open full-resolution PNG](val184_train_19422_a_1.png)

### 112. val185_train_3006_a_2

![2d val185_train_3006_a_2](val185_train_3006_a_2.png)

[Open full-resolution PNG](val185_train_3006_a_2.png)

### 113. val190_train_18569_a_2

![2d val190_train_18569_a_2](val190_train_18569_a_2.png)

[Open full-resolution PNG](val190_train_18569_a_2.png)

### 114. val191_train_19209_a_2

![2d val191_train_19209_a_2](val191_train_19209_a_2.png)

[Open full-resolution PNG](val191_train_19209_a_2.png)

### 115. val192_train_2995_a_2

![2d val192_train_2995_a_2](val192_train_2995_a_2.png)

[Open full-resolution PNG](val192_train_2995_a_2.png)

### 116. val193_train_19465_a_1

![2d val193_train_19465_a_1](val193_train_19465_a_1.png)

[Open full-resolution PNG](val193_train_19465_a_1.png)

### 117. val194_train_2568_a_5

![2d val194_train_2568_a_5](val194_train_2568_a_5.png)

[Open full-resolution PNG](val194_train_2568_a_5.png)

### 118. val197_train_18597_a_1

![2d val197_train_18597_a_1](val197_train_18597_a_1.png)

[Open full-resolution PNG](val197_train_18597_a_1.png)

### 119. val198_train_3026_b_2

![2d val198_train_3026_b_2](val198_train_3026_b_2.png)

[Open full-resolution PNG](val198_train_3026_b_2.png)

</details>
