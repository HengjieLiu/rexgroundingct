# Experiment 004 Paired Comparison

Primary results use threshold 0.5 on the fixed val200 set.

| Model | Dice | Hit rate | Hits / findings |
| --- | ---: | ---: | ---: |
| exp003_v123_epoch100 | 0.2833 | 0.6903 | 263 / 381 |
| native192_cont100 | 0.2928 | 0.7060 | 269 / 381 |
| iso2mm_global192_cont100 | 0.1711 | 0.4199 | 160 / 381 |

## Category

| Stratum | Model | N | Dice | Hit rate |
| --- | --- | ---: | ---: | ---: |
| atelectasis | exp003_v123_epoch100 | 43 | 0.3077 | 0.8140 |
| atelectasis | native192_cont100 | 43 | 0.3185 | 0.8140 |
| atelectasis | iso2mm_global192_cont100 | 43 | 0.2140 | 0.5581 |
| consolidation | exp003_v123_epoch100 | 23 | 0.3791 | 0.7391 |
| consolidation | native192_cont100 | 23 | 0.3805 | 0.7391 |
| consolidation | iso2mm_global192_cont100 | 23 | 0.3576 | 0.6957 |
| cyst_cavity | exp003_v123_epoch100 | 6 | 0.0000 | 0.0000 |
| cyst_cavity | native192_cont100 | 6 | 0.0000 | 0.0000 |
| cyst_cavity | iso2mm_global192_cont100 | 6 | 0.0001 | 0.0000 |
| effusion | exp003_v123_epoch100 | 9 | 0.4922 | 0.7778 |
| effusion | native192_cont100 | 9 | 0.4959 | 0.7778 |
| effusion | iso2mm_global192_cont100 | 9 | 0.5057 | 0.7778 |
| ground_glass | exp003_v123_epoch100 | 48 | 0.3403 | 0.7708 |
| ground_glass | native192_cont100 | 48 | 0.3372 | 0.7917 |
| ground_glass | iso2mm_global192_cont100 | 48 | 0.2827 | 0.6667 |
| nodule | exp003_v123_epoch100 | 150 | 0.3059 | 0.7733 |
| nodule | native192_cont100 | 150 | 0.3224 | 0.7933 |
| nodule | iso2mm_global192_cont100 | 150 | 0.0904 | 0.2333 |
| other | exp003_v123_epoch100 | 102 | 0.1897 | 0.5000 |
| other | native192_cont100 | 102 | 0.1969 | 0.5196 |
| other | iso2mm_global192_cont100 | 102 | 0.1577 | 0.4510 |

## Native GT Size

| Stratum | Model | N | Dice | Hit rate |
| --- | --- | ---: | ---: | ---: |
| large_over_512_voxels | exp003_v123_epoch100 | 275 | 0.2924 | 0.7055 |
| large_over_512_voxels | native192_cont100 | 275 | 0.2984 | 0.7164 |
| large_over_512_voxels | iso2mm_global192_cont100 | 275 | 0.2219 | 0.5345 |
| medium_65_512_voxels | exp003_v123_epoch100 | 96 | 0.2651 | 0.6667 |
| medium_65_512_voxels | native192_cont100 | 96 | 0.2850 | 0.6979 |
| medium_65_512_voxels | iso2mm_global192_cont100 | 96 | 0.0434 | 0.1354 |
| small_9_64_voxels | exp003_v123_epoch100 | 10 | 0.2094 | 0.5000 |
| small_9_64_voxels | native192_cont100 | 10 | 0.2129 | 0.5000 |
| small_9_64_voxels | iso2mm_global192_cont100 | 10 | 0.0000 | 0.0000 |

## Sparse Mask

| Stratum | Model | N | Dice | Hit rate |
| --- | --- | ---: | ---: | ---: |
| False | exp003_v123_epoch100 | 371 | 0.2853 | 0.6954 |
| False | native192_cont100 | 371 | 0.2949 | 0.7116 |
| False | iso2mm_global192_cont100 | 371 | 0.1757 | 0.4313 |
| True | exp003_v123_epoch100 | 10 | 0.2094 | 0.5000 |
| True | native192_cont100 | 10 | 0.2129 | 0.5000 |
| True | iso2mm_global192_cont100 | 10 | 0.0000 | 0.0000 |

Size and sparse strata use native GT voxel counts: tiny 1-8, small 9-64, medium 65-512, large >512; sparse means <=64 voxels.
