# Exp012 Category-Specialist Progress

- Status: `selection_ready`
- Run group: `exp012_category_specialists_20260801T215219Z`
- Current stage: `reported_epoch100_val200`
- Updated: `2026-08-02T12:26:44.329179+00:00`
- Selection: target Dice only; sentinel and non-target results are diagnostic.

## Target Milestones

Cells show `Dice / hit rate (hits/findings)`.

| Arm | e0 | e5 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `category_1alldiffuse_replay50` | 0.1627 / 0.365 (19/52) | 0.1424 / 0.423 (22/52) | 0.1572 / 0.442 (23/52) | 0.1350 / 0.346 (18/52) | 0.1689 / 0.404 (21/52) | 0.1603 / 0.423 (22/52) | 0.1690 / 0.423 (22/52) |
| `category_2a_replay50` | 0.3405 / 0.826 (57/69) | 0.2964 / 0.783 (54/69) | 0.3272 / 0.841 (58/69) | 0.3262 / 0.841 (58/69) | 0.3175 / 0.812 (56/69) | 0.3296 / 0.812 (56/69) | 0.3334 / 0.797 (55/69) |
| `category_2b_replay50` | 0.3569 / 0.755 (37/49) | 0.3333 / 0.714 (35/49) | 0.3271 / 0.735 (36/49) | 0.3555 / 0.776 (38/49) | 0.3488 / 0.776 (38/49) | 0.3044 / 0.776 (38/49) | 0.3495 / 0.796 (39/49) |
| `category_2c_replay50` | 0.3908 / 0.833 (50/60) | 0.3547 / 0.800 (48/60) | 0.3517 / 0.850 (51/60) | 0.3433 / 0.817 (49/60) | 0.3911 / 0.833 (50/60) | 0.3616 / 0.817 (49/60) | 0.3904 / 0.850 (51/60) |

## Val80 Total — Cross-arm Comparable

The same fixed 80 cases and 195 findings are used for every arm.

| Arm | e0 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- |
| `category_1alldiffuse_replay50` | 0.3087 / 0.682 (133/195) | 0.2847 / 0.703 (137/195) | 0.2824 / 0.662 (129/195) | 0.2921 / 0.682 (133/195) | 0.2961 / 0.677 (132/195) | 0.2960 / 0.667 (130/195) |
| `category_2a_replay50` | 0.3087 / 0.682 (133/195) | 0.2804 / 0.682 (133/195) | 0.2723 / 0.667 (130/195) | 0.2913 / 0.667 (130/195) | 0.2977 / 0.677 (132/195) | 0.2999 / 0.677 (132/195) |
| `category_2b_replay50` | 0.3087 / 0.682 (133/195) | 0.2735 / 0.641 (125/195) | 0.2848 / 0.667 (130/195) | 0.2761 / 0.667 (130/195) | 0.2943 / 0.697 (136/195) | 0.2948 / 0.692 (135/195) |
| `category_2c_replay50` | 0.3087 / 0.682 (133/195) | 0.2834 / 0.677 (132/195) | 0.2868 / 0.672 (131/195) | 0.3004 / 0.692 (135/195) | 0.2985 / 0.667 (130/195) | 0.2958 / 0.672 (131/195) |

## Val80 Non-target — Within-arm Forgetting

Each arm excludes its own target categories, so denominators differ and rows
should be compared across epochs within an arm, not across arms.

| Arm | e0 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- |
| `category_1alldiffuse_replay50` | 0.3617 / 0.797 (114/143) | 0.3311 / 0.797 (114/143) | 0.3360 / 0.776 (111/143) | 0.3369 / 0.783 (112/143) | 0.3456 / 0.769 (110/143) | 0.3421 / 0.755 (108/143) |
| `category_2a_replay50` | 0.3043 / 0.652 (107/164) | 0.2750 / 0.652 (107/164) | 0.2654 / 0.634 (104/164) | 0.2873 / 0.640 (105/164) | 0.2908 / 0.652 (107/164) | 0.2936 / 0.652 (107/164) |
| `category_2b_replay50` | 0.3017 / 0.671 (112/167) | 0.2644 / 0.629 (105/167) | 0.2735 / 0.647 (108/167) | 0.2629 / 0.647 (108/167) | 0.2906 / 0.683 (114/167) | 0.2854 / 0.671 (112/167) |
| `category_2c_replay50` | 0.2950 / 0.659 (112/170) | 0.2710 / 0.653 (111/170) | 0.2753 / 0.653 (111/170) | 0.2886 / 0.676 (115/170) | 0.2889 / 0.653 (111/170) | 0.2809 / 0.647 (110/170) |

## Epoch-100 Full Val200

| Arm | Overall Dice | Overall hit | Non-target Dice | Non-target hit |
| --- | ---: | ---: | ---: | ---: |
| `category_1alldiffuse_replay50` | 0.3288 | 0.743 | 0.3541 | 0.793 |
| `category_2a_replay50` | 0.3333 | 0.748 | 0.3333 | 0.737 |
| `category_2b_replay50` | 0.3266 | 0.753 | 0.3233 | 0.747 |
| `category_2c_replay50` | 0.3272 | 0.748 | 0.3154 | 0.729 |

## Checkpoint Recommendation

| Arm | Selected epoch | Target Dice | Delta from e0 | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| `category_1alldiffuse_replay50` | 100 | 0.169007 | 0.006257 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_1alldiffuse_replay50/checkpoints/checkpoint_update_010000.pth` |
| `category_2a_replay50` | 0 | 0.340525 | 0.000000 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth` |
| `category_2b_replay50` | 0 | 0.356857 | 0.000000 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth` |
| `category_2c_replay50` | 60 | 0.391079 | 0.000259 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_2c_replay50/checkpoints/checkpoint_update_006000.pth` |

## Epoch-100 Per-category Metrics

### `category_1alldiffuse_replay50`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.100723 | 0.333333 |
| 1b | 11 | 0.143308 | 0.363636 |
| 1c | 17 | 0.201143 | 0.411765 |
| 1d | 6 | 0.118809 | 0.166667 |
| 1e | 11 | 0.196131 | 0.727273 |
| 1f | 4 | 0.155015 | 0.250000 |
| 2a | 69 | 0.317579 | 0.797101 |
| 2b | 49 | 0.350254 | 0.714286 |
| 2c | 60 | 0.385131 | 0.816667 |
| 2d | 132 | 0.362938 | 0.833333 |
| 2e | 11 | 0.416131 | 0.727273 |
| 2g | 1 | 0.070390 | 0.000000 |
| 2h | 7 | 0.249585 | 0.571429 |

### `category_2a_replay50`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.114691 | 0.333333 |
| 1b | 11 | 0.141459 | 0.363636 |
| 1c | 17 | 0.184805 | 0.411765 |
| 1d | 6 | 0.119111 | 0.166667 |
| 1e | 11 | 0.171411 | 0.545455 |
| 1f | 4 | 0.159544 | 0.250000 |
| 2a | 69 | 0.333357 | 0.797101 |
| 2b | 49 | 0.356099 | 0.755102 |
| 2c | 60 | 0.396348 | 0.800000 |
| 2d | 132 | 0.367895 | 0.863636 |
| 2e | 11 | 0.417570 | 0.727273 |
| 2g | 1 | 0.041680 | 0.000000 |
| 2h | 7 | 0.181462 | 0.428571 |

### `category_2b_replay50`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.106424 | 0.333333 |
| 1b | 11 | 0.139171 | 0.363636 |
| 1c | 17 | 0.181400 | 0.529412 |
| 1d | 6 | 0.134926 | 0.166667 |
| 1e | 11 | 0.172638 | 0.636364 |
| 1f | 4 | 0.160637 | 0.250000 |
| 2a | 69 | 0.332225 | 0.826087 |
| 2b | 49 | 0.349522 | 0.795918 |
| 2c | 60 | 0.394157 | 0.816667 |
| 2d | 132 | 0.351100 | 0.810606 |
| 2e | 11 | 0.443557 | 0.727273 |
| 2g | 1 | 0.093038 | 0.000000 |
| 2h | 7 | 0.163912 | 0.571429 |

### `category_2c_replay50`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.121726 | 0.333333 |
| 1b | 11 | 0.154408 | 0.363636 |
| 1c | 17 | 0.184455 | 0.411765 |
| 1d | 6 | 0.108688 | 0.166667 |
| 1e | 11 | 0.168731 | 0.545455 |
| 1f | 4 | 0.229668 | 0.500000 |
| 2a | 69 | 0.323329 | 0.826087 |
| 2b | 49 | 0.358339 | 0.755102 |
| 2c | 60 | 0.390397 | 0.850000 |
| 2d | 132 | 0.356942 | 0.818182 |
| 2e | 11 | 0.424061 | 0.818182 |
| 2g | 1 | 0.089459 | 0.000000 |
| 2h | 7 | 0.126960 | 0.285714 |

## Pause Point

Training and epoch-100 val200 are complete. The pipeline has intentionally
stopped before evaluating any selected pre-100 checkpoint on val200.
