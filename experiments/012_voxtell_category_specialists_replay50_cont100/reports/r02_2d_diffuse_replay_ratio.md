# Exp012 Run 2 — 2d and Diffuse Replay-Ratio Progress

- Status: `selection_ready`
- Run group: `exp012_category_specialists_r02_replay_ratio_20260802T122657Z`
- Profile: `r02_2d_diffuse_replay_ratio`
- Exp012 umbrella status: `active` (additional runs are expected).
- Current stage: `reported_epoch100_val200`
- Updated: `2026-08-03T04:57:17.414697+00:00`
- Selection: target Dice only; sentinel and non-target results are diagnostic.

## Target Milestones

Cells show `Dice / hit rate (hits/findings)`.

| Arm | e0 | e5 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `category_2d_replay50` | 0.3785 / 0.856 (113/132) | 0.3585 / 0.856 (113/132) | 0.3637 / 0.811 (107/132) | 0.3445 / 0.818 (108/132) | 0.3683 / 0.826 (109/132) | 0.3633 / 0.833 (110/132) | 0.3638 / 0.833 (110/132) |
| `category_1alldiffuse_replay25` | 0.1627 / 0.365 (19/52) | 0.1620 / 0.423 (22/52) | 0.1435 / 0.365 (19/52) | 0.1497 / 0.346 (18/52) | 0.1683 / 0.423 (22/52) | 0.1598 / 0.423 (22/52) | 0.1575 / 0.442 (23/52) |
| `category_1alldiffuse_replay10` | 0.1627 / 0.365 (19/52) | 0.1401 / 0.365 (19/52) | 0.1600 / 0.404 (21/52) | 0.1452 / 0.404 (21/52) | 0.1562 / 0.404 (21/52) | 0.1531 / 0.404 (21/52) | 0.1443 / 0.404 (21/52) |
| `category_1alldiffuse_replay00` | 0.1627 / 0.365 (19/52) | 0.1294 / 0.404 (21/52) | 0.1297 / 0.365 (19/52) | 0.1444 / 0.385 (20/52) | 0.1549 / 0.385 (20/52) | 0.1484 / 0.404 (21/52) | 0.1610 / 0.423 (22/52) |

## Diffuse Replay-ratio Comparison

Run-1 replay50 is a frozen historical reference produced with an independent
schedule. It is not sample-paired with the three Run-2 diffuse arms.

| Arm | Schedule | e0 | e5 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `category_1alldiffuse_replay50` | historical independent | 0.1627 / 0.365 (19/52) | 0.1424 / 0.423 (22/52) | 0.1572 / 0.442 (23/52) | 0.1350 / 0.346 (18/52) | 0.1689 / 0.404 (21/52) | 0.1603 / 0.423 (22/52) | 0.1690 / 0.423 (22/52) |
| `category_1alldiffuse_replay25` | Run 2 independent | 0.1627 / 0.365 (19/52) | 0.1620 / 0.423 (22/52) | 0.1435 / 0.365 (19/52) | 0.1497 / 0.346 (18/52) | 0.1683 / 0.423 (22/52) | 0.1598 / 0.423 (22/52) | 0.1575 / 0.442 (23/52) |
| `category_1alldiffuse_replay10` | Run 2 independent | 0.1627 / 0.365 (19/52) | 0.1401 / 0.365 (19/52) | 0.1600 / 0.404 (21/52) | 0.1452 / 0.404 (21/52) | 0.1562 / 0.404 (21/52) | 0.1531 / 0.404 (21/52) | 0.1443 / 0.404 (21/52) |
| `category_1alldiffuse_replay00` | Run 2 independent | 0.1627 / 0.365 (19/52) | 0.1294 / 0.404 (21/52) | 0.1297 / 0.365 (19/52) | 0.1444 / 0.385 (20/52) | 0.1549 / 0.385 (20/52) | 0.1484 / 0.404 (21/52) | 0.1610 / 0.423 (22/52) |

## Val80 Total — Cross-arm Comparable

The same fixed 80 cases and 195 findings are used for every arm.

| Arm | e0 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- |
| `category_2d_replay50` | 0.3087 / 0.682 (133/195) | 0.2732 / 0.672 (131/195) | 0.2849 / 0.682 (133/195) | 0.2869 / 0.662 (129/195) | 0.3021 / 0.667 (130/195) | 0.2993 / 0.677 (132/195) |
| `category_1alldiffuse_replay25` | 0.3086 / 0.682 (133/195) | 0.2555 / 0.600 (117/195) | 0.2681 / 0.626 (122/195) | 0.2870 / 0.682 (133/195) | 0.2853 / 0.682 (133/195) | 0.3015 / 0.703 (137/195) |
| `category_1alldiffuse_replay10` | 0.3087 / 0.682 (133/195) | 0.2643 / 0.631 (123/195) | 0.2734 / 0.656 (128/195) | 0.2811 / 0.656 (128/195) | 0.2869 / 0.682 (133/195) | 0.2870 / 0.656 (128/195) |
| `category_1alldiffuse_replay00` | 0.3087 / 0.682 (133/195) | 0.2651 / 0.672 (131/195) | 0.2802 / 0.682 (133/195) | 0.2922 / 0.682 (133/195) | 0.2832 / 0.667 (130/195) | 0.2889 / 0.682 (133/195) |

## Val80 Non-target — Within-arm Forgetting

Each arm excludes its own target categories, so denominators differ and rows
should be compared across epochs within an arm, not across arms.

| Arm | e0 | e20 | e40 | e60 | e80 | e100 |
| --- | --- | --- | --- | --- | --- | --- |
| `category_2d_replay50` | 0.2919 / 0.639 (99/155) | 0.2509 / 0.632 (98/155) | 0.2763 / 0.652 (101/155) | 0.2662 / 0.626 (97/155) | 0.2866 / 0.632 (98/155) | 0.2827 / 0.639 (99/155) |
| `category_1alldiffuse_replay25` | 0.3617 / 0.797 (114/143) | 0.2962 / 0.685 (98/143) | 0.3112 / 0.727 (104/143) | 0.3302 / 0.776 (111/143) | 0.3309 / 0.776 (111/143) | 0.3539 / 0.797 (114/143) |
| `category_1alldiffuse_replay10` | 0.3617 / 0.797 (114/143) | 0.3022 / 0.713 (102/143) | 0.3200 / 0.748 (107/143) | 0.3265 / 0.748 (107/143) | 0.3356 / 0.783 (112/143) | 0.3389 / 0.748 (107/143) |
| `category_1alldiffuse_replay00` | 0.3617 / 0.797 (114/143) | 0.3144 / 0.783 (112/143) | 0.3296 / 0.790 (113/143) | 0.3421 / 0.790 (113/143) | 0.3323 / 0.762 (109/143) | 0.3354 / 0.776 (111/143) |

## Epoch-100 Full Val200

| Arm | Overall Dice | Overall hit | Non-target Dice | Non-target hit |
| --- | ---: | ---: | ---: | ---: |
| `category_2d_replay50` | 0.3316 | 0.748 | 0.3145 | 0.703 |
| `category_1alldiffuse_replay25` | 0.3308 | 0.764 | 0.3582 | 0.815 |
| `category_1alldiffuse_replay10` | 0.3200 | 0.732 | 0.3477 | 0.784 |
| `category_1alldiffuse_replay00` | 0.3178 | 0.751 | 0.3426 | 0.802 |

## Checkpoint Recommendation

| Arm | Selected epoch | Target Dice | Delta from e0 | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| `category_2d_replay50` | 0 | 0.378452 | 0.000000 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth` |
| `category_1alldiffuse_replay25` | 60 | 0.168278 | 0.005532 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_1alldiffuse_replay25/checkpoints/checkpoint_update_006000.pth` |
| `category_1alldiffuse_replay10` | 0 | 0.162745 | 0.000000 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth` |
| `category_1alldiffuse_replay00` | 0 | 0.162745 | 0.000000 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth` |

## Epoch-100 Per-category Metrics

### `category_2d_replay50`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.111980 | 0.333333 |
| 1b | 11 | 0.185143 | 0.454545 |
| 1c | 17 | 0.186726 | 0.411765 |
| 1d | 6 | 0.098061 | 0.166667 |
| 1e | 11 | 0.200413 | 0.636364 |
| 1f | 4 | 0.158805 | 0.250000 |
| 2a | 69 | 0.327124 | 0.811594 |
| 2b | 49 | 0.356041 | 0.755102 |
| 2c | 60 | 0.382030 | 0.800000 |
| 2d | 132 | 0.363793 | 0.833333 |
| 2e | 11 | 0.441049 | 0.727273 |
| 2g | 1 | 0.050710 | 0.000000 |
| 2h | 7 | 0.214882 | 0.571429 |

### `category_1alldiffuse_replay25`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.099041 | 0.666667 |
| 1b | 11 | 0.121415 | 0.363636 |
| 1c | 17 | 0.191395 | 0.470588 |
| 1d | 6 | 0.126720 | 0.166667 |
| 1e | 11 | 0.171909 | 0.636364 |
| 1f | 4 | 0.162791 | 0.250000 |
| 2a | 69 | 0.333239 | 0.811594 |
| 2b | 49 | 0.358727 | 0.755102 |
| 2c | 60 | 0.382636 | 0.833333 |
| 2d | 132 | 0.360115 | 0.840909 |
| 2e | 11 | 0.468498 | 0.818182 |
| 2g | 1 | 0.038706 | 0.000000 |
| 2h | 7 | 0.229484 | 0.714286 |

### `category_1alldiffuse_replay10`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.097488 | 0.333333 |
| 1b | 11 | 0.110896 | 0.272727 |
| 1c | 17 | 0.164779 | 0.470588 |
| 1d | 6 | 0.112835 | 0.166667 |
| 1e | 11 | 0.169926 | 0.636364 |
| 1f | 4 | 0.160870 | 0.250000 |
| 2a | 69 | 0.320189 | 0.797101 |
| 2b | 49 | 0.343660 | 0.734694 |
| 2c | 60 | 0.381554 | 0.800000 |
| 2d | 132 | 0.345439 | 0.803030 |
| 2e | 11 | 0.477586 | 0.818182 |
| 2g | 1 | 0.037885 | 0.000000 |
| 2h | 7 | 0.240892 | 0.571429 |

### `category_1alldiffuse_replay00`

| Category | Findings | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a | 3 | 0.099758 | 0.333333 |
| 1b | 11 | 0.119644 | 0.363636 |
| 1c | 17 | 0.199785 | 0.411765 |
| 1d | 6 | 0.120186 | 0.166667 |
| 1e | 11 | 0.180499 | 0.727273 |
| 1f | 4 | 0.162963 | 0.250000 |
| 2a | 69 | 0.313314 | 0.826087 |
| 2b | 49 | 0.357776 | 0.775510 |
| 2c | 60 | 0.370053 | 0.800000 |
| 2d | 132 | 0.338838 | 0.818182 |
| 2e | 11 | 0.464146 | 0.818182 |
| 2g | 1 | 0.012257 | 0.000000 |
| 2h | 7 | 0.218820 | 0.571429 |

## Pause Point

Training and epoch-100 val200 are complete. The pipeline has intentionally
stopped before evaluating any selected pre-100 checkpoint on val200.
