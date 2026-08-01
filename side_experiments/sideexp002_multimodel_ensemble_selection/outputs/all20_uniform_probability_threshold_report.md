# Side Experiment 002 All-20 Probability Ensemble Threshold Report

This post-hoc val200 probe applies sigmoid to each stored model logit, uniformly averages all 20 probabilities, and thresholds that mean.

> **Acceptance override:** all 20 completed exports are used. Seven have failed strict reproduction records; their stored arrays passed recorded hash verification and structural checks, but their gate warnings remain visible below.

> **Rare-category warning:** category `2g` has one finding, and `2f` has no findings. Category-specific winners are descriptive validation oracles, not generalizable threshold policies.

## Executive Summary

- Best single global threshold: `0.55`
- Global Dice: `0.344846`
- Global hits: `290 / 381` (`0.761155`)
- Per-category oracle Dice: `0.354427` (gain `+0.009580`)
- Per-category oracle hits: `292 / 381` (gain `+2`)
- Categories selecting the global threshold: `0 / 13`
- Mean category Dice loss from one global threshold: `0.025333`
- Maximum category Dice loss: `0.126872` for `2g`

## Ensemble Roster

| Candidate | Source | Model / arm | Epoch | Architecture | Dtype | Weight | Strict gate | Warning reason |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| `exp006_e5d4_e100` | Exp006 | `v123_cached_e5_d4` | 100 | standard | float32 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed; historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp006_e6d4_e100` | Exp006 | `v123_cached_e6_d4` | 100 | standard | float16 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed |
| `exp007_ddp_e050` | Exp007 | `ddp_bs4` | 50 | standard | float32 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed; historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp007_ddp_e075` | Exp007 | `ddp_bs4` | 75 | standard | float16 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed |
| `exp007_ddp_e100` | Exp007 | `ddp_bs4` | 100 | standard | float32 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed; historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp008_dual_e080` | Exp008 | `v1_dualfusion_softguide` | 80 | dual_branch | float16 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed |
| `exp008_dual_e100` | Exp008 | `v1_dualfusion_softguide` | 100 | dual_branch | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp008_joint_e080` | Exp008 | `v3_dualfusion_softguide_joint` | 80 | dual_branch | float32 | 0.050000 | bypassed | status=failed; storage_reproduction_status=failed; same_pass_mask_reproduction_status=failed; historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp008_joint_e100` | Exp008 | `v3_dualfusion_softguide_joint` | 100 | dual_branch | float16 | 0.050000 | accepted | - |
| `exp008_precision_e080` | Exp008 | `v2_dualfusion_precision` | 80 | dual_branch | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp008_precision_e100` | Exp008 | `v2_dualfusion_precision` | 100 | dual_branch | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp008_shared_e080` | Exp008 | `v1_sharedfusion_softguide` | 80 | dual_branch | float16 | 0.050000 | accepted | - |
| `exp008_shared_e100` | Exp008 | `v1_sharedfusion_softguide` | 100 | dual_branch | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp009_baseline_e100` | Exp009 | `baseline_cont100` | 100 | standard | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp009_s3v1_e100` | Exp009 | `s3v1_fixedrho_suppress_half_quarter` | 100 | s3_attention | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp009_s3v2_e100` | Exp009 | `s3v2_balanced_feature_half_quarter` | 100 | s3_attention | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp009_s3v3_e100` | Exp009 | `s3v3_logit_residual_half_quarter` | 100 | s3_attention | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |
| `exp011_clip_zscore_e100` | Exp011 | `v123_e5d4_clip1024_zscore` | 100 | standard | float16 | 0.050000 | accepted | - |
| `exp011_linear_hu_e100` | Exp011 | `v123_e5d4_clip1024_linear` | 100 | standard | float16 | 0.050000 | accepted | - |
| `exp011_native_zscore_e100` | Exp011 | `v123_e5d4_zscore` | 100 | standard | float16 | 0.050000 | accepted | historical_reproduction_status=warning; historical_reproduction_mismatch |

## Category Support

| Code | Official category | Findings | Interpretation |
| --- | --- | ---: | --- |
| 1a | Bronchial wall thickening | 3 | Very rare; descriptive only |
| 1b | Bronchiectasis | 11 | Low support |
| 1c | Emphysema | 17 | Low support |
| 1d | Septal thickening / reticulation | 6 | Very rare; descriptive only |
| 1e | Micronodules / tree-in-bud | 11 | Low support |
| 1f | Other diffuse lung/airway/pleural abnormality | 4 | Very rare; descriptive only |
| 2a | Linear opacity, scarring, fibrosis | 69 | Higher support |
| 2b | Atelectasis / consolidation | 49 | Higher support |
| 2c | Ground-glass opacity | 60 | Higher support |
| 2d | Pulmonary nodules / masses | 132 | Higher support |
| 2e | Pleural effusion / thickening | 11 | Low support |
| 2f | Honeycombing | 0 | Unavailable |
| 2g | Pneumothorax | 1 | Very rare; descriptive only |
| 2h | Other focal lung/airway/pleural finding | 7 | Very rare; descriptive only |

## Overall Threshold Sweep

The global winner is selected by full-precision mean Dice, then hits, then the lower threshold.

| Threshold | Dice | Hits | Hit rate | Macro-category Dice | Minimum category Dice | Categories choosing threshold | Mean category regret | Maximum category regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.313191 | 285 / 381 | 0.748031 | 0.230617 | 0.069067 | 2 / 13 | 0.036993 | 0.078914 |
| 0.20 | 0.330768 | 290 / 381 | 0.761155 | 0.238415 | 0.086218 | 2 / 13 | 0.029195 | 0.055142 |
| 0.25 | 0.336219 | 291 / 381 | 0.763780 | 0.240472 | 0.092889 | 0 / 13 | 0.027137 | 0.071103 |
| 0.30 | 0.339865 | 290 / 381 | 0.761155 | 0.242069 | 0.084706 | 1 / 13 | 0.025540 | 0.083810 |
| 0.35 | 0.341500 | 292 / 381 | 0.766404 | 0.242828 | 0.074025 | 0 / 13 | 0.024782 | 0.094490 |
| 0.40 | 0.343076 | 291 / 381 | 0.763780 | 0.243135 | 0.064821 | 1 / 13 | 0.024475 | 0.103695 |
| 0.45 | 0.343998 | 291 / 381 | 0.763780 | 0.242998 | 0.056374 | 1 / 13 | 0.024612 | 0.112142 |
| 0.50 | 0.344554 | 290 / 381 | 0.761155 | 0.242791 | 0.048693 | 0 / 13 | 0.024819 | 0.119823 |
| **0.55** | **0.344846** | 290 / 381 | 0.761155 | 0.242277 | 0.041644 | 0 / 13 | 0.025333 | 0.126872 |
| 0.60 | 0.344141 | 288 / 381 | 0.755906 | 0.240636 | 0.035087 | 1 / 13 | 0.026974 | 0.133429 |
| 0.65 | 0.342320 | 287 / 381 | 0.753281 | 0.237602 | 0.029129 | 2 / 13 | 0.030008 | 0.139386 |
| 0.70 | 0.340314 | 288 / 381 | 0.755906 | 0.233540 | 0.023612 | 0 / 13 | 0.034070 | 0.144903 |
| 0.75 | 0.336933 | 287 / 381 | 0.753281 | 0.227151 | 0.018149 | 1 / 13 | 0.040459 | 0.150367 |
| 0.80 | 0.332479 | 287 / 381 | 0.753281 | 0.218687 | 0.012716 | 1 / 13 | 0.048923 | 0.155799 |
| 0.90 | 0.313574 | 276 / 381 | 0.724409 | 0.192416 | 0.003003 | 1 / 13 | 0.075194 | 0.200374 |

## Global Versus Category-Best Thresholds

| Code | Category | N | Global threshold Dice | Global hits | Best threshold | Best Dice | Best hits | Dice gain | Hit gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | Bronchial wall thickening | 3 | 0.122321 | 1 / 3 | **0.80** | **0.139476** | **1 / 3** | +0.017154 | +0 |
| 1b | Bronchiectasis | 11 | 0.175764 | 4 / 11 | **0.65** | **0.179676** | **4 / 11** | +0.003911 | +0 |
| 1c | Emphysema | 17 | 0.195090 | 8 / 17 | **0.60** | **0.197481** | **8 / 17** | +0.002391 | +0 |
| 1d | Septal thickening / reticulation | 6 | 0.117095 | 1 / 6 | **0.20** | **0.131988** | **1 / 6** | +0.014893 | +0 |
| 1e | Micronodules / tree-in-bud | 11 | 0.198606 | 7 / 11 | **0.75** | **0.215179** | **7 / 11** | +0.016574 | +0 |
| 1f | Other diffuse lung/airway/pleural abnormality | 4 | 0.164634 | 1 / 4 | **0.10** | **0.208363** | **2 / 4** | +0.043728 | +1 |
| 2a | Linear opacity, scarring, fibrosis | 69 | 0.351082 | 58 / 69 | **0.65** | **0.353104** | **58 / 69** | +0.002022 | +0 |
| 2b | Atelectasis / consolidation | 49 | 0.361520 | 36 / 49 | **0.30** | **0.370047** | **38 / 49** | +0.008527 | +2 |
| 2c | Ground-glass opacity | 60 | 0.397708 | 50 / 60 | **0.40** | **0.405667** | **50 / 60** | +0.007959 | +0 |
| 2d | Pulmonary nodules / masses | 132 | 0.379972 | 112 / 132 | **0.90** | **0.388159** | **109 / 132** | +0.008187 | -3 |
| 2e | Pleural effusion / thickening | 11 | 0.428788 | 8 / 11 | **0.20** | **0.500303** | **9 / 11** | +0.071516 | +1 |
| 2f | Honeycombing | 0 | — | — | — | — | — | — | — |
| 2g | Pneumothorax | 1 | 0.041644 | 0 / 1 | **0.10** | **0.168516** | **1 / 1** | +0.126872 | +1 |
| 2h | Other focal lung/airway/pleural finding | 7 | 0.215379 | 4 / 7 | **0.45** | **0.220968** | **4 / 7** | +0.005589 | +0 |

## Diffuse Threshold Matrix

Each cell is `Dice (hits/N)`; the selected category threshold is bold.

| Threshold | 1a (N=3) | 1b (N=11) | 1c (N=17) | 1d (N=6) | 1e (N=11) | 1f (N=4) |
| ---: | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.069067 (1/3) | 0.100762 (4/11) | 0.149782 (6/17) | 0.129583 (1/6) | 0.144006 (5/11) | **0.208363 (2/4)** |
| 0.20 | 0.086218 (1/3) | 0.130247 (4/11) | 0.165121 (6/17) | **0.131988 (1/6)** | 0.162809 (6/11) | 0.173715 (1/4) |
| 0.25 | 0.092889 (1/3) | 0.141268 (4/11) | 0.170784 (6/17) | 0.131544 (1/6) | 0.168659 (6/11) | 0.162195 (1/4) |
| 0.30 | 0.098471 (1/3) | 0.150662 (4/11) | 0.175496 (6/17) | 0.130426 (1/6) | 0.173887 (6/11) | 0.161290 (1/4) |
| 0.35 | 0.103509 (1/3) | 0.157577 (4/11) | 0.179805 (7/17) | 0.128939 (1/6) | 0.179383 (6/11) | 0.162242 (1/4) |
| 0.40 | 0.108383 (1/3) | 0.163323 (4/11) | 0.183836 (7/17) | 0.126402 (1/6) | 0.184746 (6/11) | 0.162704 (1/4) |
| 0.45 | 0.112940 (1/3) | 0.168397 (4/11) | 0.187764 (7/17) | 0.123710 (1/6) | 0.188830 (7/11) | 0.163174 (1/4) |
| 0.50 | 0.117778 (1/3) | 0.172498 (4/11) | 0.191660 (8/17) | 0.120788 (1/6) | 0.193506 (7/11) | 0.163636 (1/4) |
| **0.55 (global)** | 0.122321 (1/3) | 0.175764 (4/11) | 0.195090 (8/17) | 0.117095 (1/6) | 0.198606 (7/11) | 0.164634 (1/4) |
| 0.60 | 0.126757 (1/3) | 0.178238 (4/11) | **0.197481 (8/17)** | 0.112429 (1/6) | 0.201675 (7/11) | 0.164363 (1/4) |
| 0.65 | 0.131028 (1/3) | **0.179676 (4/11)** | 0.193834 (8/17) | 0.106617 (1/6) | 0.207317 (7/11) | 0.162016 (1/4) |
| 0.70 | 0.134756 (1/3) | 0.178811 (4/11) | 0.190577 (9/17) | 0.098768 (1/6) | 0.212633 (7/11) | 0.161189 (1/4) |
| 0.75 | 0.137566 (1/3) | 0.174545 (4/11) | 0.187988 (9/17) | 0.088291 (1/6) | **0.215179 (7/11)** | 0.158768 (1/4) |
| 0.80 | **0.139476 (1/3)** | 0.168309 (4/11) | 0.184813 (9/17) | 0.074441 (1/6) | 0.214013 (7/11) | 0.158228 (1/4) |
| 0.90 | 0.119841 (1/3) | 0.151438 (4/11) | 0.165305 (8/17) | 0.016886 (0/6) | 0.200185 (7/11) | 0.154531 (1/4) |

## Focal Threshold Matrix

Each cell is `Dice (hits/N)`; the selected category threshold is bold.

| Threshold | 2a (N=69) | 2b (N=49) | 2c (N=60) | 2d (N=132) | 2e (N=11) | 2f (N=0) | 2g (N=1) | 2h (N=7) |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.307683 (55/69) | 0.357721 (37/49) | 0.384076 (48/60) | 0.331728 (112/132) | 0.495544 (9/11) | — | **0.168516 (1/1)** | 0.151192 (4/7) |
| 0.20 | 0.327539 (56/69) | 0.367740 (37/49) | 0.398267 (50/60) | 0.354558 (114/132) | **0.500303 (9/11)** | — | 0.113374 (1/1) | 0.187516 (4/7) |
| 0.25 | 0.334018 (57/69) | 0.369680 (38/49) | 0.401643 (50/60) | 0.362736 (114/132) | 0.494585 (9/11) | — | 0.097413 (0/1) | 0.198729 (4/7) |
| 0.30 | 0.338513 (57/69) | **0.370047 (38/49)** | 0.404004 (50/60) | 0.368334 (113/132) | 0.484971 (9/11) | — | 0.084706 (0/1) | 0.206095 (4/7) |
| 0.35 | 0.342142 (58/69) | 0.369579 (38/49) | 0.405284 (50/60) | 0.369945 (113/132) | 0.470906 (9/11) | — | 0.074025 (0/1) | 0.213429 (4/7) |
| 0.40 | 0.345576 (59/69) | 0.368366 (37/49) | **0.405667 (50/60)** | 0.372635 (113/132) | 0.454041 (8/11) | — | 0.064821 (0/1) | 0.220254 (4/7) |
| 0.45 | 0.348269 (59/69) | 0.366791 (37/49) | 0.403997 (50/60) | 0.374928 (112/132) | 0.442831 (8/11) | — | 0.056374 (0/1) | **0.220968 (4/7)** |
| 0.50 | 0.349922 (58/69) | 0.364505 (36/49) | 0.401156 (50/60) | 0.377324 (112/132) | 0.436066 (8/11) | — | 0.048693 (0/1) | 0.218750 (4/7) |
| **0.55 (global)** | 0.351082 (58/69) | 0.361520 (36/49) | 0.397708 (50/60) | 0.379972 (112/132) | 0.428788 (8/11) | — | 0.041644 (0/1) | 0.215379 (4/7) |
| 0.60 | 0.351771 (58/69) | 0.358041 (36/49) | 0.393521 (50/60) | 0.381290 (111/132) | 0.420750 (8/11) | — | 0.035087 (0/1) | 0.206867 (3/7) |
| 0.65 | **0.353104 (58/69)** | 0.353877 (36/49) | 0.388324 (50/60) | 0.381008 (110/132) | 0.411199 (8/11) | — | 0.029129 (0/1) | 0.191692 (3/7) |
| 0.70 | 0.353009 (58/69) | 0.348654 (36/49) | 0.381693 (50/60) | 0.382717 (110/132) | 0.399934 (8/11) | — | 0.023612 (0/1) | 0.169671 (3/7) |
| 0.75 | 0.352613 (58/69) | 0.341663 (36/49) | 0.372979 (49/60) | 0.383693 (111/132) | 0.386071 (8/11) | — | 0.018149 (0/1) | 0.135453 (2/7) |
| 0.80 | 0.351104 (58/69) | 0.332411 (36/49) | 0.360647 (49/60) | 0.386080 (111/132) | 0.368997 (8/11) | — | 0.012716 (0/1) | 0.091696 (2/7) |
| 0.90 | 0.342946 (57/69) | 0.293890 (35/49) | 0.312353 (46/60) | **0.388159 (109/132)** | 0.299929 (7/11) | — | 0.003003 (0/1) | 0.052941 (1/7) |

## Validation and Provenance

- Candidate manifest SHA256: `1e5c08ebe7c4a393be8ba9319a6fbb845865a346cb6e7bcdb1be4e082f7f622f`
- Dataset SHA256: `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`
- Array inventory SHA256: `786e230f797fce22712225b249a2fe00e4b710c3bc2ba82007474ebacc8240d3`
- Recipe hash: `8ee57211f8656d0f8e51e61bce5281a515fbd12f0c9d5617d71cf3473514094e`
- Deterministic result SHA256: `feaae2eb15497574981cb98ce805e2d4bad27cdde213929d9942542f103fb8a4`
- Arrays checked: `4000`
- Threshold recompositions passed: `True`
- Category-oracle recomposition passed: `True`

Reproduction command:

```bash
PYTHONDONTWRITEBYTECODE=1 python side_experiments/sideexp002_multimodel_ensemble_selection/analyze_all20_thresholds.py --manifest /workspace/side_experiments/sideexp002_multimodel_ensemble_selection/candidate_manifest.json --dataset-json /workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json --runtime-root /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp002_multimodel_ensemble_selection --seg-dir /data/hengjie/datasets/rexgroundingct/segmentations --output-json /workspace/side_experiments/sideexp002_multimodel_ensemble_selection/outputs/all20_uniform_probability_threshold_summary.json --output-markdown /workspace/side_experiments/sideexp002_multimodel_ensemble_selection/outputs/all20_uniform_probability_threshold_report.md --chunk-elements 4194304 --allow-unaccepted
```

Both the global winner and category-specific winners were selected on this same val200 set. The category oracle is a hindsight upper bound and must not be presented as expected test performance.
