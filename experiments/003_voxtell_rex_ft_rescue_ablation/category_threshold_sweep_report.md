# Experiment 003 Ensemble Threshold Sweep by ReX Category

This report re-aggregates the completed epoch-100 four-model probability
ensemble evaluations on the fixed val200 set. No model inference or
evaluator run was repeated.

- Cases: `200`
- Findings: `381`
- Thresholds: `13`
- Hit definition: global Dice `>= 0.1`
- Best-threshold rule: maximum category mean global Dice per finding; ties use higher hit rate, then lower threshold.
- Global reference: threshold `0.35`, Dice `0.284191`, hit rate `0.690289` (`263/381`).

## Best Threshold Per Category

| Category | Label | N | Best threshold | Best Dice | Hits / N | Hit rate | Dice change vs 0.35 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | Bronchial wall thickening | 3 | **0.45** | **0.091662** | 1 / 3 | 0.333333 | +0.007066 |
| 1b | Bronchiectasis | 11 | **0.30** | **0.134967** | 4 / 11 | 0.363636 | +0.000172 |
| 1c | Emphysema | 17 | **0.45** | **0.121249** | 7 / 17 | 0.411765 | +0.001988 |
| 1d | Septal thickening / reticulation | 6 | **0.10** | **0.135957** | 1 / 6 | 0.166667 | +0.004522 |
| 1e | Micronodules / tree-in-bud | 11 | **0.30** | **0.183573** | 7 / 11 | 0.636364 | +0.000058 |
| 1f | Other diffuse lung/airway/pleural abnormality | 4 | **0.80** | **0.151754** | 1 / 4 | 0.250000 | +0.016950 |
| 2a | Linear opacity, scarring, fibrosis | 69 | **0.30** | **0.268285** | 49 / 69 | 0.710145 | +0.000498 |
| 2b | Atelectasis / consolidation | 49 | **0.30** | **0.330177** | 36 / 49 | 0.734694 | +0.000483 |
| 2c | Ground-glass opacity | 60 | **0.20** | **0.378462** | 47 / 60 | 0.783333 | +0.001924 |
| 2d | Pulmonary nodules / masses | 132 | **0.50** | **0.303438** | 101 / 132 | 0.765152 | +0.011514 |
| 2e | Pleural effusion / thickening | 11 | **0.20** | **0.450255** | 8 / 11 | 0.727273 | +0.001482 |
| 2f | Honeycombing | 0 | — | — | — | — | — |
| 2g | Pneumothorax | 1 | **0.10** | **0.192347** | 1 / 1 | 1.000000 | +0.104692 |
| 2h | Other focal lung/airway/pleural finding | 7 | **0.90** | **0.070613** | 1 / 7 | 0.142857 | +0.015805 |

## Full Category Threshold Matrix

Each cell is `mean Dice (hits/N)`. The selected best threshold for each
category is bold.

| Category | Label | N | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.80 | 0.90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | Bronchial wall thickening | 3 | 0.061222 (1/3) | 0.072278 (1/3) | 0.080738 (1/3) | 0.084597 (1/3) | 0.087937 (1/3) | **0.091662 (1/3)** | 0.078853 (1/3) | 0.081384 (1/3) | 0.083688 (1/3) | 0.086020 (1/3) | 0.086953 (1/3) | 0.086652 (1/3) | 0.090420 (1/3) |
| 1b | Bronchiectasis | 11 | 0.124440 (3/11) | 0.132534 (3/11) | **0.134967 (4/11)** | 0.134795 (3/11) | 0.132636 (3/11) | 0.127503 (3/11) | 0.116842 (3/11) | 0.111376 (3/11) | 0.106869 (3/11) | 0.101896 (3/11) | 0.095053 (2/11) | 0.078727 (2/11) | 0.067932 (2/11) |
| 1c | Emphysema | 17 | 0.109822 (5/17) | 0.114211 (7/17) | 0.117553 (7/17) | 0.119261 (7/17) | 0.120984 (7/17) | **0.121249 (7/17)** | 0.091731 (6/17) | 0.088926 (6/17) | 0.087242 (5/17) | 0.084892 (4/17) | 0.080789 (4/17) | 0.054434 (2/17) | 0.041299 (2/17) |
| 1d | Septal thickening / reticulation | 6 | **0.135957 (1/6)** | 0.135832 (1/6) | 0.133419 (1/6) | 0.131434 (1/6) | 0.128538 (1/6) | 0.123843 (1/6) | 0.111903 (1/6) | 0.110620 (1/6) | 0.109414 (1/6) | 0.107974 (1/6) | 0.106190 (1/6) | 0.099687 (1/6) | 0.074620 (1/6) |
| 1e | Micronodules / tree-in-bud | 11 | 0.178399 (7/11) | 0.181513 (7/11) | **0.183573 (7/11)** | 0.183515 (7/11) | 0.182035 (7/11) | 0.178943 (7/11) | 0.145069 (6/11) | 0.132366 (4/11) | 0.132856 (4/11) | 0.133177 (4/11) | 0.133276 (4/11) | 0.132937 (4/11) | 0.133918 (4/11) |
| 1f | Other diffuse lung/airway/pleural abnormality | 4 | 0.135304 (1/4) | 0.135951 (1/4) | 0.135333 (1/4) | 0.134804 (1/4) | 0.134500 (1/4) | 0.134983 (1/4) | 0.139713 (1/4) | 0.145959 (1/4) | 0.144914 (1/4) | 0.144935 (1/4) | 0.143647 (1/4) | **0.151754 (1/4)** | 0.150538 (1/4) |
| 2a | Linear opacity, scarring, fibrosis | 69 | 0.263969 (50/69) | 0.267532 (49/69) | **0.268285 (49/69)** | 0.267787 (49/69) | 0.266689 (48/69) | 0.264949 (48/69) | 0.248854 (45/69) | 0.243887 (45/69) | 0.242685 (45/69) | 0.241583 (43/69) | 0.239891 (41/69) | 0.234669 (41/69) | 0.228955 (40/69) |
| 2b | Atelectasis / consolidation | 49 | 0.326566 (35/49) | 0.330100 (35/49) | **0.330177 (36/49)** | 0.329694 (36/49) | 0.328516 (36/49) | 0.326487 (36/49) | 0.318966 (35/49) | 0.316492 (35/49) | 0.314152 (35/49) | 0.311730 (35/49) | 0.308379 (35/49) | 0.295358 (34/49) | 0.271552 (34/49) |
| 2c | Ground-glass opacity | 60 | 0.375378 (48/60) | **0.378462 (47/60)** | 0.378109 (47/60) | 0.376538 (47/60) | 0.373721 (47/60) | 0.368035 (47/60) | 0.351732 (46/60) | 0.348658 (46/60) | 0.344490 (46/60) | 0.339674 (46/60) | 0.334066 (46/60) | 0.317282 (44/60) | 0.286900 (42/60) |
| 2d | Pulmonary nodules / masses | 132 | 0.282906 (97/132) | 0.285584 (98/132) | 0.290611 (101/132) | 0.291924 (101/132) | 0.293441 (100/132) | 0.295775 (100/132) | **0.303438 (101/132)** | 0.290799 (96/132) | 0.291423 (96/132) | 0.292412 (96/132) | 0.293247 (96/132) | 0.288006 (91/132) | 0.288921 (91/132) |
| 2e | Pleural effusion / thickening | 11 | 0.443148 (8/11) | **0.450255 (8/11)** | 0.450120 (8/11) | 0.448773 (8/11) | 0.446773 (8/11) | 0.443860 (8/11) | 0.439939 (8/11) | 0.436119 (8/11) | 0.431551 (8/11) | 0.426058 (8/11) | 0.418893 (8/11) | 0.396388 (8/11) | 0.335193 (8/11) |
| 2f | Honeycombing | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2g | Pneumothorax | 1 | **0.192347 (1/1)** | 0.128802 (1/1) | 0.098201 (0/1) | 0.087655 (0/1) | 0.078537 (0/1) | 0.070296 (0/1) | 0.063593 (0/1) | 0.058012 (0/1) | 0.052620 (0/1) | 0.047385 (0/1) | 0.042080 (0/1) | 0.031012 (0/1) | 0.016620 (0/1) |
| 2h | Other focal lung/airway/pleural finding | 7 | 0.046987 (2/7) | 0.051138 (2/7) | 0.053605 (2/7) | 0.054808 (2/7) | 0.055871 (2/7) | 0.057168 (1/7) | 0.058626 (1/7) | 0.059531 (1/7) | 0.060469 (1/7) | 0.061428 (1/7) | 0.062656 (1/7) | 0.065742 (1/7) | **0.070613 (1/7)** |

## Overall Threshold Sweep

| Threshold | Dice / finding | Hit rate | Hits / findings |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.278784 | 0.679790 | 259 / 381 |
| 0.20 | 0.282023 | 0.682415 | 260 / 381 |
| 0.30 | 0.284117 | 0.692913 | 264 / 381 |
| **0.35** | **0.284191** | **0.690289** | **263 / 381** |
| 0.40 | 0.283810 | 0.685039 | 261 / 381 |
| 0.45 | 0.282800 | 0.682415 | 260 / 381 |
| 0.50 | 0.276059 | 0.666667 | 254 / 381 |
| 0.55 | 0.269286 | 0.648294 | 247 / 381 |
| 0.60 | 0.267995 | 0.645669 | 246 / 381 |
| 0.65 | 0.266670 | 0.637795 | 243 / 381 |
| 0.70 | 0.264729 | 0.629921 | 240 / 381 |
| 0.80 | 0.255351 | 0.603675 | 230 / 381 |
| 0.90 | 0.243824 | 0.595801 | 227 / 381 |

## Validation-Only Per-Category Oracle

| Policy | Dice / finding | Hit rate | Hits / findings | Dice change | Hit change |
| --- | ---: | ---: | ---: | ---: | ---: |
| One global threshold (0.35) | 0.284191 | 0.690289 | 263 / 381 | — | — |
| Per-category Dice-selected thresholds | 0.289644 | 0.692913 | 264 / 381 | +0.005452 | +1 |

This oracle selects and evaluates thresholds on the same val200 set. It
measures threshold sensitivity but is optimistic and must not be treated
as a submission recipe or evidence of test-set generalization.

## Support And Reliability

- Thresholds were selected and evaluated on the same val200 set; the per-category oracle is optimistic and is not a submission policy.
- Category 2f has no val200 findings, so no category-specific threshold can be estimated.
- Category 2g has one val200 finding; its selected threshold is descriptive and not reliable.
- Additional low-support categories with fewer than 10 findings: `1a` (3), `1d` (6), `1f` (4), `2g` (1), `2h` (7).

## Validation

- All `13` threshold evaluations contain the same cases and finding alignment.
- Category support sums to `381` findings at every threshold.
- Reconstructed global Dice, hits, and hit rate match the saved summaries within `1e-12`.
- Maximum saved-summary Dice delta: `5.551e-17`.
- Maximum threshold-index Dice delta: `5.551e-17`.

## Provenance

- Dataset JSON: `/home/hengjie/code_sync/rexgroundingct/configs/evaluation/rexgroundingct_val200_seed20260723.json`
- Ensemble root: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/ensembles/epoch100_val200`
- Threshold index: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/ensembles/epoch100_val200/threshold_sweep_summary.json`
