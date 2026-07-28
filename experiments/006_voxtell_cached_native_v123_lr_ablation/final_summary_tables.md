---
created: 2026-07-25
updated: 2026-07-25
status: complete
---

# Experiment 006 Final Summary Tables

Runtime group:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z`

Notes:

- Fixed val20: `configs/evaluation/rexgroundingct_val20_seed20260723.json`.
- Fixed val200: `configs/evaluation/rexgroundingct_val200_seed20260723.json`.
- PT val20 was derived by filtering the exp001 corrected-orientation full-val200
  evaluation down to the exact fixed val20 case list.
- Cell format in the val20 table is `Dice / hits/targets (hit rate)`.

## Fixed Val20 Progress

| Model | PT | Epoch 5 | Epoch 20 | Epoch 40 | Epoch 60 | Epoch 80 | Epoch 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v123_cached_e7_d6` encoder 1e-7, decoder 1e-6 | 0.2986 / 18/31 (0.581) | 0.3130 / 20/31 (0.645) | 0.3221 / 19/31 (0.613) | 0.3394 / 21/31 (0.677) | 0.3439 / 21/31 (0.677) | 0.3480 / 21/31 (0.677) | 0.3484 / 21/31 (0.677) |
| `v123_cached_e7_d4` encoder 1e-7, decoder 1e-4 | 0.2986 / 18/31 (0.581) | 0.2638 / 19/31 (0.613) | 0.3366 / 22/31 (0.710) | 0.3481 / 22/31 (0.710) | 0.3458 / 21/31 (0.677) | 0.3680 / 23/31 (0.742) | 0.3700 / 22/31 (0.710) |
| `v123_cached_e6_d4` encoder 1e-6, decoder 1e-4 | 0.2986 / 18/31 (0.581) | 0.2809 / 20/31 (0.645) | 0.3341 / 20/31 (0.645) | 0.3312 / 21/31 (0.677) | 0.3197 / 21/31 (0.677) | 0.3749 / 21/31 (0.677) | 0.3763 / 22/31 (0.710) |
| `v123_cached_e5_d4` encoder 1e-5, decoder 1e-4 | 0.2986 / 18/31 (0.581) | 0.2881 / 20/31 (0.645) | 0.3332 / 20/31 (0.645) | 0.3462 / 22/31 (0.710) | 0.3316 / 21/31 (0.677) | 0.3531 / 22/31 (0.710) | 0.3907 / 23/31 (0.742) |

## Fixed Val200 Epoch-100 Comparison

| Row | Dice | Hit rate | Hits / targets |
| --- | ---: | ---: | ---: |
| PT VoxTell v1.1, exp001 corrected orientation | 0.2252 | 0.535 | 204 / 381 |
| Previous best pre-exp006: exp004 native192 continuation | 0.2928 | 0.706 | 269 / 381 |
| exp006 `v123_cached_e7_d6` epoch 100 | 0.2833 | 0.690 | 263 / 381 |
| exp006 `v123_cached_e7_d4` epoch 100 | 0.3132 | 0.724 | 276 / 381 |
| exp006 `v123_cached_e6_d4` epoch 100 | 0.3226 | 0.738 | 281 / 381 |
| exp006 `v123_cached_e5_d4` epoch 100 | 0.3241 | 0.761 | 290 / 381 |

## Fixed Val200 Category Breakdown

This uses the challenge category code stored in
`configs/evaluation/rexgroundingct_val200_seed20260723.json`. Counts are for
our fixed val200 set. The count column is `val [train; test]`, with each
split shown as `n (ratio within split)`. Train/val counts come from
`MICCAI_challenge_dataset.json`; test counts come from the public test-set
leaderboard screenshot. Split totals are train `7687`, val `381`, test `302`.

| Category | Count (%): val [train; test] | PT Dice | e5d4 Dice | dDice | PT Hit | e5d4 Hit | dHit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a - Bronchial wall thickening | 3 (0.8%) [236 (3.1%); 5 (1.7%)] | 0.073 | 0.100 | +0.026 | 0.333 | 0.333 | +0.000 |
| 1b - Bronchiectasis | 11 (2.9%) [282 (3.7%); 5 (1.7%)] | 0.072 | 0.120 | +0.047 | 0.182 | 0.364 | +0.182 |
| 1c - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 (4.5%) [446 (5.8%); 16 (5.3%)] | 0.064 | 0.152 | +0.088 | 0.235 | 0.471 | +0.235 |
| 1d - Septal thickening (including Interlobular, Reticulation) | 6 (1.6%) [194 (2.5%); 6 (2.0%)] | 0.125 | 0.100 | -0.025 | 0.167 | 0.167 | +0.000 |
| 1e - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 11 (2.9%) [314 (4.1%); 10 (3.3%)] | 0.102 | 0.187 | +0.085 | 0.273 | 0.545 | +0.273 |
| 1f - Other (non-focal) | 4 (1.0%) [150 (2.0%); 2 (0.7%)] | 0.002 | 0.160 | +0.158 | 0.000 | 0.250 | +0.250 |
| 2a - Linear (including subsegmental atelectasis, scarring, fibrosis) | 69 (18.1%) [1120 (14.6%); 61 (20.2%)] | 0.197 | 0.320 | +0.122 | 0.551 | 0.812 | +0.261 |
| 2b - Atelectasis, consolidation | 49 (12.9%) [1367 (17.8%); 46 (15.2%)] | 0.297 | 0.356 | +0.059 | 0.673 | 0.776 | +0.102 |
| 2c - Groundglass opacity | 60 (15.7%) [1507 (19.6%); 39 (12.9%)] | 0.309 | 0.382 | +0.073 | 0.700 | 0.833 | +0.133 |
| 2d - Pulmonary nodules/masses | 132 (34.6%) [1743 (22.7%); 97 (32.1%)] | 0.227 | 0.361 | +0.135 | 0.530 | 0.864 | +0.333 |
| 2e - Pleural effusion or thickening | 11 (2.9%) [237 (3.1%); 13 (4.3%)] | 0.431 | 0.413 | -0.018 | 0.727 | 0.727 | +0.000 |
| 2f - Honeycombing | 0 (0.0%) [16 (0.2%); 0 (0.0%)] | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| 2g - Pneumothorax | 1 (0.3%) [18 (0.2%); 1 (0.3%)] | 0.139 | 0.076 | -0.062 | 1.000 | 0.000 | -1.000 |
| 2h - Other (focal) | 7 (1.8%) [57 (0.7%); 1 (0.3%)] | 0.048 | 0.181 | +0.133 | 0.143 | 0.429 | +0.286 |

## Outcome

All requested exp006 val20 and val200 summaries are present. The strongest arm
is `v123_cached_e5_d4`: it has the best fixed val20 epoch-100 Dice and the best
fixed val200 Dice/hit rate among this experiment's four arms.
