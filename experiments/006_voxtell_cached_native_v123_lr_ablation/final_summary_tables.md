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

## Outcome

All requested exp006 val20 and val200 summaries are present. The strongest arm
is `v123_cached_e5_d4`: it has the best fixed val20 epoch-100 Dice and the best
fixed val200 Dice/hit rate among this experiment's four arms.
