# SideExp005: frozen val200 postprocessing comparison

200 CTs / 381 findings. All variants share the frozen top-four probability ensemble and official CT-RATE anatomy.

| Variant | Finding Dice | Case Dice | Hits / 381 | HIT rate |
|---|---:|---:|---:|---:|
| d1 | 0.357504194 | 0.378816306 | 296 | 0.776902887 |
| d2 | 0.361598092 | 0.383852775 | 296 | 0.776902887 |
| d3 | 0.364029702 | 0.385503284 | 295 | 0.774278215 |
| d11 | 0.366019837 | 0.386817276 | 296 | 0.776902887 |
| d12 | 0.368820199 | 0.390338690 | 297 | 0.779527559 |

## All official categories

Category Dice is the mean over findings in that category; HIT means finding Dice >= 0.1.

| Category | Method | n | Dice | Hits | HIT rate |
|---|---|---:|---:|---:|---:|
| 1a — Bronchial wall thickening | d1 | 3 | 0.121353996 | 1 | 0.333333333 |
| 1a — Bronchial wall thickening | d2 | 3 | 0.121353996 | 1 | 0.333333333 |
| 1a — Bronchial wall thickening | d3 | 3 | 0.121353996 | 1 | 0.333333333 |
| 1a — Bronchial wall thickening | d11 | 3 | 0.121630183 | 1 | 0.333333333 |
| 1a — Bronchial wall thickening | d12 | 3 | 0.121630183 | 1 | 0.333333333 |
| 1b — Bronchiectasis | d1 | 11 | 0.175400915 | 4 | 0.363636364 |
| 1b — Bronchiectasis | d2 | 11 | 0.175448348 | 4 | 0.363636364 |
| 1b — Bronchiectasis | d3 | 11 | 0.177408910 | 4 | 0.363636364 |
| 1b — Bronchiectasis | d11 | 11 | 0.178127954 | 4 | 0.363636364 |
| 1b — Bronchiectasis | d12 | 11 | 0.178127954 | 4 | 0.363636364 |
| 1c — Emphysema | d1 | 17 | 0.191700028 | 8 | 0.470588235 |
| 1c — Emphysema | d2 | 17 | 0.192421826 | 8 | 0.470588235 |
| 1c — Emphysema | d3 | 17 | 0.192312928 | 8 | 0.470588235 |
| 1c — Emphysema | d11 | 17 | 0.192492985 | 8 | 0.470588235 |
| 1c — Emphysema | d12 | 17 | 0.192492985 | 8 | 0.470588235 |
| 1d — Septal thickening | d1 | 6 | 0.125988357 | 1 | 0.166666667 |
| 1d — Septal thickening | d2 | 6 | 0.125989257 | 1 | 0.166666667 |
| 1d — Septal thickening | d3 | 6 | 0.125989239 | 1 | 0.166666667 |
| 1d — Septal thickening | d11 | 6 | 0.125989257 | 1 | 0.166666667 |
| 1d — Septal thickening | d12 | 6 | 0.125989257 | 1 | 0.166666667 |
| 1e — Micronodules | d1 | 11 | 0.197125959 | 7 | 0.636363636 |
| 1e — Micronodules | d2 | 11 | 0.200461180 | 7 | 0.636363636 |
| 1e — Micronodules | d3 | 11 | 0.198420829 | 7 | 0.636363636 |
| 1e — Micronodules | d11 | 11 | 0.201350889 | 7 | 0.636363636 |
| 1e — Micronodules | d12 | 11 | 0.201350889 | 7 | 0.636363636 |
| 1f — Other (non-focal) | d1 | 4 | 0.194984155 | 2 | 0.500000000 |
| 1f — Other (non-focal) | d2 | 4 | 0.194984155 | 2 | 0.500000000 |
| 1f — Other (non-focal) | d3 | 4 | 0.194984155 | 2 | 0.500000000 |
| 1f — Other (non-focal) | d11 | 4 | 0.194984155 | 2 | 0.500000000 |
| 1f — Other (non-focal) | d12 | 4 | 0.194984155 | 2 | 0.500000000 |
| 2a — Linear / scarring / fibrosis | d1 | 69 | 0.356342130 | 58 | 0.840579710 |
| 2a — Linear / scarring / fibrosis | d2 | 69 | 0.360910324 | 58 | 0.840579710 |
| 2a — Linear / scarring / fibrosis | d3 | 69 | 0.361950001 | 58 | 0.840579710 |
| 2a — Linear / scarring / fibrosis | d11 | 69 | 0.362022355 | 58 | 0.840579710 |
| 2a — Linear / scarring / fibrosis | d12 | 69 | 0.362022355 | 58 | 0.840579710 |
| 2b — Atelectasis / consolidation | d1 | 49 | 0.369031642 | 37 | 0.755102041 |
| 2b — Atelectasis / consolidation | d2 | 49 | 0.369533706 | 37 | 0.755102041 |
| 2b — Atelectasis / consolidation | d3 | 49 | 0.369679218 | 37 | 0.755102041 |
| 2b — Atelectasis / consolidation | d11 | 49 | 0.370076822 | 37 | 0.755102041 |
| 2b — Atelectasis / consolidation | d12 | 49 | 0.370076822 | 37 | 0.755102041 |
| 2c — Groundglass opacity | d1 | 60 | 0.404280845 | 51 | 0.850000000 |
| 2c — Groundglass opacity | d2 | 60 | 0.405200235 | 51 | 0.850000000 |
| 2c — Groundglass opacity | d3 | 60 | 0.401381247 | 50 | 0.833333333 |
| 2c — Groundglass opacity | d11 | 60 | 0.413722512 | 51 | 0.850000000 |
| 2c — Groundglass opacity | d12 | 60 | 0.427211215 | 52 | 0.866666667 |
| 2d — Pulmonary nodules/masses | d1 | 132 | 0.403233696 | 115 | 0.871212121 |
| 2d — Pulmonary nodules/masses | d2 | 132 | 0.411358843 | 115 | 0.871212121 |
| 2d — Pulmonary nodules/masses | d3 | 132 | 0.419536447 | 115 | 0.871212121 |
| 2d — Pulmonary nodules/masses | d11 | 132 | 0.419478475 | 115 | 0.871212121 |
| 2d — Pulmonary nodules/masses | d12 | 132 | 0.421428618 | 115 | 0.871212121 |
| 2e — Pleural effusion or thickening | d1 | 11 | 0.440992232 | 8 | 0.727272727 |
| 2e — Pleural effusion or thickening | d2 | 11 | 0.440992232 | 8 | 0.727272727 |
| 2e — Pleural effusion or thickening | d3 | 11 | 0.440992232 | 8 | 0.727272727 |
| 2e — Pleural effusion or thickening | d11 | 11 | 0.437071368 | 8 | 0.727272727 |
| 2e — Pleural effusion or thickening | d12 | 11 | 0.437071368 | 8 | 0.727272727 |
| 2f — Honeycombing | d1 | 0 | NA | 0 | NA |
| 2f — Honeycombing | d2 | 0 | NA | 0 | NA |
| 2f — Honeycombing | d3 | 0 | NA | 0 | NA |
| 2f — Honeycombing | d11 | 0 | NA | 0 | NA |
| 2f — Honeycombing | d12 | 0 | NA | 0 | NA |
| 2g — Pneumothorax | d1 | 1 | 0.033449764 | 0 | 0.000000000 |
| 2g — Pneumothorax | d2 | 1 | 0.033449764 | 0 | 0.000000000 |
| 2g — Pneumothorax | d3 | 1 | 0.033449764 | 0 | 0.000000000 |
| 2g — Pneumothorax | d11 | 1 | 0.033492276 | 0 | 0.000000000 |
| 2g — Pneumothorax | d12 | 1 | 0.033492276 | 0 | 0.000000000 |
| 2h — Other (focal) | d1 | 7 | 0.273464708 | 4 | 0.571428571 |
| 2h — Other (focal) | d2 | 7 | 0.279579093 | 4 | 0.571428571 |
| 2h — Other (focal) | d3 | 7 | 0.279579093 | 4 | 0.571428571 |
| 2h — Other (focal) | d11 | 7 | 0.279579093 | 4 | 0.571428571 |
| 2h — Other (focal) | d12 | 7 | 0.279607217 | 4 | 0.571428571 |

## d2 vs d1

| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 381 | 74 | 0 | 307 | 0.004093898 | 0 | 0 | 0 | 146555 | 1 |
| 1a | 3 | 0 | 0 | 3 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1b | 11 | 1 | 0 | 10 | 0.000047433 | 0 | 0 | 0 | 780 | 0 |
| 1c | 17 | 9 | 0 | 8 | 0.000721799 | 0 | 0 | 0 | 13300 | 0 |
| 1d | 6 | 2 | 0 | 4 | 0.000000900 | 0 | 0 | 0 | 840 | 0 |
| 1e | 11 | 2 | 0 | 9 | 0.003335221 | 0 | 0 | 0 | 2990 | 0 |
| 1f | 4 | 0 | 0 | 4 | 0.000000000 | 0 | 0 | 0 | 2102 | 0 |
| 2a | 69 | 16 | 0 | 53 | 0.004568195 | 0 | 0 | 0 | 13250 | 0 |
| 2b | 49 | 8 | 0 | 41 | 0.000502063 | 0 | 0 | 0 | 53016 | 0 |
| 2c | 60 | 11 | 0 | 49 | 0.000919390 | 0 | 0 | 0 | 55598 | 0 |
| 2d | 132 | 22 | 0 | 110 | 0.008125147 | 0 | 0 | 0 | 4337 | 1 |
| 2e | 11 | 0 | 0 | 11 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2f | 0 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| 2g | 1 | 0 | 0 | 1 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2h | 7 | 3 | 0 | 4 | 0.006114384 | 0 | 0 | 0 | 342 | 0 |

## d3 vs d1

| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 381 | 82 | 2 | 297 | 0.006525508 | 0 | 1 | 530631 | 259196 | 2 |
| 1a | 3 | 0 | 0 | 3 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1b | 11 | 3 | 0 | 8 | 0.002007996 | 0 | 0 | 0 | 10068 | 0 |
| 1c | 17 | 7 | 0 | 10 | 0.000612900 | 0 | 0 | 0 | 11382 | 0 |
| 1d | 6 | 1 | 0 | 5 | 0.000000882 | 0 | 0 | 0 | 763 | 0 |
| 1e | 11 | 2 | 0 | 9 | 0.001294870 | 0 | 0 | 0 | 3291 | 0 |
| 1f | 4 | 0 | 0 | 4 | 0.000000000 | 0 | 0 | 0 | 2102 | 0 |
| 2a | 69 | 15 | 0 | 54 | 0.005607872 | 0 | 0 | 0 | 16385 | 0 |
| 2b | 49 | 9 | 0 | 40 | 0.000647576 | 0 | 0 | 0 | 50762 | 0 |
| 2c | 60 | 13 | 1 | 46 | -0.002899598 | 0 | 1 | 530574 | 156802 | 0 |
| 2d | 132 | 29 | 1 | 102 | 0.016302752 | 0 | 0 | 57 | 7299 | 2 |
| 2e | 11 | 0 | 0 | 11 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2f | 0 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| 2g | 1 | 0 | 0 | 1 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2h | 7 | 3 | 0 | 4 | 0.006114384 | 0 | 0 | 0 | 342 | 0 |

## d11 vs d1

| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 381 | 105 | 5 | 271 | 0.008515643 | 0 | 0 | 99300 | 300582 | 2 |
| 1a | 3 | 2 | 0 | 1 | 0.000276186 | 0 | 0 | 0 | 1458 | 0 |
| 1b | 11 | 3 | 0 | 8 | 0.002727039 | 0 | 0 | 0 | 11047 | 0 |
| 1c | 17 | 9 | 0 | 8 | 0.000792957 | 0 | 0 | 0 | 14984 | 0 |
| 1d | 6 | 2 | 0 | 4 | 0.000000900 | 0 | 0 | 0 | 840 | 0 |
| 1e | 11 | 3 | 0 | 8 | 0.004224930 | 0 | 0 | 0 | 3409 | 0 |
| 1f | 4 | 0 | 0 | 4 | 0.000000000 | 0 | 0 | 0 | 2712 | 0 |
| 2a | 69 | 18 | 0 | 51 | 0.005680226 | 0 | 0 | 0 | 17921 | 0 |
| 2b | 49 | 13 | 0 | 36 | 0.001045180 | 0 | 0 | 0 | 83410 | 0 |
| 2c | 60 | 14 | 0 | 46 | 0.009441667 | 0 | 0 | 0 | 60500 | 0 |
| 2d | 132 | 34 | 1 | 97 | 0.016244780 | 0 | 0 | 57 | 7349 | 2 |
| 2e | 11 | 3 | 4 | 4 | -0.003920863 | 0 | 0 | 99243 | 92404 | 0 |
| 2f | 0 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| 2g | 1 | 1 | 0 | 0 | 0.000042513 | 0 | 0 | 0 | 4206 | 0 |
| 2h | 7 | 3 | 0 | 4 | 0.006114384 | 0 | 0 | 0 | 342 | 0 |

## d12 vs d1

| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 381 | 112 | 5 | 264 | 0.011316005 | 1 | 0 | 99318 | 360218 | 2 |
| 1a | 3 | 2 | 0 | 1 | 0.000276186 | 0 | 0 | 0 | 1458 | 0 |
| 1b | 11 | 3 | 0 | 8 | 0.002727039 | 0 | 0 | 0 | 11047 | 0 |
| 1c | 17 | 9 | 0 | 8 | 0.000792957 | 0 | 0 | 0 | 14984 | 0 |
| 1d | 6 | 2 | 0 | 4 | 0.000000900 | 0 | 0 | 0 | 840 | 0 |
| 1e | 11 | 3 | 0 | 8 | 0.004224930 | 0 | 0 | 0 | 3409 | 0 |
| 1f | 4 | 0 | 0 | 4 | 0.000000000 | 0 | 0 | 0 | 2712 | 0 |
| 2a | 69 | 18 | 0 | 51 | 0.005680226 | 0 | 0 | 0 | 17921 | 0 |
| 2b | 49 | 13 | 0 | 36 | 0.001045180 | 0 | 0 | 0 | 83410 | 0 |
| 2c | 60 | 19 | 0 | 41 | 0.022930370 | 1 | 0 | 18 | 119152 | 0 |
| 2d | 132 | 35 | 1 | 96 | 0.018194923 | 0 | 0 | 57 | 8271 | 2 |
| 2e | 11 | 3 | 4 | 4 | -0.003920863 | 0 | 0 | 99243 | 92404 | 0 |
| 2f | 0 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| 2g | 1 | 1 | 0 | 0 | 0.000042513 | 0 | 0 | 0 | 4206 | 0 |
| 2h | 7 | 4 | 0 | 3 | 0.006142509 | 0 | 0 | 0 | 404 | 0 |

## d12 vs d11

| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 381 | 11 | 0 | 370 | 0.002800362 | 1 | 0 | 18 | 59636 | 0 |
| 1a | 3 | 0 | 0 | 3 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1b | 11 | 0 | 0 | 11 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1c | 17 | 0 | 0 | 17 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1d | 6 | 0 | 0 | 6 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1e | 11 | 0 | 0 | 11 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 1f | 4 | 0 | 0 | 4 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2a | 69 | 0 | 0 | 69 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2b | 49 | 0 | 0 | 49 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2c | 60 | 6 | 0 | 54 | 0.013488703 | 1 | 0 | 18 | 58652 | 0 |
| 2d | 132 | 4 | 0 | 128 | 0.001950143 | 0 | 0 | 0 | 922 | 0 |
| 2e | 11 | 0 | 0 | 11 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2f | 0 | 0 | 0 | 0 | NA | 0 | 0 | 0 | 0 | 0 |
| 2g | 1 | 0 | 0 | 1 | 0.000000000 | 0 | 0 | 0 | 0 | 0 |
| 2h | 7 | 1 | 0 | 6 | 0.000028125 | 0 | 0 | 0 | 62 | 0 |

## Provenance and timing

Run identity: `71413f80671f6019dd44948a19ef4d2639e34619824a7b31f371bd50cdc230a3`. Image: `sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f`.
d2/d3 derive independently from d1; d11 is collaborator semantic-v1; d12 is d11 followed by frozen strict-v2.
Anatomy review status: `PASS_PENDING_MANUAL_VISUAL_REVIEW`. No test generation or upload is authorized by this completion.
Actual compressed prediction storage: 0.783 GiB.

| Phase | Wall minutes |
|---|---:|
| baseline | 28.01 |
| postprocessing | 70.08 |

Worker-phase seconds (summed across processes):

```json
{
  "read_hash_seconds": 3781.8065057955682,
  "averaging_seconds": 652.0185020044446,
  "anatomy_support_seconds": 12044.692460983992,
  "nifti_write_seconds": 498.0889358147979,
  "scoring_seconds": 878.3928650394082,
  "validation_seconds": 4488.132871109992,
  "wall_seconds": 22434.012523926795
}
```

Detailed private finding rows, routing and ranked changes: `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r001_d1_val200_frozen_postprocessing/reports`.
Regenerate with the frozen `source/side_experiments/sideexp005_postprocessing_merge/runner.py report` in the recorded image.
