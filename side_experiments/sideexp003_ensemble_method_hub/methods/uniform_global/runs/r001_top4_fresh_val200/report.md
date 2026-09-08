# Top-4 Uniform Global Diagnostic

Status: `diagnostic_only`; selection and evaluation both use full val200.

Wall time: `68.15 min`.
No derived logits were materialized.

| Dice | Hits | Findings |
| ---: | ---: | ---: |
| 0.357504 | 296 | 381 |

## Timing

| Component | Seconds | Minutes |
| --- | ---: | ---: |
| Source read + float conversion + sigmoid | 3098.836 | 51.647 |
| Ensemble arithmetic/generation | 87.326 | 1.455 |
| Threshold + Dice/hits evaluation | 15.425 | 0.257 |
| Ground-truth reads | 53.323 | 0.889 |
| Aggregation/report writing | 0.004 | 0.000 |
| Total active wall | 4089.250 | 68.154 |
| Total active CPU | 2078.072 | 34.635 |

Logical source bytes were 296,345,403,392 (275.99 GiB); process read bytes were
293,048,791,040 (272.92 GiB).

## Official-category metrics

| Category | Findings | Dice | Hits | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| 1a | 3 | 0.121354 | 1 | 0.333333 |
| 1b | 11 | 0.175401 | 4 | 0.363636 |
| 1c | 17 | 0.191700 | 8 | 0.470588 |
| 1d | 6 | 0.125988 | 1 | 0.166667 |
| 1e | 11 | 0.197126 | 7 | 0.636364 |
| 1f | 4 | 0.194984 | 2 | 0.500000 |
| 2a | 69 | 0.356342 | 58 | 0.840580 |
| 2b | 49 | 0.369032 | 37 | 0.755102 |
| 2c | 60 | 0.404281 | 51 | 0.850000 |
| 2d | 132 | 0.403234 | 115 | 0.871212 |
| 2e | 11 | 0.440992 | 8 | 0.727273 |
| 2f | 0 | — | 0 | — |
| 2g | 1 | 0.033450 | 0 | 0.000000 |
| 2h | 7 | 0.273465 | 4 | 0.571429 |
