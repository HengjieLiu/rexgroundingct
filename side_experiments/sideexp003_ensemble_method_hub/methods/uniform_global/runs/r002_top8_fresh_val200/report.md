# Top-8 Uniform Global Diagnostic

Status: `diagnostic_only`; selection and evaluation both use full val200.

Wall time: `98.22 min`.
No derived logits were materialized.

| Dice | Hits | Findings |
| ---: | ---: | ---: |
| 0.352344 | 291 | 381 |

## Timing

| Component | Seconds | Minutes |
| --- | ---: | ---: |
| Source read + float conversion + sigmoid | 5059.269 | 84.321 |
| Ensemble arithmetic/generation | 133.483 | 2.225 |
| Threshold + Dice/hits evaluation | 13.775 | 0.230 |
| Ground-truth reads | 46.946 | 0.782 |
| Aggregation/report preparation | 0.004 | 0.000 |
| Total active wall | 5893.068 | 98.218 |
| Total active CPU | 2584.081 | 43.068 |

## Official-category metrics

| Category | Findings | Dice | Hits | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| 1a | 3 | 0.124993 | 1 | 0.333333 |
| 1b | 11 | 0.173042 | 4 | 0.363636 |
| 1c | 17 | 0.188565 | 8 | 0.470588 |
| 1d | 6 | 0.124836 | 1 | 0.166667 |
| 1e | 11 | 0.198650 | 6 | 0.545455 |
| 1f | 4 | 0.159306 | 1 | 0.250000 |
| 2a | 69 | 0.352856 | 57 | 0.826087 |
| 2b | 49 | 0.365475 | 36 | 0.734694 |
| 2c | 60 | 0.402786 | 50 | 0.833333 |
| 2d | 132 | 0.393846 | 115 | 0.871212 |
| 2e | 11 | 0.449146 | 8 | 0.727273 |
| 2f | 0 | — | 0 | — |
| 2g | 1 | 0.041938 | 0 | 0.000000 |
| 2h | 7 | 0.256397 | 4 | 0.571429 |
