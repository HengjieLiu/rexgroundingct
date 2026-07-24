# Exp004 2 mm Epoch-100 Early Val200 Result

The completed `iso2mm_global192_cont100` arm was evaluated before the native
control finished. Evaluation used the canonical fixed val200 JSON, threshold
0.5, and saved both masks and float32 probability maps.

| Metric | Result |
| --- | ---: |
| Cases | 200 |
| Findings | 381 |
| Mean Dice per finding | 0.171086 |
| Mean Dice per case | 0.167158 |
| Hit rate | 0.419948 |
| Hits | 160/381 |

Compared with the original exp003 v123 epoch-100 val200 baseline:

| Model | Dice per finding | Hit rate | Hits |
| --- | ---: | ---: | ---: |
| Exp003 v123 epoch 100 | 0.283343 | 0.690289 | 263/381 |
| Exp004 2 mm epoch 100 | 0.171086 | 0.419948 | 160/381 |

At the primary threshold 0.5, the 2 mm continuation is substantially worse
than its initialization baseline.

Canonical runtime output:

`runs/exp004_paired_20260724T081000Z/iso2mm_global192_cont100/eval_epoch100_val200`

The early sidecar used the same `.eval.lock` and completeness contract as the
original coordinator. All 200 predictions, 200 probabilities, evaluator JSON,
and summary were verified before the lock was released. The original
post-baseline finalization will therefore skip this arm and evaluate only the
still-missing native val200 before generating the paired report.
