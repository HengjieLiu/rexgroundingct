---
created: 2026-09-09
updated: 2026-09-09
status: pending_user_review
---

# FP16 benchmark: 100 updates per arm

All four arms completed 100 measured optimizer updates and one full 69-finding
validation evaluation. These are diagnostic checkpoints; no model is ranked or
selected. Full training duration and evaluation milestones remain unset.

| Run | Dice A | Dice B | Dice full | Full Dice change | Full hits | Improved / unchanged / worsened | Mean absolute residual |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Cached base | 0.3394159 | 0.3349501 | 0.3372154 | 0 | 59/69 | — | 0 |
| run1_train | 0.3393829 | 0.3349299 | 0.3371887 | -0.0000267 | 59/69 | 24 / 2 / 43 | 0.0007236 |
| run2_train_val_a | 0.3394415 | 0.3349609 | 0.3372337 | +0.0000183 | 59/69 | 27 / 4 / 38 | 0.0031832 |
| run3_val_a | 0.3394275 | 0.3349695 | 0.3372308 | +0.0000154 | 59/69 | 23 / 43 / 3 | 0.0001918 |
| run4_val_all | 0.3394310 | 0.3349343 | 0.3372152 | -0.0000002 | 59/69 | 28 / 3 / 38 | 0.0034015 |

A contains 35 findings and B 34. A is training data for runs 2–4; B is training
data for run 4. Full-validation summaries for runs 2–3 mix training and held-out
findings. All arms retained the cached base hit counts: A 32/35, B 27/34, full
59/69 (85.507%). Findings count as improved/worsened even when Dice changes are
very small. These counts should be interpreted alongside the change magnitudes.

The maximum absolute full Dice change is 0.0000267 after this 100-update pilot.
No conclusion about converged fitting capacity follows from this short timing
run. Final interpretation and ranking remain with the user.

## Measured timing

| Run | 100-update seconds | Data-loading seconds | Peak training GiB | Evaluation seconds | Measured-update AMP retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| run1_train | 138.451 | 97.542 | 10.682 | 2319.153 | 2 |
| run2_train_val_a | 150.226 | 110.593 | 10.682 | 2320.204 | 0 |
| run3_val_a | 163.571 | 123.712 | 10.682 | 2319.871 | 0 |
| run4_val_all | 186.787 | 146.785 | 10.682 | 2319.619 | 1 |

Concurrent training group: 3.273 minutes including worker startup/warm-up.
Concurrent full evaluation group: 38.778 minutes.
Combined group wall time: 42.051 minutes.
This attempt's cache reuse/verification: 27.732 seconds; original export cost is documented separately in [the cache audit](cache_preparation_audit.md).

Each arm completed five successful synthetic warm-up updates, with one AMP
scale backoff, then restored pristine state. Measured-update retries did not
advance the update/sample cursor; all four checkpoints contain exactly 100
optimizer updates. FP32 comparison is tracked in the live report below.

[Full FP16 dashboard](runtime/reports/live_dashboard.md) · [FP16 / FP32 comparison](runtime/reports/precision_comparison.md)

All per-finding metrics, exposure labels, paired edits and timing records remain
under `runtime/benchmark/<arm>/evaluations/update_0000100/summary.json`.
