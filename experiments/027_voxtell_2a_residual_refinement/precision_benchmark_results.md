---
created: 2026-09-09
updated: 2026-09-09
status: complete_pending_user_review
---

# Matched FP16 / FP32 benchmark results

Both precision trials completed all four arms: five successful synthetic warm-up
updates followed by pristine-state restoration, 100 actual optimizer updates,
then one complete evaluation of 69 validation findings on 63 CTs. FP16 completed
at 2026-09-10 05:04:39 UTC; FP32 completed at 05:46:29 UTC. The sequential
controller finished successfully and all four GPUs were idle at the completion
audit. No full training was launched.

## Validation results

| Run | Precision | Dice A | Dice B | Dice full | Change from cached base, full | Hits full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cached base | Same inputs for both | 0.3394159 | 0.3349501 | 0.3372154 | 0 | 59/69 |
| run1_train | FP16 | 0.3393829 | 0.3349299 | 0.3371887 | -0.0000267 | 59/69 |
| run1_train | FP32 | 0.3393995 | 0.3349232 | 0.3371938 | -0.0000216 | 59/69 |
| run2_train_val_a | FP16 | 0.3394415 | 0.3349609 | 0.3372337 | +0.0000183 | 59/69 |
| run2_train_val_a | FP32 | 0.3394430 | 0.3349618 | 0.3372349 | +0.0000195 | 59/69 |
| run3_val_a | FP16 | 0.3394275 | 0.3349695 | 0.3372308 | +0.0000154 | 59/69 |
| run3_val_a | FP32 | 0.3394278 | 0.3349819 | 0.3372371 | +0.0000217 | 59/69 |
| run4_val_all | FP16 | 0.3394310 | 0.3349343 | 0.3372152 | -0.0000002 | 59/69 |
| run4_val_all | FP32 | 0.3394398 | 0.3349639 | 0.3372343 | +0.0000189 | 59/69 |

A contains 35 findings; B contains 34. A is refiner training data for runs 2–4;
B is training data for run 4. Full results for runs 2–3 mix training and held-out
findings. Every arm/precision retains A 32/35 hits, B 27/34 hits and full 59/69
hits (85.507%). Run order is fixed and does not represent a ranking.

| Run | FP32 minus FP16 full Dice | Maximum absolute per-finding Dice difference |
| --- | ---: | ---: |
| run1_train | +0.000005107 | 0.000322989 |
| run2_train_val_a | +0.000001166 | 0.000213179 |
| run3_val_a | +0.000006300 | 0.000275480 |
| run4_val_all | +0.000019031 | 0.000400357 |

Aggregate changes after 100 updates are very small in both precisions. These
timing checkpoints do not establish converged fitting capacity, and no precision
or checkpoint is selected automatically.

## Training and numerical stability

| Run | Precision | Measured 100 updates, s | Loading, s | Mean step excluding loading, s | Peak training GiB | Measured-update AMP backoffs | Warm-up AMP backoffs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run1_train | FP16 | 138.451 | 97.542 | 0.386 | 10.682 | 2 | 1 |
| run1_train | FP32 | 108.154 | 8.010 | 0.990 | 11.975 | 0 | 0 |
| run2_train_val_a | FP16 | 150.226 | 110.593 | 0.378 | 10.682 | 0 | 1 |
| run2_train_val_a | FP32 | 104.926 | 8.080 | 0.956 | 11.975 | 0 | 0 |
| run3_val_a | FP16 | 163.571 | 123.712 | 0.378 | 10.682 | 0 | 1 |
| run3_val_a | FP32 | 105.963 | 8.648 | 0.957 | 11.975 | 0 | 0 |
| run4_val_all | FP16 | 186.787 | 146.785 | 0.383 | 10.682 | 1 | 1 |
| run4_val_all | FP32 | 107.685 | 8.579 | 0.977 | 11.975 | 0 | 0 |

FP16 recovered three scaled-gradient overflows during measured training (run 1:
two; run 4: one) and one synthetic warm-up overflow per arm. Each overflowing
attempt was skipped and the same patch retried at lower AMP scale; none was
counted as an optimizer update. FP32 used no loss scaling and had no non-finite
loss/gradient/residual failures during warm-up, training or complete evaluation.
Neither precision encountered OOM.

The later FP32 trial loaded the same schedules much faster (about 8–9 seconds
per 100 updates versus 98–147 seconds for FP16). Warmer filesystem caches or
changed storage contention are plausible contributors; this was not a
counterbalanced I/O experiment. The shorter FP32 total training wall time
therefore does not establish faster FP32 computation. After subtracting recorded
loading, steps took about 0.38–0.39 s in FP16 and 0.96–0.99 s in FP32. This
includes device transfers, forward/backward, optimizer and checks; it is not
isolated CUDA-kernel profiling.

## Evaluation and complete group timing

| Precision | Concurrent four-arm training, min | Concurrent four-arm evaluation, min | Combined train + eval, min | Cache reuse/check, s |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 3.273 | 38.778 | 42.051 | 27.732 |
| FP32 | 2.023 | 38.815 | 40.838 | 28.800 |

Group training time includes worker startup and synthetic warm-up; per-arm
measured updates above exclude those costs. Cache reuse/check is separate and
does not represent the original logit generation cost.

| Run | Precision | Evaluation total, min | Inference loop, s | Metrics, s | Output, s | Peak evaluation GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| run1_train | FP16 | 38.653 | 2095.535 | 218.444 | 0.495 | 2.177 |
| run1_train | FP32 | 38.702 | 2101.567 | 214.948 | 1.092 | 1.767 |
| run2_train_val_a | FP16 | 38.670 | 2091.089 | 224.744 | 0.396 | 2.177 |
| run2_train_val_a | FP32 | 37.930 | 2061.308 | 206.120 | 1.203 | 1.767 |
| run3_val_a | FP16 | 38.665 | 2097.560 | 217.236 | 0.404 | 2.177 |
| run3_val_a | FP32 | 38.131 | 2068.031 | 214.787 | 0.852 | 1.767 |
| run4_val_all | FP16 | 38.660 | 2110.887 | 204.437 | 0.375 | 2.177 |
| run4_val_all | FP32 | 38.063 | 2077.015 | 201.470 | 1.142 | 1.767 |

The inference-loop timer includes CPU tile extraction and Gaussian blending as
well as GPU forward passes. Evaluation timing is therefore an end-to-end pipeline
measurement, not a GPU-only precision comparison. Peak allocations depend on
backend kernels/workspaces and need not scale directly with tensor element size.

## Matched inputs and completion checks

The audit verified both complete timing reports, exactly 100 ordered updates
per arm, all 69 unique finding records per evaluation, checkpoint hashes, exact
metric recomposition, identical prepared data, identical per-arm schedules and
initial model hashes, and unchanged production source fingerprints. Configs
differ only in AMP and runtime directory; FP32 additionally disables TF32 with
the recorded `NVIDIA_TF32_OVERRIDE=0` environment. The frozen base-cache storage
remains FP16 in both experiments. All final weights and optimizer/scaler/RNG
checkpoints are preserved.

[Detailed live comparison and JSON/CSV](runtime/reports/precision_comparison.md) · [FP16 dashboard](runtime/reports/live_dashboard.md) · [FP32 dashboard](runtime/precision_fp32/reports/live_dashboard.md)

Experiment status is **awaiting_schedule_decision**. Precision, total updates
and evaluation milestones await the user decision; ranking and final
interpretation remain **pending user review**.
