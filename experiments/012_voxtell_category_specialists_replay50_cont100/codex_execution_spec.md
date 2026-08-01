---
created: 2026-08-01
updated: 2026-08-01
status: active
experiment_id: "012_voxtell_category_specialists_replay50_cont100"
---

# Experiment 012 Execution Specification

## Objective

Train four single-GPU category specialists from Exp009 `baseline_cont100` with
an exact 50/50 targeted/replay event schedule, monitor their complete target
censuses at fixed milestones, and stop after epoch-100 val200 with a
target-only checkpoint recommendation.

## Arms

| GPU | Arm | Target |
| ---: | --- | --- |
| 0 | `category_1alldiffuse_replay50` | 1a--1f |
| 1 | `category_2a_replay50` | 2a |
| 2 | `category_2b_replay50` | 2b |
| 3 | `category_2c_replay50` | 2c |

All arms initialize network weights from Exp009 run group
`exp009_s3_attention_20260727T081747Z`, arm `baseline_cont100`, epoch 100.
Optimizer, momentum, scaler, RNG, update count, warmup, and polynomial horizon
start fresh. Segment resumes restore their complete Exp012 state.

## Training Contract

- 100 epochs, 100 updates per epoch, batch 1, no DDP.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- Encoder/decoder LR `1e-5/1e-4`, 100-update warmup, polynomial power `0.9`.
- Native z-score cache, `192^3` patches, v123 loss, gradient clipping `12`.
- Immutable checkpoints at updates `500/2000/4000/6000/8000/10000`.
- Every epoch contains exactly 50 targeted and 50 natural-replay events.
- Replay events are shared across arms and copied from source events
  `20000--24999`. Target events guarantee the requested finding in the crop.
- The pooled 1a--1f target allocation is
  `791/864/1087/716/912/630`; other target arms sample findings uniformly.

## Milestone State Machine

At each milestone all trainers exit, save full state, evaluate concurrently on
their assigned GPUs, wait at a global barrier, atomically refresh the live
report, and then resume.

| Epoch | Evaluation |
| ---: | --- |
| 0 | target census plus fixed-val80 non-target sentinel |
| 5 | target census only |
| 20, 40, 60, 80 | target census plus sentinel |
| 100 | full val200 only |

The sentinel is diagnostic. Selection uses target mean global Dice only. The
1a--1f arm selects one pooled checkpoint. Dice ties within `1e-6` use target hit
rate and then the earlier epoch. Epoch 0 remains a valid winner.

## Stop Condition

After all four epoch-100 val200 evaluations, refresh the report with target,
sentinel, non-target, per-category, and overall metrics; record recommended
checkpoints; write `.selection_ready`; and stop. Do not automatically evaluate
an earlier selected checkpoint on val200.
