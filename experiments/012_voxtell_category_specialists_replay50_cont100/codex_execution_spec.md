---
created: 2026-08-01
updated: 2026-08-02
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

For every sentinel milestone, the report publishes both total fixed-val80
metrics over the same 80 cases and 195 findings for cross-arm comparison and
arm-excluded fixed-val80 metrics for within-arm forgetting analysis. Epoch 5
has neither val80 view because it is target-only. Both val80 views are
diagnostic. Selection uses target mean global Dice only. The 1a--1f arm selects
one pooled checkpoint. Dice ties within `1e-6` use target hit rate and then the
earlier epoch. Epoch 0 remains a valid winner.

## Stop Condition

After all four epoch-100 val200 evaluations, refresh the report with target,
sentinel, non-target, per-category, and overall metrics; record recommended
checkpoints; write `.selection_ready`; and stop. Do not automatically evaluate
an earlier selected checkpoint on val200.

## Run Profile 2: 2d and Diffuse Replay Ratio

Profile `r02_2d_diffuse_replay_ratio` remains part of Exp012. It maps 2d
replay50 to GPU 0 and diffuse replay25/replay10/replay00 to GPUs 1/2/3. Those
diffuse labels mean 75/90/100 targeted events and 25/10/0 natural-replay events
per 100-update epoch.

Run 2 uses master seed `20260802`, independent target and placement seeds for
every arm, and disjoint replay source ranges `25000:30000`, `30000:32500`, and
`32500:33500`; replay00 consumes none. Its schedules are intentionally not
nested or sample-paired with Run 1 or with one another.

The milestone state machine and target-only selection policy are unchanged.
Diffuse arms reuse the 42-case/52-target census and 80-case union. The 2d arm
uses an exhaustive 119-case/132-target census and a 160-case/321-finding union.
The report includes Run-1 diffuse replay50 as a frozen independent historical
reference. Run 2 stops at its own `.selection_ready`, while Exp012 remains
active for later profiles.
