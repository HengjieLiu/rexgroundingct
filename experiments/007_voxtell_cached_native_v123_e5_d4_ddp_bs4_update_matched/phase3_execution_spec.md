---
created: 2026-09-05
updated: 2026-09-05
status: queued_behind_exp021
experiment_id: "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
phase: "phase3_from_abs_e200"
---

# Exp007 Phase 3: Continuation From Absolute Epoch 200

## Objective

Measure a fresh 100-epoch Exp007 continuation initialized from the retained
phase-2 absolute-epoch-200 checkpoint, using the same training method and a
fresh learning-rate schedule.

## Source and load policy

- Source run group:
  `exp007_cont100_from_ddp100_20260730T062051Z`
- Source checkpoint:
  `runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/checkpoints/checkpoint_update_010000.pth`
- Expected source SHA-256:
  `d56474a645f90e827af14e116454362d146db835ef5cf111120fb29ab8b78c0c`
- Expected source global update: `10000`.
- Load only network weights. Reset optimizer, AMP scaler, scheduler, RNG
  training state, and update counter.

## Training contract

- Same cached-native v123 preprocessing, 192³ patches, model, loss, SGD
  Nesterov optimizer, momentum, weight decay, encoder/decoder learning rates,
  gradient clipping, and deep-supervision settings as Exp007 phase 2.
- DDP world size 4, per-GPU batch size 1, gradient accumulation 1, effective
  global batch size 4.
- 100 relative epochs, 100 updates per epoch, 10,000 optimizer updates.
- Fresh 100-update warmup and polynomial power-0.9 schedule over 10,000
  updates.

## Deterministic schedule

Generate or verify a 120,000-event schedule with the established Exp007 seed
and preprocessing settings. Verify its first 80,000 events byte-for-byte
against the existing phase-2 extended schedule. Train on source events
`[80000, 120000)`, reindexed to local event indices `0..39999`.

The continuity check and slice materialization are implemented by
`scripts/rexgroundingct/prepare_007_phase3_schedule.py`.

The schedule manifest records source/output hashes, the source event range,
and the DDP rule `event_index = update * 4 + rank`.

## Evaluation and live reporting

At each relative milestone `10, 20, ..., 100`, pause training and evaluate
all 200 fixed validation cases and 381 findings on four GPUs. Use inference
threshold `0.5` and the existing ReXRank evaluator hit threshold `Dice >= 0.1`.

The live report is updated atomically after every milestone and includes:

- overall Dice, hit count, and hit rate;
- all official category metrics;
- category `2d` nodule Dice, hits, and support;
- current training update, state, completed milestone, and ETA;
- the phase-2 absolute-epoch-200 result as the initial reference.

Runtime reports:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/phase3_status.md
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/phase3_status.json
```

## Exp021 gate and poller

The poller checks Exp021 every 1,800 seconds. It requires the current Exp021
group to contain the explicit completion markers, final update-10,000
checkpoint, complete epoch-100 val200 evaluator artifacts, 200 predictions,
and a complete nodule-specialist summary. Partial logs or a partial e100
evaluation do not pass the gate.

After the gate passes, the poller verifies Docker/image availability and
launches exactly one detached phase-3 container. It records the observed
Exp021 group, every gate check, phase-3 run group, container name, and launch
state under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/phase3_autostart/
```

Install the user service after the repository is deployed:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/rexgroundingct/exp021_to_exp007_phase3_poller.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now exp021_to_exp007_phase3_poller.service
systemctl --user status exp021_to_exp007_phase3_poller.service
```

For a one-shot gate check without launching, use:

```bash
NO_LAUNCH=1 POLL_ONCE=1 bash scripts/rexgroundingct/poll_exp021_then_launch_007_phase3.sh
```

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/summarize_007_phase3_results.py
bash -n scripts/rexgroundingct/run_007_ddp_bs4_update_matched.sh
bash -n scripts/rexgroundingct/run_007_phase3_from_abs_e200.sh
bash -n scripts/rexgroundingct/run_007_phase3_from_abs_e200_docker.sh
bash -n scripts/rexgroundingct/poll_exp021_then_launch_007_phase3.sh
```
