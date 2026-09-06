---
created: 2026-09-03
updated: 2026-09-03
status: implementation
experiment_id: "021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100"
---

# Experiment 021: 0.7 mm Nodule-Only Positive Specialist

This run starts from the retained Exp017 epoch-100 network weights and keeps
the Exp017 iso07 preprocessing, DDP4 effective batch size, optimizer, loss, and
100-epoch learning-rate horizon. The supervision change is deliberate: only
official category `2d` pulmonary nodules/masses are positive targets. Other
findings and background remain present as zero-target nodule examples.

## Ownership

- Canonical config:
  `configs/experiments/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100.json`
- Execution spec:
  `experiments/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100/codex_execution_spec.md`
- Schedule generator:
  `scripts/rexgroundingct/prepare_021_nodule_schedule.py`
- Runner:
  `scripts/rexgroundingct/run_021_iso07_2d_nodule_specialist.sh`
- Human report generator:
  `scripts/rexgroundingct/summarize_021_nodule_results.py`
- External runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100`

## Data design

The released training split contains 1,305 of 2,992 cases with at least one
category-2d finding (43.6%), totaling 1,743 category-2d findings. The other
1,687 cases are sampled as nodule-negative image examples. In a positive event,
one 2d finding is the only positive prompt and two non-2d prompts are
zero-target negatives. In a negative event, a 2d query and two non-2d prompts
are all zero-target negatives.

The deterministic 40,000-event DDP schedule contains 30,000 positive and
10,000 nodule-negative-case events. This is a 75/25 event mix, with the exact
ratio enforced in each 400-event block consumed by one 100-update DDP epoch.

## Evaluation contract

Training pauses only at epochs 25, 50, 75, and 100. Every pause evaluates all
200 fixed validation cases and 381 findings. The human-readable report records
overall Dice/hit, every official category, and the nodule-specific category-2d
Dice/hit over 132 findings in 119 cases. The retained Exp017 epoch-100 result
is recorded as the initial nodule reference.

A successful run writes `.train_complete` in `ddp_bs4` and
`.experiment_complete` in the run group; a failed run writes `.train_failed`.
These markers are consumed by the Exp007 phase-3 autostart poller.

## Status

Implementation and smoke-gate launch are pending. Full training starts only
after the schedule, sample, DDP-resume, inference, and four-shard evaluation
smokes pass.
