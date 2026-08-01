---
created: 2026-08-01
updated: 2026-08-01
status: active
experiment_id: "012_voxtell_category_specialists_replay50_cont100"
---

# VoxTell Category Specialists With 50% Replay

Experiment 012 continues the Exp009 `baseline_cont100` model into four
category-routed specialists: one pooled 1a--1f arm and separate 2a, 2b, and 2c
arms. Every arm receives 50 targeted and 50 natural-replay updates per epoch.

## Files

- Canonical config:
  `configs/experiments/012_voxtell_category_specialists_replay50_cont100.json`
- Execution spec: `codex_execution_spec.md`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100`

## Runtime Policy

Large schedules, logs, predictions, checkpoints, and raw evaluator outputs stay
under the runtime directory. The live Markdown and JSON reports are refreshed
after each synchronous evaluation barrier. The orchestrator stops at
`selection_ready`; evaluating a selected pre-100 checkpoint is a separate user
decision.
