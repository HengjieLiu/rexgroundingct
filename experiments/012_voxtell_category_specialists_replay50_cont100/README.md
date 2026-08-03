---
created: 2026-08-01
updated: 2026-08-02
status: active
experiment_id: "012_voxtell_category_specialists_replay50_cont100"
---

# VoxTell Category Specialists

Experiment 012 is the umbrella for category-specialist continuations from the
Exp009 `baseline_cont100` model. Run profile `r01_core_replay50` contains the
original diffuse/2a/2b/2c arms. Profile `r02_2d_diffuse_replay_ratio` adds 2d
at 50% replay and independent diffuse schedules at 25%, 10%, and 0% replay.

## Files

- Canonical config:
  `configs/experiments/012_voxtell_category_specialists_replay50_cont100.json`
- Execution spec: `codex_execution_spec.md`
- Run-2 canonical profile: `run_specs/r02_2d_diffuse_replay_ratio.json`
- Final Run-1, Run-2, and combined reports: `reports/`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100`

## Runtime Policy

Large schedules, logs, predictions, checkpoints, and raw evaluator outputs stay
under the runtime directory. The live Markdown and JSON reports are refreshed
after each synchronous evaluation barrier. The orchestrator stops at
`selection_ready`; evaluating a selected pre-100 checkpoint is a separate user
decision. Completing one profile does not mark the Exp012 umbrella complete.
