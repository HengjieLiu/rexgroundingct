---
created: 2026-08-02
updated: 2026-08-02
status: active
experiment_id: "013_voxtell_public_category_only_specialists"
---

# VoxTell Public Category-Only Specialists

Experiment 013 trains four single-GPU category specialists directly from the
public VoxTell v1.1 checkpoint. Unlike Exp012, every training event is targeted
to the arm's category group and no natural replay is used.

## Files

- Canonical config:
  `configs/experiments/013_voxtell_public_category_only_specialists.json`
- Execution spec: `codex_execution_spec.md`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists`

## Runtime Policy

Large schedules, logs, predictions, checkpoints, and raw evaluator outputs stay
under the runtime directory. The live Markdown and JSON reports are refreshed
after each synchronous evaluation barrier. The orchestrator stops at
`selection_ready` after epoch-100 val200 and does not launch any follow-up
evaluation automatically.
