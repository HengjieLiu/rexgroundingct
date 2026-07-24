---
doc_type: ai_workflow_summary
created: 2026-07-23
updated: 2026-07-23
status: active
source_repo: /home/hengjie/code_sync/rexgroundingct
source_commit: 2ac8fc61efe4
source_commit_full: 2ac8fc61efe48d7a8c1c6b58d086048c4168233f
source_commit_date: 2026-07-23 23:26:38 -0700
source_commit_subject: Add VoxTell research brainstorm notes
inspection_date: 2026-07-23
scope: Portable snapshot of the ReXGroundingCT AI-assisted research workflow.
related_code:
  - AGENTS.md
  - docs/ai_workflow.md
  - docs/current_status.md
  - docs/templates/
  - docs/codex_prompts/
  - experiments/
---

# AI-Assisted Research Workflow Summary - ReXGroundingCT

## Executive Summary

ReXGroundingCT adapted lessons from DoseRad2026 and inrorspline into a smaller
AI-assisted workflow suited to a fast-moving medical-image grounding project.
The central choice was to add durable research memory without reorganizing the
codebase: agents read a compact operating guide, substantial work starts from
an execution spec, expensive runs pass a small gate, runtime artifacts stay
outside Git, and results return to the repo as small summaries and decisions.

This snapshot exports that workflow for future projects. It describes the
tracked state at source commit `2ac8fc61efe4`; it does not include later
documentation changes or the unrelated untracked brainstorm present during
inspection.

## Adopted Repository Structure

The workflow is distributed across a few intentionally small layers:

- `AGENTS.md` and `AGENTS/` define required reading, environment, data,
  experiment, hardware, and commit rules.
- `docs/ai_workflow.md` is the operating loop, while
  `docs/current_status.md` holds active work, settled decisions, blockers, and
  next actions.
- `docs/templates/` provides an execution-spec template and a concise
  experiment-closeout template.
- `docs/codex_prompts/` stores polished reusable prompts rather than raw chat
  transcripts.
- `configs/experiments/` owns canonical configs, and `experiments/` is a
  repo-local index containing specs, small metrics summaries, sync manifests,
  reports, and decisions.
- CT volumes, checkpoints, predictions, logs, model weights, and other heavy
  runtime artifacts live under external data and experiment roots.

This structure preserves a clear boundary between intent, configuration,
execution evidence, and heavyweight outputs. It also lets a future agent
recover the project state without relying on prior conversation history.

## Evidence From Current Use

The workflow was used immediately after it was introduced. Experiment 002
maintains an active `codex_execution_spec.md` that records the objective,
canonical decisions, prior evidence, exact paths and commands, smoke gates,
success criteria, verification, and closeout. Its reproducibility policy uses a
fixed seed, a fixed 20-case validation probe, and materialized training
schedules so batch-size or DDP comparisons do not silently change data order.

Experiment 003 extends the same pattern to a four-arm rescue ablation. It fixes
the training schedule per variant, reuses immutable val20 and val200 probes,
checks positive-crop behavior before long runs, and requires immutable
checkpoints at planned update counts. A failed launch was preserved under a
failed-runs location instead of being overwritten. Detached evaluation
sidecars use lock guards, and small synchronized manifests and metrics summaries
make runtime state visible without committing the runtime tree.

The project also keeps dynamic conclusions in `docs/current_status.md`, archives
the workflow-upgrade prompt for reuse, and exposes one canonical repository
check. That check runs challenge-source tests, experiment consistency checks,
Python compilation, Git whitespace checks, and tracked-cache detection. These
practices turn the written workflow into observable project behavior.

## Portable Workflow

Future research repositories can adapt the following loop:

```text
request -> execution spec -> smoke gate -> run -> sync/report -> decision
```

1. Ground the request by reading agent rules, relevant folder guides, current
   status, configs, prior reports, and the dirty working tree.
2. Save a compact execution spec for substantial implementation, long runs,
   data movement, or submission work. Pin inputs, paths, commands, metrics,
   artifacts, and stop conditions.
3. Run the smallest useful gate before consuming significant compute. Make the
   gate test the riskiest assumption, not merely process startup.
4. Keep runtime outputs outside Git, snapshot canonical configs at launch, and
   preserve failures rather than rewriting their history.
5. Sync only small intentional artifacts back into the repository and run
   consistency checks.
6. Update the durable decision or current-status page only when evidence changes
   what should happen next.

## Preserve, Adapt, And Avoid

Preserve stable experiment IDs, explicit source commits, immutable comparison
inputs, exact-path staging, honest failure records, and the separation between
local provenance and external runtime data. Treat local summaries as durable
truth even when dashboards or detached monitors provide live visibility.

Adapt the folder names, required metadata, smoke gates, metrics, and runtime
roots to the project. A small project does not need a large documentation
spine; the minimum useful set is an agent guide, current status, execution-spec
template, experiment index, canonical check, and artifact policy.

Avoid launching expensive work from vague chat, changing several scientific
variables without an explicit matrix, overwriting failed runs, treating a
small fixed probe as final evidence, committing private or heavyweight
artifacts, and using broad staging commands in a dirty research tree.

## Improvement Candidates

The live workflow should next absorb practices proven during experiment 003:
quarantining failed runs, lock-safe and idempotent sidecars, immutable schedules
and evaluation probes, and running planned evaluation soon after each
checkpoint becomes available. Agent-facing experiment guidance should also be
reconciled with experiment 003, and stabilization language should describe the
current fine-tuning workflow rather than experiment 002 alone.

These are focused documentation improvements, not reasons to restructure the
scripts or expand process overhead. The reusable lesson is to promote a
practice into workflow guidance only after the project demonstrates that the
practice prevents real ambiguity, lost evidence, or wasted compute.

## Provenance Note

At inspection, `main` was ahead of `origin/main` by six commits and the working
tree contained an unrelated untracked brainstorm. That file was excluded from
this source snapshot. The snapshot intentionally records the inspected source
commit rather than its own later documentation commit, avoiding circular
provenance.
