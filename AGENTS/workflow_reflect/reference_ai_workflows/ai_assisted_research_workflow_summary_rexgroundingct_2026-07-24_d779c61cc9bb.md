---
doc_type: ai_workflow_design_report
created: 2026-07-24
updated: 2026-07-24
status: draft
source_repo: /home/hengjie/code_sync/rexgroundingct
source_commit: d779c61cc9bb
source_commit_full: d779c61cc9bb8d8c49f174ad6a2576a010700492
source_commit_date: 2026-07-24 11:56:47 -0700
source_commit_subject: Record VoxTell experiments 004 and 005 summaries
inspection_date: 2026-07-24
scope: Comprehensive design, organization, and adoption blueprint for an AI-assisted research repository.
predecessor: ai_assisted_research_workflow_summary_rexgroundingct_2026-07-23_2ac8fc61efe4.md
related_code:
  - AGENTS.md
  - AGENTS/README.md
  - AGENTS/workflow.md
  - AGENTS/command.md
  - AGENTS/workflow_reflect/
  - docs/ai_workflow.md
  - docs/current_status.md
  - docs/templates/
  - experiments/
  - scripts/rexgroundingct/check_repo_workflow.py
---

# AI-Assisted Research Repository Design - ReXGroundingCT

## Executive Summary

ReXGroundingCT evolved from a lightweight research repository into a clearer
AI-assisted operating system without reorganizing its scientific code. The
first workflow snapshot emphasized durable memory: execution specs before
expensive work, smoke gates, external runtime storage, small synced summaries,
and status updates after evidence changed. The next improvement separated three
different kinds of agent guidance that had begun to compete for the same
location:

- durable rules in `AGENTS/workflow.md`;
- reusable task commands in `AGENTS/command.md`;
- workflow evidence and improvement material under
  `AGENTS/workflow_reflect/`.

That separation is the main organizational lesson in this report. Rules,
commands, reflection, project planning, execution evidence, and runtime
artifacts have different audiences and rates of change. Giving each one a clear
home makes the repository easier for future agents to read, safer to modify,
and simpler for another project to adapt.

This report describes committed state at source commit `d779c61cc9bb`. It is a
comprehensive design and adoption blueprint, not a replacement for the live
rules. The earlier July 23 export remains the concise historical snapshot.

## Evolution Of The Design

The repository's first workflow layer came from two mature AI-assisted research
projects. It adopted only the pieces that addressed immediate risks:

- a short agent entry contract and required reading order;
- current-status and experiment-index pages;
- execution-spec and closeout templates;
- canonical configs with runtime snapshots;
- small synchronized reports and manifests;
- a canonical repository check;
- strict exclusion of medical data and heavyweight runtime products.

This worked, but reusable prompts, commit procedures, workflow lessons, and
durable rules were initially too close together. A future agent could not
quickly tell whether a document was mandatory policy, an optional shortcut, or
historical evidence.

Commit `5f85ec0` made the boundary explicit. It introduced `AGENTS/command.md`
for short recurring task patterns and moved workflow-upgrade material and
reference reports into `AGENTS/workflow_reflect/`. `AGENTS/workflow.md` remained
the stable policy layer. The live `docs/ai_workflow.md` was updated to explain
how these guide types participate in research work.

This is a structural improvement rather than a cosmetic move. It prevents a
useful command from silently becoming a universal rule, prevents historical
reflection from bloating every agent session, and preserves a place where the
workflow itself can be inspected before new guidance is promoted.

## Design Principles

### Separate Artifacts By Responsibility

Every tracked artifact should answer one primary question:

- What must agents always do?
- What is a repeatable way to handle a named request?
- What is known about the project or environment?
- What work should happen next?
- What exactly will this experiment do?
- What actually ran and what did it produce?
- What did the team learn about its own workflow?

When a file tries to answer several of these questions, it becomes harder to
maintain and easier to misuse. ReXGroundingCT therefore organizes files by
responsibility rather than by whether they were authored by a human or an AI.

### Keep Durable Memory Small

The repository does not archive raw conversations. It records the minimum
project state needed to resume safely: decisions, constraints, exact configs,
execution specs, small metrics, manifests, reports, and next actions. This
keeps agent reading costs proportional to the task while preserving
reproducibility.

### Separate Canonical Intent From Runtime Evidence

Tracked configs and specs state what should run. Runtime snapshots, logs,
checkpoints, predictions, and evaluator outputs record what did run. Small
summaries bridge the two. This boundary prevents large or private artifacts
from entering Git and prevents runtime edits from becoming accidental changes
to the intended experiment.

### Promote Rules Only After Evidence

`workflow_reflect/` is the staging area for workflow knowledge. Imported
reports, local snapshots, and dated upgrade prompts can propose improvements,
but they do not automatically become policy. A practice moves into
`workflow.md` only when repeated work shows that it prevents ambiguity, lost
evidence, unsafe changes, or wasted compute.

## Repository Organization

The following tree shows the responsibility boundaries rather than every file:

```text
repo/
├── AGENTS.md                         # minimal entry contract
├── AGENTS/
│   ├── README.md                     # reading order and guide map
│   ├── workflow.md                   # durable cross-task rules
│   ├── command.md                    # reusable named task procedures
│   ├── environment.md                # execution environment and mounts
│   ├── data.md                       # dataset and Git artifact policy
│   ├── experiments.md                # experiment identity and sync rules
│   ├── literature_and_code.md        # scientific and implementation map
│   ├── hardware.md                   # capacity and storage assumptions
│   ├── goals.md                      # project direction and open decisions
│   └── workflow_reflect/
│       ├── README.md                 # reflection boundary and versioning
│       ├── ai_workflow_upgrade_*.md  # dated workflow-improvement prompts
│       └── reference_ai_workflows/   # imported evidence and local exports
├── docs/
│   ├── README.md                     # human and agent documentation map
│   ├── ai_workflow.md                # end-to-end operational lifecycle
│   ├── current_status.md             # active work, decisions, blockers, next
│   ├── templates/                    # execution and closeout scaffolds
│   ├── submission.md                 # challenge-specific release runbook
│   └── domain notes                  # preprocessing, methods, brainstorms
├── configs/experiments/              # canonical machine-readable intent
├── experiments/
│   ├── registry.yaml                 # machine-readable repo-local index
│   └── <experiment-id>/              # spec, summaries, reports, manifests
├── scripts/rexgroundingct/           # implementation and launch entrypoints
└── external runtime roots            # data, logs, models, predictions, runs
```

### Agent Entry And Context

`AGENTS.md` stays intentionally short. It tells an agent how to acquire
context, not how to perform every task. `AGENTS/README.md` gives the reading
order, and the focused context guides describe environment, data, experiments,
literature, hardware, and goals. This makes project truth discoverable without
putting all details into one global instruction file.

### Workflow Rules

`AGENTS/workflow.md` contains invariants that should survive individual
experiments: preserve unrelated work, use explicit path staging, keep
heavyweight artifacts out of Git, write specs before substantial runs, and run
the appropriate checks. Its contents should change slowly.

### Reusable Commands

`AGENTS/command.md` contains procedures triggered by recognizable user intent.
The topic-clustered commit preparation command is an example: it defines a
repeatable inspection, grouping, approval, staging, and verification sequence.
A command is more detailed than a rule but narrower in scope. Commands remain
subject to workflow rules and do not bypass approval or safety boundaries.

### Workflow Reflection

`AGENTS/workflow_reflect/` stores material about improving the operating system
itself. Reference summaries show how other repositories work. Local exports
capture this repository at a source commit. Dated upgrade prompts preserve the
reasoning used to convert evidence into proposed changes. Reflection is read
when reviewing the workflow, not necessarily for every coding task.

### Operational And Scientific Documentation

`docs/ai_workflow.md` connects the guide layers to the research lifecycle.
`docs/current_status.md` provides volatile project state: what is active, what
is settled, what is blocked, and what should happen next. Domain-specific docs
hold scientific reasoning that should not be mistaken for universal workflow
policy.

### Experiment And Runtime Layers

Each experiment has a stable ID, a canonical config, a repo-local folder, and
an external runtime root. The repo-local folder owns the execution spec,
human-readable scientific contract, small summaries, manifests, and curated
reports. The external root owns large logs, checkpoints, predictions, cached
data, and evaluator products. Synchronization is deliberate and one-way for
curated evidence.

## File Placement Guide

| Information | Correct home | Lifecycle |
| --- | --- | --- |
| Cross-task invariant | `AGENTS/workflow.md` | Rarely changed; mandatory |
| Repeatable response to a named request | `AGENTS/command.md` | Added when a task pattern recurs |
| Environment, data, or hardware fact | Focused `AGENTS/*.md` guide | Updated when project truth changes |
| Long-horizon direction or open choice | `AGENTS/goals.md` | Reviewed at project milestones |
| Active work, blocker, or next action | `docs/current_status.md` | Updated when evidence changes direction |
| Workflow critique or imported lesson | `AGENTS/workflow_reflect/` | Dated evidence; not automatically policy |
| Substantial task or experiment plan | `codex_execution_spec.md` | Written before implementation or launch |
| Scientific method contract | Experiment method/training spec | Stable across comparable runs |
| Machine-readable intended settings | `configs/experiments/` | Canonical and version controlled |
| Actual command, environment, and output | External runtime manifest/log | Written during execution |
| Curated result and provenance | Repo-local report/summary/manifest | Synced after verification |

Project planning is deliberately cross-layer rather than a fourth guide type.
`AGENTS/goals.md` states the longer horizon, `docs/current_status.md` states the
current decision frontier, and an execution spec converts one approved unit of
work into an implementable plan. A training or method spec then freezes the
scientific contract independently of the orchestration details.

## End-To-End AI-Assisted Lifecycle

The complete lifecycle is:

```text
request
  -> optional reusable command
  -> workflow constraints
  -> project and domain context
  -> project decision / execution spec
  -> canonical config and smoke gate
  -> runtime snapshot and execution
  -> checkpoint/evaluation evidence
  -> repo-local sync and consistency check
  -> result interpretation and next decision
  -> workflow reflection and selective promotion
```

1. **Ground the request.** Read the agent contract, guide map, relevant folder
   READMEs, current status, experiment material, and dirty tree. Determine
   whether the work is a small edit, substantial implementation, experiment
   launch, data movement, submission task, or workflow change.
2. **Route recurring work.** If a named command applies, use it as a procedure.
   The command cannot override durable workflow or safety rules.
3. **Make planning explicit.** Connect the request to a project goal and current
   decision. For substantial work, write an execution spec with scope, inputs,
   outputs, paths, commands, success criteria, stop conditions, and closeout.
4. **Freeze comparable intent.** Use a canonical config, fixed dataset/probe,
   deterministic schedule where needed, and an explicit scientific contract.
5. **Pass the smallest meaningful gate.** Test the riskiest assumption before
   consuming significant compute: orientation, target sampling, memory,
   geometry, loss behavior, or evaluator compatibility.
6. **Snapshot and run.** Copy the canonical config into the runtime root,
   record the source commit and command, and keep logs and heavy outputs
   external.
7. **Evaluate safely.** Treat checkpoints as immutable evidence. Give one
   process ownership of each evaluation, use locks and completeness predicates,
   and make reruns idempotent.
8. **Sync curated evidence.** Bring back only small reports, metrics, manifests,
   and decisions. Check that repo and runtime provenance agree.
9. **Close the decision loop.** Update current status only when the evidence
   changes the next action. Preserve caveats, failures, and incomplete work.
10. **Reflect selectively.** If a repeated problem reveals a workflow gap,
    document it under reflection first, then promote the proven rule or command
    into the live guide.

## Evidence From Experiments 002 Through 005

Experiment 002 demonstrated the value of planning and reproducibility policy.
Its execution spec captured the baseline, orientation assumptions, fixed seed,
val20 probe, materialized schedules, paths, launch commands, success criteria,
and closeout. This separated the scientific question from ad hoc launcher
behavior.

Experiment 003 showed why failure preservation and orchestration ownership
matter. Its four variants shared immutable schedules and fixed val20/val200
orders. A failed raw-bbox launch was quarantined rather than overwritten.
Evaluation sidecars used `.eval.lock` guards, and intermediate checkpoints were
kept as immutable evidence. These practices made a multi-day detached run
recoverable across agent sessions.

Experiment 004 strengthened the design with paired-arm contracts and explicit
evaluation ownership. The native and 2 mm arms shared network initialization
and deterministic events, while the geometry difference was documented
precisely. Training wrote atomic checkpoints; a coordinator and later sidecars
owned locks and completeness checks. The faster 2 mm arm could be evaluated
immediately without disturbing the native trainer, and the original
coordinator could safely detect and skip completed work.

Experiment 005 showed how execution specs can govern a multi-stage research
system. It separated global proposal quality from local segmentation, defined
metrics aligned with each stage, scheduled immutable intermediate reports,
provided clean stop-control files, and deferred cascade jobs when GPUs were
busy with higher-priority evaluation. Its synced final report keeps only the
comparison needed for review while the large predictions remain external.

Together, these experiments show a progression from documentation hygiene to
concurrency-safe research orchestration. The durable principle is not any
specific poll interval or filename. It is explicit ownership, immutable
evidence, verifiable completion, and a close loop from plan to decision.

## Adoption Blueprint For Another Project

### 1. Inventory Before Designing

Map the existing repository, data roots, runtime outputs, repeated user
requests, long-running tasks, decision documents, and dirty-tree risks. Identify
which facts are durable, which state changes often, and which artifacts cannot
enter Git. Do not begin by copying this directory tree.

### 2. Establish The Entry Contract

Create a minimal `AGENTS.md` that tells agents what to read and what boundaries
must not be crossed. Add an `AGENTS/README.md` only when several focused guides
need a stable reading order. Keep environment, data, hardware, experiment, and
goal facts in separate files when they have different owners or update rates.

### 3. Separate Rules, Commands, And Reflection

Put universal repository policy in `workflow.md`. Put a recurring, named task
procedure in `command.md`. Put audits, imported examples, and upgrade reasoning
in `workflow_reflect/`. Require evidence before promoting reflective advice
into mandatory rules.

### 4. Build Project Planning Across Horizons

Use a goals document for long-horizon direction and open choices. Use a current
status page for active work, settled decisions, blockers, and next actions.
Use a colocated execution spec for each substantial task or experiment. Keep
scientific method details separate when multiple executions should share the
same contract.

### 5. Define Artifact Ownership

Choose one canonical location for configs and one external root for runtime
products. Decide which small artifacts may return to Git. Record source commit,
config hash, dataset/split identity, command, and output path so intent and
evidence can be reconciled.

### 6. Add Gates And Closeout

Define the smallest useful smoke test for each expensive workflow. Add
consistency checks for registries, configs, or generated summaries. Require
closeout to state what ran, which artifacts are authoritative, what failed, and
what decision follows.

### 7. Pilot Before Expanding

Apply the organization to one real task. Observe what future sessions cannot
recover, where instructions conflict, and which manual checks are skipped.
Only then add commands, reflection reports, automation, or more elaborate
indexes.

## Organization Maturity Levels

### Minimum

Use a short agent contract, one workflow guide, a current-status page, a
canonical config location, a runtime artifact policy, and one verification
command. This is enough for a small project with few experiments.

### Intermediate

Add focused context guides, execution-spec and closeout templates, stable
experiment IDs, a repo-local registry, reusable commands, and a reflection
folder. This matches projects with recurring long runs and several active
research decisions.

### Mature

Add machine-readable provenance, deterministic schedules, immutable
checkpoints, synchronized summaries, lock-safe evaluation sidecars,
completeness predicates, submission runbooks, and automated drift checks.
These features are justified when concurrency, compute cost, or release risk
makes manual coordination unreliable.

## Governance, Provenance, And Verification

Dirty-tree preservation is part of the design. Agents inspect existing changes,
treat them as user work, and stage exact paths only. Topic-focused commits keep
code, configs, synced summaries, and documentation reviewable. Generated
experiment summaries are refreshed at the point where their recorded commit
and hashes describe the intended state.

The canonical workflow check combines challenge-source tests, experiment
consistency, Python compilation, Git whitespace checks, and tracked-cache
detection. Other projects should compose an equivalent check from their own
risks rather than copy the exact commands.

At inspection, the working tree contained an uncommitted modification to
`AGENTS/command.md` adding an index and shorthand invocation. That change is
excluded from this commit-pinned report. The report describes committed
behavior only, while acknowledging the dirty state so later readers do not
confuse uncommitted refinements with the source revision.

## Tradeoffs And Anti-Patterns

The organization adds more Markdown files, so every new guide must justify its
reading and maintenance cost. Over-separation can be as harmful as a monolith.
Create a new category only when its responsibility, audience, and update
cadence differ materially from existing files.

Avoid:

- treating raw chat as the only project memory;
- copying a mature repository scaffold before a project needs it;
- mixing mandatory rules with optional examples;
- turning every one-off request into a reusable command;
- placing active status inside a slow-changing workflow guide;
- recording intended configs only in runtime folders;
- overwriting failed or partial runs;
- launching competing evaluators without ownership and completion checks;
- promoting a reflective recommendation directly into policy without evidence;
- letting synced summaries claim a commit that does not contain their inputs.

## Current Improvement Opportunities

The organization is clearer than the July 23 version, but some live guides lag
the committed project state. `AGENTS/experiments.md` lists only experiments 001
and 002 even though the registry contains 004 and 005. `docs/current_status.md`
does not yet summarize those newer experiments, and several stabilization notes
still refer only to experiment 002. The dated workflow-upgrade prompt also
mentions the retired prompt-archive layout.

These are reconciliation tasks, not reasons to merge the guide layers again.
The correct improvement is to refresh each owner file at its natural cadence:
experiment identity in the experiment guide, dynamic decisions in current
status, and workflow-upgrade language in a new dated reflection prompt.

Future automation could check that agent experiment IDs agree with the
registry, that reference indexes point to existing exports, and that
current-status experiment references are not obviously behind the registry.
Such checks should warn about drift without regenerating human decisions.

## Bottom Line

The reusable design is a separation of responsibilities connected by an
evidence loop. Workflow rules constrain action, commands accelerate recurring
requests, project planning connects goals to executable specs, runtime systems
produce evidence, current status records the next decision, and reflection
improves the system without silently changing policy. Future projects should
adopt these boundaries incrementally and keep only the structure that solves
an observed coordination or reproducibility problem.
