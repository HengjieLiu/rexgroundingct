---
doc_type: ai_workflow_report
created: 2026-07-22
updated: 2026-07-22
status: active
source_repo: /home/hengjie/code_sync/DoseRad2026
source_commit: d4053028f76e
source_commit_full: d4053028f76ee3fcb485450189b8e1fedc72de8d
inspection_date: 2026-07-22
related_code:
  - AGENTS.md
  - docs/00_index.md
  - docs/08_codex_workflow.md
  - docs/04_experiment_registry.md
  - docs/codex_prompts/
  - experiments/
---

# AI-Assisted Research Workflow Summary - DoseRad2026

## Executive Summary

`DoseRad2026` is a strong example of a Codex-assisted research repository
because it treats AI work as durable project state, not as disposable chat
history. The repository has a clear operating guide, a numbered documentation
spine, an experiment registry, reusable Codex prompt archives, per-experiment
execution specs, technical logs, run logs, result summaries, and decision logs.

The main design lesson is simple: Codex works best when every substantial task
is converted from a free-form request into an explicit research artifact before
expensive implementation or training begins. In this repo, that pattern made it
possible to run many related medical-physics experiments while preserving enough
context for future sessions to recover the question, constraints, commands,
metrics, caveats, and next decisions.

Inspection provenance:

- Source repo: `/home/hengjie/code_sync/DoseRad2026`
- Source commit: `d4053028f76e`
- Inspection date: `2026-07-22`
- Git state at inspection: `main` was ahead of `origin/main` by 5 commits and
  the working tree had modified and untracked files.

## Current Setup

### Repository Shape

The repo is organized around a research workflow rather than only a code
workflow:

- `AGENTS.md` is the short operating guide for future Codex sessions.
- `README.md` points humans and agents to the docs entrypoint.
- `docs/00_index.md` is the navigation hub and current-status dashboard.
- `docs/01_research_plan.md` through `docs/08_codex_workflow.md` hold the
  durable research plan, literature map, methods, experiment registry, result
  summary, decision log, todo list, and Codex workflow.
- `docs/workflows/` stores reusable runbooks for data inspection, conditioning
  visualization, W&B tracking, and Geant4 reproduction.
- `docs/codex_prompts/` archives reusable prompts for major Codex-led tasks.
- `experiments/YYYY-MM-DD_short-slug/` stores experiment-specific plans,
  execution specs, run logs, technical logs, results, and notes.
- `code/doserad/` holds importable modules.
- `scripts/` holds runnable inspection, training, evaluation, launch,
  benchmark, and analysis entrypoints.
- `configs/` holds tracked experiment configs and split definitions.
- `outputs/` holds runtime products and is treated as non-committed output.

This structure makes the repo navigable across time. A new Codex session can
start at `AGENTS.md`, then read `docs/00_index.md`, `docs/07_todo.md`, and the
relevant experiment or workflow doc before acting.

### Codex Operating Model

The repo has explicit Codex conventions:

- Read the operating guide and project index before coding.
- Use `[use plan mode]` to request non-mutating planning.
- Use `[prepare commit]`, `[summarize and commit]`, or scoped variants to audit
  the working tree before staging.
- Save a `codex_execution_spec.md` before large multi-step implementation or
  long-running jobs.
- Archive reusable prompts under `docs/codex_prompts/`.
- After meaningful work, update the registry, result summary, decision log,
  todo list, and detailed todo lane when the research state changes.

This is a good pattern for AI-assisted research because it separates intent,
implementation, execution, evidence, interpretation, and next actions.

### Experiment Artifacts

The experiment folders are the strongest part of the setup. Many experiments
contain:

- `plan.md`: question, hypothesis, method, metrics, risks, and todo.
- `codex_execution_spec.md`: the converted Codex work spec.
- `technical_log.md`: data split, preprocessing, model, loss, hyperparameters,
  runtime details, artifact paths, and limitations.
- `run_log.md`: exact commands, launch times, process notes, failures, and
  follow-up actions.
- `results.md`: metrics, tables, interpretation, decision, and follow-ups.

This pattern is especially useful for long-running GPU and Monte Carlo work
where decisions may happen days after the original prompt.

### Environment And Tracking

The project uses a Docker-first GPU environment:

- Main image: `hengjieliu96/doserad2026-dev-gpu:py311-cu12`
- Persistent container: `doserad2026-dev-gpu`
- Host repo path: `/home/hengjie/code_sync/DoseRad2026`
- Container repo path: `/homebase/code_sync/DoseRad2026`
- Host data path: `/data/hengjie/datasets/DoseRAD2026`
- Container data path: `/database/datasets/DoseRAD2026`

W&B is used for live monitoring, but local files remain the durable source of
truth:

- `metrics.csv`
- `epoch_metrics.csv`
- `summary.json`
- checkpoints under each run directory

This is a healthy division: W&B helps decide whether a run is alive or worth
continuing, while local CSVs and Markdown summaries preserve reproducibility.

## What Worked Well

### 1. Free-Form Requests Became Execution Specs

The best repeated pattern is:

```text
user request -> codex_execution_spec.md -> implementation -> run_log.md -> results.md -> registry/decision/todo updates
```

The input-conditioning ablation, global-volume deep-supervision work, Geant4
reproduction, GPU geometry generation, GPU input ablation, exact GPU SDF study,
and model-scaling/debug experiments all benefited from this approach.

Good execution specs included:

- Objective and previous evidence.
- Experiment matrix.
- Shared training details.
- Required implementation constraints.
- Logging/checkpoint expectations.
- Verification gates.
- Expected documentation updates.

This reduces ambiguity for Codex and makes future sessions less dependent on
hidden conversation context.

### 2. Prompt Archives Preserved Reusable Research Intent

`docs/codex_prompts/` is valuable because it captures cleaned prompts for later
reuse. These files are not merely transcripts. They record the task framing,
constraints, accepted defaults, expected artifacts, and acceptance criteria.

This is especially useful for:

- repeating an experiment style with a new candidate method;
- onboarding a fresh Codex session;
- explaining why a task was shaped a certain way;
- avoiding drift between similar experiment waves.

### 3. Technical Logs Made Runs Auditable

The `technical_log.md` convention works well. The better logs put limitations
near the top, then record data split, normalization, input channels, target
scaling, model, loss, hyperparameters, sampling strategy, runtime, and artifact
paths.

For medical-physics research, this matters more than usual because small
choices such as dose scaling, body masks, CT-to-density assumptions, split
policy, and geometry conventions can change the interpretation of results.

### 4. Decision Logs Prevented Method Drift

`docs/06_decision_log.md` is useful because it separates durable decisions from
day-to-day observations. Examples include:

- split by patient, never by beam segment;
- treat gantry geometry as a hypothesis until visually validated;
- keep public Geant4 source unchanged and own wrappers in the main repo;
- use GPU-generated no-SDF geometry as the current Task 1 candidate;
- keep exact GPU SDF available but not default after it underperformed g4.

This helps Codex avoid re-litigating settled choices and makes it easier to
spot when new evidence should change a decision.

### 5. Smoke Tests Protected Expensive Runs

The repo repeatedly requires small checks before long runs:

- script `--help` checks;
- tiny smoke training;
- W&B smoke runs before overnight jobs;
- Geant4 low-particle smoke before official-fluence runs;
- correctness and runtime gates before adding new conditioning channels.

This is one of the most important dos for Codex-assisted research: use Codex to
move quickly, but require small validation steps before spending GPU-days or
CPU-days.

### 6. Results Reports Synthesized Across Experiments

The input-conditioning report is a good model. It did not simply repeat one
experiment's result; it connected patch evidence, full-FOV runtime failures,
GPU geometry acceleration, GPU input ablation, exact SDF validation, model
scaling, full validation, and longer training.

That kind of synthesis is where Codex can be especially useful: it can gather
many local records and turn them into a current recommendation with evidence
and caveats.

## Dos For Codex-Assisted Research

- Do start every session by reading `AGENTS.md`, `docs/00_index.md`, and
  `docs/07_todo.md`.
- Do read the relevant method, workflow, experiment, or paper note before
  editing.
- Do convert large requests into `codex_execution_spec.md` before implementation
  or launch.
- Do keep experiment IDs stable once runs, configs, manifests, or reports depend
  on them.
- Do record exact commands, commit hash, config path, data path, seed, runtime,
  and artifact path.
- Do split by patient when patient leakage is possible.
- Do keep local CSV logs and summaries as the source of truth even when W&B is
  enabled.
- Do run the smallest useful smoke test before a long training or simulation
  job.
- Do log raw and weighted loss components when losses are composite.
- Do record assumptions directly in Markdown when Codex had to infer labels,
  geometry conventions, scaling choices, or runtime policy.
- Do use decision logs for major research direction changes.
- Do update todos after results so future Codex sessions start from current
  state rather than stale plans.
- Do keep heavyweight medical data, predictions, logs, checkpoints, model
  weights, caches, and credentials out of Git.
- Do use path-limited staging commands when preparing commits.

## Don'ts For Codex-Assisted Research

- Do not rely on chat history as the only memory of a research decision.
- Do not launch long runs from vague prompts without a saved execution spec.
- Do not interpret small seeded validation subsets as final challenge evidence
  without larger or full-validation follow-up.
- Do not mix many experimental changes in one run unless the explicit goal is a
  combined recipe.
- Do not change external reference repositories, such as public simulator code,
  unless a bug fix is explicitly approved.
- Do not use validation or test labels for training normalization or model
  selection except through documented validation metrics.
- Do not assume gantry, aperture, or coordinate conventions are correct without
  visual QC.
- Do not trust W&B as the only record of a run.
- Do not commit `__pycache__`, checkpoints, raw data, generated outputs, W&B
  runtime files, secrets, or large artifacts.
- Do not use broad `git add .` in a research repo with active outputs and dirty
  runtime files.
- Do not rewrite project history or revert unrelated dirty-tree changes while
  addressing a scoped task.

## Improvements Needed

### 1. Clean Dirty-Tree And Artifact Drift

At inspection time, the repo had modified tracked files, untracked experiment
assets, and tracked `__pycache__` files marked modified. That is risky for
Codex-assisted work because later agents may not know which changes are user
work, generated artifacts, or intended commits.

Recommended improvements:

- Remove tracked Python cache files from Git if they are already tracked.
- Confirm `.gitignore` covers `__pycache__/`, notebooks checkpoints, runtime
  logs, W&B folders, checkpoints, and large generated outputs.
- Use scoped commit preparation frequently after each experiment milestone.
- Add a lightweight "dirty tree triage" checklist to the commit workflow.

### 2. Add A Reusable `codex_execution_spec` Template

The repo has an experiment template, but the execution specs are currently
mostly copied from previous examples. Add a template such as:

```text
docs/templates/codex_execution_spec.md
```

Recommended sections:

- Objective.
- Prior evidence.
- In scope / out of scope.
- Experiment matrix or implementation tasks.
- Inputs, outputs, and paths.
- Data split and leakage constraints.
- Model/config/interface changes.
- Logging/checkpoint policy.
- Verification gates.
- Documentation updates.
- Stop conditions and approval gates.

This would make the strongest workflow pattern easier to reuse.

### 3. Add Machine-Readable Experiment Status

The Markdown registry is human-readable and effective, but automated status
checks would be easier if each experiment had a small metadata file, for
example:

```text
experiments/YYYY-MM-DD_short-slug/experiment.yaml
```

Useful fields:

- `id`
- `status`
- `started`
- `updated`
- `primary_config`
- `run_root`
- `source_commit`
- `active_processes`
- `primary_metric`
- `current_decision`

This would let Codex or scripts generate dashboards, detect stale statuses, and
check whether docs and run folders agree.

### 4. Automate Preflight Checks

The repo has good written rules. A small preflight script would make them harder
to skip.

Recommended command:

```bash
python scripts/check_research_repo_hygiene.py --scope docs experiments configs scripts code
```

Recommended checks:

- dirty tracked cache files;
- files larger than 10 MB proposed for tracking;
- unignored checkpoints, logs, raw data, predictions, or W&B runtime files;
- submodule pointer changes and dirty submodule state;
- experiment folder missing `codex_execution_spec.md`, `run_log.md`, or
  `results.md`;
- registry entries pointing to missing experiment folders;
- experiment folders missing registry entries.

### 5. Make AI Workflow Reports Discoverable

This report intentionally does not update the project index, but future usage
would benefit from either:

- adding `docs/reports/ai_workflow/README.md`, or
- linking AI workflow reports from `docs/00_index.md`.

That would make workflow learnings easier for later Codex sessions to find.

### 6. Continue Tightening Validation And Runtime Gates

The current docs already identify important scientific blockers:

- geometry-channel conclusions still need expanded visual QC;
- challenge-style claims need full-validation confirmation;
- runtime claims need official-style cold/warm memory checks;
- CT HU calibration and density/material handling need review;
- plan-level metrics should wait until control-point weights or meterset values
  are verified.

Codex should keep treating these as gates, not as footnotes.

## Suggested Workflow For Future Repos

For a new AI-assisted research repo, copy the spirit of `DoseRad2026`:

1. Create a short `AGENTS.md` that tells Codex what to read first and what not
   to touch.
2. Add a numbered docs spine: index, plan, literature, methods, registry,
   results, decisions, todos, and Codex workflow.
3. Store reusable runbooks in `docs/workflows/`.
4. Store prompt archives in `docs/codex_prompts/`.
5. Require every major experiment to have a saved execution spec before
   implementation.
6. Keep per-experiment `plan.md`, `technical_log.md`, `run_log.md`, and
   `results.md`.
7. Keep runtime outputs outside Git and keep small provenance artifacts inside
   Git.
8. Use local logs as the durable truth and external dashboards as live mirrors.
9. Promote important outcomes into result summaries and decision logs.
10. Keep todos current enough that the next Codex session can resume without a
    long archaeology pass.

## Bottom Line

The best practice demonstrated by `DoseRad2026` is not any single tool choice.
It is the habit of turning AI-assisted work into auditable project artifacts:
specs before action, logs during action, results after action, and decisions
when evidence changes the plan.

The main improvement opportunity is to automate the hygiene that is already
written down. The docs are strong; the next step is making status, artifact
policy, and preflight checks executable so Codex can detect drift before it
spends time or compute.
