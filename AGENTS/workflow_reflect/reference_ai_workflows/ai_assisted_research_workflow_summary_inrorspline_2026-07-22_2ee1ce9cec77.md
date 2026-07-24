---
doc_type: ai_workflow_summary
status: draft
summary_date: 2026-07-22
repo_path: /home/hengjie/code_sync/inrorspline
head_commit: 2ee1ce9cec77d5135f7a215ed4bf23e5d59393b2
head_commit_short: 2ee1ce9cec77
head_commit_date: 2026-07-08 22:17:20 -0700
head_commit_subject: Update experiment registry for July OASIS and COPD runs
scope: Repo-inspection summary for Codex-assisted research workflow design.
---

# AI-Assisted Research Workflow Summary

## Executive Summary

`inrorspline` is a Codex-assisted research repository for studying how
implicit neural representation, spline, and direct control-point deformation
priors behave in medical image registration. The project is already organized
around several good AI-assisted research practices: explicit agent rules,
plan-first commands, experiment registries, per-run execution specs, technical
logs, protocol standards, and strict separation between Git-tracked review
artifacts and large/private data.

The strongest design idea to carry forward is that Codex is treated as a
bounded research collaborator rather than a free-running automation script.
The repo gives agents clear context, requires provenance for experiments, and
keeps safety gates around rsync, commits, public release material, and protocol
interpretation. The main improvements are mostly maintenance and tooling:
refresh stale scaffold docs, add repo-local pytest configuration, document the
intended Docker test environment, and consider a small package scaffold once
the experiment scripts stabilize.

## Current Setup

The repository supports a reproducible study of continuous deformation models
for medical image registration. The main scientific comparison is not simply
"INR versus spline"; it asks how much behavior comes from representation,
parameterization, regularization, optimization path, and motion regime.

Primary research handles:

- `brain`: OASIS brain registration, centered on seed-3 shuffled protocols.
- `lungct`: DIR-Lab 4DCT and COPD-style LungCT protocols with explicit
  preprocessing and direction conventions.

Method families and sources:

- `IDIR`: coordinate-network deformation model, anchored by `external/IDIR/`.
- `SINR`: INR plus spline/control-point structure, anchored by
  `external/SINR/`.
- `DirectCP`: repo-local direct optimization of spline/control-point
  parameters without the INR control generator.
- Baselines and references: `external/MIR/`, `external/pTVreg/`, saved OASIS
  baselines, and Greedy/pTVReg-style references where documented.

Repo model:

- Git tracks code, docs, configs, tests, split files, experiment summaries, and
  review-facing artifacts.
- Raw data, checkpoints, dense fields, large generated outputs, Docker image
  archives, credentials, and private host details stay outside Git.
- The documented path model separates repo, data, and scratch/output roots:
  `REPO_ROOT`, `DATA_ROOT`, and `SCRATCH_ROOT`.
- Docker execution is designed around separate mounts for `/workspace/repo`,
  `/workspace/data`, and `/workspace/scratch`.
- GPU workers are treated as execution mirrors unless explicitly promoted.

Implementation and provenance shape:

- Active paper-facing implementation currently lives mainly under
  `scripts/experiments/...`; `src/` remains a placeholder.
- External method repositories are tracked as Git submodules under `external/`.
- Experiment review folders under `experiments/` commonly contain
  `codex_execution_spec.md` and `technical_log.md`.
- At inspection time, the repo contained 29 Codex execution specs and 29
  matching technical logs, which is a strong provenance pattern.
- `docs/04_experiment_registry.md`, `docs/06_decision_log.md`,
  `docs/standards/`, `docs/implementation/`, and `docs/codex_prompts/` form
  the durable context layer for future agents.

Working tree note:

- The worktree was already dirty before this report was created. Existing
  modified and untracked files were treated as user work and were not changed
  by this summary task.

## Good Practices To Keep

- Keep `AGENTS.md` as the durable operating contract. It gives Codex a stable
  command vocabulary, safety rules, path model, and project intent.
- Use explicit bracket commands such as `[use plan mode]`,
  `[adapt and plan]`, `[experiment launch plan]`, `[sync-code plan]`,
  `[sync-data plan]`, `[prepare commit]`, and `[summarize and commit]`.
- Separate planning from execution for high-risk work: experiment launches,
  rsync, data movement, commits, public release preparation, and protocol
  changes.
- Register every real run before launch. Good entries include commit, branch,
  Docker image, dataset root, split config, protocol tag, command, output root,
  primary metrics, known limitations, and next decision.
- Pair every substantial experiment with a `codex_execution_spec.md` and
  `technical_log.md`. This helps future Codex sessions reconstruct what was
  intended, what actually ran, and what should be trusted.
- Preserve protocol warnings near the results. The LungCT docs correctly warn
  that native IDIR, pTVReg-style crops, SINR, and DirectCP variants may be
  solving different image-domain problems unless preprocessing is matched.
- Keep result interpretation tied to metric conventions. The repo documents
  Dice, HD95, ASD, TRE, bending energy, diffusion energy, log-Jacobian spread,
  folding percentage, and NDV conventions instead of relying on single
  operating-point claims.
- Archive reusable prompts and adapted plans in `docs/codex_prompts/`. This is
  especially useful for multi-day research where future agents need the shape
  of prior reasoning, not only final code.
- Prefer small, curated review artifacts in Git. Dense histories, raw outputs,
  checkpoints, and generated arrays should stay outside Git with summaries or
  reports committed instead.
- Use exact-path or tightly scoped ignore rules for large generated artifacts.
  The existing `.gitignore` avoids broad experiment CSV ignores, which is
  important because small review tables may be paper-facing artifacts.

## Dos And Don'ts For Codex-Assisted Research

### Do

- Do make Codex read `AGENTS.md`, `docs/00_index.md`, the methods docs,
  standards, experiment registry, decision log, and relevant experiment specs
  before changing research behavior.
- Do ask Codex to state assumptions, paths, protocol tags, launch commands,
  output roots, acceptance criteria, and verification steps before execution.
- Do ask for a smoke test before long GPU runs.
- Do keep raw datasets, scratch outputs, checkpoints, and private manifests out
  of the repo.
- Do require Codex to preserve uncommitted user work and stage only planned
  files when commits are requested.
- Do make Codex distinguish method behavior from preprocessing, loss, mask,
  metric, and motion-regime differences.
- Do have Codex record failed verification honestly. A failed or unavailable
  test is useful provenance.

### Don't

- Don't let Codex launch long experiments without a registered spec, smoke
  command, output root, and acceptance criteria.
- Don't compare LungCT rows as method-only results when preprocessing protocols
  differ.
- Don't commit raw data, checkpoints, dense displacement fields, model weights,
  private hostnames, credentials, or identity-breaking metadata.
- Don't treat raw alpha values across IDIR, SINR, and DirectCP as directly
  comparable. Use method-native context plus common posthoc regularity metrics.
- Don't let rsync run without a direction, delete policy, dry run, exclusions,
  and approval trail.
- Don't let stale scaffold docs become the only onboarding path after the repo
  has grown real scripts and tests.
- Don't use broad ignore patterns that hide potentially important review
  tables or paper artifacts.

## Improvement Opportunities

1. Refresh stale scaffold docs.

   `README.md`, `docs/00_index.md`, and `tests/README.md` still describe the
   project as mostly scaffolded or without tests. The repo now has substantial
   experiment scripts, Docker docs, tests, external submodules, and many
   completed experiment records. Updating these files would reduce onboarding
   friction for both humans and Codex.

2. Add repo-local pytest configuration.

   A default `pytest -q` currently collects upstream submodule tests under
   `external/MIR/` and then fails in the current shell because `torch` is not
   installed. Add a local pytest config that scopes collection to `tests/` by
   default and documents any optional external-submodule test command
   separately.

3. Document the intended test environment.

   The repo has Docker execution conventions and a default container noted in
   `AGENTS.md`, but the test docs do not yet say which environment should run
   repo-local tests. Add a short command such as the intended Docker-based
   `pytest tests -q` invocation once the dependency environment is confirmed.

4. Consider a minimal Python package/tooling scaffold.

   The current implementation source is script-heavy. That is workable for fast
   research, but shared metric, path, and deformation utilities are now
   important enough that a small package scaffold could reduce import hacks,
   test friction, and duplicated conventions.

5. Add a compact new-session checklist.

   Codex sessions already have excellent durable rules, but a short checklist
   could speed up future work:
   read canonical docs, inspect dirty state, identify protocol tag, confirm data
   root/output root, plan first for risky work, then verify and summarize.

6. Reconcile registry, logs, and results summaries periodically.

   The experiment registry contains many completed entries, while some summary
   docs still describe carried-forward or not-yet-final results. A periodic
   reconciliation pass would keep the narrative layer aligned with the run
   ledger.

7. Keep anonymous-release hygiene visible.

   The repo already has strong rules about private paths and review-facing
   safety. Maintain that discipline by preferring dataset handles, protocol
   tags, and environment variables in public docs, and keeping host-specific
   manifests in private or ignored locations.

## Suggested Codex Workflow Template

Use this pattern for future AI-assisted research tasks:

1. Ground the session.
   - Read `AGENTS.md`, `docs/00_index.md`, the relevant methods/standards docs,
     and the current `git status`.
   - Identify whether the task is planning, coding, experiment execution,
     data movement, or commit preparation.

2. Lock the protocol.
   - Name the dataset handle, split, fixed/moving direction, preprocessing
     protocol, metric definitions, output root, and expected artifacts.
   - State known limitations before launch, not only after results arrive.

3. Make a concrete execution spec.
   - Record commit, branch, Docker image, command, resources, smoke test,
     acceptance criteria, and failure handling.
   - Store the plan near the experiment as `codex_execution_spec.md` when it is
     a real run.

4. Execute with checkpoints.
   - Run a smoke test or dry run first.
   - For long jobs, capture logs and progress in a technical log.
   - Keep large outputs outside Git and commit only curated summaries.

5. Close the loop.
   - Aggregate results through documented scripts.
   - Update the experiment registry, technical log, results summary, and
     decision log when the outcome changes project direction.
   - Report verification honestly, including failed or skipped tests.

## Verification Notes

No code tests are required for this Markdown-only report.

During planning, `pytest -q` was tried in the current shell and did not pass.
The failure happened during collection because `torch` was unavailable in this
environment, and pytest also collected tests from the `external/MIR/` submodule.
This should be treated as a tooling/environment finding, not evidence that the
repo-local test logic is wrong.

The intended verification for this report is:

- Confirm this file exists at
  `docs/reports/ai_workflow/ai_assisted_research_workflow_summary_2026-07-22_2ee1ce9cec77.md`.
- Confirm the filename date and commit short ID match the metadata above.
- Confirm the report does not modify or rely on existing uncommitted work.
- Confirm no raw data, checkpoints, credentials, or private generated artifacts
  were added.
