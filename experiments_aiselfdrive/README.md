---
created: 2026-07-31
updated: 2026-07-31
status: active
---

# ReXGroundingCT AI Self-Drive Experiments

This directory is the complete authoring and control surface for a gated
portfolio of ReXGroundingCT experiments. The technical report in this
directory is preserved as historical scientific evidence; its remote paths,
hardware descriptions, and reported dimensionalities are not executable
provenance for this workspace.

## Agent start here

From the repository root, run:

```bash
python experiments_aiselfdrive/tools/experimentctl.py validate --all
python experiments_aiselfdrive/tools/experimentctl.py next --json
```

The first command validates schemas, dependencies, plans, state, claims,
artifacts, paths, and source hashes. The second returns the only experiment an
executor should claim next, or a no-eligible-experiment reason when the
portfolio is valid but blocked. An agent must inspect `STATUS.md` and read the
selected experiment's `README.md`, `experiment.yaml`, `claims.yaml`, and
`state.json` before claiming it.

For a second Codex window that should wait for a valid and eligible portfolio
selection before running only the first experiment, use the copy-paste work order in
`EXECUTOR_PROMPT.md`.

Claim and execute one stage at a time only after `next --json` returns
`"available": true` for that experiment:

```bash
python experiments_aiselfdrive/tools/experimentctl.py claim \
  --id asd_000_evidence_lock_error_atlas --agent-id <stable-agent-id>
python experiments_aiselfdrive/tools/experimentctl.py run-stage \
  --id asd_000_evidence_lock_error_atlas --agent-id <stable-agent-id>
```

For an `execution_mode: agent` stage, `run-stage` prints a structured work
order. Make only the listed changes, verify the listed acceptance checks, then
refresh the executor lease every 60 seconds while working:

```bash
python experiments_aiselfdrive/tools/experimentctl.py heartbeat \
  --id asd_000_evidence_lock_error_atlas --agent-id <stable-agent-id>
```

Then record evidence:

```bash
python experiments_aiselfdrive/tools/experimentctl.py complete-stage \
  --id asd_000_evidence_lock_error_atlas --agent-id <stable-agent-id> \
  --evidence <evidence-json>
```

Do not start a second experiment automatically. `experimentctl next` enforces
priority and hard dependencies.

## Source and runtime boundaries

All newly authored plans, code, configurations, tests, state, and small results
belong under this directory. Existing repository code may be imported
read-only. The executor must not add experiment-specific code to the existing
`scripts/`, `configs/`, or `side_experiments/` trees.

Large checkpoints, caches, logits, NIfTI files, and predictions live under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/<experiment-id>/<run-id>
```

Each experiment may expose that location through its ignored `runtime`
symlink. A runtime target is an allowed output location only when it is
declared by the experiment plan and artifact manifest.

## Canonical files

- `portfolio.yaml`: revisioned inventory, dependency, selection, and resource
  policy. Each intentional registry revision names its predecessor and is
  sealed by the control-plane lock manifest.
- `EXECUTOR_PROMPT.md`: guarded poll-and-run work order for a separate agent.
- `schemas/`: JSON Schemas for every machine-readable contract.
- `experiments/<id>/experiment.yaml`: canonical scientific plan for the
  active revision. A plan may change only through the audited `replan`
  transition; prior hashes and their source revision remain in the event
  ledger.
- `experiments/<id>/state.json`: canonical mutable execution state.
- `experiments/<id>/claims.yaml`: predeclared scientific claims.
- `experiments/<id>/events.jsonl`: append-only transition ledger.
- `experiments/<id>/artifacts/manifest.json`: artifact ownership and hashes.
- `STATUS.md`: generated human dashboard; never edit it manually.

Human-readable experiment READMEs are generated or checked against the same
machine-readable plans. Status is never duplicated in `portfolio.yaml`.
Claims and artifact obligations are projected into `state.json` at each plan
revision so a closeout cannot silently delete or weaken them.

## Audited revisions and promotion

Never repair a plan by manually rewriting `state.json` or truncating
`events.jsonl`. Adopt an intentional revision through the controller, naming a
recoverable predecessor:

```bash
python experiments_aiselfdrive/tools/experimentctl.py replan \
  --id <experiment-id> --agent-id <stable-agent-id> \
  --reason <specific-repair-reason> \
  --previous-plan-source git:<commit>:<repo-relative-plan-path> \
  --adopt-current
```

Once every coordinated revision is adopted, `lock-control-plane --agent-id
<stable-agent-id>` seals the registry, schemas, templates, hardware profile,
plans, protocols, and immutable claim/artifact obligations. Validation rejects
source drift after that point.

Candidate experiments stop at the portfolio T3 barrier. When every candidate
is terminal or has reached that barrier, `promote --agent-id
<stable-agent-id>` ranks eligible interventions under `portfolio.yaml`, writes
hash-verified decisions, authorizes no more than two T4 stages, and routes all
other candidates to closeout. A T4 command cannot start without its matching
controller event and promotion record. Every selected T4 runs its declared
treatment and disabled-module control as distinct, hash-matched arms.

## State semantics

Experiment status is one of:

```text
draft, ready, claimed, running, retry_wait, blocked, stopping, finished
```

Terminal outcome is one of:

```text
go, no_go, inconclusive, failed, cancelled, invalid
```

A valid negative scientific result is `finished/no_go`, not a failed run.
`finished` requires a passed closeout, verified artifacts, decided claims,
terminal required stages, and no live executor lease.

## Hardware

GPU work runs through Docker, not Slurm or direct host CUDA. The expected
profile is four RTX 6000 Ada GPUs with 47.38 GiB each. Every GPU stage must
pass the preflight in `shared/hardware_profile.yaml`; historical H100/H200
instructions in the report are prohibited.

## Scientific guardrails

- Official training data only; val80 is development. The supported executor
  denies the protected val120 artifact IDs until a controller-authorized T4;
  this is a control-plane guarantee, not an adversarial operating-system
  sandbox. A documented historical metric exposure means val120 is an
  internal held-out replication cohort, not untouched independent
  confirmation.
- Any claim of independent performance confirmation requires a newly
  acquired or externally custodied cohort that has not been inspected by this
  portfolio or its authors.
- Unknown or anatomically compatible unlabeled regions are never treated as
  global negatives.
- Counterexamples require factual, anatomy-certified supervision and lexical
  shortcut controls.
- All causal comparisons lock checkpoint ancestry, schedule, preprocessing,
  prompts, thresholds, and post-processing.
- Full-resolution S3, hard lobe clipping, direct anatomy-query addition,
  generic presence heads, and independent binary component selection are
  explicitly out of scope.
