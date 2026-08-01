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
  --id asd_000_evidence_lock_error_atlas
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
  --id asd_000_evidence_lock_error_atlas --evidence <evidence-json>
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

- `portfolio.yaml`: immutable inventory, dependency, selection, and resource
  policy.
- `EXECUTOR_PROMPT.md`: guarded poll-and-run work order for a separate agent.
- `schemas/`: JSON Schemas for every machine-readable contract.
- `experiments/<id>/experiment.yaml`: canonical immutable scientific plan.
- `experiments/<id>/state.json`: canonical mutable execution state.
- `experiments/<id>/claims.yaml`: predeclared scientific claims.
- `experiments/<id>/events.jsonl`: append-only transition ledger.
- `experiments/<id>/artifacts/manifest.json`: artifact ownership and hashes.
- `STATUS.md`: generated human dashboard; never edit it manually.

Human-readable experiment READMEs are generated or checked against the same
machine-readable plans. Status is never duplicated in `portfolio.yaml`.

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

- Official training data only; val80 is development and val120 is held out
  until T4 confirmation.
- Unknown or anatomically compatible unlabeled regions are never treated as
  global negatives.
- Counterexamples require factual, anatomy-certified supervision and lexical
  shortcut controls.
- All causal comparisons lock checkpoint ancestry, schedule, preprocessing,
  prompts, thresholds, and post-processing.
- Full-resolution S3, hard lobe clipping, direct anatomy-query addition,
  generic presence heads, and independent binary component selection are
  explicitly out of scope.
