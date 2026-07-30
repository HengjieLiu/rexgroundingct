---
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft
experiment_id: "<experiment-id-or-task-slug>"
---

# Codex Execution Spec

## Objective

State the research or engineering goal in one or two sentences.

## Prior Evidence

List the configs, reports, metrics, papers, or previous runs that motivate this
work.

## Scope

In scope:

- Item 1.

Out of scope:

- Item 1.

## Inputs And Paths

- Canonical config:
- Metadata:
- CT root:
- Segmentation root:
- Runtime experiment directory:
- Repo-local summary directory:

## Data And Preprocessing Contract

- Baseline reference:
- Changed preprocessing variables:
- Unchanged controls:
- Normalization scope:
- Resampling and interpolation:
- Patch/window policy:
- Cache root and ownership:
- Orientation/export check:
- Comparability caveat:

## Method

Describe the exact implementation or experiment plan. Include training,
inference, evaluation, or data movement commands when applicable.

## Evaluation Execution Contract

- Milestone checkpoints:
- Evaluation sets and fixed dataset hashes:
- Resource policy: synchronous milestone barrier or dedicated-GPU sidecar:
- Barrier scope: per arm or global across compared arms:
- Resume state: model, optimizer, scaler, LR horizon, update/schedule cursor,
  and RNG streams:
- Locking, idempotency, and retry behavior:

When training and validation need the same GPUs, use a synchronous milestone
barrier by default: save immutable full state, stop training to release memory,
run validation, wait for every required summary, and resume. A concurrent
sidecar is appropriate only when dedicated GPU headroom is verified and the
extra I/O and compute will not perturb the training comparison.

## Smoke Gate

Describe the smallest useful check that must pass before expensive work starts.

## Success Criteria

- Primary metric or expected output:
- Required artifacts:
- Stop condition:

## Verification

List commands to run before handoff.

## Closeout Plan

Describe which reports, manifests, metrics summaries, and status files should be
updated after completion.
