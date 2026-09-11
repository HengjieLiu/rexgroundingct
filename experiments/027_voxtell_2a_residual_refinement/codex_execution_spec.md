---
created: 2026-09-09
updated: 2026-09-09
status: full_training_authorized
experiment_id: 027_voxtell_2a_residual_refinement
---

# Exp027 execution spec

## Authorized four-arm deletion follow-up (2026-09-10)

The user authorized the new deletion-only method and a shared live dashboard.
See [deletion_four_arm/codex_execution_spec.md](deletion_four_arm/codex_execution_spec.md).
It starts four pristine FP32 models in a separate runtime, stops after the
epoch20 evaluation barrier, and preserves the stopped residual experiment below.

## Approved full FP32 run (supersedes earlier timing-only boundaries)

The user approved full 2a caching followed by four concurrent FP32 runs: 100
updates per epoch, 100 epochs (10,000 updates), with synchronous evaluation at
updates 1,000 through 10,000 in increments of 1,000. Start from pristine common
weights, retain AdamW LR 1e-4 and clipping 1.0, and disable TF32 only in refiner
workers. Preserve the frozen VoxTell export precision and FP16/storage fallback
contract. Future refinement training/evaluation defaults to FP32.

Use a separate `full_fp32_100ep` directory under the existing runtime root and
share its cache. Before any trainer starts, four disjoint cache workers must
complete and verify all 801 training CTs / 1,120 findings plus 63 validation CTs /
69 findings. Reuse the 196 completed cases with hash-verified legacy producer
compatibility; never rewrite their provenance to disguise a code change.

A dedicated CPU writer polls every second and refreshes training curves within
five seconds; per-finding provisional validation, final per-arm summaries,
phase changes and failures trigger prompt refreshes independent of other arms.
Publish matched-subset baselines and A/B/full completion counts for provisional
scores. Histories append every optimizer update, snapshots/checkpoints occur
every 100 updates and on interruption. All evaluated checkpoints are retained.
No automatic ranking. Stop at `pending_user_review` after all ten barriers.

Before launch: CPU Docker tests, synthetic dashboard inspection, cache integrity
and launch-gate tests, checkpoint/resume and partial-result checks, repository
workflow and whitespace checks. This user instruction authorizes launch after
checks pass; no additional timing approval is required. Launch command:

```bash
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_host.sh orchestrate --gpus 0 1 2 3
```

Both timing benchmarks remain preserved as historical diagnostic evidence.
The sections below record their original implementation and authorization.

## Objective and scope

Implement the four accepted 2a residual-refinement conditions and a local live
dashboard. Follow [design_decisions.md](design_decisions.md). The user explicitly
approved immediate timing-benchmark launch after implementation: "the gpus are
idle now, so move on and start the timing benchmark right away when you finish
coding". Approval covers required cache preparation, four concurrent 100-update
arms and full 2a evaluation. Full training remains unapproved and unscheduled.
Final model ranking remains with the user.

## Inputs and ownership

- Canonical config: `configs/experiments/027_voxtell_2a_residual_refinement.json`.
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`.
- CT/target cache: shared `crop_zscore_native_v1` under the existing dataset root.
- Frozen checkpoint SHA: `a499ad1c0fada9a7e4d78e352fa5510d31295e82749e00ef0a4eee6da685e4aa`.
- Base validation cache: strict SideExp003 key
  `v2_d6dcddf705766db234bf631fa0925bfd69a5e18388767448b02a31b6484fcdbe`.
- Runtime root: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement`.
- Code/config/docs/small split manifests stay in Git. Arrays, checkpoints,
  generated figures and histories stay in runtime storage.

## Data and method contract

Preprocessing is cached-equivalent native VoxTell: existing CT orientation,
crop-to-nonzero, full cropped-volume z-score once, no resampling. The method
variable is a standalone residual network. VoxTell is used only in explicit
cache preparation; its weights are never part of refiner optimization.

Use the exact split, sampler, network, loss and optimizer in the accepted
decision record. Use 192³ tiles, joint axis flips, real-voxel loss masks,
Gaussian residual blending at 50% overlap and threshold .5. Process each val
2a finding once and aggregate its metrics into A/B/full. Preserve original
finding IDs and GT counts outside the inherited nonzero crop during evaluation.

## Execution and interfaces

The experiment CLI provides `prepare`, `cache`, `train`, `evaluate`, `benchmark`,
`orchestrate`, and `report`. Preparation and reporting are CPU-only. GPU
commands require explicit opt-in; orchestration supports a non-mutating dry run.
Full-run scheduling stays null until the user supplies a budget and milestones.

Preparation freezes metadata/config/code provenance, patient halves, finding
pools and cached-base summaries. Cache preparation is separately resumable and
validates checkpoint, metadata, geometry, dtype, hashes and zero-threshold masks.

After approval, benchmark only required train cases plus all val 2a data; then
warm up each independent GPU worker for five synthetic steps, restore pristine
state, train exactly 100 timed steps concurrently, barrier, and evaluate all
69 findings per worker. Preparation and warm-up are excluded from measured
training time and reported separately. Stop at `awaiting_schedule_decision`.

Full runs start fresh, use user-selected milestones and global evaluation
barriers, and stop at `pending_user_review`. Save immutable full model,
optimizer/scaler/RNG/update state and bind resumes to config, split, schedule
and code fingerprints. Retain checkpoints and per-finding metrics; never
automatically select or promote a model. Failures stop the group and remain
visible; no automatic method/tile changes or competing evaluators are allowed.

## Live reporting

One CPU writer atomically refreshes Markdown, PNG and aggregate JSON every
30 seconds and after events. The 3×3 figure contains loss, patch Dice, residual
magnitude, then A/B/full Dice and A/B/full hit rate. Preserve arm order, expose
training/in-sample labels, cached baselines, progress and failures. Pending is
not zero. Local viewers may need reload.

## Verification and closeout

CPU-only tests cover deterministic split/membership/sampling, orientation,
padding, zero initialization, finite gradients, synthetic fitting, tiled
coverage/blending, metrics/recomposition, resume and failure handling,
dashboard partial histories and GPU/schedule gates. Use the existing VoxTell
Docker image with GPUs disabled for torch tests. Run the canonical repo checks
and whitespace checks. Preview reporting with labeled synthetic fixtures.

Real 192³ GPU feasibility and timing are explicitly deferred. Implementation
closeout records tests, dashboard paths and exact benchmark command, and leaves
status `awaiting_timing_approval`. No GPU jobs or automatic launch watchers start.

The implementation handoff and exact commands are in [verification.md](verification.md).

## Approved launch

Launch on GPUs 0–3 after confirming they are idle:

```bash
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_host.sh benchmark --gpus 0 1 2 3
```

Capture the explicit authorization, exact command and host process ID in runtime
`run_manifest.json`; save the launch log under `logs/`. Stop after the timing
report at `awaiting_schedule_decision`, or stop and report any failure.

## Previous stop condition

All 196 required CT caches are complete. After the shared-context lock fix,
attempt 5 reached the 192³ FP16 synthetic warm-up on all four GPUs and stopped
with non-finite gradient norms. No measured updates, training checkpoints or
evaluations were produced. All GPU workers exited. Follow the accepted invalid-
result stop policy: retain the data and failure reports, investigate the numerical
cause, and do not automatically switch precision, shrink tiles or relax the
finite-gradient check. At that point runtime status was `failed`; scheduling
remains unset.

## Numerical diagnosis requested after warm-up failure

The user requested investigation of the failure and comparison with previous
learning-rate and gradient-clipping settings. Run a bounded, single-GPU
synthetic diagnostic at the accepted 192³ geometry in the existing Docker image.
Record forward-loss finiteness, the failing parameter gradients, AMP scale,
whether an optimizer step actually occurred, and behavior with adaptive scale
backoff and a lower initial scale. These are disposable diagnostic updates,
separate from the four-arm benchmark and its measured update counts. Preserve
the production method/config, existing caches and stopped benchmark artifacts
while establishing the cause. Store the diagnostic script, log and JSON under
the external runtime root and summarize the evidence in the experiment folder.

The [diagnosis and fix](numerical_diagnosis.md) are complete: scaled FP16 head
gradients overflowed on synthetic update 5. Keep the accepted LR, norm threshold
and precision; allow bounded GradScaler skip/backoff and retry the same patch.
Only finite, strictly clipped gradients may reach an optimizer update. Require
exactly 100 actual optimizer calls, recording overflow retries separately.
The fix passed 32 CPU tests and the full-size GPU reproduction. Archive the
zero-measured-update failed session, retain audited unchanged caches, and retry
the previously authorized timing benchmark. Full-run scheduling remains unset.

## Authorized FP32 comparison after FP16 completion

The user explicitly requested: "after the fp16 benchmark finish ... running the
same thing with fp32 and see if there is overflow issue". This authorizes a
sequential follow-up controller. It may launch FP32 only after the current FP16
benchmark completes all four 100-update arms, all four 69-finding evaluations,
and its timing report, and releases the GPUs. Failure of the prerequisite stops
the follow-up. This supersedes the earlier prohibition on an unapproved launch
poller for this specific user-requested dependency.

Use a separate canonical variant, `fp32_benchmark_config.json` in this experiment
folder, and runtime root `runtime/precision_fp32`. The only config differences
are `training.amp=false` and the runtime output directory. Use the same code,
Docker image, GPUs, initial weights, materialized schedules, cached inputs,
optimizer, LR 1e-4, norm clipping 1.0, 192³ tiles, warm-up/reset and evaluation
barriers. Disable TF32 through `NVIDIA_TF32_OVERRIDE=0` for the FP32 diagnostic,
and record that backend policy as part of the precision comparison. Cached
base logits retain their original storage precision and are not regenerated.

The CPU controller refreshes `runtime/reports/precision_comparison.md` and its
JSON/CSV records every 30 seconds, linking the separate four-arm dashboards.
Report finite-value failures, AMP overflow retries, warm-up behavior, measured
training time, data-loading time, evaluation time and peak memory alongside
A/B/full metrics in fixed arm order. Preserve the FP16 trial and stop the FP32
trial on OOM or non-finite values without changing tile size or optimizer.
Full training and ranking remain pending user decisions.

## Learning diagnosis requested during the authorized full run

Audit completed epoch-10/20/30/40 evaluations and the first 4,000 recorded
updates per arm. Keep the running FP32 experiment and its source fingerprint
unchanged. Recompose per-finding metrics, examine edits and exposure, and replay
a deterministic, mode-stratified sample of cached patches to compare each
recorded training prediction with its exact frozen-base patch. Report empty
targets and confidence margins separately. Use CPU only, with read-only inputs;
any checkpoint probes use the existing Docker image with GPUs disabled and at
most two CPU threads. Bound patch replay to 32 patches per arm. A two-window
probe may measure whether a trained model's predictions depend on tile context.
Save the standalone audit script outside the production `exp027_*.py` glob,
small findings in `learning_diagnosis_e040.md`, and detailed JSON/CSV/figures in
the external full-run analysis directory. Distinguish measured failure modes
from hypotheses and proposed ablations; do not change training or select a
checkpoint for deployment.

## User-requested stop after epoch-50 validation

The user superseded the remaining 100-epoch schedule: finish every arm's full
69-finding validation at update 5,000, then stop before update 5,001. Preserve
all full-state checkpoints, evaluations and histories. The running coordinator
has no deferred-stop switch, so suspend only that exact coordinator process
while its four independent evaluators and CPU reporter finish. Verify all four
summaries and their checkpoint provenance, then terminate the suspended
coordinator through its interruption handler and finalize a user-requested-stop
record and dashboard. Do not modify the immutable training config, schedules,
or production source fingerprint. Record process identities and stop evidence
under the external full-run runtime. No subsequent GPU work is authorized by
this stop request.

Completed at 2026-09-10 15:27:08 UTC. All four models have exactly 5,000 updates
and complete 69-finding epoch-50 evaluations; no update 5,001 occurred. Process
exit and idle GPUs were verified. Status is `stopped_by_user`, with results
pending user review. See [stop verification](user_stop_after_val50.md).

## Audit of the proposed base-positive tile gate

The user requested an audit, not implementation or launch of a new training
scheme. Compare the accepted all-tile refiner with a per-finding gate that
enables a tile only when a valid base logit is at least zero. Use a bounded
CPU-only pass over the 69 validation finding caches on the established 192³,
50%-overlap inference grid. Measure active tiles, active tiles without GT,
the union of editable voxels, and reachable/unreachable GT (including GT lost
outside the preprocessing crop). Inspect metadata-only prediction availability
for all training findings. Preserve the stopped experiment, caches and source
fingerprint; do not run model inference or training. Store raw diagnostic
counts externally and a proposal audit in this experiment directory. Explicitly
distinguish patch gating from voxel edit restrictions, specify overlap blending,
and recommend controlled comparisons without adopting new hyperparameters.
# Authorized deletion-only diagnostic follow-up (2026-09-10)

The user authorized implementing and running the bounded diagnostic specified
in [deletion_diagnostic/codex_execution_spec.md](deletion_diagnostic/codex_execution_spec.md).
It uses a separate runtime and fresh model. The original four-arm run remains
stopped; larger editing training runs will be planned after diagnostic review.
