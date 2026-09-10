---
created: 2026-09-09
updated: 2026-09-09
status: implementation_verified
---

# Exp027 implementation handoff

The current full-run handoff is [full_run_verification.md](full_run_verification.md);
the material below preserves the benchmark implementation history.

Both authorized precision benchmarks are complete; see the
[FP16 / FP32 results](precision_benchmark_results.md). Completion checks verified
100 ordered optimizer updates and 69 unique evaluation findings per arm,
checkpoint hashes, exact metric recomposition, matched initial weights and
sampling schedules, and unchanged production source fingerprints. FP32 had no
non-finite failures. The controller exited successfully and all four GPUs were
idle at the completion audit. Full training remains unscheduled.

See the [cache coverage, preparation-time and storage audit](cache_preparation_audit.md)
for the distinction between the original export and later cache reuse checks.
The [data-loading audit](data_loading_audit.md) compares both precision trials'
actual loading histories and ranks proposed improvements without changing the
active benchmark. It identifies NFS/first-touch effects and asynchronous loading
as the first targets; the later warmed FP32 loading share is only 7–8%.

Current numerical follow-up: [the warm-up failure was reproduced and fixed](numerical_diagnosis.md).
The latest suite has 32 passing CPU tests, plus a successful 192³ FP16 GPU
reproduction using production warm-up/update functions. AMP overflow recovery
preserves the accepted LR/clipping settings and counts actual optimizer calls.

The user subsequently authorized a matched FP32 benchmark after FP16 finishes.
The CPU follow-up controller (initial PID `2654266`) successfully waited for
four complete 100-update histories, four complete 69-finding evaluations,
checkpoint hashes, the FP16 timing report, released execution locks and GPUs.
Its eight additional CPU tests pass, including prerequisite failure, missing or
duplicated findings, incomplete updates, mismatched checkpoints, dry-run launch
gating and waiting without GPU activity. Production training/evaluation source
hashes, prepared data and cache contracts remain identical to the FP16 trial.

The new canonical variant is [fp32_benchmark_config.json](fp32_benchmark_config.json).
FP32 disables AMP and loss scaling; the Docker environment additionally sets
`NVIDIA_TF32_OVERRIDE=0` to disable TF32 acceleration. All other method settings
and materialized schedules are matched. The 196 existing caches are linked into
the separate `runtime/precision_fp32` root and their hashes are rechecked by the
standard preparation stage. Frozen VoxTell logits are not regenerated.
The backend override follows NVIDIA's documented
[TF32 debugging control](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/).

The exact controller command is:

```bash
python scripts/rexgroundingct/run_027_fp32_comparison.py --execute
```

Omitting `--execute` produces a non-mutating dry run. Do not launch another copy
while the existing controller owns its lock. The controller exits on prerequisite
failure or FP32 failure; it does not change tile size, optimizer or LR. It writes
the [precision comparison](runtime/reports/precision_comparison.md) and JSON/CSV
every 30 seconds, then stops at `complete_pending_user_review`. Separate FP16
and FP32 dashboards remain available from that report. No full-run schedule is
set by this comparison. Both trials are now complete; their results remain
pending user review.

This section records the implementation-only handoff before GPU approval.
The user subsequently approved immediate benchmark launch. Current execution
state is in the [live dashboard](runtime/reports/live_dashboard.md) and
[launch manifest](runtime/run_manifest.json).

Implementation is complete; GPU timing awaits explicit user approval. No real
training, logit export, timing test, GPU smoke test or launch poller was started.
Full-run `total_updates` and `evaluation_updates` remain null. Experimental
results, ranking and checkpoint selection remain pending user review.

## Files and commands

- [Accepted design questions](design_decisions.md)
- [Execution spec](codex_execution_spec.md)
- [Canonical configuration](../../configs/experiments/027_voxtell_2a_residual_refinement.json)
- [Frozen validation halves](../../configs/evaluation/rexgroundingct_exp027_halves_seed20260909.manifest.json)
- [CLI and orchestration](../../scripts/rexgroundingct/run_027_residual_refinement.py)
- Implementation modules: `scripts/rexgroundingct/exp027_{common,data,model,runner,report}.py`.
- [Live dashboard](runtime/reports/live_dashboard.md), [subplot figure](runtime/reports/live_dashboard.png),
  [underlying dashboard JSON](runtime/reports/live_dashboard.json).

All commands below run from the repository root. Preparation only reads source
metadata/manifests and creates small split/config files plus the initial plot.
It does not convert or export logits. These commands are safe before GPU approval:

```bash
bash scripts/rexgroundingct/run_027_host.sh prepare
bash scripts/rexgroundingct/run_027_host.sh report
bash scripts/rexgroundingct/run_027_host.sh benchmark --gpus 0 1 2 3
```

The last command is a non-mutating dry run unless `START_GPU_WORK=1` is supplied.
After explicit approval, the exact timing command is:

```bash
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_host.sh benchmark --gpus 0 1 2 3
```

It resolves the existing VoxTell Docker image to its immutable image ID, records
software/GPU provenance, prepares only scheduled training cases plus all 2a
validation inputs, then runs four independent 100-update trainers and a global
evaluation barrier. Five synthetic warm-up updates are excluded and all state is
restored before measurement. All 69 findings are evaluated once per arm; A/B/full
come from the same records. The command stops at `awaiting_schedule_decision`.
Preparation, startup, warm-up, loading, updates and evaluation timing are recorded
separately. The report includes total concurrent-group wall time.

Once the user chooses a budget and evaluation updates, edit the canonical
configuration and inspect the full-run dry run:

```bash
bash scripts/rexgroundingct/run_027_host.sh orchestrate --gpus 0 1 2 3
```

Only launch that full run after authorization using the same `START_GPU_WORK=1`
opt-in. It starts in `full/` from common initial weights, independently of
diagnostic checkpoints under `benchmark/`. Repeating an unchanged full-run
command resumes saved milestones, optimizer/scaler/RNG and sampling cursor;
completed evaluations are reused. Changing code, software, split or phase config
requires a new session/root. A partially measured timing trial requires a fresh
root so resumed segments cannot be presented as a clean 100-update measurement.

The Python CLI also provides separate `cache`, `train`, `evaluate` and `report`
commands (`--help` documents their arguments). GPU workers require `--allow-gpu`
and the VoxTell environment. `cache --keys-file FILE` takes a JSON list of original
finding keys; the orchestrator materializes these files automatically. No training
sampler invokes VoxTell. Standalone `report --watch` is a CPU-only report writer;
the orchestrator already owns reporting while it runs, so a second writer fails
its lock instead of competing. No watcher launches GPU work.

## Validation cohort and baseline

Patient ID comes from the CT-RATE filename prefix (`train_N` or `valid_N`). The
5000-candidate search uses seed 20260909 and lexicographic scores: 2a finding
imbalance, all-scan imbalance, then combined quartile/bin L1 imbalance. A sorted
patient tuple breaks remaining ties. Quartiles use annotated voxel counts.

| Subset | All scans / patients | 2a findings / scans / patients | Cached Dice | Cached hits |
| --- | --- | --- | ---: | ---: |
| A | 100 / 95 | 35 / 32 / 31 | 0.339415922608 | 32 / 35 |
| B | 100 / 95 | 34 / 31 / 30 | 0.334950148355 | 27 / 34 |
| Full | 200 / 190 | 69 / 63 / 61 | 0.337215396164 | 59 / 69 |

Split content SHA256:
`c2155bffd6fbf9c3b787ff3e4cad046cf90a801804613c984edfb50cbf822d78`.
Preparation regenerates and verifies this frozen split. Original training has
1120 2a findings on 801 scans / 741 patients, with no 2a patient overlap with val.

Hits follow the existing Dice >= 0.1 convention. Voxel classification uses
probability >= 0.5, equivalently logit >= 0. The original prompt index is retained
even after non-2a channels are omitted. Training examples use each finding's
binary union of instance IDs. Available foreground coordinates are deterministically
sampled without replacement into a maximum 4096-point pool per mask for fast patch
loading; findings remain uniform within each source.

Evaluation restores thresholded predictions and cached targets through the
existing native uncrop and CT orientation transforms. This is equivalent to
thresholding restored logits because restoration has no interpolation here.
Original total GT voxel counts retain false negatives outside the inherited CT
crop. Residual statistics describe the native preprocessed extent; edit fractions
use restored geometry. A continuous restoration helper also preserves uncapped
logits. Validation baseline Dice must reproduce the exact consumed cache to
1e-10 per finding; a mismatch stops the run.

## Verification

CPU tests cover frozen-manifest coverage and grouping, deterministic split search,
all four memberships and source/patch mixtures, original indices, empty prediction
fallback, joint flips, edge padding, base-storage signs, loss masks, empty GT,
identity initialization, finite gradients, synthetic fitting, tiled identity and
blending, original-geometry restoration, per-finding edits and A+B recomposition,
checkpoint equivalence, pristine warm-up restoration, interruption recovery,
complete synthetic evaluation and reuse, failure propagation, dashboard pending
and failed states, writer locking, and GPU/schedule launch gates.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s scripts/rexgroundingct -p test_exp027_contract.py -v

docker run --rm --network none --cpus 2 --memory 8g --read-only \
  --tmpfs /tmp:rw,size=1g --user 1000:1000 \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/scripts/rexgroundingct:/workspace/external/VoxTell \
  -v "$PWD:/workspace:ro" rexgroundingct-voxtell:cu126 \
  python -m unittest discover -s /workspace/scripts/rexgroundingct \
  -p test_exp027_torch.py -v

bash -n scripts/rexgroundingct/run_027_host.sh
python scripts/rexgroundingct/check_repo_workflow.py
git diff --check
```

The Docker invocation has no `--gpus` flag, and the test explicitly asserts CUDA
is unavailable. Its existing image ID is
`sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f`.
Synthetic fitting and worker fixtures use small CPU tensors and do not measure
192³ performance. The clearly labeled populated dashboard fixture was visually
checked at `/tmp/rex027_synthetic_dashboard/reports/live_dashboard.png`; it is
separate from the real pending dashboard.

Real 192³ GPU memory feasibility, throughput, actual logit export and segmentation
quality remain untested until approval. OOM and invalid metrics stop the group;
tile size, loss and method never change automatically.

Final verification results:

- 16 host CPU contract tests passed (2.312 seconds for the unit suite).
- 10 GPU-disabled Docker PyTorch tests passed (3.417 seconds for the unit suite).
- Canonical repository workflow passed: existing challenge-info tests, experiment
  consistency, Python compilation, Git whitespace and tracked-cache checks. It
  reports 19 pre-existing missing-runtime/sync warnings for other experiments.
- Exp027 strict runtime consistency passed with matching canonical config bytes.
- Bash syntax and four-arm dry-run checks passed. The dry run reports 209 required
  finding keys across 196 CT cases and `gpu_work_started: false`.
- Both the pending dashboard and the clearly labeled populated synthetic figure
  were inspected. Runtime currently contains only metadata/configs and reports;
  no arrays, checkpoints, worker logs or measured results exist.

These unit-suite times are not the requested 192³ benchmark timings. The timing
test and all experimental conclusions remain pending. No files were staged or
committed as part of this handoff.

## Approved benchmark launch

The user approved immediate launch on idle GPUs 0–3. Initial metadata preflight
exited before cache generation or training because Python 3.12's `sum()` changed
last-bit rounding of derived baseline means relative to the host Python.
Aggregation now uses `math.fsum`; all source hashes, finding records and split
comparisons remain exact. The 16 CPU contract tests passed again with a regression
check for this case. The original prepared record and failed launch manifest are
preserved under runtime `launch_attempts/001_metadata_preflight/`.

Attempt 2 passed metadata preflight but exited during text-backbone loading:
the Docker image's default `HF_HUB_CACHE` pointed to unwritable `/workspace/.cache`.
The host launcher now explicitly sets both Hugging Face hub/Xet cache paths to
the existing mounted cache. Its failure record is preserved under runtime
`launch_attempts/002_text_cache_environment/`.

Attempt 3 successfully started VoxTell tile inference on all four GPUs. Runtime
`run_manifest.json` records its process ID, exact command, image, authorization
and source hashes. Neither prior attempt completed a case cache or reached a
measured optimizer update; the prescribed 100-update benchmark is still fresh.

## Validation checksum correction and preparation resume

Attempt 3 stopped during cache preparation with 21 completed training CT caches
and zero measured updates. The reader incorrectly compared a whole `.npy` file
SHA against SideExp003's contiguous-array-payload SHA. Both reported source files
were verified intact with the correct convention. The corrected reader also
checks shape, dtype and file size, and rejects modified payloads.

All 17 CPU contract tests passed, including the new header-versus-payload
regression. A GPU-disabled Docker check verified both actual failing cases through
the native crop/orientation restoration, reproducing their baseline Dice exactly:
`train_13013_a_1.nii.gz::1` = 0.009317348919042838 and
`train_13082_a_1.nii.gz::2` = 0.2800000004235294.

Before attempt 4, all 21 training-cache file hashes were reverified. Only the
validation reader changed: reversing that patch reproduced the exact previous
source SHA, and every other production source hash matched. The original metadata
and failed session were archived, while the unchanged training arrays were retained
with explicit producer-contract and compatibility-audit provenance. See runtime
`cache_compatibility_audit.json` and
`launch_attempts/003_validation_payload_hash/`.

The resumed preparation timing covers remaining work and reuse verification.
Earlier failed-attempt overhead is recorded separately. Training and full-volume
evaluation measurements still begin from their prescribed fresh state.

## Shared-context startup correction

Attempt 4 completed all 196 case caches, then stopped before measured updates
because sibling trainers used a fail-fast lock for their shared context file.
That short metadata critical section now waits up to 30 seconds for another
worker, while the exclusive run/report locks retain their fail-fast behavior.

All 11 GPU-disabled Docker tests passed, including a new four-process regression
that holds the context lock until all four workers are waiting, releases it, and
requires every worker to return the same context hash. Model identity, synthetic
fitting, warm-up restoration, checkpoint resume and interrupted-worker recovery
also passed. Together with the 17 CPU contract tests, this gives 28 passing tests.

The failed startup and original worker source are preserved under runtime
`launch_attempts/004_context_lock/`. Only worker lock handling changed; all cache
generation sources are byte-identical. All 196 arrays are retained with original
producer provenance and `cache_context_lock_compatibility_audit.json`; their file
hashes are checked again before training. Attempt 5 starts a fresh benchmark
context. No measured checkpoint or update was carried over.

## GPU warm-up failure: current benchmark outcome

Attempt 5 passed context creation, then all four arms failed in the 192³ FP16
synthetic warm-up when `clip_grad_norm_(error_if_nonfinite=True)` detected a
non-finite gradient norm. The failure followed backward/unscale and occurred
before any measured optimizer update. The exact numerical cause is not yet
established; AMP loss scaling is the next diagnostic target.

All 196 CT caches are complete and retained. There are zero measured histories,
zero training checkpoints, and zero evaluation summaries. All four GPU workers
exited; an explicit GPU process check confirmed the devices idle. No precision,
tile-size, loss or optimizer change was made in response. Runtime status remains
`failed`, with the failure and suggested next diagnostic recorded in
`reports/failure_report.md`. CPU workflow checks still pass; GPU numerical
stability and the requested training/evaluation timing remain unverified.
