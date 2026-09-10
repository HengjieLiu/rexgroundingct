# GPU8 test300 recovery with uninterrupted CPU search

Authorized 2026-09-09: resume and test the existing test300 inference while
leaving val200 ensemble search running without interruption. Accept resource
contention. This supersedes the CPU pause/recovery policy in the original
test300 execution spec; all inference and storage contracts remain in force.

Use a separately recorded host coordinator, `resume_test300_no_pause.py`,
which inherits the frozen runner's export scheduling and safety gates but
disables CPU pause and unpause operations. Preserve the original job spec,
source bundle, worker implementation, cache keys, and staged arrays. Record
the recovery coordinator and this policy's hashes separately from the frozen
inference provenance. Refuse an unresolved pause record without touching CPU
processes. Never change CPU worker count, supervisor, container, or scoring.

Under the existing job launch lock, archive prior supervisor state, log,
progress and pause records in a unique recovery session before launching.
Use the existing per-worker lock and hash/shape/prompt checks to reuse valid
staging. Incomplete cases are recomputed; resumed smoke inference rechecks
GPU memory while preserving saved smoke arrays. Retain the 20 TiB shared
reserve, per-wave space projection, 20,000 MiB launch and 4,096 MiB smoke gates,
label-free CT geometry, atomic publication and exact storage mask validation.

Before launch, run no-pause regression tests, SideExp003 tests, repository
checks, frozen-source validation, and a read-only validation of all saved
staging in the pinned image. Test containers must not access segmentations.
Then launch all four Wave 1 workers, observe their smoke gates and progress
beyond the saved 9/10/10/11 cases. Later waves continue automatically.

```bash
python side_experiments/sideexp003_ensemble_method_hub/resume_test300_no_pause.py \
  --job side_experiments/sideexp003_ensemble_method_hub/cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json \
  --session-id no_pause_YYYYMMDDTHHMMSSZ
```

Runtime remains the original shared job root; the recovery session contains
`execution_manifest.json`, `prior/` evidence and source snapshots. The detached
supervisor log lives under the job's `logs/` using the recovery session ID.
Monitor original `state.json` and `progress/` every minute.
Record real GPU progress, CPU identity/state, disk capacity and validation in
a compact recovery launch report. The original time estimate is provisional;
use measured completed-wave durations to update it. Completion remains 20
strict caches of 300 CTs / 582 prompts, followed by paired val/test inventory.
