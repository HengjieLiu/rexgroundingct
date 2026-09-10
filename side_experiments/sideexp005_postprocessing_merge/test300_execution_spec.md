# Automatic test300 d11/d12 generation

User authorization on 2026-09-10 supersedes the earlier validation-review gate.
Run ID: `r002_d1_test300_d11_d12_frozen_postprocessing`.

The detached supervisor polls every 30 seconds for the original val200
completion, verified report collection and successful container exit. Report
identities and coverage must pass; no Dice, HIT or relative performance
threshold applies. A failed or incompatible dependency blocks the test run.
The original val runner, collector, numerical sources and Waves 2–5 exporter
remain independent and unchanged.

Inputs are the completed s001 d1 masks, the same frozen four checkpoint
identities, the official 300-case / 582-prompt test cohort, and Exp023's audited
CT-RATE ts_total masks. Lobes are labels 10–14. d11 applies frozen collaborator
semantic-v1; d12 applies strict semantic-v2 to d11. All fallback, margins and
coordinate calculations are unchanged. The frozen adapter is called with
unused d2/d3 routes disabled, with exact parity required. There is no model
inference, re-averaging, CT reorientation or anatomy resampling. Test labels
are not mounted or read; test scores are unavailable. The source anatomy's
PASS_PENDING_MANUAL_VISUAL_REVIEW status is retained.

Code is independently snapshotted from r001's frozen source tree, plus the new
test runner/config/launcher/spec. Small configuration and aggregate records
remain in SideExp005. Heavy files, logs and private findings live in this
run's external SideExp005 runtime. Predictions are written beside existing
Exp024 outputs, exclusively in d11 and d12. Each contains exactly 300 binary
uint8 native FXYZ NIfTIs and the same CT header convention as d1. No ZIP or
upload is produced.

Four spawned CPU workers each use one numerical-library thread and
NUMPY_MADVISE_HUGEPAGE=0. Container limits: four CPU cores, 96 GiB RAM, no GPU
devices, no network. Anatomy support is cached within each case only. Smoke
cases are the largest native CT and largest finding array, with retained
outputs. Progress is published each minute and workload-normalized ETA after
ten cases. Reading/hash, postprocessing, writing and verification timings are
recorded separately. Approximately 109.03 GiB of uncompressed output is bounded
by a conservative 150 GiB shared headroom budget in addition to the 20 TiB
reserve. Insufficient space waits before launching or dispatching more cases.
Other jobs are never paused and existing files are never deleted for space.

Exclusive supervisor and worker locks prevent duplicate ownership. Each case
has a durable prepared journal before atomic destination renames, then a
complete record. Restart reuses only matching identities and hashes; foreign
outputs fail closed. A controlled stop finishes in-flight cases. Failures are
recorded with traceback and preserve completed artifacts for explicit restart.

Acceptance: frozen-method parity, v2 bypass and subset invariants, numeric
ordering, flipped/anisotropic geometry, corrupt hashes, interrupted publication,
duplicate ownership, report gating with both better and worse scores, storage
wait/recovery, and a complete synthetic 300-case/582-prompt run. Existing
SideExp005, register, relevant SideExp003 and repository checks must pass.

Completion verifies all 600 output hashes, geometry, binary masks, both
300-case / 582-prompt inventories, routing and foreground-count records.
The independent host collector publishes aggregate evidence and updates only
d11/d12 test readiness, preserving validation results and manual upload
history. ZIP status is skipped_by_request with null path/hash. Registry writes
begin only after the old val collector has finished. This job does not alter
the old collector's historical authorization record.

Commands from the repository root:

```bash
python side_experiments/sideexp005_postprocessing_merge/launch_test300.py
python side_experiments/sideexp005_postprocessing_merge/test300_postprocessing.py watch --once
```

The launcher freezes and preflights in the pinned image before detaching.
`--preflight-only` performs launch preparation without arming. The numerical
runner provides preflight/run/watch/report with `--config`; execute numerical
commands from the frozen source inside the pinned container. Logs and exact
reproduction commands are retained in the runtime launch records.
