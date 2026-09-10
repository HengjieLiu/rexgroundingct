---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Top-four test300 d1/d2/d3

**Complete: all d1/d2/d3 folders and ZIPs are validated.**
Finished at 2026-09-10T02:37:02Z (September 9, 7:37 PM PDT). Each variant
contains 300 native CT files / 582 prompts. The three ZIPs passed independent
SHA256, CRC and exact root-file-set checks. Container
`sideexp003_s001_top4_test300_d123_w16` exited with code 0 after a 1 h 40 min
continuation that retained both original smoke cases. Waves 2–5 remain held.
No upload or submission was performed.

See [the completion report](report.md), [completion record](completion.json),
[archive verification](completion_verification.json),
[16-worker execution policy](../../submission_16_execution_spec.md), and
[first-batch verification](continuation_16_first_batch.json).
Runtime state, per-case journals, manifests, source snapshots and handoff
records remain on the shared mount. Completed outputs are retained; there is
no need to restart either submission container.

Read [the execution contract](../../submission_test300_spec.md).
`job_spec.json` binds j003 ranks 1–4, test prompts, audited anatomy and source
hashes. The detached CPU-only runner waits for the Wave 1 finish guard, four
strict caches, coordinator exit and storage headroom. It leaves ranks 5–20 held.

```bash
python side_experiments/sideexp003_ensemble_method_hub/submission_test300.py \
  run --job side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/job_spec.json

python side_experiments/sideexp003_ensemble_method_hub/submission_test300.py \
  watch --job side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/job_spec.json --once
```

Use the pinned VoxTell image and the mount/launch command in `launch_report.md`.
The job owns its shared `launch.lock`; duplicate runners fail. On failure,
inspect `state.json` and the container log before restarting this same frozen
job. Never overwrite conflicting output files or modify its frozen code.

Runtime root:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/`.

Runtime artifacts include `input_binding.json`, `owner.json`, `state.json`,
`smoke_validation.json`, per-case publication journals in `cases/`,
`d1.manifest.json` / `d2.manifest.json` / `d3.manifest.json`, ZIP publication
journals, `completion.json` and `report.md`. Completion requires three validated
300-file / 582-prompt sets and three independently verified ZIPs.

Deliverables live under Exp024 `outputs/d1`, `outputs/d2`, `outputs/d3` and
`outputs/d1.zip`, `outputs/d2.zip`, `outputs/d3.zip`. The existing a/b files and
Exp024 manifests are preserved. d1 averages sigmoid probabilities; d2 applies
eligible whole-lung 20 mm support; d3 applies eligible prompt-selected 20 mm
support independently to d1. Anatomy manual-review status is retained in
provenance. This job creates packages only, without uploading them.
