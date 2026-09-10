# GPU8 test300 recovery without CPU interruption

Later instruction: finish Wave 1 only and hold ranks 5–20 for performance
investigation. Guard `2974265` is armed; scheduler `2103906` is intentionally
stopped while the four GPU workers continue. The historical recovery launch
below no longer authorizes later waves. Use runtime `finish_wave1_state.json`
and the eventual `wave1_hold_report.md` for the current state.

Launch: 2026-09-09T07:14:38Z, host `shenggpu8`, branch `gpu8`.
Supervisor PID: `2103906`; session: `no_pause_20260909T071438Z`.
Status at 07:42:42Z: recovery test passed; all four Wave 1 models are exporting
new cases. Both retained smoke cases passed on every GPU; the gate opened at
07:35:21Z. No complete 300-case model cache has been published yet.

| Rank | GPU | Cases | Post-smoke free MiB |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 11/300 | 44074 |
| 2 | 1 | 12/300 | 44074 |
| 3 | 2 | 12/300 | 44396 |
| 4 | 3 | 13/300 | 43196 |

The user authorized accepting resource contention while leaving the val200
ensemble search running. A separate `resume_test300_no_pause.py` coordinator
disables CPU pause/unpause, including startup recovery and exit cleanup.
The original frozen runner, inference worker, job spec and cache keys are
unchanged. GPU and disk gates, source drift checks, staging validation and
exact publication checks remain active. The failed coordinator's state,
progress, log and pause record were archived under the recovery session's
`prior/` directory before workers were launched.

Preflight passed all 68 SideExp003 tests, including five no-pause regressions,
the repository workflow check (existing consistency warnings), and the hub
check. Read-only validation in the pinned image, with segmentation files
hidden, verified all 40 staged arrays (50,192,192,512 bytes / 46.75 GiB):
ranks 1–4 retained 9, 10, 10 and 11 cases respectively. Checks cover original
source/job hashes, array hashes, finite clipped float32 values, prompt order,
shape and CT affine. See `recovery_validation.json` for case hashes and the
protected CPU source/container identities.

One new case from each model independently passed the same hash, finite-value,
prompt and CT geometry checks with segmentation files hidden. Evidence is in
`recovery_new_cases_validation.json`. `recovery_observation.json` records the
live worker counts, opened smoke gate, unchanged CPU container/supervisor,
unchanged protected sources and all 20 paired val200 manifest hashes. Shared
free space is 24.02 TiB and local free space is 6.33 TiB; the 20 TiB shared
reserve remains enforced.

Frozen job SHA-256:
`50ac85fb91a37a748be45e82e27d4ed09862645cdb06258b07f67fdd5bf8e912`.
The separately frozen recovery coordinator and policy hashes are recorded in
`recovery_sessions/no_pause_20260909T071438Z/execution_manifest.json` under
the runtime root. This records the changed scheduling policy without claiming
that the inference source or existing logits changed.

The CPU search retains supervisor `1578545`, container
`rex-resume-gpu8-20260909t001017z-production`, and five workers. At 07:42:42Z
it was unpaused, completed through K=12, and processing K=13 (18/200 cases).
No CPU process
was stopped, paused, unpaused or reconfigured by this recovery.

Runtime root:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j003_top20_test300_fresh_gpu8`.
Read `state.json`, `progress/` and
`logs/no_pause_20260909T071438Z.supervisor.log` on the live gpu8 host.
The initial 24–30 hour export estimate is provisional and needs replacement
using completed gpu8 waves; this restart includes repeated smoke checks.
During the largest smoke case, a nonblocking GPU-worker stack sample showed
CPU conversion to contiguous native CT layout after GPU prediction. GPU
utilization therefore varies across a case; low utilization alone does not
establish a stalled worker or CPU-search contention. The sample is archived
in the runtime logs as `no_pause_20260909T071438Z.gpu-stack.txt`.

Branch remains `gpu8`; `main` remains at
`4617db190f1e6bf93e2093fe7ef4741c2bf9bc7d`. No commits or branch switches were made.

Restart after confirming no live owner, using a new session ID:

```bash
python side_experiments/sideexp003_ensemble_method_hub/resume_test300_no_pause.py \
  --job side_experiments/sideexp003_ensemble_method_hub/cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json \
  --session-id no_pause_YYYYMMDDTHHMMSSZ
```

Do not use the original `hub.py cache run` entry point for this policy: it
retains automatic CPU pausing. `hub.py cache watch --job
j003_top20_test300_fresh_gpu8 --once` remains a read-only status command.
Completion still requires all 20 strict 300-case / 582-prompt test caches.
