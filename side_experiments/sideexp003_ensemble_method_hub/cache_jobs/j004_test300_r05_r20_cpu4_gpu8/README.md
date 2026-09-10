---
created: 2026-09-10
updated: 2026-09-10
status: active
---

# Test300 ranks 5–20 with four CPU workers

User-authorized continuation after p001 profiling. Read
[the execution specification](../../continue_test300_spec.md).
Four GPUs each feed one CPU process through at most two pending local crop
files. NumPy huge-page advice is disabled in every process. The original
checkpoint identities, preprocessing, prompts, precision and numerical
functions are retained; new v4 cache keys bind this implementation.

The original j003 Wave 1 and s001 d1/d2/d3 outputs are preserved. Only j004
schedules original waves 2–5. Runtime state and minute history live under
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j004_test300_r05_r20_cpu4_gpu8/`.

```bash
python side_experiments/sideexp003_ensemble_method_hub/continue_test300.py watch
```

The detached supervisor keeps its own and the parent j003 launch locks.
A `control/ABORT` sentinel stops owned workers; never restart the original
j003 coordinator. Restart j004 only after its previous processes have exited:
`python side_experiments/sideexp003_ensemble_method_hub/continue_test300.py run`.
Completed stages and caches are verified before reuse. Future waves wait if
GPU, RAM or projected float32 publication capacity is insufficient. The shared
20 TiB reserve remains in force; unrelated jobs are never paused.

`case_timings/` separates GPU prediction, embedding, handoff, geometry,
CPU staging and hash work. `wave_XX_timing.json` and `inventory.json` are
written at wave completion. The launch report and first-30-minute snapshots
record live progress and provisional ETA updates. Largest smoke cases are
retained as production outputs; no benchmark model inference is repeated.
