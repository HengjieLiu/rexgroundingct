---
created: 2026-09-10
updated: 2026-09-10
status: active
---

# Waves 2–5 launch and first 30 minutes

User authorized four CPU workers and automatic Waves 2–5 continuation.
Supervisor PID **3974216**, launched **2026-09-10T07:27:36Z** on `shenggpu8`,
branch `gpu8`. All four Wave 2 containers were launched by **07:28:50Z**;
that is the observation start. Original ranks 5–8 occupy GPUs 0–3.

Job content SHA:
`117cc3ffd5007a568e99dd75cc7e6b557877c804644608b59ea4dcd90dc8698f`.
The frozen job includes every checkpoint, data, preprocessing, source and
image identity. Per-container launch commands are in runtime `commands/`.
The new coordinator holds both j004 and parent j003 launch locks, and owns
only its four containers. Four CPU workers overlap restoration/staging with
GPU inference; every process disables NumPy huge-page advice. This changes
allocation behavior, not the frozen numerical inference or export functions.

Validation: **98 SideExp003 tests passed**, including five continuation tests;
the final continuation-only rerun also passed. Repository workflow checks
passed with 19 existing consistency warnings. Largest native-output iso07
saved crop passed exact values, shape, affine, orientation, prompt hashes and
numeric ordering through the new handoff. Native F32 SHA:
`e07700ef593eb9c447db36b228c2e4931cda25e9635a826c351cecb0e3e49626`.
A test fixture filename and prelaunch content-versus-file hash check were
corrected before freezing/launch; no production inference was affected.

Live root:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j004_test300_r05_r20_cpu4_gpu8`.
State: `state.json`; detailed per-case timers: `case_timings/`; requested
observations: `observation/`; supervisor output: `logs/supervisor.log`.
The read-only observation logger PID is **3978784**. Observation ends at
**07:58:50Z**, while the production coordinator continues through Wave 5.

| Observation | Ranks 5/6/7/8 inferred | Ranks 5/6/7/8 native stages | Status | Remaining Waves 2–5 ETA |
| --- | --- | --- | --- | --- |
| ~5 min, 07:33:42Z | 0 / 0 / 1 / 1 | 0 / 0 / 0 / 0 | smoke_running; no errors | 31.0–63.0 h |
| ~10 min, 07:38:54Z | 2 / 2 / 3 / 2 | 2 / 2 / 2 / 2 | exporting; no errors | 31.0–63.0 h |
| ~15 min, 07:43:39Z | 5 / 5 / 6 / 5 | 5 / 5 / 6 / 5 | exporting; no errors | 28.3–58.2 h |
| ~20 min, 07:48:53Z | 9 / 9 / 10 / 8 | 9 / 8 / 10 / 8 | exporting; no errors | 27.5–56.6 h |
| ~25 min, 07:53:59Z | 12 / 12 / 13 / 12 | 12 / 12 / 13 / 12 | exporting; no errors | 26.3–54.2 h |
| ~30 min, 07:59:08Z | 16 / 15 / 17 / 15 | 16 / 15 / 17 / 15 | exporting; no errors | 26.4–54.4 h |

“Inferred” includes a committed local crop handoff; “native stages” counts
completed geometry restoration, clipping, writes and hashes. A full model
cache becomes complete only after 300 cases / 582 prompts pass publication
and strict validation. No Wave 2 model is counted as published yet.

Initial ETA is deliberately provisional. Updated estimates separate per-case
text work from sliding-window workload, input/crop bytes and CPU output bytes,
exclude largest smoke cases once ordinary cases are available, and add an
unmeasured publication allowance. Future-wave architecture mixtures and disk
contention remain uncertainty sources. d1/d2/d3 and original Wave 1 are
unchanged; no test metrics, new ensemble decisions or submissions occur here.


## Observation closeout

The requested 30-minute watch completed at **2026-09-10T07:59:08Z**. **63/1,200 Wave 2 outputs** were staged; all four containers remained running without failures. All eight retained smoke outputs passed the memory gates. Waves 3–5 remain automatically scheduled after their preceding validation barriers. Available RAM was 335.6 GiB, shared free 23.06 TiB, and local free 6.12 TiB.

CPU postprocessing is keeping up with inference; the ordinary-case CPU queue does not justify increasing worker count. Text embedding preparation/transfers remain the dominant measured per-case cost. The current ETA includes publication/validation but remains provisional: only the beginning of Wave 2 is measured, and full-cache publication and later model mixtures have not yet run. The production supervisor continues after this observation; the short read-only observer may exit. Branch remains `gpu8`; `main` is unchanged at `4617db190f1e6bf93e2093fe7ef4741c2bf9bc7d`. Original Wave 1, scoring modules, d1/d2/d3 and unrelated jobs were preserved.
