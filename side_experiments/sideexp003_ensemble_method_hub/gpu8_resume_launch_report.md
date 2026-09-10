# GPU8 seeded-search resume launch record

Snapshot: 2026-09-09 UTC (2026-09-08 local). Branch: `gpu8`.

The authorized benchmark completed in 1040.49 seconds, within its 1200-second
budget. All 20 cases in every completed trial reproduced the archived K8 trial
rows and voxel counts exactly.

| Configuration | Compute seconds | Correctness |
| --- | ---: | --- |
| Five workers, first baseline | 523.785 | Passed |
| Ten workers | 226.069 | Passed |
| Five workers, repeated baseline | 265.661 | Passed |

Ten workers improved over the faster baseline by **14.9033%**, below the
authorized 15% selection threshold. The supervisor therefore retained **five
workers** and the original execution identity. Twenty workers was skipped to
reserve time for the baseline repeat. Filesystem caches were not cleared.

The production worker was launched at `2026-09-09T00:27:47Z`, reconstructing
completed passes from their saved metrics before continuing K9. Its target is
K16 for all five scopes. This launch record is not a completion claim.

## Runtime evidence

Parent job: `a001_top16_after_wave4_fullval`.

Session root:

```text
/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/analysis_jobs/a001_top16_after_wave4_fullval/resume_sessions/gpu8_20260909T001017Z
```

- `state.json`: current supervisor state, case counts, and subsequent ETA.
- `benchmark_report.json` / `.md`: measured results and selection rule.
- `archive/`: parent reports, execution contracts, all 1027 original partials,
  their hash inventory, historical timings, helper source, and test evidence.
- `selected_execution.json`: unchanged parent execution identity and five-worker
  selection.
- `production.log`, `production_command.json`, `production_telemetry.json`:
  command, runtime log, CPU/memory observations.
- `closeout.json` / `.md`: written only after complete K16 validation.

The shared launch lock is held throughout the session. The frozen evaluator
modules and source logits were not modified. The 14 focused helper tests, full
SideExp003 suite, hub check, and canonical repository workflow check passed;
the workflow check retained 19 existing historical provenance warnings.

All results remain optimistic same-val200 diagnostics. No derived logits,
model inference, merge, push, or final-recipe promotion is part of this run.
