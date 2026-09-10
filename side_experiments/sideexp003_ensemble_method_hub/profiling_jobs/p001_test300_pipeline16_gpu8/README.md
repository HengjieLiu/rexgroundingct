# p001: test300 pipeline profiling

**Complete:** all 16 inference jobs and six CPU comparisons validated.
Read the [final report](pipeline_report.md), [verification](completion_verification.json),
and [storage inventory](storage_inventory.json). Best measured setting: four CPU
workers with `NUMPY_MADVISE_HUGEPAGE=0`. Production Waves 2–5 remain held.

See [execution specification](../../profile_test300_spec.md). Uses ranks 5–8 of
j003 for 16 total model–case jobs with one inference pass. CPU replay uses the
same saved crop logits. Production Waves 2–5 remain held.

Commands, from repository root:

```bash
python side_experiments/sideexp003_ensemble_method_hub/profile_test300.py prepare
python side_experiments/sideexp003_ensemble_method_hub/profile_test300.py run
python side_experiments/sideexp003_ensemble_method_hub/profile_test300.py watch
```

The run command should use a detached supervisor. Immutable `job.json` binds the
checkpoint/config/image/source/dataset and selected cases. Authoritative runtime:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/profiling_jobs/p001_test300_pipeline16_gpu8`.
Progress is `state.json`, `gpu/`, and `cpu4|8|16/progress.json`. Completion writes
`report.json`, `report.md`, and `case_timings.csv`. Crop and staging arrays remain
under `/data/hengjie/sideexp003_staging/p001_test300_pipeline16_gpu8`.


## Allocation diagnostic and final report

[Huge-page diagnostic](hugepage_diagnostic.md) records the identified kernel
allocation stalls and the exact-hash real-case fix test. The CPU-only follow-up
uses `profile_test300_nohuge.py` and `NUMPY_MADVISE_HUGEPAGE=0`, in separate
`nohuge_cpu4|8|16` namespaces. It adds no inference and changes no host settings.

`profile_test300_report.py --wait` consolidates both matrices into
`pipeline_report.md`, `pipeline_analysis.json` and `pipeline_case_timings.csv`
here after all six comparisons pass. Baseline-only runtime `report.md` is an
intermediate report; use the consolidated report for the recommendation.

`prelaunch_selection_audit.json` is an unused preparation snapshot. Only
`job.json` with SHA
`62eada06c82d485229fa18dd14be455aff445eb89baac63712ed660e51c615ab`
was launched; its first two cases preserve largest-input/output smoke coverage.
