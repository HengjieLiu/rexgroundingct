# SideExp005: frozen postprocessing comparison

This independent side experiment compares d1/d2/d3/d11/d12 on the fixed
200-case / 381-finding validation cohort. It uses the same four checkpoints as
test d1 and the audited official CT-RATE `ts_total` anatomy from Exp020.
It does not enter the canonical experiment registry.

The collaborator ZIP and source audit were moved unchanged from the former
Exp027 folder; `relocation_manifest.json` records verified hashes. Exp027 is
reserved for the independent experiment on main.

## Files and execution

- `config.json`: canonical fixed run configuration.
- `codex_execution_spec.md`: method, execution and acceptance contract.
- `runner.py`: preflight, run, watch and report commands.
- `methods.py`: input adapter for frozen postprocessing methods.
- `frozen_sources/`: unchanged collaborator source and provenance.
- `test_postprocessing.py`: numerical, source-parity and recovery checks.
- `report.md` and `metrics_summary.json`: small verified aggregate closeout.

Heavy outputs and all private finding records belong under
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r001_d1_val200_frozen_postprocessing/`.

Run `python runner.py preflight`, then `python runner.py run` in the pinned
CPU-only image. `python runner.py watch --once` reads progress;
`python runner.py report` verifies completion and rebuilds reports.
`launch.py` performs preflight and starts a detached supervisor with four CPU
workers. It mounts source data read-only and only this run's runtime writable.

The original r001 run remains a validation-only audit. d12 means semantic-v1
followed by the frozen semantic-v2 strict rules; d2/d3 each derive from d1.

## Automatic test d11/d12

The user authorized separate run `r002_d1_test300_d11_d12_frozen_postprocessing`
to begin automatically after the val200 report and collector finish, regardless
of validation scores. This supersedes the earlier manual review gate. Read
[the test execution specification](test300_execution_spec.md).

`test300_config.json` freezes the completed test d1 producer, cohort, anatomy
and validation dependency. `launch_test300.py` freezes the r001 numerical
sources plus the new test coordinator, preflights, and arms a detached
supervisor. `test300_postprocessing.py` provides preflight/run/watch/report; use
`python test300_postprocessing.py watch --once` for current progress. Test artifacts
and private records live in r002's external runtime; predictions go directly
to Exp024 `outputs/d11` and `outputs/d12`, with 300 files / 582 prompts each.
No model inference, ZIP creation, upload or method retuning is performed.

Four CPU workers use one numerical thread each and
`NUMPY_MADVISE_HUGEPAGE=0`. The validation runner and its pinned collector are
unchanged. Test register updates start after that collector finishes and
preserve validation results and submission history. See
`test300_launch_report.json` for the detached launch and
`test300_report.md` / `test300_completion.json` for final aggregate evidence
when complete.
