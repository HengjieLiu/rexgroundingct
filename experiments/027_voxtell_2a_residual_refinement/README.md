---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Four-condition 2a residual logit refinement

This is a repo-local index for a runtime experiment. Edit the canonical
config in the repo, and keep large runtime artifacts under `/mnt/shengdata1`.

## Folder Map

- `README.md`: this experiment's status and ownership rules.
- `codex_execution_spec.md`: active or historical Codex execution spec.
- `metrics_summary.json`: synced metrics and provenance summary.
- `sync_manifest.json`: hashes, paths, and sync provenance.
- `report.md`: small copied runtime report when available.
- `runtime`: ignored symlink to heavyweight runtime outputs.

## Status

- Status: `cache_preparation`
- Canonical config: `configs/experiments/027_voxtell_2a_residual_refinement.json`
- Execution spec: `experiments/027_voxtell_2a_residual_refinement/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/full_fp32_100ep`
- Runtime link: `runtime` is ignored by git and points to `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement`.

## Synced Small Artifacts

- Report snapshot: `not available`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
- Keep `codex_execution_spec.md` current before substantial long-running work.

## Exp027 references

- [Accepted questions and decisions](design_decisions.md)
- [Live full-run dashboard](runtime/full_fp32_100ep/reports/live_dashboard.md)
- [Live full-run subplot figure](runtime/full_fp32_100ep/reports/live_dashboard.png)
- [Full-run verification and launch](full_run_verification.md)
- [Verification and launch commands](verification.md)
- [Warm-up numerical diagnosis and AMP recovery](numerical_diagnosis.md)
- [FP16 / FP32 comparison](runtime/reports/precision_comparison.md)
- [Completed precision benchmark results](precision_benchmark_results.md)
- [FP16 100-update results](fp16_benchmark_results.md)
- [FP32 benchmark configuration](fp32_benchmark_config.json)
- [Data-loading audit and proposed improvements](data_loading_audit.md)

The authorized full run uses FP32, 100 epochs × 100 updates, with evaluation every 10 epochs.
Complete caching of all training and validation 2a cases is required before training.
All results remain pending user review. No automatic ranking or promotion.
