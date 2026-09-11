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

- Status: `stopped_by_user`
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

- [LIVE: four losses fitted on A](runtime/deletion_loss_ablation_a_20ep/reports/live_dashboard.md)
- [A-only loss comparison: specification and verification](deletion_loss_ablation/README.md)
- [Completed four-data-source deletion dashboard](runtime/deletion_four_arm_20ep/reports/live_dashboard.md)
- [Four-arm deletion run and verification](deletion_four_arm/README.md)

- [Frozen Exp007 base probability-threshold sweep](base_probability_threshold_sweep.md)

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

- [Interim learning diagnosis through epoch 40](learning_diagnosis_e040.md)

- [User-requested stop after epoch 50](user_stop_after_val50.md)

- [Audit of the proposed base-positive tile gate](positive_tile_gate_audit.md)

- [Proposed categorical keep/remove/add editor](categorical_editing_proposal.md)

- [Authorized deletion-only fitting diagnostic](deletion_diagnostic/README.md)

The original schedule was FP32, 100 epochs × 100 updates, with evaluation every 10 epochs.
The user subsequently requested stopping after all epoch-50 validations, before update 5,001.
Complete caching of all training and validation 2a cases is required before training.
All results remain pending user review. No automatic ranking or promotion.
