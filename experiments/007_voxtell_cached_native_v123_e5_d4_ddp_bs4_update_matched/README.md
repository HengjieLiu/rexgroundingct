---
created: 2026-07-22
updated: 2026-07-30
status: active
---

# VoxTell cached-native v123 e5/d4 DDP batch4 update-matched run

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

- Status: `phase2_continuation_active`
- Canonical config: `configs/experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched.json`
- Execution spec: `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched`
- Runtime link: `runtime` is ignored by git and points to the runtime directory.

## Synced Small Artifacts

- Report snapshot: `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/report.md`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Active Continuation

- Run group: `exp007_cont100_from_ddp100_20260730T062051Z`
- Container: `rex007_cont100_from_ddp100_20260730T062051Z`
- Source checkpoint: original exp007 `exp007_full_20260725T231624Z/ddp_bs4/checkpoints/checkpoint_update_010000.pth`
- Load policy: network weights only; optimizer, AMP scaler, scheduler, and
  update counter are reset for a fresh 100-epoch continuation.
- Schedule: continuation slice from deterministic 80,000-event DDP schedule,
  using source events `40000..79999` rewritten to zero-based event indices for
  the trainer.
- Evaluation barriers: relative epochs `25/50/75/100`, labeled as absolute
  exp007 epochs `125/150/175/200`, with fixed val200 evaluation at each barrier.

## Phase 3 queued behind Exp021

- Execution spec: `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/phase3_execution_spec.md`
- Source checkpoint: this experiment's phase-2 absolute epoch-200 checkpoint,
  loaded as network weights only.
- Phase-3 barriers: relative epochs `10, 20, ..., 100`, labeled as absolute
  epochs `210, 220, ..., 300`, with full fixed-val200 evaluation at every barrier.
- Live report: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/phase3_status.md`
- Poller status: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/phase3_autostart/poller_status.md`
- The Exp021-gated poller is defined by
  `scripts/rexgroundingct/poll_exp021_then_launch_007_phase3.sh` and the
  `exp021_to_exp007_phase3_poller.service` systemd user unit.

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
- Keep `codex_execution_spec.md` current before substantial long-running work.
