---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# ReXGroundingCT Experiments

This directory is a lightweight repo-local index. It keeps configs and
small summaries easy to inspect while large outputs remain on `/mnt/shengdata1`.

## Folder Map

- `README.md`: experiment index overview and drift policy.
- `registry.yaml`: generated machine-readable experiment summary.
- `<experiment-id>/README.md`: one experiment's status and ownership rules.
- `<experiment-id>/codex_execution_spec.md`: active or historical Codex execution spec.
- `<experiment-id>/metrics_summary.json`: small synced metrics/provenance summary.
- `<experiment-id>/sync_manifest.json`: synced hash and path manifest.
- `<experiment-id>/report.md`: small copied runtime report when available.
- `<experiment-id>/runtime`: ignored symlink to heavyweight runtime outputs.

## Drift Policy

- Edit canonical configs under `configs/experiments/`.
- Write or update `codex_execution_spec.md` before substantial long-running experiment work.
- Runtime configs are hashed snapshots copied at run start.
- `experiments/*/runtime` symlinks are ignored by git.
- Use `scripts/rexgroundingct/check_experiment_consistency.py` before committing.

## Experiments

| ID | Status | Report | Runtime |
| --- | --- | --- | --- |
| `001_voxtell_v1_1_miccai200_val_eval` | `evaluation_complete` | `experiments/001_voxtell_v1_1_miccai200_val_eval/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval` |
| `002_voxtell_text_ft_miccai_train_val` | `runtime_initialized` | `` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val` |
| `003_voxtell_rex_ft_rescue_ablation` | `runtime_initialized` | `` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation` |
| `004_voxtell_v123_native_vs_2mm_global_context_ft` | `evaluation_complete` | `experiments/004_voxtell_v123_native_vs_2mm_global_context_ft/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft` |
| `005_voxtell_global_proposal_local_cascade` | `evaluation_complete` | `experiments/005_voxtell_global_proposal_local_cascade/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/005_voxtell_global_proposal_local_cascade` |
| `006_voxtell_cached_native_v123_lr_ablation` | `evaluation_complete` | `experiments/006_voxtell_cached_native_v123_lr_ablation/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation` |
| `007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched` | `evaluation_complete` | `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched` |
| `008_voxtell_dual_branch_proposal_refinement_ablation` | `evaluation_complete` | `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation` |
| `009_voxtell_s3_attention_coupling_ablation` | `evaluation_complete` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation` |
| `010_voxtell_public_anatomy_prior_fusion` | `evaluation_complete` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion` |
