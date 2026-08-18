---
created: 2026-08-13
updated: 2026-08-13
status: active
snapshot_date: 2026-08-13
scope: top20_val200_no_ensemble
---

# Top 20 Fixed-Val200 Models Without Ensembles

This snapshot ranks local evaluated checkpoints by the ReXGroundingCT fixed
val200 challenge metric, `mean_global_dice_per_finding`, on 200 cases and 381
findings.

Filtering used for this snapshot:

- included only `val_quick_global_eval.json` files whose path contains
  `val200`;
- excluded paths containing `ensembles`, `probavg`, `multiscale`,
  `lung_gating`, `oracle`, or `prompt_only`;
- treated each evaluated checkpoint as one model, so multiple checkpoints from
  the same arm can appear.

Common training defaults unless overridden in the method column: cached-native
VoxTell preprocessing with crop-to-nonzero, whole-crop z-score normalization,
native `192^3` patches, v123 prompt policy and loss, SGD Nesterov, momentum
`0.99`, weight decay `3e-5`, encoder LR `1e-5`, decoder or non-encoder LR
`1e-4`, 100-update warmup, polynomial LR decay with power `0.9`, gradient clip
`12`, and fixed-threshold `0.5` val200 evaluation.

| Rank | Model / checkpoint | Val200 Dice | Hit rate | Hits | Detailed training method |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Exp007 `ddp_bs4` continuation relative epoch 50, absolute epoch 150 | 0.346023 | 0.776903 | 296/381 | DDP global-batch-4 continuation from original Exp007 epoch 100. Loaded source network weights only, reset optimizer, AMP scaler, LR schedule, and update counter, then trained on the second deterministic 40,000-event slice. |
| 2 | Exp009 `baseline_cont100` epoch 100 | 0.339839 | 0.755906 | 288/381 | Same-source single-GPU continuation from Exp006 `v123_cached_e5_d4` epoch 100. No S3 module; fresh optimizer, scaler, warmup, and 10,000-update poly horizon. |
| 3 | Exp009 `s3v2_balanced_feature_half_quarter` epoch 100 | 0.336718 | 0.766404 | 292/381 | S3 attention continuation from Exp006 with half- and quarter-scale attention. Uses balanced feature modulation `skip' = skip * (1 + alpha_eff * (2A - 1))`, with coupling ramped to target strength by update 2000. |
| 4 | Exp008 `v1_sharedfusion_softguide` epoch 100 | 0.336590 | 0.753281 | 287/381 | Dual-branch proposal/refinement continuation from Exp006. Shared text-image fusion, detached soft proposal guidance, proposal v123 plus recall-Tversky auxiliary loss, and final v123 objective. |
| 5 | Exp008 `v1_dualfusion_softguide` epoch 100 | 0.336049 | 0.750656 | 286/381 | Dual-branch proposal/refinement continuation from Exp006. Branch-adapted fusion with detached soft proposal guidance; otherwise same objective and schedule as Exp008 shared-fusion arm. |
| 6 | Exp009 `s3v1_fixedrho_suppress_half_quarter` epoch 100 | 0.334709 | 0.748031 | 285/381 | S3 attention continuation from Exp006 using the original fixed-rho suppressive gate at half and quarter scale, with `rho=0.25` and ramped effective strength. |
| 7 | Exp007 `ddp_bs4` continuation relative epoch 100, absolute epoch 200 | 0.334659 | 0.766404 | 292/381 | Final checkpoint of the Exp007 phase-2 DDP continuation: global batch 4, weights-only initialization from original Exp007 epoch 100, fresh 10,000-update optimizer/LR horizon. |
| 8 | Exp008 `v3_dualfusion_softguide_joint` epoch 100 | 0.334520 | 0.755906 | 288/381 | Dual-branch proposal/refinement continuation from Exp006. Branch-adapted fusion with soft proposal guidance and joint guide-gradient flow into the proposal branch. |
| 9 | Exp008 `v2_dualfusion_precision` epoch 100 | 0.333285 | 0.753281 | 287/381 | Dual-branch proposal/refinement continuation from Exp006. Branch-adapted detached soft guide plus modest final precision pressure using precision-Tversky weight `0.1`. |
| 10 | Exp012 `category_2a_replay50` epoch 100 | 0.333278 | 0.748031 | 285/381 | Category specialist from Exp009 `baseline_cont100` epoch 100. Targeted category `2a`; each epoch uses 50 targeted events and 50 shared natural-replay events; fresh optimizer and v123 schedule. |
| 11 | Exp007 original `ddp_bs4` epoch 50 | 0.333268 | 0.755906 | 288/381 | Public VoxTell v1.1 initialization with cached-native v123 training under DDP world size 4, per-GPU batch 1, global batch 4. This checkpoint is after 5,000 updates. |
| 12 | Exp008 `v1_sharedfusion_softguide` epoch 80 | 0.332809 | 0.742782 | 283/381 | Same shared-fusion dual-branch proposal/refinement method as rank 4, evaluated at the update-8,000 pause checkpoint before final resume. |
| 13 | Exp009 `s3v3_logit_residual_half_quarter` epoch 100 | 0.332661 | 0.761155 | 290/381 | S3 attention continuation from Exp006 using zero-initialized logit residual coupling from half- and quarter-scale attention maps. |
| 14 | Exp007 original `ddp_bs4` epoch 75 | 0.331810 | 0.761155 | 290/381 | Original DDP global-batch-4 cached-native v123 run from public VoxTell v1.1, evaluated at the 7,500-update checkpoint. |
| 15 | Exp007 `ddp_bs4` continuation relative epoch 75, absolute epoch 175 | 0.331799 | 0.750656 | 286/381 | Phase-2 DDP continuation from original Exp007 epoch 100 with weights-only initialization and fresh optimizer/LR state, evaluated at 7,500 continuation updates. |
| 16 | Exp012 `category_2d_replay50` epoch 100 | 0.331605 | 0.748031 | 285/381 | Category specialist from Exp009 `baseline_cont100`; target category `2d`; 50 targeted and 50 natural-replay events per epoch with independent Run-2 schedule. |
| 17 | Exp007 original `ddp_bs4` epoch 100 | 0.330954 | 0.758530 | 289/381 | Original update-matched DDP global-batch-4 cached-native v123 final checkpoint: 10,000 optimizer updates, 40,000 consumed patch events. |
| 18 | Exp012 `category_1alldiffuse_replay25` epoch 100 | 0.330845 | 0.763780 | 291/381 | Diffuse-category specialist from Exp009 `baseline_cont100`; targets `1a` through `1f`; 75 targeted and 25 natural-replay events per epoch with Run-2 independent schedule. |
| 19 | Exp008 `v3_dualfusion_softguide_joint` epoch 80 | 0.330786 | 0.755906 | 288/381 | Same joint dual-fusion proposal/refinement method as rank 8, evaluated at the update-8,000 pause checkpoint. |
| 20 | Exp008 `v2_dualfusion_precision` epoch 80 | 0.330705 | 0.742782 | 283/381 | Same precision-pressure dual-branch method as rank 9, evaluated at the update-8,000 pause checkpoint. |

## Source Notes

- Exp007 phase-2 continuation runtime report:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/ddp_bs4_continue100_from_epoch100_report.md`
- Exp007 original synced report:
  `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/report.md`
- Exp008 synced report and execution spec:
  `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/report.md`,
  `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/codex_execution_spec.md`
- Exp009 runtime report and execution spec:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/reports/s3_attention_coupling_report.md`,
  `experiments/009_voxtell_s3_attention_coupling_ablation/codex_execution_spec.md`
- Exp012 combined report and execution spec:
  `experiments/012_voxtell_category_specialists_replay50_cont100/reports/exp012_r01_r02_combined.md`,
  `experiments/012_voxtell_category_specialists_replay50_cont100/codex_execution_spec.md`

## Caveats

- This is a checkpoint leaderboard, not a deduplicated experiment-family
  leaderboard.
- The Exp007 phase-2 continuation results are present in the runtime reports;
  the repo-local Exp007 `report.md` snapshot still describes the original DDP
  run.
- Exp008 lacks a same-source single-branch continuation control, so its
  improvements over epoch 0 cannot be attributed only to the dual-branch
  architecture.
- Exp012 category-specialist checkpoint selection was based on target-category
  metrics; the full-val200 rows here are diagnostic overall checkpoint scores.
