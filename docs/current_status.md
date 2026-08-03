---
created: 2026-07-23
updated: 2026-08-01
status: active
---

# Current Status

This file is the compact status, decision, and todo loop for future humans and
agents. Update it when evidence changes the next action, not after every small
edit.

## Active Work

- `001_voxtell_v1_1_miccai200_val_eval` is the corrected-orientation pretrained
  VoxTell validation baseline.
- `002_voxtell_text_ft_miccai_train_val` is the active challenge-valid
  text-conditioned fine-tuning scaffold.
- `003_voxtell_rex_ft_rescue_ablation` is implemented as the next fine-tuning
  rescue ablation: four single-GPU batch1 variants with fixed train schedules,
  fixed val20/val200 probes, smoke-tested optimizer/loss/sampler switches, and
  active run group `exp003_full_20260723T075256Z`.
- `006_voxtell_cached_native_v123_lr_ablation` completed the cached-native v123
  sanity check and learning-rate ablation for run group
  `exp006_cached_native_lr_20260725T050001Z`. All four arms finished training,
  recovery evaluation filled the planned val20/val200 summaries, and the
  strongest epoch-100 val200 arm is `v123_cached_e5_d4` with Dice `0.3241`,
  hit rate `0.7612`, and `290 / 381` hits.
- `007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched` has a phase-2
  DDP continuation active in detached container
  `rex007_cont100_from_ddp100_20260730T062051Z`, run group
  `exp007_cont100_from_ddp100_20260730T062051Z`. It initializes weights only
  from the original exp007 epoch-100 checkpoint, resets optimizer/scaler/LR
  schedule/update counter, and trains continuation events `40000..79999`.
- `008_voxtell_dual_branch_proposal_refinement_ablation` completed all four
  independent proposal/refinement arms in run group
  `exp008_dual_branch_20260726T182647Z`. The epoch-80 pause resumed from full
  optimizer and AMP state at update 8000, preserved the original poly LR
  horizon and deterministic event cursor, and finished update 10000 with LR
  zero. Fixed val20 and val200 evaluations are complete at epoch 100.
  `v1_sharedfusion_softguide` has the best val200 Dice (`0.3366`) with hit rate
  `0.7533` (`287 / 381`); `v3_dualfusion_softguide_joint` has the best hit rate
  (`0.7559`, `288 / 381`) with Dice `0.3345`.
- `010_voxtell_public_anatomy_prior_fusion` is in its public-anatomy data-audit
  phase. TotalSegmentator is pinned at `v2.16.0`; weights and outputs live
  under `/mnt/shengdata1/hengjie`. A corrected CUDA 12.6 smoke segmented the
  first fixed val200 case in 22.7 seconds with 1,269 MiB incremental GPU
  memory, exact source geometry, and 87 of 117 labels observed. Four val200
  workers and an automatic report/visualization finalizer are launched but
  gated until two consecutive checks show at least 24,000 MiB free per GPU.
- `011_voxtell_v123_e4d4_ct_normalization_ablation` completed the e4/d4
  normalization comparison in run group
  `exp011_ct_norm_e4d4_20260728T085103Z`. The paired e5/d4 rerun is active in
  `exp011_ct_norm_e5d4_20260728T225333Z`: native z-score, clipped z-score, and
  fixed linear HU all initialize from public VoxTell v1.1 and use the same
  v123 schedule. Only GPUs 0-2 are exposed. A host-side state machine stops all
  trainers at epochs `5/20/40/60/80/100`, completes the concurrent three-arm
  val20 and canonical report barrier, then resumes from full
  optimizer/scaler/RNG/sample-cursor state. Final val200 follows epoch-100
  val20.
- `012_voxtell_category_specialists_replay50_cont100` is active in detached
  run group `exp012_category_specialists_20260801T215219Z`. Four single-GPU
  arms target pooled 1a--1f, 2a, 2b, and 2c with exactly 50 targeted and 50
  shared natural-replay events per epoch. A synchronous state machine reports
  complete target censuses at epochs `0/5/20/40/60/80`, retains fixed-val80
  non-target diagnostics, runs val200 at epoch 100, and stops at
  `selection_ready` before any earlier winning checkpoint receives val200.
- `013_voxtell_public_category_only_specialists` is being prepared as the next
  public-start category-specialist test. Four single-GPU arms train from the
  public VoxTell v1.1 checkpoint using 100% targeted category-only events and
  no replay: 1a--1f, 2a--2h, 2b+2c, and 2d. The planned synchronous barriers
  evaluate target censuses at epochs `0/20/40/60/80`, run full val200 at epoch
  100, refresh a live report after every barrier, and stop at
  `selection_ready`.
- The standard CPU-only training-dynamics suite is active for Exp008, Exp009,
  and Exp011. Canonical figures live under each experiment's
  `reports/training_dynamics/` directory, with the shared gallery at
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/comparisons/training_dynamics/README.md`.
  The Exp011 refresher watches completed report barriers and excludes
  in-progress segment updates.
- Submission packaging is now documented at `docs/submission.md` but should not
  be used for final submission until a candidate checkpoint and test prediction
  set exist.

## Settled Decisions

- Keep experiment IDs stable once runtime paths, configs, manifests, or reports
  depend on them.
- Keep public VoxTell direct inference and the first fine-tuning baseline on
  VoxTell's default crop-to-nonzero plus per-volume z-score normalization.
- Experiment 002 comparable runs use seed `20260723`, the fixed
  `rexgroundingct_val20_seed20260723` validation probe, and materialized train
  schedules when batch size or DDP changes data order.
- Experiment 003 comparable runs use seed `20260723`, one 10000-event
  materialized schedule per variant, the fixed
  `rexgroundingct_val20_seed20260723` probe, and the fixed
  `rexgroundingct_val200_seed20260723` 200-case validation order.
- Future VoxTell experiments must explicitly classify preprocessing as
  unchanged, cached-equivalent, or intentionally changed. Standard cache IDs and
  required fields are tracked in `docs/voxtell/preprocessing_variants.md`.
- Training-loss comparisons use optimizer update as the x-axis and merge
  segmented metric files by `global_update`. Compare absolute loss only within
  experiments that share an objective; use the gallery convention documented
  in `docs/voxtell/training_dynamics.md`.
- The shared native VoxTell cache `crop_zscore_native_v1` is complete under
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/` with
  3,192 train/val cases, 8,068 targets, and 0 empty targets.
- Fixed CT NIfTIs are treated as already-materialized HU. Do not apply DICOM
  slope/intercept again. The validation-wide evidence and experiment 011
  interpretation are recorded in
  `docs/voxtell/ct_hu_normalization_analysis.md`.
- Experiment 002 current fine-tuning runs use fixed LR with no learning-rate
  decay; poly LR remains selectable as a later controlled option.
- Treat `experiments/` as a repo-local index. Heavy runtime outputs live under
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct`.
- Use execution specs before substantial implementation, long GPU runs, data
  movement, or submission packaging.
- Keep the current script layout. Do not refactor into an importable package
  until experiment 002 stabilizes.

## Blockers And Watch Items

- Experiment 002 uses a fixed small validation probe and schedule-backed
  comparison runs; full-run validation cadence is still a later decision.
- Experiment 003 full four-arm run is active in detached Docker container
  `rex003_rescue_ablation_full_20260723T075256Z`. The first failed launch
  exposed weak raw-bbox positive-crop scheduling; that run was quarantined under
  runtime `runs_failed/`, and the active run uses FZYX/crop-to-nonzero geometry
  plus materialized `patch_starts`.
- Experiment 003 intermediate validation sidecar
  `rex003_intermediate_eval_poller_20260723T173113Z` is polling the active run
  every 10 minutes for epoch 60/80/100 checkpoints and writes canonical val20
  and final val200 eval directories with `.eval.lock` guards.
- Experiment 003 epoch-80 val20 probability ensemble completed for the four
  fine-tuned variants. The best Dice threshold was `0.45` with Dice `0.355469`
  and hit rate `21/31`; lower thresholds `0.20` and `0.30` reached `22/31` hit
  rate with Dice above the best single epoch-80 model.
- Experiment 003 epoch-100 val200 probability ensemble poller is active in
  detached container `rex003_epoch100_val200_prob_ensemble_poller_20260724T013414Z`.
  It polls every 5 minutes and waits for `v1_opt` epoch-100 val200 eval to
  complete before launching the four-model probability ensemble under
  `ensembles/epoch100_val200`.
- Experiment 004 exposed a native preprocessing throughput bottleneck: the
  native arm repeatedly performs CT load, orientation, crop, z-score, and mask
  loading on demand, while cached variants avoid most of that CPU/I/O work.
  This does not invalidate exp004, but future native-control continuations
  should use a cached-equivalent native preprocessing cache when runtime
  comparison is not the scientific question.
- Experiment 011 is the controlled CT-normalization ablation. Its two new
  caches preserve native geometry and target hashes relative to
  `crop_zscore_native_v1`. The completed e4/d4 group remains available through
  `runs/latest_e4d4`; `runs/latest_e5d4` and `runs/latest` point to the active
  e5/d4 group. Cache, schedule, sample, one-update, and segmented-resume gates
  passed before e5/d4 training began.
- Experiment 006 resolved the native preprocessing throughput bottleneck with
  cached-equivalent preprocessing and improved fixed val200 performance over
  the previous exp004 native continuation. Treat `v123_cached_e5_d4` epoch 100
  as the current strongest validation candidate, pending submission packaging
  and test-set prediction work.
- Experiment 007 phase-2 continuation uses DDP world size 4, per-GPU batch 1
  and effective batch 4, with `1e-5` encoder LR and `1e-4` decoder LR. The
  planned barriers are relative epochs `25/50/75/100`, labeled as absolute
  exp007 epochs `125/150/175/200`, each followed by fixed val200 evaluation on
  all four GPUs before training resumes.
- Experiment 008 epoch-0 val20 preserved hit rate exactly at `23 / 31`.
  Independent mixed-precision sliding-window runs differed from the stored
  Exp006 Dice by at most `3.2e-5`, despite bitwise same-patch logit identity.
  The recorded aggregate equivalence gate therefore uses Dice tolerance
  `1e-4` while requiring exact hit counts. Inference window batches 2 and 4
  were slower than batch 1, so synchronous milestone evaluation uses batch 1.
- Experiment 008 has no same-source single-branch continuation control.
  Improvements over epoch 0 cannot be attributed entirely to the dual-branch
  architecture.
- Experiment 008 epoch-100 val200 improved Dice over the Exp006 starting model
  for all four arms, but no arm exceeded its `0.7612` hit rate. From epoch 80
  to 100, all four arms improved val200 Dice. Shared fusion is the Dice-first
  choice, while joint dual fusion is the better hit-preserving choice.
- The upstream TotalSegmentator 2.16.0 Docker image bundles a CUDA 13 PyTorch
  build and silently falls back to CPU on this host. Exp010 uses the pinned
  submodule installed with `--no-deps` into the validated CUDA 12.6 VoxTell
  base image and requires `run_report.device == "gpu"`.
- Full challenge submission workflow needs a candidate checkpoint, test CT
  readiness, prediction packaging, and official source refresh.
- Public challenge pages can change during the submission window; refresh
  `challenge_info/` before submission decisions.

## Next Actions

- Keep `experiments/002_voxtell_text_ft_miccai_train_val/codex_execution_spec.md`
  current before launching smoke or full fine-tuning.
- Monitor experiment 003 run group `exp003_full_20260723T075256Z` through
  5-epoch val20, then fresh 100-epoch training and scheduled val20/val200
  evaluations.
- Update the experiment 003 launcher so future scheduled validation is
  interleaved immediately after each checkpoint epoch is reached, instead of
  waiting for the 100-epoch training subprocess to finish before running
  epoch 20/40/60/80/100 evals.
- After new runtime results, run `sync_experiment_index.py`, run consistency
  checks, and promote conclusions into this file if the next action changes.
- Add reusable quick commands to `AGENTS/command.md` and workflow reflection or
  upgrade prompts under `AGENTS/workflow_reflect/` when a task pattern should
  be repeated later.
- Monitor experiment 007 phase-2 continuation through relative epoch 25, confirm
  the val200 barrier completes, then compare epoch 125 against original exp007
  epoch 100 and exp006 `v123_cached_e5_d4` epoch 100.
- Use the completed experiment 008 val200 results to decide whether the next
  challenge candidate should prioritize `v1_sharedfusion_softguide` for Dice
  or `v3_dualfusion_softguide_joint` for hit preservation. Retain the caveat
  that experiment 008 lacks a same-source single-branch continuation control.
- After the exp010 val200 anatomy inventory completes, inspect all-case QC and
  the label/target-overlap tables, then run the selected thoracic ROI subset
  at 1.5 mm on the fixed first 20 validation cases before generating a
  train/validation anatomy cache.
- Monitor experiment 011 e5/d4 milestone barriers through epoch 100, keep the
  training-dynamics refresher on completed reports only, then sync repo-local
  metrics and manifests after final val200 provenance is ready.
- Monitor experiment 012 through its live milestone report. At
  `selection_ready`, review the target-only recommendations and explicitly
  decide whether any selected pre-100 checkpoint should receive val200.
- Launch and monitor experiment 013 after its schedule, subset, smoke, and
  dry-run gates pass. Use the target-only curves to decide whether public-start
  specialization helps before building any routed ensemble.
