---
created: 2026-07-23
updated: 2026-09-10
status: active
---

# Current Status

This file is the compact status, decision, and todo loop for future humans and
agents. Update it when evidence changes the next action, not after every small
edit.

## Active Work

- Exp027's user-authorized four-loss A-only comparison launched at
  2026-09-11 00:18:32 UTC (September10 17:18 Pacific) in container
  `rex027_deletion_loss_ablation_a_20ep`, ID prefix `7b70eda09f13`.
  Runtime: `027_voxtell_2a_residual_refinement/deletion_loss_ablation_a_20ep`.
  Canonical config: `configs/experiments/027_deletion_loss_ablation_a.json`;
  specification/live links/verification: Exp027 `deletion_loss_ablation/README.md`.
  Fixed loss order is F+2K; F+K; D+0.25(F+2K)/3; D+0.25(F+K)/2. All four fit
  the same35 A findings using the historical run3 schedule and pristine weights.
  D is finding-aware single-patch Dice with outside predictions held at base;
  all GT including base FNs enters its denominator. B's34 findings are excluded
  from gradients and reported separately as held-out development evidence.
  The accepted budget is2000 updates with100/500/1000/2000 evaluation barriers,
  FP32/noTF32, LR1e-4, clipping1.0, batch1 and192³. The live3×3 board shows both
  removal thresholds0.50/0.90, independent provisional A/B results and raw/smoothed
  losses; a separate CPU process produces201-threshold saved-score sweeps.
  Fifty-three CPU Docker tests and workflow/whitespace checks pass. Labeled
  synthetic dashboard, test logs and exact source snapshots are saved under
  runtime `verification/`. The launch begins with hash/geometry checks of all
  63 validation CT caches/69 findings, without VoxTell inference or train-pool
  preparation. Check runtime status for current phase. Completion audits8000
  updates/16 evaluations and historical BCE reproduction before pending_user_review;
  no automatic ranking or continuation. Earlier loss proposals are superseded
  by this accepted four-objective contract.

- The user clarified the deletion objective: prioritize mean Dice and allow TP
  loss when sufficient FP removal improves it. Proposed blanket 99% pooled / 95%
  per-finding retention gates are superseded; no replacement training was launched.
  Exp027 `deletion_four_arm/dice_first_tradeoff_audit.md` records the CPU audit.
  Uniform per-finding removal of 50% FP and 10% TP would increase mean Dice from
  0.337215 to 0.379632; allocation of the same pooled budgets can change that
  outcome substantially. At a retrospective 10% pooled TP-loss operating point,
  epoch-20 run 3 removes 25.94% FP and achieves full Dice 0.351788 (B 0.340140).
  Dense 0.005-step sweeps of all sixteen evaluations and base-threshold controls
  are saved under `deletion_four_arm_20ep/reports/dice_tradeoff_audit/`; original
  coarse results and score hashes reproduce. Threshold maxima are retrospective
  diagnostics, with ranking/selection pending user review. Revise the next loss
  comparison toward retained-mask mean Dice plus auxiliary deletion BCE, with
  preservation/hit changes tracked rather than automatically vetoed.

- The requested deletion-training rethink is recorded in Exp027
  `deletion_four_arm/v2_training_proposal.md`, status proposal pending user review.
  New CPU audits compare the late online loss against an analytic constant-score
  reference and apply A-derived preservation rules to saved epoch-20 scores.
  Evidence is under `deletion_four_arm_20ep/reports/v2_loss_audit/`. Proposed first
  experiment: four loss variants on identical A-fit/calibration data (current
  BCE, preservation-weight control, hard-TP protection, plus ranking), followed
  by separate sampling/context tests and eventual return to four data conditions.
  Coefficients, split and budgets remain proposals. No new training was launched.

- Exp027 four-arm deletion stage 1 completed at 2026-09-10 20:08:43 UTC and is
  `pending_user_review`. All arms have exactly 2,000 updates and all four planned
  evaluations; the container exited with code 0 and GPUs are idle. The completion
  audit verified 21 checkpoints per arm and checkpoint hashes/69-finding coverage
  for all sixteen evaluations. At the fixed 0.90 removal threshold, epoch-20 full
  Dice is 0.338747 / 0.337217 / 0.338133 / 0.340792 in run order, versus cached
  base 0.337215. FP removal is 0.5359% / 0.0005% / 0.4957% / 1.2379%. Run 4 is
  in-sample and loses one hit, with three findings below 95% TP retention and one
  below 80%. Results and all milestones are recorded in
  `experiments/027_voxtell_2a_residual_refinement/deletion_four_arm/results.md`.
  The user requested all saved thresholds; `deletion_four_arm/all_thresholds.md`
  now contains all four checkpoints and A/B/full tables, with 288 editor summary
  rows and 6,624 per-finding rows exported under runtime `reports/threshold_sweep/`.
  Base controls are identical across all sixteen evaluations and exported once.
  Per-finding recomposition and monotonic deletion/threshold-1 identity checks
  pass. Lower thresholds reveal more editing and TP loss; for example epoch-20
  run 3 at 0.50 has full Dice 0.351784, TP retention 89.94% and 44 findings below
  95% retention, versus Dice 0.341336 and 99.46% TP retention with zero such flags
  at 0.80. No threshold/checkpoint selection or continuation was performed.
  The requested frozen-base probability sweep (0.25–0.75, interval 0.05) is also
  complete: `base_probability_threshold_sweep.md` records all 11 A/B/full results,
  with 759 per-finding rows and the figure under runtime
  `analysis/base_probability_threshold_sweep/`. Full Dice rises from 0.332122 at
  0.25 to 0.339351 at 0.75, versus 0.337215 at 0.50. The 0.75 change is +0.002136;
  mean finding precision rises from 35.282% to 37.590% and recall falls from
  47.186% to 44.030%. All 63 cached logit/target hashes and exact per-finding
  baseline reproduction passed; all cached extents cover the original CT extent.
  The sweep used four CPU workers and no inference, taking 22 seconds before
  plotting. No threshold was adopted automatically.
  At the user's request, the sweep was extended through 0.95 in 0.05 steps.
  Full Dice at 0.80/0.85/0.90/0.95 is
  0.339387 / 0.339172 / 0.338372 / 0.335165. The observed full-cohort maximum is
  at 0.80, with gain +0.002172 over 0.50 and only +0.000036 over 0.75. All prior
  rows reproduced exactly; the original sweep and source are archived under
  `analysis/base_probability_threshold_sweep/range_025_075/`. Current exports
  contain 45 aggregate rows and 1,035 per-finding rows. All hashes, geometry,
  monotonicity and baseline checks passed; CPU measurement took 18.1 seconds.
  Threshold tradeoffs and any continuation remain for user review. Its canonical
  config is `configs/experiments/027_deletion_four_arm.json`; the live entry is
  `experiments/027_voxtell_2a_residual_refinement/deletion_four_arm/README.md`.
  Container `rex027_deletion_four_20ep` (ID prefix 154b1f78847e) launched on
  2026-09-10; its immutable launch manifest records 17:14:25 UTC. The separate
  runtime is `027_voxtell_2a_residual_refinement/deletion_four_arm_20ep`.
  CPU preparation passed complete coverage of 864 CT caches / 1,189 findings in
  705.0 seconds. One base-empty training finding is explicitly bypassed, leaving
  1,119 eligible training findings. All four GPU trainers started at 17:26:05 UTC;
  initial losses/gradients are finite and allocated memory is 11.97 GiB per arm.
  Actual schedules and all four pristine FP32 checkpoints were verified.
  All arms completed update 100 and began their first concurrent evaluation at
  17:30:21 UTC. Provisional per-finding results appeared on the live board by
  17:30:59 UTC; all evaluations and later milestones subsequently completed.
  Arms remain train / train+A 50:50 / A / A+B, with a 50/25/25 TP/FP/uniform-active
  tile sampler. All models start from fresh common weights and use the tested
  FP32 deletion head, TP loss weight 2, LR 1e-4 and clipping 1.0. Main removal
  threshold 0.90; record the fixed six-threshold grid and base-threshold controls.
  The independent CPU dashboard shows training, provisional A/B/full evaluation,
  TP-retention flags and FP removal. Twenty-two CPU tests, synthetic dashboard
  inspection, dry launcher, workflow and whitespace checks passed. Stage 1 runs
  2,000 updates per arm with barriers at 100/500/1000/2000 and then stops for user
  review. No epoch 21 continuation, FN addition or model ranking is automatic.

- Exp027 deletion-only diagnostic: the user authorized coding and running the
  bounded follow-up on 2026-09-10. See
  `experiments/027_voxtell_2a_residual_refinement/deletion_diagnostic/README.md`.
  It uses eight distinct A patients, 24 fixed 192³ patches (eight GT-empty FP
  patches), 200 updates, and the existing backbone with a remove/keep head.
  FP and TP loss terms have separate means and weights1/2. FP32, no TF32,
  AdamW LR1e-4 and clipping1.0 are preserved. Eleven CPU Docker tests pass;
  cache/patch preparation completed in51.1 seconds. The diagnostic completed
  at 16:31:03 UTC with all 200 updates and eight full-volume evaluations; B is
  excluded. The session took 429.5 seconds including evaluation/reporting, with
  peak allocated memory 11.93 GiB; all values remained finite. At the fixed 0.5
  threshold, full-volume Dice is 0.322993 versus base 0.327060, with only 71.15%
  TP retained. A threshold of 0.89595 calibrated on the fitting patches gives
  Dice 0.371671, 98.81% TP retention and 23.24% FP removal; six findings improve
  and two worsen. One small finding loses 64% of its TP despite the strong
  pooled retention, so per-finding protection remains unresolved. A retrospective
  matched 99% full-volume TP-retention comparison removes 21.95% FP with the
  editor versus 3.10% with simple base-threshold suppression. This demonstrates
  useful in-sample discrimination, not held-out performance. All post-run checks
  pass, all five checkpoints are retained, and GPUs are idle. The runtime is
  `027_voxtell_2a_residual_refinement/deletion_diagnostic_v1`. Larger runs are
  not authorized by this diagnostic; the previous four-arm experiment stays
  stopped. Results and next training decisions remain pending user review.

- `027_voxtell_2a_residual_refinement`: both 100-update precision benchmarks
  completed. The user adopted FP32 for refinement training/evaluation by default
  and authorized four concurrent full runs: 100 updates × 100 epochs, with
  synchronous 69-finding A/B/full evaluations every 10 epochs. The canonical
  config targets the separate `full_fp32_100ep` runtime. All 801 training CTs /
  1,120 findings and 63 validation CTs / 69 findings must be cached and verified
  before any trainer starts; the prior 196 caches are reused with an input-bound
  producer-compatibility proof. Initial weights are pristine and common, LR
  remains 1e-4 and clipping 1.0, and TF32 is disabled only for refiner workers.
  Implementation passes 36 CPU contract/controller tests and 16 CPU PyTorch
  tests with GPUs disabled. The independent live report writer publishes
  training updates within five seconds and provisional per-finding validation
  scores before peers finish, then replaces them with each arm's final result.
  See `full_run_verification.md` and the experiment README for the current board,
  launch manifest and retained benchmark comparison. Ranking and checkpoint
  selection remain pending user review. The authorized orchestration launched
  at 2026-09-10 06:50 UTC (host PID 2754829). Complete caching and verification
  finished at 08:32 UTC: all 864 CTs / 1,189 findings passed, with 668 new caches
  and 196 reused. Preparation took 6,132.993 seconds (102.2 minutes); dedicated
  cache files occupy 353,227,386,533 bytes, excluding referenced CT images.
  At the user's request, all four runs stopped at 15:27:08 UTC after completing
  every epoch-50 validation, with exactly 5,000 updates each and no update 5,001.
  Epoch-50 full Dice is 0.3330467 / 0.3262850 / 0.3184750 / 0.3375320 in fixed
  run order, versus cached base 0.3372154; hits are 59 / 58 / 58 / 59 of 69.
  Held-out B Dice for runs 1–3 is 0.3293772 / 0.3196950 / 0.3024115 versus base
  0.3349501. Run 4's results are in-sample fitting diagnostics. All results
  remain pending user review, with no ranking or checkpoint selection.
  `full_run_progress.md` records all completed A/B/full evaluations through
  epoch 50; the final dashboard shows `stopped_by_user`.
  The user's subsequent learning diagnosis is recorded in
  `experiments/027_voxtell_2a_residual_refinement/learning_diagnosis_e040.md`.
  A CPU-only audit recomposed all sixteen completed evaluations, replayed 128
  exact historical patches, and checked all four epoch-40 checkpoints. The
  network is updating in finite FP32. Sampled nonempty patches improve, while
  new foreground on empty patches incurs a very small loss penalty; harmful
  full-volume edits offset useful corrections. Run 3 also has a pronounced B
  deficit. A one-case CPU probe measured modest tile-context dependence.
  Prioritize a fixed real-patch fitting diagnostic and a controlled negative-
  voxel objective comparison, plus visual annotation review, before changing
  LR or architecture. These are proposals for user review; the experiment was
  evaluating epoch 50 during the audit. Detailed evidence is in the full runtime's
  `analysis/learning_diagnosis_e040/` directory.
  The user then requested stopping after all epoch-50 validations, before
  update 5,001. At 15:18:35 UTC only coordinator PID 2755098 was suspended,
  preventing further launches while all four evaluators and the CPU reporter
  continued. CPU finalizer PID 3134835 verified complete summaries, checkpoint
  hashes and exact update journals, then shut down the coordinator and refreshed
  the dashboard. All processes exited and GPUs 0–3 were verified idle. See
  `user_stop_after_val50.md` and runtime `stop_after_val50.json`. Original
  config/schedules and all artifacts are preserved; the remaining 100-epoch
  schedule is superseded, and no further training should start automatically.
  The subsequent base-positive-tile proposal is audited in
  `positive_tile_gate_audit.md`, with no model inference or training. On the
  exact validation grid the gate skips 3,424/5,722 tiles (59.8%), while active
  windows still cover 55.2% of pooled CT–finding voxels. All GT is reachable in
  68/69 findings; one 116-voxel finding becomes unreachable. Of active tiles,
  977/2,298 (42.5%) have no GT and must remain available for FP-removal training.
  Prioritize a fixed-checkpoint gated-inference comparison, matched eligible
  sampling and a controlled hard-negative objective. Gate-only inference at
  fixed weights cannot change outputs on base-positive voxels. These are audit
  recommendations; no new scheme or GPU run has been adopted or launched.
  The next discussion proposes explicit keep/remove/add decisions, recorded in
  `categorical_editing_proposal.md`: start with a voxelwise FP-removal head,
  supervise base FP removal and base TP preservation with separate normalized
  terms, and later add an FN head with explicit TN protection. A GT-assisted
  perfect-removal oracle raises full mean Dice from 0.337215 to 0.590656; this
  is theoretical headroom, not an achieved or predicted model result. Compare
  against simple base-threshold suppression and verify real-patch fitting
  before a long run. Architecture, action thresholds and loss weights remain
  proposed; the experiment remains stopped and no new GPU work has started.
  The read-only `data_loading_audit.md` found synchronous patch loading, a
  two-case metadata/mapping cache and redundant buffer copies. Mean loading
  fell from 0.975–1.468 s in the earlier FP16 trial to 0.080–0.086 s in the
  later FP32 trial on identical schedules, consistent with warmer NFS file
  pages or changed contention. Later loading accounted for only 7–8% of
  training wall time; speedup claims need a controlled same-precision replay.
  Bounded prefetch, metadata retention and predictable input access are
  proposed; loader optimizations have not been implemented.
- `022_exp007_official_anatomy_val200_audit` is complete for user review:
  200 CTs / 381 finding prompts, frozen Exp007 cont e050 / abs e150 baseline
  Dice `0.346023`, official Exp020 masks only, 32 fixed diagnostic policies.
  Exact whole-lung clipping drops Dice to `0.338113` (129 worsened findings);
  the highest-observed routed comparison reaches `0.357341` (3 worsened).
  Broad prompt-eligible lung support with a 20 mm margin reaches `0.350761`
  with no measured Dice regressions. These are exploratory validation results,
  not adopted methods. All baseline/geometry checks, 12,192 comparison checks,
  and 19 tests pass; 40 diagnostic panels were AI-inspected. Official-mask
  human visual QC remains pending. Review the report before choosing next work.
- `020_ct_rate_ts_total_rex_val200_audit` sealed the 200-case CT-RATE
  `ts_total` validation mask audit and is pending manual visual review. Its
  outputs remain external and are not used downstream until that review is
  accepted.
- `021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100` has an implemented
  0.7 mm category-2d specialist path and smoke-tested schedule tooling; a full
  training run has not been accepted as complete.
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
- `023_ct_rate_ts_total_rex_test300_audit` completed no-download preflight for
  the 300-case test cohort, including the pinned 300-mask manifest and source
  checks. Authentication and manifest approval remain pending; no downstream
  test inference uses these masks.
- `024_test_inference_anatomy_audit` completed its validation-gated six-output
  test inference audit. The aggregate is diagnostic only and makes no test-set
  performance claim; manual source review remains required before adoption.
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
- `014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched` is active as
  the DDP effective batch-size-16 follow-up to exp007 in run group
  `exp014_ddp_bs16_update_matched_20260804T051845Z`. It keeps the
  cached-native v123 e5/d4 recipe and public VoxTell v1.1 initialization, uses
  local batch `1`, DDP world size `4`, gradient accumulation `4`, and has
  completed val200 barriers through epoch 75. The epoch-75 val200 report shows
  Dice `0.3258`, hit rate `0.7559`; epoch 100 remains pending.
- `015_voxtell_isotropic_resolution_audit` completed the train/val/test native
  CT header and label-geometry audit for isotropic VoxTell planning. All 3,492
  CTs are present with materialized-HU headers, train/val labels are present,
  test labels are missing as expected, and released labels match CT shape while
  using CT headers rather than label affines as physical-spacing truth.
- `016_voxtell_iso07_hu_preprocessing` completed the
  `crop_clip1024_linear_iso07_v1` cache: crop-to-nonzero, clipped linear HU,
  0.7 mm isotropic trilinear image resampling, nearest-exact mask resampling,
  image padding `-1`, target padding `0`, 3,492 cases, 8,068 targets, and zero
  empty targets after fallback.
- `017_voxtell_iso07_hu_ddp_bs4_update_matched` finished phase 1 under the
  exp007 DDP batch4 recipe with the completed exp016 cache. The best phase-1
  fixed val200 checkpoint is epoch 50 with Dice `0.3347`, hit rate `0.7769`,
  and `296 / 381` hits; phase 2 is ready to continue from epoch 100 with
  weights-only initialization and a fresh optimizer/LR/update horizon.
- `018_voxtell_category2d_nodule_audit` is complete as a read-only,
  hash-pinned fixed-val200 audit for official category `2d` pulmonary
  nodules/masses (`132` findings). Its Dice-first reference is Exp007 phase-2
  continuation relative epoch 50 / absolute epoch 150 (`0.394936`, `115/132`)
  and its hit-rate guardrail is Exp017 0.7 mm phase-1 epoch 75 (`0.392786`,
  `118/132`). Neither direct 2d fine-tune, Exp012 `category_2d_replay50` nor
  Exp013 `category_2d_target100`, exceeded the Exp007 reference on its
  available fixed-val200 result.
- `019_voxtell_iso07_lung_bbox_coverage_audit` is complete for all 200 fixed
  validation cases and 381 findings. The TotalSegmentator lung-lobe union
  bbox contains 336/381 findings without expansion (`88.2%`); the required
  isotropic expansion has median `0`, 95th percentile `3` voxels (`2.1 mm`),
  and maximum `52` voxels (`36.4 mm`). The human-readable report is indexed
  under `experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md`.
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
- The shared 0.7 mm fixed-HU VoxTell cache `crop_clip1024_linear_iso07_v1` is
  complete under
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/` with
  3,492 train/val/test cases, 8,068 released train/val targets, and 0 empty
  targets. Use `spawn` multiprocessing with one torch/OpenMP thread per worker
  for this cache family.
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
- Experiment 014 should use DDP world size 4, per-GPU batch 1, gradient
  accumulation 4, and effective batch 16. DDP accumulation uses `no_sync` on
  non-final accumulation microsteps, and the launcher should resume from the
  highest valid stable checkpoint below the next target update so mid-segment
  kills do not force rollback to the previous val200 barrier.
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
- Monitor experiment 014 through epoch 100, then compare its sample-matched
  epoch-25 and update-matched epoch-100 val200 results against exp007 DDP bs4.
- Launch experiment 017 phase 2 only after the static checks, schedule-prefix
  check, DDP resume smoke, one-case iso07 inference restore, and four-case
  sharded eval smoke pass; compare the continuation against exp007 phase 2.
- Use Exp018's fixed category-2d audit as the baseline for any later nodule
  training proposal; do not launch a training matrix or GPU job under Exp018.
