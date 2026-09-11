---
created: 2026-07-23
updated: 2026-09-09
status: active
---

# Current Status

This file is the compact status, decision, and todo loop for future humans and
agents. Update it when evidence changes the next action, not after every small
edit.

## Active Work

- **SideExp006 A1 threshold sweep complete:** all 200 val cases / 381 findings
  at 19 thresholds (0.05–0.95) produced 7,239 verified records. Saved-cache
  baseline reproduced exactly: Dice 0.34586800803778955, 295/381 hits. Best
  observed global Dice is 0.3459823617843387 at 0.60; category maxima vary.
  The executed notebook and HTML are ready under the SideExp006 external
  `runs/r001_a1_val200/` runtime. See
  [the report](../side_experiments/sideexp006_category_threshold_tuning/report.md).
  Fifteen tests and repository checks passed. Threshold adoption, instance
  metrics, and B1/D1/E1 sweeps remain deferred for review.

- **Test300 Wave 3 running under j005.** Wave 2 completed unchanged at
  2026-09-10T19:54:06Z; all ranks 5–8 have strict 300-case/582-prompt caches.
  The old j004 coordinator exited and remains retired. The automatic transition
  passed retained-logit parity and cold-source verification for all 16 benchmark
  pairs: reference 338.0 s versus streaming 226.9 s (33% shorter in this small
  benchmark; full-wave overlap remains unmeasured). Prelaunch regression and
  repository gates passed. Frozen j005 SHA:
  `6a155ab3c9908b6652f58ab4fc965b378f1964ad09a624fbd4a7e4ac880a438c`.
  Coordinator PID `765487` launched at 20:10:52Z; transition PID `604748`
  monitors it. At 20:34Z, ranks 9/10/11/12 had staged 10/10/9/10 cases,
  all four GPU workers had passed smoke gates, and no failures were reported.
  The single publisher waits for the first complete checkpoint; Waves 4–5
  follow staging barriers with publication overlap. Early remaining ETA is
  23–45 hours, combining workload-weighted live samples and Wave 2 priors;
  it is provisional. Minute monitoring and five-minute reports are active.
  See [j005 instructions](../side_experiments/sideexp003_ensemble_method_hub/cache_jobs/j005_test300_r09_r20_streaming_gpu8/README.md).
  Use `stream_test300.py watch --job j005_test300_r09_r20_streaming_gpu8 --once`.
  Do not resume j004. This supersedes its earlier automatic rollout below.


- **SideExp005 e-series complete:** r003 val200 verified 1,000 files;
  see `de_val200_report.md`. r004 test300 verified 1,500 files;
  see `e_test300_report.md`.
  Historical launch: 2026-09-10T15:21:25Z, supervisor PID `375011`.
  Original ranks 1–8 used four CPU workers and equal sigmoid-probability
  averaging. The r004 test stage followed the verified r003 validation report
  and eight strict published caches,
  regardless of validation scores. Wave 2 is checked immediately and every
  600 seconds. Later export waves continue independently. Outputs are
  Exp024 `outputs/e1`, `e2`, `e3`, `e11`, `e12`, without ZIPs or uploads.
  The e1 baseline must reproduce Dice 0.35234375344586844 and 291/381 HITs.
  Read `side_experiments/sideexp005_postprocessing_merge/e_series_execution_spec.md`;
  use `e_series.py watch --once` for progress. Final aggregate documents are
  `de_val200_report.md` and `e_test300_report.md`; existing d-only evidence is preserved.
  All 149 targeted tests and repository checks passed. Real preflight verified
  200 val / 300 test cases, numeric prompts, official anatomy and native headers.
  Launch evidence: `side_experiments/sideexp005_postprocessing_merge/e_series_launch.json`.
  Val job SHA `262ea94db0e8c19791b62926c2be25983949f173daf9f73d8de0786a83958c14`;
  test job SHA `5fee63d1859824ac8f52a6d9eec93ee62ecd6c11020f23f533fc753cf4892700`.

- **SideExp005 test d11/d12 complete:** 600 files verified; both variants cover 300 CTs / 582 prompts. ZIPs skipped by request.
  Historical launch policy: 2026-09-10T08:04:50Z.
  Separate run `r002_d1_test300_d11_d12_frozen_postprocessing`, supervisor PID
  `4025791`, waits for verified val200 report collection and successful worker
  exit, then generates both test300 variants regardless of validation scores.
  This user authorization supersedes the previous manual validation-review gate.
  Outputs are Exp024 `outputs/d11` and `outputs/d12`; no ZIPs or uploads.
  Preflight verified all 300 CTs / 582 prompts, native headers, d1 and official
  anatomy hashes. All 138 targeted tests and repository checks passed.
  The four CPU workers use one numerical thread each and
  `NUMPY_MADVISE_HUGEPAGE=0`, with 150 GiB headroom above the 20 TiB reserve.
  Original r001 runner/collector and Waves 2–5 remain independent. Use
  `side_experiments/sideexp005_postprocessing_merge/test300_postprocessing.py watch --once`.
  Job SHA: `c7b39164431a10288700e92484937cf0c30541bdaccb7e7c8f1a59ef6517af23`.
  Register changes to the separate test producer occur after the val collector
  finishes, preserving val results and manually recorded submission history.

- **SideExp003 Waves 2–5 continuation authorized and launched**, 2026-09-10T07:27:36Z.
  New source-bound job `j004_test300_r05_r20_cpu4_gpu8` runs original ranks
  5–20 on four GPUs, each feeding one spawned CPU worker (four total), with
  at most two pending crops per GPU and `NUMPY_MADVISE_HUGEPAGE=0` throughout.
  Supervisor PID `3974216`. It retains 20 TiB shared reserve, 64 GiB RAM
  headroom, and both GPU smoke gates. No unrelated CPU jobs are paused.
  All 98 SideExp003 tests and repository checks passed; the largest native
  benchmark crop matched the reference hash exactly through the new handoff.
  Original j003 remains closed after Wave 1; d1/d2/d3 remain complete.
  This user decision supersedes the earlier Waves 2–5 hold below. Use
  `continue_test300.py watch` and the j004 runtime, not the old coordinator.
  The requested 30-minute observation completed at 2026-09-10T07:59:08Z:
  63/1,200 Wave 2 outputs staged, all four workers healthy. The
  updated remaining Waves 2–5 estimate is 26–54 hours, including publication
  and validation; later-model and disk variability remain uncertain. The
  detached production coordinator continues. See j004 `launch_report.md`
  and `observation_summary.json` for all five-minute checkpoints.


- **SideExp005 postprocessing audit complete:**
  Verified all 1000 outputs / 1905 finding evaluations. d1 Dice 0.357504, 296/381 hits; d2 Dice 0.361598, 296/381 hits; d3 Dice 0.364030, 295/381 hits; d11 Dice 0.366020, 296/381 hits; d12 Dice 0.368820, 297/381 hits.
  Aggregate report: `side_experiments/sideexp005_postprocessing_merge/report.md`.
  the collaborator ZIP and
  source audit were moved unchanged from the former Exp027 folder to
  `side_experiments/sideexp005_postprocessing_merge/`, with before/after hashes.
  The independent CPU runner compares d1/d2/d3/d11/d12 on val200, first requiring
  reproduction of d1 Dice 0.3575041942161927 and 296/381 hits. d11 is frozen
  collaborator semantic-v1; d12 is d11 followed by strict semantic-v2. Four
  workers use NUMPY_MADVISE_HUGEPAGE=0. The detached supervisor launched at
  2026-09-10T06:59:11Z after full preflight, 13 SideExp005 tests, 93 SideExp003
  tests, six Exp024 tests, ten register tests and repository checks passed.
  Container: `sideexp005_r001_d1_val200_frozen_postprocessing`.
  The central register reserves d11/d12;
  test generation remains pending review. No canonical Exp027 registration or
  Waves 2–5 continuation belongs to this task. See the SideExp005 README/spec
  and external runtime state for launch and progress evidence.

- **p001 test300 profiling completed** at 2026-09-10T05:48:10Z
  (September 9, 10:48 PM PDT). All 16 model–case inference jobs and six 16-case
  CPU replay comparisons passed exact native hashes, geometry and storage masks.
  **93 SideExp003 tests passed** in the pinned image. The demonstrated slowdown
  is NumPy huge-page advice triggering kernel memory compaction: worker kernel
  CPU share was 87–93%, falling to 22–25% with process-local
  `NUMPY_MADVISE_HUGEPAGE=0`. Original CPU4/8/16 totals were
  14.62/14.44/12.08 min; fixed totals were **4.80/5.95/5.31 min**.
  Recommend the allocation setting and **four CPU workers**, with bounded
  continuous handoff for a future continuation. No production code or global
  kernel setting was changed. Waves 2–5 remain held. The 31–63 h remaining-wave
  envelope retains the measured unfixed GPU/text cost; the modified GPU path
  was not measured and needs retained smoke timing before adopting an ETA.
  See `side_experiments/sideexp003_ensemble_method_hub/profiling_jobs/p001_test300_pipeline16_gpu8/pipeline_report.md`,
  `completion_verification.json`, and `storage_inventory.json`.

- **s001 d1/d2/d3 is complete** at 2026-09-10T02:37:02Z (September 9,
  7:37 PM PDT). All three native prediction folders contain 300 cases /
  582 prompts, and all three ZIPs passed hash, CRC and exact-file-set checks.
  The 16-worker container exited successfully (code 0); continuation wall
  time was 1 h 40 min. Outputs are beside Exp024 a/b under `outputs/d1`,
  `d2`, `d3` and `d1.zip`, `d2.zip`, `d3.zip`. No upload was performed.
  Waves 2–5 remain held. Completion and verification records are in
  `side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/`.

- Sixteen-worker s001 continuation milestone (now complete). The handoff completed at
  2026-09-10T00:56:55Z (September 9 PDT), retaining both smoke cases and
  launching `sideexp003_s001_top4_test300_d123_w16` with 16 CPU cores.
  The first 16 parallel cases completed by 01:14:47Z; all 48 d1/d2/d3 file
  hashes were independently verified. At 01:15:02Z the rolling run had
  30/300 cases complete, 75/582 prompts, and about 360 GiB available RAM.
  At that milestone the runner continued; Waves 2–5 stay held. Evidence:
  `side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/continuation_16_first_batch.json`.
  The original submission container exited intentionally during handoff;
  current progress is shared s001 `state.json` and `continuation_16/state.json`.

- User authorized a 16-worker d1/d2/d3 continuation after retained smoke
  completion (2026-09-09 PDT). The old submission supervisor `3026396` is
  intentionally SIGSTOP-held so it cannot dispatch the four-worker main run;
  its remaining smoke worker continues. Tested handoff PID `3457864` waits for
  both smoke journals and all six file hashes, retires the old owner, then
  launches `sideexp003_s001_top4_test300_d123_w16` with 16 CPU cores. See
  `side_experiments/sideexp003_ensemble_method_hub/submission_16_execution_spec.md`
  and shared s001 `continuation_16/handoff_state.json`. Do not manually resume
  the old supervisor. Original numerical/export functions and outputs are
  retained; Waves 2–5 remain held. Three new continuation tests, including a
  full synthetic 300-case/582-prompt/three-ZIP run, and repository checks passed.

- Latest user decision, 2026-09-09T21:38Z: cancel the proposed Wave 2 overlap
  before implementation or launch. Keep Waves 2–5 held and wait for d1/d2/d3
  to finish. No Wave 2 workers, progress records or overlap sessions were
  created. The existing Wave 1 finish guard and automatic submission runner
  remain active.

- SideExp003 submission job `s001_top4_test300_d123` is armed on gpu8 at
  2026-09-09T19:12:07Z in detached CPU-only container
  `sideexp003_s001_top4_test300_d123` (supervisor PID `3026396`). It waits for
  the Wave 1 finish guard, coordinator exit and four strict test caches, then
  automatically writes d1/d2/d3 and three ZIPs beside Exp024 a/b outputs.
  d1 is the equally weighted sigmoid-probability ensemble; d2/d3 independently
  apply the existing 20 mm anatomy policies to d1. Ranks 5–20 remain held.
  All 300 anatomy identities and CT headers were frozen; new numerical,
  postprocessing, restart, ZIP and synthetic full-run checks passed alongside
  SideExp003 regressions. Outputs remain pending behind the dependency gate.
  Read `side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/launch_report.md`
  for exact launch/watch commands and `submission_test300_spec.md` for the
  contract. Do not edit frozen submission sources while its runner is live.

- Latest SideExp003 instruction, 2026-09-09: finish test300 Wave 1 only, then
  hold ranks 5–20 because export speed is unacceptable. Guard PID `2974265`
  has intentionally stopped scheduler PID `2103906`; all four Wave 1 Docker
  workers remain running and unpaused. At 18:34:42Z their counts were
  271/272/259/271 cases. The guard waits for all four strict publications and
  successful worker exits, terminates the scheduler, and saves the inventory
  and phase observations. Do not manually resume the scheduler or launch later
  waves. See `side_experiments/sideexp003_ensemble_method_hub/finish_test300_wave1_spec.md`.
  The val200 CPU search completed through K=16 at 15:53:05Z. Next investigation:
  compare test and val GPU compute, CT layout copying, hashes and I/O using
  matched workloads; preserve the finished caches and frozen scoring code.
- SideExp003 on `shenggpu8`, branch `gpu8`: test300 job
  `j003_top20_test300_fresh_gpu8` restarted under the separately recorded
  `resume_test300_no_pause.py` coordinator. The user accepts contention and
  requires the five-worker val200 ensemble search to remain uninterrupted.
  All 40 saved test case arrays passed hash/prompt/CT geometry validation;
  68 SideExp003 tests and repository checks passed. The frozen inference
  source, job/cache identities and 20 TiB shared reserve are preserved.
  Recovery session `no_pause_20260909T071438Z` passed all eight smoke checks
  and independent validation of one new output per GPU. At 07:42:42Z Wave 1
  was exporting at 11/12/12/13 cases, while the same unpaused CPU container
  continued K=13 (18/200). Complete test caches remain pending. See
  `side_experiments/sideexp003_ensemble_method_hub/test300_no_pause_execution_spec.md`
  and the job's `recovery_report.md` for launch and follow-up evidence.
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
