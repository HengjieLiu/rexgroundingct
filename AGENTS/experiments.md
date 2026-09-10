---
created: 2026-07-22
updated: 2026-09-06
status: active
---

# Experiment Rules

## Current Experiment IDs

- `001_voxtell_v1_1_miccai200_val_eval`: pretrained VoxTell validation
  evaluation on the 200-case MICCAI validation split.
- `002_voxtell_text_ft_miccai_train_val`: text-conditioned VoxTell fine-tuning
  scaffold for MICCAI train and validation data.
- `003_voxtell_rex_ft_rescue_ablation`: four-arm VoxTell rescue fine-tuning
  ablation with fixed schedules and fixed val20/val200 probes.
- `004_voxtell_v123_native_vs_2mm_global_context_ft`: paired continuation from
  exp003 v123 epoch 100 comparing native VoxTell preprocessing with 2 mm
  global-context preprocessing.
- `005_voxtell_global_proposal_local_cascade`: two-stage global proposal to
  local VoxTell segmentation cascade.
- `006_voxtell_cached_native_v123_lr_ablation`: cached-equivalent native v123
  fine-tuning sanity check and learning-rate ablation.
- `007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched`: cached-native
  v123 e5/d4 DDP global-batch-4 run with pause/eval/resume val200 checkpoints.
- `008_voxtell_dual_branch_proposal_refinement_ablation`: four-arm
  proposal/refinement continuation testing shared versus branch-adapted fusion,
  precision pressure, and detached versus joint soft guidance.
- `009_voxtell_s3_attention_coupling_ablation`: S3 attention coupling ablation
  comparing revised half- and quarter-scale attention variants.
- `010_voxtell_public_anatomy_prior_fusion`: public anatomy-prior fusion and
  TotalSegmentator val200 anatomy-audit experiment line.
- `011_voxtell_v123_e4d4_ct_normalization_ablation`: three-arm public-VoxTell
  v123 e4/d4 ablation comparing native z-score, clipped z-score, and fixed
  linear HU preprocessing.
- `012_voxtell_category_specialists_replay50_cont100`: category-routed
  specialist continuation from the exp009 baseline with 50% targeted and 50%
  natural-replay updates per epoch.
- `013_voxtell_public_category_only_specialists`: category-only specialists
  trained from the public VoxTell v1.1 checkpoint with no natural replay.
- `014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched`:
  cached-native v123 e5/d4 DDP effective global-batch-16 run with
  pause/eval/resume val200 checkpoints.
- `015_voxtell_isotropic_resolution_audit`: native CT spacing, label geometry,
  and isotropic-storage audit used to select the 0.7 mm VoxTell cache path.
- `016_voxtell_iso07_hu_preprocessing`: preprocessing-only build and audit for
  the `crop_clip1024_linear_iso07_v1` 0.7 mm isotropic fixed-HU cache.
- `017_voxtell_iso07_hu_ddp_bs4_update_matched`: 0.7 mm fixed-HU VoxTell DDP
  batch4 finetuning and continuation protocol matched to exp007.
- `018_voxtell_category2d_nodule_audit`: audit-only, hash-pinned ranking of
  fixed-val200 single-model checkpoints for official category `2d` pulmonary
  nodules/masses; it does not launch training or inference.
- `019_voxtell_iso07_lung_bbox_coverage_audit`: audit-only comparison of
  TotalSegmentator whole-lung bbox coverage against every fixed-val200 GT
  finding after mapping both masks to the 0.7 mm iso07 geometry.
- `020_ct_rate_ts_total_rex_val200_audit`: CT-RATE `ts_total` mask acquisition
  and validation-cohort audit before any downstream use.
- `021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100`: 0.7 mm category-2d
  nodule-only positive specialist initialized from Exp017 epoch 100.
- `022_exp007_official_anatomy_val200_audit`: Exp007 official-anatomy val200
  benefit-harm audit using the CT-RATE `ts_total` source masks.
- `023_ct_rate_ts_total_rex_test300_audit`: CT-RATE `ts_total` mask preflight
  and test-cohort audit before any downstream use.
- `024_test_inference_anatomy_audit`: validation-gated test inference audit
  comparing anatomy-supported and baseline preprocessing paths.
- `025_iso07_best_anatomy_audit`: validation-gated anatomy audit and native
  test export for the strongest existing 0.7 mm-resampled checkpoint.
- `026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr`: public-start
  cached-native v123 DDP batch4 run with fixed differential learning rates and
  full val200 barriers every five epochs.
- `027_voxtell_2a_residual_refinement`: four data-source conditions for a frozen
  Exp007 2a residual refiner, with local Markdown/PNG monitoring. GPU timing
  requires explicit approval; full-run scheduling follows the user's timing
  review, and model ranking remains a user decision.

Do not rename experiment IDs casually. Runtime paths, configs, manifests, and
reports depend on them.

## Preprocessing Contract

Every substantial experiment spec must state whether preprocessing is:

- unchanged from the VoxTell baseline;
- cached-equivalent to the baseline; or
- intentionally changed as a method variable.

For VoxTell/ReXGroundingCT work, record normalization scope, resampling,
patch/window policy, cache ownership, orientation/export checks, and expected
data-pipeline bottlenecks. Use `docs/voxtell/preprocessing_variants.md` for the
standard cache IDs and required fields.

## Refinement Precision Default

Use FP32 for new refinement-model training and evaluation by default, with
mixed precision or TF32 requiring an explicit recorded experiment decision.
This default applies to refiners, not frozen base-model cache generation or
existing cache storage. Preserve those source contracts and all historical
experiment configurations. The user adopted this rule after Exp027's matched
FP16/FP32 benchmarks and approved its 10,000-update FP32 full run.

## Milestone Evaluation

Every long-running training spec must declare whether checkpoint evaluation is
a synchronous barrier or a concurrent sidecar. When training and evaluation
share GPUs, default to the established barrier workflow: save immutable full
state, stop trainers, evaluate all compared arms, wait for every required
summary, then resume. Record model, optimizer, scaler, LR horizon, update and
sample-schedule cursors, and Python/NumPy/PyTorch RNG streams in the resume
contract. Use a sidecar only with verified dedicated headroom and an explicit
non-interference rationale.

## Canonical Files

- Canonical configs live in `configs/experiments/`.
- Runtime config snapshots live under the `/mnt/shengdata1` experiment root.
- Repo-local summaries live under `experiments/`.

## Sync And Consistency

Use:

```bash
python scripts/rexgroundingct/sync_experiment_index.py
python scripts/rexgroundingct/check_experiment_consistency.py
```

The repo-local `experiments/` directory is an index, not the runtime output
store.
