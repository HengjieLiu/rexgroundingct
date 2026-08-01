---
created: 2026-07-22
updated: 2026-07-22
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
