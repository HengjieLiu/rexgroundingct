---
created: 2026-07-23
updated: 2026-07-25
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
- The shared native VoxTell cache `crop_zscore_native_v1` is complete under
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/` with
  3,192 train/val cases, 8,068 targets, and 0 empty targets.
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
- CT-specific HU normalization remains a later controlled ablation, not part of
  the first fine-tuning baseline.
- Experiment 006 resolved the native preprocessing throughput bottleneck with
  cached-equivalent preprocessing and improved fixed val200 performance over
  the previous exp004 native continuation. Treat `v123_cached_e5_d4` epoch 100
  as the current strongest validation candidate, pending submission packaging
  and test-set prediction work.
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
- Review experiment 006 `v123_cached_e5_d4` epoch 100 as the next candidate
  checkpoint for packaging and test-set prediction work.
