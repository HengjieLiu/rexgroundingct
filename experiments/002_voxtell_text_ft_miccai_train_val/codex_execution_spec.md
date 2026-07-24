---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "002_voxtell_text_ft_miccai_train_val"
---

# Experiment 002 Codex Execution Spec

## Objective

Establish a challenge-valid, text-conditioned VoxTell fine-tuning path on the
MICCAI train split and evaluate it against the validation split. The first real
baseline is a paper-aligned single-GPU run that preserves VoxTell's default
`192^3` patch size and z-score preprocessing.

## Canonical Decisions

- Baseline spec:
  `experiments/002_voxtell_text_ft_miccai_train_val/training_finetuning_spec.md`
- One-finding sampler fallback:
  `experiments/002_voxtell_text_ft_miccai_train_val/sampler_fallback_decision.md`
- Reproducibility policy:
  `experiments/002_voxtell_text_ft_miccai_train_val/reproducibility_policy.md`
- Canonical config:
  `configs/experiments/002_voxtell_text_ft_miccai_train_val.json`

The fallback decision is intentionally labeled as
`confusion_not_solved_but_moving_forward`: multi-finding cases use
`2` positive + `1` negative prompt, while one-finding cases use
`1` positive + `2` negatives with forced foreground anchoring on the sole
positive target. Training always has `3` prompt channels; inference uses only
the actual case prompts.

## Prior Evidence

- Experiment 001 corrected-orientation quick/global validation Dice per finding:
  `0.225228`.
- Experiment 001 fixed the ReX/VoxTell orientation mismatch at inference export.
- Experiment 002 must apply the inverse orientation transform to masks before
  training, because CTs are read through `NibabelIOWithReorient` but released GT
  masks are stored in raw evaluator layout.
- `docs/voxtell/normalization.md` recommends keeping VoxTell z-score
  normalization for the first fine-tuning baseline.
- Public `voxtell-finetune` is an nnU-Net encoder-transfer workflow and is not
  the primary free-text grounding path for this challenge.

## In Scope

- Precompute ReXGroundingCT train+val prompt embeddings.
- Probe the largest feasible single-GPU batch size with real
  forward/backward/optimizer steps.
- Run the first baseline on a single GPU, no DDP:
  `5` epochs, `100` optimizer updates per epoch.
- Save an inference-compatible fine-tuned VoxTell model directory.
- Optionally run a 20-case validation quick/global evaluation after training.
- Use the fixed val20 seed-20260723 probe for quick validation.
- Use materialized train patch schedules for batch/DDP comparisons.
- Use fixed LR with no learning-rate decay for the current runs; keep
  `--lr-schedule poly` available for later controlled comparisons.
- Keep runtime logs, checkpoints, predictions, and evaluator outputs under
  `/mnt/shengdata1`.

## Out of Scope For This Baseline

- HU/window input.
- Non-`192^3` patch-size fine-tuning.
- 4-GPU DDP training.
- Full nnU-Net augmentation parity.
- Test-set submission packaging.
- Changes to the upstream VoxTell submodule.

## Inputs And Paths

- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Segmentations: `/data/hengjie/datasets/rexgroundingct/segmentations`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Runtime experiment directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val`
- Repo-local index:
  `experiments/002_voxtell_text_ft_miccai_train_val`

## Execution

Inside the VoxTell Docker container, run:

```bash
bash /workspace/scripts/rexgroundingct/run_002_precompute_text_embeddings.sh
bash /workspace/scripts/rexgroundingct/run_002_single_gpu_zscore192_baseline.sh
```

The baseline launcher snapshots the canonical config, polls CT readiness, runs a
single-GPU batch-size probe, trains for `500` optimizer updates, and writes time
and peak-memory reports.

## Success Criteria

- Batch probe writes `reports/batch_probe.json` and `reports/batch_probe.md`.
- Training writes `reports/training_metrics.json` and
  `reports/training_report.md`.
- Final checkpoint exists under the run directory.
- Inference-compatible model exists under `model/`.
- If quick validation is enabled, validation predictions and quick/global eval
  reports are written under the same run directory.
- No heavyweight runtime artifacts are tracked in Git.

## Verification

Host-side:

```bash
python -m py_compile scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  scripts/rexgroundingct/precompute_text_embeddings.py
python scripts/rexgroundingct/check_repo_workflow.py
```

Docker-side, when dependencies are needed:

```bash
bash /workspace/scripts/rexgroundingct/run_002_single_gpu_zscore192_baseline.sh
```

## Closeout Plan

After a runtime result exists:

1. Inspect the batch probe and training reports.
2. Decide whether the largest single-GPU batch size makes 4-GPU DDP worthwhile.
3. Run `python scripts/rexgroundingct/sync_experiment_index.py`.
4. Update `docs/current_status.md` if the next action changes.
5. Keep heavyweight runtime artifacts out of Git.
