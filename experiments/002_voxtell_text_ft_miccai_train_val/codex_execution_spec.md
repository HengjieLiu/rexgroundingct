---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "002_voxtell_text_ft_miccai_train_val"
---

# Experiment 002 Codex Execution Spec

## Objective

Establish a challenge-valid text-conditioned VoxTell fine-tuning path on the
MICCAI train split and evaluate it against the validation split. The first
baseline should isolate ReXGroundingCT fine-tuning while preserving VoxTell's
default preprocessing.

## Prior Evidence

- Experiment 001 corrected-orientation quick/global validation Dice per finding:
  `0.225228`.
- `docs/voxtell/preprocessing.md` documents the corrected orientation export
  and VoxTell direct inference path.
- `docs/voxtell/normalization.md` recommends keeping VoxTell z-score
  normalization for the first fine-tuning baseline.
- Public `voxtell-finetune` is an nnU-Net encoder-transfer workflow and is not
  the primary free-text grounding path for this challenge.

## Scope

In scope:

- Precompute ReXGroundingCT prompt embeddings when available.
- Run the 4-GPU text-conditioned smoke loop after train+val CT readiness passes.
- Run the longer fine-tuning loop only after the smoke run succeeds.
- Keep runtime logs, checkpoints, predictions, and evaluator outputs under
  `/mnt/shengdata1`.

Out of scope:

- CT-specific HU/window normalization ablations.
- Script-to-package refactors.
- Test-set submission packaging.
- Changes to the upstream VoxTell submodule.

## Inputs And Paths

- Canonical config: `configs/experiments/002_voxtell_text_ft_miccai_train_val.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Segmentations: `/data/hengjie/datasets/rexgroundingct/segmentations`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Runtime experiment directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val`
- Repo-local index:
  `experiments/002_voxtell_text_ft_miccai_train_val`

## Method

Inside the VoxTell Docker container, run:

```bash
bash /workspace/scripts/rexgroundingct/run_002_precompute_text_embeddings.sh
bash /workspace/scripts/rexgroundingct/run_002_voxtell_text_ft_smoke.sh
MAX_ITERATIONS=10000 CHECKPOINT_EVERY=500 \
  bash /workspace/scripts/rexgroundingct/run_002_voxtell_text_ft_full.sh
```

The launchers snapshot the canonical config, poll CT readiness, and pass
`--allow-experimental-train` explicitly.

## Smoke Gate

The smoke run must complete on `train + val` readiness with:

- no missing CT or segmentation errors;
- a runtime `run_manifest.json`;
- a smoke log under runtime `logs/`;
- smoke metrics under runtime `reports/`;
- no tracked runtime artifacts in Git.

## Success Criteria

- Full run completes and writes a final checkpoint under runtime `checkpoints/`.
- Validation evaluation produces a report and metrics JSON under runtime
  `reports/` and `eval/`.
- `sync_experiment_index.py` copies small summaries into the repo-local
  experiment index.
- `check_experiment_consistency.py` passes.

## Verification

```bash
python scripts/rexgroundingct/check_repo_workflow.py
```

If Docker-only dependencies are needed, run the experiment commands inside
`rexgroundingct-voxtell:cu126` and run repo checks from the host afterward.

## Closeout Plan

After smoke or full results:

1. Run `python scripts/rexgroundingct/sync_experiment_index.py`.
2. Inspect `experiments/002_voxtell_text_ft_miccai_train_val/metrics_summary.json`.
3. Add or sync a small report when runtime results exist.
4. Update `docs/current_status.md` if the next action changes.
5. Keep heavyweight runtime artifacts out of Git.
