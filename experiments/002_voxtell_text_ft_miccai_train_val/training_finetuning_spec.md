---
created: 2026-07-23
updated: 2026-07-23
status: active
experiment_id: "002_voxtell_text_ft_miccai_train_val"
baseline: "single_gpu_zscore_192"
---

# Experiment 002 Training and Fine-Tuning Specification

This is the human-readable specification for the first real VoxTell v1.1
fine-tuning baseline on ReXGroundingCT.

## Baseline Definition

Experiment 002 is a conservative text-conditioned fine-tuning baseline:

- Base checkpoint: VoxTell `voxtell_v1.1`.
- Input normalization: VoxTell default crop-to-nonzero plus per-volume z-score.
- Patch size: `192 x 192 x 192`.
- Training sample shape: one CT patch and `3` prompt-conditioned target channels.
- First run: single GPU, no DDP.
- Epoch definition: `100` optimizer updates per epoch.
- First training duration: `5` epochs, therefore `500` optimizer updates.
- Reproducibility seed: `20260723`.

An optimizer update is not the same as one sampled patch. With batch size `B`,
one epoch contains `100 * B` case-patches and `300 * B` prompt-target channels on
a single GPU. With gradient accumulation `G`, one epoch contains
`100 * B * G` case-patches. DDP is not used for the first baseline; if DDP is
used later with `W` GPUs, the per-epoch patch count becomes `100 * B * G * W`.

## Paper-Alignment Table

| Setting | Experiment 002 baseline | VoxTell paper/pretraining setting | Same? |
| --- | --- | --- | --- |
| Base model | Initialize from VoxTell `voxtell_v1.1` | VoxTell pretrained checkpoint | Same starting checkpoint |
| Task head | Keep text-conditioned VoxTell decoder | Text-conditioned VoxTell decoder | Same |
| Patch size | `192^3` | `192^3` | Same |
| Image preprocessing | nnU-Net `NibabelIOWithReorient`, crop-to-nonzero, z-score | VoxTell predictor crop-to-nonzero and z-score | Same intent |
| HU/window input | Not used | Not used for public VoxTell predictor | Same |
| Orientation | GT masks converted from raw evaluator `(F,X,Y,Z)` into VoxTell/nnU-Net `(F,Z,Y,X)` before training | Paper not ReX-specific | ReX-specific fix |
| Prompts per sample | Always `3` prompt slots | `2` positive + `1` negative | Same for multi-finding cases only |
| One-finding fallback | `1` positive + `2` negatives | Not clear from paper/code | Different/unknown |
| Foreground oversampling | Probability `0.85` for multi-finding cases; one-finding fallback forces foreground anchoring | Foreground structures oversampled `85%` | Same intent; fallback differs |
| Patch positivity guarantee | Anchor positive non-empty for foreground-oversampled patches; one-finding fallback forced non-empty when mask exists; other positives not guaranteed | Not fully specified | Same intent, exact behavior unknown |
| Negative targets | Empty masks | Negative prompt target absent/empty | Same intent |
| Loss | BCEWithLogits + soft Dice, deep supervision | Dice + BCE, deep supervision | Same intent |
| Deep supervision weights | Normalized geometric weights from `[1, 1/2, 1/4, 1/8, 1/16]` | Reported weights `[1, 1/2, 1/4, 1/8, 1/16]` | Same ratios; normalization detail differs/unknown |
| Optimizer | SGD, LR `1e-4`, momentum `0.99`, Nesterov, weight decay `3e-5` | SGD LR `1e-4`; nnU-Net-style details not fully explicit | Same LR; extra optimizer details inferred from nnU-Net |
| LR schedule | Fixed LR, no learning-rate decay; `poly` remains selectable | Poly LR | Different current choice |
| Augmentation | Not implemented in this lightweight trainer | Standard nnU-Net augmentation, no left-right mirroring | Different |
| Epoch length | `100` optimizer updates | `250` iterations per epoch | Different by design |
| Training length | First run `5` epochs / `500` updates | `2000` epochs | Different by design |
| Text embeddings | Precomputed Qwen3-Embedding-4B vectors loaded from `.npz` | Frozen Qwen3 embeddings | Same intent |
| Distributed training | First baseline single GPU/no DDP | Paper training hardware not reproduced here | Different |

## Reproducibility

Use `reproducibility_policy.md` for all comparable experiment 002 runs. In
short:

- fixed seed: `20260723`;
- fixed quick validation probe:
  `configs/evaluation/rexgroundingct_val20_seed20260723.json`;
- materialized JSONL train schedules for batch-size or DDP comparisons.

## Learning-Rate Schedule

Current runs use fixed LR with no learning-rate decay:

```text
--lr 1e-4 --lr-schedule fixed
```

This is different from the VoxTell paper's poly LR schedule. The trainer keeps
selection flexible with `--lr-schedule fixed` or `--lr-schedule poly`. Here,
"no decay" means no learning-rate decay; optimizer weight decay remains the
configured `3e-5` unless changed explicitly.

## Batch-Size Probe

Before the first baseline train run, run the batch-size probe on one GPU. The
probe performs real forward/backward/optimizer steps on real sampled patches and
records:

- largest successful batch size;
- elapsed time per update;
- peak allocated CUDA memory;
- peak reserved CUDA memory;
- sample-composition counters.

The training script writes these files under the selected run directory:

- `reports/batch_probe.json`
- `reports/batch_probe.md`

The first baseline launcher reports both the largest successful batch size and a
recommended batch size. By default it trains with the recommended batch size,
chosen as the fastest observed seconds per case-patch, unless `BATCH_SIZE` is
set explicitly. Set `BATCH_SELECTION=largest` to force the largest fitting
batch.

## DDP Decision Rule

The first run is intentionally single GPU. DDP should be considered only after
the single-GPU probe gives a memory and throughput baseline.

For this experiment definition, `100` steps per epoch means `100` optimizer
updates. If DDP is later used across `4` GPUs without changing
`steps_per_epoch`, each epoch sees about `4x` as many case-patches as the
single-GPU run at the same per-GPU batch size. That is acceptable only if we
intend the epoch to mean optimizer updates rather than a fixed number of
patches. To compare strictly by sampled patches, reduce `steps_per_epoch`
proportionally.

## Future Ablations

Future patch-size and HU-input runs should be separate experiments or named
subruns, not silent changes to this baseline. Directly changing patch size is
not only a dataloader change: VoxTell has positional embeddings tied to the
feature grid, and inference must be patched consistently. Directly switching
from z-score to HU/windowed input is also an input-distribution shift and should
be fine-tuned before inference is trusted.

For ensemble work, use the same exported model directory format for each
subrun:

- `model/plans.json`
- `model/fold_0/checkpoint_final.pth`

Inference should then run each fine-tuned model with the actual case prompts and
ensemble their probability/logit outputs before thresholding.
