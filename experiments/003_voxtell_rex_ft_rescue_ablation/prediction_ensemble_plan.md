---
created: 2026-07-23
updated: 2026-07-23
status: draft
experiment_id: "003_voxtell_rex_ft_rescue_ablation"
---

# Exp003 Prediction Ensemble Plan

## Purpose

Evaluate whether ensembling the four exp003 fine-tuned VoxTell variants improves
ReXGroundingCT validation performance and provides a stronger candidate for
challenge prediction packaging.

This plan is intentionally prediction-level. Do not average model weights across
variants, because the variants differ in loss behavior, positive-crop sampling,
and deep-supervision weighting.

## Current Candidate Pool

Runtime group:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z`

Variants:

- `v1_opt`
- `v12_opt_poscrop`
- `v123_opt_poscrop_emptyloss`
- `v1234_opt_poscrop_emptyloss_ds`

As of this plan, scheduled checkpoints and fixed val20 evals exist through
epoch 80 for all four variants. Epoch 100 checkpoints, val20 evals, and val200
evals are still pending.

## Standard Segmentation Ensemble

For a regular segmentation challenge, the preferred ensemble is:

1. Run each model and save soft foreground probabilities.
2. Average probabilities voxelwise across models.
3. Apply one final threshold after averaging.
4. Apply any postprocessing once to the ensembled output.
5. Tune threshold and optional model weights on validation data.

Hard-mask voting is a fallback when only binary predictions exist. It is faster
but less ideal because each model has already discarded uncertainty at its own
threshold.

## ReXGroundingCT Adaptation

The challenge output is prompt-conditioned and finding-indexed, so ensemble must
preserve case and finding alignment exactly:

- Ensemble per case and per finding channel, never across findings.
- Preserve the 4D NIfTI layout `finding x X x Y x Z`.
- Use the fixed dataset JSON to guarantee identical case and prompt order.
- Ensemble after orientation-aware export, because predictions are already in
  evaluator-compatible layout.
- Preserve affine/header/shape from a reference prediction or ground truth.
- Validate prediction counts, channel counts, and filenames before scoring.

The current VoxTell inference path exports hard masks after `sigmoid > 0.5`.
A proper probability ensemble therefore needs a small inference extension to
save pre-threshold sigmoid probabilities.

## Planned Evaluation Sequence

1. Quick hard-vote screen on fixed val20.
   - Use existing binary predictions from epoch 80.
   - Test 4-model majority vote and 2-model/3-model subsets motivated by val20.
   - Treat this only as a complementarity check, not final model selection.

2. Probability export implementation.
   - Add an inference mode that writes float probability maps before threshold.
   - Keep the existing hard-mask path unchanged for current eval compatibility.
   - Store probability outputs outside Git under the exp003 runtime tree.

3. Probability ensemble on fixed val20.
   - Average probabilities for the four epoch-80 models.
   - Compare uniform averaging, simple subset averaging, and threshold sweep.
   - Prefer simple thresholds unless val200 shows a strong reason otherwise.

4. Probability ensemble on fixed val200.
   - Repeat for epoch 100 after all models finish.
   - Select final candidate using val200, not val20 alone.
   - Candidate choices should include best single model, best hard-vote ensemble,
     best uniform probability ensemble, and any clearly justified subset.

5. Submission-facing decision.
   - Use the final selected recipe to regenerate predictions for the intended
     split.
   - Record exact variants, checkpoints, thresholds, dataset JSON, command, and
     output paths before packaging.

## Defaults And Cautions

- Default first serious ensemble: uniform probability average of the four epoch
  100 checkpoints, selected by fixed val200.
- Default quick sanity check: hard-vote epoch 80 val20 using existing binary
  predictions.
- Avoid overfitting to fixed val20; it has only 20 cases and 31 findings.
- Do not mix pretrained and fine-tuned checkpoints in the first ensemble unless
  val200 later shows complementary behavior.
- Do not use `checkpoint_latest.pth` for documented ensemble selection; use
  named scheduled checkpoints such as `checkpoint_update_008000.pth` or
  `checkpoint_update_010000.pth`.

## Epoch 80 Val20 Probability Ensemble Result

Executed on 2026-07-23 as a sidecar Docker run, without touching the active
training or intermediate-eval containers.

Runtime output root:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/ensembles/epoch080_val20`

Implementation:

- Four epoch-80 model directories were evaluated in probability mode.
- Probabilities were averaged uniformly with weight `0.25` per model.
- Thresholds tested:
  `0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90`.
- Fixed validation JSON:
  `configs/evaluation/rexgroundingct_val20_seed20260723.json`.
- Summary artifact:
  `threshold_sweep_summary.md` under the runtime output root.

Val20 results:

| Threshold | Dice/Finding | Hit Rate | Hits/Targets |
| --- | ---: | ---: | ---: |
| 0.10 | 0.342764 | 0.709677 | 22/31 |
| 0.20 | 0.349273 | 0.709677 | 22/31 |
| 0.30 | 0.352684 | 0.709677 | 22/31 |
| 0.35 | 0.353955 | 0.677419 | 21/31 |
| 0.40 | 0.354974 | 0.677419 | 21/31 |
| 0.45 | 0.355469 | 0.677419 | 21/31 |
| 0.50 | 0.348037 | 0.612903 | 19/31 |
| 0.55 | 0.343190 | 0.580645 | 18/31 |
| 0.60 | 0.342310 | 0.580645 | 18/31 |
| 0.65 | 0.340978 | 0.580645 | 18/31 |
| 0.70 | 0.339086 | 0.580645 | 18/31 |
| 0.80 | 0.332339 | 0.580645 | 18/31 |
| 0.90 | 0.308931 | 0.580645 | 18/31 |

Best Dice/Finding was threshold `0.45`: Dice `0.355469`, hit rate `0.677419`
(`21/31`). Compared with the best single epoch-80 model on val20
(`v123_opt_poscrop_emptyloss`, Dice `0.347774`, hit rate `0.677419`), this is a
small Dice gain without hit-rate gain. Lower thresholds `0.20` and `0.30`
increase hit rate to `22/31` while still improving Dice over the best single
model.

## Epoch 100 Val200 Probability Ensemble Poller

Started on 2026-07-24 as detached container
`rex003_epoch100_val200_prob_ensemble_poller_20260724T013414Z`.

The poller checks every 5 minutes and waits for the canonical
`v1_opt/full_100ep/eval_epoch100_val200` summary to complete before launching
the four-model epoch-100 val200 probability ensemble. It does not touch the
active v1 eval poller.

Runtime output root:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/ensembles/epoch100_val200`

Runtime log:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/logs/epoch100_val200_probability_ensemble_poller_exp003_full_20260723T075256Z_20260724T013414Z.log`
