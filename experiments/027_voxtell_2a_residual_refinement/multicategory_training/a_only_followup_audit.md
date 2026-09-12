# Identifying the 2a A-only result and a matched category follow-up

Audit date: 2026-09-11. This is an evidence review and proposed follow-up,
not a launched experiment or a change to the current training lifecycle.

## Verified historical result

Two prior deletion-only models fit the same 35 validation-A findings and
improved mean Dice on the 34 gradient-held-out B findings. Both ran for 2000
updates (20 epochs of 100 updates). The original `run3_val_a` used `F + 2K`;
its reference was reproduced exactly as `bce_tp2` in the A-only loss ablation.
The later `dice_bce_tp2` used `D + 0.25(F + 2K)/3`.

| Recipe, final update 2000 | A Dice at 0.50 | B Dice at 0.50 | B change | B Dice at 0.90 |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 0.339416 | 0.334950 | 0 | 0.334950 |
| Original deletion BCE: F + 2K | 0.363095 | 0.340140 | +0.005190 | 0.335697 |
| Finding-aware Dice + TP2 auxiliary BCE | 0.369552 | 0.344549 | +0.009599 | 0.337133 |

The two final checkpoint hashes were reverified against their evaluation
summaries, and the B means were recomposed from all 34 per-finding records.
The Dice+TP2 checkpoint SHA256 is
`b8fa176b2280060d03eaf0292faf4083ab3569062c501caf8015132d7e25b460`.
Its summary SHA256 is
`9cea6c397617fbc7d83bed689db17c736e867aa3326bb5acc263b0558790a292`.
At threshold 0.50 it removed 23.01% of pooled B FP and 11.07% of pooled B TP;
21 findings improved, 13 worsened. This was one seed on development B, not an
independent final test or a robustness result. No threshold is selected here.

## Exact loss and difference from the current runs

For removal logit r and p = sigmoid(r):

- F = mean over valid base-FP voxels of softplus(-r).
- K = mean over valid base-TP voxels of softplus(r).
- a = sum of p over patch base-TP; b = sum of p over patch base-FP.
- D = 1 - [2(T0-a)+1e-6] / [T0+F0+G0-a-b+1e-6].
- L = D + 0.25(F + 2K)/3.

T0 and F0 are full-finding base TP and FP. G0 is all full-finding GT,
including base FN. D hypothetically deletes within the sampled patch and leaves
predictions elsewhere unchanged. Empty FP/TP groups contribute differentiable
zero without changing normalization. Padding is excluded. Hard removal
thresholds apply only during evaluation, never inside the loss.

The current 2b/2c/2d/2e models already use this exact loss and architecture.
They fit original training data. The proposed follow-up changes the fitting
source to validation A. It does not introduce a new loss. Through completed
update 1000, full-validation Dice at threshold 0.50 is
0.317402/0.399372/0.382052/0.405602, versus baselines
0.357030/0.415254/0.394980/0.410408. At 0.90 it is
0.355381/0.415267/0.389554/0.410211. The scheduled 50-epoch runs are still
in progress; these are interim results.

## Proposed matched pilot

Use one fresh Dice+TP2 deletion editor per category, all trained on A only.
Match the historical 2a pilot: 2000 updates, evaluation at 100/500/1000/2000,
common pristine weights, FP32/no TF32, native 192-cubed patches, AdamW LR1e-4,
clipping1.0, existing patch mixtures and full-volume deletion inference.
Reuse the completed cache, retain original patient A/B partition and all
validation findings. No new base inference is required. A-only training does
not use any original training findings, including the previously excluded
2d training finding from patient train_2936.

| Category | A fitting findings | B evaluation findings | A baseline Dice | B baseline Dice |
| --- | ---: | ---: | ---: | ---: |
| 2b | 25 | 24 | 0.360920 | 0.352978 |
| 2c | 31 | 29 | 0.416814 | 0.413587 |
| 2d | 61 | 71 | 0.393874 | 0.395930 |
| 2e | 8 | 3 | 0.404691 | 0.425653 |

Record A fitting and B development performance separately; full A+B becomes a
mixed-exposure diagnostic. Retain both 0.50/0.90 and all 201 thresholds without
choosing a B-optimized operating point. Publish validation only after the
individual category's full evaluation completes. Compare source conditions at
matched updates and thresholds. B should be prominent in interpreting this
follow-up; a full-only curve cannot distinguish A memorization from B benefit.
The 2e B sample is too small to establish stable generalization.

If A fitting improves and B also benefits, that supports investigating training
source/annotation mismatch. It does not establish incomplete training labels:
cohort differences and many more updates per finding are alternative causes.
If A improves while B does not, this points to fitting without demonstrated
transfer. If A itself cannot improve even across thresholds, inspect objective,
patch coverage, model capacity and implementation before scaling training.

[Historical A-only results](../deletion_loss_ablation/results.md).
[Current category training](README.md).
