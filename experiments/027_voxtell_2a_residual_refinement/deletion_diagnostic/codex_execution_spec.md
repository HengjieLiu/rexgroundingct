# Deletion diagnostic execution specification

Date: 2026-09-10. Status: completed; results pending user review.

The user authorized implementation and GPU execution. The200-update diagnostic
and all eight full-volume evaluations completed at 16:31:03 UTC; see
[results.md](results.md) for outcomes, caveats, verification and next decisions.

## Objective and scope

Test whether an explicit removal head can fit annotated FP versus TP on a
small fixed real-patch set, then retain useful behavior on the same complete
volumes. Preserve the stopped Exp027 run, caches, benchmarks and source files.
No FN addition, architecture search, long four-arm launch or checkpoint ranking.

## Inputs and preprocessing

Use the canonical Exp027 config and `full_fp32_100ep/prepared.json`; validate
their source hashes and the selected cache arrays using existing verification.
Reuse frozen Exp007 cont e050 / absolute e150, SHA a499ad1c..., strict 2a
validation logits. No VoxTell inference or cache regeneration. Keep native
cropped z-score CT, original finding IDs/prompts, no resampling, cached clipping
and dtype provenance. Refiner inputs and arithmetic are FP32, TF32 disabled.

Select eight different half-A patients deterministically across baseline-Dice
strata among findings with both base TP and FP, using only frozen-base metrics.
Save keys, prompts, patient IDs, counts and selection rationale before training.
For each finding select three distinct base-positive tiles from the established
192³ / 50%-overlap grid: a GT-empty FP tile when available (otherwise most FP),
most TP, then strongest remaining mixed FP/TP tile (otherwise most FP).
Save all tile counts and coordinates. No augmentation. Cycle a seeded shuffled
24-patch order for exactly 200 updates; every patch receives 8 or 9 updates.

## Model, loss and operating points

Use the existing 16-channel GroupNorm/GELU backbone, dilations 1/2/4/1, fresh
initialization. Replace residual meaning with a removal logit. Zero head
weights and bias logit(0.05) initially preserve the binary base at threshold
0.5. Train all parameters. Only valid base-positive voxels receive supervision.

Loss = mean_FP BCEWithLogits(remove,1) + 2 × mean_TP BCEWithLogits(remove,0).
Each term is separately normalized; an absent group contributes zero. These
weights are a diagnostic implementation decision, not future-run defaults.
No Dice, L1, additions, intensity augmentation or spatial flips. AdamW LR1e-4,
weight decay1e-4, betas0.9/0.999, eps1e-8, clip norm1.0, batch1, no AMP or TF32.

Record fixed removal threshold 0.5 and the full predeclared threshold curve.
At update200 additionally determine a threshold retaining at least99% of TP
on the fitting patches, then freeze it for full-volume use. Fit a simple
base-logit suppression threshold to the same patch TP-retention target.
These are fitting-data operating points, not calibrated probabilities.

## Execution and artifacts

Canonical config: `configs/experiments/027_deletion_diagnostic.json`.
Runtime: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/deletion_diagnostic_v1`.
One authorized GPU worker; preparation and CPU tests have GPUs disabled.
Save source/config/environment hashes, fixed schedule, update journals and
full-state checkpoints at 0/50/100/150/200. Evaluate all fixed patches at each
checkpoint synchronously, then run full-volume evaluation at 200 only. Runtime
ownership lock prevents concurrent writers; existing completed artifacts are
not overwritten. Save failure/interruption checkpoint and stop on OOM,
non-finite values, integrity failure or mismatched geometry. No silent method
changes or automatic restart. A new run needs a separate runtime directory.

Inference skips tiles without valid base-positive voxels. Blend removal
probabilities in FP32 with Gaussian weights over the full grid, inactive tiles
contributing zero action with their weights retained. Threshold once after
blending; final mask is strictly a subset of base. Restore masks to original
geometry and verify frozen-base Dice/counts before accepting each result.

Write append-only training JSONL, per-patch and per-finding JSON/CSV, threshold
curves, timing, checkpoint hashes, diagnostic slice panels, and an atomically
refreshed local Markdown/PNG report. Publish phase/update/finding/tile progress.

## Smoke gate and interpretation

CPU PyTorch tests in the existing Docker image with GPUs disabled cover loss
group normalization/empty groups/padding, finite gradients and tiny synthetic
fitting, initial keep identity, subset guarantee, tiling/blending including
small volumes, metric arithmetic, operating-point ties and fixed schedules.
Run repository workflow and whitespace checks before GPU launch.

Evidence of useful fitting requires FP removal alongside high TP retention and
paired Dice benefit, with individual harms reported. Compare threshold-only
suppression at comparable retention; merely reducing foreground is insufficient.
Patch improvement that fails on complete training volumes is a context/sampling
failure, not evidence of generalization. All eight volumes are in-sample.
Report outcomes without automatically promoting a checkpoint or launching
the larger training runs. Completion status: pending_user_review.
