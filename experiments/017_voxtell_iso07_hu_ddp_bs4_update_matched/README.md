---
created: 2026-08-15
updated: 2026-08-16
status: phase2_continuation_ready
experiment_id: "017_voxtell_iso07_hu_ddp_bs4_update_matched"
---

# Experiment 017: VoxTell 0.7 mm HU DDP Batch4

This experiment tests the completed Exp016 cache under the Exp007 DDP batch4
finetuning recipe. The only intended method change from Exp007 is data
geometry and normalization: `0.7 mm` isotropic, crop-to-nonzero, clipped linear
HU `[-1024, 1024] / 1024`, and image padding `-1`.

## Ownership

- Canonical config:
  `configs/experiments/017_voxtell_iso07_hu_ddp_bs4_update_matched.json`
- Execution spec:
  `experiments/017_voxtell_iso07_hu_ddp_bs4_update_matched/codex_execution_spec.md`
- Runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched`
- Cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`

## Status

- Phase 1 completed: best fixed val200 checkpoint was epoch50 with Dice
  `0.3347`, hit rate `0.7769`, `296 / 381`.
- Phase 2 protocol is ready: continue from phase-1 epoch100 with weights-only
  initialization, fresh optimizer/scaler/LR/update state, and the second
  deterministic iso07 40,000-event schedule slice.
- Phase 2 full training should not start until static checks, schedule
  preflight, DDP resume smoke, one-case iso07 inference restore, and small
  val-shard smoke pass.

## Primary Comparison

Primary baseline is Exp007 DDP batch4 native z-score e5/d4 epoch100 val200:
Dice `0.3310`, hit rate `0.7585`, `289 / 381`.

Primary phase-2 baseline is Exp007 continuation from epoch100 with the same
weights-only fresh-optimizer protocol. Exp007 phase-2 best fixed val200 result
was relative epoch50 / absolute epoch150: Dice `0.3460`, hit rate `0.7769`,
`296 / 381`.
