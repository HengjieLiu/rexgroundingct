---
created: 2026-09-06
updated: 2026-09-06
status: complete
experiment_id: "025_iso07_best_anatomy_audit"
---

# Codex execution specification

Exp025 audits the strongest existing iso07 checkpoint before test export:
Exp017 continuation relative epoch 50 / absolute epoch 150. It freezes the
checkpoint and iso07 cache hashes, reuses the outcome-blind prompt routes, and
keeps official anatomy masks in native CT space.

Validation uses unchanged, exact whole-lung clipping, prompt-eligible whole-
lung +20 mm, and prompt-supported side/lobe +20 mm. The baseline must match
the published `0.33752638623854136` Dice and `290/381` hits. Test export uses
unchanged, whole-lung +20 mm, and fine-region +20 mm only; test ground truth is
never read. Every output is restored against the original CT header and
checked for native `(F,H,W,D)` shape, affine, binary uint8 values, and official
filename census.

The reproducible coordinator is
`scripts/rexgroundingct/run_025_iso07_anatomy_audit.py`; the immutable
configuration is `configs/experiments/025_iso07_best_anatomy_audit.json`.
Runtime artifacts and the detailed report are external under the Exp025
runtime directory. Anatomy source status remains
`PASS_PENDING_MANUAL_VISUAL_REVIEW`.
