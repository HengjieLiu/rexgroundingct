---
created: 2026-09-09
updated: 2026-09-09
status: verified_formulas_publishing_mapping_unconfirmed
---

# Dice aggregation: case means and finding means

The initial V++_b3 snapshot note called the difference between published overall
Dice and category-weighted Dice a discrepancy. That wording was too strong:
these values need not agree if overall Dice gives each CT equal weight while
category summaries give each finding equal weight. The user's case-wise
explanation is consistent with the evaluator, but the live publishing field
mapping has not been independently confirmed.

## Verified evaluator behavior

The locally available official dataset evaluator is
`/data/hengjie/datasets/rexgroundingct/rexrank_eval.py`, SHA-256
`92d7badc482356de0aad69e494f9c81251102ff23aa3f6936b03f72bea29f898`.
Its `compute_summary_overall` function (lines 61–138) computes both:

- `mean_global_dice_per_case`: average the finding Dice scores inside each CT,
  then average those CT means. Each nonempty CT has equal weight.
- `mean_global_dice_per_finding`: average all finding Dice scores directly.
  CTs with more findings contribute more total weight.
- `hit_rate`: total hit findings divided by total findings.

A synthetic execution of that exact function using one CT with a single Dice
of 1, and another CT with three Dice values of 0, returned case-wise Dice 0.5,
finding-wise Dice 0.25, and hit rate 0.25. No images or labels were accessed.

## What the public leaderboard establishes

The [challenge page](https://rexrank.ai/ReXGroundingCT/challenge.html) describes
Dice as an average per finding per case. Its JavaScript sorts and displays the
stored `dice` field without recomputing it. The
[official publishing documentation](https://github.com/rajpurkarlab/ReXrank/blob/gh-pages/ReXGroundingCT/FIREBASE_SETUP.md)
says to evaluate with `rexrank_eval.py` and publish `dice`, `hitRate`, and
`instanceF1`, but does not identify which Dice summary field feeds `dice`.
The linked Hugging Face evaluator requires authentication; the available
local copy was inspected instead. The organizer's current evaluator/publisher
implementation and per-case public scores were not available in these sources.

## Interpretation for V++_b3

- Official published overall Dice: **0.332147596250632**.
- Finding-count-weighted mean of the public category Dice values:
  **0.31461865183074966**.
- These can both be correct under different weighting conventions. The
  second value cannot reconstruct a case-wise mean because the category
  aggregates do not retain which findings belong to each CT.
- Case-wise averaging is a plausible explanation, not a verified reconstruction
  of the 0.332147596250632 value from per-case results.

Preserve both values with their definitions; do not treat their difference
alone as a leaderboard error. No local scoring method or existing ranking
was changed by this investigation. The original source snapshot remains intact.
