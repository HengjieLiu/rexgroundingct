---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# Prior Ensemble Evidence

This is a read-only synthesis of earlier work. The linked experiment artifacts
remain authoritative; SideExp003 does not rewrite SideExp002 results.

## Exp003: orientation-rescue ensemble

### Four-model probability averaging

The four v1/v2/v3 rescue arms were averaged after sigmoid. On the 20-case probe,
epoch 80 reached Dice `0.355469` and `21/31` hits at threshold `0.45`; the best
single model reached `0.347774`. On fixed val200, epoch 100 reached Dice
`0.284191` and `263/381` hits at the best global threshold `0.35`.

The per-category threshold sweep reached Dice `0.289644` and `264/381` hits. It
selected thresholds and reported them on the same val200 findings, so it is a
hindsight/category-threshold oracle, not a generalization estimate.

- data: Exp003 val20 probe and fixed val200;
- members: four orientation-rescue checkpoints;
- fusion: equal probability average;
- threshold: swept (`0.45` on val20, `0.35` global on val200);
- limitations: threshold selection reused the evaluation split; Exp003 lacks
  catalog-grade fixed-`0.5` threshold provenance;
- sources: `experiments/003_voxtell_rex_ft_rescue_ablation/prediction_ensemble_plan.md`
  and `experiments/003_voxtell_rex_ft_rescue_ablation/category_threshold_sweep_report.md`.

### Hard voting and multiscale

Hard voting was planned, but no complete, provenance-backed result artifact was
found. Multiscale single-model val20 evaluations were executed and did not beat
native scale; a multiscale probability ensemble was contemplated but did not
produce a durable, trustworthy complete result. These are historical plans, not
positive ensemble evidence.

Sources: `experiments/003_voxtell_rex_ft_rescue_ablation/prediction_ensemble_plan.md`
and `experiments/003_voxtell_rex_ft_rescue_ablation/multiscale_inference_plan.md`.

## SideExp002: 20-model ensemble selection

### Hard-mask complementarity

The 20 selected fixed-val200 models were compared using saved hard-mask outcome
evidence. A best-per-finding hindsight oracle reached Dice `0.403015` when all 20
models were available. This demonstrates complementarity but is not an
achievable deployed ensemble because it chooses the winner after seeing each
finding's ground truth.

Source:
`side_experiments/sideexp002_multimodel_ensemble_selection/outputs/hard_mask_complementarity.md`.

### Uniform probability ensemble

All 20 exported models were averaged in probability space. Its best same-split
global threshold was `0.55`, with Dice `0.344846` and `290/381` hits. The
category-specific threshold oracle reached Dice `0.354427` and `292/381` hits;
because category thresholds were chosen and evaluated on the same val200 set,
that value is diagnostic only.

Sources:
`side_experiments/sideexp002_multimodel_ensemble_selection/outputs/all20_uniform_probability_threshold_report.md`
and `all20_uniform_probability_threshold_summary.json` beside it.

### Export integrity and unfinished search

All 20 exports contain the full structural inventory: 200 cases each and 4,000
case arrays in total. The final runtime audit accepted 13 through the strict
gate. Seven are analysis-usable but failed the strict final gate:

- `exp006_e6d4_e100`
- `exp006_e5d4_e100`
- `exp007_ddp_e050`
- `exp007_ddp_e075`
- `exp007_ddp_e100`
- `exp008_dual_e080`
- `exp008_joint_e080`

Those seven may support exploratory or OOF search. If one becomes a final
member, it must be strictly re-exported into a new versioned cache; its legacy
array must not be overwritten.

SideExp002 implements 5-fold subset/weight search, but the finalizer timed out
and no complete trusted formal search result/report was produced. The existence
of code is not counted as a result.

Sources: the runtime `finalization_manifest.json`, tracked
`candidate_manifest.json`, `search_ensembles.py`, and the absence of completed
formal selection outputs documented by SideExp002's status.

## Lessons carried into SideExp003

- Save logits/probabilities, never only binary masks, for model selection.
- Reuse base-model inference; search is a CPU/storage problem after caching.
- Separate same-split oracle diagnostics from OOF estimates.
- Hold probability threshold fixed at `0.5` in the first methods so member
  selection is the only searched degree of freedom.
- Prefer a small near-peak ensemble after observing the full `M` curve, but let
  the user decide `M` after the search rather than selecting it implicitly.
