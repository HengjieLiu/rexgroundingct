---
created: 2026-07-26
updated: 2026-07-26
status: research_design
experiment_id: 010_voxtell_public_anatomy_prior_fusion
---

# VoxTell Public Anatomy-Prior Fusion

This experiment line studies whether masks from a public anatomical
segmentation model can improve free-text finding grounding when they are fused
into both VoxTell's visual features and its text-conditioned decoder.

## Current Status

- Data-audit implementation is active; no anatomy-fusion training has started.
- Proposed starting point: public VoxTell v1.1.
- Proposed training recipe: experiment 006
  `v123_cached_e5_d4`.
- Recommended first anatomy source: TotalSegmentator anatomical tasks, with
  lesion-predicting tasks deliberately excluded.
- TotalSegmentator is pinned as `external/TotalSegmentator` at release
  `v2.16.0`.
- The first executable phase inventories all 117 `total` labels over fixed
  val200 using the 3 mm model, then selects a smaller thoracic ontology for a
  1.5 mm quality check.

## Files

- `research_report_v1.md`: literature and code audit, candidate fusion designs,
  recommended architecture, ablation matrix, cache contract, and evaluation
  plan.
- `prompt_anatomy_roi_audit.md`: complete train/validation/test prompt audit,
  TotalSegmentator ROI priorities, derived-region requirements, exclusions,
  and cache/fusion recommendations.
- `anatomy_constraint_applicability_val_test.md`: val/test applicability
  estimates, conservative and expanded constraint cohorts, routing policy,
  boundary-risk analysis, and required validation gate.
- `lung_gating_oracle_val200_preliminary.md`: preliminary oracle whole-lung
  gating analysis on exp006 `v123_cached_e5_d4` epoch-100 val200 predictions,
  using validation target containment to estimate the upper-bound value of
  selective lung restriction.
- `lung_gating_prompt_only_val200.md`: prompt-only whole-lung gating
  validation on exp006 `v123_cached_e5_d4` epoch-100 val200 predictions,
  freezing text-only routing before GT containment and prediction metrics.
- `codex_execution_spec.md`: executable val200 anatomy-inventory contract,
  storage paths, smoke gate, and completion criteria.
- `codex_execution_spec_thorax_val200.md`: second-stage informative val200
  thorax-cache contract using high-resolution selected `total` ROIs plus
  `lung_vessels`, `trunk_cavities`, and `body`.
- `../../docs/voxtell/bronchopulmonary_segments_and_pseudo_segment_priors.md`:
  reusable S1-S10 segment reference, geometric pseudo-segment design,
  alternatives, safeguards, and validation requirements.

Heavy masks, checkpoints, predictions, and runtime outputs must remain outside
Git under the shared data or experiment roots.
