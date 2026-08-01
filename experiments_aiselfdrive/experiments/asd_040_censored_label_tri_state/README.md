# ASD-040: Censored-Label Tri-State Laboratory

This experiment asks whether incomplete ReXGroundingCT masks can support safer
training when supervision distinguishes `known_positive`,
`certified_negative`, and `unknown`.

The released multi-entity masks are used to create a synthetic censorship
laboratory. One recorded entity is hidden by each of four deterministic
annotation rules. The hidden entity is available only to a separate evaluator;
training code receives it as `unknown`.

The canonical plan is `experiment.yaml`, and live state is `state.json`. The
experiment starts blocked until `asd_000_evidence_lock_error_atlas` finishes
with outcome `go`.

## Core safeguards

- Official training cases build the censorship laboratory.
- Folds are separated by patient.
- Hidden masks never enter features, sampling, losses, thresholds, or final
  training.
- Anatomically compatible unlabeled tissue remains `unknown`.
- `certified_negative` is restricted to deterministic synthetic/body-padding
  controls; it is not inferred from missing finding annotations.
- val80 may be used for development after the cross-fitted policy is frozen.
- The supported executor rejects declared val120 and val200 access before controller-authorized T4; this is a control-plane guard, not an operating-system secrecy boundary. Val120 is internal replication, not independent confirmation.

Four censorship rules are evaluated separately: volume priority, centrality
priority, annotation/report-order proxy, and seeded random. The main diagnostic
must recover at least 30% of hidden entities at at least 80% precision under
every rule and outperform base confidence. A PU pilot must improve hidden
recall without increasing certified false positives by 10% or more.

Heavy outputs belong under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_040_censored_label_tri_state/run_v1`.
The ignored local `runtime` symlink points to the experiment's external
runtime root. The scaffold runs no training.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `1`
- Updated: `2026-08-01T05:05:20.369094Z`

Blockers:

- `dependency_unsatisfied`: asd_000_evidence_lock_error_atlas: dependency status is ready
<!-- experimentctl:state:end -->
