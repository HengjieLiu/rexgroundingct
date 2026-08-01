# ASD-060: Multi-Threshold Persistence-Tree Set Verifier

This experiment treats a prediction as a variable-cardinality set of persistent
component branches, not as independently classified blobs.

For thresholds `0.05, 0.10, ..., 0.80`, connected components form a
low-to-high-threshold forest. An exact ground-truth oracle is computed first
with an antichain constraint: an ancestor and descendant cannot both be
selected. No learned verifier runs unless the oracle improves Dice by at least
0.07 and candidate recall is at least 85%.

Learned selection additionally requires ASD-040 to finish with outcome `go`
and publish a verified tri-state policy. If ASD-040 is `no_go` or
`inconclusive`, learned stages are skipped. High oracle headroom without safe
labels closes `inconclusive`; low oracle headroom closes `no_go`.

The canonical plan is `experiment.yaml`; live status is only in `state.json`.
The experiment starts blocked only by ASD-000. The CPU oracle may run as soon
as ASD-000 is `go`; `resolve_tri_state_branch` then waits for a terminal
ASD-040 evidence outcome before any learned stage.

The learned model uses set context and exact tree decoding. It must recover at
least 20% of the oracle gain, outperform an identical-feature independent
scorer, preserve at least 99% of baseline true-positive voxels, and obtain a
positive case-clustered interval.

Heavy outputs belong under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_060_persistence_tree_set_verifier/run_v1`.
The ignored local `runtime` symlink points to the experiment's external
runtime root. The scaffold runs no oracle or training.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `0`
- Updated: `2026-07-31T00:00:00Z`

Blockers:

- `dependency_unsatisfied`: The locked lineage, cohorts, and verified val80 logits are required.
<!-- experimentctl:state:end -->
