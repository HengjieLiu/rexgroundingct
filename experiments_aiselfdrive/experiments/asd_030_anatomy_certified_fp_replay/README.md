# ASD-030: Anatomy-Certified False-Positive Replay

This experiment tests one narrow intervention: replay high-scoring false
components mined by the locked baseline, while applying negative supervision
only where prompt and anatomy evidence certify incompatibility.

The canonical plan is `experiment.yaml`; live status is only in `state.json`.
The experiment starts blocked until both `asd_000_evidence_lock_error_atlas`
and `asd_005_train_anatomy_asset_extension` finish with outcome `go`.

## Scientific contract

- A voxel or component is `known_positive` when it overlaps the released
  finding mask.
- A component is `certified_negative` only when a high-confidence prompt
  parser supplies a supported side or lobe, the anatomy case passed geometry
  QC, and the component is completely disjoint from the 15-mm-dilated allowed
  region while lying in a verified incompatible thoracic region.
- Every anatomically compatible, ambiguous, unlabeled, or QC-failed region is
  `unknown` and receives zero negative loss.
- Mining always uses the frozen baseline locked by ASD-000. The replay manifest
  is immutable after its hash is recorded.
- This experiment does not add anatomy features, change the decoder, refresh
  candidates from a trained intervention, or use uncertified regions as empty
  labels.

## Decision path

1. Implement and test the isolated replay pipeline.
2. Validate dependency hashes, cohort separation, geometry, and Docker
   hardware.
3. Mine baseline components from official training cases.
4. Certify labels and freeze the replay manifest.
5. Require certified top-5 candidates to explain at least 30% of val80
   prompt-off-location false-positive volume.
6. Run a 500-update matched pilot only if the coverage gate passes.
7. Run a 2,000-update decision study only if the pilot passes.
8. Access val120 and val200 only after external T4 promotion authorization.

A failed scientific gate closes as `finished/no_go`; a lineage, leakage,
geometry, identity, device, or non-finite failure is invalid and is never
automatically weakened.

## Files

- `configs/protocol.yaml`: frozen method and arm configuration.
- `scripts/`: entrypoints an implementation agent must create.
- `src/`: experiment-local library code an implementation agent must create.
- `tests/`: required unit and integration tests.
- `artifacts/manifest.json`: artifact ownership and retention.
- `results/result.json`: scaffold result record, replaced at verified closeout.

Heavy artifacts belong under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_030_anatomy_certified_fp_replay/run_v1`.
The ignored local `runtime` symlink points to the experiment's external
runtime root. No training is launched by the portfolio scaffold.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `0`
- Updated: `2026-07-31T00:00:00Z`

Blockers:

- `dependency_unsatisfied`: The locked baseline lineage, cohorts, prompt ontology, and error atlas must be verified first.
- `dependency_unsatisfied`: The geometry-verified training anatomy manifest is required for certification.
<!-- experimentctl:state:end -->
