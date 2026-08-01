---
created: 2026-07-31
updated: 2026-07-31
experiment_id: asd_005_train_anatomy_asset_extension
---

# Train/Validation Fast Anatomy Asset

This experiment creates one reusable TotalSegmentator `total --fast` anatomy
cache for all 3,192 released train and validation CTs. It runs the pinned
3-mm model only, preserves each source CT shape and affine, and uses four
independent single-GPU Docker workers.

It is blocked until `asd_000_evidence_lock_error_atlas` finishes with outcome
`go`. It does not train VoxTell, generate lesion labels, run high-resolution
TotalSegmentator, use Slurm, or inspect test data.

The durable unified manifest will be:

```text
experiments_aiselfdrive/shared/manifests/trainval_fast3mm_anatomy_v1.json
```

Heavy masks live under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_005_train_anatomy_asset_extension/anatomy_fast_3mm_trainval_v1
```

Every case is atomic and receives a hashed JSON completion marker. Empty touch
files are not completion evidence.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `1`
- Updated: `2026-08-01T05:04:42.863884Z`

Blockers:

- `dependency_unsatisfied`: asd_000_evidence_lock_error_atlas: dependency status is ready
<!-- experimentctl:state:end -->
