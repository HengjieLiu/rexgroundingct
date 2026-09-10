---
created: 2026-07-23
updated: 2026-07-23
status: active
---

# Submission Workflow

This is the canonical challenge-level packaging checklist. It is model-agnostic:
VoxTell-derived outputs are the current likely path, but the checks apply to any
ReXGroundingCT submission.

## Central submission records

Open [the submission register](../submissions/REGISTER.md) before choosing a
prediction directory or an archive. It maps a1–d3 and reserved d11/d12 to their actual names and
paths, model/checkpoint recipes, postprocessing and recorded upload history.
The c-series aliases retain Exp025's baseline/whole_lung20/fine20 directories.
See [the register workflow](../submissions/README.md) for check, refresh and
render commands. Future leaderboard evidence and reviews belong under the
same top-level `submissions/` folder.

d11 is the frozen collaborator semantic-v1 applied to d1; d12 applies frozen
strict semantic-v2 to d11. Their [SideExp005 validation audit](../side_experiments/sideexp005_postprocessing_merge/README.md)
is separate from test readiness. Separate SideExp005 run
`r002_d1_test300_d11_d12_frozen_postprocessing` is authorized to generate test
d11/d12 after the val200 report is verified, regardless of validation scores.
ZIPs are skipped by request. Existing d1/d2/d3 completion evidence cannot cover
d11/d12; their own producer and per-file manifests establish prediction readiness.

Keep prediction readiness, ZIP readiness and upload history separate. Record
the exact submitted name, uploaded ZIP hash and external ID after a confirmed
upload; do not infer submission from completion of predictions. Existing
prediction directories and producer manifests remain authoritative.

## Before Packaging

1. Refresh or verify official sources:

```bash
python challenge_info/update_challenge_info.py --verify-latest
```

2. Read:

- `challenge_info/README.md`
- `challenge_info/SUMMARY.md`
- `challenge_info/rexgroundingct_challenge.md`
- `challenge_info/rexrankct_submission_guideline.md`

3. Record the candidate run:

- repo commit;
- model checkpoint path;
- canonical config;
- runtime experiment directory;
- prediction directory;
- source snapshot timestamp from `challenge_info/README.md`.

## Prediction Requirements

Follow the current challenge page as the submission authority. At the time of
the archived source snapshot, MICCAI submission expects a single `.zip` file
containing prediction `.nii.gz` files for the 300-case test split.

Before zipping:

- verify there are exactly 300 test prediction files;
- verify filenames match the official test input names;
- verify each file is a NIfTI prediction, not a directory or symlink target;
- verify no logs, configs, checkpoints, hidden files, or parent folders are in
  the zip payload;
- verify output masks use the evaluator-compatible orientation and shape.

## Packaging Pattern

Use a clean staging directory outside Git and outside the runtime prediction
directory:

```bash
export SUBMISSION_STAGE=/mnt/shengdata1/hengjie/experiments/rexgroundingct/submission_stage
export PRED_DIR=/mnt/shengdata1/hengjie/experiments/rexgroundingct/<experiment-id>/predictions

rm -rf "$SUBMISSION_STAGE"
mkdir -p "$SUBMISSION_STAGE"
```

Copy only validated test prediction files into the staging directory. Then zip
the files themselves, not the containing folder:

```bash
cd "$SUBMISSION_STAGE"
zip -9 ../rexgroundingct_submission_<date>_<slug>.zip ./*.nii.gz
```

Do not run destructive cleanup commands against the source prediction directory.

## Final Checks

Run a zip listing and compare it against the expected test file list:

```bash
zipinfo -1 /mnt/shengdata1/hengjie/experiments/rexgroundingct/rexgroundingct_submission_<date>_<slug>.zip | sort | head
zipinfo -1 /mnt/shengdata1/hengjie/experiments/rexgroundingct/rexgroundingct_submission_<date>_<slug>.zip | wc -l
```

Record the final zip path, byte size, SHA-256, source commit, source snapshot,
and candidate run in a repo-local submission note or experiment report. Keep
the zip itself outside Git.
