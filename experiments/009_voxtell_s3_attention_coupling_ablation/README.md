---
created: 2026-07-27
updated: 2026-07-27
status: active
experiment_id: "009_voxtell_s3_attention_coupling_ablation"
---

# VoxTell S3 Attention Coupling Ablation

Experiment 009 compares a same-source exp006 e5/d4 continuation baseline with
three revised `1/2 + 1/4` S3 attention coupling variants labeled `s3v1`,
`s3v2`, and `s3v3`.

The first all-scale attempt is preserved as runtime evidence: full-resolution
`1/1` S3 attention OOMed during one-update smoke on the S3 arms, so the old run
was stopped and the launch plan was revised to remove `1/1`.

## Files

- Canonical config:
  `configs/experiments/009_voxtell_s3_attention_coupling_ablation.json`
- Execution spec:
  `experiments/009_voxtell_s3_attention_coupling_ablation/codex_execution_spec.md`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation`

## Artifact Policy

Logs, predictions, checkpoints, and raw evaluator outputs stay under the
runtime directory on `/mnt/shengdata1`. Only small curated summaries and
provenance files belong in the repo.

## Training Dynamics

Canonical total-loss and S3 objective-component figures are generated under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/reports/training_dynamics/`.
The shared gallery is
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/comparisons/training_dynamics/README.md`.
Compare absolute losses within Exp009 only; the base-v123 panel isolates the
component shared by its baseline and S3 variants.
