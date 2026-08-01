---
created: 2026-07-31
updated: 2026-07-31
experiment_id: asd_000_evidence_lock_error_atlas
---

# Revision-2 Evidence Lock and Error Atlas

This CPU-first foundation experiment converts the locally available Exp009
baseline into a single reproducible lineage. Revision 2 recomputes Dice and
hits directly from stored predictions plus released ground truth; evaluator
summary JSON is a historical cross-check, never the reproduction source. It
also requires verified val80 logits before any atlas or downstream GO.

The technical report is historical evidence. Its `/common/lidxxlab/...`,
Slurm, H100/H200, and 2048-dimensional text references are not executable
inputs. The local contract uses the 6467-by-2560 embedding bank and the Exp009
RTX 6000 Ada run.

## Primary acceptance gate

The stored Exp009 baseline must reproduce Dice `0.339839 ± 0.000005` and
exactly `288/381` hits at threshold `0.5` and hit threshold `0.1`. The accepted
80-case category-aware probe is development data. Its 120-case complement has
already been historically exposed and is therefore internal replication only:
it is not blinded, untouched, external, or independently confirmatory. An
independent performance claim requires new externally unseen data with a
recorded custodian/access history.

## Execution

Read `experiment.yaml`, `claims.yaml`, `state.json`, and
`configs/protocol.yaml` before taking any action. `experiment.yaml` revision 2
was adopted through the controller's audited replan transition. The exact
revision-1 plan, state, result, claims, and artifact manifest are hash-verified
in the replan archive and event ledger. Execute revision 2 only through
`tools/experimentctl.py`. No training is part of this experiment.

The command DAG is `selfcheck`, `lineage`, `cohorts`, `logits`, `atlas`, then
`closeout`. The logit phase first accepts only a fully verified Side Experiment
002 export; otherwise it runs a single-GPU, val80-only Docker adapter under the
external runtime. That fallback uses the exact lowercased lookup contract for
the locked 6467-by-2560 bank (195/195 val80 and 381/381 val200 prompt
occurrences, with no Qwen fallback), disables the network, mounts sources
read-only, and overlays only this experiment's runtime as writable. It performs
an atomic write/fsync/replace/directory-fsync/read/delete probe before GPU
inference. The atlas treats compatible unlabeled tissue as `unknown`, never as
a global negative. For laterality-only prompts, outside-lung tissue is also
unknown.

Heavy or regenerated artifacts belong under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_000_evidence_lock_error_atlas
```

The source report is immutable. Existing checkpoints, masks, logits, metadata,
and anatomy caches are read-only.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `ready`
- Outcome: `pending`
- Current stage: `selfcheck_revision_2_implementation`
- State revision: `12`
- Updated: `2026-08-01T05:03:27.677790Z`
<!-- experimentctl:state:end -->
