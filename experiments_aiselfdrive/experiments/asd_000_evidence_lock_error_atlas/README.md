---
created: 2026-07-31
updated: 2026-07-31
status: blocked
experiment_id: asd_000_evidence_lock_error_atlas
---

# Evidence Lock and Error Atlas

This CPU-first foundation experiment converts the locally available Exp009
baseline into a single reproducible lineage. It locks the report, checkpoint,
embedding bank, val200, accepted val80, preprocessing, prompt, threshold, and
post-processing contracts before any new method is trained.

The technical report is historical evidence. Its `/common/lidxxlab/...`,
Slurm, H100/H200, and 2048-dimensional text references are not executable
inputs. The local contract uses the 6467-by-2560 embedding bank and the Exp009
RTX 6000 Ada run.

## Primary acceptance gate

The stored Exp009 baseline must reproduce Dice `0.339839 ± 0.000005` and
exactly `288/381` hits at threshold `0.5` and hit threshold `0.1`. The accepted
80-case category-aware probe becomes development-only data. Its 120-case
complement is sealed and must not be evaluated before a T4 stage.

## Execution

Read `experiment.yaml`, `claims.yaml`, `state.json`, and
`configs/protocol.yaml` before taking any action. The current state is blocked
after the initial implementation stage; do not claim or reopen it unless a
future repair plan explicitly resolves the recorded blockers. No training is
part of this experiment.

Heavy or regenerated artifacts belong under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_000_evidence_lock_error_atlas
```

The source report is immutable. Existing checkpoints, masks, logits, metadata,
and anatomy caches are read-only.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `11`
- Updated: `2026-07-31T15:46:44.528595Z`

Blockers:

- `stage_failed`: command exited 2; stderr tail: StageContractError: invalid completion contract for stage lock_local_lineage: evidence_file 'experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/lineage_lock.json' is also a declared output. The active experimentctl verifies every file output SHA-256 from inside that same evidence JSON, which would require an impossible SHA-256 self-reference. The immutable claimed plan and controller are outside this stage's allowed write paths.

- `confirmatory_blind_compromised`: Confirmatory blind compromised during repair diagnosis: /mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json, the per-case val200 evaluator, was loaded in full before any T4 stage. This materialized val120 metrics and displayed train_13075_a_2.nii.gz, which is absent from locked val80 source SHA-256 61048d78be0d72ff28619ebfff3efac8501dc0a6154deaa38881e26bca054faf. Full val200 anatomy metadata JSONs were also loaded for geometry diagnosis; those contain label-derived statistics even though only geometry was inspected. No val120 label mask or prediction array was loaded, but the metric-access stop condition is violated; ASD-000 must not be reopened or claimed in this session.
- `post_stage_repair_noncanonical`: Authorized post-stage repair is noncanonical and unexecuted: scripts/run_stage.py changed from sealed implementation evidence SHA-256 cc1d2fef5168af795bde3769a6a50538b036be8150b5ad696ec6948d35c01f15 size 16255 to fb54762dc1f95b798932b112719c1375789415cdb12662364740d7b2c631782e size 84031; tests/test_error_atlas.py changed from c96809995cafccf227a388c67eba9d620dbaa562ba6be9986cef921d5493676c size 16934 to 95b973958c2c6eeb23069548015900ccc36cff4af1c7266af1b425af0d054da5 size 24188; configs/protocol.yaml changed from pre-repair df019e4158488a8a5bc8cd3013aacaa3a9d64dd0c227133537facf7c5e006374 to 6ac6717af234212010694bfc1082f78297c66893c53871aaa1062355e5c0361f. The repaired files compile and 53 synthetic tests pass, but completed-stage evidence and its event hash were correctly left unchanged. Full execution was not run. Before any future clean execution, re-enter through a revised implementation stage and resolve the remaining atlas review findings: vectorize O(K*N) component scans, do not certify outside-lung tissue for laterality-only prompts, and emit null totals with observation counts when spatial supervision is unavailable.
<!-- experimentctl:state:end -->
