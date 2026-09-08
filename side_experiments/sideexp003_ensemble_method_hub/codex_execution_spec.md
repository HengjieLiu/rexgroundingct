---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# SideExp003 Execution Specification

## Authority and scope

This file is the operational contract for SideExp003. The tracked repository is
the control plane; `/mnt` is the data plane. The catalog and run specifications
may authorize CPU-only analysis, but neither authorizes model inference or final
recipe materialization without the explicit user gates below.

The foundation milestone ended after steps 1–4. The user subsequently
authorized the frozen top-20 fresh val200 cache job. That authorization covers
the 20 base-model inference passes only; it does not authorize greedy search or
a final `M`.

## Immutable inputs

- fixed val200 dataset:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`;
- dataset SHA-256:
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`;
- expected shape: 200 cases / 381 findings;
- evaluator hit definition: finding Dice `>= 0.1`;
- first-method prediction rule: sigmoid, equal probability average, fixed
  probability threshold `0.5`.

All category metrics must be recomposed by exact `(case name, finding index)`
join. Serialized order is never an alignment key.

## Catalog transaction

For “add ExpXXX to SideExp003 leaderboards”:

1. Run `catalog add --experiment XXX --dry-run`.
2. Require 200 cases, 381 findings, the fixed dataset hash, an unambiguous
   checkpoint mapping, checkpoint SHA deduplication, and explicit threshold
   provenance.
3. If one checkpoint maps to inconsistent evaluators, stop. Preserve all source
   paths and emit an override stub with `canonical_evaluator_path: null`; a human
   must decide. Merge the reviewed path, rationale, reviewer, and UTC time into
   `catalog_overrides.json`, then repeat the dry-run. The tool accepts only a
   path present in that conflict set.
4. Run the same command with `--apply`. This replaces that experiment's snapshot
   and atomically renders both leaderboards from `checkpoint_catalog.json`.
5. Run `hub.py check`, SideExp003 tests, the repo workflow check, and Git
   whitespace check.

`needs_review` records are retained but are roster-ineligible unless a reviewed
override supplies the missing provenance. Markdown ranking rows are generated
only; manual edits are invalid.

## Roster gate

A roster is an immutable JSON document under `rosters/`. Its candidate list is
deduplicated by checkpoint SHA and contains only catalog IDs. Freeze:

- catalog SHA and roster SHA;
- dataset/fold SHA;
- candidate checkpoint/config/cache hashes;
- candidate order and `N`;
- any explicit eligibility overrides and their rationale.

Do not create a run until the user explicitly confirms the roster.

## Cache gate

Reuse a cache only when its key and audit match the frozen run contract. The
top-20 job is an explicit `fresh_only` exception: it must run all 20 models and
must neither read nor fall back to legacy SideExp002 arrays. A new
cache key is a canonical hash over:

```text
dataset SHA + checkpoint SHA + config/inference-contract SHA +
preprocessing-manifest SHA + code version + storage contract version
```

One inference pass must both save logits and calculate the fixed-threshold
metrics used for comparison to the catalog evaluator. Never run a second
“standard evaluation” inference for the same cache.

Fresh caches stage clipped float32 arrays. If an offline float16 cast preserves
every threshold-zero voxel, publish float16; otherwise publish the staged
float32 arrays. This makes dtype fallback inference-free.

Base caches are long-lived. Intermediate greedy rounds save recipes/metrics,
not duplicate logits. Only the user-confirmed final recipe may save derived
ensemble logits.

Legacy SideExp002 exports keep their old logical paths through a symlink. Its 13
strict-gate exports may be reused under their legacy contract. The seven
analysis-usable exports may participate in exploratory/OOF analysis, but an
actually selected final member requires a new versioned strict re-export; legacy
arrays are immutable.

## Search and finalization gates

After a complete cache audit, report:

- deduplicated `N`;
- proposed `Mmax=min(N,10)`;
- greedy candidate evaluations
  `Mmax*N - Mmax*(Mmax-1)/2` for one forward path;
- estimated bytes read, separated by method/fold if applicable.

Wait for user confirmation of `Mmax`. Then run the specified `M=1..Mmax` search
and report candidates for final `M`; do not freeze it. Wait for the user's final
`M` decision before full-val refit and derived-logit materialization.

The final saved logit is:

```text
z_ens = logit(clamp(mean_i(sigmoid(z_i)), epsilon, 1-epsilon))
```

Thus `z_ens >= 0` exactly corresponds to the fixed probability threshold `0.5`.

## Run identity and immutability

Run IDs are `rNNN_<roster-slug>`. Before execution, copy the templates into:

```text
methods/<method>/runs/rNNN_<roster-slug>/
```

Freeze roster/fold/cache hashes, `N`, `Mmax`, parameters, tie-breaks, commands,
code revision, and tracked/runtime outputs. Once execution begins, do not edit
the run contract. A changed roster, cache contract, method, fold, or parameter
requires a new run ID. A new ensemble algorithm requires a new method folder and
`method_spec.md`; never hide it inside an old run.

## Verification boundary

Foundation tests cover catalog alignment/deduplication/conflicts/rendering and
transactional cache migration with rollback. Cache export, greedy, fold leakage,
category routing, and end-to-end materialization tests become mandatory when
those step-5 executors are added; their specifications alone are not permission
to execute them now.
