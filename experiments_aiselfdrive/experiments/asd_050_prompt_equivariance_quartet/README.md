# ASD-050: Prompt-Aware Left/Right Equivariance Quartet

This experiment supplies exact synthetic text-image alignment supervision for
unilateral findings. It coordinates a left/right CT reflection with a
deterministic prompt rewrite and evaluates two factual and two counterfactual
conditions:

```text
original CT + original prompt + original mask
flipped CT  + swapped prompt  + flipped mask
original CT + swapped prompt  + negative only on the known original mask
flipped CT  + original prompt + negative only on the known flipped mask
```

All space outside the known counterfactual-negative mask remains `unknown`.
The package excludes bilateral/multifocal language, right-middle-lobe or
lingular mappings, diffuse/vague findings, mixed laterality, masks crossing
the midline, and any orientation or round-trip failure.

The canonical plan is `experiment.yaml`; live execution state is only in
`state.json`. The experiment starts blocked until ASD-000 finishes `go`.

The synthetic gate requires both factual scores to exceed their paired
counterfactual scores by 0.05 for at least 80% of eligible quartets without
increasing factual empty predictions. The real val80 gate requires at least
15% lower laterality off-target FP, at least 10 percentage points higher
laterality compliance, and no aggregate Dice loss beyond 0.001.

Heavy outputs belong under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/asd_050_prompt_equivariance_quartet/run_v1`.
The ignored local `runtime` symlink points to the experiment's external
runtime root. The scaffold launches no training.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `0`
- Updated: `2026-07-31T00:00:00Z`

Blockers:

- `dependency_unsatisfied`: The locked lineage, prompt ontology, cohorts, and laterality error atlas are required.
<!-- experimentctl:state:end -->
