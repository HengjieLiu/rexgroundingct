# Scripts

`run_stage.py` is the argument-array command entry point and exposes the
`selfcheck`, `lineage`, `cohorts`, `logits`, `atlas`, and `closeout` phases.
`export_val80_logits.py` is the pinned Docker adapter used only when no
accepted Side Experiment 002 export exists. It writes compressed arrays only
under the ASD-000 external runtime and enforces the 80-case/195-finding,
32-GiB-free-memory, two-GPU-hour, 64-GiB-storage, geometry, finite-value, hash,
exact text-bank/checkpoint lineage, offline execution, durability-probe, and
exact threshold-mask contracts. Nested Docker uses a controller-owned cidfile;
on timeout the runner verifies both lease and experiment labels before stopping
the owned container. Every command failure writes attempt- and plan-bound
structured evidence with normalized error class and measured GPU, CPU, and
storage resources.
