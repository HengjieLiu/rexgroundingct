---
created: 2026-09-11
updated: 2026-09-11
status: implementation_verified
---

# Verification and launch record

The previous category experiment stopped at exactly 2000 updates per arm,
without starting its epoch-20 validation. Its coordinator was held while the
remaining bounded trainers finished; all full-state checkpoints and journals
were verified before TERM then CONT allowed the coordinator to exit. At
2026-09-11 23:26:12 UTC its stop receipt was published. Earlier complete
validation barriers remain 100/500/1000. Old source/config/schedule hashes are
unchanged, and all checkpoints and results are preserved for user review.

[Verified old stop receipt](../runtime/deletion_categories_bcde_50ep/reports/user_stop_update2000.json).

All 16 A-only CPU tests passed in 22.702 seconds in the existing VoxTell Docker
image with GPU devices disabled. Coverage includes original-train/B exclusion,
A membership, patient separation, complete schedule coverage, per-epoch patch
mixtures/fallbacks, common pristine initialization, exact loss values/gradients,
empty groups, padding, overlapping/small-volume deletion inference, strict
threshold ties and identity, four variable-cohort barriers, interruption at
update 2107 with identical resumed weights through 2200, partial-evaluation
recovery without repeated inference, 201-threshold analysis and A/B/full tables,
completed-only A/B publication, case counting for multiple findings in one CT,
pending/failure states, single-writer ownership and Docker-free GPU dry runs.

The shared cache's 29 CPU regression tests passed in 3.685 seconds, covering
geometry round trips, storage-mask preservation, channel/finding ordering,
cache-only loading, integrity failures and tile indexing. The repository
workflow check passed with 19 pre-existing missing-runtime/sync warnings;
Python/shell syntax and whitespace checks passed.

The clearly labeled synthetic 3x3 dashboard was visually inspected. One
category has separate A/B results and baselines while other categories show
pending CT/finding counts. A uses dashed/hollow lines and B solid/filled;
circles denote 0.50 and squares 0.90. All tables label A fitted, B held-out
development and full mixed exposure. Partial values never enter the curves.

Real CPU preparation passed the required 177-CT/252-finding validation inventory,
original IDs, exact baseline reproduction and A/B patient checks. Training
pools are 25/31/61/8 A findings, each visited by its fixed 2000-event schedule;
B cohorts are 24/29/71/3 and excluded from gradients. Full validation retains
all findings, including the three base-empty 2d B examples. The initial state
matches historical pristine weights ff8d94ec... . New execution context:
`57d3363ebe943a5178a46e4997857437002ecabc6946405b374d744109ea1766`.

Four real 192-cubed patch probes passed using only the cache loader. Each was
finite FP32 and matched scheduled TP/FP counts. Per-subset A/B/full baselines,
all-A schedule coverage, pristine initialization and the old-stop gate passed.
Source snapshots, tests, workflow log, synthetic figure and real-input evidence
are preserved under the new runtime verification directory.

The authorized detached launch started container
`rex027_deletion_categories_bcde_val_a_20ep` / `2e6c1064960a`, assigning
2b/2c/2d/2e to GPUs 0/1/2/3. Target: 2000 updates each, evaluation at
100/500/1000/2000, then pending_user_review. Live startup verification follows.


At 2026-09-11 23:31:15 UTC (16:31 Pacific), initial live verification found
14/13/11/15 updates in 2b/2c/2d/2e order. Every recorded update used an A finding,
all losses and gradients were finite, and all four initial checkpoints matched
the historical pristine weight hash. FP32/no TF32 and the shared Dice+TP2 loss
were verified. The dashboard was refreshing live; validation remains pending
until update 100. Evidence: runtime `verification/initial_live_check.json`.
