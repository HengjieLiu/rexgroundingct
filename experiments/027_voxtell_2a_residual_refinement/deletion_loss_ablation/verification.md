# Verification and launch

Implementation uses a separate runner/config/runtime and leaves historical
deletion source files, cached arrays and checkpoints unchanged.

CPU Docker verification: 53 tests passed, including the new loss formulas and
exact historical BCE gradients; finding-aware Dice/full-volume equivalence;
empty groups and gradient directions; original geometry/tiling; exact real
A-only schedule; actual training-loop interruption at update107 and identical
resumed weights at200; evaluation interruption/reuse of completed findings;
201-threshold analysis and strict ties; provisional dashboard and failures.
The Docker image had zero visible GPUs. Image SHA prefix: `8ff421d05fbf`.

Workflow check and shell/Python syntax checks pass. The workflow reports19
existing warnings about older missing runtime/config/sync artifacts; none is a
new error. Whitespace checks pass. A labeled synthetic 3×3 dashboard was
rendered and visually inspected. All unavailable scores remain pending.

The no-opt-in host command reports `gpu_work_started: False` and the four loss
identities without creating runtime state. Full scheduling is pinned to
2000 updates and100/500/1000/2000 milestones; altered/unset settings are rejected.

## Commands

```bash
# CPU preparation only; also performed automatically by orchestration.
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh prepare

# Dry run, no Docker or GPU launch.
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh orchestrate --gpus 0 1 2 3

# Authorized four-GPU launch; preparation verifies all69 findings first.
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh orchestrate --gpus 0 1 2 3

# Recovery after an interrupted container has exited; choose a unique name.
START_GPU_WORK=1 DETACH=1 DELETION_ABLATION_CONTAINER_NAME=rex027_loss_resume bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh orchestrate --resume --gpus 0 1 2 3

# Independent CPU report/analysis/audit, only when its corresponding writer is idle.
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh report
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh analyze
bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh audit

# Standalone worker commands require prepared initialization and exclusive GPU ownership.
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh train --arm bce_tp2 --update 100 --gpus 0
START_GPU_WORK=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh evaluate --arm bce_tp2 --update 100 --gpus 0
```

Runtime launch, preparation and status JSON files are authoritative. The
orchestrator performs a completion audit and comparison with historical run3
before setting `pending_user_review`. A reference mismatch stops closeout for
investigation, without selecting or ranking results.

## Launched

The authorized container `rex027_deletion_loss_ablation_a_20ep` launched at
2026-09-11 00:18:32 UTC (September10 17:18 Pacific), ID prefix `7b70eda09f13`.
Its first phase verifies existing validation inputs before starting trainers.

- [Launch manifest](../runtime/deletion_loss_ablation_a_20ep/launch_manifest.json)
- [Current status](../runtime/deletion_loss_ablation_a_20ep/status.json)
- [Durable CPU verification and source hashes](../runtime/deletion_loss_ablation_a_20ep/verification/verification.json)
- [CPU test log](../runtime/deletion_loss_ablation_a_20ep/verification/cpu_tests.log)
- [Labeled synthetic dashboard](../runtime/deletion_loss_ablation_a_20ep/verification/synthetic_dashboard/reports/live_dashboard.md)

The source snapshot matches every file hash in the launched context. No trained
historical weights or optimizer state were reused.

Validation-cache verification passed all63 CTs/69 findings in88.80 seconds.
All four trainers subsequently completed100 updates with finite losses and
gradients, peak allocated memory about12.0 GiB, and began the first concurrent
69-finding evaluation. The dashboard publishes partial A/B results independently.

The update100 BCE-reference model digest is exactly identical to the historical
A-only run3 checkpoint. All four new checkpoints share the pinned initial and
schedule digests, while their trained model digests are distinct. The early
reference losses, both BCE components and pre-clipping gradient norms also
reproduce exactly. See the [checkpoint100 proof](../runtime/deletion_loss_ablation_a_20ep/verification/checkpoint100.json).
Full experiment results remain pending; orchestration continues to2000 updates.

## Completion

Completed at2026-09-11 02:14:41 UTC, status `pending_user_review`. All8000
updates,16 evaluations and16 dense analyses passed the automated audit. The
BCE reference reproduces historical A-only model weights and per-finding
metrics exactly at all four evaluated checkpoints. See [completed results](results.md)
and the [completion audit](../runtime/deletion_loss_ablation_a_20ep/reports/completion.json).
No automatic continuation, threshold selection or model ranking occurred.
