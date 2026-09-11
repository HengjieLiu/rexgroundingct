# Four-arm deletion verification and launch

Date: 2026-09-10. Implementation checks passed; authorized stage 1 completed.

Completion audit: all arms have exactly 2,000 consecutive updates and 21 saved
checkpoints, including initialization. All sixteen evaluations contain the
expected 69 findings and match their retained checkpoint hashes. The coordinator
stopped at 20:08:43 UTC with `pending_user_review`; the container exited with
code 0. All losses and gradients were finite. See [results](results.md).

Container: `rex027_deletion_four_20ep`, ID
`154b1f78847ebe02de4ec39ff2852e425a7b57c78d04896ec96ce6bdeb408ec8`.
The runtime launch manifest records 17:14:25 UTC. CPU preparation completed
in 705.0 seconds, and all four GPU trainers started at 17:26:05 UTC.

- Complete input coverage passed: 864 CTs / 1,189 findings. One training finding,
  `train_7777_a_2.nii.gz::4`, has no base-positive voxels and is explicitly
  bypassed. Eligible pools contain 1,119 training findings, 35 A findings and
  69 A+B findings.
- All four immutable 2,000-update schedules passed actual-data membership
  checks and exact per-epoch branch counts. Run 2 has exactly 50 training and
  50 A events per epoch.
- All four update-zero checkpoints match the pristine FP32 initialization
  (weight SHA beginning `ff8d94ec3648`). Their cursor is zero and TF32 is off.
- All four completed exactly 100 optimizer updates with finite losses and
  gradients, saved their update-100 checkpoint, and entered the first concurrent
  evaluation at 17:30:21 UTC. Peak allocated training memory is 11.97 GiB per arm.
- At 17:30:59 UTC, each model had completed its first validation finding. The
  live Markdown and figure had already published provisional A/full metrics,
  the matching subset baseline, and pending B metrics with 0/34 completed.
  All six removal thresholds were present in the underlying partial records.
  Full evaluation and the remaining training milestones are still running;
  these checks establish launch health, not completed experimental results.
- The previous residual run remains stopped at update 5,000, and the completed
  200-update deletion diagnostic remains pending user review.

- 22 CPU tests passed in the existing VoxTell Docker image, with GPUs disabled:
  exact tile counts including padding, empty branch fallback, all four source
  memberships, per-epoch source and branch counts, fixed schedules, incomplete
  cache gate, receipt invalidation, model/loss behavior, subset-preserving
  tiling, checkpoint/optimizer/RNG replay from initial and later states, journal
  recovery, all-four evaluation barrier, A+B aggregation, provisional/final
  dashboard replacement, failures, undefined values and single-writer ownership.
- The 4×3 synthetic dashboard was rendered and visually inspected. It shows
  partial results before other arms finish, an absent B subset, readable
  failure details and consistent colors. It is explicitly labeled synthetic.
- Canonical repository workflow, script compilation, shell syntax, dry launcher
  and whitespace checks passed. The workflow retains 19 existing missing-runtime
  or missing-sync-artifact warnings unrelated to this method.
- The launch uses the existing image SHA beginning `8ff421d05fbf`; exact image,
  source hashes and config are recorded in the external launch/context manifests.

```bash
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_four_host.sh orchestrate --gpus 0 1 2 3
```

Preparation is CPU-only and reuses all verified arrays. All four GPU trainers
are gated on complete eligibility indexes and the verified input manifest.
The run stops at 2,000 updates per arm after evaluations at 100/500/1000/2000.
Epoch 21 and later are not authorized by this launch.

- [Live dashboard](../runtime/deletion_four_arm_20ep/reports/live_dashboard.md)
- [Source/config/cache context](../runtime/deletion_four_arm_20ep/context.json)
- [Launch manifest](../runtime/deletion_four_arm_20ep/launch_manifest.json)
- [Input coverage and bypasses](../runtime/deletion_four_arm_20ep/input_manifest.json)
- [Synthetic fixture](../runtime/deletion_four_arm_20ep/verification_fixture/reports/live_dashboard.md)

Recovery is explicit via the same command with `--resume` and a fresh
`DELETION_CONTAINER_NAME` (the completed container is retained). It validates code,
inputs, schedules and checkpoint hashes, resumes the saved cursor, and completes
missing evaluations before the next training segment. Do not use it to resume
a numerical failure without diagnosing that failure first.

While this run is active, preserve the source files enumerated in `context.json`.
Each training/evaluation phase verifies those hashes; changing a pinned module
would stop the run at the next phase boundary. Documentation can be updated
without changing the recorded method.
