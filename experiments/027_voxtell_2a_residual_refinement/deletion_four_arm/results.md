# Four-arm deletion stage 1 results

Status: **pending_user_review**. All arms completed 2,000 updates (20 epochs) and evaluations at epochs 1, 5, 10 and 20. The coordinator stopped successfully at 2026-09-10 20:08:43 UTC (13:08:43 Pacific). No continuation or checkpoint selection has been made.

Total orchestration time was 10,463.3 seconds (2 h 54 min), including 705.0 seconds of CPU input preparation. All training losses and gradients were finite. The container exited with code 0; all four GPUs were idle at the completion audit.

The post-run audit verified exactly 2,000 consecutive updates, 21 retained checkpoints (including initialization), and four complete 69-finding evaluations with matching checkpoint hashes for every arm.

## Epoch 20 at the fixed removal threshold 0.90

Rows remain in run order. A has 35 findings; B has 34. B is gradient-held-out for runs 1–3, and in-sample for run 4. A is development data. Full results mix exposure for runs 2–3; run 4 measures empirical fitting capacity.

| Run | A Dice | B Dice | Full Dice | Full change | TP retained | FP removed | Hits | Findings below 95% / 80% TP retention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cached base | 0.339416 | 0.334950 | 0.337215 | 0 | 100% | 0% | 59/69 | 0 / 0 |
| run1_train | 0.340546 | 0.336894 | 0.338747 | +0.001531 | 99.9901% | 0.5359% | 59/69 | 0 / 0 |
| run2_train_val_a | 0.339418 | 0.334951 | 0.337217 | +0.000001 | 100.0000% | 0.0005% | 59/69 | 0 / 0 |
| run3_val_a | 0.340498 | 0.335697 | 0.338133 | +0.000917 | 99.9577% | 0.4957% | 59/69 | 0 / 0 |
| run4_val_all | 0.340014 | 0.341592 | 0.340792 | +0.003576 | 99.8700% | 1.2379% | 58/69 | 3 / 1 |

TP retention and FP removal above are pooled voxel ratios. High pooled retention can conceal damage to a small finding. Run 4 loses one hit: `train_19279_c_1.nii.gz::2` in A retains 54.44% of its base TP, with Dice falling from 0.124412 to 0.076467. The other two run-4 retention flags are `train_19180_a_2.nii.gz::1` (84.62%) and `train_19341_a_2.nii.gz::1` (94.07%).

At this fixed threshold the edits are small: run 2 removes only five FP voxels across the full cohort. These are descriptive results, not a ranking. [All saved thresholds](all_thresholds.md) now tabulates all four checkpoints, all six removal thresholds and six base-logit controls, with A/B/full metrics, per-finding harm counts, CSV/JSON exports and a comparison figure. Threshold and checkpoint selection remain with the user.

## Training-loss comparison

[One-figure comparison](../runtime/deletion_four_arm_20ep/reports/loss_comparison/four_run_training_loss.png) overlays all four runs: trailing 100-update means above and all raw step losses below. [PDF](../runtime/deletion_four_arm_20ep/reports/loss_comparison/four_run_training_loss.pdf) and [unrounded curve CSV](../runtime/deletion_four_arm_20ep/reports/loss_comparison/loss_curves.csv) are retained externally, together with the source-history hashes and plotting script.

All curves drop sharply during roughly the first 300 updates, then improve more slowly with continued fluctuations. Epoch-1 to epoch-20 mean losses are 2.1518 → 1.5756 / 2.0904 → 1.6809 / 2.1835 → 1.5325 / 2.1923 → 1.6123 in run order. Raw step spikes remain finite (maximum 10.56); the figure retains their full range. The loss is FP-removal BCE + 2 × TP-preservation BCE, independent of the inference threshold. The arms sample different training pools.

## All completed primary-threshold evaluations

| Run | Epoch | A Dice | B Dice | Full Dice | TP retained | FP removed |
| --- | --- | --- | --- | --- | --- | --- |
| run1_train | 1 | 0.334832 | 0.329104 | 0.332010 | 99.4016% | 0.2535% |
| run1_train | 5 | 0.340366 | 0.336074 | 0.338251 | 99.9932% | 0.2928% |
| run1_train | 10 | 0.339514 | 0.335046 | 0.337312 | 100.0000% | 0.0356% |
| run1_train | 20 | 0.340546 | 0.336894 | 0.338747 | 99.9901% | 0.5359% |
| run2_train_val_a | 1 | 0.339416 | 0.334829 | 0.337156 | 99.9963% | 0.0000% |
| run2_train_val_a | 5 | 0.339530 | 0.334960 | 0.337278 | 100.0000% | 0.0187% |
| run2_train_val_a | 10 | 0.339636 | 0.335113 | 0.337407 | 99.9940% | 0.0441% |
| run2_train_val_a | 20 | 0.339418 | 0.334951 | 0.337217 | 100.0000% | 0.0005% |
| run3_val_a | 1 | 0.339416 | 0.334950 | 0.337215 | 100.0000% | 0.0000% |
| run3_val_a | 5 | 0.340221 | 0.335611 | 0.337949 | 99.9540% | 0.2543% |
| run3_val_a | 10 | 0.340560 | 0.336845 | 0.338729 | 99.9076% | 0.6034% |
| run3_val_a | 20 | 0.340498 | 0.335697 | 0.338133 | 99.9577% | 0.4957% |
| run4_val_all | 1 | 0.339416 | 0.334950 | 0.337215 | 100.0000% | 0.0000% |
| run4_val_all | 5 | 0.339293 | 0.337534 | 0.338426 | 99.9527% | 0.3779% |
| run4_val_all | 10 | 0.339556 | 0.341874 | 0.340698 | 99.8433% | 1.0308% |
| run4_val_all | 20 | 0.340014 | 0.341592 | 0.340792 | 99.8700% | 1.2379% |

[Final dashboard](../runtime/deletion_four_arm_20ep/reports/live_dashboard.md) contains the curves, per-finding retention flags and links to the underlying records. [Verification](verification.md) records implementation and launch checks.
