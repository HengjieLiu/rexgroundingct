# Experiment 004 Execution Specification

## Question

Does continuing the exp003 v123 epoch-100 model with a 384 mm physical field
of view improve ReXGroundingCT localization and segmentation compared with an
otherwise paired native-resolution continuation?

## Paired Arms

| Arm | Training GPU | Spatial preprocessing |
| --- | ---: | --- |
| `native192_cont100` | 2 | Original VoxTell crop-to-nonzero and full cropped-volume z-score, then native-resolution `192^3` positive crops |
| `iso2mm_global192_cont100` | 3 | The same crop and z-score, followed by 2 mm isotropic resampling and `192^3` positive crops |

Both arms load network weights only from exp003
`v123_opt_poscrop_emptyloss` epoch 100. Optimizer state, momentum, LR
scheduler, AMP scaler, and update count are reset.

## 2 mm Geometry Contract

1. Read CTs through `NibabelIOWithReorient`.
2. Convert released masks from raw `F,X,Y,Z` to VoxTell `F,Z,Y,X`.
3. Crop the image to nonzero and apply the same crop to every target.
4. Z-score the complete cropped native image once.
5. Resample the normalized image to 2 mm with trilinear interpolation,
   `align_corners=False`, and no antialiasing.
6. Resample masks with nearest-exact interpolation.
7. If a nonempty mask disappears, map source foreground voxel centers into
   the 2 mm grid and record the fallback.
8. Model padding uses image value `0.0` and mask value `0`.
9. Validation uses overlapping windows and restores probabilities to the
   native cropped grid before undoing the crop and orientation.

The complete cache audit must report `8,068` nonempty train/validation targets.

## Reproducibility

- Seed: `20260723`.
- Generate the deterministic v123 stream through event 19,999.
- Verify events 0-9,999 exactly reproduce the exp003 schedule.
- Use events 10,000-19,999 for both exp004 arms.
- Pair case names, prompt slots, positive/negative choices, and patch seeds.
- Recompute only spatial points and starts in the 2 mm FZYX geometry.
- Fixed validation sets are passed by JSON path; `--limit` is forbidden.

## Optimization

- 100 epochs, 100 optimizer updates per epoch, batch 1, accumulation 1.
- Single GPU, no DDP, no added augmentation.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- Encoder LR `1e-7`, other parameters LR `1e-6`.
- 100-update warmup, polynomial decay power `0.9` over 10,000 updates.
- Gradient norm clipping at `12`.
- v123 empty-target and weighted-BCE loss.
- Deep supervision weights `[1, 1/2, 1/4, 1/8, 1/16]`, normalized.

## Checkpoint And Evaluation Ownership

Training processes only train and save checkpoints. Immutable checkpoints are
written atomically at epochs 20, 40, 60, 80, and 100. A separate rolling
recovery checkpoint is updated every 500 optimizer updates.

A coordinator polls every 60 seconds and owns eval locks:

- Native val20 uses GPU 0.
- 2 mm val20 uses GPU 1.
- Val20 runs immediately at epochs 20, 40, 60, 80, and 100.
- After both training arms exit, val200 runs concurrently and saves threshold
  0.5 masks plus float32 probabilities.
- Evaluation directories are complete only when prediction counts and summary
  case/finding counts match the fixed dataset JSON.

### Early 2 mm Val200 Amendment

The 2 mm arm completed substantially before the native control. Its epoch-100
val200 is therefore launched immediately by
`scripts/rexgroundingct/run_004_iso2mm_val200_now.sh` on GPUs 1 and 3.

This sidecar writes the same canonical `eval_epoch100_val200` directory and
holds the same `.eval.lock`. Completion requires all 200 masks, all 200
probability files, the evaluator JSON, and matching case/finding counts. When
the original coordinator reaches its post-training finalization, its existing
`eval_complete` check must skip this already-complete 2 mm evaluation. It will
still run the native val200 and generate the paired comparison report. The
baseline trainer on GPU 2 is not signaled, restarted, or modified.

The early evaluation completed on 2026-07-24 with all 200 masks and probability
maps. At threshold 0.5 it obtained mean Dice per finding `0.171086`, hit rate
`0.419948`, and `160/381` finding hits. The shared lock was released only after
the evaluator JSON and summary were complete.

A second fresh-container sidecar,
`scripts/rexgroundingct/run_004_native_val200_after_train.sh`, waits without
allocating a GPU for the native epoch-100 checkpoint and `.train_complete`
marker. It then runs the canonical native val200 on GPUs 2 and 3, after the
trainer has released GPU 2, and generates the paired comparison report. The
same lock and completeness checks make both the sidecar and original
coordinator safe to rerun or skip.

## Required Report

The final report compares both arms and the original exp003 v123 epoch-100
baseline by Dice, hit rate, finding category, lesion size, and sparse/small
target strata. Threshold 0.5 is primary; any threshold sweep is supplementary.
