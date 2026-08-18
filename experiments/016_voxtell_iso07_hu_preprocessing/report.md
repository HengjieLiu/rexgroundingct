# Experiment 016 Preprocessing Smoke Report

Status: `preprocessing_complete`

The new cache variant `crop_clip1024_linear_iso07_v1` was smoke-tested in the
VoxTell Docker environment on one train, one validation, and one test case.
The final smoke cache was written to
`/tmp/rex016_iso07_smoke_cache_per_split`; the standard cache root on
`/mnt/shengdata1` was not populated.

## Smoke Command

```bash
docker run --rm \
  -v /home/hengjie/code_sync/rexgroundingct:/workspace \
  -v /data:/data:ro \
  -v /mnt/shengdata1:/mnt/shengdata1:ro \
  -v /tmp:/tmp \
  rexgroundingct-voxtell:cu126 \
  python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
    --preprocess-id crop_clip1024_linear_iso07_v1 \
    --splits train val test \
    --max-cases-per-split 1 \
    --cache-root /tmp/rex016_iso07_smoke_cache_per_split \
    --num-workers 1 \
    --estimate-cases 3 \
    --storage-buffer-fraction 0.0 \
    --overwrite
```

## Result

| Check | Result |
| --- | --- |
| Cases | `3` |
| Complete case markers | `3` |
| Splits | train `1`, val `1`, test `1` |
| Empty train/val targets | `0` |
| Foreground fallback targets | `0` |
| HU header failures | `0` |
| Normalization failures | `0` |
| Target spacing | `[0.7, 0.7, 0.7]` |
| Image padding | `-1.0` |
| Target padding | `0` |
| Test targets file | omitted as expected |

Case-level resampled shapes and ranges:

| Case | Split | Targets | Shape ZYX | Resampled min/max |
| --- | --- | --- | --- | --- |
| `train_1741_b_2.nii.gz` | train | yes | `[425, 463, 463]` | `[-1.0, 1.0]` |
| `train_13082_a_1.nii.gz` | val | yes | `[439, 420, 420]` | `[-1.0, 1.0]` |
| `train_13195_a_1.nii.gz` | test | no | `[433, 500, 500]` | `[-1.0, 1.0]` |

## Next Command

Full cache build, after the successful fixed multiprocessing calibration:

```bash
docker run --rm \
  -v /home/hengjie/code_sync/rexgroundingct:/workspace \
  -v /data:/data:ro \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  rexgroundingct-voxtell:cu126 \
  python /workspace/scripts/rexgroundingct/prepare_voxtell_preprocessed_cache.py \
    --preprocess-id crop_clip1024_linear_iso07_v1 \
    --splits train val test \
    --cache-root /mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1 \
    --num-workers 16 \
    --multiprocessing-start-method spawn \
    --worker-threads 1 \
    --estimate-cases 30 \
    --overwrite
```

Run under a log path in the Exp016 runtime directory, as with calibration.

## Calibration Result

The 45-case mixed calibration completed on `2026-08-14`. The fastest validated
fixed parallel 16-worker output is under:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/016_voxtell_iso07_hu_preprocessing/calibration/crop_clip1024_linear_iso07_v1_45cases_spawn_w16
```

Earlier fork-style `8`-worker and `2`-worker attempts were stopped because they
made no case-level writes after several minutes in the build phase. The fix is
to use `spawn` and limit torch/OpenMP threads per worker.

| Check | Result |
| --- | --- |
| Cases | `45` |
| Split mix | train `15`, val `15`, test `15` |
| Complete case markers | `45` |
| Targets | `89` |
| Empty targets | `0` |
| Foreground fallback targets | `0` |
| Geometry/target mismatches | `0` |
| HU header failures | `0` |
| Normalization failures | `0` |
| Target spacing | `[0.7, 0.7, 0.7]` |
| Shape min ZYX | `[360, 383, 383]` |
| Shape max ZYX | `[569, 704, 704]` |
| On-disk size | `21G` |
| Uncompressed image bytes | `20.940 GiB` |
| Uncompressed target bytes | `10.818 GiB` |
| Storage estimate for this 45-case subset | `26.418 GiB` |

Timing:

- storage estimation: `1:25`;
- serial cache build: `14:03`;
- fixed `spawn`/4-worker cache build: `4:26`;
- fixed `spawn`/8-worker cache build: `2:15`;
- fixed `spawn`/16-worker cache build: `1:14`;
- content hash matched serial:
  `a28d2e49aaa4eca09e5d7fcc75cc4e189df8c4fec1d9e95dd023221f01044b8e`;
- target hash matched serial:
  `e8153d3ec5e3782de4e8f084d2397ca895db2f46205d6059b082b32a1f6cec61`.

Implication for the full `3492`-case all-split cache:

- fixed `spawn`/16-worker build extrapolates to roughly `1.6 h` for
  preprocessing/writes, plus storage estimation and final audit;
- practical full-run ETA should be treated as `3-6 h`;
- use `--multiprocessing-start-method spawn --worker-threads 1` for the full
  run.

## Full Build Launch

The full all-split cache build was launched on `2026-08-14` in detached Docker
container:

```text
rex016_iso07_full_w16_build_20260814T0925Z
```

Runtime log:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/016_voxtell_iso07_hu_preprocessing/logs/full_cache_spawn_w16_build_20260814T0925Z.log
```

Standard cache root:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1
```

Launch settings:

```text
--num-workers 16 --multiprocessing-start-method spawn --worker-threads 1 --skip-storage-estimate
```

The pre-build storage-estimation pass was skipped for the full launch because
it spent several minutes before the first progress tick on the full split
selection. The calibration outputs and `/mnt/shengdata1` free-space check were
used instead. The root manifest and `.complete` marker will still be written
only after the full case build and audit pass.

## Full Build Result

The full cache completed successfully. The Docker container exited with status
`0`, and the standard cache root has both `manifest.json` and `.complete`.

| Check | Result |
| --- | --- |
| Cases | `3492` |
| Complete case markers | `3492` |
| Split mix | train `2992`, val `200`, test `300` |
| Targets | `8068` |
| Empty targets | `0` |
| Foreground fallback targets | `0` |
| Geometry/target mismatches | `0` |
| HU header failures | `0` |
| Normalization failures | `0` |
| Target spacing | `[0.7, 0.7, 0.7]` |
| Shape min ZYX | `[255, 347, 347]` |
| Shape max ZYX | `[1077, 714, 714]` |
| Image padding | `-1.0` |
| Target padding | `0` |
| On-disk size | `1.7T` |
| Uncompressed image bytes | `1859577464504` |
| Uncompressed target bytes | `1078293677734` |
| Workers | `16` |
| Multiprocessing | `spawn`, `worker_threads=1` |
| Storage estimate | skipped for full launch |

Content hashes:

- Case content index:
  `162a079364b92f07a7c8fbf923c467abc141eb7d1170e2482527be79db53b4d4`.
- Target index:
  `fd01d10bd0dcce5eea66f91faa36563caccba23f6afd03c380e8bccc40fa14b0`.
