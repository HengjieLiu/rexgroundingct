# Detached submission runner launch

Launched 2026-09-09T19:12:07.458600+00:00 on gpu8.

Container: `sideexp003_s001_top4_test300_d123`. Job SHA256: `1b981c0420b308ecde216eba6feef6d200986e83fb7f9a9636e2febae1765e6e`.

The CPU-only supervisor waits for all four strict Wave 1 caches and coordinator exit. Ranks 5–20 remain held. The container cannot access challenge segmentation labels, GPUs, network or the Docker socket. Source and input mounts are read-only; only its runtime and Exp024 output parent are writable.

Reproduction command (requires the recorded container name to be unused and no live job lock):

```bash
docker run -d --init --name sideexp003_s001_top4_test300_d123 --network none --pid host --cpus 4 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,size=2g,mode=1777 --user 1009:1009 --label rex.submission_job=1b981c0420b308ecde216eba6feef6d200986e83fb7f9a9636e2febae1765e6e -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -v /home/hengjie/code_sync/rexgroundingct:/home/hengjie/code_sync/rexgroundingct:ro -v /mnt/shengdata1:/mnt/shengdata1:ro -v /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123:/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123:rw -v /mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs:/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs:rw -v /data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json:/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json:ro -w /home/hengjie/code_sync/rexgroundingct sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f python /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_test300.py run --job /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/job_spec.json
```

Watch the live state:

```bash
docker exec sideexp003_s001_top4_test300_d123 python /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_test300.py watch --job /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/job_spec.json --once

docker logs --tail 10 sideexp003_s001_top4_test300_d123
```

After container exit, inspect `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/state.json` and `completion.json`. Output generation and final validation remain pending until the dependency gate opens.
