# TotalSegmentator CUDA 12.6 Runtime

This image installs the pinned `external/TotalSegmentator` submodule into the
project's validated CUDA 12.6 VoxTell environment. It exists because the
upstream TotalSegmentator 2.16.0 image bundles a CUDA 13 PyTorch build that
cannot initialize on this host's CUDA 12.8-capable NVIDIA driver.

Build from the repository root:

```bash
docker build \
  -f docker/totalsegmentator/Dockerfile \
  -t rexgroundingct-totalsegmentator:2.16.0-cu126 \
  .
```

The image reuses the base image's PyTorch, nnU-Net v2, and medical imaging
dependencies. TotalSegmentator itself is installed with `--no-deps` so pip
cannot replace the known-working CUDA PyTorch build. The three import-time
DICOM packages are pinned to the versions in the upstream 2.16.0 image even
though this experiment supplies NIfTI inputs.

Persistent weights must be mounted from:

```text
/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0
```

Set:

```bash
TOTALSEG_WEIGHTS_PATH=/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0/nnunet/results
```

and use a separate small `TOTALSEG_HOME_DIR` per concurrent worker to avoid
configuration-file write races.
