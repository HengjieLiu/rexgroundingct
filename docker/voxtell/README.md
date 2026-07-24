# VoxTell Docker Environment

Build from the repository root:

```bash
docker build -f docker/voxtell/Dockerfile -t rexgroundingct-voxtell:cu126 .
```

Run an interactive container:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  -v /home/hengjie/code_sync/rexgroundingct:/workspace \
  -v /data/hengjie:/data/hengjie \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e HOME=/tmp \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
  rexgroundingct-voxtell:cu126
```

Sanity checks inside the container:

```bash
python3 - <<'PY'
import torch
print(torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

python3 -c "import voxtell; print(voxtell.__file__)"
voxtell-predict --help
```

Run JupyterLab for repo notebooks:

```bash
docker run --rm -it \
  --ipc=host \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  -p 8888:8888 \
  -v /home/hengjie/code_sync/rexgroundingct:/workspace \
  -v /data/hengjie:/data/hengjie \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e HOME=/tmp \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
  rexgroundingct-voxtell:cu126 \
  bash -lc "cd /workspace && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
```

The visualization notebooks do not require GPU access. Add `--gpus all` to the
Jupyter command only when testing GPU code from the same container.

Do not bake Hugging Face tokens into the image. Use the mounted `HF_HOME` above
or pass `HF_TOKEN` at runtime.

The `--user "$(id -u):$(id -g)"` flag is required on `/mnt/shengdata1` because
the mount behaves like an NFS filesystem with root-squash; a root process inside
the container cannot write to experiment directories owned by the host user.
