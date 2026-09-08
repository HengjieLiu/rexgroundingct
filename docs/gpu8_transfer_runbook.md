---
created: 2026-09-08
updated: 2026-09-08
status: active
---

# GPU9 To GPU8 Transfer Runbook

This runbook prepares GPU8 to run ReXGroundingCT jobs independently of GPU9.
It transfers:

1. the Git repository and submodules;
2. `/data/hengjie/datasets/rexgroundingct`;
3. the exact `rexgroundingct-voxtell:cu126` Docker image.

The large data and runtime trees under `/mnt/shengdata1` are not copied. They
are verified as shared and usable.

This is not a multi-node DDP setup. GPU8 and GPU9 must use different run/output
directories when they execute jobs concurrently.

## 1. Define The Hosts And Paths On GPU8

Run these commands on GPU8 and keep the same terminal open:

```bash
export GPU9_SSH="hengjie@10.72.22.74"
export GPU9_REPO="/home/hengjie/code_sync/rexgroundingct"
export GPU8_REPO="/home/hengjie/code_sync/rexgroundingct"
export REX_DATA_ROOT="/data/hengjie/datasets/rexgroundingct"
export SHARED_ROOT="/mnt/shengdata1/hengjie"
export TRANSFER_ROOT="$SHARED_ROOT/transfers/rexgroundingct_gpu8"
export VOXTELL_IMAGE="rexgroundingct-voxtell:cu126"
```

If the repository will live at another location on GPU8, change `GPU8_REPO`.
Pass `REPO_ROOT="$GPU8_REPO"` to the project's Docker launchers.

## 2. Check GPU8

Run on GPU8:

```bash
hostname
id
nvidia-smi
docker version
docker info | grep -i runtime
df -h /data /mnt/shengdata1
ssh "$GPU9_SSH" hostname
mkdir -p "$TRANSFER_ROOT"
```

Confirm:

- SSH reaches GPU9 as the expected user.
- GPU8 exposes the expected GPUs.
- Docker and the NVIDIA container runtime work.
- `/data` has at least 20 GB free for the approximately 14 GB project tree.
- Your user can read `/mnt/shengdata1` and write below
  `/mnt/shengdata1/hengjie`.

If the CUDA base image is already cached on GPU8, this optional command tests
GPU access inside Docker:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 \
  nvidia-smi
```

Otherwise skip this command to avoid downloading the base image. GPU access is
tested with the transferred project image later.

## 3. Verify That `/mnt/shengdata1` Is Shared

Inspect the mount on both machines:

```bash
findmnt -T /mnt/shengdata1
ssh "$GPU9_SSH" 'findmnt -T /mnt/shengdata1'
```

Ask GPU9 to create a small probe on the shared mount:

```bash
export MOUNT_PROBE="$TRANSFER_ROOT/mount_probe_from_gpu9.txt"

ssh "$GPU9_SSH" \
  "date -u > '$MOUNT_PROBE' && hostname >> '$MOUNT_PROBE' && sha256sum '$MOUNT_PROBE'"

test -r "$MOUNT_PROBE"
cat "$MOUNT_PROBE"
sha256sum "$MOUNT_PROBE"
```

The file must be immediately visible on GPU8, and the two SHA-256 values must
match. If not, stop: the mounts are not shared.

Check the required shared paths on GPU8:

```bash
test -d "$SHARED_ROOT/datasets/rexgroundingct/ct"
test -d "$SHARED_ROOT/datasets/rexgroundingct/preprocessed"
test -d "$SHARED_ROOT/experiments/rexgroundingct"
test -w "$SHARED_ROOT/experiments/rexgroundingct"

find "$SHARED_ROOT/datasets/rexgroundingct/ct" \
  -type f -name '*.incomplete' -print -quit
```

The final command should print nothing.

After installing the code, run the stronger readiness check:

```bash
cd "$GPU8_REPO"
python scripts/rexgroundingct/poll_ct_subset.py --splits train val test
```

It should report 3,492 ready CT files. A train-and-validation-only check should
report 3,192 files:

```bash
python scripts/rexgroundingct/poll_ct_subset.py --splits train val
```

## 4. Transfer The Repository

### 4.1 Inspect GPU9

Run from GPU8:

```bash
ssh "$GPU9_SSH" \
  "cd '$GPU9_REPO' && git status --short --branch && git rev-parse HEAD && git submodule status --recursive"
```

A Git clone transfers only committed and pushed content. Modified and
untracked files shown here will not transfer. Decide whether each change
should be committed and pushed, intentionally left on GPU9, or treated as a
separate runtime artifact.

If required, run on GPU9 after committing the intended work:

```bash
cd /home/hengjie/code_sync/rexgroundingct
git status --short --branch
git push origin main
```

### 4.2 Clone Or Update On GPU8

For a new GPU8 checkout:

```bash
mkdir -p "$(dirname "$GPU8_REPO")"
git clone --recurse-submodules \
  git@github.com:HengjieLiu/rexgroundingct.git \
  "$GPU8_REPO"
```

If the checkout already exists, inspect it and then update it:

```bash
cd "$GPU8_REPO"
git status --short --branch
git fetch origin
git switch main
git pull --ff-only
git submodule update --init --recursive
```

Do not update over uncommitted GPU8 work until you decide how that work should
be preserved.

### 4.3 Verify Code And Submodules

Run on GPU8:

```bash
export GPU9_COMMIT="$(ssh "$GPU9_SSH" "cd '$GPU9_REPO' && git rev-parse HEAD")"
export GPU8_COMMIT="$(git -C "$GPU8_REPO" rev-parse HEAD)"

printf 'GPU9 commit: %s\n' "$GPU9_COMMIT"
printf 'GPU8 commit: %s\n' "$GPU8_COMMIT"
test "$GPU8_COMMIT" = "$GPU9_COMMIT"

git -C "$GPU8_REPO" status --short --branch
git -C "$GPU8_REPO" submodule status --recursive
ssh "$GPU9_SSH" "cd '$GPU9_REPO' && git submodule status --recursive"
```

The main commit hashes and both submodule commit hashes must match. A leading
`-`, `+`, or `U` in submodule status indicates an uninitialized, mismatched, or
conflicted submodule.

## 5. Transfer `/data/hengjie` Project Data

The project currently needs data under
`/data/hengjie/datasets/rexgroundingct` that is absent from the main `/mnt`
dataset root:

- `MICCAI_challenge_dataset.json` and related metadata;
- 3,192 ground-truth masks under `segmentations/`;
- the pretrained VoxTell model under `.hf_home/`;
- local evaluator scripts.

These commands transfer the project dataset root, not unrelated content under
`/data/hengjie`.

### 5.1 Check Size And Space

Run on GPU8:

```bash
ssh "$GPU9_SSH" "du -sh '$REX_DATA_ROOT'"
df -h /data
mkdir -p "$REX_DATA_ROOT"
```

### 5.2 Dry Run

Run on GPU8. The trailing slashes are intentional:

```bash
rsync -aHn --info=progress2 --stats \
  "$GPU9_SSH:$REX_DATA_ROOT/" \
  "$REX_DATA_ROOT/"
```

Review the proposed transfer. This does not delete extra files already on
GPU8.

### 5.3 Actual Transfer

Run on GPU8:

```bash
rsync -aH --partial --info=progress2 --stats \
  "$GPU9_SSH:$REX_DATA_ROOT/" \
  "$REX_DATA_ROOT/"
```

If SSH disconnects, run the same command again. Rsync will compare the trees
and continue transferring what is still needed.

The transferred `.hf_home` can contain Hugging Face authentication files.
Inspect permissions without printing their contents:

```bash
stat -c '%a %U:%G %n' "$REX_DATA_ROOT/.hf_home"
test ! -f "$REX_DATA_ROOT/.hf_home/token" || \
  stat -c '%a %U:%G %n' "$REX_DATA_ROOT/.hf_home/token"
test ! -f "$REX_DATA_ROOT/.hf_home/stored_tokens" || \
  stat -c '%a %U:%G %n' "$REX_DATA_ROOT/.hf_home/stored_tokens"
```

The cache directory and token files should not be group- or world-readable.

### 5.4 Verify The Transfer

Run a checksum-mode rsync dry run. It reads every file and may take several
minutes:

```bash
rsync -aHnc --itemize-changes --stats \
  "$GPU9_SSH:$REX_DATA_ROOT/" \
  "$REX_DATA_ROOT/"
```

There should be no file-change lines.

Check required files and the pretrained model:

```bash
test -r "$REX_DATA_ROOT/MICCAI_challenge_dataset.json"
test -d "$REX_DATA_ROOT/segmentations"
test -r "$REX_DATA_ROOT/rexrank_eval.py"
test -d "$REX_DATA_ROOT/.hf_home/hub/models--mrokuss--VoxTell"
```

Compare metadata hashes:

```bash
sha256sum "$REX_DATA_ROOT/MICCAI_challenge_dataset.json"
ssh "$GPU9_SSH" "sha256sum '$REX_DATA_ROOT/MICCAI_challenge_dataset.json'"
```

Compare segmentation counts:

```bash
find "$REX_DATA_ROOT/segmentations" \
  -maxdepth 1 -type f -name '*.nii.gz' | wc -l

ssh "$GPU9_SSH" \
  "find '$REX_DATA_ROOT/segmentations' -maxdepth 1 -type f -name '*.nii.gz' | wc -l"
```

Both should currently report 3,192. Compare overall sizes as a final sanity
check; filesystem block accounting can cause small differences:

```bash
du -sh "$REX_DATA_ROOT"
ssh "$GPU9_SSH" "du -sh '$REX_DATA_ROOT'"
```

## 6. Transfer The Exact Docker Image

The bind-mounted repository, `/data/hengjie`, and `/mnt/shengdata1` are not
inside the image. They must be transferred or mounted separately as described
above.

### 6.1 Save The Image On GPU9

Run on GPU9:

```bash
export TRANSFER_ROOT="/mnt/shengdata1/hengjie/transfers/rexgroundingct_gpu8"
export VOXTELL_IMAGE="rexgroundingct-voxtell:cu126"
export IMAGE_TAR="$TRANSFER_ROOT/rexgroundingct-voxtell-cu126.tar"

mkdir -p "$TRANSFER_ROOT"
docker image inspect "$VOXTELL_IMAGE" --format '{{.Id}} {{.Size}}'
test ! -e "$IMAGE_TAR"
docker save -o "$IMAGE_TAR" "$VOXTELL_IMAGE"
sha256sum "$IMAGE_TAR" > "$IMAGE_TAR.sha256"
ls -lh "$IMAGE_TAR" "$IMAGE_TAR.sha256"
```

If `test ! -e "$IMAGE_TAR"` fails, an archive already exists. Verify it is the
intended image rather than overwriting it silently.

### 6.2 Verify And Load On GPU8

Run on GPU8:

```bash
export GPU9_SSH="hengjie@10.72.22.74"
export TRANSFER_ROOT="/mnt/shengdata1/hengjie/transfers/rexgroundingct_gpu8"
export VOXTELL_IMAGE="rexgroundingct-voxtell:cu126"
export IMAGE_TAR="$TRANSFER_ROOT/rexgroundingct-voxtell-cu126.tar"

cd "$TRANSFER_ROOT"
sha256sum -c "$IMAGE_TAR.sha256"
docker load -i "$IMAGE_TAR"
docker image inspect "$VOXTELL_IMAGE" --format '{{.Id}} {{.Size}}'
```

Compare image IDs:

```bash
export GPU9_IMAGE_ID="$(ssh "$GPU9_SSH" "docker image inspect '$VOXTELL_IMAGE' --format '{{.Id}}'")"
export GPU8_IMAGE_ID="$(docker image inspect "$VOXTELL_IMAGE" --format '{{.Id}}')"

printf 'GPU9 image: %s\n' "$GPU9_IMAGE_ID"
printf 'GPU8 image: %s\n' "$GPU8_IMAGE_ID"
test "$GPU8_IMAGE_ID" = "$GPU9_IMAGE_ID"
```

## 7. End-To-End Container Test On GPU8

Run on GPU8:

```bash
export GPU8_REPO="/home/hengjie/code_sync/rexgroundingct"
export VOXTELL_IMAGE="rexgroundingct-voxtell:cu126"

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  -v "$GPU8_REPO:/workspace" \
  -v /data/hengjie:/data/hengjie \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e HOME=/tmp \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
  "$VOXTELL_IMAGE" \
  bash -lc '
    cd /workspace
    python3 -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    python3 -c "import voxtell; print(voxtell.__file__)"
    python3 -c "import json; p=\"/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json\"; d=json.load(open(p)); print(type(d).__name__)"
    test -d /data/hengjie/datasets/rexgroundingct/segmentations
    test -d /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct
    test -w /mnt/shengdata1/hengjie/experiments/rexgroundingct
    voxtell-predict --help >/dev/null
    python scripts/rexgroundingct/check_repo_workflow.py
  '
```

Every command must succeed. The output should show the expected PyTorch
version, all GPU8 GPUs, and the loaded VoxTell package.

## 8. Smoke-Test An Experiment

Before starting a long job:

1. Select the exact existing launcher and read its experiment README/spec.
2. Confirm whether it starts from public VoxTell or a checkpoint under `/mnt`.
3. Use its smallest supported smoke mode or case subset.
4. Confirm it reads a CT and, when required, a segmentation.
5. Confirm GPU activity with `nvidia-smi`.
6. Confirm logs and outputs are owned by your host user.
7. Stop if paths, permissions, checkpoint provenance, or GPU count differ from
   GPU9.

For an interactive project container:

```bash
cd "$GPU8_REPO"
REPO_ROOT="$GPU8_REPO" \
IMAGE="$VOXTELL_IMAGE" \
./docker/voxtell/run_dev.sh bash
```

## 9. Prevent Shared-Output Collisions

Never launch both machines into the same run directory. Create a unique GPU8
label:

```bash
export GPU8_RUN_LABEL="gpu8_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$GPU8_RUN_LABEL"
```

Pass it through the launcher's `RUN_GROUP`, run-name, or output-root option.
Inspect the selected launcher first because some scripts contain fixed paths.

Before every full launch, record:

```bash
hostname
git -C "$GPU8_REPO" status --short --branch
git -C "$GPU8_REPO" rev-parse HEAD
git -C "$GPU8_REPO" submodule status --recursive
docker image inspect "$VOXTELL_IMAGE" --format '{{.Id}}'
nvidia-smi
```

## Completion Checklist

- [ ] GPU8 SSH access to GPU9 works.
- [ ] GPU8 Docker can access every expected GPU.
- [ ] The `/mnt/shengdata1` probe written by GPU9 is visible on GPU8.
- [ ] The CT readiness check passes for the required splits.
- [ ] GPU8 and GPU9 repository commits match.
- [ ] GPU8 and GPU9 submodule commits match.
- [ ] The `/data/hengjie` checksum rsync reports no differences.
- [ ] Both hosts report 3,192 segmentation masks.
- [ ] The metadata SHA-256 values match.
- [ ] The pretrained VoxTell model exists on GPU8.
- [ ] Hugging Face authentication files have private permissions.
- [ ] The Docker archive checksum passes.
- [ ] GPU8 and GPU9 Docker image IDs match.
- [ ] The end-to-end container test succeeds.
- [ ] A small experiment smoke test succeeds.
- [ ] The GPU8 run has a unique output directory on shared `/mnt`.
