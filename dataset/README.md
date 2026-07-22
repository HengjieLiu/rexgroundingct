---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Download ReXGroundingCT from Hugging Face

This guide downloads the gated Hugging Face dataset
`rajpurkarlab/ReXGroundingCT` into:

```bash
/data/hengjie/datasets/rexgroundingct
```

Local project sources used for this guide:

- [`../challenge_info/huggingface_dataset_card.md`](../challenge_info/huggingface_dataset_card.md)
- [`../challenge_info/github_dataset_generation.md`](../challenge_info/github_dataset_generation.md)
- [`../challenge_info/SUMMARY.md`](../challenge_info/SUMMARY.md)
- [`../challenge_info/rexrankct_submission_guideline.md`](../challenge_info/rexrankct_submission_guideline.md)

Hugging Face command references checked while writing this guide:

- [Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Download files from the Hub](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Downloading datasets](https://huggingface.co/docs/hub/datasets-downloading)
- [Using Xet Storage](https://huggingface.co/docs/hub/xet/using-xet-storage)

The local archive says the dataset is hosted at:

```text
https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT
```

It also records that access is gated: you must log in to Hugging Face and agree
to share contact information before the files can be downloaded. The archived
dataset card reports a total file size of about `3.57 GB`, but leave extra disk
space for temporary files and Hugging Face/Xet metadata.

The archived Hugging Face card lists the license as `cc-by-nc-sa-4.0`; make
sure your intended use follows the dataset terms you accept on Hugging Face.

## 1. Confirm the target location has enough space

```bash
df -h /data/hengjie/datasets
mkdir -p /data/hengjie/datasets/rexgroundingct
```

Recommended free space: at least `10 GB`, even though the archived dataset card
lists the dataset itself as about `3.57 GB`.

## 2. Install the Hugging Face CLI

Use a small virtual environment so the download tooling is isolated from the
rest of the system:

```bash
python3 -m venv ~/.venvs/hf-hub
source ~/.venvs/hf-hub/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub hf_xet
hf --help
```

Notes:

- The modern Hugging Face CLI command is `hf`.
- `hf_xet` is useful because the ReXGroundingCT Hugging Face page is marked as
  Xet-backed in the local archive.
- If your system already has `hf`, you can still use the existing install; just
  run `hf --help` and `hf auth whoami` to confirm it works.
- In any fresh terminal, reactivate this environment before running the later
  commands:

```bash
source ~/.venvs/hf-hub/bin/activate
```

## 3. Accept the gated dataset terms in a browser

Open the dataset page:

```text
https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT
```

Then:

1. Log in to your Hugging Face account.
2. Review the dataset conditions.
3. Accept the request to share your contact information.
4. Wait until the page shows that you have access to the files.

Do this before running the download command. If the terms are not accepted, the
CLI usually fails with a `401`, `403`, or gated-repository access error.

## 4. Log in from the terminal

Keep Hugging Face auth/cache files under the dataset area so large-transfer
metadata does not silently go into your home directory:

```bash
source ~/.venvs/hf-hub/bin/activate
export REXGROUNDINGCT_DIR=/data/hengjie/datasets/rexgroundingct
export HF_HOME="$REXGROUNDINGCT_DIR/.hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
mkdir -p "$REXGROUNDINGCT_DIR" "$HF_HOME"
chmod 700 "$HF_HOME"
```

Interactive browser login:

```bash
hf auth login
hf auth whoami
```

Headless/token login:

```bash
read -rsp "HF token: " HF_TOKEN
echo
hf auth login --token "$HF_TOKEN"
unset HF_TOKEN
hf auth whoami
```

Use a Hugging Face user access token that can read gated repositories. Do not
paste tokens directly into shell history.

## 5. Preview the download

Run a dry run first. This verifies authentication, lists what would be
downloaded, and gives a current size estimate from Hugging Face:

```bash
export DATASET_REPO_ID=rajpurkarlab/ReXGroundingCT
export REXGROUNDINGCT_DIR=/data/hengjie/datasets/rexgroundingct
export HF_HOME="$REXGROUNDINGCT_DIR/.hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"

hf download "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --dry-run
```

If this fails, fix authentication/access before continuing.

## 6. Optional: download metadata first

This step is useful if you want to inspect the JSON schema and evaluation code
before pulling every binary file:

```bash
hf download "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --include "*.json" \
  --include "*.md" \
  --include "*.py"
```

Expected important files, based on the local archive:

- `dataset.json`
- `MICCAI_challenge_dataset.json`
- `reports_dataset.json`
- `MLHC_dataset_version.json`
- `rexrank_eval.py`

The upstream repository may add, remove, or rename files, so treat this list as
an expected checkpoint rather than a permanent contract.

## 7. Download the full dataset

```bash
hf download "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --max-workers 8
```

If the download is interrupted, rerun the same command. The `--local-dir`
metadata under:

```text
/data/hengjie/datasets/rexgroundingct/.cache/huggingface
```

helps the CLI avoid re-downloading files that are already current.

## 8. Verify the local files

Check size and list the top-level contents:

```bash
du -sh "$REXGROUNDINGCT_DIR"
find "$REXGROUNDINGCT_DIR" -maxdepth 2 -type f \
  | sed "s#^$REXGROUNDINGCT_DIR/##" \
  | sort \
  | head -200
```

Check the expected project files:

```bash
for f in \
  dataset.json \
  MICCAI_challenge_dataset.json \
  reports_dataset.json \
  MLHC_dataset_version.json \
  rexrank_eval.py
do
  if [ -f "$REXGROUNDINGCT_DIR/$f" ]; then
    echo "OK: $f"
  else
    echo "MISSING: $f"
  fi
done
```

Count likely CT/mask payload files:

```bash
find "$REXGROUNDINGCT_DIR" -type f \
  \( -name "*.nii" -o -name "*.nii.gz" -o -name "*.npy" -o -name "*.npz" \) \
  | wc -l
```

If your installed `hf` supports cache verification, run:

```bash
hf cache verify "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --fail-on-missing-files
```

## 9. Inspect the JSON split files

Use this quick Python check to summarize whatever JSON files are present:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/data/hengjie/datasets/rexgroundingct")
for name in [
    "dataset.json",
    "MICCAI_challenge_dataset.json",
    "reports_dataset.json",
    "MLHC_dataset_version.json",
]:
    path = root / name
    if not path.exists():
        print(f"{name}: missing")
        continue

    data = json.loads(path.read_text())
    print(f"\n{name}")
    print(f"  type: {type(data).__name__}")

    if isinstance(data, dict):
        keys = list(data)
        print(f"  top-level keys: {keys[:20]}")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} records")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} entries")
    elif isinstance(data, list):
        print(f"  records: {len(data)}")
        if data and isinstance(data[0], dict):
            print(f"  first record keys: {list(data[0])[:20]}")
PY
```

From the archived dataset card, each dataset item includes fields such as
`name`, `findings`, `entity_counts`, `shape`, `pixels`, `categories`, and
`protocol`. Segmentation masks are described as having shape `(F, H, W, D)`,
where `F` is the number of findings for a scan and `H, W, D` match the CT
volume shape.

## 10. Python fallback instead of CLI

If you prefer Python, or if the CLI wrapper behaves differently on your system:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rajpurkarlab/ReXGroundingCT",
    repo_type="dataset",
    local_dir="/data/hengjie/datasets/rexgroundingct",
)
PY
```

To preview the file list after authentication:

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(
    repo_id="rajpurkarlab/ReXGroundingCT",
    repo_type="dataset",
)
for file in files:
    print(file)
PY
```

## 11. Common problems

### `401`, `403`, or gated access error

Fixes:

```bash
hf auth whoami
```

- Make sure you accepted the terms at
  `https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT`.
- Make sure the terminal is using the same account/token that accepted access.
- If needed, rerun `hf auth login --force`.

### `Repository not found`

Use the dataset repo type explicitly:

```bash
hf download rajpurkarlab/ReXGroundingCT --repo-type dataset --dry-run
```

Without `--repo-type dataset`, the CLI may look for a model repository instead.

### Files downloaded to `~/.cache/huggingface` instead of `/data`

Use `--local-dir "$REXGROUNDINGCT_DIR"` for a real local copy under
`/data/hengjie/datasets/rexgroundingct`. If you only use `--cache-dir`, the
dataset remains in the Hugging Face cache layout.

### Slow network or timeout

Increase the Hugging Face download timeout and rerun the same command:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=60
hf download "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --max-workers 4
```

### Verification fails after an interrupted download

First rerun the normal full download command. If verification still fails, force
refresh the local files:

```bash
hf download "$DATASET_REPO_ID" \
  --repo-type dataset \
  --local-dir "$REXGROUNDINGCT_DIR" \
  --force-download
```

### Need the older CLI name

Some older installations expose `huggingface-cli` instead of `hf`. The equivalent
full download command is:

```bash
huggingface-cli download rajpurkarlab/ReXGroundingCT \
  --repo-type dataset \
  --local-dir /data/hengjie/datasets/rexgroundingct
```

Prefer `hf` for new installs.

## 12. Download the matching CT-RATE CT volumes

ReXGroundingCT provides the finding metadata and segmentation masks, but the CT
volumes themselves come from the larger gated CT-RATE dataset:

```text
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
```

Accept the CT-RATE terms separately in your browser before downloading. The
ReXGroundingCT access approval does not automatically grant CT-RATE file access.

Use a different local root for the CT volumes:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct
```

Reason for the path difference: `/data/hengjie/datasets/rexgroundingct` already
contains the smaller ReXGroundingCT metadata and segmentation masks, which are
about `3.4 GB`. The matching CT-RATE CT subset is much larger: about
`336.78 GiB` for `train + val` or `369.15 GiB` for `train + val + test`, so it
should live on the larger `/mnt/shengdata1` mount.

There is also a source-layout difference. ReXGroundingCT segmentation masks are
stored locally as:

```text
/data/hengjie/datasets/rexgroundingct/segmentations/<filename>
```

The matching CT volumes are stored in CT-RATE's nested `*_fixed` layout under:

```text
/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct/dataset/<split>_fixed/...
```

CT-RATE organizes volume files as:

```text
dataset/<split>_fixed/<split>_<patient_id>/<split>_<patient_id>_<scan_id>/<filename>
```

For example:

```text
train_1741_b_2.nii.gz
```

maps to:

```text
dataset/train_fixed/train_1741/train_1741_b/train_1741_b_2.nii.gz
```

Use the helper script in this folder to download only the files referenced by
`MICCAI_challenge_dataset.json`:

```bash
source ~/.venvs/hf-hub/bin/activate

export HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"

df -h /mnt/shengdata1
mkdir -p /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct

python /home/hengjie/code_sync/rexgroundingct/dataset/download_ct_rate_challenge_subset.py \
  --output-dir /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct
```

By default this downloads the MICCAI challenge `train`, `val`, and `test`
volumes: `3,492` CT files. If you only want files that have released
segmentation masks, use:

```bash
python /home/hengjie/code_sync/rexgroundingct/dataset/download_ct_rate_challenge_subset.py \
  --splits train val \
  --output-dir /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct
```

That downloads `3,192` CT files.

To list paths without downloading:

```bash
python /home/hengjie/code_sync/rexgroundingct/dataset/download_ct_rate_challenge_subset.py \
  --list-only \
  --manifest /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct/ct_rate_rexgroundingct_paths.txt
```

## 13. Check CT readiness for VoxTell experiments

Use the experiment poller to verify whether the downloaded CT files are complete
for the next stage:

```bash
cd /home/hengjie/code_sync/rexgroundingct

python scripts/rexgroundingct/poll_ct_subset.py --splits val
python scripts/rexgroundingct/poll_ct_subset.py --splits train val
python scripts/rexgroundingct/poll_ct_subset.py --splits train val test
```

Expected complete counts:

- `val`: `200` CT files, needed for pretrained VoxTell validation evaluation.
- `train val`: `3,192` CT files, needed for fine-tuning and validation.
- `train val test`: `3,492` CT files, the full MICCAI challenge subset.

The poller returns exit code `0` only when all requested files are present and
there are no Hugging Face `.incomplete` files under the CT root. The VoxTell
experiment launchers use the same readiness condition.
