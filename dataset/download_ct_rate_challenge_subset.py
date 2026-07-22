#!/usr/bin/env python3
"""Download only the CT-RATE volumes referenced by ReXGroundingCT metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


CT_RATE_REPO_ID = "ibrahimhamamci/CT-RATE"
DEFAULT_OUTPUT_DIR = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")


def ct_rate_path(filename: str) -> str:
    if not filename.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz filename, got {filename!r}")

    stem = filename[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE filename format: {filename!r}")

    split = parts[0]
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    return f"dataset/{split}_fixed/{patient}/{scan}/{filename}"


def load_filenames(metadata_path: Path, splits: list[str]) -> list[str]:
    metadata = json.loads(metadata_path.read_text())
    filenames: list[str] = []
    for split in splits:
        if split not in metadata:
            raise KeyError(f"Split {split!r} not found in {metadata_path}")
        filenames.extend(item["name"] for item in metadata[split])
    return sorted(set(filenames))


def write_manifest(paths: list[str], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(paths) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CT-RATE fixed CT volumes used by ReXGroundingCT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json"),
        help="ReXGroundingCT JSON metadata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for CT-RATE subset files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="ReXGroundingCT splits to download. Use train val for released masks only.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write the CT-RATE file list.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print/write paths; do not download.",
    )
    args = parser.parse_args()

    filenames = load_filenames(args.metadata, args.splits)
    paths = [ct_rate_path(name) for name in filenames]

    manifest = args.manifest or args.output_dir / "ct_rate_rexgroundingct_paths.txt"
    write_manifest(paths, manifest)
    print(f"Prepared {len(paths)} CT-RATE paths from {args.metadata}")
    print(f"Wrote manifest: {manifest}")

    if args.list_only:
        for path in paths:
            print(path)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        hf_hub_download(
            repo_id=CT_RATE_REPO_ID,
            repo_type="dataset",
            filename=path,
            local_dir=args.output_dir,
        )

    print(f"Done. CT-RATE subset is under: {args.output_dir}")


if __name__ == "__main__":
    main()
