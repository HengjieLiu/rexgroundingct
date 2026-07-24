#!/usr/bin/env python3
"""Build the full-FOV 4 mm proposal cache for experiment 005."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import REX_METADATA, REX_SEG_DIR, command_string, load_split_entries, sha256_file, utc_now_iso, write_json
from global_proposal import GLOBAL_SHAPE, global_case_paths, preprocess_global_case


def atomic_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def build_one(job: tuple[dict, str, str, str]) -> dict:
    entry, source_root, output_root, seg_dir = job
    paths = global_case_paths(Path(output_root), entry["name"])
    if paths["complete"].is_file():
        return json.loads(paths["metadata"].read_text())
    image, targets, metadata = preprocess_global_case(entry, Path(source_root), Path(seg_dir))
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_npy(paths["image"], image)
    atomic_npz(paths["native_targets"], targets=targets)
    tmp = paths["metadata"].with_name(f".metadata.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, paths["metadata"])
    marker = paths["complete"].with_name(f".complete.tmp.{os.getpid()}")
    marker.write_text(f"{entry['name']}\n")
    os.replace(marker, paths["complete"])
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--source-2mm-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-name", action="append", default=None)
    args = parser.parse_args()

    entries = load_split_entries(args.metadata, args.splits)
    if args.case_name:
        selected = set(args.case_name)
        entries = [entry for entry in entries if entry["name"] in selected]
        missing = selected - {entry["name"] for entry in entries}
        if missing:
            raise ValueError(f"Requested case(s) not found: {sorted(missing)}")
    if args.max_cases is not None:
        entries = entries[: args.max_cases]
    jobs = [
        (entry, str(args.source_2mm_cache), str(args.output_cache), str(args.seg_dir))
        for entry in entries
    ]
    args.output_cache.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=args.num_workers) if args.num_workers > 1 else None
    iterator = pool.map(build_one, jobs) if pool is not None else map(build_one, jobs)
    records = []
    try:
        for record in tqdm(iterator, total=len(jobs), desc="Building 4mm full-FOV cache"):
            records.append(record)
    finally:
        if pool is not None:
            pool.shutdown()

    stats: Counter = Counter()
    total_targets = 0
    inclusion_fallbacks = 0
    shape_min = [10**9] * 3
    shape_max = [0] * 3
    for record in records:
        stats[f"split_{record['split']}_cases"] += 1
        total_targets += len(record["targets"])
        inclusion_fallbacks += sum(not target["inclusion_possible"] for target in record["targets"])
        shape = record["shape_4mm_zyx"]
        shape_min = [min(old, int(value)) for old, value in zip(shape_min, shape)]
        shape_max = [max(old, int(value)) for old, value in zip(shape_max, shape)]
    if any(value > GLOBAL_SHAPE[index] for index, value in enumerate(shape_max)):
        raise RuntimeError(f"4 mm cache exceeds {GLOBAL_SHAPE}: max={shape_max}")
    manifest_path = args.manifest_json or args.output_cache / "manifest.json"
    manifest = {
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "source_2mm_cache": str(args.source_2mm_cache),
        "source_2mm_manifest_sha256": sha256_file(args.source_2mm_cache / "manifest.json"),
        "output_cache": str(args.output_cache),
        "cases": len(records),
        "targets": total_targets,
        "global_shape_zyx": list(GLOBAL_SHAPE),
        "global_spacing_mm": 4.0,
        "shape_4mm_min_zyx": shape_min,
        "shape_4mm_max_zyx": shape_max,
        "full_inclusion_impossible_target_fallbacks": inclusion_fallbacks,
        "target_policy": "valid_native_192_cube_center_region; hit-region fallback if lesion extent exceeds 192",
        "stats": dict(stats),
    }
    write_json(manifest_path, manifest)
    if len(records) == 3192 and total_targets != 8068:
        raise RuntimeError(f"Expected 8,068 targets, got {total_targets}")
    (args.output_cache / ".complete").write_text(sha256_file(manifest_path) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
