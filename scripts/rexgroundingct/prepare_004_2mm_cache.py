#!/usr/bin/env python3
"""Build and audit the reusable train/validation 2 mm cache for experiment 004."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from common import (
    CT_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    command_string,
    load_split_entries,
    sha256_file,
    utc_now_iso,
    write_json,
)
from voxtell_2mm import CACHE_SCHEMA_VERSION, cache_case_paths, write_cached_case


def _build_one(job: tuple[dict, str, str, str, bool]) -> dict:
    entry, cache_root, ct_root, seg_dir, overwrite = job
    return write_cached_case(
        Path(cache_root),
        entry,
        ct_root=Path(ct_root),
        seg_dir=Path(seg_dir),
        overwrite=overwrite,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-name", action="append", default=None)
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    entries = load_split_entries(args.metadata, args.splits)
    if args.case_name:
        selected = set(args.case_name)
        entries = [entry for entry in entries if entry["name"] in selected]
        missing = selected - {entry["name"] for entry in entries}
        if missing:
            raise ValueError(f"Requested case(s) not found in selected splits: {sorted(missing)}")
    if args.max_cases is not None:
        entries = entries[: args.max_cases]
    args.cache_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (entry, str(args.cache_root), str(args.ct_root), str(args.seg_dir), args.overwrite)
        for entry in entries
    ]
    if args.num_workers == 1:
        iterator = map(_build_one, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.num_workers)
        iterator = pool.map(_build_one, jobs)

    records = []
    try:
        for record in tqdm(iterator, total=len(jobs), desc="Caching crop-zscore-2mm cases"):
            records.append(record)
    finally:
        if pool is not None:
            pool.shutdown()

    stats: Counter = Counter()
    shape_min = [10**9, 10**9, 10**9]
    shape_max = [0, 0, 0]
    total_targets = 0
    empty_targets = 0
    fallback_targets = 0
    for record in records:
        stats[f"split_{record['split']}_cases"] += 1
        shape = record["resampled_shape_zyx"]
        shape_min = [min(old, int(value)) for old, value in zip(shape_min, shape)]
        shape_max = [max(old, int(value)) for old, value in zip(shape_max, shape)]
        total_targets += len(record["resampled_target_voxels"])
        empty_targets += sum(int(value) <= 0 for value in record["resampled_target_voxels"])
        fallback_targets += len(record["mask_resampling"]["fallback_target_indices"])
    if empty_targets:
        raise RuntimeError(f"Cache audit failed: {empty_targets} targets are empty")

    manifest_path = args.manifest_json or (args.cache_root / "manifest.json")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "ct_root": str(args.ct_root),
        "seg_dir": str(args.seg_dir),
        "cache_root": str(args.cache_root),
        "splits": args.splits,
        "cases": len(records),
        "targets": total_targets,
        "empty_targets": empty_targets,
        "foreground_fallback_targets": fallback_targets,
        "resampled_shape_zyx_min": shape_min,
        "resampled_shape_zyx_max": shape_max,
        "normalization": "crop_to_nonzero_then_full_cropped_volume_zscore_once",
        "target_spacing_zyx_mm": [2.0, 2.0, 2.0],
        "image_interpolation": "trilinear_align_corners_false_no_antialias",
        "mask_interpolation": "nearest_exact_with_foreground_center_splat_fallback",
        "case_stats": dict(stats),
        "complete_case_markers": sum(
            cache_case_paths(args.cache_root, entry["name"])["complete"].is_file()
            for entry in entries
        ),
    }
    write_json(manifest_path, manifest)
    if len(records) == len(load_split_entries(args.metadata, ["train", "val"])) and total_targets != 8068:
        raise RuntimeError(f"Expected 8,068 train/validation targets, found {total_targets}")
    (args.cache_root / ".complete").write_text(sha256_file(manifest_path) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
