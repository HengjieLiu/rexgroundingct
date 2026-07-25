#!/usr/bin/env python3
"""Build and audit standard VoxTell preprocessing caches."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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
from voxtell_preprocessed_cache import (
    CACHE_SCHEMA_VERSION,
    STANDARD_CACHE_ROOT,
    SUPPORTED_PREPROCESS_IDS,
    cache_case_paths,
    preprocess_case,
    write_cached_case,
)


def _split_allows_missing_targets(entry: dict[str, Any]) -> bool:
    return entry.get("_split") == "test"


def _estimate_one(job: tuple[dict, str, str, str]) -> dict[str, Any]:
    entry, preprocess_id, ct_root, seg_dir = job
    image, targets, metadata = preprocess_case(
        entry,
        preprocess_id=preprocess_id,
        ct_root=Path(ct_root),
        seg_dir=Path(seg_dir),
        allow_missing_targets=_split_allows_missing_targets(entry),
    )
    target_bytes = int(targets.nbytes) if targets is not None else 0
    return {
        "name": entry["name"],
        "split": entry.get("_split"),
        "image_bytes": int(image.nbytes),
        "target_bytes": target_bytes,
        "total_bytes": int(image.nbytes) + target_bytes,
        "shape_zyx": metadata["resampled_shape_zyx"],
        "has_targets": targets is not None,
    }


def _build_one(job: tuple[dict, str, str, str, str, bool]) -> dict[str, Any]:
    entry, cache_root, preprocess_id, ct_root, seg_dir, overwrite = job
    return write_cached_case(
        Path(cache_root),
        entry,
        preprocess_id=preprocess_id,
        ct_root=Path(ct_root),
        seg_dir=Path(seg_dir),
        overwrite=overwrite,
        allow_missing_targets=_split_allows_missing_targets(entry),
    )


def _estimate_storage(
    entries: list[dict[str, Any]],
    preprocess_id: str,
    ct_root: Path,
    seg_dir: Path,
    sample_cases: int,
) -> dict[str, Any]:
    if sample_cases >= len(entries):
        sample = entries
    elif sample_cases == 1:
        sample = [entries[0]]
    else:
        indices = {
            int(round(index * (len(entries) - 1) / float(sample_cases - 1)))
            for index in range(sample_cases)
        }
        sample = [entries[index] for index in sorted(indices)]
    records = [
        _estimate_one((entry, preprocess_id, str(ct_root), str(seg_dir)))
        for entry in tqdm(sample, desc="Estimating cache storage")
    ]
    if not records:
        return {
            "sample_cases": 0,
            "mean_case_bytes": 0,
            "estimated_total_bytes": 0,
            "sample_records": [],
        }
    mean_bytes = sum(record["total_bytes"] for record in records) / float(len(records))
    return {
        "sample_cases": len(records),
        "mean_case_bytes": int(round(mean_bytes)),
        "estimated_total_bytes": int(round(mean_bytes * len(entries))),
        "sample_records": records,
    }


def _check_storage(cache_root: Path, estimate: dict[str, Any], free_buffer: float) -> dict[str, Any]:
    probe_path = cache_root if cache_root.exists() else cache_root.parent
    probe_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(probe_path)
    estimated = int(estimate["estimated_total_bytes"])
    required = int(round(estimated * (1.0 + free_buffer)))
    result = {
        "path": str(probe_path),
        "free_bytes": int(usage.free),
        "estimated_total_bytes": estimated,
        "required_free_bytes_with_buffer": required,
        "free_buffer_fraction": float(free_buffer),
        "passed": int(usage.free) >= required,
    }
    if not result["passed"]:
        raise RuntimeError(
            "Insufficient free space for cache generation: "
            f"free={usage.free} required={required} estimate={estimated}"
        )
    return result


def _run_build_jobs(
    jobs: list[tuple[dict, str, str, str, str, bool]],
    num_workers: int,
) -> list[dict[str, Any]]:
    if num_workers == 1:
        return [
            record
            for record in tqdm(
                map(_build_one, jobs),
                total=len(jobs),
                desc="Caching VoxTell preprocessed cases",
            )
        ]
    pool = ProcessPoolExecutor(max_workers=num_workers)
    try:
        return [
            record
            for record in tqdm(
                pool.map(_build_one, jobs),
                total=len(jobs),
                desc="Caching VoxTell preprocessed cases",
            )
        ]
    finally:
        pool.shutdown()


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats: Counter = Counter()
    shape_min = [10**9, 10**9, 10**9]
    shape_max = [0, 0, 0]
    total_targets = 0
    empty_targets = 0
    fallback_targets = 0
    image_bytes = 0
    target_bytes = 0
    for record in records:
        stats[f"split_{record.get('split')}_cases"] += 1
        shape = record["resampled_shape_zyx"]
        shape_min = [min(old, int(value)) for old, value in zip(shape_min, shape)]
        shape_max = [max(old, int(value)) for old, value in zip(shape_max, shape)]
        image_bytes += int(record.get("image_nbytes", 0))
        target_bytes += int(record.get("targets_nbytes", 0))
        counts = record.get("resampled_target_voxels") or []
        total_targets += len(counts)
        empty_targets += sum(int(value) <= 0 for value in counts)
        mask_resampling = record.get("mask_resampling") or {}
        fallback_targets += len(mask_resampling.get("fallback_target_indices") or [])
    if not records:
        shape_min = [0, 0, 0]
    return {
        "case_stats": dict(stats),
        "resampled_shape_zyx_min": shape_min,
        "resampled_shape_zyx_max": shape_max,
        "targets": total_targets,
        "empty_targets": empty_targets,
        "foreground_fallback_targets": fallback_targets,
        "image_bytes_uncompressed": image_bytes,
        "target_bytes_uncompressed": target_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--preprocess-id", choices=SUPPORTED_PREPROCESS_IDS, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--estimate-cases", type=int, default=8)
    parser.add_argument("--storage-buffer-fraction", type=float, default=0.20)
    parser.add_argument("--skip-storage-check", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.estimate_cases < 1:
        raise ValueError("--estimate-cases must be >= 1")
    if args.storage_buffer_fraction < 0:
        raise ValueError("--storage-buffer-fraction must be >= 0")

    cache_root = args.cache_root or (STANDARD_CACHE_ROOT / args.preprocess_id)
    entries = load_split_entries(args.metadata, args.splits)
    if args.case_name:
        selected = set(args.case_name)
        entries = [entry for entry in entries if entry["name"] in selected]
        missing = selected - {entry["name"] for entry in entries}
        if missing:
            raise ValueError(f"Requested case(s) not found in selected splits: {sorted(missing)}")
    if args.max_cases is not None:
        entries = entries[: args.max_cases]
    if not entries:
        raise ValueError("No entries selected for cache generation")

    cache_root.mkdir(parents=True, exist_ok=True)
    storage_estimate = _estimate_storage(
        entries=entries,
        preprocess_id=args.preprocess_id,
        ct_root=args.ct_root,
        seg_dir=args.seg_dir,
        sample_cases=args.estimate_cases,
    )
    storage_check = None
    if not args.skip_storage_check:
        storage_check = _check_storage(
            cache_root,
            storage_estimate,
            free_buffer=args.storage_buffer_fraction,
        )

    jobs = [
        (
            entry,
            str(cache_root),
            args.preprocess_id,
            str(args.ct_root),
            str(args.seg_dir),
            args.overwrite,
        )
        for entry in entries
    ]
    records = _run_build_jobs(jobs, args.num_workers)
    summary = _summarize_records(records)
    if summary["empty_targets"]:
        raise RuntimeError(f"Cache audit failed: {summary['empty_targets']} targets are empty")

    manifest_path = args.manifest_json or (cache_root / "manifest.json")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "ct_root": str(args.ct_root),
        "seg_dir": str(args.seg_dir),
        "cache_root": str(cache_root),
        "preprocess_id": args.preprocess_id,
        "splits": args.splits,
        "cases": len(records),
        "complete_case_markers": sum(
            cache_case_paths(cache_root, entry["name"])["complete"].is_file()
            for entry in entries
        ),
        "normalization": "crop_to_nonzero_then_full_cropped_volume_zscore_once",
        "storage_estimate": storage_estimate,
        "storage_check": storage_check,
        **summary,
    }
    if args.preprocess_id == "crop_zscore_2mm_v1":
        manifest.update(
            {
                "target_spacing_zyx_mm": [2.0, 2.0, 2.0],
                "image_interpolation": "trilinear_align_corners_false_no_antialias",
                "mask_interpolation": "nearest_exact_with_foreground_center_splat_fallback",
            }
        )
    else:
        manifest.update(
            {
                "target_spacing_zyx_mm": None,
                "image_interpolation": None,
                "mask_interpolation": None,
            }
        )
    write_json(manifest_path, manifest)
    (cache_root / ".complete").write_text(sha256_file(manifest_path) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
