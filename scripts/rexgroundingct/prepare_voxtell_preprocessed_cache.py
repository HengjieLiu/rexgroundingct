#!/usr/bin/env python3
"""Build and audit standard VoxTell preprocessing caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_WORKER_THREADS = 1
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, str(DEFAULT_WORKER_THREADS))

from common import (
    CT_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    command_string,
    load_split_entries,
    sha256_file,
    utc_now_iso,
)
from voxtell_preprocessed_cache import (
    CACHE_SCHEMA_VERSION,
    CLIPPED_LINEAR_ISO07_PREPROCESS_ID,
    CLIPPED_LINEAR_NATIVE_PREPROCESS_ID,
    CLIPPED_ZSCORE_NATIVE_PREPROCESS_ID,
    NATIVE_GEOMETRY_PREPROCESS_IDS,
    PREPROCESS_SPECS,
    STANDARD_CACHE_ROOT,
    SUPPORTED_PREPROCESS_IDS,
    cache_case_paths,
    preprocess_case,
    preprocess_native_variants,
    write_cached_native_variants,
    write_cached_case,
)


def _configure_process_threads(num_threads: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(num_threads)
    try:
        import torch

        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
    except Exception:
        pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _split_allows_missing_targets(entry: dict[str, Any]) -> bool:
    return entry.get("_split") == "test"


def _estimate_one(job: tuple[dict, tuple[str, ...], str, str]) -> dict[str, dict[str, Any]]:
    entry, preprocess_ids, ct_root, seg_dir = job
    if len(preprocess_ids) > 1:
        if any(value not in NATIVE_GEOMETRY_PREPROCESS_IDS for value in preprocess_ids):
            raise ValueError("Multi-output estimation only supports native-geometry caches")
        variants = preprocess_native_variants(
            entry,
            preprocess_ids,
            ct_root=Path(ct_root),
            seg_dir=Path(seg_dir),
            allow_missing_targets=_split_allows_missing_targets(entry),
        )
    else:
        preprocess_id = preprocess_ids[0]
        variants = {
            preprocess_id: preprocess_case(
                entry,
                preprocess_id=preprocess_id,
                ct_root=Path(ct_root),
                seg_dir=Path(seg_dir),
                allow_missing_targets=_split_allows_missing_targets(entry),
            )
        }
    records: dict[str, dict[str, Any]] = {}
    for preprocess_id, (image, targets, metadata) in variants.items():
        target_bytes = int(targets.nbytes) if targets is not None else 0
        records[preprocess_id] = {
            "name": entry["name"],
            "split": entry.get("_split"),
            "image_bytes": int(image.nbytes),
            "target_bytes": target_bytes,
            "total_bytes": int(image.nbytes) + target_bytes,
            "shape_zyx": metadata["resampled_shape_zyx"],
            "has_targets": targets is not None,
        }
    return records


def _build_one(
    job: tuple[dict, dict[str, str], tuple[str, ...], str, str, bool],
) -> dict[str, dict[str, Any]]:
    entry, cache_roots, preprocess_ids, ct_root, seg_dir, overwrite = job
    if len(preprocess_ids) > 1:
        return write_cached_native_variants(
            {key: Path(value) for key, value in cache_roots.items()},
            entry,
            ct_root=Path(ct_root),
            seg_dir=Path(seg_dir),
            overwrite=overwrite,
            allow_missing_targets=_split_allows_missing_targets(entry),
        )
    preprocess_id = preprocess_ids[0]
    return {
        preprocess_id: write_cached_case(
            Path(cache_roots[preprocess_id]),
            entry,
            preprocess_id=preprocess_id,
            ct_root=Path(ct_root),
            seg_dir=Path(seg_dir),
            overwrite=overwrite,
            allow_missing_targets=_split_allows_missing_targets(entry),
        )
    }


def _estimate_storage(
    entries: list[dict[str, Any]],
    preprocess_ids: tuple[str, ...],
    ct_root: Path,
    seg_dir: Path,
    sample_cases: int,
) -> dict[str, dict[str, Any]]:
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
    records_by_case = [
        _estimate_one((entry, preprocess_ids, str(ct_root), str(seg_dir)))
        for entry in tqdm(sample, desc="Estimating cache storage")
    ]
    estimates: dict[str, dict[str, Any]] = {}
    for preprocess_id in preprocess_ids:
        records = [record[preprocess_id] for record in records_by_case]
        if not records:
            estimates[preprocess_id] = {
                "sample_cases": 0,
                "mean_case_bytes": 0,
                "estimated_total_bytes": 0,
                "sample_records": [],
            }
            continue
        mean_bytes = sum(record["total_bytes"] for record in records) / float(len(records))
        estimates[preprocess_id] = {
            "sample_cases": len(records),
            "mean_case_bytes": int(round(mean_bytes)),
            "estimated_total_bytes": int(round(mean_bytes * len(entries))),
            "sample_records": records,
        }
    return estimates


def _skipped_storage_estimates(
    entries: list[dict[str, Any]],
    preprocess_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        preprocess_id: {
            "sample_cases": 0,
            "mean_case_bytes": 0,
            "estimated_total_bytes": 0,
            "sample_records": [],
            "skipped": True,
            "reason": "disabled_by_--skip-storage-estimate",
            "selected_cases": len(entries),
        }
        for preprocess_id in preprocess_ids
    }


def _check_storage(
    cache_roots: dict[str, Path],
    estimates: dict[str, dict[str, Any]],
    free_buffer: float,
) -> dict[str, Any]:
    first_root = next(iter(cache_roots.values()))
    probe_path = first_root if first_root.exists() else first_root.parent
    probe_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(probe_path)
    estimated = sum(int(value["estimated_total_bytes"]) for value in estimates.values())
    required = int(round(estimated * (1.0 + free_buffer)))
    result = {
        "path": str(probe_path),
        "free_bytes": int(usage.free),
        "estimated_bytes_by_preprocess_id": {
            key: int(value["estimated_total_bytes"]) for key, value in estimates.items()
        },
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
    jobs: list[tuple[dict, dict[str, str], tuple[str, ...], str, str, bool]],
    num_workers: int,
    multiprocessing_start_method: str,
    worker_threads: int,
) -> list[dict[str, dict[str, Any]]]:
    _configure_process_threads(worker_threads)
    if num_workers == 1:
        return [
            record
            for record in tqdm(
                map(_build_one, jobs),
                total=len(jobs),
                desc="Caching VoxTell preprocessed cases",
            )
        ]
    context = mp.get_context(multiprocessing_start_method)
    pool = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_configure_process_threads,
        initargs=(worker_threads,),
    )
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
    case_content_index = [
        {
            "name": record["name"],
            "split": record.get("split"),
            "image_sha256": record.get("image_sha256"),
            "targets_sha256": record.get("targets_sha256"),
            "resampled_shape_zyx": record.get("resampled_shape_zyx"),
        }
        for record in records
    ]
    target_index = [
        {
            "name": record["name"],
            "targets_sha256": record.get("targets_sha256"),
        }
        for record in records
    ]

    def index_sha256(value: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "case_stats": dict(stats),
        "resampled_shape_zyx_min": shape_min,
        "resampled_shape_zyx_max": shape_max,
        "targets": total_targets,
        "empty_targets": empty_targets,
        "foreground_fallback_targets": fallback_targets,
        "image_bytes_uncompressed": image_bytes,
        "target_bytes_uncompressed": target_bytes,
        "case_content_index_sha256": index_sha256(case_content_index),
        "target_index_sha256": index_sha256(target_index),
    }


def _load_reference_records(
    entries: list[dict[str, Any]],
    reference_cache_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for entry in tqdm(entries, desc="Loading reference cache metadata"):
        paths = cache_case_paths(reference_cache_root, entry["name"])
        if not paths["complete"].is_file() or not paths["metadata"].is_file():
            raise FileNotFoundError(
                f"Missing reference cache metadata for {entry['name']}: {paths['root']}"
            )
        records.append(json.loads(paths["metadata"].read_text()))
    return records


def _cross_cache_audit(
    entries: list[dict[str, Any]],
    records_by_id: dict[str, list[dict[str, Any]]],
    reference_cache_root: Path | None,
) -> dict[str, Any]:
    comparison_records = dict(records_by_id)
    reference_label = None
    if reference_cache_root is not None:
        reference_label = reference_cache_root.name
        comparison_records[f"reference:{reference_label}"] = _load_reference_records(
            entries,
            reference_cache_root,
        )

    fields = (
        "targets_sha256",
        "crop_bbox_zyx",
        "original_reoriented_shape_zyx",
        "native_cropped_shape_zyx",
        "resampled_shape_zyx",
        "orientation",
    )
    mismatches: list[dict[str, Any]] = []
    labels = list(comparison_records)
    baseline_label = labels[0]
    baseline_records = comparison_records[baseline_label]
    for index, entry in enumerate(entries):
        baseline = baseline_records[index]
        for label in labels[1:]:
            candidate = comparison_records[label][index]
            for field in fields:
                if candidate.get(field) != baseline.get(field):
                    mismatches.append(
                        {
                            "name": entry["name"],
                            "field": field,
                            "baseline": baseline_label,
                            "candidate": label,
                        }
                    )
                    break
    hu_header_failures = [
        record["name"]
        for records in records_by_id.values()
        for record in records
        if not record.get("ct_intensity_header", {}).get("materialized_hu_check_passed", False)
    ]
    normalization_failures: list[dict[str, Any]] = []
    for preprocess_id, records in records_by_id.items():
        for record in records:
            stats = record.get("normalized_cropped_statistics") or {}
            if preprocess_id == CLIPPED_ZSCORE_NATIVE_PREPROCESS_ID:
                if abs(float(stats.get("mean", 1.0))) > 5e-5 or abs(
                    float(stats.get("std", 0.0)) - 1.0
                ) > 5e-5:
                    normalization_failures.append(
                        {"name": record["name"], "preprocess_id": preprocess_id, "stats": stats}
                    )
            elif preprocess_id == CLIPPED_LINEAR_NATIVE_PREPROCESS_ID:
                if float(stats.get("min", -2.0)) < -1.000001 or float(
                    stats.get("max", 2.0)
                ) > 1.000001:
                    normalization_failures.append(
                        {"name": record["name"], "preprocess_id": preprocess_id, "stats": stats}
                    )
            elif preprocess_id == CLIPPED_LINEAR_ISO07_PREPROCESS_ID:
                resampled_stats = record.get("normalized_resampled_statistics") or {}
                if float(resampled_stats.get("min", -2.0)) < -1.000001 or float(
                    resampled_stats.get("max", 2.0)
                ) > 1.000001:
                    normalization_failures.append(
                        {
                            "name": record["name"],
                            "preprocess_id": preprocess_id,
                            "stats": resampled_stats,
                        }
                    )
                target_spacing = (
                    (record.get("image_resampling") or {}).get("target_spacing_zyx_mm")
                    or []
                )
                if [round(float(value), 6) for value in target_spacing] != [0.7, 0.7, 0.7]:
                    normalization_failures.append(
                        {
                            "name": record["name"],
                            "preprocess_id": preprocess_id,
                            "target_spacing_zyx_mm": target_spacing,
                        }
                    )
    audit = {
        "compared_preprocess_ids": labels,
        "reference_cache_root": (
            str(reference_cache_root) if reference_cache_root is not None else None
        ),
        "reference_label": reference_label,
        "cases_compared": len(entries),
        "fields_compared": list(fields),
        "geometry_or_target_mismatches": len(mismatches),
        "geometry_or_target_mismatch_examples": mismatches[:20],
        "materialized_hu_header_failures": len(hu_header_failures),
        "materialized_hu_header_failure_examples": hu_header_failures[:20],
        "normalization_failures": len(normalization_failures),
        "normalization_failure_examples": normalization_failures[:20],
    }
    if mismatches or hu_header_failures or normalization_failures:
        raise RuntimeError(f"Cross-cache audit failed: {json.dumps(audit, sort_keys=True)}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument(
        "--preprocess-id",
        choices=SUPPORTED_PREPROCESS_IDS,
        action="append",
        required=True,
        help="Repeat for a one-pass multi-output native cache build.",
    )
    parser.add_argument(
        "--standard-cache-root",
        type=Path,
        default=STANDARD_CACHE_ROOT,
        help="Parent directory used when --cache-root is not supplied.",
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--reference-cache-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--multiprocessing-start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
        help="Use spawn/forkserver for torch-backed resampling; fork is legacy.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=DEFAULT_WORKER_THREADS,
        help="Torch/OpenMP/MKL threads per cache worker.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--max-cases-per-split",
        type=int,
        default=None,
        help="Select up to this many cases from each requested split before --max-cases.",
    )
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--estimate-cases", type=int, default=8)
    parser.add_argument(
        "--skip-storage-estimate",
        action="store_true",
        help="Skip the pre-build sample preprocessing pass and record a skipped estimate.",
    )
    parser.add_argument("--storage-buffer-fraction", type=float, default=0.20)
    parser.add_argument("--skip-storage-check", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.worker_threads < 1:
        raise ValueError("--worker-threads must be >= 1")
    if args.estimate_cases < 1:
        raise ValueError("--estimate-cases must be >= 1")
    if args.storage_buffer_fraction < 0:
        raise ValueError("--storage-buffer-fraction must be >= 0")
    if args.max_cases_per_split is not None and args.max_cases_per_split < 1:
        raise ValueError("--max-cases-per-split must be >= 1")
    _configure_process_threads(args.worker_threads)

    preprocess_ids = tuple(dict.fromkeys(args.preprocess_id))
    if len(preprocess_ids) > 1:
        if args.cache_root is not None or args.manifest_json is not None:
            raise ValueError("--cache-root/--manifest-json require exactly one --preprocess-id")
        if any(value not in NATIVE_GEOMETRY_PREPROCESS_IDS for value in preprocess_ids):
            raise ValueError("Multi-output builds only support native-geometry preprocessing IDs")
    cache_roots = {
        preprocess_id: (
            args.cache_root
            if args.cache_root is not None
            else args.standard_cache_root / preprocess_id
        )
        for preprocess_id in preprocess_ids
    }
    entries = load_split_entries(args.metadata, args.splits)
    if args.case_name:
        selected = set(args.case_name)
        entries = [entry for entry in entries if entry["name"] in selected]
        missing = selected - {entry["name"] for entry in entries}
        if missing:
            raise ValueError(f"Requested case(s) not found in selected splits: {sorted(missing)}")
    if args.max_cases_per_split is not None:
        counts_by_split: dict[str, int] = {}
        selected_entries = []
        for entry in entries:
            split = str(entry.get("_split"))
            count = counts_by_split.get(split, 0)
            if count >= args.max_cases_per_split:
                continue
            selected_entries.append(entry)
            counts_by_split[split] = count + 1
        entries = selected_entries
    if args.max_cases is not None:
        entries = entries[: args.max_cases]
    if not entries:
        raise ValueError("No entries selected for cache generation")

    for cache_root in cache_roots.values():
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / ".complete").unlink(missing_ok=True)
    if args.skip_storage_estimate:
        storage_estimates = _skipped_storage_estimates(entries, preprocess_ids)
    else:
        storage_estimates = _estimate_storage(
            entries=entries,
            preprocess_ids=preprocess_ids,
            ct_root=args.ct_root,
            seg_dir=args.seg_dir,
            sample_cases=args.estimate_cases,
        )
    storage_check = None
    if not args.skip_storage_check and not args.skip_storage_estimate:
        storage_check = _check_storage(
            cache_roots,
            storage_estimates,
            free_buffer=args.storage_buffer_fraction,
        )
    elif args.skip_storage_estimate and not args.skip_storage_check:
        storage_check = {
            "path": str(next(iter(cache_roots.values()))),
            "free_bytes": None,
            "estimated_bytes_by_preprocess_id": {
                preprocess_id: None for preprocess_id in preprocess_ids
            },
            "estimated_total_bytes": None,
            "required_free_bytes_with_buffer": None,
            "free_buffer_fraction": float(args.storage_buffer_fraction),
            "passed": None,
            "skipped": True,
            "reason": "storage estimate was skipped",
        }

    jobs = [
        (
            entry,
            {key: str(value) for key, value in cache_roots.items()},
            preprocess_ids,
            str(args.ct_root),
            str(args.seg_dir),
            args.overwrite,
        )
        for entry in entries
    ]
    records_by_case = _run_build_jobs(
        jobs,
        args.num_workers,
        multiprocessing_start_method=args.multiprocessing_start_method,
        worker_threads=args.worker_threads,
    )
    records_by_id = {
        preprocess_id: [record[preprocess_id] for record in records_by_case]
        for preprocess_id in preprocess_ids
    }
    summaries = {
        preprocess_id: _summarize_records(records)
        for preprocess_id, records in records_by_id.items()
    }
    for preprocess_id, summary in summaries.items():
        if summary["empty_targets"]:
            raise RuntimeError(
                f"{preprocess_id} cache audit failed: "
                f"{summary['empty_targets']} targets are empty"
            )

    cross_cache_audit = _cross_cache_audit(
        entries,
        records_by_id,
        reference_cache_root=args.reference_cache_root,
    )
    manifests: dict[str, dict[str, Any]] = {}
    for preprocess_id in preprocess_ids:
        cache_root = cache_roots[preprocess_id]
        manifest_path = (
            args.manifest_json
            if args.manifest_json is not None
            else cache_root / "manifest.json"
        )
        spec = PREPROCESS_SPECS[preprocess_id]
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "command": command_string(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256_file(args.metadata),
            "ct_root": str(args.ct_root),
            "seg_dir": str(args.seg_dir),
            "cache_root": str(cache_root),
            "preprocess_id": preprocess_id,
            "splits": args.splits,
            "cases": len(records_by_id[preprocess_id]),
            "num_workers": args.num_workers,
            "multiprocessing_start_method": args.multiprocessing_start_method,
            "worker_threads": args.worker_threads,
            "complete_case_markers": sum(
                cache_case_paths(cache_root, entry["name"])["complete"].is_file()
                for entry in entries
            ),
            "normalization": spec["normalization"],
            "normalization_parameters": spec["normalization_parameters"],
            "image_padding_value": spec["image_padding_value"],
            "target_padding_value": spec["target_padding_value"],
            "storage_estimate": storage_estimates[preprocess_id],
            "combined_storage_check": storage_check,
            "cross_cache_audit": cross_cache_audit,
            **summaries[preprocess_id],
        }
        manifest.update(
            {
                "target_spacing_zyx_mm": spec.get("target_spacing_zyx_mm"),
                "image_interpolation": spec.get("image_interpolation"),
                "mask_interpolation": spec.get("mask_interpolation"),
            }
        )
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_text(
            cache_root / ".complete",
            sha256_file(manifest_path) + "\n",
        )
        manifests[preprocess_id] = manifest
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
