#!/usr/bin/env python3
"""Export and strictly validate one fresh SideExp003 val200 logit cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SIDEEXP002 = REPO_ROOT / "side_experiments/sideexp002_multimodel_ensemble_selection"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SIDEEXP002) not in sys.path:
    sys.path.insert(0, str(SIDEEXP002))

from fresh_cache import (  # noqa: E402
    FreshCacheError,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
    validate_job,
)
from analyze_candidates import flatten_eval  # noqa: E402
from export_logits import (  # noqa: E402
    LOGIT_CLIP,
    acquire_lock,
    array_sha256,
    atomic_save_npy,
    build_predictor,
    load_dataset,
    mask_dice,
    predict_case_logits,
    storage_mask_mismatch_voxels,
    validate_candidate_sources,
)


SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
REPO_PREFIX = Path("/home/hengjie/code_sync/rexgroundingct")


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.exists() or not value.is_absolute():
        return value if value.is_absolute() else REPO_ROOT / value
    try:
        return REPO_ROOT / value.relative_to(REPO_PREFIX)
    except ValueError:
        return value


def _case_stem(name: str) -> str:
    return name.removesuffix(".nii.gz")


def ordered_cases(cases: list[dict[str, Any]], cache_root: Path) -> list[dict[str, Any]]:
    """Put the largest cached input first for the retained smoke case."""

    ranked = []
    for original_index, case in enumerate(cases):
        metadata_path = cache_root / "cases" / _case_stem(case["name"]) / "metadata.json"
        voxels = 0
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            shape = metadata.get("resampled_shape_zyx")
            if isinstance(shape, list) and len(shape) == 3:
                voxels = math.prod(int(value) for value in shape)
        ranked.append((-voxels, original_index, case))
    ranked.sort(key=lambda value: (value[0], value[1]))
    return [value[2] for value in ranked]


def _stage_paths(root: Path, name: str) -> tuple[Path, Path]:
    return (
        root / "staging_float32/cases" / f"{name}.npy",
        root / "staging_float32/case_manifests" / f"{name}.json",
    )


def _published_paths(root: Path, name: str) -> tuple[Path, Path]:
    return root / "cases" / f"{name}.npy", root / "case_manifests" / f"{name}.json"


def _valid_stage_case(
    root: Path, case: dict[str, Any], verify_hash: bool = True
) -> dict[str, Any] | None:
    array_path, metadata_path = _stage_paths(root, case["name"])
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = read_json(metadata_path)
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if (
            str(array.dtype) != "float32"
            or array.ndim != 4
            or array.shape[0] != len(case["findings"])
            or list(array.shape) != record.get("shape")
            or record.get("status") != "complete"
        ):
            return None
        if verify_hash and array_sha256(array) != record.get("array_sha256"):
            return None
        return record
    except (OSError, ValueError, FreshCacheError):
        return None


def _write_progress(path: Path, base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    record = {**base, **updates, "updated_at_utc": utc_now()}
    atomic_write_json(path, record)
    return record


def _wait_for_smoke_release(
    job_root: Path,
    wave: int,
    progress_path: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    continue_path = job_root / "control" / f"wave_{wave:02d}.continue"
    abort_path = job_root / "control" / "ABORT"
    progress = _write_progress(progress_path, progress, status="smoke_waiting")
    while not continue_path.is_file():
        if abort_path.is_file():
            reason = read_json(abort_path).get("reason", "job aborted")
            raise RuntimeError(f"smoke gate aborted: {reason}")
        time.sleep(2)
    progress = _write_progress(progress_path, progress, status="exporting")
    return progress


def _remove_controlled_tree(path: Path, cache_root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(cache_root.resolve(strict=False))
    except ValueError as exc:
        raise RuntimeError(f"refusing to remove path outside cache root: {path}") from exc
    if path.exists():
        shutil.rmtree(path)


def _publish(
    root: Path,
    cases: list[dict[str, Any]],
    progress_path: Path,
    progress: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    progress = _write_progress(progress_path, progress, status="finalizing")
    finalizing = root / "finalizing_float16"
    _remove_controlled_tree(finalizing, root)
    _remove_controlled_tree(root / "cases", root)
    _remove_controlled_tree(root / "case_manifests", root)
    (finalizing / "cases").mkdir(parents=True, exist_ok=True)
    (finalizing / "case_manifests").mkdir(parents=True, exist_ok=True)
    converted: list[dict[str, Any]] = []
    mismatch_total = 0
    for case in cases:
        name = case["name"]
        stage_array_path, stage_metadata_path = _stage_paths(root, name)
        raw = np.load(stage_array_path, mmap_mode="r", allow_pickle=False)
        compact = np.ascontiguousarray(raw, dtype=np.float16)
        mismatch = storage_mask_mismatch_voxels(raw, compact)
        mismatch_total += mismatch
        output_array, output_metadata = _published_paths(root, name)
        temporary_array = finalizing / "cases" / f"{name}.npy"
        temporary_metadata = finalizing / "case_manifests" / f"{name}.json"
        stage_record = read_json(stage_metadata_path)
        record = {
            **stage_record,
            "array_path": str(output_array),
            "dtype": "float16",
            "npy_bytes": None,
            "array_sha256": array_sha256(compact),
            "same_pass_mask_mismatch_voxels": mismatch,
            "same_pass_mask_comparison": "exact_threshold_zero_voxels",
        }
        atomic_save_npy(temporary_array, compact)
        record["npy_bytes"] = temporary_array.stat().st_size
        atomic_write_json(temporary_metadata, record)
        converted.append(record)
    if mismatch_total == 0:
        os.rename(finalizing / "cases", root / "cases")
        os.rename(finalizing / "case_manifests", root / "case_manifests")
        finalizing.rmdir()
        return "float16", converted, progress

    _remove_controlled_tree(finalizing, root)
    stage_root = root / "staging_float32"
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "case_manifests").mkdir(parents=True, exist_ok=True)
    for case in cases:
        name = case["name"]
        array_path, metadata_path = _published_paths(root, name)
        stage_array_path, stage_metadata_path = _stage_paths(root, name)
        os.link(stage_array_path, array_path)
        record = read_json(stage_metadata_path)
        record.update(
            {
                "array_path": str(array_path),
                "dtype": "float32",
                "same_pass_mask_mismatch_voxels": 0,
                "same_pass_mask_comparison": "exact_threshold_zero_voxels",
            }
        )
        atomic_write_json(metadata_path, record)
    published = [read_json(_published_paths(root, case["name"])[1]) for case in cases]
    return "float32", published, progress


def _validate_published(
    candidate: dict[str, Any],
    root: Path,
    cases: list[dict[str, Any]],
    dtype: str,
) -> dict[str, Any]:
    historical, _summary = flatten_eval(candidate, verify_hash=True)
    values: list[float] = []
    same_pass_values: list[float] = []
    hits = 0
    same_pass_hits = 0
    mismatch_total = 0
    bytes_total = 0
    case_results = []
    for case in cases:
        name = case["name"]
        array_path, metadata_path = _published_paths(root, name)
        record = read_json(metadata_path)
        logits = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if str(logits.dtype) != dtype or list(logits.shape) != record.get("shape"):
            raise RuntimeError(f"{name}: published dtype/shape mismatch")
        if array_sha256(logits) != record.get("array_sha256"):
            raise RuntimeError(f"{name}: published array SHA mismatch")
        if not np.isfinite(logits).all():
            raise RuntimeError(f"{name}: published logits are non-finite")
        ground_truth = np.asanyarray(nib.load(str(SEG_DIR / name)).dataobj)
        if logits.shape != ground_truth.shape:
            raise RuntimeError(f"{name}: published shape {logits.shape} != GT {ground_truth.shape}")
        reference = record.get("same_pass_reference")
        reference_values = reference.get("finding_dice") if isinstance(reference, dict) else None
        if not isinstance(reference_values, list) or len(reference_values) != logits.shape[0]:
            raise RuntimeError(f"{name}: missing same-pass finding metrics")
        case_values = [
            mask_dice(ground_truth[index], logits[index] >= 0.0)
            for index in range(logits.shape[0])
        ]
        case_hits = sum(value >= 0.1 for value in case_values)
        reference_hits = sum(float(value) >= 0.1 for value in reference_values)
        values.extend(case_values)
        same_pass_values.extend(float(value) for value in reference_values)
        hits += case_hits
        same_pass_hits += reference_hits
        mismatch_total += int(record.get("same_pass_mask_mismatch_voxels", -1))
        bytes_total += array_path.stat().st_size
        case_results.append(
            {
                "name": name,
                "findings": len(case_values),
                "mean_dice": math.fsum(case_values) / len(case_values),
                "hits": case_hits,
                "same_pass_mean_dice": math.fsum(float(value) for value in reference_values) / len(reference_values),
                "same_pass_hits": reference_hits,
            }
        )
    if len(cases) != 200 or len(values) != 381:
        raise RuntimeError(f"incomplete published cache: {len(cases)} cases/{len(values)} findings")
    dice = math.fsum(values) / len(values)
    same_pass_dice = math.fsum(same_pass_values) / len(same_pass_values)
    storage_delta = dice - same_pass_dice
    storage_passed = abs(storage_delta) <= 1e-6 and hits == same_pass_hits and mismatch_total == 0
    historical_values = [
        float(historical[(case["name"], int(index))]["dice"])
        for case in cases
        for index in sorted(case["findings"], key=int)
    ]
    historical_hits = sum(
        bool(historical[(case["name"], int(index))]["hit"])
        for case in cases
        for index in sorted(case["findings"], key=int)
    )
    historical_dice = math.fsum(historical_values) / len(historical_values)
    historical_delta = dice - historical_dice
    warnings = []
    if abs(historical_delta) > 1e-4 or hits != historical_hits:
        warnings.append("historical_reproduction_mismatch")
    result = {
        "schema_version": 1,
        "candidate_id": candidate["id"],
        "status": "passed" if storage_passed else "failed",
        "dtype": dtype,
        "cases": len(cases),
        "findings": len(values),
        "array_bytes": bytes_total,
        "array_hashes_verified": True,
        "mean_global_dice_per_finding": dice,
        "total_hits": hits,
        "hit_rate": hits / len(values),
        "same_pass_mean_global_dice_per_finding": same_pass_dice,
        "same_pass_total_hits": same_pass_hits,
        "storage_dice_delta": storage_delta,
        "storage_reproduction_status": "passed" if storage_passed else "failed",
        "same_pass_mask_mismatch_voxels": mismatch_total,
        "expected_mean_global_dice_per_finding": historical_dice,
        "expected_total_hits": historical_hits,
        "dice_delta": historical_delta,
        "historical_reproduction_status": "passed" if not warnings else "warning",
        "historical_reproduction_is_gate": False,
        "warnings": warnings,
        "case_results": case_results,
    }
    atomic_write_json(root / "reproduction_validation.json", result)
    if not storage_passed:
        raise RuntimeError(
            f"storage reproduction failed: delta={storage_delta:+.8f}, "
            f"hits={hits}/{same_pass_hits}, mismatches={mismatch_total}"
        )
    return result


def export_candidate(job_path: Path, candidate_id: str, force_stale_lock: bool) -> dict[str, Any]:
    job = read_json(job_path)
    errors = validate_job(job)
    if errors:
        raise FreshCacheError("invalid job spec: " + "; ".join(errors))
    matches = [candidate for candidate in job["candidates"] if candidate["id"] == candidate_id]
    if len(matches) != 1:
        raise FreshCacheError(f"unknown or duplicated candidate {candidate_id}")
    candidate = matches[0]
    validate_candidate_sources(candidate, verify_checkpoint_hash=False)
    if sha256_file(Path(candidate["checkpoint"]["path"])) != candidate["checkpoint"]["sha256"]:
        raise FreshCacheError(f"{candidate_id}: checkpoint SHA mismatch")

    dataset_path = resolve_repo_path(job["dataset"]["path"])
    if sha256_file(dataset_path) != job["dataset"]["sha256"]:
        raise FreshCacheError("val200 dataset SHA mismatch")
    cases = ordered_cases(load_dataset(dataset_path), Path(candidate["cache"]["root"]))
    root = Path(candidate["cache_root"])
    progress_path = Path(candidate["progress_path"])
    job_root = Path(job["runtime_root"])
    root.mkdir(parents=True, exist_ok=True)
    existing_manifest = root / "export_manifest.json"
    if existing_manifest.is_file():
        existing = read_json(existing_manifest)
        if existing.get("job_id") != job["job_id"] or existing.get("candidate_id") != candidate_id:
            raise FreshCacheError(f"fresh cache target belongs to another job: {root}")
        if existing.get("status") == "strict_passed":
            raise FreshCacheError(f"fresh-only job refuses to reuse completed cache: {root}")
    acquire_lock(root / ".export.lock", candidate_id, force_stale_lock)
    started_at = utc_now()
    progress = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "candidate_id": candidate_id,
        "rank": candidate["rank"],
        "wave": candidate["wave"],
        "gpu": candidate["gpu"],
        "status": "smoke_running",
        "started_at_utc": started_at,
        "updated_at_utc": started_at,
        "completed_at_utc": None,
        "cases": 0,
        "findings": 0,
        "bytes": 0,
        "dtype": None,
        "elapsed_seconds": 0.0,
        "seconds_per_case": None,
        "eta_seconds": None,
        "warnings": [],
        "error": None,
    }
    if progress_path.is_file():
        previous = read_json(progress_path)
        if previous.get("job_id") == job["job_id"]:
            progress["started_at_utc"] = previous.get("started_at_utc", started_at)
    progress = _write_progress(progress_path, progress)
    export_manifest: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "candidate_id": candidate_id,
        "cache_key": candidate["cache_key"],
        "cache_version_id": candidate["cache_version_id"],
        "status": "exporting",
        "started_at_utc": progress["started_at_utc"],
        "updated_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "physical_gpu": candidate["gpu"],
        "visible_device": "cuda:0",
        "staging_dtype": "float32",
        "clip": [-LOGIT_CLIP, LOGIT_CLIP],
        "dataset": job["dataset"],
        "candidate_source": candidate,
        "cases": [],
    }
    atomic_write_json(existing_manifest, export_manifest)
    process_started = time.monotonic()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"worker requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        predictor = build_predictor(candidate, device, embeddings=None)
        case_records: list[dict[str, Any]] = []
        total_findings = 0
        total_bytes = 0
        for case_index, case in enumerate(cases):
            abort_path = job_root / "control" / "ABORT"
            if abort_path.is_file():
                raise RuntimeError(f"job aborted: {read_json(abort_path).get('reason')}")
            name = case["name"]
            existing = _valid_stage_case(root, case)
            if existing is None:
                case_started = time.monotonic()
                logits, details = predict_case_logits(predictor, candidate, case, SEG_DIR)
                raw_min = float(np.min(logits))
                raw_max = float(np.max(logits))
                low_clips = int(np.count_nonzero(logits < -LOGIT_CLIP))
                high_clips = int(np.count_nonzero(logits > LOGIT_CLIP))
                np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP, out=logits)
                stored = np.ascontiguousarray(logits, dtype=np.float32)
                if not np.isfinite(stored).all():
                    raise RuntimeError(f"{name}: non-finite staged logits")
                stage_array, stage_metadata = _stage_paths(root, name)
                logical_sha = array_sha256(stored)
                atomic_save_npy(stage_array, stored)
                published_array, _published_metadata = _published_paths(root, name)
                existing = {
                    "name": name,
                    "status": "complete",
                    "staging_array_path": str(stage_array),
                    "array_path": str(published_array),
                    "npy_bytes": stage_array.stat().st_size,
                    "array_sha256": logical_sha,
                    "shape": list(stored.shape),
                    "dtype": "float32",
                    "unclipped_range": [raw_min, raw_max],
                    "clip_low_count": low_clips,
                    "clip_high_count": high_clips,
                    "same_pass_mask_mismatch_voxels": 0,
                    "same_pass_mask_comparison": "exact_threshold_zero_voxels",
                    "elapsed_seconds": time.monotonic() - case_started,
                    **details,
                }
                atomic_write_json(stage_metadata, existing)
            case_records.append(existing)
            total_findings += int(existing["shape"][0])
            total_bytes += int(existing["npy_bytes"])
            elapsed = time.monotonic() - process_started
            progress = _write_progress(
                progress_path,
                progress,
                status="smoke_running" if case_index == 0 else "exporting",
                cases=len(case_records),
                findings=total_findings,
                bytes=total_bytes,
                elapsed_seconds=elapsed,
                seconds_per_case=elapsed / len(case_records),
                eta_seconds=(elapsed / len(case_records)) * (len(cases) - len(case_records)),
            )
            export_manifest.update(
                {
                    "status": progress["status"],
                    "updated_at_utc": utc_now(),
                    "case_count": len(case_records),
                    "finding_count": total_findings,
                    "cases": case_records,
                }
            )
            atomic_write_json(existing_manifest, export_manifest)
            if case_index == 0:
                free_bytes, total_gpu_bytes = torch.cuda.mem_get_info(device)
                smoke = {
                    "candidate_id": candidate_id,
                    "wave": candidate["wave"],
                    "physical_gpu": candidate["gpu"],
                    "case": name,
                    "free_mib_after_case": free_bytes // (1024 * 1024),
                    "total_mib": total_gpu_bytes // (1024 * 1024),
                    "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) // (1024 * 1024),
                    "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) // (1024 * 1024),
                    "recorded_at_utc": utc_now(),
                }
                atomic_write_json(
                    job_root / "smoke" / f"{candidate_id}.json", smoke
                )
                progress = _wait_for_smoke_release(
                    job_root, candidate["wave"], progress_path, progress
                )
        if len(case_records) != 200 or total_findings != 381:
            raise RuntimeError(
                f"incomplete staging export: {len(case_records)} cases/{total_findings} findings"
            )
        del predictor
        torch.cuda.empty_cache()
        dtype, published_records, progress = _publish(
            root, cases, progress_path, progress
        )
        export_manifest.update(
            {
                "status": "validating",
                "updated_at_utc": utc_now(),
                "published_dtype": dtype,
                "case_count": 200,
                "finding_count": 381,
                "cases": published_records,
            }
        )
        atomic_write_json(existing_manifest, export_manifest)
        progress = _write_progress(
            progress_path, progress, status="validating", dtype=dtype
        )
        validation = _validate_published(candidate, root, cases, dtype)
        _remove_controlled_tree(root / "staging_float32", root)
        completed = utc_now()
        elapsed = time.monotonic() - process_started
        export_manifest.update(
            {
                "status": "strict_passed",
                "updated_at_utc": completed,
                "completed_at_utc": completed,
                "elapsed_seconds": elapsed,
                "validation_path": str(root / "reproduction_validation.json"),
            }
        )
        atomic_write_json(existing_manifest, export_manifest)
        progress = _write_progress(
            progress_path,
            progress,
            status="strict_passed",
            dtype=dtype,
            cases=200,
            findings=381,
            bytes=int(validation["array_bytes"]),
            completed_at_utc=completed,
            elapsed_seconds=elapsed,
            seconds_per_case=elapsed / 200,
            eta_seconds=0.0,
            warnings=validation["warnings"],
        )
        (root / ".export.lock").unlink(missing_ok=True)
        return progress
    except Exception as exc:
        progress = _write_progress(
            progress_path,
            progress,
            status="failed",
            elapsed_seconds=time.monotonic() - process_started,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        export_manifest.update(
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(existing_manifest, export_manifest)
        (root / ".export.lock").unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--force-stale-lock", action="store_true")
    args = parser.parse_args()
    result = export_candidate(args.job, args.candidate_id, args.force_stale_lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
