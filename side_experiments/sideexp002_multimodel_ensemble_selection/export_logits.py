#!/usr/bin/env python3
"""Export one sideexp002 candidate's evaluator-aligned pre-sigmoid logits."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "rexgroundingct"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_candidates import (  # noqa: E402
    atomic_write_json,
    flatten_eval,
    load_manifest,
    require_candidate_config,
    require_sha256,
    resolve_source_path,
    sha256_file,
)
from common import REX_SEG_DIR, sorted_prompts  # noqa: E402
from run_voxtell_val_inference import (  # noqa: E402
    export_prediction_to_gt_layout,
    load_model_spec_kind,
    predict_preprocessed_crop_branch_logits,
    predict_preprocessed_crop_logits,
    restore_cached_native_crop,
)
from voxtell.inference.predictor import VoxTellPredictor  # noqa: E402
from voxtell_dual_branch import DualBranchVoxTellPredictor  # noqa: E402
from voxtell_preprocessed_cache import (  # noqa: E402
    image_padding_value,
    load_cached_case,
)
from voxtell_s3_attention import S3AttentionVoxTellPredictor  # noqa: E402
from storage_validation import validate_existing_case  # noqa: E402


DEFAULT_MANIFEST = HERE / "candidate_manifest.json"
DEFAULT_DATASET = (
    REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp002_multimodel_ensemble_selection"
)
LOGIT_CLIP = 30.0
LOCK_STALE_SECONDS = 48 * 60 * 60


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    cases = data.get("test") if isinstance(data, dict) else None
    if not isinstance(cases, list) or len(cases) != 200:
        raise ValueError(f"{path}: expected 200 val200 cases under 'test'")
    seen: set[str] = set()
    findings = 0
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"{path}: duplicate or invalid case name {name!r}")
        seen.add(name)
        if not isinstance(case.get("findings"), dict):
            raise ValueError(f"{path} {name}: missing findings")
        findings += len(case["findings"])
    if findings != 381:
        raise ValueError(f"{path}: expected 381 findings, got {findings}")
    return cases


def acquire_lock(path: Path, candidate_id: str, force_stale: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and force_stale:
        age = max(0.0, time.time() - path.stat().st_mtime)
        if age < LOCK_STALE_SECONDS:
            raise RuntimeError(
                f"{path}: lock is only {age / 3600:.1f} hours old; "
                f"stale threshold is {LOCK_STALE_SECONDS / 3600:.0f} hours"
            )
        path.unlink()
    metadata = {
        "candidate_id": candidate_id,
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "command": sys.argv,
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(f"candidate export lock already exists: {path}") from exc
    with os.fdopen(descriptor, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def array_sha256(array: np.ndarray, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), chunk_bytes):
        digest.update(view[offset : offset + chunk_bytes])
    return digest.hexdigest()


def mask_dice(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-6) -> float:
    """Match the challenge evaluator's smoothed binary Dice."""
    first = mask1 > 0
    second = mask2 > 0
    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(first, second).sum())
    return float((2 * intersection + eps) / (denominator + eps))


def mask_metrics(
    ground_truth: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    """Record the in-memory threshold-zero reference before storage casting."""
    if logits.shape != ground_truth.shape:
        raise ValueError(
            f"logits shape {logits.shape} does not match GT {ground_truth.shape}"
        )
    values = [
        mask_dice(ground_truth[index], logits[index] >= 0.0)
        for index in range(logits.shape[0])
    ]
    hits = sum(value >= 0.1 for value in values)
    return {
        "threshold_logit": 0.0,
        "finding_dice": values,
        "mean_dice": float(np.mean(values, dtype=np.float64)),
        "hits": hits,
        "findings": len(values),
    }


def storage_mask_mismatch_voxels(
    logits: np.ndarray,
    stored: np.ndarray,
) -> int:
    """Count threshold-zero mask changes introduced by clipping/storage."""
    if logits.shape != stored.shape:
        raise ValueError(
            f"raw logits shape {logits.shape} != stored shape {stored.shape}"
        )
    return sum(
        int(
            np.count_nonzero(
                np.greater_equal(logits[index], 0.0)
                != np.greater_equal(stored[index], 0.0)
            )
        )
        for index in range(logits.shape[0])
    )


def case_output_paths(candidate_root: Path, name: str) -> tuple[Path, Path]:
    return (
        candidate_root / "cases" / f"{name}.npy",
        candidate_root / "case_manifests" / f"{name}.json",
    )


def build_predictor(
    candidate: dict[str, Any],
    device: torch.device,
    embeddings: Path | None,
) -> VoxTellPredictor:
    model_dir = Path(candidate["model_dir"])
    model_spec = load_model_spec_kind(model_dir)
    kwargs = {
        "model_dir": model_dir,
        "device": device,
        "embedding_bank": str(embeddings) if embeddings else None,
        "use_precomputed_embeddings": embeddings is None,
    }
    if model_spec is not None and model_spec[0] == "dual":
        return DualBranchVoxTellPredictor(**kwargs)
    if model_spec is not None and model_spec[0] == "s3":
        return S3AttentionVoxTellPredictor(**kwargs)
    return VoxTellPredictor(
        model_dir=str(model_dir),
        device=device,
        embedding_bank=str(embeddings) if embeddings else None,
        use_precomputed_embeddings=embeddings is None,
    )


def predict_case_logits(
    predictor: VoxTellPredictor,
    candidate: dict[str, Any],
    case: dict[str, Any],
    seg_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = case["name"]
    cache_root = Path(candidate["cache"]["root"])
    image, _targets, cache_metadata = load_cached_case(
        cache_root,
        name,
        require_targets=False,
    )
    prompts = sorted_prompts(case)
    padding = image_padding_value(cache_metadata)
    if isinstance(predictor, DualBranchVoxTellPredictor):
        branch_logits = predict_preprocessed_crop_branch_logits(
            predictor,
            image,
            prompts,
            padding_value=padding,
        )
        crop_logits = branch_logits["final"]
    else:
        crop_logits = predict_preprocessed_crop_logits(
            predictor,
            image,
            prompts,
            padding_value=padding,
        )
    logits_reoriented = restore_cached_native_crop(
        crop_logits,
        cache_metadata,
        fill_value=-LOGIT_CLIP,
    )
    gt_path = seg_dir / name
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing ground truth: {gt_path}")
    gt_img = nib.load(str(gt_path))
    exported, orientation = export_prediction_to_gt_layout(
        logits_reoriented,
        gt_img,
        cache_metadata["ct_properties"],
        name,
        output_dtype=None,
    )
    if exported.shape[0] != len(prompts):
        raise ValueError(
            f"{name}: exported {exported.shape[0]} logits for {len(prompts)} prompts"
        )
    ground_truth = np.asanyarray(gt_img.dataobj)
    return exported, {
        "orientation": orientation,
        "same_pass_reference": mask_metrics(ground_truth, exported),
        "cache_case": {
            "preprocess_id": cache_metadata.get("preprocess_id"),
            "image_sha256": cache_metadata.get("image_sha256"),
            "targets_sha256": cache_metadata.get("targets_sha256"),
            "image_padding_value": padding,
        },
    }


def validate_candidate_sources(
    candidate: dict[str, Any],
    verify_checkpoint_hash: bool,
) -> None:
    require_candidate_config(candidate)
    require_sha256(
        Path(candidate["cache"]["manifest_path"]),
        candidate["cache"]["manifest_sha256"],
        f"{candidate['id']} cache manifest",
    )
    require_sha256(
        Path(candidate["evaluation"]["path"]),
        candidate["evaluation"]["sha256"],
        f"{candidate['id']} evaluation",
    )
    if verify_checkpoint_hash:
        require_sha256(
            Path(candidate["checkpoint"]["path"]),
            candidate["checkpoint"]["sha256"],
            f"{candidate['id']} checkpoint",
        )
    flatten_eval(candidate, verify_hash=False)
    model_dir = Path(candidate["model_dir"])
    if not (model_dir / "plans.json").is_file():
        raise FileNotFoundError(f"{candidate['id']}: missing plans.json in {model_dir}")
    actual_checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    if actual_checkpoint.resolve() != Path(candidate["checkpoint"]["path"]).resolve():
        raise ValueError(
            f"{candidate['id']}: model directory checkpoint {actual_checkpoint} "
            f"does not match manifest {candidate['checkpoint']['path']}"
        )


def export_candidate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    matches = [
        candidate
        for candidate in manifest["candidates"]
        if candidate["id"] == args.candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate ID: {args.candidate_id}")
    candidate = matches[0]
    validate_candidate_sources(candidate, not args.skip_checkpoint_hash)
    cases = load_dataset(args.dataset_json)
    if args.case_name is not None:
        cases = [case for case in cases if case["name"] == args.case_name]
        if not cases:
            raise ValueError(f"case not found in val200: {args.case_name}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]
    dtype = np.dtype(args.dtype)
    candidate_root = args.runtime_root / "logits" / candidate["id"]
    lock_path = candidate_root / ".export.lock"
    acquire_lock(lock_path, candidate["id"], args.force_stale_lock)
    started = utc_now()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    run_record: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": candidate["id"],
        "status": "running",
        "started_at_utc": started,
        "updated_at_utc": started,
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
        "gpu": args.gpu,
        "dtype": str(dtype),
        "clip": [-LOGIT_CLIP, LOGIT_CLIP],
        "dataset_json": str(args.dataset_json),
        "dataset_sha256": sha256_file(args.dataset_json),
        "candidate_source": candidate,
        "cases": [],
    }
    candidate_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(candidate_root / "export_manifest.json", run_record)
    try:
        predictor = build_predictor(candidate, device, args.embeddings)
        for case in tqdm(cases, desc=candidate["id"]):
            name = case["name"]
            array_path, metadata_path = case_output_paths(candidate_root, name)
            existing = None
            if not args.overwrite:
                existing = validate_existing_case(
                    array_path,
                    metadata_path,
                    str(dtype),
                    len(case["findings"]),
                )
            if existing is not None:
                run_record["cases"].append(existing)
                continue
            case_started = time.monotonic()
            logits, details = predict_case_logits(
                predictor,
                candidate,
                case,
                args.seg_dir,
            )
            raw_min = float(np.min(logits))
            raw_max = float(np.max(logits))
            low_clips = int(np.count_nonzero(logits < -LOGIT_CLIP))
            high_clips = int(np.count_nonzero(logits > LOGIT_CLIP))
            np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP, out=logits)
            stored = np.ascontiguousarray(logits, dtype=dtype)
            if not np.isfinite(stored).all():
                raise RuntimeError(f"{name}: non-finite stored logits")
            mask_mismatches = storage_mask_mismatch_voxels(logits, stored)
            logical_sha256 = array_sha256(stored)
            atomic_save_npy(array_path, stored)
            case_record = {
                "name": name,
                "status": "complete",
                "array_path": str(array_path),
                "npy_bytes": array_path.stat().st_size,
                "array_sha256": logical_sha256,
                "shape": list(stored.shape),
                "dtype": str(stored.dtype),
                "unclipped_range": [raw_min, raw_max],
                "clip_low_count": low_clips,
                "clip_high_count": high_clips,
                "same_pass_mask_mismatch_voxels": mask_mismatches,
                "same_pass_mask_comparison": "exact_threshold_zero_voxels",
                "elapsed_seconds": time.monotonic() - case_started,
                **details,
            }
            atomic_write_json(metadata_path, case_record)
            run_record["cases"].append(case_record)
            run_record["updated_at_utc"] = utc_now()
            atomic_write_json(candidate_root / "export_manifest.json", run_record)
        full_export = args.case_name is None and args.limit is None
        run_record["status"] = "complete" if full_export else "partial"
        run_record["completed_at_utc"] = utc_now()
        run_record["case_count"] = len(run_record["cases"])
        run_record["finding_count"] = sum(
            int(case["shape"][0]) for case in run_record["cases"]
        )
        if full_export and (
            run_record["case_count"] != 200 or run_record["finding_count"] != 381
        ):
            raise RuntimeError(
                f"{candidate['id']}: incomplete export "
                f"{run_record['case_count']} cases/{run_record['finding_count']} findings"
            )
        atomic_write_json(candidate_root / "export_manifest.json", run_record)
        lock_path.unlink(missing_ok=True)
        return run_record
    except Exception as exc:
        run_record["status"] = "failed"
        run_record["updated_at_utc"] = utc_now()
        run_record["error"] = str(exc)
        run_record["traceback"] = traceback.format_exc()
        atomic_write_json(candidate_root / "export_manifest.json", run_record)
        lock_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-stale-lock", action="store_true")
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip the multi-gigabyte checkpoint read; other source hashes remain required.",
    )
    args = parser.parse_args()
    result = export_candidate(args)
    print(
        f"{result['candidate_id']}: {result['status']} with "
        f"{len(result['cases'])} cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
