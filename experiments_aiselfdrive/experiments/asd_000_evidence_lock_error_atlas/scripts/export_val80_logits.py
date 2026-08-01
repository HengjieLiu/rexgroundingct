#!/usr/bin/env python3
"""Export and verify the ASD-000 val80 baseline logits inside pinned Docker.

This adapter reuses the read-only Side Experiment 002 inference primitives but
owns all state and arrays under the ASD-000 runtime root.  Its 80/195 validator
is intentionally independent of Side Experiment 002's 200-case validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SIDEEXP_ROOT = REPO_ROOT / "side_experiments" / "sideexp002_multimodel_ensemble_selection"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SIDEEXP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDEEXP_ROOT))

import export_logits as upstream  # noqa: E402
from hardware_contract import (  # noqa: E402
    atomic_runtime_durability_probe,
    validate_cuda_inventory,
)


EXPECTED_CASES = 80
EXPECTED_FINDINGS = 195
EXPECTED_VAL200_CASES = 200
EXPECTED_VAL200_FINDINGS = 381
EXPECTED_VAL80_EXACT_CASE_COVERAGE = 7
EXPECTED_VAL200_EXACT_CASE_COVERAGE = 14
MAX_DURABLE_STORAGE_GIB = 64


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_compressed(path: Path, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, logits=logits)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_prompt_cohort(
    path: Path,
    expected_sha256: str,
    *,
    expected_cases: int,
    expected_findings: int,
    label: str,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} source SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("test") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise RuntimeError(
            f"{label} must contain exactly {expected_cases} cases"
        )
    findings = 0
    names: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError(f"{label} contains a non-mapping case")
        name = case.get("name")
        prompts = case.get("findings")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise RuntimeError(f"invalid or duplicate {label} name: {name!r}")
        if not isinstance(prompts, dict) or sorted(prompts, key=int) != [
            str(index) for index in range(len(prompts))
        ]:
            raise RuntimeError(f"non-contiguous {label} findings for {name}")
        if not all(isinstance(value, str) and value for value in prompts.values()):
            raise RuntimeError(f"{label} contains an invalid prompt for {name}")
        names.add(name)
        findings += len(prompts)
    if findings != expected_findings:
        raise RuntimeError(
            f"{label} must contain exactly {expected_findings} findings, "
            f"got {findings}"
        )
    return cases


def ordered_prompts(cases: list[dict[str, Any]]) -> list[str]:
    return [
        str(case["findings"][str(index)])
        for case in cases
        for index in range(len(case["findings"]))
    ]


def validate_embedding_bank(
    path: Path,
    expected_sha256: str,
    *,
    expected_labels: int,
    expected_dimension: int,
    val80_cases: list[dict[str, Any]],
    val200_cases: list[dict[str, Any]],
    lookup_normalization: str,
) -> dict[str, Any]:
    if lookup_normalization != "python_str_lower":
        raise RuntimeError("embedding lookup normalization contract drifted")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("explicit embedding bank SHA-256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"labels", "embeddings"}:
                raise RuntimeError("embedding bank members drifted")
            labels = np.asarray(archive["labels"])
            embeddings = np.asarray(archive["embeddings"])
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot load explicit embedding bank: {exc}") from exc
    if (
        labels.ndim != 1
        or len(labels) != expected_labels
        or embeddings.shape != (expected_labels, expected_dimension)
        or embeddings.dtype != np.float16
        or not np.all(np.isfinite(embeddings))
    ):
        raise RuntimeError("explicit embedding bank shape/dtype/value contract drifted")
    label_values = [str(value) for value in labels.tolist()]
    label_set = set(label_values)
    if len(label_set) != expected_labels:
        raise RuntimeError("explicit embedding bank labels are not unique")

    coverage: dict[str, dict[str, int]] = {}
    for cohort_name, cases, expected_occurrences in (
        ("val80", val80_cases, EXPECTED_FINDINGS),
        ("val200", val200_cases, EXPECTED_VAL200_FINDINGS),
    ):
        prompts = ordered_prompts(cases)
        normalized = [prompt.lower() for prompt in prompts]
        missing = [prompt for prompt in normalized if prompt not in label_set]
        if missing:
            raise RuntimeError(
                f"{cohort_name} has {len(missing)} prompts absent from the "
                "explicit lowercased embedding bank"
            )
        if len(prompts) != expected_occurrences:
            raise RuntimeError(f"{cohort_name} prompt occurrence count drifted")
        coverage[cohort_name] = {
            "prompt_occurrences": len(prompts),
            "exact_case_covered_occurrences": sum(
                prompt in label_set for prompt in prompts
            ),
            "lowercase_covered_occurrences": len(prompts),
            "unique_lowercase_prompts": len(set(normalized)),
            "missing_after_lowercase": 0,
        }
    if (
        coverage["val80"]["exact_case_covered_occurrences"]
        != EXPECTED_VAL80_EXACT_CASE_COVERAGE
        or coverage["val200"]["exact_case_covered_occurrences"]
        != EXPECTED_VAL200_EXACT_CASE_COVERAGE
    ):
        raise RuntimeError("embedding bank exact-case coverage contract drifted")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "label_count": expected_labels,
        "shape": [expected_labels, expected_dimension],
        "embedding_dimension": expected_dimension,
        "dtype": "float16",
        "lookup_normalization": lookup_normalization,
        "source_text_form": "original_finding_text_verbatim",
        "coverage": coverage,
        "qwen_fallback_permitted": False,
        "predictor_embedding_mode": (
            "explicit_local_bank_with_precomputed_autoload_disabled"
        ),
        "verified": True,
    }


def load_candidate(
    path: Path,
    candidate_id: str,
    *,
    expected_checkpoint_path: Path,
    expected_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = upstream.load_manifest(path)
    matches = [row for row in manifest["candidates"] if row["id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"candidate {candidate_id!r} is not unique")
    candidate = matches[0]
    checkpoint = candidate.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("path") != str(expected_checkpoint_path)
        or checkpoint.get("sha256") != expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "candidate checkpoint path/hash does not match locked baseline"
        )
    upstream.validate_candidate_sources(candidate, verify_checkpoint_hash=True)
    cache = candidate.get("cache")
    if not isinstance(cache, dict):
        raise RuntimeError("candidate cache lineage is missing")
    try:
        candidate_manifest_path = path.resolve(strict=True).relative_to(
            REPO_ROOT.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise RuntimeError("candidate manifest is outside the repository") from exc
    return candidate, {
        "candidate_id": candidate_id,
        "checkpoint_path": str(expected_checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": sha256_file(path),
        "preprocessing_cache_id": cache.get("id"),
        "preprocessing_cache_manifest_sha256": cache.get("manifest_sha256"),
        "verified": True,
    }


def geometry_and_equivalence(
    logits: np.ndarray,
    prediction_path: Path,
    ground_truth_path: Path,
) -> dict[str, Any]:
    if logits.dtype != np.float16 or not np.all(np.isfinite(logits)):
        raise RuntimeError("logits must be finite float16 before geometry validation")
    prediction_image = nib.load(str(prediction_path))
    ground_truth_image = nib.load(str(ground_truth_path))
    prediction = np.asanyarray(prediction_image.dataobj)
    if prediction.dtype != np.uint8 or not np.all(np.isin(np.unique(prediction), [0, 1])):
        raise RuntimeError(f"locked prediction is not uint8 binary: {prediction_path}")
    if tuple(logits.shape) != tuple(prediction_image.shape) or tuple(
        prediction_image.shape
    ) != tuple(ground_truth_image.shape):
        raise RuntimeError("logit/prediction/GT shape mismatch")
    if not np.array_equal(prediction_image.affine, ground_truth_image.affine):
        raise RuntimeError("prediction/GT affine mismatch")
    prediction_codes = tuple(nib.aff2axcodes(prediction_image.affine))
    ground_truth_codes = tuple(nib.aff2axcodes(ground_truth_image.affine))
    if prediction_codes != ground_truth_codes:
        raise RuntimeError("prediction/GT orientation mismatch")
    prediction_qform = int(prediction_image.header["qform_code"])
    prediction_sform = int(prediction_image.header["sform_code"])
    if prediction_qform != int(ground_truth_image.header["qform_code"]):
        raise RuntimeError("prediction/GT qform-code mismatch")
    if prediction_sform != int(ground_truth_image.header["sform_code"]):
        raise RuntimeError("prediction/GT sform-code mismatch")
    mismatch_voxels = int(np.count_nonzero((logits >= 0.0) != (prediction > 0)))
    if mismatch_voxels:
        raise RuntimeError(
            f"threshold-0.5 mask differs from locked prediction at {mismatch_voxels} voxels"
        )
    return {
        "verified": True,
        "shape_fxyz": list(logits.shape),
        "affine_sha256": canonical_sha256(prediction_image.affine.tolist()),
        "axis_codes": list(prediction_codes),
        "qform_code": prediction_qform,
        "sform_code": prediction_sform,
        "threshold_probability": 0.5,
        "threshold_logit": 0.0,
        "inclusive": True,
        "mismatch_voxels": 0,
    }


def case_paths(root: Path, case_name: str) -> tuple[Path, Path]:
    return root / "cases" / f"{case_name}.npz", root / "cases" / f"{case_name}.json"


def validate_existing_case(
    array_path: Path,
    record_path: Path,
    prediction_path: Path,
    ground_truth_path: Path,
    finding_count: int,
    embedding_bank_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    if not array_path.is_file() or not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "complete"
        or record.get("file_sha256") != sha256_file(array_path)
        or record.get("embedding_bank_sha256") != embedding_bank_sha256
        or record.get("checkpoint_sha256") != checkpoint_sha256
    ):
        return None
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            logits = np.asarray(archive["logits"])
    except (OSError, ValueError, KeyError):
        return None
    if logits.dtype != np.float16 or logits.ndim != 4 or logits.shape[0] != finding_count:
        return None
    if not np.all(np.isfinite(logits)):
        return None
    geometry = geometry_and_equivalence(logits, prediction_path, ground_truth_path)
    if record.get("logical_array_sha256") != upstream.array_sha256(logits):
        return None
    if record.get("geometry") != geometry:
        return None
    return record


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_durable_storage_gib != MAX_DURABLE_STORAGE_GIB:
        raise RuntimeError(
            "val80 local export durable-storage contract must be exactly "
            f"{MAX_DURABLE_STORAGE_GIB} GiB"
        )
    max_durable_bytes = args.max_durable_storage_gib * 1024**3
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable}=1 is required for local logit export")
    runtime_probe = atomic_runtime_durability_probe(args.runtime_root)
    runtime_probe["completed_before_cuda_and_predictor_initialization"] = True
    cases = load_prompt_cohort(
        args.val80_json,
        args.val80_sha256,
        expected_cases=EXPECTED_CASES,
        expected_findings=EXPECTED_FINDINGS,
        label="val80",
    )
    val200_cases = load_prompt_cohort(
        args.val200_json,
        args.val200_sha256,
        expected_cases=EXPECTED_VAL200_CASES,
        expected_findings=EXPECTED_VAL200_FINDINGS,
        label="val200",
    )
    embedding_proof = validate_embedding_bank(
        args.embeddings,
        args.embeddings_sha256,
        expected_labels=args.expected_embedding_labels,
        expected_dimension=args.expected_embedding_dimension,
        val80_cases=cases,
        val200_cases=val200_cases,
        lookup_normalization=args.embedding_lookup_normalization,
    )
    candidate, candidate_lineage = load_candidate(
        args.candidate_manifest,
        args.candidate_id,
        expected_checkpoint_path=args.expected_checkpoint_path,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable in the pinned image")
    visible_device_count = torch.cuda.device_count()
    visible_device_names = [
        torch.cuda.get_device_name(index) for index in range(visible_device_count)
    ]
    cuda_inventory = validate_cuda_inventory(
        visible_device_names,
        expected_count=args.expected_device_count,
        expected_name=args.expected_device_name,
        require_exact=args.preflight_only,
    )
    if args.gpu < 0 or args.gpu >= visible_device_count:
        raise RuntimeError(
            f"selected CUDA device {args.gpu} is unavailable; "
            f"visible count is {visible_device_count}"
        )
    device = torch.device(f"cuda:{args.gpu}")
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if free_before < 32 * 1024**3:
        raise RuntimeError(
            f"GPU has only {free_before / 1024**3:.2f} GiB free; 32 GiB required"
        )
    predictor = upstream.build_predictor(candidate, device, args.embeddings)

    first = cases[0]
    if args.preflight_only:
        started = time.monotonic()
        logits, _ = upstream.predict_case_logits(
            predictor, candidate, first, args.segmentation_root
        )
        np.clip(logits, -30.0, 30.0, out=logits)
        stored = np.ascontiguousarray(logits, dtype=np.float16)
        geometry = geometry_and_equivalence(
            stored,
            args.prediction_root / first["name"],
            args.segmentation_root / first["seg_path"],
        )
        return {
            "status": "passed",
            "mode": "preflight_only",
            "candidate_id": args.candidate_id,
            "candidate_lineage": candidate_lineage,
            "embedding_bank": embedding_proof,
            "runtime_durability_probe": runtime_probe,
            "network_mode": "none",
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            **cuda_inventory,
            "device_total_memory_bytes": total_memory,
            "device_free_memory_bytes_before_checkpoint_load": free_before,
            "one_case": first["name"],
            "one_case_elapsed_seconds": time.monotonic() - started,
            "geometry": geometry,
        }

    export_root = args.runtime_root / "val80_logits_revision_2"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_path = export_root / "export_manifest.json"
    records: list[dict[str, Any]] = []
    durable_bytes = 0
    for order, case in enumerate(cases):
        name = case["name"]
        array_path, record_path = case_paths(export_root, name)
        prediction_path = args.prediction_root / name
        ground_truth_path = args.segmentation_root / case["seg_path"]
        existing = validate_existing_case(
            array_path,
            record_path,
            prediction_path,
            ground_truth_path,
            len(case["findings"]),
            args.embeddings_sha256,
            args.expected_checkpoint_sha256,
        )
        if existing is not None:
            durable_bytes += int(existing["size_bytes"])
            if durable_bytes > max_durable_bytes:
                raise RuntimeError(
                    "resumed val80 logit export exceeds the "
                    f"{args.max_durable_storage_gib} GiB durable cap"
                )
            records.append(existing)
            continue
        started = time.monotonic()
        logits, details = upstream.predict_case_logits(
            predictor, candidate, case, args.segmentation_root
        )
        np.clip(logits, -30.0, 30.0, out=logits)
        stored = np.ascontiguousarray(logits, dtype=np.float16)
        if not np.all(np.isfinite(stored)):
            raise RuntimeError(f"non-finite logits for {name}")
        geometry = geometry_and_equivalence(
            stored, prediction_path, ground_truth_path
        )
        atomic_save_compressed(array_path, stored)
        array_size = array_path.stat().st_size
        if durable_bytes + array_size > max_durable_bytes:
            array_path.unlink()
            raise RuntimeError(
                "compressed val80 logit export would exceed the "
                f"{args.max_durable_storage_gib} GiB durable cap"
            )
        record = {
            "status": "complete",
            "case_order": order,
            "case_name": name,
            "finding_count": len(case["findings"]),
            "array_path": str(array_path),
            "file_sha256": sha256_file(array_path),
            "size_bytes": array_size,
            "logical_array_sha256": upstream.array_sha256(stored),
            "dtype": "float16",
            "shape_fxyz": list(stored.shape),
            "finite": True,
            "geometry": geometry,
            "embedding_bank_sha256": args.embeddings_sha256,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "same_pass_reference": details["same_pass_reference"],
            "cache_case": details["cache_case"],
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(record_path, record)
        records.append(record)
        durable_bytes += array_size

    manifest = {
        "schema_version": "1.0",
        "experiment_id": "asd_000_evidence_lock_error_atlas",
        "plan_revision": 2,
        "status": "verified_complete",
        "candidate_id": args.candidate_id,
        "source": "local_docker_export",
        "candidate_lineage": candidate_lineage,
        "embedding_bank": embedding_proof,
        "runtime_durability_probe": runtime_probe,
        "network_mode": "none",
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "val80_path": str(args.val80_json),
        "val80_sha256": args.val80_sha256,
        "case_count": len(records),
        "finding_count": sum(int(row["finding_count"]) for row in records),
        "dtype": "float16",
        "visible_cuda_device_count_during_export": visible_device_count,
        "visible_cuda_device_names_during_export": visible_device_names,
        "compression": "numpy_savez_compressed",
        "durable_storage_cap_gib": args.max_durable_storage_gib,
        "logical_uncompressed_bytes": sum(
            int(np.prod(row["shape_fxyz"], dtype=np.int64) * 2)
            for row in records
        ),
        "logical_array_count": len(records),
        "threshold_probability": 0.5,
        "threshold_logit": 0.0,
        "threshold_mask_equivalence": "exact",
        "total_size_bytes": sum(int(row["size_bytes"]) for row in records),
        "cases": records,
    }
    if manifest["case_count"] != EXPECTED_CASES or manifest["finding_count"] != EXPECTED_FINDINGS:
        raise RuntimeError("final val80 logit counts are incomplete")
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--val80-json", required=True, type=Path)
    parser.add_argument("--val80-sha256", required=True)
    parser.add_argument("--val200-json", required=True, type=Path)
    parser.add_argument("--val200-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--segmentation-root", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--expected-device-count", required=True, type=int)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--max-durable-storage-gib", required=True, type=int)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--embeddings-sha256", required=True)
    parser.add_argument("--expected-embedding-labels", required=True, type=int)
    parser.add_argument("--expected-embedding-dimension", required=True, type=int)
    parser.add_argument(
        "--embedding-lookup-normalization",
        required=True,
        choices=("python_str_lower",),
    )
    parser.add_argument("--expected-checkpoint-path", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), sort_keys=True, allow_nan=False))
