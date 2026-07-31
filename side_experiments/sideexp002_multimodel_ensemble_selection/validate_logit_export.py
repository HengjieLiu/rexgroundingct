#!/usr/bin/env python3
"""Validate a completed logit export against its original threshold-0.5 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from analyze_candidates import (
    atomic_write_json,
    flatten_eval,
    load_manifest,
    reproduction_decision,
)
from export_logits import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    DEFAULT_RUNTIME_ROOT,
    case_output_paths,
    load_dataset,
    mask_dice,
)
from storage_validation import case_mask_storage_proof


def memmap_data_sha256(
    array: np.ndarray,
    chunk_elements: int = 32 * 1024 * 1024,
) -> str:
    flat = array.reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.size, chunk_elements):
        chunk = np.ascontiguousarray(flat[offset : offset + chunk_elements])
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def validate(
    manifest_path: Path,
    candidate_id: str,
    runtime_root: Path,
    dataset_json: Path,
    seg_dir: Path,
    verify_array_hashes: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    candidates = {
        candidate["id"]: candidate for candidate in manifest["candidates"]
    }
    if candidate_id not in candidates:
        raise ValueError(f"unknown candidate ID: {candidate_id}")
    candidate = candidates[candidate_id]
    cases = load_dataset(dataset_json)
    candidate_root = runtime_root / "logits" / candidate_id
    export_manifest_path = candidate_root / "export_manifest.json"
    if not export_manifest_path.is_file():
        raise FileNotFoundError(f"missing export manifest: {export_manifest_path}")
    with export_manifest_path.open() as handle:
        export_manifest = json.load(handle)
    allowed_statuses = {"complete", "partial"} if allow_partial else {"complete"}
    if export_manifest.get("status") not in allowed_statuses:
        raise ValueError(
            f"{candidate_id}: export status is {export_manifest.get('status')!r}"
        )
    observed_cases = {
        case.get("name") for case in export_manifest.get("cases", [])
    }
    expected_cases = {case["name"] for case in cases}
    if allow_partial:
        if not observed_cases or not observed_cases <= expected_cases:
            raise ValueError(f"{candidate_id}: partial export has invalid cases")
        cases = [case for case in cases if case["name"] in observed_cases]
    elif observed_cases != expected_cases or len(observed_cases) != 200:
        raise ValueError(f"{candidate_id}: export has duplicate or missing cases")
    source_rows, _ = flatten_eval(candidate, verify_hash=True)

    values: list[float] = []
    hits = 0
    same_pass_values: list[float] = []
    same_pass_hits = 0
    mask_proofs: list[dict[str, Any]] = []
    case_results = []
    for case in cases:
        name = case["name"]
        array_path, metadata_path = case_output_paths(candidate_root, name)
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"{candidate_id} {name}: incomplete case output")
        with metadata_path.open() as handle:
            case_manifest = json.load(handle)
        logits = np.load(array_path, mmap_mode="r", allow_pickle=False)
        gt_img = nib.load(str(seg_dir / name))
        ground_truth = np.asanyarray(gt_img.dataobj)
        if logits.shape != ground_truth.shape:
            raise ValueError(
                f"{candidate_id} {name}: logits {logits.shape} != GT "
                f"{ground_truth.shape}"
            )
        if verify_array_hashes:
            actual_hash = memmap_data_sha256(logits)
            if actual_hash != case_manifest.get("array_sha256"):
                raise ValueError(
                    f"{candidate_id} {name}: logical array SHA256 mismatch"
                )
        reference = case_manifest.get("same_pass_reference")
        if not isinstance(reference, dict):
            raise ValueError(
                f"{candidate_id} {name}: missing same-pass reference metrics"
            )
        reference_values = reference.get("finding_dice")
        if (
            not isinstance(reference_values, list)
            or len(reference_values) != logits.shape[0]
        ):
            raise ValueError(
                f"{candidate_id} {name}: invalid same-pass finding metrics"
            )
        case_dice = []
        for finding_index in range(logits.shape[0]):
            value = mask_dice(
                ground_truth[finding_index],
                logits[finding_index] >= 0.0,
            )
            values.append(value)
            case_dice.append(value)
            hits += int(value >= 0.1)
            same_pass_values.append(float(reference_values[finding_index]))
        reference_case_hits = sum(
            float(value) >= 0.1 for value in reference_values
        )
        if int(reference.get("hits", -1)) != reference_case_hits:
            raise ValueError(
                f"{candidate_id} {name}: inconsistent same-pass hit count"
            )
        same_pass_hits += reference_case_hits
        mask_proof = case_mask_storage_proof(logits, case_manifest)
        mask_proofs.append(mask_proof)
        case_results.append(
            {
                "name": name,
                "findings": len(case_dice),
                "mean_dice": math.fsum(case_dice) / len(case_dice),
                "hits": sum(value >= 0.1 for value in case_dice),
                "same_pass_mean_dice": (
                    math.fsum(float(value) for value in reference_values)
                    / len(reference_values)
                ),
                "same_pass_hits": reference_case_hits,
                "same_pass_mask_reproduction": mask_proof,
            }
        )
    expected_finding_count = sum(len(case["findings"]) for case in cases)
    if len(values) != expected_finding_count:
        raise ValueError(
            f"{candidate_id}: expected {expected_finding_count} findings, "
            f"got {len(values)}"
        )
    mean_dice = math.fsum(values) / len(values)
    same_pass_mean_dice = math.fsum(same_pass_values) / len(same_pass_values)
    storage_dice_delta = mean_dice - same_pass_mean_dice
    exact_masks_passed = all(proof["passed"] for proof in mask_proofs)
    storage_passed = (
        abs(storage_dice_delta) <= 1e-6 and hits == same_pass_hits
        and exact_masks_passed
    )
    source_values = [
        float(source_rows[(case["name"], int(finding_index))]["dice"])
        for case in cases
        for finding_index in sorted(case["findings"], key=int)
    ]
    source_hits = sum(
        bool(source_rows[(case["name"], int(finding_index))]["hit"])
        for case in cases
        for finding_index in sorted(case["findings"], key=int)
    )
    expected_dice = math.fsum(source_values) / len(source_values)
    dice_delta = mean_dice - expected_dice
    historical_passed = abs(dice_delta) <= 1e-4 and hits == source_hits
    decision = reproduction_decision(storage_passed, historical_passed)
    passed = bool(decision["accepted"])
    result = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "status": "passed" if passed else "failed",
        "dtype": export_manifest["dtype"],
        "cases": len(cases),
        "findings": len(values),
        "mean_global_dice_per_finding": mean_dice,
        "total_hits": hits,
        "hit_rate": hits / len(values),
        "same_pass_mean_global_dice_per_finding": same_pass_mean_dice,
        "same_pass_total_hits": same_pass_hits,
        "storage_dice_delta": storage_dice_delta,
        "storage_dice_tolerance": 1e-6,
        "storage_exact_hit_count_required": True,
        "same_pass_exact_masks_required": True,
        "same_pass_mask_reproduction_status": (
            "passed" if exact_masks_passed else "failed"
        ),
        "same_pass_mask_reproduction_methods": sorted(
            {str(proof["method"]) for proof in mask_proofs}
        ),
        "same_pass_mask_mismatch_voxels": (
            sum(int(proof["mismatch_voxels"]) for proof in mask_proofs)
            if all(proof["mismatch_voxels"] is not None for proof in mask_proofs)
            else None
        ),
        "legacy_stored_zero_voxels": sum(
            int(proof.get("stored_zero_voxels", 0)) for proof in mask_proofs
        ),
        "storage_reproduction_status": (
            "passed" if storage_passed else "failed"
        ),
        "expected_mean_global_dice_per_finding": expected_dice,
        "expected_total_hits": source_hits,
        "dice_delta": dice_delta,
        "dice_tolerance": 1e-4,
        "exact_hit_count_required": True,
        "historical_reproduction_status": (
            "passed" if historical_passed else "warning"
        ),
        "historical_reproduction_is_gate": False,
        "warnings": decision["warnings"],
        "array_hashes_verified": verify_array_hashes,
        "case_results": case_results,
    }
    output_name = (
        "smoke_reproduction_validation.json"
        if allow_partial
        else "reproduction_validation.json"
    )
    atomic_write_json(candidate_root / output_name, result)
    if not passed:
        raise RuntimeError(
            f"{candidate_id}: storage reproduction failed: Dice delta "
            f"{storage_dice_delta:+.8f}, hits {hits} != {same_pass_hits}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=Path("/data/hengjie/datasets/rexgroundingct/segmentations"),
    )
    parser.add_argument("--verify-array-hashes", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = validate(
        manifest_path=args.manifest,
        candidate_id=args.candidate_id,
        runtime_root=args.runtime_root,
        dataset_json=args.dataset_json,
        seg_dir=args.seg_dir,
        verify_array_hashes=args.verify_array_hashes,
        allow_partial=args.allow_partial,
    )
    print(
        f"{args.candidate_id}: reproduced Dice "
        f"{result['mean_global_dice_per_finding']:.6f}, "
        f"hits {result['total_hits']}/{result['findings']}, "
        f"historical={result['historical_reproduction_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
