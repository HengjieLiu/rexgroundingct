#!/usr/bin/env python3
"""Revalidate existing exports without running inference, continuing on errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from analyze_candidates import atomic_write_json, load_manifest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "candidate_manifest.json"
DEFAULT_DATASET = (
    REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp002_multimodel_ensemble_selection"
)


def revalidate_candidates(
    candidate_ids: list[str],
    validator: Callable[..., dict[str, Any]],
    validator_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for candidate_id in candidate_ids:
        try:
            result = validator(candidate_id=candidate_id, **validator_kwargs)
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "dtype": result["dtype"],
                    "historical_reproduction_status": result[
                        "historical_reproduction_status"
                    ],
                    "warnings": result["warnings"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - fail-soft audit queue
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return results


def main() -> int:
    from validate_logit_export import validate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=Path("/data/hengjie/datasets/rexgroundingct/segmentations"),
    )
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <runtime-root>/completed_revalidation_manifest.json.",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    known = {candidate["id"] for candidate in manifest["candidates"]}
    unknown = set(args.candidate_id) - known
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    if len(set(args.candidate_id)) != len(args.candidate_id):
        raise ValueError("duplicate --candidate-id values")
    results = revalidate_candidates(
        args.candidate_id,
        validate,
        {
            "manifest_path": args.manifest,
            "runtime_root": args.runtime_root,
            "dataset_json": args.dataset_json,
            "seg_dir": args.seg_dir,
            "verify_array_hashes": True,
            "allow_partial": False,
        },
    )
    output = (
        args.output
        if args.output is not None
        else args.runtime_root / "completed_revalidation_manifest.json"
    )
    record = {
        "schema_version": 1,
        "historical_reproduction_is_gate": False,
        "inference_performed": False,
        "candidate_ids": args.candidate_id,
        "status": (
            "complete_with_failures"
            if any(result["status"] == "failed" for result in results)
            else "complete"
        ),
        "results": results,
    }
    atomic_write_json(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
