#!/usr/bin/env python3
"""Prepare the fixed val200 case plan for the exp010 anatomy audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL_JSON = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
DEFAULT_CT_ROOT = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_total_fast_3mm_v2_16_0"
)
DEFAULT_EXP_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion"
)
EXPECTED_VAL_SHA256 = "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ct_rate_path(filename: str, ct_root: Path) -> Path:
    if not filename.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz filename, got {filename!r}")
    stem = filename[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE filename: {filename!r}")
    split = parts[0]
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    return ct_root / "dataset" / f"{split}_fixed" / patient / scan / filename


def case_key(filename: str) -> str:
    return filename[: -len(".nii.gz")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--expected-val-sha256", default=EXPECTED_VAL_SHA256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    observed_val_sha = sha256_file(args.val_json)
    if observed_val_sha != args.expected_val_sha256:
        raise SystemExit(
            "Fixed val JSON hash mismatch: "
            f"expected={args.expected_val_sha256} observed={observed_val_sha}"
        )

    payload = json.loads(args.val_json.read_text())
    entries = payload.get("test")
    if not isinstance(entries, list) or len(entries) != args.expected_cases:
        raise SystemExit(
            f"Expected {args.expected_cases} entries under 'test', got "
            f"{len(entries) if isinstance(entries, list) else type(entries).__name__}"
        )

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    shapes: list[tuple[int, int, int]] = []
    spacings: list[tuple[float, float, float]] = []
    for index, entry in enumerate(entries):
        name = entry["name"]
        if name in seen:
            raise SystemExit(f"Duplicate validation case: {name}")
        seen.add(name)
        ct_path = ct_rate_path(name, args.ct_root)
        if not ct_path.is_file():
            missing.append(str(ct_path))
            continue
        image = nib.load(str(ct_path))
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        shapes.append(shape)
        spacings.append(spacing)
        rows.append(
            {
                "val_order": index,
                "case_key": case_key(name),
                "name": name,
                "ct_path": str(ct_path),
                "source_shape_xyz": list(shape),
                "source_spacing_xyz_mm": list(spacing),
                "source_axcodes": list(nib.aff2axcodes(image.affine)),
                "finding_count": len(entry.get("findings", {})),
                "categories": entry.get("categories", {}),
                "findings": entry.get("findings", {}),
            }
        )

    if missing:
        sample = "\n".join(missing[:20])
        raise SystemExit(f"{len(missing)} validation CT files are missing:\n{sample}")

    plan_path = args.exp_dir / "config/val200_totalsegmentator_case_plan.jsonl"
    plan_manifest_path = args.exp_dir / "config/val200_totalsegmentator_case_plan.json"
    if plan_path.exists() and not args.overwrite:
        existing_sha = sha256_file(plan_path)
        print(f"Case plan already exists: {plan_path} sha256={existing_sha}")
        return 0

    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "cases").mkdir(parents=True, exist_ok=True)
    for folder in ["config", "logs", "reports", "visualizations", "runtime"]:
        (args.exp_dir / folder).mkdir(parents=True, exist_ok=True)

    write_jsonl_atomic(plan_path, rows)
    shape_axes = list(zip(*shapes, strict=True))
    spacing_axes = list(zip(*spacings, strict=True))
    manifest = {
        "created_at_utc": utc_now(),
        "purpose": "exp010 TotalSegmentator total_fast val200 label inventory",
        "val_json": str(args.val_json),
        "val_json_sha256": observed_val_sha,
        "ct_root": str(args.ct_root),
        "cache_root": str(args.cache_root),
        "experiment_dir": str(args.exp_dir),
        "cases": len(rows),
        "plan_jsonl": str(plan_path),
        "plan_jsonl_sha256": sha256_file(plan_path),
        "source_shape_xyz": {
            "min": [min(axis) for axis in shape_axes],
            "max": [max(axis) for axis in shape_axes],
        },
        "source_spacing_xyz_mm": {
            "min": [min(axis) for axis in spacing_axes],
            "max": [max(axis) for axis in spacing_axes],
        },
        "inference": {
            "task": "total",
            "fast": True,
            "model_spacing_mm": 3.0,
            "save_lowres": False,
            "multilabel": True,
            "statistics_extra": True,
        },
    }
    write_json_atomic(plan_manifest_path, manifest)
    write_json_atomic(args.cache_root / "plan_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
