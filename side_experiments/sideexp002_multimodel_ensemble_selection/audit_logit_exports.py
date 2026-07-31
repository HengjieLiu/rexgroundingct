#!/usr/bin/env python3
"""Generate a deterministic all-candidate Side Experiment 002 export audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_candidates import (
    atomic_write_json,
    atomic_write_text,
    load_manifest,
    sha256_file,
)
from launch_logit_exports import DEFAULT_RUNTIME_ROOT


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "candidate_manifest.json"
DEFAULT_OUTPUT_JSON = HERE / "outputs" / "logit_export_audit.json"
DEFAULT_OUTPUT_MD = HERE / "outputs" / "logit_export_audit.md"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_candidate(
    candidate: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    candidate_id = candidate["id"]
    root = runtime_root / "logits" / candidate_id
    export_path = root / "export_manifest.json"
    gate_path = root / "reproduction_validation.json"
    if not export_path.is_file():
        return {
            "candidate_id": candidate_id,
            "status": "not_started",
            "accepted": False,
            "warnings": [],
        }
    export = read_json(export_path)
    cases = export.get("cases", [])
    names = [row.get("name") for row in cases if isinstance(row, dict)]
    gate = read_json(gate_path) if gate_path.is_file() else None
    ranges = [
        row["unclipped_range"]
        for row in cases
        if isinstance(row, dict)
        and isinstance(row.get("unclipped_range"), list)
        and len(row["unclipped_range"]) == 2
    ]
    complete = (
        export.get("status") == "complete"
        and int(export.get("case_count", -1)) == 200
        and int(export.get("finding_count", -1)) == 381
        and len(names) == 200
        and len(set(names)) == 200
    )
    source_matches = export.get("candidate_source") == candidate
    storage_passed = (
        gate is not None
        and gate.get("status") == "passed"
        and gate.get("storage_reproduction_status") == "passed"
        and gate.get("same_pass_mask_reproduction_status") == "passed"
        and gate.get("historical_reproduction_is_gate") is False
        and gate.get("array_hashes_verified") is True
    )
    accepted = complete and source_matches and storage_passed
    warnings = list(gate.get("warnings", [])) if gate else []
    array_records = [
        {
            "name": row.get("name"),
            "array_sha256": row.get("array_sha256"),
            "shape": row.get("shape"),
            "dtype": row.get("dtype"),
        }
        for row in cases
        if isinstance(row, dict)
    ]
    return {
        "candidate_id": candidate_id,
        "status": (
            "accepted"
            if accepted
            else "complete_unaccepted"
            if complete
            else str(export.get("status", "incomplete"))
        ),
        "accepted": accepted,
        "dtype": export.get("dtype"),
        "cases": len(names),
        "findings": export.get("finding_count"),
        "unique_cases": len(set(names)),
        "npy_bytes": sum(
            int(row.get("npy_bytes", 0))
            for row in cases
            if isinstance(row, dict)
        ),
        "unclipped_range": (
            [
                min(float(value[0]) for value in ranges),
                max(float(value[1]) for value in ranges),
            ]
            if ranges
            else None
        ),
        "clip_low_count": sum(
            int(row.get("clip_low_count", 0))
            for row in cases
            if isinstance(row, dict)
        ),
        "clip_high_count": sum(
            int(row.get("clip_high_count", 0))
            for row in cases
            if isinstance(row, dict)
        ),
        "source_matches_manifest": source_matches,
        "storage_reproduction_status": (
            gate.get("storage_reproduction_status") if gate else "missing"
        ),
        "same_pass_mask_reproduction_status": (
            gate.get("same_pass_mask_reproduction_status")
            if gate
            else "missing"
        ),
        "same_pass_mask_reproduction_methods": (
            gate.get("same_pass_mask_reproduction_methods", [])
            if gate
            else []
        ),
        "array_hashes_verified": (
            bool(gate.get("array_hashes_verified")) if gate else False
        ),
        "array_hash_record_count": sum(
            isinstance(row.get("array_sha256"), str)
            for row in cases
            if isinstance(row, dict)
        ),
        "array_and_geometry_records_sha256": stable_sha256(array_records),
        "source_hashes": {
            "checkpoint_sha256": candidate["checkpoint"]["sha256"],
            "config_sha256": candidate["config"]["sha256"],
            "cache_manifest_sha256": candidate["cache"]["manifest_sha256"],
            "evaluation_sha256": candidate["evaluation"]["sha256"],
            "dataset_sha256": export.get("dataset_sha256"),
        },
        "historical_reproduction_status": (
            gate.get("historical_reproduction_status") if gate else "missing"
        ),
        "historical_reproduction_is_gate": (
            gate.get("historical_reproduction_is_gate") if gate else None
        ),
        "historical_dice_delta": gate.get("dice_delta") if gate else None,
        "observed_hits": gate.get("total_hits") if gate else None,
        "historical_hits": gate.get("expected_total_hits") if gate else None,
        "warnings": warnings,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Side Experiment 002 Logit Export Audit",
        "",
        f"- Accepted: `{summary['accepted_candidates']} / "
        f"{summary['total_candidates']}`",
        f"- Complete but unaccepted: `{summary['complete_unaccepted']}`",
        f"- Incomplete or not started: `{summary['incomplete_candidates']}`",
        f"- Historical warnings: `{summary['historical_warning_candidates']}`",
        "",
        "Historical reproduction is diagnostic. Acceptance requires complete "
        "arrays, exact same-pass storage reproduction, verified array hashes, "
        "and unchanged candidate provenance.",
        "",
        "| Candidate | Status | Dtype | Cases | Findings | Storage | Masks | "
        "Historical | Dice delta | Hits | Logit range | Clips L/H | "
        "Provenance | Array hashes | Warnings |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | "
        "--- | ---: | --- | ---: | --- |",
    ]
    for row in summary["candidates"]:
        delta = (
            f"{row['historical_dice_delta']:+.8f}"
            if row.get("historical_dice_delta") is not None
            else "-"
        )
        hits = (
            f"{row['observed_hits']} / {row['historical_hits']}"
            if row.get("observed_hits") is not None
            else "-"
        )
        value_range = (
            f"{row['unclipped_range'][0]:.6f} to "
            f"{row['unclipped_range'][1]:.6f}"
            if row.get("unclipped_range") is not None
            else "-"
        )
        provenance = (
            "match"
            if row.get("source_matches_manifest") is True
            else "mismatch"
            if row.get("source_matches_manifest") is False
            else "-"
        )
        lines.append(
            f"| `{row['candidate_id']}` | {row['status']} | "
            f"{row.get('dtype', '-')} | {row.get('cases', 0)} | "
            f"{row.get('findings') or 0} | "
            f"{row.get('storage_reproduction_status') or '-'} | "
            f"{row.get('same_pass_mask_reproduction_status') or '-'} | "
            f"{row.get('historical_reproduction_status') or '-'} | "
            f"{delta} | {hits} | "
            f"{value_range} | {row.get('clip_low_count', 0)} / "
            f"{row.get('clip_high_count', 0)} | "
            f"{provenance} | "
            f"{row.get('array_hash_record_count', 0)} | "
            f"{', '.join(row.get('warnings', [])) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(
    manifest_path: Path,
    runtime_root: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    candidates = [
        audit_candidate(candidate, runtime_root)
        for candidate in manifest["candidates"]
    ]
    summary = {
        "schema_version": 1,
        "side_experiment": "sideexp002_multimodel_ensemble_selection",
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "runtime_root": str(runtime_root),
        "total_candidates": len(candidates),
        "accepted_candidates": sum(row["accepted"] for row in candidates),
        "complete_unaccepted": sum(
            row["status"] == "complete_unaccepted" for row in candidates
        ),
        "incomplete_candidates": sum(
            row["status"] not in {"accepted", "complete_unaccepted"}
            for row in candidates
        ),
        "historical_warning_candidates": sum(
            "historical_reproduction_mismatch" in row.get("warnings", [])
            for row in candidates
        ),
        "candidates": candidates,
    }
    atomic_write_json(output_json, summary)
    atomic_write_text(output_markdown, build_report(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.manifest,
        args.runtime_root,
        args.output_json,
        args.output_markdown,
    )
    print(
        f"Accepted {summary['accepted_candidates']}/"
        f"{summary['total_candidates']} candidates."
    )
    if (
        summary["accepted_candidates"] != summary["total_candidates"]
        and not args.allow_incomplete
    ):
        raise RuntimeError("logit export audit is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
