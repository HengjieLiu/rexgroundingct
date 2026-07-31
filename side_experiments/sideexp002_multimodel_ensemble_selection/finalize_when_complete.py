#!/usr/bin/env python3
"""Wait for all exports, write the strict audit, then run ensemble selection."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze_candidates import atomic_write_json, load_manifest
from audit_logit_exports import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_MD,
    audit_candidate,
)
from launch_logit_exports import DEFAULT_RUNTIME_ROOT


HERE = Path(__file__).resolve().parent
AUDITOR = HERE / "audit_logit_exports.py"
SEARCH = HERE / "search_ensembles.py"


def acceptance_snapshot(
    manifest_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = [
        audit_candidate(candidate, runtime_root)
        for candidate in manifest["candidates"]
    ]
    return {
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "total": len(rows),
        "incomplete_candidate_ids": [
            row["candidate_id"] for row in rows if not row["accepted"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--audit-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=48 * 60 * 60)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Defaults to <runtime-root>/finalization_manifest.json.",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_seconds < args.poll_seconds:
        raise ValueError("invalid poll or timeout interval")
    state_path = (
        args.state
        if args.state is not None
        else args.runtime_root / "finalization_manifest.json"
    )
    started = time.time()
    while True:
        snapshot = acceptance_snapshot(args.manifest, args.runtime_root)
        state = {
            "schema_version": 1,
            "status": "waiting_for_exports",
            "historical_reproduction_is_gate": False,
            "test_inference_performed": False,
            **snapshot,
        }
        atomic_write_json(state_path, state)
        if snapshot["accepted"] == snapshot["total"]:
            break
        if time.time() - started >= args.timeout_seconds:
            state["status"] = "timed_out"
            atomic_write_json(state_path, state)
            raise TimeoutError("timed out waiting for all candidate exports")
        time.sleep(args.poll_seconds)

    audit_command = [
        sys.executable,
        str(AUDITOR),
        "--manifest",
        str(args.manifest),
        "--runtime-root",
        str(args.runtime_root),
        "--output-json",
        str(args.audit_json),
        "--output-markdown",
        str(args.audit_markdown),
    ]
    subprocess.run(audit_command, check=True)
    state["status"] = "audit_passed"
    atomic_write_json(state_path, state)
    if not args.skip_search:
        state["status"] = "ensemble_search_running"
        atomic_write_json(state_path, state)
        subprocess.run(
            [
                sys.executable,
                str(SEARCH),
                "--manifest",
                str(args.manifest),
                "--runtime-root",
                str(args.runtime_root),
            ],
            check=True,
        )
    state["status"] = "complete"
    state["ensemble_search_performed"] = not args.skip_search
    atomic_write_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
