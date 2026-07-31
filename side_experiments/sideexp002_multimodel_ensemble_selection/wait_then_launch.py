#!/usr/bin/env python3
"""Start a no-memory-wait queue after one existing candidate finishes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from launch_logit_exports import DEFAULT_MANIFEST, DEFAULT_RUNTIME_ROOT


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "launch_logit_exports.py"


def candidate_finished(runtime_root: Path, candidate_id: str) -> bool:
    root = runtime_root / "logits" / candidate_id
    export_path = root / "export_manifest.json"
    gate_path = root / "reproduction_validation.json"
    if (root / ".export.lock").exists():
        return False
    if not export_path.is_file() or not gate_path.is_file():
        return False
    try:
        with export_path.open() as handle:
            export = json.load(handle)
        with gate_path.open() as handle:
            gate = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        export.get("status") == "complete"
        and int(export.get("case_count", -1)) == 200
        and int(export.get("finding_count", -1)) == 381
        and gate.get("dtype") == export.get("dtype")
        and int(gate.get("cases", -1)) == 200
        and int(gate.get("findings", -1)) == 381
        and gate.get("status") in {"passed", "failed"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--wait-for-candidate", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=6 * 60 * 60)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_seconds < args.poll_seconds:
        raise ValueError("invalid poll or timeout interval")
    deadline = time.monotonic() + args.timeout_seconds
    while not candidate_finished(args.runtime_root, args.wait_for_candidate):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {args.wait_for_candidate}"
            )
        time.sleep(args.poll_seconds)
    command = [
        sys.executable,
        str(LAUNCHER),
        "--manifest",
        str(args.manifest),
        "--runtime-root",
        str(args.runtime_root),
        "--gpus",
        str(args.gpu),
        "--no-memory-wait",
        "--launch-manifest",
        str(args.launch_manifest),
    ]
    for candidate_id in args.candidate_id:
        command.extend(["--candidate-id", candidate_id])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
