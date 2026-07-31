#!/usr/bin/env python3
"""Run resumable sideexp002 logit exports with one candidate per gated GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from analyze_candidates import DEFAULT_MANIFEST, atomic_write_json, load_manifest


HERE = Path(__file__).resolve().parent
EXPORTER = HERE / "export_logits.py"
VALIDATOR = HERE / "validate_logit_export.py"
DEFAULT_GPUS = (0, 1, 2, 3)
DEFAULT_MIN_FREE_MIB = 32768
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp002_multimodel_ensemble_selection"
)
SMOKE_CANDIDATES = (
    "exp009_baseline_e100",
    "exp008_shared_e100",
    "exp009_s3v2_e100",
    "exp011_clip_zscore_e100",
    "exp011_linear_hu_e100",
)


def gpu_free_mib() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in completed.stdout.splitlines():
        index, free = [part.strip() for part in line.split(",", maxsplit=1)]
        values[int(index)] = int(free)
    return values


def export_complete(runtime_root: Path, candidate_id: str) -> bool:
    path = runtime_root / "logits" / candidate_id / "export_manifest.json"
    validation = (
        runtime_root / "logits" / candidate_id / "reproduction_validation.json"
    )
    if not path.is_file() or not validation.is_file():
        return False
    try:
        with path.open() as handle:
            manifest = json.load(handle)
        with validation.open() as handle:
            gate = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("case_count", -1)) == 200
        and int(manifest.get("finding_count", -1)) == 381
        and gate.get("status") == "passed"
    )


def wait_for_gpu(gpu: int, min_free_mib: int, poll_seconds: int) -> None:
    while True:
        free = gpu_free_mib()
        if gpu not in free:
            raise RuntimeError(f"GPU {gpu} is not visible to nvidia-smi")
        if free[gpu] >= min_free_mib:
            return
        print(
            f"GPU {gpu}: {free[gpu]} MiB free; waiting for {min_free_mib} MiB",
            flush=True,
        )
        time.sleep(poll_seconds)


@contextmanager
def gpu_export_lock(runtime_root: Path, gpu: int):
    """Prevent two Side Experiment 002 inference workers on one GPU."""
    path = runtime_root / "locks" / f"gpu{gpu}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Side Experiment 002 worker owns GPU {gpu}: {path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "gpu": gpu,
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "command": sys.argv,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def storage_retry_required(runtime_root: Path, candidate_id: str) -> bool:
    path = (
        runtime_root
        / "logits"
        / candidate_id
        / "reproduction_validation.json"
    )
    if not path.is_file():
        return False
    try:
        with path.open() as handle:
            gate = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        gate.get("status") == "failed"
        and gate.get("dtype") == "float16"
        and gate.get("storage_reproduction_status") == "failed"
    )


def memory_wait_enabled(args: argparse.Namespace) -> bool:
    return not bool(args.no_memory_wait)


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )


def run_candidate(
    candidate_id: str,
    gpu: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runtime_root = args.runtime_root / "smoke" if args.smoke else args.runtime_root
    if (
        not args.smoke
        and export_complete(runtime_root, candidate_id)
        and not args.overwrite
    ):
        return {"candidate_id": candidate_id, "gpu": gpu, "status": "skipped_complete"}
    if memory_wait_enabled(args):
        wait_for_gpu(gpu, args.min_free_mib, args.poll_seconds)
    log_path = runtime_root / "logs" / f"{candidate_id}.log"
    command = [
        sys.executable,
        str(EXPORTER),
        "--manifest",
        str(args.manifest),
        "--candidate-id",
        candidate_id,
        "--runtime-root",
        str(runtime_root),
        "--gpu",
        str(gpu),
        "--dtype",
        "float16",
    ]
    if args.embeddings is not None:
        command.extend(["--embeddings", str(args.embeddings)])
    if args.skip_checkpoint_hash:
        command.append("--skip-checkpoint-hash")
    if args.overwrite:
        command.append("--overwrite")
    if args.smoke:
        command.extend(["--limit", "1"])
    run_command(command, log_path)
    validation_command = [
        sys.executable,
        str(VALIDATOR),
        "--manifest",
        str(args.manifest),
        "--candidate-id",
        candidate_id,
        "--runtime-root",
        str(runtime_root),
        "--verify-array-hashes",
    ]
    if args.smoke:
        validation_command.append("--allow-partial")
    try:
        run_command(validation_command, log_path)
        return {
            "candidate_id": candidate_id,
            "gpu": gpu,
            "status": "complete_float16",
            "smoke": args.smoke,
        }
    except RuntimeError:
        if not storage_retry_required(runtime_root, candidate_id):
            raise
        print(
            f"{candidate_id}: float16 storage reproduction failed; "
            "retrying float32",
            flush=True,
        )
        if memory_wait_enabled(args):
            wait_for_gpu(gpu, args.min_free_mib, args.poll_seconds)
        fallback = command.copy()
        dtype_index = fallback.index("float16")
        fallback[dtype_index] = "float32"
        if "--overwrite" not in fallback:
            fallback.append("--overwrite")
        run_command(fallback, log_path)
        run_command(validation_command, log_path)
        return {
            "candidate_id": candidate_id,
            "gpu": gpu,
            "status": "complete_float32_fallback",
            "smoke": args.smoke,
        }


def worker(
    gpu: int,
    candidates: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    runtime_root = args.runtime_root / "smoke" if args.smoke else args.runtime_root
    with gpu_export_lock(runtime_root, gpu):
        for candidate_id in candidates:
            try:
                result = run_candidate(candidate_id, gpu, args)
            except Exception as exc:  # noqa: BLE001 - isolate candidate failures
                result = {
                    "candidate_id": candidate_id,
                    "gpu": gpu,
                    "status": "failed",
                    "smoke": args.smoke,
                    "error": str(exc),
                }
            results.append(result)
            atomic_write_json(
                runtime_root / "queue_results" / f"{candidate_id}.json",
                result,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument(
        "--launch-manifest",
        type=Path,
        default=None,
        help="Optional isolated launcher-state path for a recovery queue.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--no-memory-wait",
        action="store_true",
        help="Start immediately without polling free GPU memory.",
    )
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Export one case for standard, dual, S3, clipped-z-score, and linear-HU representatives.",
    )
    args = parser.parse_args()
    if args.min_free_mib < 1024:
        raise ValueError("--min-free-mib is implausibly small")
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain unique GPU indices")
    manifest = load_manifest(args.manifest)
    known = [candidate["id"] for candidate in manifest["candidates"]]
    candidates = args.candidate_id or (
        list(SMOKE_CANDIDATES) if args.smoke else known
    )
    unknown = set(candidates) - set(known)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    assignments = {gpu: [] for gpu in gpus}
    for index, candidate_id in enumerate(candidates):
        assignments[gpus[index % len(gpus)]].append(candidate_id)
    launch_root = args.runtime_root / "smoke" if args.smoke else args.runtime_root
    launch_root.mkdir(parents=True, exist_ok=True)
    launch_manifest = {
        "schema_version": 1,
        "command": sys.argv,
        "hostname": os.uname().nodename,
        "gpus": list(gpus),
        "min_free_mib": args.min_free_mib,
        "memory_wait_enabled": not args.no_memory_wait,
        "assignments": assignments,
        "status": "running",
        "smoke": args.smoke,
    }
    launch_path = (
        args.launch_manifest
        if args.launch_manifest is not None
        else launch_root / "launch_manifest.json"
    )
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(launch_path, launch_manifest)
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [
                pool.submit(worker, gpu, assignments[gpu], args)
                for gpu in gpus
                if assignments[gpu]
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        launch_manifest["status"] = (
            "complete_with_failures"
            if any(result["status"] == "failed" for result in results)
            else "complete"
        )
        launch_manifest["results"] = sorted(
            results,
            key=lambda item: item["candidate_id"],
        )
        atomic_write_json(launch_path, launch_manifest)
    except Exception as exc:
        launch_manifest["status"] = "failed"
        launch_manifest["error"] = str(exc)
        launch_manifest["results"] = results
        atomic_write_json(launch_path, launch_manifest)
        raise
    print(f"Completed {len(results)} candidate exports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
