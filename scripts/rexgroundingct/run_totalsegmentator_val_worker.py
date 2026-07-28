#!/usr/bin/env python3
"""Run one resumable TotalSegmentator shard for the exp010 val200 audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_class_map(task: str) -> dict[int, str]:
    output = subprocess.check_output(
        ["totalseg_info", "--classes", "-ta", task, "--json"],
        text=True,
    )
    return {int(key): value for key, value in json.loads(output).items()}


def gpu_memory_mib() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU, observed {len(rows)}: {rows}")
    used, free = [int(value.strip()) for value in rows[0].split(",")]
    return used, free


def wait_for_gpu(
    minimum_free_mib: int,
    poll_seconds: int,
    stability_checks: int,
) -> None:
    passing_checks = 0
    while True:
        used, free = gpu_memory_mib()
        passing_checks = passing_checks + 1 if free >= minimum_free_mib else 0
        print(
            f"{utc_now()} GPU wait check: used={used} MiB free={free} MiB "
            f"required_free={minimum_free_mib} MiB "
            f"stable={passing_checks}/{stability_checks}",
            flush=True,
        )
        if passing_checks >= stability_checks:
            return
        time.sleep(poll_seconds)


class GPUMemoryMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                used, _ = gpu_memory_mib()
                self.samples.append(used)
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "GPUMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def acquire_case_lock(lock_dir: Path, stale_seconds: int) -> bool:
    try:
        lock_dir.mkdir()
    except FileExistsError:
        age = time.time() - lock_dir.stat().st_mtime
        if age <= stale_seconds:
            return False
        shutil.rmtree(lock_dir)
        lock_dir.mkdir()
    write_json_atomic(
        lock_dir / "owner.json",
        {"pid": os.getpid(), "hostname": os.uname().nodename, "created_at_utc": utc_now()},
    )
    return True


def validate_output(
    source_path: Path,
    output_path: Path,
    class_map: dict[int, str],
) -> dict[str, Any]:
    source = nib.load(str(source_path))
    output = nib.load(str(output_path))
    if tuple(output.shape) != tuple(source.shape):
        raise ValueError(f"Shape mismatch: output={output.shape} source={source.shape}")
    if not np.allclose(output.affine, source.affine, atol=1e-5, rtol=1e-5):
        raise ValueError("Output affine does not match source CT affine")
    data = np.asarray(output.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("Output contains non-finite values")
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise ValueError("Multilabel output contains non-integer values")
    labels, counts = np.unique(rounded.astype(np.uint8, copy=False), return_counts=True)
    invalid = [int(label) for label in labels if int(label) != 0 and int(label) not in class_map]
    if invalid:
        raise ValueError(f"Output contains unknown label IDs: {invalid}")
    voxel_volume_mm3 = float(np.prod(source.header.get_zooms()[:3]))
    class_voxels = {
        str(int(label)): int(count)
        for label, count in zip(labels, counts, strict=True)
        if int(label) != 0
    }
    return {
        "source_shape_xyz": list(source.shape),
        "output_shape_xyz": list(output.shape),
        "source_spacing_xyz_mm": [float(value) for value in source.header.get_zooms()[:3]],
        "output_spacing_xyz_mm": [float(value) for value in output.header.get_zooms()[:3]],
        "source_affine": np.asarray(source.affine).tolist(),
        "output_affine": np.asarray(output.affine).tolist(),
        "source_axcodes": list(nib.aff2axcodes(source.affine)),
        "output_axcodes": list(nib.aff2axcodes(output.affine)),
        "output_dtype": str(output.get_data_dtype()),
        "voxel_volume_mm3": voxel_volume_mm3,
        "observed_label_ids": [int(label) for label in labels if int(label) != 0],
        "observed_class_count": sum(int(label) != 0 for label in labels),
        "class_voxels": class_voxels,
        "class_volume_mm3": {
            label: float(count * voxel_volume_mm3) for label, count in class_voxels.items()
        },
    }


def run_case(
    row: dict[str, Any],
    cache_root: Path,
    class_map: dict[int, str],
    task: str,
    fast: bool,
    output_name: str,
    roi_subset: list[str] | None,
    overwrite: bool,
    require_device: str | None,
) -> str:
    case_dir = cache_root / "cases" / row["case_key"]
    legacy_output = output_name == "total_labels.nii.gz"
    if output_name.endswith(".nii.gz"):
        output_prefix = output_name[: -len(".nii.gz")]
    else:
        raise ValueError(f"output_name must end with .nii.gz, got {output_name!r}")
    if output_prefix.endswith("_labels"):
        output_prefix = output_prefix[: -len("_labels")]

    complete_path = case_dir / (".complete" if legacy_output else f".complete_{output_prefix}")
    output_path = case_dir / output_name
    metadata_path = case_dir / (
        "metadata.json" if legacy_output else f"{output_prefix}_metadata.json"
    )
    statistics_path = case_dir / (
        "statistics.json" if legacy_output else f"{output_prefix}_statistics.json"
    )
    report_path = case_dir / (
        "run_report.json" if legacy_output else f"{output_prefix}_run_report.json"
    )
    log_path = case_dir / (
        "inference.log" if legacy_output else f"{output_prefix}_inference.log"
    )
    failure_path = case_dir / (
        "failure.json" if legacy_output else f"{output_prefix}_failure.json"
    )
    if (
        not overwrite
        and complete_path.is_file()
        and output_path.is_file()
        and metadata_path.is_file()
    ):
        return "skipped_complete"

    case_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and complete_path.exists():
        complete_path.unlink()
    lock_dir = case_dir / (".case.lock" if legacy_output else f".case_{output_prefix}.lock")
    if not acquire_case_lock(lock_dir, stale_seconds=12 * 60 * 60):
        return "skipped_locked"

    work_dir = case_dir / f".work-{uuid.uuid4().hex}"
    work_dir.mkdir()
    work_output = work_dir / "total_labels.nii.gz"
    work_statistics = work_dir / "statistics.json"
    work_report = work_dir / "run_report.json"
    work_log = work_dir / "inference.log"
    source_path = Path(row["ct_path"])
    started_at = utc_now()
    started = time.monotonic()
    baseline_used, baseline_free = gpu_memory_mib()
    command = [
        "TotalSegmentator",
        "-i",
        str(source_path),
        "-o",
        str(work_output),
        "--task",
        task,
        "--device",
        "gpu",
        "--ml",
        "--statistics",
        str(work_statistics),
        "--statistics_extra",
        "--report",
        str(work_report),
        "--resampling_order",
        "1",
        "--nr_thr_resamp",
        "1",
        "--nr_thr_saving",
        "1",
        "--quiet",
    ]
    if fast:
        command.append("--fast")
    if roi_subset:
        command.append("--roi_subset")
        command.extend(roi_subset)

    try:
        with work_log.open("w") as log_handle, GPUMemoryMonitor() as monitor:
            log_handle.write(f"started_at_utc={started_at}\n")
            log_handle.write("command=" + " ".join(command) + "\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError(f"TotalSegmentator exited with code {result.returncode}")
        for required in [work_output, work_statistics, work_report]:
            if not required.is_file():
                raise FileNotFoundError(f"Missing TotalSegmentator artifact: {required}")

        run_report = json.loads(work_report.read_text())
        report_classes = run_report.get("classes")
        if isinstance(report_classes, dict) and report_classes:
            validation_class_map = {
                int(key): value for key, value in report_classes.items()
            }
        else:
            validation_class_map = class_map
        geometry = validate_output(source_path, work_output, validation_class_map)
        resolved_device = run_report.get("device")
        if require_device is not None and resolved_device != require_device:
            raise RuntimeError(
                f"Resolved device mismatch: required={require_device!r} "
                f"observed={resolved_device!r}"
            )
        peak_total_mib = max(monitor.samples, default=baseline_used)
        metadata = {
            "completed_at_utc": utc_now(),
            "started_at_utc": started_at,
            "case": row,
            "task": task,
            "fast": fast,
            "roi_subset": roi_subset,
            "model_spacing_mm": model_spacing_mm(task, fast),
            "save_lowres": False,
            "multilabel": True,
            "class_map": {str(key): value for key, value in class_map.items()},
            "output_class_map": {
                str(key): value for key, value in validation_class_map.items()
            },
            "configured_class_count": len(class_map),
            "output_class_count": len(validation_class_map),
            "geometry": geometry,
            "source_ct_sha256": sha256_file(source_path),
            "output_sha256": sha256_file(work_output),
            "runtime_seconds": elapsed,
            "gpu_memory_baseline_used_mib": baseline_used,
            "gpu_memory_baseline_free_mib": baseline_free,
            "gpu_memory_peak_total_used_mib": peak_total_mib,
            "gpu_memory_peak_delta_mib": max(0, peak_total_mib - baseline_used),
            "official_run_report_versions": {
                key: run_report.get(key)
                for key in [
                    "totalsegmentator_version",
                    "nnunetv2_version",
                    "torch_version",
                    "device",
                ]
            },
            "command": command,
        }
        write_json_atomic(work_dir / "metadata.json", metadata)

        final_files = {
            work_output: output_path,
            work_statistics: statistics_path,
            work_report: report_path,
            work_log: log_path,
            work_dir / "metadata.json": metadata_path,
        }
        for source, destination in final_files.items():
            os.replace(source, destination)
        if failure_path.exists():
            failure_path.unlink()
        write_json_atomic(
            complete_path,
            {
                "completed_at_utc": metadata["completed_at_utc"],
                "output_sha256": metadata["output_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
            },
        )
        work_dir.rmdir()
        return "completed"
    except Exception as error:
        failure = {
            "failed_at_utc": utc_now(),
            "case": row,
            "error_type": type(error).__name__,
            "error": str(error),
            "work_dir": str(work_dir),
            "command": command,
        }
        write_json_atomic(case_dir / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        return "failed"
    finally:
        if lock_dir.exists():
            shutil.rmtree(lock_dir)


def model_spacing_mm(task: str, fast: bool) -> float | list[float] | None:
    if task in {"total", "total_v3", "total_mr"}:
        return 3.0 if fast else 1.5
    if task == "body":
        return 6.0 if fast else 1.5
    if task == "lung_vessels":
        return [0.703125, 0.703125, 1.0]
    if task == "lung_vessels_LEGACY":
        return None
    if task == "trunk_cavities":
        return [1.5, 1.5, 1.5]
    return None


def validate_roi_subset(task: str, roi_subset: list[str] | None, class_map: dict[int, str]) -> None:
    if not roi_subset:
        return
    if not task.startswith("total"):
        raise SystemExit("--roi-subset is supported only for TotalSegmentator total tasks")
    known = set(class_map.values())
    missing = [name for name in roi_subset if name not in known]
    if missing:
        raise SystemExit(
            "Unknown ROI names for task "
            f"{task!r}: {', '.join(missing)}"
        )


def case_already_complete(cache_root: Path, row: dict[str, Any], output_name: str) -> bool:
    if not output_name.endswith(".nii.gz"):
        raise ValueError(f"output_name must end with .nii.gz, got {output_name!r}")
    output_prefix = output_name[: -len(".nii.gz")]
    if output_prefix.endswith("_labels"):
        output_prefix = output_prefix[: -len("_labels")]

    case_dir = cache_root / "cases" / row["case_key"]
    legacy_output = output_name == "total_labels.nii.gz"
    complete_path = case_dir / (".complete" if legacy_output else f".complete_{output_prefix}")
    output_path = case_dir / output_name
    metadata_path = case_dir / (
        "metadata.json" if legacy_output else f"{output_prefix}_metadata.json"
    )
    return complete_path.is_file() and output_path.is_file() and metadata_path.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", default="total")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--roi-subset", nargs="+", default=None)
    parser.add_argument("--output-name", default="total_labels.nii.gz")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--wait-free-mib", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--free-stability-checks", type=int, default=1)
    parser.add_argument("--wait-before-each-case", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-device", choices=["gpu", "cpu"], default=None)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    if args.free_stability_checks < 1:
        raise SystemExit("free-stability-checks must be at least 1")
    if args.wait_free_mib and not args.wait_before_each_case:
        wait_for_gpu(
            args.wait_free_mib,
            args.poll_seconds,
            args.free_stability_checks,
        )

    class_map = query_class_map(args.task)
    validate_roi_subset(args.task, args.roi_subset, class_map)
    rows = [
        row
        for index, row in enumerate(read_jsonl(args.plan))
        if index % args.num_shards == args.shard_index
    ]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    state_counts: dict[str, int] = {}
    worker_started = time.monotonic()
    print(
        f"{utc_now()} shard={args.shard_index}/{args.num_shards} cases={len(rows)} "
        f"task={args.task} classes={len(class_map)} output={args.output_name}",
        flush=True,
    )
    for ordinal, row in enumerate(rows, start=1):
        print(
            f"{utc_now()} [{ordinal}/{len(rows)}] case={row['name']} "
            f"val_order={row['val_order']}",
            flush=True,
        )
        already_complete = (
            not args.overwrite and case_already_complete(args.cache_root, row, args.output_name)
        )
        if args.wait_free_mib and args.wait_before_each_case and not already_complete:
            wait_for_gpu(
                args.wait_free_mib,
                args.poll_seconds,
                args.free_stability_checks,
            )
        state = run_case(
            row,
            args.cache_root,
            class_map,
            args.task,
            args.fast,
            args.output_name,
            args.roi_subset,
            args.overwrite,
            args.require_device,
        )
        state_counts[state] = state_counts.get(state, 0) + 1
        print(f"{utc_now()} case={row['name']} state={state}", flush=True)
        if state == "failed" and not args.continue_on_error:
            return 1

    summary = {
        "completed_at_utc": utc_now(),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cases_selected": len(rows),
        "states": state_counts,
        "elapsed_seconds": time.monotonic() - worker_started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if state_counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
