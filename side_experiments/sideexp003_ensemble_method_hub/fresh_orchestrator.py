#!/usr/bin/env python3
"""Guarded four-GPU wave orchestrator for SideExp003 fresh logit exports."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fresh_cache import (  # noqa: E402
    EXPECTED_CONTAINER_IMAGE_ID,
    FreshCacheError,
    atomic_write_json,
    atomic_write_text,
    read_json,
    source_bundle,
    utc_now,
    validate_job,
)


MILESTONES = (1, 10, 50, 100, 150, 200)
BASELINE_SECONDS = {
    ("standard", "crop_zscore_native_v1"): 5_700.0,
    ("standard", "crop_clip1024_linear_iso07_v1"): 6_900.0,
    ("dual", "crop_zscore_native_v1"): 8_000.0,
    ("s3", "crop_zscore_native_v1"): 6_600.0,
}


def run_command(
    command: list[str], *, check: bool = True, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def docker_image_id(image: str) -> str:
    result = run_command(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    return result.stdout.strip()


def docker_ps() -> list[dict[str, Any]]:
    result = run_command(["docker", "ps", "-q"])
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    records = []
    for container_id in ids:
        value = run_command(["docker", "inspect", container_id]).stdout
        inspected = json.loads(value)[0]
        records.append(inspected)
    return records


def gpu_cotenants(job_id: str) -> list[dict[str, Any]]:
    result = []
    for inspected in docker_ps():
        name = str(inspected.get("Name", "")).lstrip("/")
        if name.startswith(f"sideexp003_{job_id}_"):
            continue
        requests = inspected.get("HostConfig", {}).get("DeviceRequests") or []
        if not requests:
            continue
        result.append(
            {
                "id": inspected["Id"],
                "name": name,
                "started_at_utc": inspected.get("State", {}).get("StartedAt"),
                "command": inspected.get("Config", {}).get("Cmd"),
            }
        )
    return result


def check_cotenant_health(cotenants: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors = []
    for cotenant in cotenants:
        completed = run_command(
            ["docker", "inspect", cotenant["id"]], check=False
        )
        if completed.returncode != 0:
            continue
        state = json.loads(completed.stdout)[0].get("State", {})
        if state.get("Running"):
            continue
        exit_code = int(state.get("ExitCode", 0))
        if state.get("OOMKilled") or exit_code != 0:
            errors.append(
                f"GPU co-tenant {cotenant['name']} exited abnormally: "
                f"exit={exit_code}, oom={state.get('OOMKilled')}, error={state.get('Error')}"
            )
    return not errors, errors


def _parse_gpu_csv(text: str) -> dict[int, int]:
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        index, free = [part.strip() for part in line.split(",", maxsplit=1)]
        values[int(index)] = int(free)
    return values


def gpu_free_mib() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(command, check=False)
    if result.returncode == 0:
        return _parse_gpu_csv(result.stdout)
    containers = [
        value for value in docker_ps()
        if value.get("HostConfig", {}).get("DeviceRequests")
    ]
    if not containers:
        raise FreshCacheError(f"nvidia-smi failed and no probe container exists: {result.stderr}")
    probe = containers[0]["Id"]
    fallback = run_command(["docker", "exec", probe, *command])
    return _parse_gpu_csv(fallback.stdout)


def _worker_name(job_id: str, wave: int, gpu: int) -> str:
    return f"sideexp003_{job_id}_w{wave:02d}_g{gpu}"


def _container_state(name: str) -> dict[str, Any] | None:
    result = run_command(["docker", "inspect", name], check=False)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)[0].get("State", {})


def _capture_and_remove_worker(name: str, log_path: Path) -> None:
    logs = run_command(["docker", "logs", name], check=False)
    atomic_write_text(log_path, logs.stdout + logs.stderr)
    run_command(["docker", "rm", name], check=False)


def _stop_workers(names: list[str]) -> None:
    for name in names:
        state = _container_state(name)
        if state and state.get("Running"):
            run_command(["docker", "stop", "--time", "30", name], check=False)


def _container_job_path(job_path: Path) -> str:
    resolved = job_path.resolve()
    try:
        return str(Path("/workspace") / resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise FreshCacheError(f"job spec must be tracked below {REPO_ROOT}: {job_path}") from exc


def launch_worker(job: dict[str, Any], job_path: Path, candidate: dict[str, Any]) -> str:
    name = _worker_name(job["job_id"], candidate["wave"], candidate["gpu"])
    existing = _container_state(name)
    if existing is not None:
        if existing.get("Running"):
            return name
        _capture_and_remove_worker(
            name, Path(job["runtime_root"]) / "logs" / f"{candidate['id']}.previous.log"
        )
    command = [
        "docker", "run", "-d", "--name", name,
        "--gpus", f"device={candidate['gpu']}",
        "--ipc=host", "--shm-size=32g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/workspace",
        "-v", f"{REPO_ROOT}:/workspace",
        "-v", "/data/hengjie:/data/hengjie",
        "-v", "/mnt/shengdata1:/mnt/shengdata1",
        "-e", "HOME=/tmp",
        "-e", "TZ=US/Pacific",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home",
        "-e", "HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub",
        "-e", "HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet",
        job["container"]["image"],
        "python", "/workspace/side_experiments/sideexp003_ensemble_method_hub/fresh_export.py",
        "--job", _container_job_path(job_path),
        "--candidate-id", candidate["id"],
    ]
    result = run_command(command)
    if not result.stdout.strip():
        raise FreshCacheError(f"docker did not return a container ID for {candidate['id']}")
    return name


def _progress(job: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(candidate["progress_path"])
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except FreshCacheError:
        return None
    return value if value.get("job_id") == job["job_id"] else None


def _progress_map(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for candidate in job["candidates"]:
        value = _progress(job, candidate)
        if value is not None:
            result[candidate["id"]] = value
    return result


def _sync_catalog(job_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "hub.py"),
            "cache", "sync", "--job", str(job_path), "--apply",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FreshCacheError(
            f"catalog sync failed: {completed.stdout}\n{completed.stderr}"
        )


def _write_abort(job: dict[str, Any], reason: str) -> None:
    atomic_write_json(
        Path(job["runtime_root"]) / "control" / "ABORT",
        {"job_id": job["job_id"], "reason": reason, "created_at_utc": utc_now()},
    )


def _job_state_path(job: dict[str, Any]) -> Path:
    return Path(job["runtime_root"]) / "job_state.json"


def _write_state(job: dict[str, Any], state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    value = {**state, **updates, "updated_at_utc": utc_now()}
    atomic_write_json(_job_state_path(job), value)
    return value


def _format_progress(candidate: dict[str, Any], progress: dict[str, Any] | None) -> str:
    if progress is None:
        return f"r{candidate['rank']}:{candidate['id']} pending"
    cases = int(progress.get("cases", 0))
    status = progress.get("status")
    eta = progress.get("eta_seconds")
    eta_text = f", eta={float(eta)/3600:.2f}h" if isinstance(eta, (int, float)) else ""
    return f"r{candidate['rank']}:{status} {cases}/200{eta_text}"


def _wait_for_start_memory(job: dict[str, Any], members: list[dict[str, Any]]) -> None:
    required = int(job["gpu_safety"]["minimum_start_free_mib"])
    poll = int(job["gpu_safety"]["poll_seconds"])
    while True:
        values = gpu_free_mib()
        insufficient = {
            candidate["gpu"]: values.get(candidate["gpu"], -1)
            for candidate in members
            if values.get(candidate["gpu"], -1) < required
        }
        if not insufficient:
            return
        print(f"GPU start gate waiting: {insufficient}; require {required} MiB", flush=True)
        time.sleep(poll)


def _wait_for_smoke(
    job: dict[str, Any], members: list[dict[str, Any]], names: list[str], cotenants: list[dict[str, Any]]
) -> None:
    job_root = Path(job["runtime_root"])
    poll = min(10, int(job["gpu_safety"]["poll_seconds"]))
    while True:
        smoke = []
        for candidate in members:
            path = job_root / "smoke" / f"{candidate['id']}.json"
            if path.is_file():
                smoke.append(read_json(path))
        progress = {candidate["id"]: _progress(job, candidate) for candidate in members}
        failed = [value for value in progress.values() if value and value.get("status") == "failed"]
        if failed:
            raise FreshCacheError(f"wave smoke worker failed: {[value.get('candidate_id') for value in failed]}")
        for name in names:
            state = _container_state(name)
            if state is not None and not state.get("Running") and len(smoke) < len(members):
                raise FreshCacheError(f"worker exited before smoke gate: {name}, state={state}")
        healthy, errors = check_cotenant_health(cotenants)
        if not healthy:
            raise FreshCacheError("; ".join(errors))
        if len(smoke) == len(members):
            required = int(job["gpu_safety"]["minimum_post_smoke_free_mib"])
            low = [
                value for value in smoke
                if int(value.get("free_mib_after_case", -1)) < required
            ]
            if low:
                raise FreshCacheError(
                    f"post-smoke GPU reserve below {required} MiB: "
                    f"{[(value['physical_gpu'], value.get('free_mib_after_case')) for value in low]}"
                )
            atomic_write_json(
                job_root / "control" / f"wave_{members[0]['wave']:02d}.continue",
                {"status": "passed", "created_at_utc": utc_now(), "smoke": smoke},
            )
            return
        time.sleep(poll)


def _monitor_wave(
    job: dict[str, Any], job_path: Path, members: list[dict[str, Any]], names: list[str], cotenants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    poll = int(job["gpu_safety"]["poll_seconds"])
    seen_milestones = {candidate["id"]: set() for candidate in members}
    while True:
        records = [_progress(job, candidate) for candidate in members]
        for candidate, record in zip(members, records):
            if record is None:
                continue
            count = int(record.get("cases", 0))
            reached = {value for value in MILESTONES if count >= value}
            if reached - seen_milestones[candidate["id"]]:
                seen_milestones[candidate["id"]] = reached
                _sync_catalog(job_path)
        print(" | ".join(_format_progress(candidate, record) for candidate, record in zip(members, records)), flush=True)
        if all(record and record.get("status") == "strict_passed" for record in records):
            return [record for record in records if record is not None]
        if any(record and record.get("status") == "failed" for record in records):
            while any(
                (_container_state(name) or {}).get("Running") for name in names
            ):
                time.sleep(min(30, poll))
            raise FreshCacheError(
                "wave candidate failure: "
                + "; ".join(
                    f"{record.get('candidate_id')}: {record.get('error')}"
                    for record in records if record and record.get("status") == "failed"
                )
            )
        healthy, errors = check_cotenant_health(cotenants)
        if not healthy:
            raise FreshCacheError("; ".join(errors))
        exited_bad = []
        for name, record in zip(names, records):
            state = _container_state(name)
            if state is not None and not state.get("Running"):
                if int(state.get("ExitCode", 1)) != 0 or not record or record.get("status") != "strict_passed":
                    exited_bad.append((name, state, record))
        if exited_bad:
            raise FreshCacheError(f"worker container failure: {exited_bad}")
        time.sleep(poll)


def _estimate_after_wave1(job: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {candidate["id"]: candidate for candidate in job["candidates"]}
    factors = []
    observations = []
    for record in records:
        candidate = by_id[record["candidate_id"]]
        key = (candidate["architecture_kind"], candidate["cache"]["id"])
        baseline = BASELINE_SECONDS.get(key, 6_600.0)
        observed = float(record["elapsed_seconds"])
        factors.append(observed / baseline)
        observations.append(
            {
                "candidate_id": candidate["id"],
                "rank": candidate["rank"],
                "architecture_kind": candidate["architecture_kind"],
                "preprocessing": candidate["cache"]["id"],
                "elapsed_seconds": observed,
                "seconds_per_case": observed / 200,
                "dtype": record.get("dtype"),
            }
        )
    factor = statistics.median(factors)
    wave_predictions = []
    for wave in range(2, max(candidate["wave"] for candidate in job["candidates"]) + 1):
        members = [candidate for candidate in job["candidates"] if candidate["wave"] == wave]
        predictions = []
        for candidate in members:
            baseline = BASELINE_SECONDS.get(
                (candidate["architecture_kind"], candidate["cache"]["id"]), 6_600.0
            )
            predictions.append(baseline * factor)
        wave_predictions.append({"wave": wave, "seconds": max(predictions), "members": len(members)})
    remaining = sum(value["seconds"] for value in wave_predictions)
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "created_at_utc": utc_now(),
        "wave_1": observations,
        "measured_contention_factor_median": factor,
        "remaining_wave_predictions": wave_predictions,
        "remaining_eta_hours_point": remaining / 3600,
        "remaining_eta_hours_range": [remaining * 0.8 / 3600, remaining * 1.25 / 3600],
        "all_20_eta_hours_range": [
            (max(float(record["elapsed_seconds"]) for record in records) + remaining * 0.8) / 3600,
            (max(float(record["elapsed_seconds"]) for record in records) + remaining * 1.25) / 3600,
        ],
    }


def _render_wave1_report(report: dict[str, Any]) -> str:
    lines = [
        "# SideExp003 Fresh Top-20 Wave 1 Timing Report",
        "",
        f"Generated: `{report['created_at_utc']}`.",
        "",
        "| Rank | Candidate | Architecture | Preprocessing | Hours | Seconds/case | Dtype |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for value in report["wave_1"]:
        lines.append(
            f"| {value['rank']} | `{value['candidate_id']}` | `{value['architecture_kind']}` "
            f"| `{value['preprocessing']}` | {value['elapsed_seconds']/3600:.3f} "
            f"| {value['seconds_per_case']:.2f} | `{value['dtype']}` |"
        )
    low, high = report["remaining_eta_hours_range"]
    all_low, all_high = report["all_20_eta_hours_range"]
    lines.extend(
        [
            "",
            f"Measured median contention factor: `{report['measured_contention_factor_median']:.3f}`.",
            f"Remaining ETA after wave 1: `{low:.2f}–{high:.2f} hours`.",
            f"Estimated total launch-to-20/20 time: `{all_low:.2f}–{all_high:.2f} hours`.",
            "",
            "The queue is configured to continue automatically after this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_job(job_path: Path) -> int:
    job = read_json(job_path)
    errors = validate_job(job)
    if errors:
        raise FreshCacheError("invalid job: " + "; ".join(errors))
    if job["container"]["image_id"] != EXPECTED_CONTAINER_IMAGE_ID:
        raise FreshCacheError("job container image ID is not pinned")
    observed_image = docker_image_id(job["container"]["image"])
    if observed_image != job["container"]["image_id"]:
        raise FreshCacheError(
            f"container image drift: expected {job['container']['image_id']}, got {observed_image}"
        )
    observed_bundle = source_bundle()["sha256"]
    if observed_bundle != job["source_bundle"]["sha256"]:
        raise FreshCacheError("source bundle changed after job planning")
    job_root = Path(job["runtime_root"])
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "logs").mkdir(parents=True, exist_ok=True)
    (job_root / "control").mkdir(parents=True, exist_ok=True)
    cotenants = gpu_cotenants(job["job_id"])
    state = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "status": "running",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "current_wave": None,
        "gpu_cotenants_at_start": cotenants,
        "error": None,
    }
    if _job_state_path(job).is_file():
        previous = read_json(_job_state_path(job))
        state["started_at_utc"] = previous.get("started_at_utc", state["started_at_utc"])
    state = _write_state(job, state)
    candidates = job["candidates"]
    ranks = [int(candidate["rank"]) for candidate in candidates]
    eta_range = job["initial_eta_hours"].get(
        "remaining", job["initial_eta_hours"].get("all_20")
    )
    print(
        f"{job['job_id']}: fresh-only {len(candidates)}-model export for ranks "
        f"{min(ranks)}-{max(ranks)}; initial ETA {eta_range[0]:.0f}-{eta_range[1]:.0f}h",
        flush=True,
    )
    active_names: list[str] = []
    try:
        min_wave = min(candidate["wave"] for candidate in candidates)
        max_wave = max(candidate["wave"] for candidate in candidates)
        for wave in range(min_wave, max_wave + 1):
            members = [candidate for candidate in job["candidates"] if candidate["wave"] == wave]
            completed = [_progress(job, candidate) for candidate in members]
            if all(record and record.get("status") == "strict_passed" for record in completed):
                print(f"wave {wave}: already complete for this job; resuming after it", flush=True)
                continue
            state = _write_state(job, state, status=f"waiting_for_wave_{wave}_gpu", current_wave=wave)
            _wait_for_start_memory(job, members)
            state = _write_state(job, state, status=f"running_wave_{wave}", current_wave=wave)
            active_names = [launch_worker(job, job_path, candidate) for candidate in members]
            _sync_catalog(job_path)
            _wait_for_smoke(job, members, active_names, cotenants)
            _sync_catalog(job_path)
            print(f"wave {wave}: guarded smoke passed; full export released", flush=True)
            records = _monitor_wave(job, job_path, members, active_names, cotenants)
            for candidate, name in zip(members, active_names):
                while (_container_state(name) or {}).get("Running"):
                    time.sleep(2)
                _capture_and_remove_worker(
                    name, job_root / "logs" / f"{candidate['id']}.log"
                )
            active_names = []
            _sync_catalog(job_path)
            print(f"wave {wave}: {len(records)}/{len(members)} strict_passed", flush=True)
            if wave == 1:
                report = _estimate_after_wave1(job, records)
                tracked_root = job_path.parent
                atomic_write_json(tracked_root / "wave1_timing_report.json", report)
                atomic_write_text(tracked_root / "wave1_timing_report.md", _render_wave1_report(report))
                atomic_write_json(job_root / "wave1_timing_report.json", report)
                low, high = report["remaining_eta_hours_range"]
                all_low, all_high = report["all_20_eta_hours_range"]
                print(
                    f"WAVE 1 COMPLETE: remaining ETA {low:.2f}-{high:.2f}h; "
                    f"total ETA {all_low:.2f}-{all_high:.2f}h; auto-continuing",
                    flush=True,
                )
        records = _progress_map(job)
        if len(records) != len(job["candidates"]) or any(
            value.get("status") != "strict_passed" for value in records.values()
        ):
            raise FreshCacheError(
                f"job ended without {len(job['candidates'])}/{len(job['candidates'])} strict caches"
            )
        completed_count = len(candidates)
        parent_count = len(job.get("continuation", {}).get("completed_parent_ranks", []))
        state = _write_state(
            job,
            state,
            status="strict_passed",
            current_wave=max_wave,
            completed_at_utc=utc_now(),
            strict_passed=completed_count,
            overall_strict_passed=parent_count + completed_count,
        )
        _sync_catalog(job_path)
        print(
            f"{job['job_id']}: {completed_count}/{completed_count} strict_passed; "
            f"overall top-20 {parent_count + completed_count}/"
            f"{job.get('continuation', {}).get('original_top_n', completed_count)}",
            flush=True,
        )
        return 0
    except Exception as exc:
        reason = str(exc)
        _write_abort(job, reason)
        _stop_workers(active_names)
        for name in active_names:
            if _container_state(name) is not None:
                candidate_id = next(
                    (candidate["id"] for candidate in job["candidates"] if _worker_name(job["job_id"], candidate["wave"], candidate["gpu"]) == name),
                    name,
                )
                _capture_and_remove_worker(name, job_root / "logs" / f"{candidate_id}.log")
        _sync_catalog(job_path)
        _write_state(job, state, status="failed", error=reason)
        raise


def watch_job(job_path: Path, interval: int, once: bool = False) -> int:
    job = read_json(job_path)
    errors = validate_job(job)
    if errors:
        raise FreshCacheError("invalid job: " + "; ".join(errors))
    while True:
        state = read_json(_job_state_path(job)) if _job_state_path(job).is_file() else {"status": "not_started"}
        records = _progress_map(job)
        print(
            json.dumps(
                {
                    "job_id": job["job_id"],
                    "status": state.get("status"),
                    "strict_passed": sum(value.get("status") == "strict_passed" for value in records.values()),
                    "failed": [key for key, value in records.items() if value.get("status") == "failed"],
                    "progress": {
                        key: {
                            "status": value.get("status"),
                            "cases": value.get("cases"),
                            "eta_hours": (
                                float(value["eta_seconds"]) / 3600
                                if isinstance(value.get("eta_seconds"), (int, float)) else None
                            ),
                        }
                        for key, value in records.items()
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if once or state.get("status") in {"strict_passed", "failed"}:
            return 0 if state.get("status") != "failed" else 1
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--job", type=Path, required=True)
    watch = subparsers.add_parser("watch")
    watch.add_argument("--job", type=Path, required=True)
    watch.add_argument("--interval", type=int, default=60)
    watch.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        return run_job(args.job)
    return watch_job(args.job, args.interval, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
