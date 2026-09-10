#!/usr/bin/env python3
"""Exp027 CLI. CPU preparation/reporting; GPU work always requires explicit opt-in."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from exp027_common import (ARMS, DEFAULT_CONFIG, EXPERIMENT, atomic_json, code_fingerprint, digest,
                          full_schedule, gpu_gate, lock, materialize_schedule, now, read_json, schedule)


def validate_config(config):
    t, e, m = config["training"], config["evaluation"], config["model"]
    if config["experiment"] != EXPERIMENT:
        raise ValueError("Wrong experiment config")
    if t["batch_size"] != 1 or t["gradient_accumulation"] != 1 or t["updates_per_epoch"] != 100:
        raise ValueError("This implementation requires batch 1, accumulation 1 and 100-update reporting epochs")
    if t["lr_schedule"] != "constant":
        raise ValueError("Supported LR schedule is constant; any new schedule needs an explicit method revision")
    if m["patch_size"] != [192, 192, 192] or e != {"threshold": .5, "overlap": .5}:
        raise ValueError("Accepted tile/threshold/overlap contract changed")


def benchmark_plan(config, prepared=None):
    result = {"mode": "benchmark", "arms": list(ARMS), "updates_per_arm": config["benchmark"]["updates"],
              "warmup_updates": config["benchmark"]["warmup_updates"], "gpu_work_started": False,
              "evaluation": "one full 69-finding 2a pass per arm, then A/B/full aggregation",
              "end_status": "awaiting_schedule_decision", "ranking": "pending_user_review"}
    if prepared:
        events = {arm: schedule(prepared, arm, config["benchmark"]["updates"], config["seed"]) for arm in ARMS}
        keys = {e["key"] for rows in events.values() for e in rows} | {r["key"] for r in prepared["val"]}
        records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
        result.update(required_findings=len(keys), required_cases=len({records[k]["name"] for k in keys}))
    return result


def parallel_workers(commands, *, root, refresh):
    """Barrier-owned children; failure terminates peers and never launches the next stage."""
    children, logs = [], []
    started = time.perf_counter()
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(*_):
        raise KeyboardInterrupt("Orchestration interrupted")

    for s in old_handlers:
        signal.signal(s, interrupted)
    try:
        for name, command, env in commands:
            path = Path(root) / "logs" / f"{name}_{time.time_ns()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w")
            logs.append(handle)
            env = {**env, "EXP027_WORKER_LAUNCH_NS": str(time.time_ns())}
            children.append((name, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        last_refresh = 0.0
        prior_states = None
        while True:
            states = [(name, proc.poll()) for name, proc in children]
            failed = [(name, code) for name, code in states if code not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker failure: {failed}; inspect {Path(root) / 'logs'}")
            if states != prior_states or time.monotonic() - last_refresh >= 30:
                refresh()
                last_refresh = time.monotonic()
                prior_states = states
            if all(code == 0 for _, code in states):
                break
            time.sleep(1)
        refresh()
        return time.perf_counter() - started
    finally:
        for _, proc in children:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 30
        for _, proc in children:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for f in logs:
            f.close()
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


def worker_environment(task, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    if task in ("train", "evaluate"):
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    elif task == "cache":
        env.pop("NVIDIA_TF32_OVERRIDE", None)  # Keep the original frozen-base export environment.
    return env


@contextmanager
def live_writer(root, snapshot):
    """The reporter alone renders figures; coordinators/trainers never wait on it."""
    stop = root / "reports" / ".stop_writer"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    log_path = root / "logs" / f"report_writer_{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "report", "--config", str(snapshot),
               "--root", str(root), "--watch", "--stop-file", str(stop)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        def check():
            if proc.poll() is not None:
                raise RuntimeError(f"Live report writer exited ({proc.returncode}); see {log_path}")
        try:
            yield check
        finally:
            stop.touch()
            try:
                code = proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
                raise RuntimeError(f"Live report writer did not stop: {log_path}")
            if code != 0 and sys.exc_info()[0] is None:
                raise RuntimeError(f"Live report writer failed ({code}); see {log_path}")


def orchestration(config, args, *, benchmark):
    if args.dry_run:
        prepared_path = Path(args.root) / "prepared.json"
        prepared = read_json(prepared_path) if prepared_path.is_file() else None
        plan = benchmark_plan(config, prepared) if benchmark else {"mode": "full", "schedule": full_schedule(config), "arms": ARMS,
                                                       "precision": "FP32", "cache_scope": "all train and validation 2a",
                                                       "expected_cache_cases": 864, "expected_cache_findings": 1189,
                                                       "runtime": str(args.root), "initialization": "pristine common weights",
                                                       "end_status": "pending_user_review", "gpu_work_started": False}
        plan["gpu_ids_after_approval"] = args.gpus
        print(__import__("json").dumps(plan, indent=2))
        return
    gpu_gate(args.allow_gpu)
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("Four independent, distinct GPU IDs are required")
    if benchmark:
        total, milestones = config["benchmark"]["updates"], [config["benchmark"]["updates"]]
        if total != 100:
            raise ValueError("Approved timing design requires exactly 100 updates")
    else:
        total, milestones = full_schedule(config)
        if config["training"]["amp"]:
            raise ValueError("The approved full run requires FP32")
    from exp027_data import cache_gate, prepare, required_cache_keys
    from exp027_report import timing_report
    phase = "benchmark" if benchmark else "full"
    root = Path(args.root)
    with lock(root / ".orchestrator.lock"):
        if benchmark and (root / "timing_report.json").exists():
            raise ValueError("Timing report already exists; use a new runtime root for another timing trial")
        if benchmark and any((root / phase).glob("*/training_timing_until_*.json")):
            raise ValueError("A prior timing trial entered training; use a fresh root for a comparable four-arm measurement")
        prepared = prepare(config, root, config_path=args.config)
        shared = config.get("cache", {}).get("shared_root")
        if shared:
            path = root / "cache"
            if path.exists() or path.is_symlink():
                if path.resolve() != Path(shared).resolve():
                    raise ValueError("Runtime cache points to a different shared cache")
            else:
                path.symlink_to(shared, target_is_directory=True)
        phase_context = root / phase / "context.json"
        if phase_context.exists() and read_json(phase_context)["config"] != config:
            raise ValueError("Existing phase config differs; use a fresh root")
        snapshot = root / phase / "config.json"
        atomic_json(snapshot, config)
        atomic_json(root / phase / "code_manifest.json", code_fingerprint())
        stage = "cache_preparation"

        def status(value, **extra):
            atomic_json(root / "status.json", {"status": value, "active_phase": phase, "updated_at": now(),
                        "ranking": "pending_user_review", **extra})

        def command(task, extra, gpu):
            env = worker_environment(task, gpu)
            return [sys.executable, str(Path(__file__).resolve()), task, "--config", str(snapshot),
                    "--root", str(root), "--allow-gpu", *extra], env

        start = time.perf_counter()
        with live_writer(root, snapshot) as check_writer:
            try:
                status(stage)
                all_events = {arm: materialize_schedule(root / phase / arm / "schedule.json", prepared, arm,
                                                        total, config["seed"]) for arm in ARMS}
                keys = required_cache_keys(prepared, all_events, benchmark)
                records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
                names = sorted({records[k]["name"] for k in keys})
                commands = []
                for i, gpu in enumerate(args.gpus):
                    selected_names = set(names[i::4])
                    path = root / phase / f"cache_keys_gpu{i}.json"
                    atomic_json(path, [k for k in keys if records[k]["name"] in selected_names])
                    cmd, env = command("cache", ["--keys-file", str(path)], gpu)
                    commands.append((f"{phase}_cache_gpu{i}", cmd, env))
                preparation_seconds = parallel_workers(commands, root=root, refresh=check_writer)
                inventory = cache_gate(root, prepared, config, keys)
                atomic_json(root / "preparation_timing.json", {"seconds": preparation_seconds, "inventory": inventory,
                            "updated_at": now()})
                train_group_seconds = eval_group_seconds = 0.0
                for milestone in milestones:
                    stage = "training"
                    status(stage, until=milestone)
                    commands = []
                    for arm, gpu in zip(ARMS, args.gpus):
                        cmd, env = command("train", ["--phase", phase, "--arm", arm, "--until", str(milestone)], gpu)
                        commands.append((f"{phase}_{arm}_train_{milestone}", cmd, env))
                    train_group_seconds += parallel_workers(commands, root=root, refresh=check_writer)
                    initial_hashes = {read_json(root / phase / arm / "training.json")["initial_weights_sha256"] for arm in ARMS}
                    if len(initial_hashes) != 1:
                        raise RuntimeError("Arms did not share identical initial weights")
                    stage = "evaluating"
                    status(stage, update=milestone)
                    commands = []
                    for arm, gpu in zip(ARMS, args.gpus):
                        cmd, env = command("evaluate", ["--phase", phase, "--arm", arm, "--update", str(milestone)], gpu)
                        commands.append((f"{phase}_{arm}_eval_{milestone}", cmd, env))
                    eval_group_seconds += parallel_workers(commands, root=root, refresh=check_writer)
                    for arm in ARMS:
                        summary = read_json(root / phase / arm / "evaluations" / f"update_{milestone:07d}" / "summary.json")
                        if summary["metrics"]["full"]["findings"] != len(prepared["val"]):
                            raise RuntimeError("Incomplete evaluation barrier")
                        if (len(summary["findings"]) != len(prepared["val"]) or
                                {r["key"] for r in summary["findings"]} != {r["key"] for r in prepared["val"]}):
                            raise RuntimeError("Evaluation barrier finding coverage differs")
                    atomic_json(root / "barriers.json", {"completed": [v for v in milestones if v <= milestone],
                                "updated_at": now(), "latest_update": milestone})
                if benchmark:
                    arms = {}
                    for arm in ARMS:
                        folder = root / phase / arm
                        arms[arm] = {"training": read_json(folder / f"training_timing_until_{total:07d}.json"),
                                     "evaluation": read_json(folder / "evaluations" / f"update_{total:07d}" / "summary.json")["timing"]}
                    timing_report(root, {"preparation_seconds": preparation_seconds, "training_group_seconds": train_group_seconds,
                                        "evaluation_group_seconds": eval_group_seconds, "total_seconds": time.perf_counter() - start,
                                        "arms": arms, "gpu_ids": args.gpus, "status": "awaiting_schedule_decision"})
                status("awaiting_schedule_decision" if benchmark else "pending_user_review")
            except BaseException as exc:
                status("failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
                raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    for task in ("prepare", "cache", "train", "evaluate", "benchmark", "orchestrate", "report"):
        sub = subs.add_parser(task)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--root", type=Path)
        if task in ("cache", "train", "evaluate", "benchmark", "orchestrate"):
            sub.add_argument("--allow-gpu", action="store_true", help="Use only after explicit user approval")
        if task in ("cache", "train", "evaluate"):
            sub.add_argument("--device", type=int, default=0, help="Logical device inside CUDA_VISIBLE_DEVICES")
        if task == "cache":
            sub.add_argument("--keys-file", type=Path, required=True)
        if task in ("train", "evaluate"):
            sub.add_argument("--phase", choices=("benchmark", "full"), required=True)
            sub.add_argument("--arm", choices=ARMS, required=True)
            sub.add_argument("--until" if task == "train" else "--update", type=int, required=True)
        if task in ("benchmark", "orchestrate"):
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
        if task == "report":
            sub.add_argument("--watch", action="store_true")
            sub.add_argument("--stop-file", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = read_json(args.config)
    validate_config(config)
    args.root = args.root or Path(config["experiment_dir"])
    if args.command in ("benchmark", "orchestrate"):
        orchestration(config, args, benchmark=args.command == "benchmark")
        return 0
    if args.command == "prepare":
        from exp027_data import prepare
        from exp027_report import report_command
        prepare(config, args.root, config_path=args.config)
        report_command(args.root)
        print(f"Prepared CPU metadata; dashboard: {args.root / 'reports/live_dashboard.md'}")
        return 0
    if args.command == "report":
        from exp027_report import report_command
        report_command(args.root, args.watch, config["report"]["interval_seconds"], args.stop_file)
        return 0
    gpu_gate(args.allow_gpu)  # Gate before importing torch or opening any CUDA context.
    prepared = read_json(args.root / "prepared.json")
    if args.command == "cache":
        from exp027_data import build_cache
        build_cache(config, prepared, args.root, read_json(args.keys_file), allow_gpu=True, device=args.device)
    else:
        if args.phase == "full":
            if config["training"]["amp"]:
                raise ValueError("The approved full run requires FP32")
            total, _ = full_schedule(config)
            requested = args.until if args.command == "train" else args.update
            if not 0 <= requested <= total:
                raise ValueError("Requested update exceeds approved full-run horizon")
            from exp027_data import require_full_cache
            require_full_cache(args.root, prepared, config)
        elif (args.until if args.command == "train" else args.update) != 100:
            raise ValueError("Benchmark workers must use update 100")
        from exp027_runner import train_worker, evaluate_worker
        try:
            if args.command == "train":
                train_worker(config, prepared, args.root, args.phase, args.arm, args.until,
                             allow_gpu=True, device_index=args.device)
            else:
                evaluate_worker(config, prepared, args.root, args.phase, args.arm, args.update,
                                allow_gpu=True, device_index=args.device)
        except BaseException as exc:
            status_path = args.root / args.phase / args.arm / "status.json"
            previous = read_json(status_path) if status_path.exists() else {}
            atomic_json(status_path, {**previous, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                      "updated_at": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
