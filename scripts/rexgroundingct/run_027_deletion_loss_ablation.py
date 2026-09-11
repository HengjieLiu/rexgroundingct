#!/usr/bin/env python3
"""Four A-only loss arms with explicit GPU opt-in and independent CPU reports."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from exp027_common import atomic_json, atomic_text, lock, now, read_json, sha256
from deletion027_ablation_data import ARMS, LOSSES, DEFAULT_CONFIG, check_context, create_context, finish_preparation, prepare_shard


def group_status(root, value, **extra):
    atomic_json(root / "status.json", {"status": value, "updated_at": now(), "ranking": "pending_user_review", **extra})


def barrier_commands(root, commands, check_reporter):
    children, handles = [], []
    try:
        for name, command, environment in commands:
            path = root / "logs" / f"{name}_{time.time_ns()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w")
            handles.append(handle)
            children.append((name, subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)))
        atomic_json(root / "active_workers.json", {"workers": [{"name": n, "pid": p.pid} for n, p in children], "updated_at": now()})
        while True:
            check_reporter()
            states = [(name, p.poll()) for name, p in children]
            failed = [(n, code) for n, code in states if code not in (None, 0)]
            if failed:
                raise RuntimeError(f"Workers failed: {failed}; see {root/'logs'}")
            if all(code == 0 for _, code in states):
                return
            time.sleep(1)
    finally:
        for _, process in children:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic()+45
        for _, process in children:
            try:
                process.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for handle in handles:
            handle.close()


def verify_barrier(root, update, prepared, context_hash):
    expected = {r["key"] for r in prepared["val"]}
    for arm in ARMS:
        folder = root / "runs" / arm
        summary = read_json(folder / "evaluations" / f"update_{update:07d}" / "summary.json")
        if (summary.get("status") != "complete" or summary["context_sha256"] != context_hash or summary["update"] != update
                or summary.get("arm") != arm or summary.get("loss_spec") != LOSSES[arm]
                or len(summary["findings"]) != len(expected) or {r["key"] for r in summary["findings"]} != expected
                or summary["checkpoint_sha256"] != sha256(folder / "checkpoints" / f"update_{update:07d}.pth")):
            raise ValueError("Incomplete or mismatched evaluation barrier")


def orchestrate(config, args):
    if args.dry_run:
        print({"gpu_work_started": False, "arms": ARMS, "losses": LOSSES, "training_pool": "35 A findings; no B gradients", "updates_per_arm": config["total_updates"],
               "milestones": config["evaluation_updates"], "precision": config["precision"],
               "runtime": config["runtime"], "stop": "pending_user_review"})
        return
    if not args.allow_gpu or os.environ.get("START_GPU_WORK") != "1":
        raise ValueError("GPU orchestration needs explicit opt-in")
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("Four distinct GPU IDs required")
    root = Path(config["runtime"])
    with lock(root / ".orchestrator.lock"):
        if any(root.glob("runs/*/training.jsonl")) and not args.resume:
            raise ValueError("Existing training requires explicit --resume")
        (root / "logs").mkdir(parents=True, exist_ok=True)
        stopped = root / "reports" / ".stop"
        stopped.parent.mkdir(parents=True, exist_ok=True)
        stopped.unlink(missing_ok=True)
        if (root / "config.json").exists() and read_json(root / "config.json") != config:
            raise ValueError("Existing runtime config differs; it cannot be overwritten")
        atomic_json(root / "config.json", config)
        reporter_log = (root / "logs" / f"reporter_{time.time_ns()}.log").open("w")
        cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        command_base = [sys.executable, str(Path(__file__).resolve()), "--config", str(root / "config.json")]
        reporter = subprocess.Popen([*command_base, "report", "--watch"], env=cpu_env, stdout=reporter_log, stderr=subprocess.STDOUT)

        analysis_stop = root / "reports" / ".analysis_stop"
        analysis_stop.unlink(missing_ok=True)
        analysis_log = (root / "logs" / f"analysis_{time.time_ns()}.log").open("w")
        analyzer = subprocess.Popen([*command_base, "analyze", "--watch"], env=cpu_env, stdout=analysis_log, stderr=subprocess.STDOUT)

        def check_reporter():
            if reporter.poll() is not None or analyzer.poll() is not None:
                raise RuntimeError("CPU reporter/analysis worker exited; inspect logs")

        def interrupted(*_):
            raise KeyboardInterrupt("Orchestrator interruption requested")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        started = time.perf_counter()
        try:
            group_status(root, "verifying_inputs")
            context = create_context(config, root)
            atomic_json(root / "launch_manifest.json", {"started_at": now(), "argv": sys.argv, "gpus": args.gpus,
                        "context_sha256": context["sha256"], "image": os.environ.get("DELETION027_IMAGE_ID"), "resume": args.resume})
            group_status(root, "indexing_eligible_tiles")
            commands = [(f"prepare_{i}", [*command_base, "prepare-shard", "--shard", str(i)], cpu_env) for i in range(config["cache_workers"])]
            barrier_commands(root, commands, check_reporter)
            manifest = finish_preparation(config, root)
            from deletion027_ablation_data import initial_state
            initial = initial_state(config, root)
            atomic_json(root / "preparation.json", {"seconds": time.perf_counter()-started, "cases": manifest["cases"],
                        "findings": manifest["findings"], "verified_cache_bytes": sum(r["cache_bytes"] for r in manifest["cache_proofs"]), "initial_sha256": initial["weights_sha256"]})
            prepared = read_json(root / "prepared.json")
            prior = read_json(root / "barriers.json")["completed"] if (root / "barriers.json").exists() else []
            for milestone in config["evaluation_updates"]:
                if milestone in prior:
                    verify_barrier(root, milestone, prepared, context["sha256"])
                    continue
                for phase in ("train", "evaluate"):
                    group_status(root, "training" if phase == "train" else "evaluating", milestone=milestone)
                    commands = []
                    for arm, gpu in zip(ARMS, args.gpus):
                        env = {**cpu_env, "CUDA_VISIBLE_DEVICES": str(gpu), "NVIDIA_TF32_OVERRIDE": "0"}
                        command = [*command_base, phase, "--arm", arm, "--update", str(milestone), "--allow-gpu"]
                        commands.append((f"{phase}_{arm}_{milestone}", command, env))
                    barrier_commands(root, commands, check_reporter)
                verify_barrier(root, milestone, prepared, context["sha256"])
                prior.append(milestone)
                atomic_json(root / "barriers.json", {"completed": prior, "updated_at": now()})
            group_status(root, "finalizing_analysis", update=config["total_updates"])
            analysis_stop.touch()
            while analyzer.poll() is None:
                if reporter.poll() is not None:
                    raise RuntimeError("Reporter exited during closeout")
                time.sleep(1)
            if analyzer.returncode != 0:
                raise RuntimeError("Threshold analysis failed")
            from deletion027_ablation_analysis import audit
            completion = audit(config, root)
            group_status(root, "pending_user_review", update=config["total_updates"],
                         reference_reproduced=completion["reference_reproduced"], elapsed_seconds=time.perf_counter()-started)
        except BaseException as error:
            atomic_text(root / f"failure_{time.time_ns()}.txt", traceback.format_exc())
            group_status(root, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
            raise
        finally:
            analysis_stop.touch()
            if analyzer.poll() is None:
                analyzer.terminate()
                try:
                    analyzer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    analyzer.kill(); analyzer.wait()
            analysis_log.close()
            stopped.touch()
            try:
                reporter.wait(timeout=30)
            except subprocess.TimeoutExpired:
                reporter.terminate(); reporter.wait(timeout=10)
            reporter_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("orchestrate", "prepare", "prepare-shard", "train", "evaluate", "report", "analyze", "audit"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--update", type=int)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command in ("train", "evaluate") and (args.arm is None or args.update not in config["evaluation_updates"]):
        parser.error("train/evaluate requires --arm and a scheduled --update")
    if args.command == "prepare-shard" and (args.shard is None or not 0 <= args.shard < config["cache_workers"]):
        parser.error("prepare-shard requires a valid --shard")
    root = Path(config["runtime"])
    if args.command == "orchestrate":
        orchestrate(config, args)
    elif args.command == "prepare":
        with lock(root / ".orchestrator.lock"):
            group_status(root, "verifying_inputs")
            create_context(config, root)
            for i in range(config["cache_workers"]):
                prepare_shard(config, root, i)
            finish_preparation(config, root)
            from deletion027_ablation_data import initial_state
            initial_state(config, root)
            group_status(root, "prepared")
    elif args.command == "prepare-shard":
        prepare_shard(config, root, args.shard)
    elif args.command in ("train", "evaluate"):
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            if len(args.gpus) != 1:
                raise ValueError("Standalone train/evaluate requires one --gpus ID")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        from deletion027_ablation_worker import train, evaluate
        (train if args.command == "train" else evaluate)(config, root, args.arm, args.update, args.allow_gpu)
    elif args.command == "analyze":
        from deletion027_ablation_analysis import analyze_available, watch
        if args.watch:
            watch(config, root)
        else:
            with lock(root / ".analysis.lock"):
                analyze_available(config, root)
    elif args.command == "audit":
        from deletion027_ablation_analysis import audit
        audit(config, root)
    else:
        from deletion027_ablation_report import render, watch
        if args.watch:
            watch(root, root / "reports" / ".stop")
        else:
            with lock(root / ".report_writer.lock"):
                render(root)


if __name__ == "__main__":
    main()
