#!/usr/bin/env python3
"""Explicit standalone cache preparation; never launches refinement training."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from exp027_common import atomic_json, lock, now, read_json
from exp027_multicategory_data import DEFAULT_CONFIG, check_context, finalize, prepare
from exp027_multicategory_report import render, report
from exp027_multicategory_worker import interrupted, require_gpu, run_worker


def command_line(config_path, command, *args):
    return [sys.executable, str(Path(__file__).resolve()), command, "--config", str(config_path), *map(str, args)]


def dry_run(config, command):
    return {"command": command, "gpu_work": False, "dry_run": True,
            "runtime": config["runtime"], "categories": config["categories"],
            "expected": config["expected"], "gpu_policy": config["gpu"],
            "phases": ["CPU metadata verification", "CPU validation import", "guarded training-logit smoke",
                       "all-worker smoke barrier", "training-logit export", "complete verification"],
            "completion": "cache_ready_for_training", "new_refiner_training": False}


def stop_children(children):
    for p, stream in children:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    for p, stream in children:
        if p.poll() is None:
            try:
                p.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        stream.close()


def orchestrate(config, args, verify_only=False, val_only=False):
    root = Path(config["runtime"])
    if not verify_only and not val_only:
        require_gpu(args.allow_gpu, args.share_gpus)
        if len(args.gpus) != config["workers"] or len(set(args.gpus)) != len(args.gpus):
            raise ValueError("Four distinct GPUs required")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with lock(root / "locks/orchestrator.lock"):
        prior = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if prior.get("status") == "cache_ready_for_training" and not verify_only:
            check_context(config, root)
            print("Cache already ready; use verify for an explicit full recheck.")
            return
        if prior.get("status") in ("failed", "interrupted", "exporting", "smoke", "preparing") and not args.resume:
            raise ValueError("Existing execution requires explicit --resume")
        attempt = root / "attempts" / now().replace(":", "").replace("+", "_")
        attempt.mkdir(parents=True, exist_ok=False)
        children = []
        mode = "verify" if verify_only else "export"
        started = time.monotonic()

        def status(phase, error=None):
            atomic_json(root / "status.json", {"status": phase, "mode": mode, "attempt": str(attempt),
                        "updated_at": now(), "elapsed_seconds": time.monotonic() - started, "error": error})

        def launch(command, label, env=None):
            stream = (attempt / f"{label}.log").open("w")
            p = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
            children.append((p, stream))
            return p

        try:
            status("preparing")
            render(root)
            context, prepared = prepare(config, root)
            preparation_seconds = time.monotonic() - started
            atomic_json(attempt / "run_manifest.json", {"argv": sys.argv, "config": config,
                        "context_sha256": context["sha256"], "started_at": now(),
                        "image_id": os.environ.get("MULTICATEGORY_IMAGE_ID"), "mode": mode,
                        "metadata_preparation_seconds": preparation_seconds,
                        "code": context["code"], "gpus": args.gpus if not verify_only and not val_only else []})
            reporter = launch(command_line(args.config, "report", "--watch"), "report")
            workers = []
            common = ["--attempt", attempt]
            if verify_only:
                for shard in range(config["workers"]):
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
                    workers.append(launch(command_line(args.config, "worker", *common, "--kind", "verify", "--shard", shard), f"verify{shard}", env))
            else:
                workers.append(launch(command_line(args.config, "worker", *common, "--kind", "val"), "val", {**os.environ, "CUDA_VISIBLE_DEVICES": ""}))
                if not val_only:
                    for shard, gpu in enumerate(args.gpus):
                        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                        workers.append(launch(command_line(args.config, "worker", *common, "--kind", "train",
                                              "--shard", shard, "--allow-gpu", "--share-gpus"), f"train{shard}", env))
            released = verify_only or val_only
            status("verifying" if verify_only else ("importing_validation" if val_only else "smoke"))
            while any(p.poll() is None for p in workers):
                failed = [p.pid for p in workers if p.poll() not in (None, 0)]
                if failed or (attempt / "ABORT.json").exists():
                    raise RuntimeError(f"Cache worker failure; inspect {attempt}; failed PIDs={failed}")
                if reporter.poll() is not None:
                    raise RuntimeError(f"CPU report writer exited unexpectedly: {reporter.returncode}")
                if not released:
                    paths = [attempt / "smoke" / f"train{s}.json" for s in range(config["workers"])]
                    if all(p.exists() for p in paths):
                        smoke = [read_json(p) for p in paths]
                        if any(s["status"] != "passed" or s["memory"]["minimum_free_mib"] < config["gpu"]["minimum_free_mib"] for s in smoke):
                            raise RuntimeError("Memory or integrity smoke gate failed")
                        atomic_json(attempt / "RELEASE.json", {"status": "passed", "smoke": smoke, "at": now()})
                        released = True
                        status("exporting")
                time.sleep(1)
            if any(p.returncode != 0 for p in workers):
                raise RuntimeError(f"Cache worker failed: {attempt}")
            if val_only:
                status("validation_import_complete")
            else:
                status("verifying")
                inventory = finalize(config, root)
                atomic_json(attempt / "completion.json", {"cases": inventory["cases"], "findings": inventory["findings"],
                            "cache_bytes": inventory["cache_bytes"], "total_seconds": time.monotonic() - started,
                            "metadata_preparation_seconds": preparation_seconds, "completed_at": now()})
                status("cache_ready_for_training")
            if reporter.poll() is None:
                os.killpg(reporter.pid, signal.SIGTERM)
                reporter.wait(timeout=15)
            report(root)
        except BaseException as exc:
            atomic_json(attempt / "ABORT.json", {"error": str(exc), "traceback": traceback.format_exc(), "at": now()})
            stop_children(children)
            children = []
            status("interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed", str(exc))
            report(root)
            raise
        finally:
            stop_children(children)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "export", "import-validation", "verify", "report", "dry-run", "orchestrate", "worker"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--share-gpus", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--attempt", type=Path)
    p.add_argument("--kind", choices=("train", "val", "verify"))
    p.add_argument("--shard", type=int, default=0)
    return p


def main():
    args = parser().parse_args()
    config = read_json(args.config)
    root = Path(config["runtime"])
    if args.command == "dry-run" or args.dry_run:
        import json
        print(json.dumps(dry_run(config, args.command), indent=2))
    elif args.command == "prepare":
        prepare(config, root)
        report(root)
    elif args.command == "report":
        report(root, config["report_interval_seconds"], args.watch)
    elif args.command == "worker":
        if args.attempt is None or args.kind is None:
            raise ValueError("Internal worker requires --attempt and --kind")
        run_worker(config, root, args.attempt, args.kind, args.shard, args.allow_gpu, args.share_gpus)
    else:
        orchestrate(config, args, args.command == "verify", args.command == "import-validation")


if __name__ == "__main__":
    main()
