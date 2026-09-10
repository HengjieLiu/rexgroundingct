#!/usr/bin/env python3
"""Explicitly authorized FP16-completion -> FP32 benchmark, with CPU reporting."""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from exp027_common import (ARMS, DEFAULT_CONFIG, REPO, atomic_csv, atomic_json, atomic_text,
                          code_fingerprint, digest, gpu_gate, lock, now, read_json, schedule, sha256)
from exp027_data import cache_contract, prepare

DEFAULT_CONFIG = REPO / "experiments/027_voxtell_2a_residual_refinement/fp16_benchmark_config.json"
FP32_CONFIG = REPO / "experiments/027_voxtell_2a_residual_refinement/fp32_benchmark_config.json"


def validate_pair(fp16, fp32):
    if fp16["training"]["amp"] is not True or fp32["training"]["amp"] is not False:
        raise ValueError("Expected AMP FP16 prerequisite and AMP-disabled FP32 follow-up")
    expected = copy.deepcopy(fp16)
    expected["training"]["amp"] = False
    expected["experiment_dir"] = str(Path(fp16["experiment_dir"]) / "precision_fp32")
    if fp32 != expected:
        raise ValueError("FP32 must differ only in AMP and its separate runtime directory")
    if fp16["benchmark"] != {"updates": 100, "warmup_updates": 5}:
        raise ValueError("Comparison requires five warm-up and 100 measured updates")


def completion(root):
    """Pending is False; declared completion with invalid evidence is an error."""
    root = Path(root)
    path = root / "status.json"
    if not path.exists():
        return False
    status = read_json(path)
    if status["status"] in ("failed", "interrupted"):
        raise RuntimeError(f"Benchmark failed at {root}: {status}")
    if status["status"] != "awaiting_schedule_decision":
        return False
    prepared = read_json(root / "prepared.json")
    expected_keys = {r["key"] for r in prepared["val"]}
    if len(expected_keys) != 69:
        raise ValueError("Expected exactly 69 validation findings")
    timing = read_json(root / "timing_report.json")
    if timing["status"] != "awaiting_schedule_decision" or set(timing["arms"]) != set(ARMS):
        raise ValueError("Incomplete timing report")
    initial_hashes = set()
    for arm in ARMS:
        folder = root / "benchmark" / arm
        history = read_json(folder / "training.json")
        updates = history["updates"]
        if [r["update"] for r in updates] != list(range(1, 101)):
            raise ValueError(f"Incomplete or repeated measured updates: {arm}")
        for r in updates:
            if not all(math.isfinite(r[k]) for k in ("loss", "grad_norm", "patch_dice", "residual_mean_abs")):
                raise ValueError(f"Non-finite training metrics: {arm}")
        train_timing = timing["arms"][arm]["training"]
        if train_timing["updates_completed"] != 100 or train_timing["started_at_update"] != 0:
            raise ValueError(f"Training was incomplete or resumed: {arm}")
        initial_hashes.add(history["initial_weights_sha256"])
        result = read_json(folder / "evaluations/update_0000100/summary.json")
        if (len(result["findings"]) != 69 or {r["key"] for r in result["findings"]} != expected_keys
                or result["metrics"]["full"]["findings"] != 69):
            raise ValueError(f"Incomplete evaluation: {arm}")
        if sha256(folder / "checkpoints/update_0000100.pth") != result["checkpoint_sha256"]:
            raise ValueError(f"Evaluation/checkpoint hash mismatch: {arm}")
    if len(initial_hashes) != 1:
        raise ValueError("Initial weights differ between arms")
    return True


def execution_released(root):
    try:
        with lock(Path(root) / ".orchestrator.lock"), lock(Path(root) / ".report_writer.lock"):
            return True
    except RuntimeError as exc:
        if str(exc).startswith("Another process owns "):
            return False
        raise


def idle_gpus():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = [[int(v.strip()) for v in line.split(",")] for line in output.strip().splitlines()]
    return (len(rows) >= 4 and all(i == index and memory < 100 and utilization == 0
                                for index, (i, memory, utilization) in enumerate(rows[:4]))), rows


def docker_command(config, root, image):
    # Keep the original worker code unchanged during the active FP16 evaluation.
    # NVIDIA's override disables TF32 in child processes as well as the parent.
    env = {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "EXP027_CONTAINER_IMAGE_ID": image,
           "NVIDIA_TF32_OVERRIDE": "0", "HF_HOME": "/data/hengjie/datasets/rexgroundingct/.hf_home",
           "HF_HUB_CACHE": "/data/hengjie/datasets/rexgroundingct/.hf_home/hub",
           "HF_XET_CACHE": "/data/hengjie/datasets/rexgroundingct/.hf_home/xet",
           "PYTHONPATH": f"{REPO}/scripts/rexgroundingct:{REPO}/external/VoxTell"}
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--shm-size=16g",
               "--user", f"{os.getuid()}:{os.getgid()}"]
    for mount in (f"{REPO}:{REPO}:ro", "/data/hengjie:/data/hengjie:ro", "/mnt/shengdata1:/mnt/shengdata1"):
        command += ["-v", mount]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    return command + [image, "python", str(REPO / "scripts/rexgroundingct/run_027_residual_refinement.py"),
                      "benchmark", "--allow-gpu", "--config", str(config), "--root", str(root),
                      "--gpus", "0", "1", "2", "3"]


def read_optional(path):
    return read_json(path) if path.is_file() else {}


def report(fp16_root, fp32_root, controller):
    rows = []
    for arm in ARMS:
        for precision, root in (("FP16 AMP", fp16_root), ("FP32", fp32_root)):
            folder = root / "benchmark" / arm
            status = read_optional(folder / "status.json")
            training = read_optional(folder / "training.json").get("updates", [])
            warmup = read_optional(folder / "warmup.json").get("updates", [])
            timing = read_optional(folder / "training_timing_until_0000100.json")
            result = read_optional(folder / "evaluations/update_0000100/summary.json")
            metrics, evaluation = result.get("metrics", {}), result.get("timing", {})
            row = {"arm": arm, "precision": precision, "status": status.get("status", "pending"),
                   "updates": training[-1]["update"] if training else None,
                   "evaluation_findings": metrics.get("full", {}).get("findings", status.get("findings_done")),
                   "training_seconds": timing.get("measured_wall_seconds"),
                   "step_median_seconds": timing.get("step_median_seconds"),
                   "step_p95_seconds": timing.get("step_p95_seconds"),
                   "data_loading_seconds": timing.get("data_seconds"),
                   "peak_training_gib": timing["peak_memory_bytes"] / 2**30 if timing else None,
                   "evaluation_seconds": evaluation.get("total_seconds"),
                   "evaluation_inference_seconds": evaluation.get("inference_seconds"),
                   "evaluation_metric_seconds": evaluation.get("metric_seconds"),
                   "evaluation_output_seconds": evaluation.get("output_seconds"),
                   "peak_evaluation_gib": evaluation["peak_memory_bytes"] / 2**30 if evaluation else None,
                   "training_amp_backoffs": sum(r.get("amp_overflow_retries", 0) for r in training) if training else None,
                   "warmup_amp_backoffs": sum(r.get("amp_overflow_retries", 0) for r in warmup) if warmup else None,
                   "error": status.get("error")}
            for half in ("A", "B", "full"):
                for metric in ("dice", "hit_rate"):
                    row[f"{half}_{metric}"] = metrics.get(half, {}).get(metric)
            rows.append(row)
    payload = {"updated_at": now(), "controller": controller, "rows": rows,
               "report_renderer_sha256": sha256(__file__),
               "fp16_root": str(fp16_root), "fp32_root": str(fp32_root), "ranking": "pending_user_review"}
    path = fp16_root / "reports/precision_comparison"
    atomic_json(path.with_suffix(".json"), payload)
    atomic_csv(path.with_suffix(".csv"), rows)
    def show(value, digits=3):
        return "pending" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    lines = ["# Exp027 FP16 / FP32 comparison", "", f"Last refreshed: {payload['updated_at']}", "",
             f"Controller: **{controller['status']}**. Results remain pending user review.", "",
             "[FP16 dashboard](live_dashboard.md) · [FP32 dashboard](../precision_fp32/reports/live_dashboard.md)", "",
             "FP32 disables autocast, loss scaling and TF32 acceleration. Inputs, initial weights, schedules,",
             "192³ tiles, LR 1e-4, clipping 1.0 and full-volume evaluation are matched. No automatic ranking.", "",
             "| Run | Precision | Phase | Updates | Eval findings | Train s | Load s | Peak train GiB | Eval s | Training / warm-up AMP backoffs |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append("| " + " | ".join(show(r[k]) for k in ("arm", "precision", "status", "updates",
                     "evaluation_findings", "training_seconds", "data_loading_seconds", "peak_training_gib",
                     "evaluation_seconds")) + f" | {show(r['training_amp_backoffs'])} / {show(r['warmup_amp_backoffs'])} |")
    lines += ["", "| Run | Precision | Dice A | Dice B | Dice full | Hit rate A | Hit rate B | Hit rate full |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| " + " | ".join(show(r[k], 7) for k in ("arm", "precision", "A_dice", "B_dice", "full_dice",
                                                             "A_hit_rate", "B_hit_rate", "full_hit_rate")) + " |")
    lines += ["", "A is refiner training data for runs 2–4; B for run 4. Full results for runs 2–3 have mixed exposure.",
              "Warm-up and data preparation are separate from measured training. Sequential trials can have different",
              "filesystem-cache effects; report loading time separately when interpreting speed differences.",
              "FP32 has no AMP backoff mechanism; any non-finite loss/gradient stops that trial.",
              "A successful 100-update test does not establish stability for every future training schedule."]
    errors = [f"{r['arm']} {r['precision']}: {r['error']}" for r in rows if r["error"]]
    if controller.get("error"):
        errors.insert(0, controller["error"])
    if errors:
        lines += ["", "Failures:", "", *[f"- {e}" for e in errors]]
    atomic_text(path.with_suffix(".md"), "\n".join(lines) + "\n")


def initialize(fp16_root, fp32_root, fp16, fp32, config_path):
    if read_json(fp16_root / "benchmark/code_manifest.json") != code_fingerprint():
        raise ValueError("Code changed since the FP16 trial; matched comparison refused")
    prepared = prepare(fp32, fp32_root, config_path=config_path)
    if prepared != read_json(fp16_root / "prepared.json"):
        raise ValueError("Prepared data differ between precisions")
    contract = cache_contract(fp32, prepared)
    if contract != cache_contract(fp16, prepared):
        raise ValueError("Precision change unexpectedly alters the base cache contract")
    caches = sorted((fp16_root / "cache").glob("*/metadata.json"))
    if len(caches) != 196:
        raise ValueError("All 196 cached cases must exist; no logit generation in this comparison")
    for p in caches:
        meta = read_json(p)
        if meta["contract"] != contract or not all((p.parent / name).is_file() for name in meta["hashes"]):
            raise ValueError(f"Missing/incompatible cache: {p}")
    cache = fp32_root / "cache"
    if cache.is_symlink():
        if cache.resolve() != (fp16_root / "cache").resolve():
            raise ValueError("FP32 cache points to different inputs")
    elif cache.exists():
        raise ValueError("FP32 cache must reference the unchanged FP16 inputs")
    else:
        cache.symlink_to(fp16_root / "cache", target_is_directory=True)
    for arm in ARMS:
        expected = schedule(prepared, arm, 100, fp32["seed"])
        prior = read_json(fp16_root / "benchmark" / arm / "schedule.json")
        if prior != {"events": expected, "sha256": digest(expected)}:
            raise ValueError(f"Sampling schedule differs: {arm}")
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FP32_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorized wait and GPU launch")
    args = parser.parse_args(argv)
    fp16, fp32 = read_json(DEFAULT_CONFIG), read_json(args.config)
    validate_pair(fp16, fp32)
    fp16_root, fp32_root = Path(fp16["experiment_dir"]), Path(fp32["experiment_dir"])
    launch = read_json(fp16_root / "run_manifest.json")
    command = docker_command(args.config.resolve(), fp32_root, launch["image_id"])
    if not args.execute:
        print({"gpu_work_started": False, "prerequisite": str(fp16_root), "fp32_root": str(fp32_root),
               "updates_per_arm": 100, "validation_findings_per_arm": 69, "command": command})
        return 0
    gpu_gate(args.execute)
    with lock(fp16_root / ".precision_comparison.lock"):
        controller = {"status": "waiting_for_fp16", "started_at": now(), "host_pid": os.getpid(),
                      "authorization": "User requested the same FP32 benchmark after FP16 finishes.",
                      "script_sha256": sha256(__file__), "config_sha256": sha256(args.config),
                      "code": code_fingerprint(), "tf32_override": "0", "command": command}
        controller_path = fp16_root / "precision_comparison_controller.json"
        child = None
        def refresh(status=None, **extra):
            if status:
                controller["status"] = status
            controller.update(updated_at=now(), **extra)
            atomic_json(controller_path, controller)
            report(fp16_root, fp32_root, controller)
        def interrupted(*_):
            raise KeyboardInterrupt("Precision comparison interrupted")
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for s in old_handlers:
            signal.signal(s, interrupted)
        try:
            initialize(fp16_root, fp32_root, fp16, fp32, args.config)
            if read_optional(fp32_root / "status.json").get("status") == "awaiting_timing_approval":
                atomic_json(fp32_root / "status.json", {"status": "waiting_for_fp16", "updated_at": now(),
                            "active_phase": "benchmark", "ranking": "pending_user_review"})
            from exp027_report import report_command
            report_command(fp32_root)
            refresh()
            deadline = time.monotonic() + 24 * 3600
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("FP16 prerequisite did not finish/release GPUs within 24 hours")
                if completion(fp16_root) and execution_released(fp16_root):
                    idle, gpu_rows = idle_gpus()
                    if idle:
                        break
                    refresh("waiting_for_gpu_release", gpu_preflight=gpu_rows)
                else:
                    refresh("waiting_for_fp16")
                time.sleep(30)
            if code_fingerprint() != controller["code"] or sha256(args.config) != controller["config_sha256"]:
                raise ValueError("Code/config changed while waiting; refusing unmatched launch")
            if (fp32_root / "timing_report.json").exists() or list((fp32_root / "benchmark").glob("*/training_timing_until_*.json")):
                raise ValueError("FP32 trial already entered training; never overwrite/repeat a timing trial")
            refresh("launching_fp32", gpu_preflight=gpu_rows, fp16_finished_at=now())
            log = fp32_root / "logs/benchmark_launch.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                atomic_json(fp32_root / "run_manifest.json", {**controller, "host_pid": child.pid,
                            "image_id": launch["image_id"], "launch_log": str(log), "launched_at": now(),
                            "fp16_prerequisite_timing_sha256": sha256(fp16_root / "timing_report.json"),
                            "cached_cases_reused": 196, "full_training_authorized": False})
                refresh("fp32_running", child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(30)
                    refresh()
            if child.returncode != 0 or not completion(fp32_root):
                raise RuntimeError(f"FP32 benchmark failed (exit {child.returncode}); see {log}")
            for arm in ARMS:
                a = read_json(fp16_root / "benchmark" / arm / "training.json")
                b = read_json(fp32_root / "benchmark" / arm / "training.json")
                if a["initial_weights_sha256"] != b["initial_weights_sha256"]:
                    raise ValueError(f"Initial weights differ between precisions: {arm}")
            refresh("complete_pending_user_review")
        except BaseException as exc:
            if child and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            refresh("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
