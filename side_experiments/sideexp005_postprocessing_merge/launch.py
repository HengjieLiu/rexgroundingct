#!/usr/bin/env python3
"""Preflight and detach the authorized SideExp005 CPU audit on gpu8."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def command(cfg, runtime):
    args = ["docker", "run", "--network", "none", "--cpus", "4", "--memory", "96g", "--shm-size", "1g",
            "--user", f"{os.getuid()}:{os.getgid()}", "--init",
            "--mount", f"type=bind,src={REPO},dst={REPO},readonly",
            "--mount", "type=bind,src=/mnt/shengdata1,dst=/mnt/shengdata1,readonly",
            "--mount", "type=bind,src=/data/hengjie/datasets/rexgroundingct/segmentations,dst=/data/hengjie/datasets/rexgroundingct/segmentations,readonly",
            "--mount", f"type=bind,src={runtime},dst={runtime}",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "PYTHONUNBUFFERED=1"]
    for key, value in cfg["environment"].items():
        args += ["-e", f"{key}={value}"]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text())
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "gpu8":
        raise SystemExit("SideExp005 launch requires branch gpu8")
    image = subprocess.check_output(["docker", "image", "inspect", cfg["image_id"], "--format", "{{.Id}}"], text=True).strip()
    if image != cfg["image_id"]:
        raise SystemExit("pinned image identity mismatch")
    runtime = Path(cfg["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    base = command(cfg, runtime)
    name = "sideexp005_r001_d1_val200_frozen_postprocessing"
    listing = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
    if name in listing:
        raise SystemExit("Existing named supervisor; inspect its state before an explicit restart")
    preflight = base + ["--rm", cfg["image_id"], "python", str(ROOT / "runner.py"), "preflight"]
    if (runtime / "job_manifest.json").exists() and not args.preflight_only:
        from runner import read, verify_job
        job = read(runtime / "job_manifest.json")
        verify_job(cfg, job)
        gate = read(runtime / "preflight.json")
        if gate.get("status") != "passed" or gate.get("job_sha256") != job["job_sha256"]:
            raise SystemExit("Frozen preflight evidence does not match this job")
    else:
        subprocess.run(preflight, check=True)
    if args.preflight_only:
        return
    frozen = runtime / "source" / ROOT.relative_to(REPO) / "runner.py"
    launch = base + ["--detach", "--name", name, cfg["image_id"], "python", str(frozen), "run"]
    container = subprocess.check_output(launch, text=True).strip()
    record = {"container_name": name, "container_id": container, "image_id": image,
              "preflight_command": preflight, "launch_command": launch,
              "watch_command": ["python", str(ROOT / "runner.py"), "watch", "--once"],
              "log_command": ["docker", "logs", name], "runtime_root": str(runtime)}
    (runtime / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    (ROOT / "launch_report.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
