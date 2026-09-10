#!/usr/bin/env python3
"""Host supervisor for the immutable four-GPU test300 export job."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import fresh_cache as fc
import test300_cache as tc

CPU_STATE = (
    fc.RUNTIME_ROOT
    / "analysis_jobs/a001_top16_after_wave4_fullval/orchestrator_state.json"
)


def command(args, check=True):
    p = subprocess.run(args, capture_output=True, text=True)
    if check:
        tc.require(p.returncode == 0, p.stderr or p.stdout)
    return p


def docker_state(name):
    p = command(["docker", "inspect", name], check=False)
    if p.returncode:
        tc.require("No such" in p.stderr, "cannot establish Docker state: " + p.stderr)
        return None
    return json.loads(p.stdout)[0]


def gpu_free():
    raw = command(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).stdout
    return {
        int(line.split(",")[0]): int(line.split(",")[1]) for line in raw.splitlines()
    }


def progress(job):
    records = {}
    for c in job["candidates"]:
        p = Path(c["progress_path"])
        if p.exists():
            d = fc.read_json(p)
            tc.require(
                d["job_spec_sha256"] == job["job_spec_sha256"], "foreign progress"
            )
            records[c["id"]] = d
    return records


def completed(job, c, verify_hashes=False):
    root = Path(c["cache_root"])
    p = root / "export_manifest.json"
    v = root / "reproduction_validation.json"
    if not p.exists():
        return False
    m = fc.read_json(p)
    tc.require(m["job_spec_sha256"] == job["job_spec_sha256"], "foreign cache")
    if m["status"] != "strict_passed":
        return False
    validation = fc.read_json(v)
    tc.require(
        validation.get("job_spec_sha256") == job["job_spec_sha256"]
        and validation["status"] == "passed"
        and validation["cases"] == 300
        and validation["findings"] == 582
        and validation["array_hashes_verified"]
        and validation["same_pass_mask_mismatch_voxels"] == 0,
        "invalid completed cache",
    )
    tc.require(len(m["cases"]) == 300, "incomplete published manifest")
    for rec in m["cases"]:
        p = Path(rec["array_path"])
        tc.require(
            p.is_file() and p.stat().st_size == rec["npy_bytes"],
            "completed cache file missing/size drift",
        )
    if verify_hashes:
        cases = tc.load_cases(tc.resolve(job["dataset"]["path"]))
        by_name = {r["name"]: r for r in m["cases"]}
        tc.require(
            set(by_name) == {case["name"] for case in cases}, "completed case set drift"
        )
        for case in cases:
            rec = by_name[case["name"]]
            tc.check_array(
                Path(rec["array_path"]),
                rec,
                case,
                job["job_spec_sha256"],
                c["cache_key"],
            )
    return True


def sync(job_path):
    command(
        [
            "python",
            str(tc.HERE / "hub.py"),
            "cache",
            "sync",
            "--job",
            job_path.parent.name,
            "--apply",
        ]
    )


def summarize(job, root):
    rows = []
    for c in job["candidates"]:
        if not completed(job, c):
            continue
        m = fc.read_json(Path(c["cache_root"]) / "export_manifest.json")
        v = fc.read_json(Path(c["cache_root"]) / "reproduction_validation.json")
        rows.append(
            {
                "rank": c["rank"],
                "candidate_id": c["id"],
                "checkpoint_sha256": c["checkpoint"]["sha256"],
                "paired_val200": c["paired_val200"],
                "test_cache_key": c["cache_key"],
                "cache_path": c["cache_root"],
                "manifest_sha256": fc.sha256_file(
                    Path(c["cache_root"]) / "export_manifest.json"
                ),
                "validation_sha256": fc.sha256_file(
                    Path(c["cache_root"]) / "reproduction_validation.json"
                ),
                "cases": v["cases"],
                "findings": v["findings"],
                "dtype": v["dtype"],
                "bytes": v["array_bytes"],
                "elapsed_seconds": m["elapsed_seconds"],
            }
        )
    inventory = {
        "job_id": job["job_id"],
        "job_spec_sha256": job["job_spec_sha256"],
        "completed_models": len(rows),
        "expected_models": 20,
        "status": "complete" if len(rows) == 20 else "running",
        "metric_status": tc.METRIC_STATUS,
        "array_bytes": sum(r["bytes"] for r in rows),
        "models": rows,
        "wave_timings": [
            fc.read_json(p) for p in sorted(root.glob("wave_*_timing.json"))
        ],
        "warnings": [],
        "shared_free_bytes": shutil.disk_usage(root).free,
        "local_free_bytes": shutil.disk_usage(job["staging_root"]).free,
        "preprocessing_bytes": sum(
            p.stat().st_size for p in tc.NATIVE_ROOT.rglob("*") if p.is_file()
        ),
        "updated_at_utc": fc.utc_now(),
    }
    inventory["warnings"] = [
        f"Rank {r['rank']} required float32 to preserve threshold masks"
        for r in rows
        if r["dtype"] == "float32"
    ]
    fc.atomic_write_json(root / "inventory.json", inventory)
    lines = [
        "# GPU8 test300 base-logit inventory",
        "",
        f"Status: {inventory['status']}; {len(rows)}/20 models.",
        "",
        "Test labels are withheld; no Dice/hit metrics or ensemble decisions are produced.",
        "",
        "| Rank | Candidate | Cases/prompts | Dtype | GiB | Worker hours |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    lines += [
        f"| {r['rank']} | `{r['candidate_id']}` | {r['cases']}/{r['findings']} | {r['dtype']} | {r['bytes']/1024**3:.2f} | {r['elapsed_seconds']/3600:.2f} |"
        for r in rows
    ]
    lines += [
        "",
        f"Total array storage: {inventory['array_bytes']/1024**4:.3f} TiB.",
        f"Native preprocessing: {inventory['preprocessing_bytes']/1024**3:.2f} GiB.",
        f"Shared free: {inventory['shared_free_bytes']/1024**4:.2f} TiB.",
        "",
        "Paired checkpoint, val200, test300, and validation hashes: `inventory.json`.",
        "",
        "Wave timings:",
    ]
    lines += [
        f"- Wave {v['wave']}: {v['wall_seconds']/3600:.2f} hours."
        for v in inventory["wave_timings"]
    ]
    lines += [
        "",
        "Warnings: " + ("; ".join(inventory["warnings"]) or "none"),
        "",
        f"Resume: `python side_experiments/sideexp003_ensemble_method_hub/hub.py cache run --job {job['job_id']} --auto-continue`.",
    ]
    fc.atomic_write_text(root / "report.md", "\n".join(lines) + "\n")
    return inventory


class Runner:
    def __init__(self, path):
        self.path = Path(path)
        self.job = fc.read_json(self.path)
        self.root = Path(self.job["runtime_root"])
        self.active = []
        self.interrupted = False
        self.paused = None
        self.paused_id = None
        self.degraded = 0
        self.recovered = 0
        self.started = time.monotonic()
        self.baseline = {}
        for c in self.job["candidates"]:
            tc.require(
                fc.sha256_file(Path(c["paired_val200"]["export_manifest"]))
                == c["paired_val200"]["export_sha256"],
                "paired val timing provenance drift",
            )
            m = fc.read_json(Path(c["paired_val200"]["export_manifest"]))
            good = [r for r in m["cases"] if r.get("elapsed_seconds", 0) > 0]
            self.baseline[c["id"]] = sum(r["elapsed_seconds"] for r in good) / sum(
                __import__("math").prod(r["shape"]) for r in good
            )

    def state(self, status, **kw):
        d = {
            "job_id": self.job["job_id"],
            "job_spec_sha256": self.job["job_spec_sha256"],
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "status": status,
            "updated_at_utc": fc.utc_now(),
            "elapsed_seconds": time.monotonic() - self.started,
            "paused_cpu_container": self.paused,
            **kw,
        }
        fc.atomic_write_json(self.root / "state.json", d)
        print(json.dumps(d), flush=True)

    def check(self):
        tc.require(not self.interrupted, "operator interruption")
        tc.require(not (self.root / "control/ABORT").exists(), "ABORT sentinel present")
        tc.validate_job(self.job)
        tc.require(
            command(["git", "branch", "--show-current"], check=True).stdout.strip()
            == "gpu8",
            "branch changed",
        )
        tc.require(
            shutil.disk_usage(self.root).free >= tc.MIN_SHARED,
            "shared free disk below 20 TiB",
        )

    def sleep(self, seconds=60):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tc.require(not self.interrupted, "operator interruption")
            time.sleep(min(1, max(0, end - time.monotonic())))

    def cpu_ready(self):
        if not CPU_STATE.exists():
            return False
        d = fc.read_json(CPU_STATE)
        session = Path(d.get("session", "/nonexistent"))
        return (session / "selected_execution.json").is_file() and d.get(
            "status"
        ) not in ("benchmarking", "launching_benchmark")

    def release_pause(self):
        if self.paused:
            d = docker_state(self.paused)
            if (
                d
                and d["Id"] == self.paused_id
                and d["State"]["Running"]
                and d["State"]["Paused"]
            ):
                command(["docker", "unpause", self.paused])
            fc.atomic_write_json(
                self.root / "cpu_pause.json",
                {
                    "status": "released",
                    "container": self.paused,
                    "updated_at_utc": fc.utc_now(),
                },
            )
            self.paused = None
            self.paused_id = None

    def contention(self, records):
        ratios = []
        for cid, p in records.items():
            recent = p.get("recent_case_costs", [])
            if p["status"] == "exporting" and len(recent) >= 10:
                cost = sum(r["seconds"] for r in recent) / sum(
                    r["elements"] for r in recent
                )
                ratios.append(cost / self.baseline[cid])
        ratio = sorted(ratios)[len(ratios) // 2] if ratios else None
        if self.paused:
            self.recovered = (
                self.recovered + 1 if ratio is not None and ratio < 1.5 else 0
            )
            if self.recovered >= 2 or not ratios:
                self.release_pause()
        else:
            self.degraded = self.degraded + 1 if ratio is not None and ratio > 2 else 0
            if self.degraded >= 2 and CPU_STATE.exists():
                cpu = fc.read_json(CPU_STATE)
                name = cpu.get("container")
                if (
                    cpu.get("host") == "shenggpu8"
                    and name
                    and name.startswith("rex-resume-")
                    and name.endswith("-production")
                ):
                    d = docker_state(name)
                    if d and d["State"]["Running"] and not d["State"]["Paused"]:
                        fc.atomic_write_json(
                            self.root / "cpu_pause.json",
                            {
                                "status": "requesting",
                                "container": name,
                                "container_id": d["Id"],
                                "ratio": ratio,
                                "updated_at_utc": fc.utc_now(),
                            },
                        )
                        self.paused = name
                        self.paused_id = d["Id"]
                        command(["docker", "pause", name])
                        fc.atomic_write_json(
                            self.root / "cpu_pause.json",
                            {
                                "status": "owned",
                                "container": name,
                                "container_id": d["Id"],
                                "ratio": ratio,
                                "updated_at_utc": fc.utc_now(),
                            },
                        )
        return ratio

    def launch(self, c):
        name = f"sideexp003_{self.job['job_id']}_w{c['wave']:02d}_g{c['gpu']}"
        d = docker_state(name)
        if d:
            tc.require(
                d.get("Config", {}).get("Labels", {}).get("rex.test_job")
                == self.job["job_spec_sha256"],
                "foreign container with matching name",
            )
            tc.require(not d["State"]["Running"], f"existing live worker: {name}")
            logs = command(["docker", "logs", name], check=False)
            fc.atomic_write_text(
                self.root / "logs" / f"{name}.previous.log", logs.stdout + logs.stderr
            )
            command(["docker", "rm", name])
        cache = Path(c["cache_root"])
        stage = Path(c["staging_root"])
        cache.mkdir(parents=True, exist_ok=True)
        stage.mkdir(parents=True, exist_ok=True)
        fc.atomic_write_json(
            Path(c["progress_path"]),
            {
                "job_id": self.job["job_id"],
                "job_spec_sha256": self.job["job_spec_sha256"],
                "candidate_id": c["id"],
                "status": "queued",
                "cases": 0,
                "findings": 0,
                "updated_at_utc": fc.utc_now(),
            },
        )
        relative = self.path.resolve().relative_to(tc.REPO)
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"rex.test_job={self.job['job_spec_sha256']}",
            "--gpus",
            f"device={c['gpu']}",
            "--ipc=host",
            "--shm-size=32g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-w",
            "/workspace",
            "-v",
            f"{tc.REPO}:/workspace:ro",
            "-v",
            "/data/hengjie:/data/hengjie:ro",
            "-v",
            "/mnt/shengdata1:/mnt/shengdata1:ro",
            "-v",
            f"{cache}:{cache}:rw",
            "--tmpfs",
            "/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m",
            "-v",
            f"{stage}:{stage}:rw",
            "-v",
            f"{self.root}:{self.root}:rw",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home",
            "-e",
            "HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            "OMP_NUM_THREADS=4",
            "-e",
            "MKL_NUM_THREADS=4",
            self.job["container"]["image_id"],
            "python",
            "/workspace/side_experiments/sideexp003_ensemble_method_hub/test300_cache.py",
            "--job",
            "/workspace/" + str(relative),
            "--candidate-id",
            c["id"],
        ]
        fc.atomic_write_json(
            self.root / "commands" / f'{c["id"]}.json',
            {"argv": cmd, "created_at_utc": fc.utc_now()},
        )
        command(cmd)
        self.active.append(name)

    def collect(self):
        for name in self.active:
            d = docker_state(name)
            tc.require(
                d is not None and not d["State"]["Running"], "worker has not exited"
            )
            logs = command(["docker", "logs", name], check=False)
            fc.atomic_write_text(
                self.root / "logs" / f"{name}.log", logs.stdout + logs.stderr
            )
            tc.require(d["State"]["ExitCode"] == 0, f"worker exit failure: {name}")
            command(["docker", "rm", name])
        self.active = []

    def stop(self):
        for name in self.active:
            d = docker_state(name)
            if d and d["State"]["Running"]:
                command(["docker", "stop", "--time", "20", name])
            d = docker_state(name)
            tc.require(
                not d or not d["State"]["Running"],
                f"cannot confirm stopped worker {name}",
            )
            if d:
                p = command(["docker", "logs", name], check=False)
                fc.atomic_write_text(
                    self.root / "logs" / f"{name}.interrupted.log", p.stdout + p.stderr
                )
        self.active = []

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with tc.exclusive(self.root / "launch.lock"):
            # Recover only a pause explicitly owned by an earlier instance of this job.
            pause = self.root / "cpu_pause.json"
            if pause.exists() and fc.read_json(pause).get("status") in (
                "owned",
                "requesting",
            ):
                previous = fc.read_json(pause)
                self.paused = previous["container"]
                self.paused_id = previous["container_id"]
                self.release_pause()
            self.check()
            tc.require(
                socket.gethostname() == "shenggpu8",
                "test job is pinned to gpu8 local staging",
            )
            tc.require(
                command(
                    [
                        "docker",
                        "image",
                        "inspect",
                        self.job["container"]["image"],
                        "--format",
                        "{{.Id}}",
                    ]
                ).stdout.strip()
                == self.job["container"]["image_id"],
                "image ID drift",
            )
            Path(self.job["staging_root"]).mkdir(parents=True, exist_ok=True)
            try:
                while not self.cpu_ready():
                    self.check()
                    self.state("waiting_for_cpu_benchmark")
                    self.sleep()
                for c in self.job["candidates"]:
                    cache = Path(c["cache_root"])
                    if cache.exists():
                        with tc.exclusive(cache / ".worker.lock"):
                            if completed(self.job, c, verify_hashes=True):
                                stage = Path(c["staging_root"])
                                if stage.exists():
                                    tc.cleanup_stage(stage)
                                    stage.rmdir()
                for wave in range(1, 6):
                    members = [c for c in self.job["candidates"] if c["wave"] == wave]
                    todo = [c for c in members if not completed(self.job, c)]
                    if not todo:
                        continue
                    while True:
                        self.check()
                        free = gpu_free()
                        space = tc.wave_space_ok(
                            shutil.disk_usage(self.root).free,
                            shutil.disk_usage(self.job["staging_root"]).free,
                            len(todo),
                            self.job["output_elements_per_model"],
                        )
                        if space and all(free[c["gpu"]] >= 20000 for c in todo):
                            break
                        self.state(
                            "waiting_for_resources",
                            wave=wave,
                            space_ok=space,
                            free_gpu_mib=free,
                        )
                        self.sleep()
                    gate = self.root / "control" / f"wave_{wave:02d}.continue"
                    gate.unlink(missing_ok=True)
                    for c in todo:
                        self.launch(c)
                    begin = time.monotonic()
                    released = False
                    while True:
                        self.check()
                        records = progress(self.job)
                        current = {
                            c["id"]: records[c["id"]]
                            for c in todo
                            if c["id"] in records
                        }
                        tc.require(
                            not any(p["status"] == "failed" for p in current.values()),
                            "candidate failed: "
                            + str(
                                {
                                    k: v.get("error")
                                    for k, v in current.items()
                                    if v["status"] == "failed"
                                }
                            ),
                        )
                        for name in self.active:
                            d = docker_state(name)
                            tc.require(d is not None, "worker disappeared")
                            tc.require(
                                d["State"]["Running"] or d["State"]["ExitCode"] == 0,
                                f"worker exited unsuccessfully: {name}",
                            )
                        if (
                            not released
                            and len(current) == len(todo)
                            and all(
                                p["status"] in ("smoke_waiting", "strict_passed")
                                for p in current.values()
                            )
                        ):
                            tc.require(
                                all(
                                    p["status"] == "strict_passed"
                                    or p["free_mib_after_smoke"] >= 4096
                                    for p in current.values()
                                ),
                                "smoke memory gate failed",
                            )
                            fc.atomic_write_json(
                                gate,
                                {
                                    "job_spec_sha256": self.job["job_spec_sha256"],
                                    "released_at_utc": fc.utc_now(),
                                },
                            )
                            released = True
                        ratio = self.contention(current)
                        self.state(
                            "exporting" if released else "smoke_running",
                            wave=wave,
                            completed_models=sum(
                                p["status"] == "strict_passed" for p in records.values()
                            ),
                            candidates={
                                k: {
                                    f: v.get(f)
                                    for f in ("status", "cases", "eta_seconds")
                                }
                                for k, v in current.items()
                            },
                            normalized_cost_ratio=ratio,
                            free_gpu_mib=gpu_free(),
                            shared_free_bytes=shutil.disk_usage(self.root).free,
                            local_free_bytes=shutil.disk_usage(
                                self.job["staging_root"]
                            ).free,
                            cpu_search=fc.read_json(CPU_STATE),
                        )
                        sync(self.path)
                        if len(current) == len(todo) and all(
                            p["status"] == "strict_passed" for p in current.values()
                        ):
                            if all(
                                not docker_state(n)["State"]["Running"]
                                for n in self.active
                            ):
                                break
                        self.sleep()
                    self.collect()
                    for c in todo:
                        stage = Path(c["staging_root"])
                        if stage.exists():
                            tc.require(
                                not any(stage.iterdir()),
                                "worker left staging contents after completion",
                            )
                            stage.rmdir()
                    self.release_pause()
                    inventory = summarize(self.job, self.root)
                    timing = {
                        "wave": wave,
                        "wall_seconds": time.monotonic() - begin,
                        "completed_models": inventory["completed_models"],
                        "updated_at_utc": fc.utc_now(),
                    }
                    if wave == 1:
                        factor = timing["wall_seconds"] / (2.929109628625835 * 3600)
                        remaining = (
                            sum(
                                [
                                    2.494742310586282,
                                    2.4969962151534855,
                                    2.435219897758216,
                                    2.4369275028920834,
                                ]
                            )
                            * factor
                        )
                        timing["remaining_eta_hours"] = [
                            remaining * 0.8,
                            remaining * 1.3,
                        ]
                    fc.atomic_write_json(
                        self.root / f"wave_{wave:02d}_timing.json", timing
                    )
                final = summarize(self.job, self.root)
                tc.require(final["completed_models"] == 20, "incomplete inventory")
                sync(self.path)
                tracked = self.path.parent
                for name in ("inventory.json", "report.md"):
                    fc.atomic_write_text(tracked / name, (self.root / name).read_text())
                self.state(
                    "complete", completed_models=20, array_bytes=final["array_bytes"]
                )
            except BaseException as exc:
                self.stop()
                self.state("failed", error=str(exc))
                raise
            finally:
                self.release_pause()


def watch(path, interval=60, once=False):
    job = fc.read_json(path)
    root = Path(job["runtime_root"])
    while True:
        state = (
            fc.read_json(root / "state.json")
            if (root / "state.json").exists()
            else {"status": "not_started"}
        )
        print(
            json.dumps({"state": state, "candidates": progress(job)}, indent=2),
            flush=True,
        )
        if once or state["status"] in ("complete", "failed"):
            return 0 if state["status"] != "failed" else 2
        time.sleep(interval)


def run_job(path):
    runner = Runner(path)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(runner, "interrupted", True))
    runner.run()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    a = p.parse_args()
    run_job(a.job)
