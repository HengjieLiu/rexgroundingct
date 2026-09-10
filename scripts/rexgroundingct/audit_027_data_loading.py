#!/usr/bin/env python3
"""Read existing histories and schedules; no image reads, timing reruns or GPUs."""
from collections import OrderedDict
from pathlib import Path
import subprocess

import numpy as np

from exp027_common import ARMS, DEFAULT_CONFIG, atomic_csv, atomic_json, code_fingerprint, now, read_json


def summarize(root):
    root = Path(root)
    rows, reuse = [], []
    for arm in ARMS:
        first = root / "benchmark" / arm
        second = root / "precision_fp32/benchmark" / arm
        events = read_json(first / "schedule.json")
        if events != read_json(second / "schedule.json"):
            raise ValueError(f"Schedules differ: {arm}")
        for precision, folder in (("FP16", first), ("FP32", second)):
            history = read_json(folder / "training.json")["updates"]
            timing = read_json(folder / "training_timing_until_0000100.json")
            if [(v["update"], v["key"], v["mode"], v["source"]) for v in history] != [
                    (v["update"], v["key"], v["mode"], v["source"]) for v in events["events"]]:
                raise ValueError(f"Training did not follow the schedule: {arm} {precision}")
            loading = np.array([v["data_seconds"] for v in history])
            other = np.array([v["step_seconds"] - v["data_seconds"] for v in history])
            wall = timing["measured_wall_seconds"]
            rows.append({"arm": arm, "precision": precision, "updates": len(history),
                         "load_mean_seconds": float(loading.mean()), "load_median_seconds": float(np.median(loading)),
                         "load_p95_seconds": float(np.percentile(loading, 95)), "load_max_seconds": float(loading.max()),
                         "loads_over_1_second": int((loading > 1).sum()),
                         "non_loading_step_mean_seconds": float(other.mean()),
                         "load_fraction_of_training_wall": float(loading.sum() / wall),
                         "training_wall_seconds": wall,
                         "zero_loading_cost_speedup_ceiling": float(wall / (wall - loading.sum()))})
        names = [v["key"].split("::")[0] for v in events["events"]]
        for capacity in (2, 8, 16, 32, 64, 128, 196):
            resident, hits = OrderedDict(), 0
            for name in names:
                hits += int(name in resident)
                resident[name] = True
                resident.move_to_end(name)
                while len(resident) > capacity:
                    resident.popitem(last=False)
            reuse.append({"arm": arm, "capacity_cases": capacity, "accesses": len(names), "metadata_mapping_hits": hits,
                          "note": "Serial schedule simulation, not OS page-cache hits or measured latency."})
    return rows, reuse


def main():
    root = Path(read_json(DEFAULT_CONFIG)["experiment_dir"])
    rows, reuse = summarize(root)
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "Cached", "Buffers"):
            meminfo[key + "_bytes"] = int(value.strip().split()[0]) * 1024
    storage = subprocess.check_output(["findmnt", "-T", str(root / "cache"), "-o", "TARGET,SOURCE,FSTYPE"], text=True)
    audit = {"recorded_at": now(), "scope": "read-only history/code/storage audit; no performance workloads",
             "code": code_fingerprint(), "rows": rows, "lru_simulation": reuse,
             "host_memory_snapshot": meminfo, "storage_mount": storage,
             "interpretation": "FP16 ran first. FP32 used identical scheduled patches later. Warmer filesystem pages and different resource contention are confounders; precision does not change this CPU loader.",
             "batch_payload_bytes": 4 * 192**3 * 4,
             "four_arms_four_ready_batches_bytes": 4 * 4 * 4 * 192**3 * 4}
    folder = root / "audits/data_loading"
    atomic_json(folder / "audit.json", audit)
    atomic_csv(folder / "timings.csv", rows)
    atomic_csv(folder / "metadata_mapping_reuse.csv", reuse)
    print(folder / "audit.json")


if __name__ == "__main__":
    main()
