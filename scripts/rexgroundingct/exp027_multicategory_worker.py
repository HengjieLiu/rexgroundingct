"""Resumable CPU imports and guarded GPU exports, isolated from refiner workers."""
from __future__ import annotations

import os
import signal
import threading
import time
import traceback
from pathlib import Path

from exp027_common import append_update, atomic_json, lock, now, read_json, require_hash
from exp027_data import case_key, validate_sources, write_case_cache
from exp027_multicategory_data import build_index, check_context, verify_case


def require_gpu(allowed, share):
    if not allowed or os.environ.get("START_GPU_WORK") != "1" or not share:
        raise PermissionError("Export requires START_GPU_WORK=1, --allow-gpu and --share-gpus")


class MemoryGuard:
    def __init__(self, root, attempt, worker, config):
        self.root, self.attempt, self.worker, self.config = Path(root), Path(attempt), worker, config
        self.stop = threading.Event()
        self.minimum_free = None
        self.failure = None

    def sample(self):
        import torch
        free, total = torch.cuda.mem_get_info(0)
        mib = 1024 ** 2
        self.minimum_free = min(self.minimum_free if self.minimum_free is not None else free, free)
        value = {"free_mib": free / mib, "minimum_free_mib": self.minimum_free / mib,
                 "total_mib": total / mib, "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / mib,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / mib, "updated_at": now()}
        atomic_json(self.root / "progress" / f"{self.worker}_memory.json", value)
        if free / mib < self.config["gpu"]["minimum_free_mib"]:
            raise RuntimeError(f"GPU free-memory reserve breached: {value}")
        return value

    def monitor(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except BaseException as exc:
                self.failure = f"{type(exc).__name__}: {exc}"
                atomic_json(self.attempt / "ABORT.json", {"worker": self.worker, "error": self.failure, "at": now()})
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.stop.wait(self.config["gpu"]["poll_seconds"])

    def start(self):
        import torch
        free, total = torch.cuda.mem_get_info(0)
        if free < self.config["gpu"]["minimum_start_free_mib"] * 1024 ** 2:
            raise RuntimeError(f"GPU requires 32 GiB free before export; found {free / 1024 ** 3:.2f}")
        torch.cuda.set_per_process_memory_fraction(self.config["gpu"]["allocator_limit_mib"] * 1024 ** 2 / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        self.sample()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)


def interrupted(signum, frame):
    raise InterruptedError(f"Cache worker received signal {signum}")


def process_case(root, name, config, context, prepared, entries, sources, predictor,
                 verify_only=False):
    root = Path(root)
    folder = root / "cases" / case_key(name)
    records = {r["key"]: r for r in prepared["records"]}
    with lock(root / "locks" / f"{case_key(name)}.lock"):
        reused = (folder / "complete.json").exists()
        if verify_only or reused:
            return predictor, verify_case(root, name, config, context, prepared, sources), reused
        # A crash before the marker leaves an incomplete case. Reuse complete
        # array metadata and finish its index; absent metadata rebuilds arrays.
        if not (folder / "metadata.json").exists():
            predictor = write_case_cache(config, records, entries, sources, folder, name,
                                         predictor, 0, context["cache_contract"])
        selected = [records[k] for k in prepared["cases"][name]["keys"]]
        build_index(folder, config, selected)
        proof = verify_case(root, name, config, context, prepared, sources, completing=True)
        return predictor, proof, False


def run_worker(config, root, attempt, kind, shard=0, allow_gpu=False, share=False, verify_only=False):
    root, attempt = Path(root), Path(attempt)
    context, prepared = check_context(config, root)
    source = validate_sources(config)
    sources = {r["name"]: r for r in source["cases"]}
    metadata = read_json(config["metadata"])
    entries = {r["name"]: r for split in ("train", "val") for r in metadata[split]}
    worker = f"{kind}{shard}" if kind != "val" else "val"
    if kind == "train":
        plan = prepared["shards"][shard]
        names, smoke = plan["names"], plan["smoke"]
    elif kind == "val":
        names = sorted(n for n, c in prepared["cases"].items() if c["split"] == "val")
        smoke = []
    elif kind == "verify":
        names, smoke = sorted(prepared["cases"])[shard::config["workers"]], []
        verify_only = True
    else:
        raise ValueError("Unknown worker kind")
    if kind == "train" and not verify_only:
        require_gpu(allow_gpu, share)
        require_hash(config["base"]["checkpoint"], config["base"]["checkpoint_sha256"])
        require_hash(source["candidate_source"]["plans"]["path"], source["candidate_source"]["plans"]["sha256"])
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    predictor, guard, cases, reused = None, None, [], 0
    name, phase = None, "starting"
    progress_path = root / "progress" / f"{worker}.json"
    receipt_path = root / "receipts" / f"{worker}.json"

    def publish(status, error=None):
        value = {"worker": worker, "status": status, "case": name, "done": len(cases), "total": len(names),
                 "reused": reused, "generated": len(cases) - reused if not verify_only else 0,
                 "verified": len(cases), "findings": sum(len(c["keys"]) for c in cases),
                 "cache_bytes": sum(c["cache_bytes"] for c in cases),
                 "elapsed_seconds": time.monotonic() - started, "updated_at": now(), "error": error}
        atomic_json(progress_path, value)
        return value

    with lock(root / "locks" / f"worker_{worker}.lock"):
        try:
            publish(phase)
            atomic_json(receipt_path, {"context_sha256": context["sha256"], "cases": []})
            if kind == "train" and not verify_only:
                import torch
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("One visible CUDA device required per cache worker")
                torch.set_num_threads(2)
                guard = MemoryGuard(root, attempt, worker, config)
                guard.start()
            for name in names:
                if (attempt / "ABORT.json").exists():
                    raise RuntimeError(f"Cache group aborted: {read_json(attempt / 'ABORT.json')}")
                phase = "verifying" if verify_only else ("smoke_export" if name in smoke else "exporting")
                publish(phase)
                t = time.monotonic()
                predictor, proof, was_reused = process_case(root, name, config, context, prepared, entries,
                                                          sources, predictor, verify_only)
                cases.append(proof)
                reused += int(was_reused)
                atomic_json(receipt_path, {"context_sha256": context["sha256"], "cases": cases})
                event = {"attempt": attempt.name, "worker": worker, "name": name, "reused": was_reused,
                         "seconds": time.monotonic() - t, "cache_bytes": proof["cache_bytes"],
                         "findings": len(proof["keys"]), "finished_at": now()}
                append_update(root / "histories" / f"{worker}.jsonl", event)
                publish(phase)
                if guard is not None and name == smoke[-1]:
                    memory = guard.sample()
                    atomic_json(attempt / "smoke" / f"{worker}.json",
                                {"status": "passed", "names": smoke, "memory": memory, "at": now()})
                    publish("smoke_waiting")
                    while not (attempt / "RELEASE.json").exists():
                        if (attempt / "ABORT.json").exists():
                            raise RuntimeError("Cache group aborted at smoke barrier")
                        time.sleep(.5)
            publish("complete")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if guard is not None and guard.failure:
                error += f"; memory guard: {guard.failure}"
            publish("failed", error)
            atomic_json(attempt / "failures" / f"{worker}.json",
                        {"error": error, "case": name, "phase": phase, "traceback": traceback.format_exc(), "at": now()})
            raise
        finally:
            if guard is not None:
                guard.close()
            del predictor
