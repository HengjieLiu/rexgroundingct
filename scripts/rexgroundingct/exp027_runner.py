"""Explicit refiner train/evaluation workers, full-state checkpoints and timing."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import os
import platform
import random
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from exp027_common import (EXPOSURE, aggregate, append_update, atomic_csv, atomic_json, context, digest, finding_metrics,
                          gpu_gate, lock, now, read_json, materialize_schedule, reconcile_updates, sha256)
from exp027_data import FindingStore, restore_crop_to_original
from exp027_model import make_model, refiner_loss, refine_volume


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda"]])


def weight_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_optimizer(model, config):
    spec = config["optimizer"]
    return torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"],
                             betas=tuple(spec["betas"]), eps=spec["eps"])


def checkpoint_payload(model, optimizer, scaler, update, history, fingerprint, arm, initial_hash):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "update": update, "sampling_cursor": update, "history": history, "rng": rng_state(),
            "context_sha256": fingerprint, "arm": arm, "initial_weights_sha256": initial_hash}


def restore_checkpoint(state, model, optimizer, scaler, fingerprint, arm):
    if state["context_sha256"] != fingerprint or state["arm"] != arm:
        raise ValueError("Checkpoint config/split/code/arm differs; refusing resume")
    if state["update"] != state["sampling_cursor"]:
        raise ValueError("Checkpoint sampling cursor mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])


def update_step(model, optimizer, scaler, x, y, valid, config, device):
    model.train()
    retry_rng, overflows = rng_state(), []
    optimizer_steps = [0]

    def counted_step(*_):
        optimizer_steps[0] += 1

    # Count real optimizer calls: AMP-skipped attempts must never advance a
    # finding schedule, checkpoint cursor, or the 100-update timing budget.
    with optimizer.register_step_post_hook(counted_step):
        while True:
            optimizer.zero_grad(set_to_none=True)
            scale = scaler.get_scale()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config["training"]["amp"] and device.type == "cuda"):
                residual = model(x)
            loss, parts = refiner_loss(x[:, 1:2], residual, y, valid,
                                      config["loss"]["residual_l1"], config["loss"]["dice_epsilon"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at AMP scale {scale}; no optimizer update")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            named_grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
            if not named_grads:
                raise RuntimeError("No parameter gradients")
            finite = torch.stack([torch.isfinite(g).all() for _, g in named_grads]).tolist()
            bad = [name for (name, _), ok in zip(named_grads, finite) if not ok]
            if bad:
                if not scaler.is_enabled() or scale <= 1 or len(overflows) >= 16:
                    raise FloatingPointError(f"Non-finite gradients at AMP scale {scale} in {bad}; "
                                             f"recovery stopped after {len(overflows)} backoffs")
                # unscale_ already registered the overflow. GradScaler safely
                # skips optimizer.step and halves its scale; do not clip infs.
                scaler.step(optimizer)
                scaler.update()
                if optimizer_steps[0] != 0 or not 0 < scaler.get_scale() < scale:
                    raise RuntimeError("AMP overflow did not skip the optimizer and lower its scale")
                event = {"scale_before": scale, "scale_after": scaler.get_scale(),
                         "parameters": bad, "loss": float(loss.detach())}
                overflows.append(event)
                print(f"AMP overflow: {event}; retrying the same patch without advancing the update", flush=True)
                restore_rng(retry_rng)
                # Release the failed graph/activations before retrying 192³.
                del residual, loss, parts, named_grads
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["grad_clip"],
                                                 error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if optimizer_steps[0] != 1:
                raise RuntimeError("Expected exactly one successful optimizer update")
            return {"loss": float(loss.detach()), "grad_norm": float(norm), "amp_scale": scaler.get_scale(),
                    "amp_overflow_retries": len(overflows), "amp_overflows": overflows,
                    **{k: float(v.detach()) for k, v in parts.items()}}


def warmup(model, optimizer, scaler, config, device, count):
    if count <= 0:
        return []
    state = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "rng": rng_state()})
    try:
        size = config["model"]["patch_size"]
        x = torch.randn((1, 2, *size), device=device)
        y = (torch.rand((1, 1, *size), device=device) > .9).float()
        history = [update_step(model, optimizer, scaler, x, y, torch.ones_like(y), config, device)
                   for _ in range(count)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return history
    finally:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        optimizer.zero_grad(set_to_none=True)
        restore_rng(state["rng"])


def worker_device(index, allowed):
    gpu_gate(allowed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return device


def session_context(config, prepared, session):
    session = Path(session)
    value = context(config, prepared)
    value["software"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "container_image_id": os.environ.get("EXP027_CONTAINER_IMAGE_ID")}
    # All four workers intentionally share this short metadata critical section.
    # Exclusive run/report locks remain fail-fast; sibling context access waits.
    with ExitStack() as stack:
        deadline = time.monotonic() + 30
        while True:
            try:
                stack.enter_context(lock(session / ".context.lock"))
                break
            except RuntimeError as exc:
                if not str(exc).startswith("Another process owns ") or time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        path = session / "context.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("Run context changed: create a new run/session instead of overwriting provenance")
        atomic_json(path, value)
    return digest(value)


def train_worker(config, prepared, root, phase, arm, until, *, allow_gpu=False, device_index=0, resume=True):
    setup_start = time.perf_counter()
    launch_ns = os.environ.get("EXP027_WORKER_LAUNCH_NS")
    startup_seconds = (time.time_ns() - int(launch_ns)) / 1e9 if launch_ns else None
    if not config["training"]["amp"]:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = worker_device(device_index, allow_gpu)
    root = Path(root)
    folder = root / phase / arm
    with lock(folder / ".train.lock"):
        fingerprint = session_context(config, prepared, root / phase)
        atomic_json(folder / "environment.json", {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU fixture",
                    "container_image_id": os.environ.get("EXP027_CONTAINER_IMAGE_ID"), "started_at": now(),
                    "amp": config["training"]["amp"], "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
        seed_all(config["seed"])
        model = make_model(config).to(device)
        initial_hash = weight_hash(model)
        optimizer = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"])
        horizon = config["training"]["total_updates"] if phase == "full" else config["benchmark"]["updates"]
        events = materialize_schedule(folder / "schedule.json", prepared, arm, horizon, config["seed"])
        latest = folder / "checkpoints/latest.json"
        history, update, warm_seconds = [], 0, 0.0
        requested_checkpoint = folder / "checkpoints" / f"update_{until:07d}.pth"
        if requested_checkpoint.exists():
            existing = load_state(requested_checkpoint)
            if existing["context_sha256"] != fingerprint or existing["arm"] != arm:
                raise ValueError("Existing milestone has different provenance")
            return
        if latest.exists():
            if not resume:
                raise ValueError("Existing run; choose a fresh root instead of overwriting it")
            state = load_state(folder / "checkpoints" / read_json(latest)["filename"])
            restore_checkpoint(state, model, optimizer, scaler, fingerprint, arm)
            update, history = state["update"], state["history"]
            if state["initial_weights_sha256"] != initial_hash or update > until:
                raise ValueError("Invalid initial weights or resume horizon")
        else:
            start = time.perf_counter()
            warm_history = warmup(model, optimizer, scaler, config, device,
                                  config["benchmark"]["warmup_updates"] if phase == "benchmark" else 0)
            warm_seconds = time.perf_counter() - start
            atomic_json(folder / "warmup.json", {"synthetic": True, "updates": warm_history,
                        "seconds": warm_seconds, "pristine_state_restored": True,
                        "restored_amp_scale": scaler.get_scale()})
        journal = folder / "training.jsonl"
        reconcile_updates(journal, history)

        def snapshot():
            atomic_json(folder / "training.json", {"updates": history, "initial_weights_sha256": initial_hash})
            atomic_csv(folder / "training.csv", history)

        snapshot()
        store = FindingStore(root, prepared, config)
        stop = {"requested": False}
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for s in old_handlers:
            signal.signal(s, lambda *_: stop.update(requested=True))

        def save():
            filename = f"update_{update:07d}.pth"
            path = folder / "checkpoints" / filename
            if not path.exists():
                torch_save(path, checkpoint_payload(model, optimizer, scaler, update, history, fingerprint, arm, initial_hash))
            atomic_json(latest, {"filename": filename, "update": update})

        save()
        setup_seconds = time.perf_counter() - setup_start - warm_seconds
        torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        started_update = update
        status = "training"
        error = None
        try:
            for event in events[update:until]:
                step_start = time.perf_counter()
                x, y, valid, sample_info = store.patch(event, config["model"]["patch_size"])
                loading = time.perf_counter() - step_start
                tensors = [torch.from_numpy(a)[None].to(device) for a in (x, y, valid)]
                metrics = update_step(model, optimizer, scaler, *tensors, config, device)
                torch.cuda.synchronize(device)
                update = event["update"]
                metrics.update(update=update, source=event["source"], key=event["key"], mode=event["mode"],
                               prediction_fallback=sample_info["fallback"], data_seconds=loading,
                               step_seconds=time.perf_counter() - step_start, lr=optimizer.param_groups[0]["lr"])
                history.append(metrics)
                append_update(journal, metrics)
                atomic_json(folder / "status.json", {"status": "training", "update": update, "until": until,
                            "updated_at": now(), "elapsed_seconds": time.perf_counter() - begin,
                            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)})
                if update % 100 == 0 or update == until or stop["requested"]:
                    save()
                    snapshot()
                if stop["requested"]:
                    status = "interrupted"
                    break
            else:
                status = "barrier_ready"
        except BaseException as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
            measured = time.perf_counter() - begin
            timing = {"measured_wall_seconds": measured, "setup_seconds": setup_seconds,
                      "process_startup_seconds": startup_seconds,
                      "warmup_seconds": warm_seconds, "updates_completed": update,
                      "amp_overflow_retries": sum(r.get("amp_overflow_retries", 0) for r in history),
                      "started_at_update": started_update, "resumed": started_update > 0,
                      "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                      "step_mean_seconds": float(np.mean([r["step_seconds"] for r in history])) if history else None,
                      "step_median_seconds": float(np.median([r["step_seconds"] for r in history])) if history else None,
                      "step_p95_seconds": float(np.percentile([r["step_seconds"] for r in history], 95)) if history else None,
                      "data_seconds": sum(r["data_seconds"] for r in history)}
            atomic_json(folder / f"training_timing_until_{until:07d}.json", timing)
            snapshot()
            atomic_json(folder / "status.json", {"status": status, "update": update, "updated_at": now(),
                                                "error": error, **timing})
        if status == "interrupted":
            raise RuntimeError("Training interrupted after saving full state")


def evaluate_worker(config, prepared, root, phase, arm, update, *, allow_gpu=False, device_index=0):
    begin = time.perf_counter()
    launch_ns = os.environ.get("EXP027_WORKER_LAUNCH_NS")
    startup_seconds = (time.time_ns() - int(launch_ns)) / 1e9 if launch_ns else None
    if not config["training"]["amp"]:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = worker_device(device_index, allow_gpu)
    root = Path(root)
    folder = root / phase / arm
    out = folder / "evaluations" / f"update_{update:07d}"
    checkpoint = folder / "checkpoints" / f"update_{update:07d}.pth"
    with lock(out / ".eval.lock"):
        fingerprint = session_context(config, prepared, root / phase)
        state = load_state(checkpoint)
        if state["context_sha256"] != fingerprint or state["arm"] != arm or state["update"] != update:
            raise ValueError("Evaluation checkpoint context mismatch")
        checkpoint_hash = sha256(checkpoint)
        summary_path = out / "summary.json"
        if summary_path.exists():
            previous = read_json(summary_path)
            if (previous["context_sha256"] != fingerprint or previous["checkpoint_sha256"] != checkpoint_hash
                    or {r["key"] for r in previous["findings"]} != {r["key"] for r in prepared["val"]}
                    or len(previous["findings"]) != len(prepared["val"])):
                raise ValueError("Existing evaluation has invalid provenance or coverage")
            return previous
        model = make_model(config).to(device)
        model.load_state_dict(state["model"])
        store = FindingStore(root, prepared, config, capacity=1)
        rows = []
        inference_seconds = metric_seconds = loading_seconds = output_seconds = 0.0
        resumed = False
        last_progress = 0.0

        def publish_partial():
            atomic_json(out / "partial.json", {
                "status": "provisional", "updated_at": now(), "arm": arm, "update": update,
                "checkpoint_sha256": checkpoint_hash, "context_sha256": fingerprint,
                "findings_done": len(rows), "findings_total": len(prepared["val"]),
                "expected_counts": {h: sum(h == "full" or r["half"] == h for r in prepared["val"])
                                    for h in ("A", "B", "full")},
                "metrics": aggregate(rows), "keys": [r["key"] for r in rows]})
            atomic_json(folder / "status.json", {"status": "evaluating", "update": update,
                        "findings_done": len(rows), "findings_total": len(prepared["val"]), "updated_at": now(),
                        "elapsed_seconds": time.perf_counter() - begin,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)})

        publish_partial()
        torch.cuda.reset_peak_memory_stats(device)
        for i, record in enumerate(prepared["val"]):
            path = out / "findings" / f"{digest(record['key'])}.json"
            if path.exists():
                saved = read_json(path)
                if saved["checkpoint_sha256"] != checkpoint_hash or saved["key"] != record["key"]:
                    raise ValueError("Stale per-finding evaluation")
                rows.append(saved)
                resumed = True
                publish_partial()
                continue
            start = time.perf_counter()
            image, base, target, _, cache_metadata = store.get(record["key"])
            loading_seconds += time.perf_counter() - start

            def progress(done, total):
                nonlocal last_progress
                if time.monotonic() - last_progress < 1 and done != total:
                    return
                last_progress = time.monotonic()
                atomic_json(folder / "status.json", {"status": "evaluating", "update": update,
                            "findings_done": i, "findings_total": len(prepared["val"]),
                            "tiles_done": done, "tiles_total": total, "updated_at": now(),
                            "elapsed_seconds": time.perf_counter() - begin,
                            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)})

            start = time.perf_counter()
            final, residual = refine_volume(model, image, base, config["model"]["patch_size"], device,
                                            overlap=config["evaluation"]["overlap"], amp=config["training"]["amp"], progress=progress)
            torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            start = time.perf_counter()
            metrics = finding_metrics(base, final, target, residual, record["voxels"],
                                      mask_transform=lambda mask: restore_crop_to_original(mask, cache_metadata["ct_metadata"]))
            expected = next(r["dice"] for r in prepared["baseline_findings"] if r["key"] == record["key"])
            if abs(metrics["base_dice"] - expected) > 1e-10:
                raise ValueError(f"Cached baseline mismatch for {record['key']}: {metrics['base_dice']} != {expected}")
            metric_seconds += time.perf_counter() - start
            result = {**record, **metrics, "checkpoint_sha256": checkpoint_hash}
            output_start = time.perf_counter()
            atomic_json(path, result)
            output_seconds += time.perf_counter() - output_start
            rows.append(result)
            publish_partial()
            del final, residual
        output_start = time.perf_counter()
        summary = {"status": "pending_user_review", "arm": arm, "update": update, "exposure": EXPOSURE[arm],
                   "checkpoint_sha256": checkpoint_hash, "context_sha256": fingerprint,
                   "metrics": aggregate(rows), "findings": rows,
                   "timing": {"inference_seconds": inference_seconds, "metric_seconds": metric_seconds,
                              "process_startup_seconds": startup_seconds,
                              "loading_seconds": loading_seconds, "resumed": resumed,
                              "peak_memory_bytes": torch.cuda.max_memory_allocated(device)}}
        atomic_csv(out / "per_finding.csv", rows)
        summary["timing"].update(output_seconds=output_seconds + time.perf_counter() - output_start,
                                 total_seconds=time.perf_counter() - begin)
        atomic_json(out / "summary.json", summary)
        atomic_json(folder / "status.json", {"status": "evaluation_complete", "update": update,
                    "findings_done": len(rows), "findings_total": len(rows), "updated_at": now(), "timing": summary["timing"]})
        return summary
