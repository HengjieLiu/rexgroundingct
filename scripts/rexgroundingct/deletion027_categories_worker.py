"""FP32 category-specific workers using the existing Dice + BCE TP2 editor."""
from __future__ import annotations

import os
import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from exp027_common import (append_update, atomic_csv, atomic_json, atomic_text, digest,
                          lock, now, read_json, reconcile_updates, sha256)
from exp027_data import restore_crop_to_original
from exp027_multicategory_data import FindingStore
from fit_027_deletion import infer_removal, make_editor
from deletion027_worker import (edit_metrics, aggregate as legacy_aggregate, weight_hash,
                                save_torch, rng_state, restore_rng, setup_device, status)
from deletion027_categories_data import (LOSSES, check_context, initial_state, load_event, verify_gate)
from deletion027_ablation_loss import loss_for_arm


def aggregate(rows):
    result = legacy_aggregate(rows)
    for half in ("A", "B", "full"):
        subset = [r for r in rows if half == "full" or r["half"] == half]
        result[half].update(
            hits_gained=sum(r["hit"] > r["base_hit"] for r in subset) if subset else None,
            hits_lost=sum(r["hit"] < r["base_hit"] for r in subset) if subset else None,
            tp_loss=1-result[half]["tp_retention"] if result[half]["tp_retention"] is not None else None)
    return result


def save_checkpoint(folder, model, optimizer, update, history, context, schedule_hash, initial_hash, *, interrupted=False):
    path = folder / "checkpoints" / f"update_{update:07d}.pth"
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
             "update": update, "sampling_cursor": update, "history": history, "precision": "fp32", "tf32": False,
             "context_sha256": context, "schedule_sha256": schedule_hash, "initial_sha256": initial_hash,
             "model_sha256": weight_hash(model), "interrupted": interrupted,
             "arm": folder.name, "category": folder.name, "loss_arm": "dice_bce_tp2", "loss_spec": LOSSES[folder.name]}
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        if (old["context_sha256"], old["schedule_sha256"], old["update"], old["model_sha256"]) != (context, schedule_hash, update, state["model_sha256"]):
            raise ValueError("Immutable checkpoint conflict")
    else:
        save_torch(path, state)
    atomic_json(folder / "latest_checkpoint.json", {"path": str(path), "sha256": sha256(path), "update": update})
    atomic_json(folder / "training.json", {"update": update, "initial_sha256": initial_hash, "history": history})
    atomic_csv(folder / "training.csv", history)


def load_checkpoint(folder, context, schedule_hash):
    pointer = read_json(folder / "latest_checkpoint.json")
    if sha256(pointer["path"]) != pointer["sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    state = torch.load(pointer["path"], map_location="cpu", weights_only=False)
    if (state["context_sha256"] != context or state["schedule_sha256"] != schedule_hash
            or state["sampling_cursor"] != state["update"] or state["precision"] != "fp32" or state["tf32"]):
        raise ValueError("Checkpoint context/cursor/precision mismatch")
    if (state.get("arm") != folder.name or state.get("category") != folder.name
            or state.get("loss_arm") != "dice_bce_tp2" or state.get("loss_spec") != LOSSES[folder.name]):
        raise ValueError("Checkpoint loss identity differs")
    if len(state["history"]) != state["update"]:
        raise ValueError("Checkpoint journal length differs")
    return state


def train(config, root, arm, until, allow_gpu):
    context = check_context(config, root)
    manifest = verify_gate(root, context)
    device = setup_device(allow_gpu)
    base, prepared = context["base_config"], read_json(root / "prepared.json")[arm]
    folder = root / "runs" / arm
    with lock(folder / ".train.lock"):
        fixed = read_json(folder / "schedule.json")
        if (fixed["sha256"] != digest(fixed["events"]) or fixed["context_sha256"] != context["sha256"]
                or fixed["sha256"] != context["schedule_hashes"][arm]
                or len(fixed["events"]) != config["total_updates"]):
            raise ValueError("Invalid schedule provenance")
        if until not in config["evaluation_updates"]:
            raise ValueError("Unscheduled training barrier")
        model = make_editor(config, base).to(device)
        spec = base["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"],
                                     betas=tuple(spec["betas"]), eps=spec["eps"])
        if not (root / "initial.pth").exists():
            raise ValueError("Coordinator has not prepared common initial weights")
        initial = initial_state(config, root)
        history, update = [], 0
        if (folder / "latest_checkpoint.json").exists():
            state = load_checkpoint(folder, context["sha256"], fixed["sha256"])
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            history, update = state["history"], state["update"]
            if state["initial_sha256"] != initial["weights_sha256"]:
                raise ValueError("Initial weights differ")
            reconcile_updates(folder / "training.jsonl", history)
        else:
            model.load_state_dict(initial["model"])
            if weight_hash(model) != initial["weights_sha256"]:
                raise ValueError("Common initial weights failed")
            seed = config["training_rng_seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            save_checkpoint(folder, model, optimizer, 0, [], context["sha256"], fixed["sha256"], initial["weights_sha256"])
        if update > until:
            raise ValueError("Trainer state passed the requested barrier")
        store = FindingStore(Path(config["cache_runtime"]), capacity=2)
        seen_findings = {r["key"] for r in history}
        seen_tiles = {(r["key"], r["tile_index"]) for r in history}
        stopping = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
        started = time.perf_counter()
        try:
            for event in fixed["events"][update:until]:
                if (event["source"] != "train" or event["key"] not in prepared["train_keys"]
                        or manifest["index"][event["key"]]["record"]["category"] != arm):
                    raise ValueError("Non-training finding in train-only worker")
                if stopping[0]:
                    save_checkpoint(folder, model, optimizer, update, history, context["sha256"], fixed["sha256"], initial["weights_sha256"], interrupted=True)
                    raise InterruptedError("Graceful training interruption")
                begin = time.perf_counter()
                arrays = load_event(store, event, base["model"]["patch_size"])
                x, y, valid = [torch.from_numpy(a)[None].to(device) for a in arrays]
                torch.cuda.synchronize()
                loading = time.perf_counter() - begin
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                item = manifest["index"][event["key"]]
                counts = {"base_tp": item["base_tp"], "base_fp": item["base_fp"], "total_gt": item["total_gt"]}
                loss, parts = loss_for_arm(logits, x[:, 1:2], y, valid, counts, config["loss_arm"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite deletion loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite editor parameters")
                with torch.no_grad():
                    b, gt = (x[:, 1:2] >= 0) & (valid > 0), (y > 0) & (valid > 0)
                    deleted = b & (logits.sigmoid() > config["threshold"])
                    metric = edit_metrics(event["tp"], event["fp"], int(gt.sum()), int((deleted & gt).sum()), int((deleted & ~gt).sum()))
                torch.cuda.synchronize()
                update = event["update"]
                row = {**event, "scheduled_tp": event["tp"], "scheduled_fp": event["fp"],
                       "loss_arm": config["loss_arm"], "category": arm, "training_source": "train", "loss": float(loss.detach()),
                       **{k: float(v.detach()) for k, v in parts.items()},
                       **metric, "patch_dice": metric["dice"], "gt_empty": metric["total_gt"] == 0,
                       "grad_norm_before_clip": float(grad), "loading_seconds": loading,
                       "step_seconds": time.perf_counter()-begin, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                       "updated_at": now(), "metric_weight_state": "before_this_optimizer_update"}
                append_update(folder / "training.jsonl", row)
                history.append(row)
                seen_findings.add(event["key"]); seen_tiles.add((event["key"], event["tile_index"]))
                status(folder, "training", update=update, until=until, unique_findings=len(seen_findings),
                       unique_tiles=len(seen_tiles), elapsed_seconds=time.perf_counter()-started,
                       peak_memory_bytes=torch.cuda.max_memory_allocated())
                del x, y, valid, logits, loss, parts, b, gt, deleted
                if update % 100 == 0:
                    save_checkpoint(folder, model, optimizer, update, history, context["sha256"], fixed["sha256"], initial["weights_sha256"])
            status(folder, "waiting_at_evaluation_barrier", update=update, unique_findings=len(seen_findings),
                   unique_tiles=len(seen_tiles), peak_memory_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.perf_counter()-started)
        except BaseException as error:
            atomic_text(folder / f"failure_{time.time_ns()}.txt", traceback.format_exc())
            status(folder, "interrupted" if isinstance(error, InterruptedError) else "failed", update=update, error=repr(error))
            raise


def metric_from_vectors(labels, scores, total_gt, threshold, base=False):
    deleted = scores < threshold if base else scores > threshold
    return edit_metrics(int(labels.sum()), int((~labels).sum()), total_gt,
                        int((deleted & labels).sum()), int((deleted & ~labels).sum()))


def evaluate(config, root, arm, update, allow_gpu):
    context = check_context(config, root)
    base, prepared = context["base_config"], read_json(root / "prepared.json")[arm]
    folder = root / "runs" / arm
    out = folder / "evaluations" / f"update_{update:07d}"
    checkpoint = folder / "checkpoints" / f"update_{update:07d}.pth"
    with lock(out / ".eval.lock"):
        checksum = sha256(checkpoint)
        if (out / "summary.json").exists():
            summary = read_json(out / "summary.json")
            if (summary["checkpoint_sha256"] != checksum or summary["context_sha256"] != context["sha256"]
                    or summary.get("status") != "complete" or summary.get("loss_spec") != LOSSES[arm]
                    or summary["arm"] != arm or len(summary["findings"]) != len(prepared["val"])
                    or {r["key"] for r in summary["findings"]} != {r["key"] for r in prepared["val"]}):
                raise ValueError("Completed evaluation context/coverage mismatch")
            return summary
        verify_gate(root, context)
        device = setup_device(allow_gpu)
        model = make_editor(config, base).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (state["context_sha256"] != context["sha256"] or state["update"] != update
                or state.get("arm") != arm or state.get("loss_spec") != LOSSES[arm]):
            raise ValueError("Evaluation checkpoint mismatch")
        model.load_state_dict(state["model"])
        store = FindingStore(Path(config["cache_runtime"]), capacity=1)
        rows = []
        started = time.perf_counter()
        stopped = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

        def publish(complete=False):
            summary = {"status": "complete" if complete else "provisional", "arm": arm, "update": update,
                       "checkpoint_sha256": checksum, "context_sha256": context["sha256"], "loss_spec": LOSSES[arm],
                       "findings_done": len(rows), "findings_total": len(prepared["val"]),
                       "expected_counts": {h: sum(h == "full" or r["half"] == h for r in prepared["val"]) for h in ("A", "B", "full")},
                       "training_source": "train",
                       "exposure": prepared["exposure"],
                       "threshold": config["threshold"], "findings": rows,
                       "metrics": aggregate([{**r, **r["thresholds"][str(config["threshold"]) ]} for r in rows]),
                       "threshold_metrics": {str(t): aggregate([{**r, **r["thresholds"][str(t)]} for r in rows]) for t in config["thresholds"]},
                       "base_threshold_metrics": {str(t): aggregate([{**r, **r["base_thresholds"][str(t)]} for r in rows]) for t in config["base_thresholds"]},
                       "updated_at": now(), "elapsed_seconds": time.perf_counter()-started,
                       "peak_memory_bytes": torch.cuda.max_memory_allocated()}
            atomic_json(out / ("summary.json" if complete else "partial.json"), summary)
            return summary

        try:
            status(folder, "evaluating", update=update, findings_done=0, findings_total=len(prepared["val"]))
            publish()
            for record in prepared["val"]:
                path = out / "findings" / f"{digest(record['key'])}.json"
                if stopped[0]:
                    raise InterruptedError("Evaluation interrupted between findings")
                if path.exists():
                    result = read_json(path)
                    if (result["checkpoint_sha256"] != checksum or result["key"] != record["key"]
                            or result.get("context_sha256") != context["sha256"]
                            or sha256(out / "scores" / f"{digest(record['key'])}.npz") != result["scores_sha256"]):
                        raise ValueError("Partial finding provenance mismatch")
                    rows.append(result); publish(); continue
                load_started = time.perf_counter()
                image, z, y, _, meta = store.get(record["key"])
                loading_seconds = time.perf_counter()-load_started
                last = [0.]

                def progress(done, total, active):
                    if stopped[0]:
                        raise InterruptedError("Evaluation interrupted during a tile")
                    if time.monotonic()-last[0] >= 1 or done == total:
                        last[0] = time.monotonic()
                        status(folder, "evaluating", update=update, findings_done=len(rows), findings_total=len(prepared["val"]),
                               tiles_done=done, tiles_total=total, active_tiles=active, current_finding=record["key"],
                               elapsed_seconds=time.perf_counter()-started, peak_memory_bytes=torch.cuda.max_memory_allocated())

                begin = time.perf_counter()
                scores, tiling = infer_removal(model, image, z, base["model"]["patch_size"], device, progress)
                inference_seconds = time.perf_counter()-begin
                metric_started = time.perf_counter()
                b = z >= 0
                labels, values = np.asarray(y[b], bool), scores[b]
                original_b = restore_crop_to_original(b, meta["ct_metadata"])
                original_y = restore_crop_to_original(y > 0, meta["ct_metadata"])
                if (int((original_b & original_y).sum()), int((original_b & ~original_y).sum())) != (int(labels.sum()), int((~labels).sum())):
                    raise ValueError("Original/native baseline counts differ")
                thresholds = {str(t): metric_from_vectors(labels, values, record["voxels"], t) for t in config["thresholds"]}
                expected = next(r["dice"] for r in prepared["baseline_findings"] if r["key"] == record["key"])
                if abs(thresholds[str(config["threshold"])]["base_dice"] - expected) > 1e-10:
                    raise ValueError("Strict cache baseline reproduction failed")
                final = b & ~(scores > config["threshold"])
                original_final = restore_crop_to_original(final, meta["ct_metadata"])
                if np.any(original_final & ~original_b):
                    raise ValueError("Editor introduced foreground")
                if int((original_final & original_y).sum()) != thresholds[str(config["threshold"])]["tp"]:
                    raise ValueError("Refined original-geometry counts differ")
                z_values = np.asarray(z[b], np.float32)
                metric_seconds = time.perf_counter()-metric_started
                output_started = time.perf_counter()
                score_path = out / "scores" / f"{digest(record['key'])}.npz"
                score_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = score_path.with_suffix(".tmp")
                with tmp.open("wb") as stream:
                    np.savez(stream, native_indices=np.flatnonzero(b), native_shape=b.shape, scores=values,
                             labels=labels, base_logits=z_values, total_gt=record["voxels"])
                os.replace(tmp, score_path)
                result = {**record, "thresholds": thresholds,
                          "base_thresholds": {str(t): metric_from_vectors(labels, z_values, record["voxels"], t, True) for t in config["base_thresholds"]},
                          "checkpoint_sha256": checksum, "context_sha256": context["sha256"], "scores_sha256": sha256(score_path),
                          "loading_seconds": loading_seconds, "metric_seconds": metric_seconds,
                          "output_seconds": time.perf_counter()-output_started,
                          "inference_seconds": inference_seconds, "tiling": tiling, "metric_geometry": "original_CT"}
                atomic_json(path, result)
                rows.append(result)
                publish()
                del scores, b, original_b, original_y, original_final, final
            primary = [{**{k: r[k] for k in ("key", "name", "half", "prompt")}, **r["thresholds"][str(config["threshold"]) ]} for r in rows]
            atomic_csv(out / "per_finding.csv", primary)
            atomic_csv(out / "retention_flags.csv", [r for r in primary if r["tp_retention"] is not None and r["tp_retention"] < .95])
            summary = publish(True)
            status(folder, "evaluation_complete", update=update, findings_done=len(rows), findings_total=len(rows),
                   elapsed_seconds=time.perf_counter()-started, peak_memory_bytes=torch.cuda.max_memory_allocated())
            return summary
        except BaseException as error:
            atomic_text(out / f"failure_{time.time_ns()}.txt", traceback.format_exc())
            status(folder, "interrupted" if isinstance(error, InterruptedError) else "failed", update=update, error=repr(error))
            raise
