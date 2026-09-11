#!/usr/bin/env python3
"""Bounded FP-only fitting diagnostic; never resumes or launches Exp027 arms."""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exp027_common import (REPO, append_update, atomic_csv, atomic_json, atomic_text,
                           digest, lock, now, read_json, sha256)
from exp027_data import (FindingStore, case_key, compatible_contracts, extract_patch,
                        restore_crop_to_original, save_npy, validate_sources, verify_cache_case)
from exp027_model import ResidualRefiner, gaussian_weights, tile_starts

DEFAULT_CONFIG = REPO / "configs/experiments/027_deletion_diagnostic.json"


def make_editor(config, base_config):
    spec = base_config["model"]
    model = ResidualRefiner(spec["channels"], spec["groups"], spec["dilations"])
    probability = config["initial_remove_probability"]
    torch.nn.init.constant_(model.head.bias, math.log(probability / (1 - probability)))
    return model.float()


def deletion_loss(logits, base, target, valid, fp_weight=1., tp_weight=2.):
    if not (logits.shape == base.shape == target.shape == valid.shape):
        raise ValueError("Mismatched loss shapes")
    foreground = (base >= 0) & (valid > 0)
    fp, tp = foreground & (target == 0), foreground & (target > 0)
    # Independent group means, with a differentiable zero for absent groups.
    zero = logits.float().sum() * 0
    remove = F.softplus(-logits.float()[fp]).mean() if fp.any() else zero
    keep = F.softplus(logits.float()[tp]).mean() if tp.any() else zero
    return fp_weight * remove + tp_weight * keep, {
        "remove_loss": remove, "preserve_loss": keep,
        "fp_voxels": fp.sum(), "tp_voxels": tp.sum()}


def metrics(labels, deleted, total_gt):
    """Vectors contain only base-positive voxels; missing GT is counted as FN."""
    labels, deleted = np.asarray(labels, bool), np.asarray(deleted, bool)
    if labels.shape != deleted.shape:
        raise ValueError("Metric shape mismatch")
    base_tp, base_fp = int(labels.sum()), int((~labels).sum())
    if total_gt < base_tp:
        raise ValueError("GT count smaller than base TP")
    removed_tp, removed_fp = int((deleted & labels).sum()), int((deleted & ~labels).sum())
    tp, fp = base_tp - removed_tp, base_fp - removed_fp
    eps = 1e-6
    dice = (2 * tp + eps) / (tp + fp + total_gt + eps)
    before = (2 * base_tp + eps) / (base_tp + base_fp + total_gt + eps)
    return {"base_tp": base_tp, "base_fp": base_fp, "total_gt": int(total_gt),
            "tp": tp, "fp": fp, "fn": int(total_gt - tp),
            "tp_removed": removed_tp, "fp_removed": removed_fp,
            "tp_retention": tp / base_tp if base_tp else None,
            "fp_removal": removed_fp / base_fp if base_fp else None,
            "deletion_precision": removed_fp / int(deleted.sum()) if deleted.any() else None,
            "base_dice": before, "dice": dice, "delta_dice": dice - before,
            "new_foreground": 0}


def summarize(rows):
    if not rows:
        return {"count": 0, "dice": None, "base_dice": None}
    result = {"count": len(rows)}
    for key in ("dice", "base_dice", "delta_dice"):
        result[key] = math.fsum(r[key] for r in rows) / len(rows)
    for key in ("base_tp", "base_fp", "tp", "fp", "fp_removed", "tp_removed", "total_gt"):
        result[key] = sum(r[key] for r in rows)
    result.update(tp_retention=result["tp"] / result["base_tp"] if result["base_tp"] else None,
                  fp_removal=result["fp_removed"] / result["base_fp"] if result["base_fp"] else None,
                  deletion_precision=result["fp_removed"] / (result["fp_removed"] + result["tp_removed"])
                  if result["fp_removed"] + result["tp_removed"] else None,
                  improved=sum(r["delta_dice"] > 1e-12 for r in rows),
                  worsened=sum(r["delta_dice"] < -1e-12 for r in rows),
                  unchanged=sum(abs(r["delta_dice"]) <= 1e-12 for r in rows))
    return result


def retention_threshold(tp_scores, retention=.99, higher_means_delete=True):
    """Conservative order statistic; strict comparisons preserve tied TP."""
    scores = np.sort(np.asarray(tp_scores))
    if not len(scores) or not 0 < retention <= 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid TP calibration samples")
    allowed = int(math.floor((1 - retention) * len(scores) + 1e-9))
    return float(scores[len(scores) - allowed - 1] if higher_means_delete else scores[allowed])


def make_schedule(patches, updates, seed):
    order = np.random.default_rng(seed).permutation(patches).tolist()
    return [order[i % patches] for i in range(updates)]


def source_manifest(config_path, base_config, prepared_path):
    scripts = ("fit_027_deletion.py", "exp027_model.py", "exp027_data.py", "exp027_common.py",
               "run_voxtell_val_inference.py", "voxtell_preprocessed_cache.py",
               "train_text_conditioned_voxtell.py")
    return {"config_sha256": sha256(config_path), "base_config_sha256": digest(base_config),
            "prepared_sha256": sha256(prepared_path),
            "code": {name: sha256(Path(__file__).parent / name) for name in scripts}}


def write_status(root, phase, **kwargs):
    value = {"phase": phase, "updated_at": now(), **kwargs}
    atomic_json(root / "status.json", value)
    print(value, flush=True)


def prepare(config, config_path):
    root = Path(config["runtime"])
    base_config = read_json(REPO / config["base_config"])
    full = Path(base_config["experiment_dir"])
    prepared_path = full / "prepared.json"
    prepared = read_json(prepared_path)
    provenance = source_manifest(config_path, base_config, prepared_path)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["provenance"] != provenance:
            raise ValueError("Diagnostic inputs/code changed; use a separate runtime")
        for patch in previous["patches"]:
            for filename, checksum in patch["hashes"].items():
                if sha256(root / "patches" / patch["id"] / filename) != checksum:
                    raise ValueError("Prepared patch integrity failure")
        return previous, base_config, prepared
    write_status(root, "preparing")
    started = time.perf_counter()
    validate_sources(base_config)
    reference_path = full / "full/run1_train/evaluations/update_0005000/summary.json"
    # This historical result supplies reconstructed FROZEN-BASE counts only.
    reference = {r["key"]: r for r in read_json(reference_path)["findings"] if r["half"] == "A"}
    candidates = []
    patients = set()
    for record in sorted(prepared["val"], key=lambda r: r["key"]):
        if record["half"] != "A" or record["patient"] in patients:
            continue
        old = reference[record["key"]]
        tp = old["tp"] - old["fn_recovered"] + old["tp_removed"]
        fp = old["fp"] + old["fp_removed"] - old["tn_added"]
        if tp > 0 and fp > 0:
            candidates.append({**record, "base_tp": tp, "base_fp": fp, "base_dice": old["base_dice"]})
            patients.add(record["patient"])
    candidates.sort(key=lambda r: (r["base_dice"], r["key"]))
    if len(candidates) < config["findings"]:
        raise ValueError("Insufficient distinct eligible A patients")
    indices = np.rint(np.linspace(0, len(candidates) - 1, config["findings"])).astype(int)
    selected = [candidates[i] for i in indices]
    records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
    contracts = compatible_contracts(base_config, prepared)
    val_manifest = read_json(Path(base_config["base"]["validation_cache"]) / "export_manifest.json")
    sources = {r["name"]: r for r in val_manifest["cases"]}
    store = FindingStore(full, prepared, base_config, capacity=1)
    size = base_config["model"]["patch_size"]
    patches, receipts = [], []
    for record in selected:
        receipt = verify_cache_case(full / "cache" / case_key(record["name"]), record["name"],
                                    records, base_config, contracts, sources)
        receipts.append(receipt)
        image, base, target, _, meta = store.get(record["key"])
        b = base >= 0
        actual = metrics(target[b], np.zeros(int(b.sum()), bool), record["voxels"])
        if (actual["base_tp"], actual["base_fp"]) != (record["base_tp"], record["base_fp"]):
            raise ValueError("Reconstructed baseline counts differ from verified arrays")
        if abs(actual["base_dice"] - record["base_dice"]) > 1e-10:
            raise ValueError("Baseline Dice mismatch")
        eligible = []
        for starts in tile_starts(base.shape, size):
            slices = tuple(slice(max(0, s), min(n, s + p)) for s, n, p in zip(starts, base.shape, size))
            pred, gt = b[slices], target[slices] > 0
            tp, fp = int((pred & gt).sum()), int((pred & ~gt).sum())
            if tp + fp:
                eligible.append({"starts": list(starts), "tp": tp, "fp": fp, "gt": int(gt.sum())})
        if len(eligible) < config["patches_per_finding"]:
            raise ValueError("Too few distinct eligible grid windows")
        choices = []
        for role in ("fp_focus", "tp_focus", "mixed"):
            available = [t for t in eligible if t not in choices]
            if role == "fp_focus":
                empty = [t for t in available if t["gt"] == 0]
                tile = max(empty or available, key=lambda t: t["fp"])
            elif role == "tp_focus":
                tile = max(available, key=lambda t: t["tp"])
            else:
                tile = max(available, key=lambda t: (min(t["tp"], t["fp"]), t["fp"]))
            choices.append(tile)
            folder = root / "patches" / f"p{len(patches):02d}"
            ct, valid = extract_patch(image, tile["starts"], size, 0)
            z, _ = extract_patch(base, tile["starts"], size, -30)
            y, _ = extract_patch(target, tile["starts"], size, 0)
            save_npy(folder / "x.npy", np.stack([ct, z]).astype(np.float32))
            save_npy(folder / "y.npy", y.astype(np.uint8)[None])
            save_npy(folder / "valid.npy", valid.astype(np.uint8)[None])
            patches.append({"id": folder.name, "key": record["key"], "role": role, **tile,
                            "hashes": {n: sha256(folder / n) for n in ("x.npy", "y.npy", "valid.npy")}})
        write_status(root, "preparing", findings_done=len(receipts), findings_total=len(selected))
    schedule = make_schedule(len(patches), config["updates"], config["seed"])
    manifest = {"config": config, "base_config": base_config, "provenance": provenance,
                "selection": "eight evenly spaced baseline-Dice ranks; one finding per A patient; base TP and FP required",
                "reference_path": str(reference_path), "reference_sha256": sha256(reference_path),
                "findings": selected, "patches": patches, "schedule": schedule,
                "schedule_sha256": digest(schedule), "cache_verification": receipts,
                "preparation_seconds": time.perf_counter() - started, "created_at": now()}
    atomic_json(manifest_path, manifest)
    atomic_json(root / "config.json", config)
    atomic_json(root / "base_config.json", base_config)
    atomic_json(root / "environment.json", {"torch": torch.__version__, "numpy": np.__version__,
                "image": os.environ.get("DIAGNOSTIC_IMAGE_ID"),
                "repo_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()})
    write_status(root, "prepared", patches=len(patches), findings=len(selected))
    return manifest, base_config, prepared


def load_patch(root, patch, device):
    folder = root / "patches" / patch["id"]
    return [torch.from_numpy(np.load(folder / n, allow_pickle=False))[None].to(device)
            for n in ("x.npy", "y.npy", "valid.npy")]


def checkpoint(root, model, optimizer, update, manifest, tag=None):
    path = root / "checkpoints" / (tag or f"update_{update:04d}.pth")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Refusing to overwrite checkpoint {path}")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
               "sampling_cursor": update, "manifest_sha256": digest(manifest), "rng_python": random.getstate(),
               "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "precision": "fp32", "tf32": False}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sha256(path)


def curve_rows(vectors, thresholds):
    return [{"threshold": threshold, **summarize([
        metrics(v["labels"], v["scores"] > threshold, v["total_gt"]) for v in vectors])}
            for threshold in thresholds]


@torch.no_grad()
def evaluate_patches(root, model, manifest, device, update, config):
    model.eval()
    rows, vectors = [], []
    out = root / "patch_evaluations" / f"update_{update:04d}"
    out.mkdir(parents=True, exist_ok=True)
    for patch in manifest["patches"]:
        x, y, valid = load_patch(root, patch, device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite patch inference")
        loss, parts = deletion_loss(logits, x[:, 1:2], y, valid, config["fp_weight"], config["tp_weight"])
        b = (x[:, 1:2] >= 0) & (valid > 0)
        scores = logits.sigmoid()[b].cpu().numpy()
        labels, z = y[b].cpu().numpy().astype(bool), x[:, 1:2][b].cpu().numpy()
        gt = int(((y > 0) & (valid > 0)).sum())
        row = {"patch": patch["id"], "key": patch["key"], "role": patch["role"],
               "loss": float(loss), **metrics(labels, scores > config["remove_threshold"], gt)}
        rows.append(row)
        vectors.append({"labels": labels, "scores": scores, "base_logits": z, "total_gt": gt})
        np.savez(out / f"{patch['id']}.npz", labels=labels, scores=scores, base_logits=z, total_gt=gt)
    summary = {"update": update, "findings_exposure": "fitting_A", "threshold": config["remove_threshold"],
               "metrics": summarize(rows), "mean_loss": float(np.mean([r["loss"] for r in rows])),
               "patches": rows, "curve": curve_rows(vectors, config["thresholds"]), "updated_at": now()}
    atomic_json(out / "summary.json", summary)
    atomic_csv(out / "patches.csv", rows)
    atomic_csv(out / "curve.csv", summary["curve"])
    return vectors, summary


@torch.no_grad()
def infer_removal(model, image, base, size, device, progress=None):
    if image.shape != base.shape or len(base.shape) != 3:
        raise ValueError("Inference geometry mismatch")
    model.eval()
    summed, denominator = np.zeros(base.shape, np.float32), np.zeros(base.shape, np.float32)
    weights = gaussian_weights(size)
    starts_list = tile_starts(base.shape, size, .5)
    active = 0
    for i, starts in enumerate(starts_list):
        dest = tuple(slice(max(0, s), min(n, s + p)) for s, n, p in zip(starts, base.shape, size))
        src = tuple(slice(max(0, -s), min(n, s + p) - s) for s, n, p in zip(starts, base.shape, size))
        denominator[dest] += weights[src]
        if np.any(base[dest] >= 0):
            ct, _ = extract_patch(image, starts, size, 0)
            z, _ = extract_patch(base, starts, size, -30)
            x = torch.from_numpy(np.stack([ct, z]).astype(np.float32))[None].to(device)
            value = model(x).sigmoid()[0, 0].float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Non-finite full-volume removal scores")
            summed[dest] += value[src] * weights[src]
            active += 1
        if progress:
            progress(i + 1, len(starts_list), active)
    if np.any(denominator <= 0):
        raise ValueError("Uncovered inference voxels")
    summed /= denominator
    return summed, {"tiles": len(starts_list), "active_tiles": active}


def diagnostic_panel(root, record, image, base, target, scores, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b, y = base >= 0, target > 0
    removed = b & (scores > threshold)
    helpful, harmful = removed & ~y, removed & y
    # One slice with most edits, one with most GT; disclose the selection.
    slices = [int(np.argmax(removed.sum(axis=(1, 2)))), int(np.argmax(y.sum(axis=(1, 2))))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, z in enumerate(slices):
        for col, ax in enumerate(axes[row]):
            ax.imshow(image[z], cmap="gray", vmin=-1.5, vmax=2)
            overlay = np.zeros((*image.shape[1:], 4), np.float32)
            if col == 0:
                overlay[y[z]] = (0, 1, 0, .65)
            elif col == 1:
                overlay[b[z]] = (1, .7, 0, .65)
            else:
                overlay[helpful[z]] = (0, 1, 1, .85)
                overlay[harmful[z]] = (1, 0, 0, .9)
            ax.imshow(overlay)
            ax.set_title(("GT green", "Base amber", "Deleted: FP cyan / TP red")[col] + f"; z={z}")
            ax.axis("off")
    fig.suptitle(f"TRAINING A diagnostic: {record['key']}\n{record['prompt']}", fontsize=9)
    fig.tight_layout()
    path = root / "panels" / f"{digest(record['key'])[:12]}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(root))


def evaluate_volumes(root, model, manifest, prepared, base_config, device, config, operating):
    store = FindingStore(Path(base_config["experiment_dir"]), prepared, base_config, capacity=1)
    rows, vectors = [], []
    start = time.perf_counter()
    for index, record in enumerate(manifest["findings"]):
        image, base, target, _, meta = store.get(record["key"])
        last = [0.]

        def progress(done, total, active):
            if time.monotonic() - last[0] >= 5 or done == total:
                last[0] = time.monotonic()
                write_status(root, "full_volume_evaluation", update=config["updates"],
                             findings_done=index, findings_total=len(manifest["findings"]),
                             tiles_done=done, tiles_total=total, active_tiles=active)

        begin = time.perf_counter()
        scores, tiling = infer_removal(model, image, base, base_config["model"]["patch_size"], device, progress)
        inference_seconds = time.perf_counter() - begin
        b = base >= 0
        original_b = restore_crop_to_original(b, meta["ct_metadata"])
        original_y = restore_crop_to_original(target > 0, meta["ct_metadata"])
        check = metrics(original_y[original_b], np.zeros(int(original_b.sum()), bool), record["voxels"])
        if abs(check["base_dice"] - record["base_dice"]) > 1e-10:
            raise ValueError("Original-geometry baseline mismatch")
        vector = {"labels": np.asarray(target[b], bool), "scores": scores[b],
                  "base_logits": np.asarray(base[b], np.float32), "total_gt": record["voxels"]}
        fixed = metrics(vector["labels"], vector["scores"] > config["remove_threshold"], record["voxels"])
        calibrated = metrics(vector["labels"], vector["scores"] > operating["remove_threshold"], record["voxels"])
        simple = metrics(vector["labels"], vector["base_logits"] < operating["base_logit_threshold"], record["voxels"])
        final = b & ~(scores > config["remove_threshold"])
        original_final = restore_crop_to_original(final, meta["ct_metadata"])
        if np.any(original_final & ~original_b):
            raise AssertionError("Deletion editor introduced foreground")
        original_metrics = metrics(original_y[original_b], ~original_final[original_b], record["voxels"])
        if original_metrics != fixed:
            raise ValueError("Native/original metric mismatch")
        panel = diagnostic_panel(root, record, image, base, target, scores, config["remove_threshold"])
        out = root / "full_volume" / digest(record["key"])
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "base_positive_scores.npz", **vector, native_indices=np.flatnonzero(b), native_shape=b.shape)
        row = {**record, "fixed": fixed, "patch_calibrated": calibrated, "simple_threshold": simple,
               "tiling": tiling, "inference_seconds": inference_seconds, "panel": panel,
               "metric_geometry": "original_CT", "exposure": "fitting_A"}
        atomic_json(out / "result.json", row)
        rows.append(row)
        vectors.append(vector)
        summary = {"status": "provisional" if len(rows) < len(manifest["findings"]) else "pending_user_review",
                   "findings_done": len(rows), "findings_total": len(manifest["findings"]),
                   "findings": rows, "operating_points": operating,
                   "fixed": summarize([r["fixed"] for r in rows]),
                   "patch_calibrated": summarize([r["patch_calibrated"] for r in rows]),
                   "simple_threshold": summarize([r["simple_threshold"] for r in rows]),
                   "curve": curve_rows(vectors, config["thresholds"]),
                   "elapsed_seconds": time.perf_counter() - start, "updated_at": now()}
        atomic_json(root / "full_volume" / "summary.json", summary)
        report(root)
        del scores, b, final, original_b, original_y, original_final
    atomic_csv(root / "full_volume" / "per_finding.csv", [
        {"key": r["key"], "policy": policy, **r[policy]} for r in rows
        for policy in ("fixed", "patch_calibrated", "simple_threshold")])
    return summary


def report(root):
    """Single diagnostic worker owns atomic report updates; no shared-arm writer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    status = read_json(root / "status.json")
    history_path = root / "training.jsonl"
    history = [__import__("json").loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    evaluations = [read_json(p) for p in sorted((root / "patch_evaluations").glob("*/summary.json"))]
    full_path = root / "full_volume" / "summary.json"
    full = read_json(full_path) if full_path.exists() else None
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    if history:
        axes[0, 0].plot([r["update"] for r in history], [r["loss"] for r in history], alpha=.6, label="Training patch loss")
    for ax, field, title in zip(axes.flat[1:5], ("dice", "tp_retention", "fp_removal", "deletion_precision"),
                                ("Fixed-patch Dice", "TP retention", "FP removal", "Deletion precision")):
        xs = [r["update"] for r in evaluations]
        ys = [r["metrics"].get(field) for r in evaluations]
        ax.plot(xs, [np.nan if y is None else y for y in ys], marker="o", label="All 24 fitting patches; threshold 0.5")
        ax.set_title(title)
        if field == "dice" and evaluations:
            ax.axhline(evaluations[0]["metrics"]["base_dice"], color="gray", linestyle="--", label="Base")
    axes[0, 0].set_title("Group-normalized deletion loss")
    if evaluations:
        curve = evaluations[-1]["curve"]
        axes[1, 2].plot([r["tp_retention"] for r in curve], [r["fp_removal"] for r in curve], "o-", label="Fitting patches")
    if full:
        axes[1, 2].plot([r["tp_retention"] for r in full["curve"]], [r["fp_removal"] for r in full["curve"]], "s-", label="Full training volumes")
    axes[1, 2].set_title("Removal/retention tradeoff")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel("Optimizer update")
        if ax.lines:
            ax.legend(fontsize=7)
    axes[1, 2].set_xlabel("TP retention")
    fig.suptitle("DELETION DIAGNOSTIC — selected A training data; no held-out results")
    fig.tight_layout()
    tmp = root / ".curves.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, root / "curves.png")
    text = ["# Deletion-only fitting diagnostic", "", "All results are in-sample, selected half-A diagnostics. B is excluded.",
            "", f"Last refreshed: {now()}; viewers may need to reload.", "", f"Phase: `{status['phase']}`", "",
            f"Progress: `{status}`", "", "![Curves](curves.png)", "",
            "| Update | Mean fixed-patch loss | Dice | Base Dice | TP retained | FP removed |", "| --- | --- | --- | --- | --- | --- |"]
    for row in evaluations:
        m = row["metrics"]
        text.append(f"| {row['update']} | {row['mean_loss']:.5f} | {m['dice']:.5f} | {m['base_dice']:.5f} | {m['tp_retention']:.3%} | {m['fp_removal']:.3%} |")
    if full:
        text += ["", f"Full-volume evaluation: {full['findings_done']}/{full['findings_total']} — {full['status']}", "",
                 "| Policy | Dice | Base Dice | TP retained | FP removed | Improved / unchanged / worsened |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for name in ("fixed", "patch_calibrated", "simple_threshold"):
            m = full[name]
            text.append(f"| {name} | {m['dice']:.5f} | {m['base_dice']:.5f} | {m['tp_retention']:.3%} | {m['fp_removal']:.3%} | {m['improved']} / {m['unchanged']} / {m['worsened']} |")
        text += ["", "Patch-calibrated operating points: `" + str(full["operating_points"]) + "`", ""]
        text += [f"- [{r['key']}]({r['panel']})" for r in full["findings"]]
    atomic_text(root / "report.md", "\n".join(text) + "\n")


def run(config, config_path, allow_gpu):
    if not allow_gpu or os.environ.get("START_GPU_WORK") != "1":
        raise ValueError("GPU execution requires --allow-gpu and START_GPU_WORK=1")
    if config["precision"] != "fp32" or config["tf32"]:
        raise ValueError("Diagnostic requires FP32 with TF32 disabled")
    root = Path(config["runtime"])
    with lock(root / ".owner.lock"):
        if (root / "training.jsonl").exists() or (root / "checkpoints").exists():
            raise ValueError("Existing diagnostic training evidence; no automatic overwrite or restart")
        manifest, base_config, prepared = prepare(config, config_path)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        device = torch.device("cuda:0")
        atomic_json(root / "run_manifest.json", {"started_at": now(), "argv": __import__("sys").argv,
                    "gpu": torch.cuda.get_device_name(device), "precision": "fp32", "tf32": False,
                    "cuda_runtime": torch.version.cuda, "manifest_sha256": digest(manifest),
                    "source": manifest["provenance"], "image": os.environ.get("DIAGNOSTIC_IMAGE_ID")})
        model = make_editor(config, base_config).to(device)
        spec = base_config["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"],
                                     betas=tuple(spec["betas"]), eps=spec["eps"])
        update = 0
        start = time.perf_counter()
        train_seconds = 0.
        interrupted = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
        try:
            checkpoint(root, model, optimizer, 0, manifest)
            write_status(root, "patch_evaluation", update=0)
            evaluate_patches(root, model, manifest, device, 0, config)
            report(root)
            for cursor, patch_index in enumerate(manifest["schedule"]):
                if interrupted[0]:
                    raise InterruptedError("Graceful interruption requested")
                model.train()
                begin = time.perf_counter()
                x, y, valid = load_patch(root, manifest["patches"][patch_index], device)
                torch.cuda.synchronize()
                loaded = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss, parts = deletion_loss(logits, x[:, 1:2], y, valid, config["fp_weight"], config["tp_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite deletion loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite editor parameters")
                torch.cuda.synchronize()
                update = cursor + 1
                elapsed = time.perf_counter() - begin
                train_seconds += elapsed
                row = {"update": update, "patch": manifest["patches"][patch_index]["id"], "loss": float(loss),
                       **{k: float(v) for k, v in parts.items()}, "grad_norm_before_clip": float(grad),
                       "step_seconds": elapsed, "loading_seconds": loaded - begin,
                       "peak_memory_bytes": torch.cuda.max_memory_allocated(), "updated_at": now()}
                append_update(root / "training.jsonl", row)
                del x, y, valid, logits, loss, parts
                if update % 10 == 0:
                    write_status(root, "training", update=update, total_updates=config["updates"],
                                 elapsed_seconds=time.perf_counter() - start, latest_loss=row["loss"])
                    report(root)
                if update in config["checkpoints"]:
                    checkpoint(root, model, optimizer, update, manifest)
                    write_status(root, "patch_evaluation", update=update)
                    vectors, _ = evaluate_patches(root, model, manifest, device, update, config)
                    report(root)
            tp_scores = np.concatenate([v["scores"][v["labels"]] for v in vectors])
            tp_base = np.concatenate([v["base_logits"][v["labels"]] for v in vectors])
            operating = {"target_patch_tp_retention": config["target_tp_retention"],
                         "remove_threshold": retention_threshold(tp_scores, config["target_tp_retention"]),
                         "base_logit_threshold": retention_threshold(tp_base, config["target_tp_retention"], False),
                         "source": "update200 fixed fitting patches; overlapping voxels counted per patch"}
            operating["patch_editor"] = summarize([metrics(v["labels"], v["scores"] > operating["remove_threshold"],
                                                            v["total_gt"]) for v in vectors])
            operating["patch_simple_threshold"] = summarize([metrics(v["labels"], v["base_logits"] < operating["base_logit_threshold"],
                                                                       v["total_gt"]) for v in vectors])
            atomic_json(root / "operating_points.json", operating)
            evaluate_volumes(root, model, manifest, prepared, base_config, device, config, operating)
            atomic_json(root / "timing.json", {"training_seconds": train_seconds,
                        "training_plus_evaluation_seconds": time.perf_counter() - start,
                        "preparation_seconds": manifest["preparation_seconds"],
                        "peak_memory_bytes": torch.cuda.max_memory_allocated()})
            write_status(root, "pending_user_review", update=update, total_updates=config["updates"],
                         elapsed_seconds=time.perf_counter() - start)
            report(root)
        except BaseException as error:
            checkpoint(root, model, optimizer, update, manifest, f"failure_after_{update:04d}.pth")
            atomic_text(root / "failure.txt", traceback.format_exc())
            write_status(root, "interrupted" if isinstance(error, InterruptedError) else "failed",
                         update=update, error=repr(error))
            report(root)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config["runtime"])
    if args.command == "run":
        run(config, args.config, args.allow_gpu)
    elif args.command == "prepare":
        with lock(root / ".owner.lock"):
            prepare(config, args.config)
    else:
        report(root)


if __name__ == "__main__":
    main()
