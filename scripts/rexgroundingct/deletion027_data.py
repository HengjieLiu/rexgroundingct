"""CPU input verification, exact eligible tiles and immutable deletion schedules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp027_common import ARMS, REPO, atomic_json, digest, lock, now, read_json, sha256
from exp027_data import (FindingStore, cache_contract, case_key, extract_patch,
                        require_full_cache, validate_sources)

DEFAULT_CONFIG = REPO / "configs/experiments/027_deletion_four_arm.json"


def inputs(config):
    base = read_json(REPO / config["base_config"])
    prepared = read_json(Path(base["experiment_dir"]) / "prepared.json")
    return base, prepared


def validate_config(config, base):
    if (config["precision"] != "fp32" or config["tf32"] or config["updates_per_epoch"] != 100
            or config["total_updates"] != 2000 or config["evaluation_updates"] != [100, 500, 1000, 2000]
            or config["threshold"] != .9 or config["thresholds"] != [.5, .8, .9, .95, .99, 1.]
            or config["fp_weight"] != 1 or config["tp_weight"] != 2):
        raise ValueError("Accepted deletion stage1 contract changed")
    if base["model"] != {"channels": 16, "groups": 4, "dilations": [1, 2, 4, 1], "patch_size": [192]*3}:
        raise ValueError("Base model feature architecture differs")
    if base["optimizer"] != {"lr": 1e-4, "weight_decay": 1e-4, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1}:
        raise ValueError("Optimizer contract differs")


def code_manifest():
    names = ["deletion027_data.py", "deletion027_worker.py", "deletion027_report.py",
             "run_027_deletion_four.py", "run_027_deletion_four_host.sh", "fit_027_deletion.py",
             "exp027_common.py", "exp027_data.py", "exp027_model.py",
             "run_voxtell_val_inference.py", "voxtell_preprocessed_cache.py", "train_text_conditioned_voxtell.py"]
    return {n: sha256(Path(__file__).parent / n) for n in names}


def verify_receipts(base, prepared):
    """Reuse cryptographic verification only while every covered file is unchanged."""
    old = Path(base["experiment_dir"])
    inventory = require_full_cache(old, prepared, base)
    receipts = sorted(old.glob("cache_verified_gpu*.json"))
    rows = []
    for path in receipts:
        proof = read_json(path)
        if proof["prepared_sha256"] != digest(prepared) or proof["cache_contract"] != cache_contract(base, prepared):
            raise ValueError("Historical receipt source mismatch")
        rows.extend(proof["cases"])
    if len(rows) != inventory["cases"] or sorted(r["name"] for r in rows) != inventory["names"]:
        raise ValueError("Missing/duplicate verified CT coverage")
    if sorted(k for r in rows for k in r["keys"]) != inventory["keys"]:
        raise ValueError("Verified finding coverage mismatch")
    for row in rows:
        for filename, expected in row["files"].items():
            stat = Path(filename).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
                raise ValueError(f"Previously hash-verified cache changed: {filename}")
    return {"cases": len(rows), "findings": len(inventory["keys"]), "cache_contract": inventory["cache_contract"],
            "receipts": {str(p): sha256(p) for p in receipts}, "prepared_sha256": digest(prepared)}


def create_context(config, root):
    base, prepared = inputs(config)
    validate_config(config, base)
    validate_sources(base)
    proof = verify_receipts(base, prepared)
    value = {"config": config, "base_config": base, "prepared_sha256": digest(prepared),
             "cache_proof": proof, "code": code_manifest()}
    value["sha256"] = digest(value)
    path = root / "context.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Immutable runtime context changed; use a separate run")
    atomic_json(path, value)
    atomic_json(root / "config.json", config)
    atomic_json(root / "prepared.json", prepared)
    return value


def check_context(config, root):
    context = read_json(root / "context.json")
    body = {k: v for k, v in context.items() if k != "sha256"}
    if (context["sha256"] != digest(body) or context["config"] != config
            or context["code"] != code_manifest()):
        raise ValueError("Runtime config/code provenance differs")
    return context


def eligible_tiles(base, target, size):
    from exp027_model import tile_starts
    if base.shape != target.shape:
        raise ValueError("Tile target geometry mismatch")
    coordinates = np.column_stack(np.nonzero(base >= 0))
    labels = target[tuple(coordinates.T)] > 0 if len(coordinates) else np.zeros(0, bool)
    tiles = []
    for starts in tile_starts(base.shape, size, .5):
        inside = np.ones(len(coordinates), bool)
        for axis, (start, width) in enumerate(zip(starts, size)):
            inside &= (coordinates[:, axis] >= start) & (coordinates[:, axis] < start + width)
        count = int(inside.sum())
        if count:
            tp = int(labels[inside].sum())
            tiles.append({"starts": list(starts), "tp": tp, "fp": count - tp})
    return {"shape": list(base.shape), "base_tp": int(labels.sum()),
            "base_fp": int(len(labels) - labels.sum()), "tiles": tiles,
            "tp_indices": [i for i, t in enumerate(tiles) if t["tp"]],
            "fp_indices": [i for i, t in enumerate(tiles) if t["fp"]]}


def prepare_shard(config, root, shard):
    context = check_context(config, root)
    base, prepared = context["base_config"], read_json(root / "prepared.json")
    store = FindingStore(Path(base["experiment_dir"]), prepared, base, capacity=1)
    records = prepared["train"] + prepared["val"]
    names = sorted({r["name"] for r in records})[shard::config["cache_workers"]]
    with lock(root / f".prepare_{shard}.lock"):
        for done, name in enumerate(names):
            path = root / "tile_index" / f"{case_key(name)}.json"
            if path.exists():
                saved = read_json(path)
                if saved["context_sha256"] != context["sha256"]:
                    raise ValueError("Tile index context changed")
            else:
                findings = {}
                for record in (r for r in records if r["name"] == name):
                    _, z, y, _, _ = store.get(record["key"])
                    findings[record["key"]] = {"record": record, **eligible_tiles(z, y, base["model"]["patch_size"])}
                atomic_json(path, {"name": name, "context_sha256": context["sha256"], "findings": findings})
            atomic_json(root / f"prepare_progress_{shard}.json", {"done": done + 1, "total": len(names),
                        "case": name, "updated_at": now()})


def pool_keys(prepared, index, arm):
    train = [r["key"] for r in prepared["train"] if index[r["key"]]["tiles"]]
    a = [r["key"] for r in prepared["val"] if r["half"] == "A" and index[r["key"]]["tiles"]]
    all_val = [r["key"] for r in prepared["val"] if index[r["key"]]["tiles"]]
    return {"train": train} if arm == ARMS[0] else {"train": train, "A": a} if arm == ARMS[1] else {"A": a} if arm == ARMS[2] else {"A+B": all_val}


def schedule(prepared, index, arm, total, seed):
    rng = np.random.default_rng(seed + 10007 * (ARMS.index(arm) + 1))
    pools = pool_keys(prepared, index, arm)
    if any(not keys for keys in pools.values()) or total % 100:
        raise ValueError("Empty source pool or incomplete reporting epoch")
    events = []
    for epoch in range(total // 100):
        sources = ["train"]*50 + ["A"]*50 if arm == ARMS[1] else [next(iter(pools))]*100
        branches = ["tp"]*50 + ["fp"]*25 + ["eligible"]*25
        rng.shuffle(sources)
        rng.shuffle(branches)
        for source, branch in zip(sources, branches):
            key = pools[source][int(rng.integers(len(pools[source])))]
            item = index[key]
            candidates = list(range(len(item["tiles"]))) if branch == "eligible" else item[branch + "_indices"]
            fallback = not candidates
            if fallback:
                candidates = list(range(len(item["tiles"])))
            tile_id = int(rng.choice(candidates))
            events.append({"update": len(events) + 1, "key": key, "source": source, "branch": branch,
                           "fallback": fallback, "tile_index": tile_id, **item["tiles"][tile_id]})
    return events


def finish_preparation(config, root):
    context = check_context(config, root)
    prepared = read_json(root / "prepared.json")
    expected = {r["key"] for r in prepared["train"] + prepared["val"]}
    index, hashes = {}, {}
    for path in sorted((root / "tile_index").glob("*.json")):
        row = read_json(path)
        if row["context_sha256"] != context["sha256"] or index.keys() & row["findings"].keys():
            raise ValueError("Duplicate or incompatible tile indexes")
        index.update(row["findings"])
        hashes[path.name] = sha256(path)
    if set(index) != expected:
        raise ValueError("Complete-pool tile gate failed")
    verify_receipts(context["base_config"], prepared)
    for arm in ARMS:
        events = schedule(prepared, index, arm, config["total_updates"], config["seed"])
        value = {"context_sha256": context["sha256"], "events": events, "sha256": digest(events)}
        path = root / "runs" / arm / "schedule.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("Immutable schedule changed")
        atomic_json(path, value)
    baseline = []
    from deletion027_worker import edit_metrics
    for r in prepared["val"]:
        item = index[r["key"]]
        metric = edit_metrics(item["base_tp"], item["base_fp"], r["voxels"], 0, 0)
        expected_dice = next(q["dice"] for q in prepared["baseline_findings"] if q["key"] == r["key"])
        if abs(metric["base_dice"] - expected_dice) > 1e-10:
            raise ValueError("Indexed baseline differs from strict cached baseline")
        baseline.append({**r, **metric})
    manifest = {"context_sha256": context["sha256"], "cases": len(hashes), "findings": len(index),
                "index_hashes": hashes, "bypassed": [k for k, v in index.items() if not v["tiles"]],
                "baseline_findings": baseline, "eligible_counts": {arm: {k: len(v) for k, v in pool_keys(prepared, index, arm).items()} for arm in ARMS}}
    if (root / "input_manifest.json").exists() and read_json(root / "input_manifest.json") != manifest:
        raise ValueError("Verified input manifest/index hashes changed")
    atomic_json(root / "input_manifest.json", manifest)
    return manifest


def load_event(store, event, size):
    image, base, target, _, _ = store.get(event["key"])
    ct, valid = extract_patch(image, event["starts"], size, 0)
    z, _ = extract_patch(base, event["starts"], size, -30)
    y, _ = extract_patch(target, event["starts"], size, 0)
    b = (z >= 0) & (valid > 0)
    if (int((b & (y > 0)).sum()), int((b & (y == 0)).sum())) != (event["tp"], event["fp"]):
        raise ValueError("Scheduled tile counts differ from cached arrays")
    return np.stack([ct, z]).astype(np.float32), y[None].astype(np.float32), valid[None]
