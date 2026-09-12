"""Category membership, verified shared caches, pristine state and fixed schedules."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exp027_common import REPO, atomic_json, digest, now, read_json, require_hash, sha256
from exp027_multicategory_data import (FindingStore, check_context as check_cache_context,
                                      file_proof, immutable_json)
from deletion027_data import load_event
from deletion027_ablation_data import LOSSES as LOSS_RECIPES

ARMS = ("2b", "2c", "2d", "2e")
LOSSES = {arm: LOSS_RECIPES["dice_bce_tp2"] for arm in ARMS}
DEFAULT_CONFIG = REPO / "configs/experiments/027_deletion_categories_bcde_50ep.json"
MILESTONES = [100, 500, 1000, 2000, 3000, 4000, 5000]


def code_manifest():
    names = ("deletion027_categories_data.py", "deletion027_categories_worker.py",
             "deletion027_categories_report.py", "deletion027_categories_analysis.py",
             "run_027_deletion_categories.py", "run_027_deletion_categories_host.sh",
             "deletion027_ablation_loss.py", "deletion027_ablation_data.py", "deletion027_data.py",
             "deletion027_worker.py", "fit_027_deletion.py", "exp027_common.py", "exp027_data.py",
             "exp027_model.py", "exp027_multicategory_data.py", "deletion027_train_ablation_worker.py",
             "deletion027_train_ablation_analysis.py", "deletion027_report.py")
    return {n: sha256(Path(__file__).parent / n) for n in names}


def validate_config(config, base):
    wanted = {"arms": list(ARMS), "loss_arm": "dice_bce_tp2", "precision": "fp32", "tf32": False,
              "total_updates": 5000, "updates_per_epoch": 100, "evaluation_updates": MILESTONES,
              "auxiliary_bce_weight": .25, "dice_smoothing": 1e-6,
              "initial_remove_probability": .05, "training_source": "train",
              "display_thresholds": [.5, .9], "thresholds": [.5, .8, .9, .95, .99, 1.],
              "validation_plot_policy": "complete_category_only", "validation_plot_subsets": ["full"]}
    if any(config.get(k) != v for k, v in wanted.items()):
        raise ValueError("Accepted category training contract differs")
    if base["model"] != {"channels": 16, "groups": 4, "dilations": [1, 2, 4, 1], "patch_size": [192]*3}:
        raise ValueError("Architecture differs")
    if base["optimizer"] != {"lr": 1e-4, "weight_decay": 1e-4, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1}:
        raise ValueError("Optimizer differs")
    if Path(config["runtime"]).resolve() in (Path(config["cache_runtime"]).resolve(), Path(config["historical_runtime"]).resolve()):
        raise ValueError("Separate training runtime required")


def category_prepared(records, index, halves, exclusions):
    if set(exclusions) - {r["key"] for r in records if r["split"] == "train"}:
        raise ValueError("Unknown/non-training exclusion")
    from deletion027_worker import edit_metrics
    result = {}
    for arm in ARMS:
        train_all = [r for r in records if r["category"] == arm and r["split"] == "train"]
        train = [r for r in train_all if r["key"] not in exclusions]
        val = [{**r, "half": halves[r["name"]]} for r in records if r["category"] == arm and r["split"] == "val"]
        keys = [r["key"] for r in train if index[r["key"]]["tiles"]]
        if not keys or not val:
            raise ValueError("Empty category training or validation pool")
        if {r["name"] for r in train} & {r["name"] for r in val}:
            raise ValueError("CT leakage")
        patients = {r["patient"] for r in train if r["key"] in keys}
        if patients & {r["patient"] for r in val}:
            raise ValueError("Training/validation patient overlap remains after exclusions")
        exposure = {h: ("patient_overlap_with_training" if patients & {r["patient"] for r in val if h == "full" or r["half"] == h}
                        else "held_out_development") for h in ("A", "B", "full")}
        if {r["patient"] for r in val if r["half"] == "A"} & {r["patient"] for r in val if r["half"] == "B"}:
            raise ValueError("A/B patient overlap")
        baseline = []
        for r in val:
            item = index[r["key"]]
            baseline.append({**r, **edit_metrics(item["base_tp"], item["base_fp"], item["total_gt"], 0, 0)})
        result[arm] = {"train": train, "val": val, "train_keys": keys, "baseline_findings": baseline,
                       "excluded_empty_base": [r["key"] for r in train_all if not index[r["key"]]["tiles"]],
                       "excluded_overlap": [r["key"] for r in train_all if r["key"] in exclusions], "exposure": exposure}
    return result


def schedule(keys, index, updates, seed):
    if not keys or updates % 100:
        raise ValueError("Empty pool or partial epoch")
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(updates // 100):
        branches = ["tp"]*50 + ["fp"]*25 + ["eligible"]*25
        rng.shuffle(branches)
        for branch in branches:
            key = keys[int(rng.integers(len(keys)))]
            item = index[key]
            candidates = list(range(len(item["tiles"]))) if branch == "eligible" else item[branch + "_indices"]
            fallback = not candidates
            if fallback:
                candidates = list(range(len(item["tiles"])))
            tile = int(rng.choice(candidates))
            events.append({"update": len(events)+1, "source": "train", "key": key, "branch": branch,
                           "fallback": fallback, "tile_index": tile, **item["tiles"][tile]})
    return events


def create_context(config, root):
    root = Path(root)
    base = read_json(REPO / config["base_config"])
    validate_config(config, base)
    cache = Path(config["cache_runtime"])
    cache_config = read_json(cache / "config.json")
    cache_context, source = check_cache_context(cache_config, cache)
    require_hash(cache / "input_manifest.json", config["cache_inventory_sha256"])
    inventory = read_json(cache / "input_manifest.json")
    if (inventory["status"] != "verified" or inventory["context_sha256"] != cache_context["sha256"]
            or (inventory["cases"], inventory["findings"]) != (2730, 5106)):
        raise ValueError("Complete verified multicategory cache required")
    index, files = {}, {}
    for i, proof in enumerate(inventory["case_proofs"]):
        if file_proof(proof["files"]) != proof["files"]:
            raise ValueError(f"Verified cache changed: {proof['name']}")
        files.update(proof["files"])
        folder = cache / "cases" / proof["name"].removesuffix(".nii.gz")
        marker = read_json(folder / "complete.json")
        require_hash(folder / "tile_index.json", marker["hashes"]["tile_index.json"])
        finding = read_json(folder / "tile_index.json")["findings"]
        if index.keys() & finding.keys():
            raise ValueError("Duplicate finding index")
        index.update(finding)
        if i % 50 == 0 or i+1 == len(inventory["case_proofs"]):
            atomic_json(root / "preparation_progress.json", {"verified_cases": i+1, "total_cases": 2730, "at": now()})
    if set(index) != {r["key"] for r in source["records"]}:
        raise ValueError("Index coverage differs")
    prepared = category_prepared(source["records"], index, source["halves"], config["exclude_training_keys"])
    schedules = {}
    for arm in ARMS:
        if (len(prepared[arm]["train_keys"]) != config["expected_eligible_train"][arm]
                or len(prepared[arm]["val"]) != config["expected_val"][arm]):
            raise ValueError(f"Category membership differs: {arm}")
        for row in prepared[arm]["baseline_findings"]:
            if abs(row["base_dice"] - source["source_baseline"][row["key"]]) > 1e-12:
                raise ValueError("Category baseline does not reproduce strict cache")
        schedules[arm] = schedule(prepared[arm]["train_keys"], index, config["total_updates"], config["schedule_seeds"][arm])
    old = Path(config["historical_runtime"])
    historical = read_json(old / "context.json")
    context = {"config": config, "base_config": base, "cache_context_sha256": cache_context["sha256"],
               "prepared_sha256": digest(prepared), "index_sha256": digest(index),
               "input_files_sha256": digest(files), "code": code_manifest(),
               "schedule_hashes": {a: digest(e) for a, e in schedules.items()},
               "historical_context_sha256": historical["sha256"],
               "historical_initial_file_sha256": sha256(old / "initial.pth")}
    context["sha256"] = digest(context)
    immutable_json(root / "config.json", config)
    immutable_json(root / "prepared.json", prepared)
    immutable_json(root / "context.json", context)
    manifest = {"context_sha256": context["sha256"], "cache_inventory_sha256": config["cache_inventory_sha256"],
                "files": files, "index": index,
                "baseline_findings": [r for a in ARMS for r in prepared[a]["baseline_findings"]],
                "eligible_counts": config["expected_eligible_train"], "cases": 2730, "findings": 5106}
    immutable_json(root / "input_manifest.json", manifest)
    for arm in ARMS:
        immutable_json(root / "runs" / arm / "schedule.json", {"context_sha256": context["sha256"],
                       "events": schedules[arm], "sha256": context["schedule_hashes"][arm]})
    return context


def check_context(config, root):
    root = Path(root)
    c = read_json(root / "context.json")
    if (c["config"] != config or c["code"] != code_manifest()
            or c["sha256"] != digest({k: v for k, v in c.items() if k != "sha256"})
            or digest(read_json(root / "prepared.json")) != c["prepared_sha256"]):
        raise ValueError("Training context/config/code changed")
    require_hash(Path(config["cache_runtime"]) / "input_manifest.json", config["cache_inventory_sha256"])
    return c


def verify_gate(root, context):
    manifest = read_json(Path(root) / "input_manifest.json")
    if (manifest["context_sha256"] != context["sha256"]
            or manifest["cache_inventory_sha256"] != context["config"]["cache_inventory_sha256"]
            or digest(manifest["index"]) != context["index_sha256"]
            or digest(manifest["files"]) != context["input_files_sha256"]
            or (manifest["cases"], manifest["findings"]) != (2730, 5106)
            or file_proof(manifest["files"]) != manifest["files"]):
        raise ValueError("Verified input gate failed")
    return manifest


def initial_state(config, root):
    import torch
    from fit_027_deletion import make_editor
    from deletion027_worker import save_torch, weight_hash
    context = check_context(config, root)
    path = Path(config["historical_runtime"]) / "initial.pth"
    require_hash(path, context["historical_initial_file_sha256"])
    old = torch.load(path, map_location="cpu", weights_only=False)
    model = make_editor(config, context["base_config"])
    model.load_state_dict(old["model"])
    if (weight_hash(model) != config["initial_weights_sha256"]
            or old["context_sha256"] != context["historical_context_sha256"]
            or torch.count_nonzero(model.head.weight).item()
            or not torch.allclose(model.head.bias.sigmoid(), torch.full_like(model.head.bias, .05))):
        raise ValueError("Historical pristine initialization differs")
    value = {"model": old["model"], "weights_sha256": config["initial_weights_sha256"], "context_sha256": context["sha256"]}
    local = Path(root) / "initial.pth"
    if local.exists():
        saved = torch.load(local, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        if saved["context_sha256"] != context["sha256"] or weight_hash(model) != value["weights_sha256"]:
            raise ValueError("Local pristine initialization differs")
    else:
        save_torch(local, value)
    return value


def verify_2a_stop(config):
    receipt = read_json(config["stop_receipt"])
    if (receipt.get("status") != "stopped_by_user" or receipt.get("update") != 6000
            or receipt.get("evaluation_count") != 4 or not receipt.get("coordinator_exited")):
        raise ValueError("Verified 2a epoch-60 stop required before GPU launch")
    if len(receipt.get("evaluation_summaries", {})) != 4 or receipt.get("container_state", {}).get("Running", True):
        raise ValueError("Four verified 2a summaries and exited coordinator required")
    for path, checksum in receipt["evaluation_summaries"].items():
        require_hash(path, checksum)
    return receipt
