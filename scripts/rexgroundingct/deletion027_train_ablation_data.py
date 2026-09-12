"""Complete train/validation verification and immutable train-only loss schedules."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import time

import numpy as np

from exp027_common import REPO, atomic_json, digest, lock, now, read_json, sha256
from exp027_data import (FindingStore, case_key, compatible_contracts,
                        restore_crop_to_original, validate_sources, verify_cache_case)
from deletion027_data import load_event, pool_keys, schedule
from deletion027_ablation_data import (ARMS, LOSSES, immutable_json,
                                      code_manifest as historical_code_manifest)

DEFAULT_CONFIG = REPO / "configs/experiments/027_deletion_loss_ablation_train.json"
MILESTONES = [100, 500, 1000, *range(2000, 10001, 1000)]
REFERENCE_UPDATES = [100, 500, 1000, 2000]
EXCLUDED = ["train_7777_a_2.nii.gz::4"]


def validate_config(config, base):
    wanted = {"arms": list(ARMS), "precision": "fp32", "tf32": False,
              "updates_per_epoch": 100, "total_updates": 10000,
              "evaluation_updates": MILESTONES, "auxiliary_bce_weight": .25,
              "dice_smoothing": 1e-6, "dice_scope": "whole_finding_single_patch_edit",
              "initial_remove_probability": .05, "seed": 20260911, "training_rng_seed": 20261612,
              "training_source": "train", "historical_arm": "run1_train",
              "schedule_sha256": "6975fc98840f2b589d88294fff628923ad01989ea14ea32528368ef074ecfeda",
              "schedule_prefix_sha256": "cfe5c1bc3a450b4a2e4355c97e4ebeac605d9e8dab5a2c5f1e5b56856177e70f",
              "threshold": .9, "display_thresholds": [.5, .9],
              "thresholds": [.5, .8, .9, .95, .99, 1.], "dense_threshold_step": .005,
              "base_thresholds": [0., .1910400390625, .5, 1., 2., 4.], "cache_workers": 4,
              "report_interval_seconds": 5}
    if any(config.get(k) != v for k, v in wanted.items()):
        raise ValueError("Accepted loss-ablation contract changed")
    if len({Path(config[k]).resolve() for k in ("runtime", "historical_runtime", "a_only_runtime")}) != 3:
        raise ValueError("A separate runtime is required")
    if base["model"] != {"channels": 16, "groups": 4, "dilations": [1, 2, 4, 1], "patch_size": [192]*3}:
        raise ValueError("Architecture contract differs")
    if base["optimizer"] != {"lr": 1e-4, "weight_decay": 1e-4, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1}:
        raise ValueError("Optimizer contract differs")


def code_manifest():
    names = ["deletion027_train_ablation_data.py", "deletion027_ablation_loss.py",
             "deletion027_train_ablation_worker.py", "deletion027_train_ablation_report.py",
             "deletion027_train_ablation_comparison.py",
             "deletion027_train_ablation_analysis.py", "run_027_deletion_loss_ablation_train.py",
             "run_027_deletion_loss_ablation_train_host.sh"]
    return {**historical_code_manifest(), **{n: sha256(Path(__file__).parent/n) for n in names}}


def validate_membership(prepared):
    rows = prepared["val"]
    if len(rows) != 69 or len({r["key"] for r in rows}) != 69 or len({r["name"] for r in rows}) != 63:
        raise ValueError("Validation finding/CT coverage differs")
    a, b = ([r for r in rows if r["half"] == h] for h in ("A", "B"))
    if len(a) != 35 or len(b) != 34 or {r["patient"] for r in a} & {r["patient"] for r in b}:
        raise ValueError("A/B patient membership differs")
    if {r["name"] for r in a} & {r["name"] for r in b}:
        raise ValueError("A/B CT overlap")
    train = prepared["train"]
    if (len(train) != 1120 or len({r["key"] for r in train}) != 1120
            or len({r["name"] for r in train}) != 801 or len({r["patient"] for r in train}) != 741):
        raise ValueError("Training finding/CT/patient coverage differs")
    for field in ("key", "name", "patient"):
        if {r[field] for r in train} & {r[field] for r in rows}:
            raise ValueError(f"Training/validation {field} overlap")
    if any(r["split"] != "train" for r in train):
        raise ValueError("Non-training record in training pool")


def create_context(config, root):
    base = read_json(REPO/config["base_config"])
    validate_config(config, base)
    prepared_path = Path(base["experiment_dir"])/"prepared.json"
    prepared = read_json(prepared_path)
    validate_membership(prepared)
    validate_sources(base)
    old = Path(config["historical_runtime"])
    prior = read_json(old/"context.json")
    if (prior["sha256"] != digest({k: v for k, v in prior.items() if k != "sha256"})
            or prior["prepared_sha256"] != digest(prepared)
            or prior["base_config"] != base or read_json(old/"prepared.json") != prepared):
        raise ValueError("Historical prepared/context provenance differs")
    saved = read_json(old/"runs/run1_train/schedule.json")
    if (saved["context_sha256"] != prior["sha256"] or saved["sha256"] != digest(saved["events"])
            or saved["sha256"] != config["schedule_prefix_sha256"]):
        raise ValueError("Historical schedule provenance differs")
    references = {}
    for update in REFERENCE_UPDATES:
        for tail in (f"checkpoints/update_{update:07d}.pth", f"evaluations/update_{update:07d}/summary.json"):
            path = old/"runs/run1_train"/tail
            references[str(path)] = sha256(path)
        for arm in ARMS:
            path = Path(config["a_only_runtime"])/"runs"/arm/f"evaluations/update_{update:07d}/summary.json"
            references[str(path)] = sha256(path)
    value = {"config": config, "base_config": base, "prepared_sha256": digest(prepared),
             "code": code_manifest(), "historical_context_sha256": prior["sha256"],
             "historical_manifest_sha256": sha256(old/"input_manifest.json"),
             "historical_initial_file_sha256": sha256(old/"initial.pth"),
             "historical_schedule_file_sha256": sha256(old/"runs/run1_train/schedule.json"),
             "reference_files": references}
    value["sha256"] = digest(value)
    immutable_json(root/"context.json", value)
    immutable_json(root/"config.json", config)
    # Preserve the complete metadata object because its digest binds cache contracts.
    # All arrays are verified; only original training findings enter schedules.
    immutable_json(root/"prepared.json", prepared)
    return value


def check_context(config, root):
    context = read_json(root/"context.json")
    if (context["sha256"] != digest({k: v for k, v in context.items() if k != "sha256"})
            or context["config"] != config or context["code"] != code_manifest()
            or digest(read_json(root/"prepared.json")) != context["prepared_sha256"]):
        raise ValueError("Runtime config/code/input provenance differs")
    return context


def prepare_shard(config, root, shard):
    context = check_context(config, root)
    base, prepared = context["base_config"], read_json(root/"prepared.json")
    old = Path(config["historical_runtime"])
    manifest = read_json(old/"input_manifest.json")
    if sha256(old/"input_manifest.json") != context["historical_manifest_sha256"]:
        raise ValueError("Historical index manifest changed")
    sources = {r["name"]: r for r in validate_sources(base)["cases"]}
    records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
    store = FindingStore(Path(base["experiment_dir"]), prepared, base, capacity=1)
    names = sorted({r["name"] for r in records.values()})[shard::config["cache_workers"]]
    contracts = compatible_contracts(base, prepared)
    started, verified_bytes, reused = time.perf_counter(), 0, 0
    with lock(root/f".prepare_{shard}.lock"):
        for done, name in enumerate(names):
            destination = root/"tile_index"/f"{case_key(name)}.json"
            if destination.exists():
                previous = read_json(destination)
                if previous["context_sha256"] != context["sha256"]:
                    raise ValueError("Resumed tile index has a different context")
                verify_proof_files(previous["cache_proof"])
                old_index = old/"tile_index"/destination.name
                if (sha256(old_index) != manifest["index_hashes"][old_index.name]
                        or previous["findings"] != read_json(old_index)["findings"]):
                    raise ValueError("Resumed tile index differs from historical source")
                verified_bytes += previous["cache_proof"]["cache_bytes"]
                reused += 1
                atomic_json(root/f"prepare_progress_{shard}.json",
                            {"done": done+1, "total": len(names), "case": name, "updated_at": now(),
                             "verified_cache_bytes": verified_bytes, "reused_verifications": reused,
                             "elapsed_seconds": time.perf_counter()-started})
                continue
            proof = verify_cache_case(Path(base["experiment_dir"])/"cache"/case_key(name),
                                      name, records, base, contracts, sources)
            old_index = old/"tile_index"/f"{case_key(name)}.json"
            if sha256(old_index) != manifest["index_hashes"][old_index.name]:
                raise ValueError("Historical tile index SHA differs")
            index = read_json(old_index)
            expected = {k for k, r in records.items() if r["name"] == name}
            if (set(index["findings"]) != expected
                    or index["context_sha256"] != context["historical_context_sha256"]):
                raise ValueError("Historical tile index finding context differs")
            for key, item in index["findings"].items():
                _, z, y, _, meta = store.get(key)
                b = z >= 0
                counts = (int((b & (y > 0)).sum()), int((b & (y == 0)).sum()), int(y.sum()))
                if item["record"] != records[key] or counts != (item["base_tp"], item["base_fp"], records[key]["voxels"]):
                    raise ValueError("Whole-finding counts or original identifier differ")
                restored = restore_crop_to_original(y, meta["ct_metadata"])
                if int(restored.sum()) != records[key]["voxels"]:
                    raise ValueError("GT count changed on original geometry restoration")
            immutable_json(destination,
                           {"context_sha256": context["sha256"], "name": name,
                            "findings": index["findings"], "cache_proof": proof})
            verified_bytes += proof["cache_bytes"]
            atomic_json(root/f"prepare_progress_{shard}.json",
                        {"done": done+1, "total": len(names), "case": name, "updated_at": now(),
                         "verified_cache_bytes": verified_bytes, "reused_verifications": reused,
                         "elapsed_seconds": time.perf_counter()-started})


def validate_events(config, prepared, index, saved):
    if digest(saved) != config["schedule_sha256"] or len(saved) != config["total_updates"]:
        raise ValueError("Immutable schedule hash/length differs")
    regenerated = schedule(prepared, index, "run1_train", config["total_updates"], config["seed"])
    eligible = set(pool_keys(prepared, index, "run1_train")["train"])
    if (len(eligible) != 1119 or saved != regenerated or {e["key"] for e in saved} != eligible
            or {e["source"] for e in saved} != {"train"}
            or sorted(r["key"] for r in prepared["train"] if r["key"] not in eligible) != EXCLUDED
            or digest(saved[:2000]) != config["schedule_prefix_sha256"]):
        raise ValueError("Train-only schedule replay/coverage/prefix differs")
    for start in range(0, len(saved), 100):
        if Counter(e["branch"] for e in saved[start:start+100]) != {"tp": 50, "fp": 25, "eligible": 25}:
            raise ValueError("Patch branch mixture differs")


def verify_proof_files(proof):
    for filename, expected in proof["files"].items():
        stat = Path(filename).stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"Verified cache changed: {filename}")


def verify_gate(root, context):
    manifest = read_json(root/"input_manifest.json")
    prepared = read_json(root/"prepared.json")
    if (manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest["context_sha256"] != context["sha256"] or manifest["findings"] != 1189
            or manifest["cases"] != 864
            or set(manifest["index"]) != {r["key"] for r in prepared["train"] + prepared["val"]}
            or manifest.get("bypassed") != EXCLUDED):
        raise ValueError("Complete train/validation input gate missing")
    expected = {r["name"] for r in prepared["train"] + prepared["val"]}
    if (len(manifest["cache_proofs"]) != 864 or {r["name"] for r in manifest["cache_proofs"]} != expected
            or sorted(k for r in manifest["cache_proofs"] for k in r["keys"]) != sorted(manifest["index"])):
        raise ValueError("Complete cache proof coverage gate failed")
    for row in manifest["cache_proofs"]:
        verify_proof_files(row)
    return manifest


def finish_preparation(config, root):
    context = check_context(config, root)
    prepared = read_json(root/"prepared.json")
    index, proofs = {}, []
    for path in sorted((root/"tile_index").glob("*.json")):
        row = read_json(path)
        if row["context_sha256"] != context["sha256"] or index.keys() & row["findings"].keys():
            raise ValueError("Duplicate/incompatible tile index")
        index.update(row["findings"])
        proofs.append(row["cache_proof"])
    if len(proofs) != 864 or set(index) != {r["key"] for r in prepared["train"] + prepared["val"]}:
        raise ValueError("Complete train/validation index gate failed")
    saved = schedule(prepared, index, "run1_train", config["total_updates"], config["seed"])
    validate_events(config, prepared, index, saved)
    old_path = Path(config["historical_runtime"])/"runs/run1_train/schedule.json"
    if (sha256(old_path) != context["historical_schedule_file_sha256"]
            or saved[:2000] != read_json(old_path)["events"]):
        raise ValueError("Historical schedule prefix changed")
    for arm in ARMS:
        immutable_json(root/"runs"/arm/"schedule.json",
                       {"context_sha256": context["sha256"], "events": saved, "sha256": digest(saved)})
    from deletion027_worker import edit_metrics, aggregate
    baseline = []
    for r in prepared["val"]:
        item = index[r["key"]]
        m = edit_metrics(item["base_tp"], item["base_fp"], r["voxels"], 0, 0)
        expected = next(q["dice"] for q in prepared["baseline_findings"] if q["key"] == r["key"])
        if abs(m["dice"]-expected) > 1e-10:
            raise ValueError("Strict cached baseline differs")
        baseline.append({**r, **m})
    summary = aggregate(baseline)
    for half, expected in (("A", .3394159226078028), ("B", .334950148355145), ("full", .3372153961644642)):
        if abs(summary[half]["dice"]-expected) > 1e-10:
            raise ValueError("Baseline A/B recomposition differs")
    manifest = {"context_sha256": context["sha256"], "cases": 864, "findings": 1189,
                "index": index, "cache_proofs": proofs, "baseline_findings": baseline,
                "eligible_counts": {arm: {"train": 1119} for arm in ARMS},
                "bypassed": sorted(k for k, v in index.items() if not v["tiles"]),
                "sampling": {"updates": len(saved), "unique_findings": len({e["key"] for e in saved}),
                             "fallbacks": sum(e["fallback"] for e in saved), "sha256": digest(saved)}}
    manifest["sha256"] = digest(manifest)
    immutable_json(root/"input_manifest.json", manifest)
    verify_gate(root, context)
    return manifest


def initial_state(config, root):
    import torch
    from fit_027_deletion import make_editor
    from deletion027_worker import save_torch, weight_hash
    context = check_context(config, root)
    path = Path(config["historical_runtime"])/"initial.pth"
    if sha256(path) != context["historical_initial_file_sha256"]:
        raise ValueError("Historical initial file changed")
    old = torch.load(path, map_location="cpu", weights_only=False)
    model = make_editor(config, context["base_config"])
    model.load_state_dict(old["model"])
    if (weight_hash(model) != config["initial_weights_sha256"]
            or old["weights_sha256"] != config["initial_weights_sha256"]
            or old["context_sha256"] != context["historical_context_sha256"]
            or torch.count_nonzero(model.head.weight).item() != 0
            or not torch.allclose(model.head.bias.sigmoid(), torch.full_like(model.head.bias, .05))):
        raise ValueError("Pristine initial weights differ")
    value = {"model": old["model"], "weights_sha256": old["weights_sha256"], "context_sha256": context["sha256"]}
    local = root/"initial.pth"
    if not local.exists():
        save_torch(local, value)
    else:
        current = torch.load(local, map_location="cpu", weights_only=False)
        model.load_state_dict(current["model"])
        if current["context_sha256"] != context["sha256"] or weight_hash(model) != value["weights_sha256"]:
            raise ValueError("Local initial state differs")
    return value
