"""Validation-only inputs and immutable historical replay for the loss comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exp027_common import REPO, atomic_json, digest, lock, now, read_json, sha256
from exp027_data import (FindingStore, case_key, compatible_contracts,
                        restore_crop_to_original, validate_sources, verify_cache_case)
from deletion027_data import code_manifest as historical_code_manifest, load_event, schedule

ARMS = ("bce_tp2", "bce_tp1", "dice_bce_tp2", "dice_bce_tp1")
LOSSES = {
    "bce_tp2": {"dice_weight": 0., "fp_weight": 1., "tp_weight": 2., "formula": "F + 2K"},
    "bce_tp1": {"dice_weight": 0., "fp_weight": 1., "tp_weight": 1., "formula": "F + K"},
    "dice_bce_tp2": {"dice_weight": 1., "fp_weight": .25/3, "tp_weight": .5/3,
                     "formula": "D + 0.25(F + 2K)/3"},
    "dice_bce_tp1": {"dice_weight": 1., "fp_weight": .125, "tp_weight": .125,
                     "formula": "D + 0.25(F + K)/2"},
}
DEFAULT_CONFIG = REPO / "configs/experiments/027_deletion_loss_ablation_a.json"


def validate_config(config, base):
    wanted = {"arms": list(ARMS), "precision": "fp32", "tf32": False,
              "updates_per_epoch": 100, "total_updates": 2000,
              "evaluation_updates": [100, 500, 1000, 2000], "auxiliary_bce_weight": .25,
              "dice_smoothing": 1e-6, "dice_scope": "whole_finding_single_patch_edit",
              "initial_remove_probability": .05, "seed": 20260911, "training_rng_seed": 20263014,
              "threshold": .9, "display_thresholds": [.5, .9],
              "thresholds": [.5, .8, .9, .95, .99, 1.], "dense_threshold_step": .005,
              "base_thresholds": [0., .1910400390625, .5, 1., 2., 4.], "cache_workers": 4,
              "report_interval_seconds": 5}
    if any(config.get(k) != v for k, v in wanted.items()):
        raise ValueError("Accepted loss-ablation contract changed")
    if Path(config["runtime"]).resolve() == Path(config["historical_runtime"]).resolve():
        raise ValueError("A separate runtime is required")
    if base["model"] != {"channels": 16, "groups": 4, "dilations": [1, 2, 4, 1], "patch_size": [192]*3}:
        raise ValueError("Architecture contract differs")
    if base["optimizer"] != {"lr": 1e-4, "weight_decay": 1e-4, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1}:
        raise ValueError("Optimizer contract differs")


def code_manifest():
    names = ["deletion027_ablation_data.py", "deletion027_ablation_loss.py",
             "deletion027_ablation_worker.py", "deletion027_ablation_report.py",
             "deletion027_ablation_analysis.py", "run_027_deletion_loss_ablation.py",
             "run_027_deletion_loss_ablation_host.sh"]
    return {**historical_code_manifest(), **{n: sha256(Path(__file__).parent/n) for n in names}}


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact changed: {path}")
    else:
        atomic_json(path, value)


def validate_membership(prepared):
    rows = prepared["val"]
    if len(rows) != 69 or len({r["key"] for r in rows}) != 69 or len({r["name"] for r in rows}) != 63:
        raise ValueError("Validation finding/CT coverage differs")
    a, b = ([r for r in rows if r["half"] == h] for h in ("A", "B"))
    if len(a) != 35 or len(b) != 34 or {r["patient"] for r in a} & {r["patient"] for r in b}:
        raise ValueError("A/B patient membership differs")
    if {r["name"] for r in a} & {r["name"] for r in b}:
        raise ValueError("A/B CT overlap")


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
    saved = read_json(old/"runs/run3_val_a/schedule.json")
    if (saved["context_sha256"] != prior["sha256"] or saved["sha256"] != digest(saved["events"])
            or saved["sha256"] != config["schedule_sha256"]):
        raise ValueError("Historical schedule provenance differs")
    value = {"config": config, "base_config": base, "prepared_sha256": digest(prepared),
             "code": code_manifest(), "historical_context_sha256": prior["sha256"],
             "historical_manifest_sha256": sha256(old/"input_manifest.json"),
             "historical_initial_file_sha256": sha256(old/"initial.pth"),
             "historical_schedule_file_sha256": sha256(old/"runs/run3_val_a/schedule.json")}
    value["sha256"] = digest(value)
    immutable_json(root/"context.json", value)
    immutable_json(root/"config.json", config)
    # Preserve the complete metadata object because its digest binds cache contracts.
    # Only validation arrays are accessed; train metadata never enters schedules.
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
    records = {r["key"]: r for r in prepared["val"]}
    store = FindingStore(Path(base["experiment_dir"]), prepared, base, capacity=1)
    names = sorted({r["name"] for r in records.values()})[shard::config["cache_workers"]]
    contracts = compatible_contracts(base, prepared)
    with lock(root/f".prepare_{shard}.lock"):
        for done, name in enumerate(names):
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
            immutable_json(root/"tile_index"/old_index.name,
                           {"context_sha256": context["sha256"], "name": name,
                            "findings": index["findings"], "cache_proof": proof})
            atomic_json(root/f"prepare_progress_{shard}.json",
                        {"done": done+1, "total": len(names), "case": name, "updated_at": now()})


def validate_events(config, prepared, index, saved):
    if digest(saved) != config["schedule_sha256"] or len(saved) != 2000:
        raise ValueError("Immutable schedule hash/length differs")
    regenerated = schedule({"train": [], "val": prepared["val"]}, index,
                           "run3_val_a", 2000, config["seed"])
    if saved != regenerated or {e["key"] for e in saved} != {r["key"] for r in prepared["val"] if r["half"] == "A"}:
        raise ValueError("A-only schedule replay differs")


def verify_gate(root, context):
    manifest = read_json(root/"input_manifest.json")
    prepared = read_json(root/"prepared.json")
    if (manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest["context_sha256"] != context["sha256"] or manifest["findings"] != 69
            or manifest["cases"] != 63 or set(manifest["index"]) != {r["key"] for r in prepared["val"]}):
        raise ValueError("Complete validation input gate missing")
    for row in manifest["cache_proofs"]:
        for filename, expected in row["files"].items():
            stat = Path(filename).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
                raise ValueError(f"Verified cache changed: {filename}")
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
    if len(proofs) != 63 or set(index) != {r["key"] for r in prepared["val"]}:
        raise ValueError("Complete validation index gate failed")
    saved = read_json(Path(config["historical_runtime"])/"runs/run3_val_a/schedule.json")["events"]
    validate_events(config, prepared, index, saved)
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
    manifest = {"context_sha256": context["sha256"], "cases": 63, "findings": 69,
                "index": index, "cache_proofs": proofs, "baseline_findings": baseline,
                "eligible_counts": {arm: {"A": 35} for arm in ARMS}}
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
