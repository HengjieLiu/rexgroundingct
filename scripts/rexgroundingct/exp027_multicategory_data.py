"""Independent, cache-only inputs for 2b–2e; historical producers stay untouched."""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from exp027_common import (REPO, atomic_csv, atomic_json, digest, lock, now,
                          patient_id, read_json, require_hash, sha256)
from exp027_data import (cache_producer_fingerprint, case_key, load_validation_array,
                         restore_crop_to_original, validate_sources, verify_cache_case)

DEFAULT_CONFIG = REPO / "configs/experiments/027_voxtell_2bcde_cache.json"
CATEGORIES = ("2b", "2c", "2d", "2e")


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable input changed: {path}")
    else:
        atomic_json(path, value)


def code_manifest():
    names = ("exp027_multicategory_data.py", "exp027_multicategory_worker.py",
             "exp027_multicategory_report.py", "run_027_multicategory_cache.py",
             "run_027_multicategory_cache_host.sh", "exp027_common.py", "exp027_data.py",
             "exp027_model.py")
    return {**cache_producer_fingerprint(),
            **{name: sha256(Path(__file__).parent / name) for name in names},
            "voxtell_predictor": sha256(REPO / "external/VoxTell/voxtell/inference/predictor.py")}


def cache_contract(config):
    # A/B membership, runtime location and future loss/schedule choices are views.
    return digest({"version": "native_multicategory_v1", "base": config["base"],
                   "preprocessing": config["preprocessing"], "metadata": config["metadata_sha256"],
                   "categories": config["categories"], "producer": cache_producer_fingerprint(),
                   "predictor": sha256(REPO / "external/VoxTell/voxtell/inference/predictor.py")})


def finding_records(metadata, categories):
    records = []
    for split in ("train", "val"):
        for row in sorted(metadata[split], key=lambda r: r["name"]):
            for channel, fid in enumerate(sorted(row["findings"], key=int)):
                if row["categories"][fid] in categories:
                    records.append({"name": row["name"], "key": f"{row['name']}::{fid}",
                                    "finding_id": fid, "channel": channel,
                                    "prompt": row["findings"][fid], "category": row["categories"][fid],
                                    "patient": patient_id(row["name"]), "split": split,
                                    "voxels": int(row["pixels"][fid]),
                                    "entities": int(row["entity_counts"][fid])})
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate finding identifiers")
    return records


def census(records, halves, categories=CATEGORIES):
    result = {}
    for category in categories:
        selected = [r for r in records if r["category"] == category]
        row = {}
        for split in ("train", "val"):
            items = [r for r in selected if r["split"] == split]
            row[split + "_cases"] = len({r["name"] for r in items})
            row[split + "_findings"] = len(items)
        for half in ("A", "B"):
            row[half] = sum(r["split"] == "val" and halves[r["name"]] == half for r in selected)
        result[category] = row
    return result


def validate_membership(records, halves, config):
    for split in ("train", "val"):
        rows = [r for r in records if r["split"] == split]
        actual = {"cases": len({r["name"] for r in rows}), "findings": len(rows)}
        if actual != config["expected"][split]:
            raise ValueError(f"Wrong {split} census: {actual}")
    train, val = ([r for r in records if r["split"] == s] for s in ("train", "val"))
    for field in ("key", "name"):
        if {r[field] for r in train} & {r[field] for r in val}:
            raise ValueError(f"Train/validation {field} overlap")
    overlap = sorted({r["patient"] for r in train} & {r["patient"] for r in val})
    if overlap != config.get("expected_train_val_patient_overlap", []):
        raise ValueError(f"Unexpected train/validation patient overlap: {overlap}")
    by_half = {h: {patient_id(n) for n, v in halves.items() if v == h} for h in ("A", "B")}
    if set(halves.values()) != {"A", "B"} or by_half["A"] & by_half["B"]:
        raise ValueError("Invalid patient partition")
    if census(records, halves, config["categories"]) != config["category_counts"]:
        raise ValueError("Category or A/B census differs")


def shard_plan(cases, workers):
    plans = []
    names = sorted(n for n, c in cases.items() if c["split"] == "train")
    for shard in range(workers):
        group = names[shard::workers]
        if not group:
            raise ValueError("Empty training shard")
        largest = max(group, key=lambda n: (math.prod(cases[n]["shape"]), n))
        busiest = max(group, key=lambda n: (math.prod(cases[n]["shape"]) * cases[n]["prompt_count"], n))
        smoke = list(dict.fromkeys([largest, busiest]))
        plans.append({"shard": shard, "smoke": smoke,
                      "names": smoke + [n for n in group if n not in smoke]})
    return plans


def prepare(config, root):
    root = Path(root)
    with lock(root / ".prepare.lock"):
        if config["categories"] != list(CATEGORIES) or config["workers"] != 4:
            raise ValueError("Accepted category/worker contract changed")
        if config["model"]["patch_size"] != [192] * 3 or config["overlap"] != .5:
            raise ValueError("Accepted tiling contract changed")
        source = validate_sources(config)
        require_hash(config["base"]["checkpoint"], config["base"]["checkpoint_sha256"])
        require_hash(source["candidate_source"]["plans"]["path"], source["candidate_source"]["plans"]["sha256"])
        split_path = REPO / config["split_manifest"]
        require_hash(split_path, config["split_manifest_sha256"])
        split = read_json(split_path)
        if split["metadata_sha256"] != config["metadata_sha256"] or digest(split["split"]) != split["split_sha256"]:
            raise ValueError("Frozen split provenance differs")
        metadata = read_json(config["metadata"])
        halves = split["split"]["halves"]
        if set(halves) != {r["name"] for r in metadata["val"]}:
            raise ValueError("Split does not cover the complete validation cohort")
        records = finding_records(metadata, config["categories"])
        validate_membership(records, halves, config)
        entries = {r["name"]: r for s in ("train", "val") for r in metadata[s]}
        cases = {}
        for name in sorted({r["name"] for r in records}):
            path = Path(config["preprocessing"]["cache_root"]) / "cases" / case_key(name)
            m = read_json(path / "metadata.json")
            if not (path / ".complete").exists():
                raise ValueError(f"Native cache incomplete: {name}")
            prompts = [entries[name]["findings"][k] for k in sorted(entries[name]["findings"], key=int)]
            if m["prompts"] != prompts or m["preprocess_id"] != "crop_zscore_native_v1":
                raise ValueError(f"Native prompt/preprocessing mismatch: {name}")
            if m["native_cropped_shape_zyx"] != m["resampled_shape_zyx"]:
                raise ValueError(f"Native geometry changed: {name}")
            rows = [r for r in records if r["name"] == name]
            cases[name] = {"split": rows[0]["split"], "keys": [r["key"] for r in rows],
                           "shape": m["resampled_shape_zyx"], "prompt_count": len(prompts),
                           "native_metadata_sha256": sha256(path / "metadata.json")}
        sources = {c["name"]: c for c in source["cases"]}
        baseline = {}
        for r in records:
            if r["split"] == "val":
                s = sources[r["name"]]
                if s["status"] != "complete" or s["same_pass_mask_mismatch_voxels"] != 0:
                    raise ValueError(f"Invalid strict validation case: {r['name']}")
                baseline[r["key"]] = s["same_pass_reference"]["finding_dice"][r["channel"]]
        overlap = config["expected_train_val_patient_overlap"]
        exposure = {"status": "inherited_original_split_patient_overlap" if overlap else "patient_disjoint",
                    "patients": overlap, "findings": [r for r in records if r["patient"] in overlap],
                    "note": "Preserve complete caches; resolve patient exposure before future training/evaluation."}
        prepared = {"records": records, "halves": halves, "cases": cases, "patient_separation": exposure,
                    "split_sha256": split["split_sha256"], "counts": census(records, halves),
                    "source_baseline": baseline, "shards": shard_plan(cases, config["workers"])}
        context = {"config": config, "prepared_sha256": digest(prepared), "code": code_manifest(),
                   "cache_contract": cache_contract(config),
                   "source_manifest_sha256": sha256(Path(config["base"]["validation_cache"]) / "export_manifest.json")}
        context["sha256"] = digest(context)
        immutable_json(root / "config.json", config)
        immutable_json(root / "prepared.json", prepared)
        immutable_json(root / "context.json", context)
        return context, prepared


def check_context(config, root):
    root = Path(root)
    context, prepared = read_json(root / "context.json"), read_json(root / "prepared.json")
    if (context["sha256"] != digest({k: v for k, v in context.items() if k != "sha256"})
            or context["config"] != config or context["prepared_sha256"] != digest(prepared)
            or context["code"] != code_manifest() or context["cache_contract"] != cache_contract(config)):
        raise ValueError("Cache execution provenance changed")
    require_hash(Path(config["base"]["validation_cache"]) / "export_manifest.json", context["source_manifest_sha256"])
    return context, prepared


def finding_index(base, target, size=(192, 192, 192), overlap=.5):
    """Exact historical tile membership, without materializing foreground coordinates."""
    from exp027_model import tile_starts
    if base.shape != target.shape or base.ndim != 3:
        raise ValueError("Tile geometry mismatch")
    tp = fp = gt = 0
    # Bounded z slabs; dense or diffuse predictions must not allocate N x 3 indices.
    for start in range(0, base.shape[0], 8):
        b, y = base[start:start + 8] >= 0, target[start:start + 8] > 0
        t = int(np.count_nonzero(b & y))
        tp += t
        fp += int(np.count_nonzero(b)) - t
        gt += int(np.count_nonzero(y))
    tiles = []
    for starts in tile_starts(base.shape, size, overlap):
        slices = tuple(slice(max(0, a), min(n, a + w)) for a, w, n in zip(starts, size, base.shape))
        b, y = base[slices] >= 0, target[slices] > 0
        count = int(np.count_nonzero(b))
        if count:
            t = int(np.count_nonzero(b & y))
            tiles.append({"starts": list(starts), "tp": t, "fp": count - t})
    denominator = tp + fp + gt
    return {"shape": list(base.shape), "base_tp": tp, "base_fp": fp,
            "base_fn": gt - tp, "total_gt": gt, "base_positive": tp + fp,
            "dice": (2 * tp + 1e-6) / (denominator + 1e-6),
            "deletion_eligible": bool(tiles), "tiles": tiles,
            "tp_indices": [i for i, t in enumerate(tiles) if t["tp"]],
            "fp_indices": [i for i, t in enumerate(tiles) if t["fp"]]}


def build_index(folder, config, records):
    folder = Path(folder)
    base, target = (np.load(folder / n, mmap_mode="r", allow_pickle=False) for n in ("logits.npy", "targets.npy"))
    findings = {}
    for channel, record in enumerate(records):
        value = finding_index(base[channel], target[channel], config["model"]["patch_size"], config["overlap"])
        if value["total_gt"] != record["voxels"]:
            raise ValueError(f"GT voxel count differs: {record['key']}")
        findings[record["key"]] = {"record": record, **value}
    value = {"name": records[0]["name"], "tile_size": config["model"]["patch_size"],
             "overlap": config["overlap"], "findings": findings}
    atomic_json(folder / "tile_index.json", value)
    return value


def verify_validation_geometry(folder, records, source, expected):
    folder = Path(folder)
    meta = read_json(folder / "metadata.json")
    original = load_validation_array(source)
    logits = np.load(folder / "logits.npy", mmap_mode="r", allow_pickle=False)
    index = read_json(folder / "tile_index.json")["findings"]
    for channel, record in enumerate(records):
        restored = restore_crop_to_original(logits[channel], meta["ct_metadata"], fill_value=-30)
        if not np.array_equal(restored >= 0, original[record["channel"]] >= 0):
            raise ValueError(f"Original-space baseline mask changed: {record['key']}")
        if abs(index[record["key"]]["dice"] - expected[record["key"]]) > 1e-12:
            raise ValueError(f"Per-finding baseline differs: {record['key']}")


def file_proof(paths):
    return {str(p): {"bytes": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths}


def verify_case(root, name, config, context, prepared, sources, completing=False):
    folder = Path(root) / "cases" / case_key(name)
    marker_path = folder / "complete.json"
    if not completing and not marker_path.exists():
        raise ValueError(f"Incomplete cache: {name}")
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["cache_contract"] != context["cache_contract"] or marker["name"] != name:
            raise ValueError(f"Completed cache contract changed: {name}")
        for filename, value in marker["hashes"].items():
            require_hash(folder / filename, value)
        if set(marker["hashes"]) != {"metadata.json", "tile_index.json", "logits.npy", "targets.npy"}:
            raise ValueError("Incomplete completion marker")
    records = {r["key"]: r for r in prepared["records"]}
    proof = verify_cache_case(folder, name, records, config, {context["cache_contract"]}, sources)
    selected = [records[k] for k in prepared["cases"][name]["keys"]]
    native = Path(config["preprocessing"]["cache_root"]) / "cases" / case_key(name) / "metadata.json"
    require_hash(native, prepared["cases"][name]["native_metadata_sha256"])
    index = read_json(folder / "tile_index.json")
    if (set(index["findings"]) != {r["key"] for r in selected}
            or index["tile_size"] != config["model"]["patch_size"] or index["overlap"] != config["overlap"]):
        raise ValueError("Tile index membership/geometry differs")
    for r in selected:
        item = index["findings"][r["key"]]
        if (item["record"] != r or item["total_gt"] != r["voxels"]
                or item["base_tp"] + item["base_fn"] != item["total_gt"]):
            raise ValueError("Finding index integrity differs")
    if selected[0]["split"] == "val":
        verify_validation_geometry(folder, selected, sources[name], prepared["source_baseline"])
    if not marker_path.exists():
        meta = read_json(folder / "metadata.json")
        atomic_json(marker_path, {"cache_contract": context["cache_contract"], "name": name,
                    "hashes": {**meta["hashes"], **{n: sha256(folder / n) for n in ("metadata.json", "tile_index.json")}},
                    "completed_at": now()})
    proof["files"].update(file_proof([folder / "tile_index.json", marker_path]))
    proof["cache_bytes"] += (folder / "tile_index.json").stat().st_size + marker_path.stat().st_size
    proof["baseline_findings"] = [{k: v for k, v in item.items() if k not in ("tiles", "tp_indices", "fp_indices")}
                                  for item in index["findings"].values()]
    return proof


def summarize(rows, prepared, categories=CATEGORIES):
    result = []
    for category in categories:
        for subset in ("train", "A", "B", "full"):
            values = [r for r in rows if r["record"]["category"] == category and
                      (r["record"]["split"] == "train" if subset == "train" else
                       r["record"]["split"] == "val" and
                       (subset == "full" or prepared["halves"][r["record"]["name"]] == subset))]
            result.append({"category": category, "subset": subset, "findings": len(values),
                           "cases": len({r["record"]["name"] for r in values}),
                           "mean_dice": math.fsum(r["dice"] for r in values) / len(values) if values else None,
                           "hits": sum(r["dice"] >= .1 for r in values),
                           "base_tp": sum(r["base_tp"] for r in values),
                           "base_fp": sum(r["base_fp"] for r in values),
                           "base_fn": sum(r["base_fn"] for r in values),
                           "deletion_eligible": sum(r["deletion_eligible"] for r in values)})
    return result


def finalize(config, root):
    root = Path(root)
    context, prepared = check_context(config, root)
    verifying = read_json(root / "status.json").get("mode") == "verify"
    receipts = [read_json(p) for p in sorted((root / "receipts").glob("*.json"))
                if p.name.startswith("verify") == verifying]
    if any(r["context_sha256"] != context["sha256"] for r in receipts):
        raise ValueError("Verification receipt provenance changed")
    cases = [c for r in receipts for c in r["cases"]]
    if len(cases) != len(prepared["cases"]) or {c["name"] for c in cases} != set(prepared["cases"]):
        raise ValueError("Incomplete/duplicate verified CT coverage")
    rows = [r for c in cases for r in c["baseline_findings"]]
    if len(rows) != len(prepared["records"]) or {r["record"]["key"] for r in rows} != {r["key"] for r in prepared["records"]}:
        raise ValueError("Incomplete/duplicate verified finding coverage")
    for case in cases:
        if file_proof(case["files"]) != case["files"]:
            raise ValueError(f"Cache changed since verification: {case['name']}")
    summaries = summarize(rows, prepared)
    for row in summaries:
        if row["subset"] == "full" and abs(row["mean_dice"] - config["validation_baselines"][row["category"]]) > 1e-12:
            raise ValueError(f"Category baseline differs: {row}")
    atomic_csv(root / "reports/baselines.csv", summaries)
    atomic_json(root / "reports/baselines.json", summaries)
    atomic_csv(root / "reports/baseline_findings.csv", [{**r["record"], **{k: v for k, v in r.items() if k != "record"},
                "half": prepared["halves"].get(r["record"]["name"])} for r in rows])
    for category in config["categories"]:
        for subset in ("train", "A", "B", "full"):
            keys = [r["key"] for r in prepared["records"] if r["category"] == category and
                    (r["split"] == "train" if subset == "train" else r["split"] == "val" and
                     (subset == "full" or prepared["halves"][r["name"]] == subset))]
            atomic_json(root / "manifests" / category / f"{subset}.json",
                        {"cache_contract": context["cache_contract"], "split_sha256": prepared["split_sha256"],
                         "category": category, "subset": subset, "keys": keys,
                         "cross_split_patient_keys": [r["key"] for r in prepared.get("patient_separation", {}).get("findings", []) if r["key"] in keys]})
    inventory = {"status": "verified", "context_sha256": context["sha256"],
                 "cache_contract": context["cache_contract"], "cases": len(cases), "findings": len(rows),
                 "cache_bytes": sum(c["cache_bytes"] for c in cases), "verified_at": now(),
                 "baselines": summaries, "case_proofs": cases,
                 "patient_separation": prepared.get("patient_separation")}
    atomic_json(root / "input_manifest.json", inventory)
    return inventory


class FindingStore:
    """Read-only, memory-mapped inputs. Missing/incompatible caches always fail."""
    def __init__(self, root, capacity=1):
        self.root, self.capacity, self.loaded = Path(root), capacity, OrderedDict()
        self.config = read_json(self.root / "config.json")
        self.context, self.prepared = check_context(self.config, self.root)
        inventory = read_json(self.root / "input_manifest.json")
        if inventory["status"] != "verified" or inventory["context_sha256"] != self.context["sha256"]:
            raise ValueError("Complete verified cache required")
        self.records = {r["key"]: r for r in self.prepared["records"]}
        self.proofs = {r["name"]: r for r in inventory["case_proofs"]}

    def get(self, key):
        record = self.records[key]
        name = record["name"]
        if name not in self.loaded:
            proof = self.proofs[name]
            if file_proof(proof["files"]) != proof["files"]:
                raise ValueError(f"Verified inputs changed: {name}")
            folder = self.root / "cases" / case_key(name)
            meta = read_json(folder / "metadata.json")
            image = np.load(meta["image_path"], mmap_mode="r", allow_pickle=False)
            z, y = (np.load(folder / f, mmap_mode="r", allow_pickle=False) for f in ("logits.npy", "targets.npy"))
            index = read_json(folder / "tile_index.json")["findings"]
            self.loaded[name] = (image, z, y, index, meta)
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(name)
        image, z, y, index, meta = self.loaded[name]
        channel = next(i for i, r in enumerate(meta["records"]) if r["key"] == key)
        return image[0], z[channel], y[channel], index[key], meta
