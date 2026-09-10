"""Explicit preparation and memory-mapped finding data; never infer in a sampler."""
from __future__ import annotations

import os
import hashlib
import inspect
import json
import time
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from exp027_common import (DEFAULT_CONFIG, EXPERIMENT, REPO, aggregate, atomic_json, atomic_text, code_fingerprint, digest, finding_records,
                          gpu_gate, lock, make_split, now, read_json, require_hash, sha256)


def case_key(name):
    return name.removesuffix(".nii.gz")


def validate_sources(config):
    require_hash(config["metadata"], config["metadata_sha256"])
    require_hash(Path(config["preprocessing"]["cache_root"]) / "manifest.json",
                 config["preprocessing"]["manifest_sha256"])
    cache = Path(config["base"]["validation_cache"])
    manifest = read_json(cache / "export_manifest.json")
    proof = read_json(cache / "reproduction_validation.json")
    if (manifest["status"] != "strict_passed" or manifest["candidate_id"] != config["base"]["candidate_id"]
            or proof["status"] != "passed" or proof["same_pass_mask_mismatch_voxels"] != 0
            or proof["storage_reproduction_status"] != "passed"):
        raise ValueError("Validation cache does not satisfy strict same-pass reproduction")
    source = manifest["candidate_source"]
    if source["checkpoint"]["sha256"] != config["base"]["checkpoint_sha256"]:
        raise ValueError("Validation cache checkpoint mismatch")
    if source["cache"]["manifest_sha256"] != config["preprocessing"]["manifest_sha256"]:
        raise ValueError("Validation preprocessing mismatch")
    if manifest["clip"] != config["base"]["logit_clip"]:
        raise ValueError("Logit clipping contract mismatch")
    return manifest


def prepare(config, root, config_path=DEFAULT_CONFIG):
    """Metadata-only: no torch import, inference, logit conversion, or GPU access."""
    root = Path(root)
    with lock(root / ".prepare.lock"):
        manifest = validate_sources(config)
        metadata = read_json(config["metadata"])
        split = make_split(metadata["val"], config["seed"], config["split"]["candidates"])
        if config["split"].get("manifest"):
            frozen = read_json(REPO / config["split"]["manifest"])
            if frozen["metadata_sha256"] != config["metadata_sha256"] or frozen["split_sha256"] != digest(split):
                raise ValueError("Generated split differs from the frozen repository manifest")
        train, val = finding_records(metadata["train"], "train"), split["findings"]
        expected = config["expected"]
        actual = (len(train), len(val), len({r["name"] for r in val}), len(metadata["val"]))
        wanted = tuple(expected[k] for k in ("train_findings", "val_findings", "val_cases", "val_total_cases"))
        if "train_cases" in expected and len({r["name"] for r in train}) != expected["train_cases"]:
            raise ValueError("Training CT census mismatch")
        if actual != wanted:
            raise ValueError(f"Metadata census mismatch: {actual} != {wanted}")
        train_patients = {r["patient"] for r in train}
        if train_patients & {r["patient"] for r in val}:
            raise ValueError("2a train/validation patient overlap")
        by_name = {r["name"]: r for r in manifest["cases"]}
        baseline = []
        for r in val:
            c = by_name[r["name"]]
            if c["status"] != "complete" or c["same_pass_mask_mismatch_voxels"] != 0:
                raise ValueError(f"Incomplete validation cache: {r['name']}")
            value = c["same_pass_reference"]["finding_dice"][r["channel"]]
            baseline.append({**r, "dice": value, "base_dice": value,
                             "hit": int(value >= .1), "base_hit": int(value >= .1)})
        prepared = {"schema_version": 1, "metadata_sha256": config["metadata_sha256"],
                    "split": split, "train": train, "val": val,
                    "base_manifest_sha256": sha256(Path(config["base"]["validation_cache"]) / "export_manifest.json"),
                    "baseline": aggregate(baseline), "baseline_findings": baseline}
        path = root / "prepared.json"
        if path.exists():
            previous = read_json(path)
            # Derived averages must be stable across Python's 3.12 sum() change.
            # All source records/hashes/splits still require exact equality.
            previous["baseline"] = aggregate(previous["baseline_findings"])
            if previous != prepared:
                raise ValueError("Prepared inputs changed; use a new runtime root")
        atomic_json(path, prepared)
        atomic_json(root / "split_manifest.json", split)
        if read_json(config_path) != config:
            raise ValueError("Configuration source differs from supplied configuration")
        atomic_text(root / "config" / f"{EXPERIMENT}.json", Path(config_path).read_text())
        atomic_json(root / "config" / "code_manifest.json", code_fingerprint())
        if not (root / "status.json").exists():
            full = config["training"].get("total_updates") is not None
            atomic_json(root / "status.json", {"status": config.get("status", "awaiting_timing_approval"), "updated_at": now(),
                                              "active_phase": "full" if full else "benchmark", "ranking": "pending_user_review"})
        return prepared


def original_to_native(array, properties):
    """Preserve continuous values; unlike the existing target helper, never binarize."""
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
    info = properties["nibabel_stuff"]
    transform = ornt_transform(io_orientation(np.asarray(info["original_affine"])),
                               io_orientation(np.asarray(info["reoriented_affine"])))
    return np.transpose(apply_orientation(array, transform), (2, 1, 0))


def native_to_original(array, properties):
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
    info = properties["nibabel_stuff"]
    transform = ornt_transform(io_orientation(np.asarray(info["reoriented_affine"])),
                               io_orientation(np.asarray(info["original_affine"])))
    return apply_orientation(np.transpose(array, (2, 1, 0)), transform)


def restore_crop_to_original(array, metadata, fill_value=0):
    """Established native uncrop followed by CT orientation, without logit clipping."""
    from run_voxtell_val_inference import restore_cached_native_crop
    if metadata["native_cropped_shape_zyx"] != metadata["resampled_shape_zyx"]:
        raise ValueError("Exp027 requires native geometry without resampling")
    native = restore_cached_native_crop(array[None], metadata, fill_value=fill_value)[0]
    return native_to_original(native, metadata["ct_properties"])


def save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".array.")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stored_logits(value, clip):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite source logits")
    clipped = np.clip(value, *clip)
    half = clipped.astype(np.float16)
    # A negative subnormal can round to -0.0 and change a >=0 mask.
    if np.array_equal(half >= 0, clipped >= 0):
        return half
    return clipped


def load_validation_array(record):
    """SideExp003 hashes contiguous array payload bytes, excluding the NPY header."""
    path = Path(record["array_path"])
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
        raise ValueError(f"Validation logit dtype/shape mismatch: {path}")
    if path.stat().st_size != record["npy_bytes"]:
        raise ValueError(f"Validation logit file size mismatch: {path}")
    payload = memoryview(np.ascontiguousarray(array)).cast("B")
    checksum = hashlib.sha256()
    for start in range(0, len(payload), 64 * 1024 * 1024):
        checksum.update(payload[start:start + 64 * 1024 * 1024])
    if checksum.hexdigest() != record["array_sha256"]:
        raise ValueError(f"Validation logit payload SHA256 mismatch: {path}")
    return array


def candidate_points(mask, seed, limit=4096):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    if len(indices) > limit:
        indices = np.random.default_rng(seed).choice(indices, limit, replace=False)
    return np.stack(np.unravel_index(indices, mask.shape), axis=1).tolist()


def cache_producer_fingerprint():
    """Hash only cache-producing code; refiner/report changes cannot stale data."""
    functions = (original_to_native, native_to_original, stored_logits,
                 load_validation_array, candidate_points, write_case_cache)
    return {**{fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in functions},
            **{name: sha256(Path(__file__).parent / name) for name in
               ("run_voxtell_val_inference.py", "voxtell_preprocessed_cache.py",
                "train_text_conditioned_voxtell.py")}}


def cache_contract(config, prepared):
    return digest({"base": config["base"], "preprocessing": config["preprocessing"],
                   "prepared": digest(prepared), "storage": "native_2a_v1",
                   "producer": cache_producer_fingerprint()})


def compatible_contracts(config, prepared):
    current = cache_contract(config, prepared)
    accepted = {current}
    spec = config.get("cache", {}).get("compatibility_manifest")
    if spec:
        path = REPO / spec["path"]
        require_hash(path, spec["sha256"])
        proof = read_json(path)
        legacy = digest({"base": config["base"], "preprocessing": config["preprocessing"],
                         "prepared": digest(prepared), "storage": "native_2a_v1", "code": proof["previous_code"]})
        if proof["current_contract"] == current and proof["legacy_contract"] == legacy:
            accepted.add(legacy)
    return accepted


def required_cache_keys(prepared, events, benchmark):
    records = prepared["train"] + prepared["val"]
    return sorted(({e["key"] for rows in events.values() for e in rows} |
                   {r["key"] for r in prepared["val"]}) if benchmark else {r["key"] for r in records})


def verify_cache_case(folder, name, records, config, contracts, validation_sources=None):
    """Full verification before launch; no GPU and no inference."""
    folder = Path(folder)
    meta_path = folder / "metadata.json"
    meta = read_json(meta_path)
    selected = [r for r in records.values() if r["name"] == name]
    if meta["contract"] not in contracts or meta["name"] != name or meta["records"] != selected:
        raise ValueError(f"Cache provenance/finding coverage mismatch: {name}")
    if set(meta["hashes"]) != {"logits.npy", "targets.npy"}:
        raise ValueError(f"Missing cache array hashes: {name}")
    original = Path(config["preprocessing"]["cache_root"]) / "cases" / case_key(name)
    ct_meta = read_json(original / "metadata.json")
    if meta["ct_metadata"] != ct_meta or Path(meta["image_path"]) != original / "image.npy":
        raise ValueError(f"Cached CT geometry/source mismatch: {name}")
    if ct_meta["preprocess_id"] != "crop_zscore_native_v1" or ct_meta["native_cropped_shape_zyx"] != ct_meta["resampled_shape_zyx"]:
        raise ValueError(f"Unexpected cache geometry: {name}")
    shape = tuple(meta["shape"])
    if shape != tuple(ct_meta["resampled_shape_zyx"]):
        raise ValueError(f"Cache extent mismatch: {name}")
    image = np.load(meta["image_path"], mmap_mode="r", allow_pickle=False)
    if image.shape != (1, *shape) or image.dtype != np.float32:
        raise ValueError(f"CT array shape/dtype mismatch: {name}")
    payload = memoryview(image).cast("B")
    checksum = hashlib.sha256()
    # Match voxtell_preprocessed_cache.sha256_array: dtype + shape + payload.
    checksum.update(str(image.dtype).encode("ascii"))
    checksum.update(json.dumps(list(image.shape)).encode("ascii"))
    for start in range(0, len(payload), 64 * 1024 * 1024):
        checksum.update(payload[start:start + 64 * 1024 * 1024])
    if checksum.hexdigest() != ct_meta["image_sha256"]:
        raise ValueError(f"CT payload SHA256 mismatch: {name}")
    for filename, checksum in meta["hashes"].items():
        require_hash(folder / filename, checksum)
        array = np.load(folder / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(selected), *shape):
            raise ValueError(f"Cache array shape mismatch: {name}")
        if filename == "logits.npy":
            if array.dtype not in (np.float16, np.float32) or not np.isfinite(array).all():
                raise ValueError(f"Invalid cache logits: {name}")
            if np.min(array) < config["base"]["logit_clip"][0] or np.max(array) > config["base"]["logit_clip"][1]:
                raise ValueError(f"Cache clipping mismatch: {name}")
            if str(array.dtype) != meta["origin"]["dtype"]:
                raise ValueError(f"Cache dtype provenance mismatch: {name}")
        elif array.dtype != np.uint8 or not np.all((array == 0) | (array == 1)):
            raise ValueError(f"Invalid cache target: {name}")
    for r in selected:
        if ct_meta["prompts"][r["channel"]] != r["prompt"] or not meta["points"][r["key"]]["gt"]:
            raise ValueError(f"Finding index/prompt mismatch: {r['key']}")
    if selected[0]["split"] == "train":
        if (meta["origin"]["kind"] != "frozen_exp007_inference" or
                meta["origin"]["checkpoint_sha256"] != config["base"]["checkpoint_sha256"] or
                meta["origin"].get("same_pass_mask_mismatch_voxels") != 0):
            raise ValueError(f"Wrong training base checkpoint: {name}")
    elif meta["origin"]["kind"] != "strict_validation_cache":
        raise ValueError(f"Wrong validation base cache: {name}")
    elif validation_sources is not None and meta["origin"]["array_sha256"] != validation_sources[name]["array_sha256"]:
        raise ValueError(f"Validation source payload differs: {name}")
    paths = [meta_path, folder / "logits.npy", folder / "targets.npy", Path(meta["image_path"]), original / "metadata.json"]
    return {"name": name, "keys": [r["key"] for r in selected], "contract": meta["contract"],
            "metadata_sha256": sha256(meta_path),
            "files": {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths},
            "cache_bytes": sum((folder / f).stat().st_size for f in ("logits.npy", "targets.npy", "metadata.json"))}


def cache_gate(root, prepared, config, keys):
    """Consume all worker receipts, verifying complete coverage and unchanged files."""
    root = Path(root)
    records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
    required_names = {records[k]["name"] for k in keys}
    expected_keys = {r["key"] for r in records.values() if r["name"] in required_names}
    receipts = [read_json(path) for path in sorted(root.glob("cache_verified_gpu*.json"))]
    rows = [row for receipt in receipts for row in receipt["cases"]]
    if any(r["prepared_sha256"] != digest(prepared) or r["cache_contract"] != cache_contract(config, prepared) for r in receipts):
        raise ValueError("Cache verification receipts have different provenance")
    if len(rows) != len(required_names) or {r["name"] for r in rows} != required_names:
        raise ValueError("Incomplete or duplicate CT cache coverage")
    if {key for row in rows for key in row["keys"]} != expected_keys or sum(len(row["keys"]) for row in rows) != len(expected_keys):
        raise ValueError("Incomplete or duplicate finding cache coverage")
    for row in rows:
        for path, expected in row["files"].items():
            stat = Path(path).stat()
            if expected != {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError(f"Cache changed after verification: {path}")
    value = {"status": "verified", "updated_at": now(), "prepared_sha256": digest(prepared),
             "cache_contract": cache_contract(config, prepared), "cases": len(rows), "findings": len(expected_keys),
             "names": sorted(required_names), "keys": sorted(expected_keys),
             "cache_bytes": sum(row["cache_bytes"] for row in rows)}
    atomic_json(root / "cache_inventory.json", value)
    return value


def require_full_cache(root, prepared, config):
    value = read_json(Path(root) / "cache_inventory.json")
    records = prepared["train"] + prepared["val"]
    if (value["status"] != "verified" or value["prepared_sha256"] != digest(prepared)
            or value["cache_contract"] != cache_contract(config, prepared)
            or value["keys"] != sorted(r["key"] for r in records)
            or value["names"] != sorted({r["name"] for r in records})):
        raise ValueError("Full training requires the verified complete 2a cache")
    return value


def build_cache(config, prepared, root, keys, *, allow_gpu=False, device=0):
    """Explicit GPU-gated preparation, including conversion of existing val logits."""
    gpu_gate(allow_gpu)
    import torch
    from voxtell_preprocessed_cache import load_cached_case, sha256_array
    from run_voxtell_val_inference import predict_preprocessed_crop_logits
    from voxtell.inference.predictor import VoxTellPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("GPU cache preparation requires CUDA; no CPU fallback")
    manifest = validate_sources(config)
    require_hash(config["base"]["checkpoint"], config["base"]["checkpoint_sha256"])
    source = manifest["candidate_source"]
    require_hash(source["plans"]["path"], source["plans"]["sha256"])
    records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
    if set(keys) - records.keys():
        raise ValueError("Unknown finding requested for cache")
    names = sorted({records[k]["name"] for k in keys})
    entries = {r["name"]: r for rows in read_json(config["metadata"]).values() for r in rows}
    val_sources = {r["name"]: r for r in manifest["cases"]}
    contract = cache_contract(config, prepared)
    contracts = compatible_contracts(config, prepared)
    predictor = None
    worker_id = os.environ.get("CUDA_VISIBLE_DEVICES", str(device)).replace(",", "_")
    receipt_path = Path(root) / f"cache_verified_gpu{worker_id}.json"
    progress_path = Path(root) / f"cache_progress_gpu{worker_id}.json"
    cases, reused, generated = [], 0, 0
    started = time.perf_counter()
    name = "pending"

    def publish(status, error=None):
        atomic_json(progress_path, {"status": status, "done": len(cases), "verified": len(cases),
                    "total": len(names), "reused": reused, "generated": generated, "name": name,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cache_bytes": sum(row["cache_bytes"] for row in cases), "error": error, "updated_at": now()})

    atomic_json(receipt_path, {"cases": [], "prepared_sha256": digest(prepared), "cache_contract": contract})
    try:
        publish("cache_preparation")
        for name in names:
            folder = Path(root) / "cache" / case_key(name)
            publish("cache_preparation")
            with lock(folder / ".cache.lock"):
                if (folder / "metadata.json").exists():
                    reused += 1
                else:
                    predictor = write_case_cache(config, records, entries, val_sources, folder, name, predictor, device, contract)
                    generated += 1
                cases.append(verify_cache_case(folder, name, records, config, contracts, val_sources))
            atomic_json(receipt_path, {"cases": cases, "prepared_sha256": digest(prepared), "cache_contract": contract})
            publish("cache_preparation")
        publish("complete")
    except BaseException as exc:
        publish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del predictor


def write_case_cache(config, records, entries, val_sources, folder, name, predictor, device, contract):
    """Numerically unchanged case export from the completed benchmark producer."""
    import torch
    from voxtell_preprocessed_cache import load_cached_case, sha256_array
    from run_voxtell_val_inference import predict_preprocessed_crop_logits
    from voxtell.inference.predictor import VoxTellPredictor
    meta_path = folder / "metadata.json"
    original = Path(config["preprocessing"]["cache_root"]) / "cases" / case_key(name)
    image, all_targets, meta = load_cached_case(Path(config["preprocessing"]["cache_root"]), name, mmap_image=True)
    if meta["preprocess_id"] != "crop_zscore_native_v1":
        raise ValueError("Unexpected native preprocessing ID")
    if sha256_array(image) != meta["image_sha256"] or sha256_array(all_targets) != meta["targets_sha256"]:
        raise ValueError(f"Preprocessed array hash mismatch: {name}")
    selected = [r for r in records.values() if r["name"] == name]
    row = entries[name]
    prompts = [row["findings"][k] for k in sorted(row["findings"], key=int)]
    if meta["prompts"] != prompts or len(all_targets) != len(prompts):
        raise ValueError(f"Original prompt/channel ordering mismatch: {name}")
    # Keep every 2a channel from a scheduled case, avoiding repeat VoxTell inference.
    channels = [r["channel"] for r in selected]
    if name in val_sources:
        c = val_sources[name]
        array = load_validation_array(c)
        bbox = tuple(slice(*p) for p in meta["crop_bbox_zyx"])
        logits = np.stack([original_to_native(array[i], meta["ct_properties"])[bbox] for i in channels])
        origin = {"kind": "strict_validation_cache", "array_sha256": c["array_sha256"], "dtype": str(array.dtype)}
    else:
        if predictor is None:
            predictor = VoxTellPredictor(model_dir=str(Path(config["base"]["checkpoint"]).parents[1]),
                                        device=torch.device(f"cuda:{device}"), use_precomputed_embeddings=False)
        generated = predict_preprocessed_crop_logits(predictor, image, prompts, padding_value=0.0)
        logits = stored_logits(generated[channels], config["base"]["logit_clip"])
        origin = {"kind": "frozen_exp007_inference", "dtype": str(logits.dtype),
                  "same_pass_mask_mismatch_voxels": 0, "checkpoint_sha256": config["base"]["checkpoint_sha256"]}
        del generated
    targets = np.asarray(all_targets[channels] > 0, dtype=np.uint8)
    if logits.shape != targets.shape or tuple(logits.shape[1:]) != tuple(image.shape[1:]):
        raise ValueError(f"CT/logit/target geometry mismatch: {name}")
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite cache: {name}")
    hashes = {}
    for filename, array in (("logits.npy", logits), ("targets.npy", targets)):
        save_npy(folder / filename, array)
        hashes[filename] = sha256(folder / filename)
    points = {}
    for i, record in enumerate(selected):
        points[record["key"]] = {"gt": candidate_points(targets[i], config["seed"] + i),
                                  "prediction": candidate_points(logits[i] >= 0, config["seed"] + i)}
        if not points[record["key"]]["gt"]:
            raise ValueError(f"Empty annotated 2a finding: {record['key']}")
    atomic_json(meta_path, {"contract": contract, "name": name, "records": selected, "hashes": hashes,
                           "points": points, "image_path": str(original / "image.npy"), "origin": origin,
                           "shape": list(image.shape[1:]), "ct_metadata": meta, "completed_at": now()})
    return predictor


def extract_patch(array, starts, size, fill):
    result = np.full(size, fill, dtype=array.dtype)
    valid = np.zeros(size, dtype=np.float32)
    src, dst = [], []
    for n, start, length in zip(array.shape, starts, size):
        lo, hi = max(0, start), min(n, start + length)
        if hi <= lo:
            raise ValueError("Patch does not intersect image")
        src.append(slice(lo, hi))
        dst.append(slice(lo - start, hi - start))
    result[tuple(dst)] = array[tuple(src)]
    valid[tuple(dst)] = 1
    return result, valid


class FindingStore:
    def __init__(self, root, prepared, config, capacity=2):
        self.root, self.capacity = Path(root), capacity
        self.records = {r["key"]: r for r in prepared["train"] + prepared["val"]}
        self.contracts = compatible_contracts(config, prepared)
        self.loaded = OrderedDict()

    def get(self, key):
        record = self.records[key]
        name = record["name"]
        if name not in self.loaded:
            folder = self.root / "cache" / case_key(name)
            meta = read_json(folder / "metadata.json")  # Missing cache is a hard failure, never online inference.
            if meta["contract"] not in self.contracts:
                raise ValueError("Cache contract differs from prepared inputs")
            arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in
                      (meta["image_path"], folder / "logits.npy", folder / "targets.npy")]
            self.loaded[name] = (meta, arrays)
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(name)
        meta, arrays = self.loaded[name]
        i = next(i for i, r in enumerate(meta["records"]) if r["key"] == key)
        return arrays[0][0], arrays[1][i], arrays[2][i], meta["points"][key], meta

    def patch(self, event, size):
        image, base, target, points, _ = self.get(event["key"])
        rng = np.random.default_rng(event["patch_seed"])
        mode = event["mode"]
        fallback = mode == "prediction" and not points["prediction"]
        actual = "random" if fallback else mode
        point = None if actual == "random" else points[actual][int(rng.integers(len(points[actual])))]
        starts = []
        for axis, (n, width) in enumerate(zip(image.shape, size)):
            if n < width:
                starts.append(-(width - n) // 2)
            elif point is None:
                starts.append(int(rng.integers(n - width + 1)))
            else:
                starts.append(int(rng.integers(max(0, point[axis] - width + 1), min(point[axis], n - width) + 1)))
        ct, valid = extract_patch(image, starts, size, 0)
        z, _ = extract_patch(base, starts, size, -30)
        y, _ = extract_patch(target, starts, size, 0)
        arrays = [ct, z, y, valid]
        flips = [bool(rng.integers(2)) for _ in range(3)]
        for axis, flip in enumerate(flips):
            if flip:
                arrays = [np.flip(a, axis) for a in arrays]
        ct, z, y, valid = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
        return np.stack([ct, z]), y[None], valid[None], {"fallback": fallback, "actual_mode": actual,
                                                     "starts": starts, "flips": flips}
