#!/usr/bin/env python3
"""Split-specific, label-free test300 contracts and resumable GPU exporter."""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import fresh_cache as fc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_test300_v1"
)
STAGING_ROOT = Path("/data/hengjie/sideexp003_staging")
MIN_SHARED = 20 * 1024**4
METRIC_STATUS = "not_available_withheld_labels"


def require(ok, message):
    if not ok:
        raise fc.FreshCacheError(message)


def resolve(path):
    p = Path(path)
    if p.exists():
        return p
    for prefix in (Path("/workspace"), Path("/home/hengjie/code_sync/rexgroundingct")):
        try:
            return REPO / p.relative_to(prefix)
        except ValueError:
            pass
    return p


def load_cases(path, expected_cases=300, expected_findings=582):
    data = fc.read_json(Path(path))
    cases = data.get("test") if isinstance(data, dict) else None
    require(
        isinstance(cases, list) and len(cases) == expected_cases,
        "test case count mismatch",
    )
    names = [c["name"] for c in cases]
    require(len(set(names)) == len(names), "duplicate test cases")
    require(
        all(Path(n).name == n and n.endswith(".nii.gz") for n in names),
        "unsafe case name",
    )
    for c in cases:
        indices = sorted(int(i) for i in c["findings"])
        require(
            indices == list(range(len(indices))),
            "finding indices must be contiguous from zero",
        )
        require(
            all(isinstance(v, str) for v in c["findings"].values()), "invalid prompt"
        )
        require(
            len(c["shape"]) == 3 and all(int(v) > 0 for v in c["shape"]),
            "invalid CT shape",
        )
    require(
        sum(len(c["findings"]) for c in cases) == expected_findings,
        "test prompt count mismatch",
    )
    return cases


def prompt_contract(case):
    ids = sorted(case["findings"], key=int)
    return {
        "finding_indices": [int(i) for i in ids],
        "prompt_sha256s": [
            hashlib.sha256(case["findings"][i].encode()).hexdigest() for i in ids
        ],
    }


def output_elements(cases):
    return sum(math.prod(c["shape"]) * len(c["findings"]) for c in cases)


def wave_space_ok(shared_free, local_free, unfinished, elements):
    # Full float32 destination and staging for every unfinished worker, plus atomic-file slack.
    need = unfinished * elements * 4 + 32 * 1024**3
    return shared_free >= MIN_SHARED + need and local_free >= need


@contextlib.contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise fc.FreshCacheError(f"live owner: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def source_bundle():
    entries = fc.source_bundle()["files"]
    for name in ("test300_cache.py", "test300_runner.py", "test300_execution_spec.md"):
        p = HERE / name
        entries.append({"path": str(p.relative_to(REPO)), "sha256": fc.sha256_file(p)})
    entries.sort(key=lambda x: x["path"])
    return {"files": entries, "sha256": fc.json_sha256(entries)}


def build_job(roster, job_id, staging_root=STAGING_ROOT, native_root=NATIVE_ROOT):
    require(job_id == "j003_top20_test300_fresh_gpu8", "unexpected test300 job ID")
    require(
        Path(staging_root) == STAGING_ROOT,
        "test300 staging base is frozen under local /data",
    )
    errors = fc.validate_real_sources(roster, verify_checkpoints=False)
    require(not errors, "; ".join(errors))
    require(roster["N"] == 20, "test export requires frozen top 20")
    dataset = {
        **fc.validate_test300_contract(),
        "split": "test300",
        "labels": "withheld",
    }
    cases = load_cases(dataset["path"])
    native_manifest = Path(native_root) / "manifest.json"
    require(native_manifest.is_file(), "build separate native test300 cache first")
    native = fc.read_json(native_manifest)
    require(
        native["preprocess_id"] == "crop_zscore_native_v1"
        and native["cases"] == 300
        and native["splits"] == ["test"]
        and native["targets"] == 0,
        "invalid native test-only cache",
    )
    require(
        native["metadata_sha256"] == dataset["sha256"],
        "native test cache dataset drift",
    )
    require(
        (Path(native_root) / ".complete").read_text().strip()
        == fc.sha256_file(native_manifest),
        "native cache incomplete",
    )
    job = fc.build_job(roster, job_id=job_id)
    job.update(
        {
            "dataset": dataset,
            "source_bundle": source_bundle(),
            "kind": "test300_base_logits",
            "staging_root": str(Path(staging_root) / job_id),
            "host": "shenggpu8",
            "parent_roster": {
                "path": str(HERE / "rosters" / f"roster_{roster['roster_id']}.json"),
                "roster_sha256": roster["roster_sha256"],
            },
            "metric_status": METRIC_STATUS,
            "minimum_shared_free_bytes": MIN_SHARED,
            "output_elements_per_model": output_elements(cases),
            "initial_eta_hours": {"all_20": [24, 30]},
            "storage_contract": {
                "layout": "F,X,Y,Z",
                "kind": "pre_sigmoid_logits",
                "clip": [-30, 30],
                "staging_dtype": "float32",
                "publication_policy": "float16_if_exact_threshold_sign_else_float32",
                "geometry_reference": "native_CT_header",
                "same_pass_probability_threshold": 0.5,
            },
        }
    )
    job.pop("test300", None)
    preprocessing_cases = {}
    for c in job["candidates"]:
        parent_id = (
            "j001_top20_val200_fresh"
            if c["rank"] <= 8
            else "j002_top20_val200_fresh_r09_r20"
        )
        parent = fc.read_json(HERE / "cache_jobs" / parent_id / "job_spec.json")
        source = next(v for v in parent["candidates"] if v["id"] == c["id"])
        require(
            source["checkpoint"]["sha256"] == c["checkpoint"]["sha256"]
            and source["config"]["sha256"] == c["config"]["sha256"],
            "paired val checkpoint/config drift",
        )
        vp = Path(source["cache_root"]) / "reproduction_validation.json"
        ep = Path(source["cache_root"]) / "export_manifest.json"
        validation = fc.read_json(vp)
        require(
            validation["status"] == "passed"
            and validation["cases"] == 200
            and validation["findings"] == 381
            and validation["array_hashes_verified"],
            "paired val cache not strict",
        )
        c["paired_val200"] = {
            "job_id": parent_id,
            "cache_key": source["cache_key"],
            "export_manifest": str(ep),
            "export_sha256": fc.sha256_file(ep),
            "validation": str(vp),
            "validation_sha256": fc.sha256_file(vp),
        }
        c["training_preprocessing"] = copy.deepcopy(c["cache"])
        if c["cache"]["id"] == "crop_zscore_native_v1":
            c["cache"].update(
                root=str(native_root),
                manifest_path=str(native_manifest),
                manifest_sha256=fc.sha256_file(native_manifest),
            )
        else:
            require(
                c["cache"]["id"] == "crop_clip1024_linear_iso07_v1",
                "unhandled preprocessing",
            )
        if c["cache"]["id"] not in preprocessing_cases:
            index = {}
            for case in cases:
                case_root = (
                    Path(c["cache"]["root"])
                    / "cases"
                    / case["name"].removesuffix(".nii.gz")
                )
                require(
                    all(
                        (case_root / f).is_file()
                        for f in ("image.npy", "metadata.json", ".complete")
                    ),
                    f"missing test cache: {case_root}",
                )
                meta = fc.read_json(case_root / "metadata.json")
                require(
                    meta.get("targets_sha256") is None,
                    "test preprocessing unexpectedly contains labels",
                )
                require(
                    (case_root / ".complete").read_text().strip()
                    == meta["image_sha256"],
                    "incomplete preprocessing case",
                )
                index[case["name"]] = {
                    "metadata_sha256": fc.sha256_file(case_root / "metadata.json"),
                    "image_sha256": meta["image_sha256"],
                }
            preprocessing_cases[c["cache"]["id"]] = index
        c["test_preprocessing_cases"] = preprocessing_cases[c["cache"]["id"]]
        c["inference_contract"] = {
            **c["inference_contract"],
            "restore": "cached crop to native CT header layout",
            "geometry_reference": "CT",
            "split": "test300",
        }
        c["inference_contract_sha256"] = fc.json_sha256(c["inference_contract"])
        key = "v3_" + fc.json_sha256(
            {
                "split": "test300",
                "dataset": dataset,
                "checkpoint": c["checkpoint"]["sha256"],
                "config": c["config"]["sha256"],
                "preprocessing": c["cache"]["manifest_sha256"],
                "inference": c["inference_contract_sha256"],
                "preprocessing_cases": fc.json_sha256(c["test_preprocessing_cases"]),
                "source": job["source_bundle"]["sha256"],
                "image": job["container"]["image_id"],
                "storage": job["storage_contract"],
            }
        )
        c.update(
            cache_key=key,
            cache_version_id=f"sideexp003_{key}",
            cache_root=str(fc.RUNTIME_ROOT / "cache/logits/by_cache_key" / key),
            staging_root=str(Path(job["staging_root"]) / key),
        )
    job.pop("job_spec_sha256", None)
    job["job_spec_sha256"] = fc.json_sha256(job)
    return job


def validate_job(job, verify_sources=True):
    errors = fc.validate_job(job)
    require(not errors, "; ".join(errors))
    require(
        job["job_id"] == "j003_top20_test300_fresh_gpu8"
        and len(job["candidates"]) == 20
        and job["wave_size"] == 4
        and job["gpus"] == [0, 1, 2, 3],
        "test wave/roster contract drift",
    )
    require(
        Path(job["runtime_root"]) == fc.RUNTIME_ROOT / "cache/jobs" / job["job_id"],
        "unsafe runtime root",
    )
    require(
        job.get("kind") == "test300_base_logits"
        and job["dataset"].get("split") == "test300",
        "wrong split/job kind",
    )
    require(
        job["dataset"]["cases"] == 300 and job["dataset"]["findings"] == 582,
        "wrong cohort",
    )
    require(
        job["dataset"]["sha256"] == fc.TEST300_SHA256, "test metadata identity drift"
    )
    require(
        Path(job["staging_root"]).parent == STAGING_ROOT
        and Path(job["staging_root"]).name == job["job_id"],
        "staging must stay in the managed local job directory",
    )
    if verify_sources:
        for entry in job["source_bundle"]["files"]:
            require(
                fc.sha256_file(REPO / entry["path"]) == entry["sha256"],
                f"source drift: {entry['path']}",
            )
        require(
            fc.sha256_file(resolve(job["dataset"]["path"])) == job["dataset"]["sha256"],
            "dataset drift",
        )
    for c in job["candidates"]:
        require(
            Path(c["staging_root"]).resolve() == Path(c["staging_root"]),
            "symlinked staging directory",
        )
        require(
            Path(c["progress_path"])
            == Path(job["runtime_root"]) / "progress" / f"{c['id']}.json",
            "unsafe progress path",
        )
        require(
            Path(c["staging_root"]) == Path(job["staging_root"]) / c["cache_key"],
            "unsafe staging root",
        )
        require(
            Path(c["cache_root"])
            == fc.RUNTIME_ROOT / "cache/logits/by_cache_key" / c["cache_key"],
            "unsafe publication root",
        )


def sync_catalog(catalog, job):
    result = copy.deepcopy(catalog)
    by_id = {c["candidate_id"]: c for c in result["candidates"]}
    for c in job["candidates"]:
        split = by_id[c["id"]]["inference_artifacts"]["test300"]
        split["dataset"] = job["dataset"]
        p = Path(c["progress_path"])
        progress = (
            fc.read_json(p)
            if p.is_file()
            else {"status": "queued", "cases": 0, "findings": 0}
        )
        if p.is_file():
            require(
                progress.get("job_spec_sha256") == job["job_spec_sha256"],
                "foreign progress",
            )
        version = fc.fresh_version_from_job(c, job)
        version.update(
            {
                k: v
                for k, v in progress.items()
                if k in version or k in ("bytes", "metric_status")
            }
        )
        version["metrics_path"] = None
        version["metric_status"] = METRIC_STATUS
        version["strict_eligible"] = progress["status"] == "strict_passed"
        split["versions"] = [
            v
            for v in split["versions"]
            if v["cache_version_id"] != c["cache_version_id"]
        ] + [version]
        split.update(status=progress["status"], metric_status=METRIC_STATUS)
        if progress["status"] == "strict_passed":
            split.update(
                active_cache_version=c["cache_version_id"],
                cache_path=c["cache_root"],
                result_path=str(Path(c["cache_root"]) / "reproduction_validation.json"),
            )
    result["artifact_state_updated_at_utc"] = fc.utc_now()
    return result


def inference_imports():
    sys.path.insert(
        0, str(REPO / "side_experiments/sideexp002_multimodel_ensemble_selection")
    )
    import export_logits as ex

    return ex


def array_sha256(array):
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), 64 * 1024 * 1024):
        digest.update(view[offset : offset + 64 * 1024 * 1024])
    return digest.hexdigest()


def atomic_save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_test_image(candidate, case):
    # Deliberately avoid load_cached_case: it opens targets when present even with require_targets=False.
    root = (
        Path(candidate["cache"]["root"])
        / "cases"
        / case["name"].removesuffix(".nii.gz")
    )
    frozen = candidate["test_preprocessing_cases"][case["name"]]
    require(
        fc.sha256_file(root / "metadata.json") == frozen["metadata_sha256"],
        "case preprocessing metadata drift",
    )
    meta = fc.read_json(root / "metadata.json")
    require(
        (root / ".complete").read_text().strip() == frozen["image_sha256"],
        "incomplete preprocessing case",
    )
    image = np.load(root / "image.npy", allow_pickle=False)
    from voxtell_preprocessed_cache import sha256_array

    require(sha256_array(image) == frozen["image_sha256"], "preprocessing image drift")
    return image, meta


def predict_test(predictor, candidate, case):
    import nibabel as nib

    ex = inference_imports()
    from common import ct_rate_abs_path, CT_ROOT, sorted_prompts
    from run_voxtell_val_inference import (
        predict_preprocessed_crop_branch_logits,
        predict_preprocessed_crop_logits,
        restore_cached_native_crop,
        export_prediction_to_ct_layout,
    )
    from voxtell_preprocessed_cache import image_padding_value

    image, meta = load_test_image(candidate, case)
    prompts = sorted_prompts(case)
    kwargs = {"padding_value": image_padding_value(meta)}
    if isinstance(predictor, ex.DualBranchVoxTellPredictor):
        crop = predict_preprocessed_crop_branch_logits(
            predictor, image, prompts, **kwargs
        )["final"]
    else:
        crop = predict_preprocessed_crop_logits(predictor, image, prompts, **kwargs)
    restored = restore_cached_native_crop(crop, meta, fill_value=-30)
    ct_path = ct_rate_abs_path(case["name"], CT_ROOT)
    ct = nib.load(str(ct_path))
    require(
        list(ct.shape) == case["shape"], "CT shape differs from frozen test metadata"
    )
    require(
        np.array_equal(
            ct.affine,
            np.asarray(meta["ct_properties"]["nibabel_stuff"]["original_affine"]),
        ),
        "CT affine differs from frozen preprocessing geometry",
    )
    exported, orientation = export_prediction_to_ct_layout(
        restored, ct, meta["ct_properties"], case["name"], output_dtype=None
    )
    require(
        tuple(exported.shape) == (len(prompts), *ct.shape),
        "test native output shape mismatch",
    )
    return exported, {
        "orientation": orientation,
        "affine": ct.affine.tolist(),
        "ct_shape": list(ct.shape),
        "ct_path": str(ct_path),
        "metric_status": METRIC_STATUS,
        **prompt_contract(case),
        "preprocessing_image_sha256": meta["image_sha256"],
    }


def stage_paths(candidate, name):
    root = Path(candidate["staging_root"])
    return root / "cases" / f"{name}.npy", root / "records" / f"{name}.json"


def cleanup_stage(root):
    # The worker's root is a Docker bind mount: remove its contents, not the mount point.
    require(root.resolve() == root and not root.is_symlink(), "unsafe staging cleanup")
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def check_array(path, record, case, job_sha, cache_key):
    require(
        record.get("job_spec_sha256") == job_sha and record.get("name") == case["name"],
        "foreign case record",
    )
    require(record.get("cache_key") == cache_key, "foreign checkpoint case record")
    require(
        record.get("finding_indices") == prompt_contract(case)["finding_indices"]
        and record.get("prompt_sha256s") == prompt_contract(case)["prompt_sha256s"],
        "prompt alignment drift",
    )
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    require(
        tuple(a.shape) == (len(case["findings"]), *case["shape"]),
        "array geometry mismatch",
    )
    require(
        str(a.dtype) == record["dtype"] and list(a.shape) == record["shape"],
        "array dtype/shape drift",
    )
    require(array_sha256(a) == record["array_sha256"], "array hash mismatch")
    for start in range(0, a.size, 4 * 1024 * 1024):
        chunk = a.reshape(-1)[start : start + 4 * 1024 * 1024]
        require(
            np.isfinite(chunk).all() and (abs(chunk) <= 30).all(),
            "nonfinite/out-of-range logits",
        )
    return a


def load_stage(candidate, case, job_sha):
    path, record_path = stage_paths(candidate, case["name"])
    if not record_path.exists():
        return None  # An orphan .npy has no completion record and is safely recomputed.
    require(path.is_file(), "recorded staging array missing")
    record = fc.read_json(record_path)
    check_array(path, record, case, job_sha, candidate["cache_key"])
    require(record["dtype"] == "float32", "staging dtype mismatch")
    return record


def smoke_cases(cases, cache_root):
    def size(c):
        m = fc.read_json(
            Path(cache_root)
            / "cases"
            / c["name"].removesuffix(".nii.gz")
            / "metadata.json"
        )
        return math.prod(m["resampled_shape_zyx"])

    largest_input = max(cases, key=size)
    largest_output = max(
        cases, key=lambda c: math.prod(c["shape"]) * len(c["findings"])
    )
    return list(dict.fromkeys([largest_input["name"], largest_output["name"]]))


def publish(job, candidate, cases):
    dtype = "float16"
    # Decide one dtype for the candidate without materializing another whole cache.
    for case in cases:
        a = np.load(
            stage_paths(candidate, case["name"])[0], mmap_mode="r", allow_pickle=False
        )
        flat = a.reshape(-1)
        for start in range(0, a.size, 4 * 1024 * 1024):
            v = flat[start : start + 4 * 1024 * 1024]
            if np.count_nonzero((v >= 0) != (v.astype(np.float16) >= 0)):
                dtype = "float32"
                break
        if dtype == "float32":
            break
    root = Path(candidate["cache_root"])
    records = []
    for case in cases:
        source, rp = stage_paths(candidate, case["name"])
        record = fc.read_json(rp)
        raw = np.load(source, mmap_mode="r", allow_pickle=False)
        path = root / "cases" / f"{case['name']}.npy"
        manifest = root / "case_manifests" / f"{case['name']}.json"
        if manifest.is_file():
            previous = fc.read_json(manifest)
            check_array(
                path, previous, case, job["job_spec_sha256"], candidate["cache_key"]
            )
            require(previous["dtype"] == dtype, "interrupted publication dtype changed")
            require(
                np.count_nonzero((raw >= 0) != (np.load(path, mmap_mode="r") >= 0))
                == 0,
                "resumed publication mask mismatch",
            )
            records.append(previous)
            continue
        require(
            shutil.disk_usage(root).free
            >= MIN_SHARED + raw.size * np.dtype(dtype).itemsize + 1024**3,
            "shared disk reserve reached during publication",
        )
        compact = np.ascontiguousarray(raw, dtype=dtype)
        require(
            np.count_nonzero((raw >= 0) != (compact >= 0)) == 0,
            "publication mask mismatch",
        )
        atomic_save_npy(path, compact)
        record.update(
            dtype=dtype,
            array_path=str(path),
            array_sha256=array_sha256(compact),
            npy_bytes=path.stat().st_size,
            same_pass_mask_mismatch_voxels=0,
            storage_reproduction_status="passed",
            metric_status=METRIC_STATUS,
        )
        fc.atomic_write_json(manifest, record)
        records.append(record)
    return dtype, records


def validate_published(job, candidate, cases):
    import nibabel as nib
    from common import ct_rate_abs_path, CT_ROOT

    root = Path(candidate["cache_root"])
    records = []
    total = 0
    for case in cases:
        p = root / "cases" / f"{case['name']}.npy"
        rec = fc.read_json(root / "case_manifests" / f"{case['name']}.json")
        a = check_array(p, rec, case, job["job_spec_sha256"], candidate["cache_key"])
        require(
            Path(rec["ct_path"]) == ct_rate_abs_path(case["name"], CT_ROOT),
            "wrong CT reference",
        )
        ct = nib.load(rec["ct_path"])
        require(
            tuple(a.shape[1:]) == ct.shape
            and np.array_equal(np.asarray(rec["affine"]), ct.affine),
            "CT affine/shape drift",
        )
        require(
            rec.get("same_pass_mask_mismatch_voxels") == 0
            and rec.get("storage_reproduction_status") == "passed",
            "missing exact storage proof",
        )
        require(
            not any(k in rec for k in ("dice", "hits", "same_pass_reference"))
            and rec.get("metric_status") == METRIC_STATUS,
            "test metrics must be unavailable",
        )
        records.append(rec)
        total += p.stat().st_size
    require(len(records) == job["dataset"]["cases"], "incomplete test cache")
    require(
        sum(r["shape"][0] for r in records) == job["dataset"]["findings"],
        "incomplete prompt coverage",
    )
    return {
        "status": "passed",
        "cases": len(records),
        "findings": sum(r["shape"][0] for r in records),
        "dtype": records[0]["dtype"],
        "array_bytes": total,
        "array_hashes_verified": True,
        "same_pass_mask_mismatch_voxels": 0,
        "storage_reproduction_status": "passed",
        "metric_status": METRIC_STATUS,
        "job_spec_sha256": job["job_spec_sha256"],
    }, records


def export_worker(job_path, candidate_id):
    job = fc.read_json(job_path)
    validate_job(job)
    c = next(c for c in job["candidates"] if c["id"] == candidate_id)
    root = Path(c["cache_root"])
    root.mkdir(parents=True, exist_ok=True)
    stages = Path(c["staging_root"])
    stages.mkdir(parents=True, exist_ok=True)
    pp = Path(c["progress_path"])
    started = time.monotonic()
    progress = {
        "job_id": job["job_id"],
        "job_spec_sha256": job["job_spec_sha256"],
        "candidate_id": candidate_id,
        "rank": c["rank"],
        "wave": c["wave"],
        "gpu": c["gpu"],
        "cases": 0,
        "findings": 0,
        "status": "smoke_running",
        "started_at_utc": fc.utc_now(),
        "metric_status": METRIC_STATUS,
        "warnings": [],
    }

    def update(**kw):
        progress.update(
            kw, updated_at_utc=fc.utc_now(), elapsed_seconds=time.monotonic() - started
        )
        fc.atomic_write_json(pp, progress)

    with exclusive(root / ".worker.lock"):
        existing = root / "export_manifest.json"
        if existing.exists():
            old = fc.read_json(existing)
            require(
                old["job_spec_sha256"] == job["job_spec_sha256"], "foreign publication"
            )
            if old["status"] == "strict_passed":
                validation, _ = validate_published(
                    job, c, load_cases(resolve(job["dataset"]["path"]))
                )
                update(
                    status="strict_passed",
                    cases=300,
                    findings=582,
                    dtype=validation["dtype"],
                    bytes=validation["array_bytes"],
                )
                return
        fc.atomic_write_json(
            existing,
            {
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "candidate_id": candidate_id,
                "cache_key": c["cache_key"],
                "dataset": job["dataset"],
                "paired_val200": c["paired_val200"],
                "status": "exporting",
            },
        )
        try:
            ex = inference_imports()
            import torch
            from analyze_candidates import require_candidate_config

            require_candidate_config(c)
            for label in ("checkpoint", "plans"):
                require(
                    fc.sha256_file(resolve(c[label]["path"])) == c[label]["sha256"],
                    f"{label} hash drift",
                )
            if c["model_spec"]["path"]:
                require(
                    fc.sha256_file(resolve(c["model_spec"]["path"]))
                    == c["model_spec"]["sha256"],
                    "model spec drift",
                )
            require(
                fc.sha256_file(Path(c["cache"]["manifest_path"]))
                == c["cache"]["manifest_sha256"],
                "preprocessing drift",
            )
            require(
                torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "worker must have exactly one GPU",
            )
            cases = load_cases(resolve(job["dataset"]["path"]))
            smokes = smoke_cases(cases, c["cache"]["root"])
            ordered = sorted(
                cases,
                key=lambda case: (
                    case["name"] not in smokes,
                    next(i for i, v in enumerate(cases) if v["name"] == case["name"]),
                ),
            )
            predictor = None
            records = []
            inference_needed = any(
                not stage_paths(c, case["name"])[1].exists() for case in cases
            )
            for case in ordered:
                require(
                    not (Path(job["runtime_root"]) / "control/ABORT").exists(),
                    "job aborted",
                )
                rec = load_stage(c, case, job["job_spec_sha256"])
                if rec is not None and case["name"] in smokes and inference_needed:
                    # Recheck the current GPU's smoke memory without replacing verified staged outputs.
                    if predictor is None:
                        predictor = ex.build_predictor(
                            c, torch.device("cuda:0"), embeddings=None
                        )
                    warmed, _ = predict_test(predictor, c, case)
                    require(
                        np.isfinite(warmed).all(), "nonfinite resumed smoke inference"
                    )
                    del warmed
                if rec is None:
                    require(
                        shutil.disk_usage(stages).free
                        >= math.prod(case["shape"]) * len(case["findings"]) * 4
                        + 1024**3,
                        "local staging full",
                    )
                    if predictor is None:
                        predictor = ex.build_predictor(
                            c, torch.device("cuda:0"), embeddings=None
                        )
                    begin = time.monotonic()
                    arr, details = predict_test(predictor, c, case)
                    require(np.isfinite(arr).all(), "nonfinite inference")
                    bounds = [float(arr.min()), float(arr.max())]
                    np.clip(arr, -30, 30, out=arr)
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    ap, rp = stage_paths(c, case["name"])
                    ex.atomic_save_npy(ap, arr)
                    rec = {
                        **details,
                        "name": case["name"],
                        "shape": list(arr.shape),
                        "dtype": "float32",
                        "job_spec_sha256": job["job_spec_sha256"],
                        "cache_key": c["cache_key"],
                        "array_sha256": ex.array_sha256(arr),
                        "unclipped_range": bounds,
                        "inference_seconds": time.monotonic() - begin,
                        "output_elements": arr.size,
                        "completed_at_utc": fc.utc_now(),
                    }
                    fc.atomic_write_json(rp, rec)
                    del arr
                records.append(rec)
                update(
                    cases=len(records),
                    findings=sum(len(r["finding_indices"]) for r in records),
                    seconds_per_case=(time.monotonic() - started) / len(records),
                    eta_seconds=(time.monotonic() - started)
                    / len(records)
                    * (300 - len(records)),
                    recent_case_costs=[
                        {
                            "seconds": r["inference_seconds"],
                            "elements": r["output_elements"],
                        }
                        for r in records[-10:]
                    ],
                )
                if len(records) == len(smokes):
                    free, _ = torch.cuda.mem_get_info()
                    require(free // 1024**2 >= 4096, "post-smoke GPU memory gate")
                    update(
                        status="smoke_waiting",
                        smoke_cases=smokes,
                        free_mib_after_smoke=free // 1024**2,
                    )
                    release = (
                        Path(job["runtime_root"])
                        / "control"
                        / f"wave_{c['wave']:02d}.continue"
                    )
                    while not release.exists():
                        require(
                            not (release.parent / "ABORT").exists(),
                            "job aborted at smoke gate",
                        )
                        time.sleep(2)
                    require(
                        fc.read_json(release)["job_spec_sha256"]
                        == job["job_spec_sha256"],
                        "foreign smoke gate",
                    )
                    update(status="exporting")
            del predictor
            torch.cuda.empty_cache()
            update(status="finalizing")
            dtype, published = publish(job, c, cases)
            update(status="validating", dtype=dtype)
            validation, published = validate_published(job, c, cases)
            fc.atomic_write_json(root / "reproduction_validation.json", validation)
            fc.atomic_write_json(
                existing,
                {
                    "job_id": job["job_id"],
                    "job_spec_sha256": job["job_spec_sha256"],
                    "candidate_id": candidate_id,
                    "cache_key": c["cache_key"],
                    "candidate_source": c,
                    "dataset": job["dataset"],
                    "paired_val200": c["paired_val200"],
                    "status": "strict_passed",
                    "metric_status": METRIC_STATUS,
                    "cases": published,
                    "case_count": 300,
                    "finding_count": 582,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at_utc": fc.utc_now(),
                },
            )
            # Only this exact source-bound local stage belongs to this worker.
            require(
                stages == Path(job["staging_root"]) / c["cache_key"],
                "unsafe staging cleanup",
            )
            cleanup_stage(stages)
            update(
                status="strict_passed",
                cases=300,
                findings=582,
                bytes=validation["array_bytes"],
                dtype=dtype,
                completed_at_utc=fc.utc_now(),
                eta_seconds=0,
            )
        except BaseException as exc:
            update(status="failed", error=str(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    export_worker(args.job, args.candidate_id)
