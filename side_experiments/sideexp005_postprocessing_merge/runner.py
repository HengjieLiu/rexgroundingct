#!/usr/bin/env python3
"""Independent, restartable SideExp005 val200 postprocessing audit."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
STOP = False
CFG = JOB = None


def utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def pin(ref):
    require(sha(ref["path"]) == ref["sha256"], "source hash mismatch: " + ref["path"])


def scientific():
    global np, nib, methods
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    import nibabel as nib
    import methods


@contextlib.contextmanager
def job_lock(runtime):
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another live SideExp005 owner holds run.lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_space(cfg, fraction=1.0, free_bytes=None):
    if free_bytes is None:
        free_bytes = shutil.disk_usage(cfg["runtime_root"]).free
    need = cfg["storage"]["reserve_bytes"] + math.ceil(cfg["storage"]["headroom_bytes"] * fraction)
    require(free_bytes >= need, f"insufficient shared space: free={free_bytes}, required={need}; no files deleted")
    return free_bytes


def code_records():
    own = [p for p in ROOT.rglob("*") if p.is_file()
           and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".md", ".zip"}
           and p.name not in {"report.md", "metrics_summary.json", "launch_report.json", "verification.json"}]
    dependencies = [REPO / "scripts/rexgroundingct/run_024_test_inference_anatomy.py",
                    REPO / "side_experiments/sideexp003_ensemble_method_hub/preliminary_ensemble.py"]
    return [{"relative_path": str(p.relative_to(REPO)), "sha256": sha(p)}
            for p in sorted(set(own + dependencies))]


def job_digest(job):
    return digest({k: v for k, v in job.items() if k not in {"job_sha256", "frozen_at_utc"}})


def verify_job(cfg, job):
    require(job_digest(job) == job["job_sha256"], "job manifest hash mismatch")
    require(digest(cfg) == job["config_sha256"], "configuration differs from frozen job")
    for ref in cfg["inputs"].values():
        pin(ref)
    for ref in job["code"]:
        require(sha(REPO / ref["relative_path"]) == ref["sha256"], "frozen implementation drift: " + ref["relative_path"])
    for candidate in job["candidates"]:
        pin({"path": candidate["export_manifest_path"], "sha256": candidate["export_manifest_sha256"]})
        pin({"path": candidate["validation_path"], "sha256": candidate["validation_sha256"]})


def preflight(cfg):
    scientific()
    runtime = Path(cfg["runtime_root"])
    require(cfg["experiment"] == "sideexp005_postprocessing_merge", "experiment identity mismatch")
    require(cfg["expected"] == {"cases": 200, "findings": 381} and cfg["workers"] == 4, "fixed cohort/worker contract")
    with job_lock(runtime):
        check_space(cfg)
        for ref in cfg["inputs"].values():
            pin(ref)
        archive = read(ROOT / "frozen_sources/manifest.json")
        require(sha(ROOT / archive["archive"]) == archive["archive_sha256"], "collaborator archive drift")
        for ref in archive["files"]:
            require(sha(ROOT / "frozen_sources" / ref["path"]) == ref["sha256"], "collaborator source drift")
        from apply_semantic_constraint_best_policy import load_config
        load_config(ROOT / "frozen_sources/semantic_v1_policy.json")
        cases = read(cfg["inputs"]["dataset"]["path"])["test"]
        names = [c["name"] for c in cases]
        require(len(names) == len(set(names)) == 200, "val200 case set mismatch")
        roster = read(cfg["inputs"]["roster"]["path"])
        candidates = roster["candidates"]
        require([c["id"] for c in candidates] == cfg["candidate_ids"], "checkpoint rank order drift")
        require([c["rank"] for c in candidates] == [1, 2, 3, 4], "roster ranks drift")
        require(read(cfg["inputs"]["baseline_result"]["path"])["candidate_ids"] == cfg["candidate_ids"], "baseline checkpoint mismatch")
        exports = []
        for c in candidates:
            pin({"path": c["export_manifest_path"], "sha256": c["export_manifest_sha256"]})
            pin({"path": c["validation_path"], "sha256": c["validation_sha256"]})
            v = read(c["validation_path"])
            require(v["status"] == v["storage_reproduction_status"] == "passed" and v["array_hashes_verified"], "cache validation failed")
            require(v["cases"] == 200 and v["findings"] == 381 and v["same_pass_mask_mismatch_voxels"] == 0, "cache storage/coverage mismatch")
            e = read(c["export_manifest_path"])
            require(e["cache_key"] == c["cache_key"] and e["candidate_id"] == c["id"], "foreign cache")
            require(e["candidate_source"]["checkpoint"] == c["checkpoint"], "checkpoint identity mismatch")
            mapped = {r["name"]: r for r in e["cases"]}
            require(len(e["cases"]) == len(mapped) == 200 and set(mapped) == set(names), "cache case coverage")
            exports.append(mapped)
        audit = read(cfg["inputs"]["anatomy_audit"]["path"])
        require(audit["lut_status"] == "PASS", "official anatomy LUT gate")
        anatomy = {c["volume_name"]: c for c in audit["cases"]}
        require(set(anatomy) == set(names), "anatomy cohort mismatch")
        routing = read(cfg["inputs"]["routes"]["path"])
        require(routing["split"] == "val" and len(routing["rows"]) == 381, "wrong routing split")
        routes = {(r["case"], r["finding_id"]): r for r in routing["rows"]}
        require(len(routes) == 381, "duplicate routing entries")
        counts = collections.Counter()
        frozen = []
        for n, case in enumerate(cases, 1):
            name = case["name"]
            require(Path(name).name == name and name.endswith(".nii.gz"), "unsafe case name")
            findings = methods.ordered_findings(case)
            a = anatomy[name]
            geom = a["geometry"]
            require(geom["index_grid_status"] == geom["world_header_status"] == "PASS", "anatomy geometry audit failed")
            require(a["integrity"]["status"] == "VALID" and a["source_provenance"]["status"] == "FULL_HASH_MATCH", "anatomy provenance failed")
            mask_ref = {"path": a["integrity"]["path"], "sha256": a["integrity"]["observed_sha256"]}
            pin(mask_ref)
            ct_path = a["source_provenance"]["path"]
            ct = nib.load(ct_path)
            mask = nib.load(mask_ref["path"])
            require(list(ct.shape) == geom["ct_shape"] == list(mask.shape), "native CT/anatomy shape drift")
            require(np.allclose(ct.affine, geom["ct_affine"], atol=1e-5, rtol=0)
                    and np.allclose(mask.affine, ct.affine, atol=1e-5, rtol=0), "native CT/anatomy affine drift")
            require(set(range(10, 15)).issubset(a["labels"]["unique_label_ids"]), "missing official lung lobes")
            methods.world_axis_info(ct.affine, ct.shape)
            gt_path = Path(cfg["ground_truth_root"]) / case.get("seg_path", name)
            require(gt_path.parent == Path(cfg["ground_truth_root"]), "GT path escape")
            gt = nib.load(str(gt_path))
            shape = [len(findings), *ct.shape]
            require(list(gt.shape) == shape, "GT native index shape mismatch")
            sources = []
            for e in exports:
                r = e[name]
                require(r["shape"] == shape and r["status"] == "complete", "cache native shape/status mismatch")
                require(r["orientation"]["applied_transform"] == "nnunet_FZYX_to_reoriented_FXYZ_then_original_ct_orientation", "unverified cache orientation")
                require(r["orientation"]["ct_original_axcodes"] == list(nib.aff2axcodes(ct.affine)), "cache CT orientation drift")
                array = np.load(r["array_path"], mmap_mode="r", allow_pickle=False)
                require(list(array.shape) == shape and array.dtype.name == r["dtype"] and array.dtype.name in {"float16", "float32"}
                        and array.flags.c_contiguous, "cache array layout/dtype mismatch")
                sources.append({"path": r["array_path"], "sha256": r["array_sha256"], "dtype": r["dtype"], "bytes": array.nbytes})
                del array
            case_routes = []
            for f in findings:
                route = routes[(name, f["finding_idx"])]
                expected = methods.route_prompt(f["prompt"], f["category"])
                require(all(route[k] == v for k, v in expected.items()), "frozen d2/d3 prompt route drift")
                case_routes.append(route)
                parsed = methods.parse_finding(f["prompt"])
                counts[parsed["roi_kind"]] += 1
                counts["v2_selected"] += int(parsed["v2_selected"])
            frozen.append({"name": name, "findings": findings, "routes": case_routes,
                           "shape": shape, "affine": ct.affine.tolist(), "ct_path": ct_path,
                           "ct_source": a["source_provenance"], "anatomy": mask_ref,
                           "gt": {"path": str(gt_path), "sha256": sha(gt_path)}, "sources": sources})
            if n % 25 == 0:
                print(f"preflight: {n}/200 native input records validated", flush=True)
        require(sum(len(c["findings"]) for c in frozen) == 381, "finding coverage mismatch")
        require(dict(counts) == cfg["routing_counts"], "frozen semantic routing count mismatch")
        smoke = list(dict.fromkeys([max(frozen, key=lambda c: math.prod(c["shape"][1:]))["name"],
                                   max(frozen, key=lambda c: math.prod(c["shape"]))["name"]]))
        job = {"schema_version": 1, "run_id": cfg["run_id"], "config_sha256": digest(cfg),
               "code": code_records(), "candidates": candidates, "cases": frozen,
               "smoke_cases": smoke, "routing_counts": dict(counts), "image_id": cfg["image_id"],
               "frozen_at_utc": utc()}
        job["job_sha256"] = job_digest(job)
        dest = runtime / "job_manifest.json"
        if dest.exists():
            old = read(dest)
            require(old["job_sha256"] == job["job_sha256"], "incompatible existing run; use a new run ID")
            verify_job(cfg, old)
            return old
        for ref in job["code"]:
            source = REPO / ref["relative_path"]
            target = runtime / "source" / ref["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(sha(target) == ref["sha256"], "conflicting source snapshot")
            else:
                shutil.copyfile(source, target)
        atomic_json(runtime / "config.json", cfg)
        atomic_json(dest, job)
        atomic_json(runtime / "preflight.json", {"status": "passed", "job_sha256": job["job_sha256"],
                    "cases": 200, "findings": 381, "routing_counts": dict(counts), "checked_at_utc": utc()})
        return job


def average_logits(sources, shape, chunk, timing):
    start = time.monotonic()
    arrays = [np.load(r["path"], mmap_mode="r", allow_pickle=False) for r in sources]
    for array, ref in zip(arrays, sources):
        require(list(array.shape) == list(shape) and array.dtype.name == ref["dtype"] and array.flags.c_contiguous,
                "source array metadata changed")
    timing["read_hash_seconds"] += time.monotonic() - start
    out = np.empty(math.prod(shape), dtype=np.uint8)
    hashes = [hashlib.sha256() for _ in arrays]
    for lo in range(0, out.size, chunk):
        hi = min(lo + chunk, out.size)
        acc = np.zeros(hi - lo, dtype=np.float32)
        for array, h in zip(arrays, hashes):
            start = time.monotonic()
            values = array.reshape(-1)[lo:hi]
            h.update(memoryview(values).cast("B"))
            require(np.isfinite(values).all(), "nonfinite source logits")
            timing["read_hash_seconds"] += time.monotonic() - start
            start = time.monotonic()
            acc += methods.sigmoid_float32(values)
            timing["averaging_seconds"] += time.monotonic() - start
        out[lo:hi] = acc >= np.float32(2.0)
    for h, ref in zip(hashes, sources):
        require(h.hexdigest() == ref["sha256"], "source array hash mismatch: " + ref["path"])
    return out.reshape(shape)


def score_case(pred, gt, case, variant):
    require(pred.shape == gt.shape, "prediction/GT shape mismatch")
    rows = []
    for f in case["findings"]:
        idx = f["finding_idx"]
        truth, mask = gt[idx] > 0, pred[idx] > 0
        g, p, i = int(np.count_nonzero(truth)), int(np.count_nonzero(mask)), int(np.count_nonzero(truth & mask))
        dice = methods.dice_from_counts(g, p, i)
        rows.append({"case": case["name"], "finding_index": idx, "category": f["category"],
                     "prompt": f["prompt"], "variant": variant, "gt_voxels": g,
                     "pred_voxels": p, "intersection_voxels": i, "dice": dice, "hit": dice >= 0.1})
    return rows


def validate_nifti(path, case, expected=None):
    img = nib.load(str(path))
    require(list(img.shape) == case["shape"] and img.get_data_dtype() == np.dtype("uint8"), "prediction shape/dtype mismatch")
    require(np.allclose(img.affine, case["affine"], atol=1e-5, rtol=0), "prediction affine mismatch")
    ct = nib.load(case["ct_path"])
    for getter in ("get_qform", "get_sform"):
        ours, code = getattr(img, getter)(coded=True)
        target, target_code = getattr(ct, getter)(coded=True)
        require(int(code) == int(target_code), "prediction header form-code mismatch")
        if code:
            require(np.allclose(ours, target, atol=1e-5, rtol=0), "prediction header form mismatch")
    if expected is not None:
        data = np.asanyarray(img.dataobj)
        require(np.isin(data, [0, 1]).all() and np.array_equal(data, expected), "prediction publication value mismatch")
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size,
            "shape": list(img.shape), "affine": img.affine.tolist(), "dtype": "uint8"}


def artifact_paths(runtime, case, variant):
    directory = Path(runtime) / "predictions" / variant
    return directory / case["name"], directory / (".sideexp005." + case["name"])


def journal_path(runtime, case, phase):
    return Path(runtime) / "progress" / phase / (case["name"] + ".json")


def recover_case(cfg, job, case, phase):
    path = journal_path(cfg["runtime_root"], case, phase)
    variants = ["d1"] if phase == "baseline" else list(methods.VARIANTS[1:])
    if not path.exists():
        for variant in variants:
            dest, _ = artifact_paths(cfg["runtime_root"], case, variant)
            require(not dest.exists(), "conflicting unowned prediction: " + str(dest))
        return None
    record = read(path)
    require(record["job_sha256"] == job["job_sha256"] and record["case"] == case["name"]
            and record["phase"] == phase and record["status"] in {"prepared", "complete"}, "incompatible case journal")
    require(set(record["artifacts"]) == set(variants), "case artifact coverage mismatch")
    for variant in variants:
        dest, temp = artifact_paths(cfg["runtime_root"], case, variant)
        r = record["artifacts"][variant]
        require(r["path"] == str(dest) and r["temporary_path"] == str(temp), "artifact path mismatch")
        source = dest if dest.exists() else temp
        require(source.exists(), "journal output missing")
        validated = validate_nifti(source, case)
        require(validated["sha256"] == r["sha256"], "published output hash mismatch")
        if source == temp:
            os.replace(temp, dest)
        elif temp.exists():
            require(sha(temp) == r["sha256"], "conflicting owned temporary file")
            temp.unlink()
    if record["status"] != "complete":
        record.update(status="complete", completed_at_utc=utc())
        atomic_json(path, record)
    return record


def worker_init(cfg, job):
    global CFG, JOB
    CFG, JOB = cfg, job
    scientific()


def case_worker(task):
    phase, case = task
    recovered = recover_case(CFG, JOB, case, phase)
    if recovered is not None:
        return recovered
    started = time.monotonic()
    timing = {k: 0.0 for k in ("read_hash_seconds", "averaging_seconds", "anatomy_support_seconds",
                               "nifti_write_seconds", "scoring_seconds", "validation_seconds")}
    details = []
    if phase == "baseline":
        outputs = {"d1": average_logits(case["sources"], case["shape"], CFG["chunk_elements"], timing)}
    else:
        baseline = recover_case(CFG, JOB, case, "baseline")
        require(baseline is not None, "baseline case not complete")
        start = time.monotonic()
        raw = np.asanyarray(nib.load(baseline["artifacts"]["d1"]["path"]).dataobj)
        pin(case["anatomy"])
        img = nib.load(case["anatomy"]["path"])
        anatomy = np.asanyarray(img.dataobj)
        require(list(anatomy.shape) == case["shape"][1:]
                and np.allclose(img.affine, case["affine"], atol=1e-5, rtol=0), "anatomy drift")
        timing["read_hash_seconds"] += time.monotonic() - start
        start = time.monotonic()
        outputs, details = methods.postprocess(raw, anatomy, np.asarray(case["affine"]), case["findings"], case["routes"])
        timing["anatomy_support_seconds"] += time.monotonic() - start
        start = time.monotonic()
        for variant, arr in outputs.items():
            require(not np.any((arr > 0) & (raw == 0)), "postprocessor added foreground: " + variant)
        require(not np.any((outputs["d12"] > 0) & (outputs["d11"] == 0)), "d12 is not a subset of d11")
        for d in details:
            idx = d["finding_idx"]
            if not d["collaborator"]["v2_selected"]:
                require(np.array_equal(outputs["d11"][idx], outputs["d12"][idx]), "unselected v2 finding changed")
        timing["validation_seconds"] += time.monotonic() - start
        del raw, anatomy
    start = time.monotonic()
    pin(case["gt"])
    gt = np.asanyarray(nib.load(case["gt"]["path"]).dataobj)
    timing["read_hash_seconds"] += time.monotonic() - start
    start = time.monotonic()
    rows = [r for variant, arr in outputs.items() for r in score_case(arr, gt, case, variant)]
    timing["scoring_seconds"] += time.monotonic() - start
    del gt
    artifacts = {}
    ct = nib.load(case["ct_path"])
    for variant, arr in outputs.items():
        dest, temp = artifact_paths(CFG["runtime_root"], case, variant)
        require(not dest.exists(), "refusing to overwrite an existing prediction")
        dest.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        header = ct.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(arr, ct.affine, header), str(temp))
        timing["nifti_write_seconds"] += time.monotonic() - start
        start = time.monotonic()
        rec = validate_nifti(temp, case, expected=arr)
        timing["validation_seconds"] += time.monotonic() - start
        artifacts[variant] = {**rec, "path": str(dest), "temporary_path": str(temp), "variant": variant}
    timing["wall_seconds"] = time.monotonic() - started
    record = {"status": "prepared", "phase": phase, "case": case["name"], "job_sha256": JOB["job_sha256"],
              "artifacts": artifacts, "rows": rows, "routing": details, "timings": timing,
              "elements": math.prod(case["shape"]), "prepared_at_utc": utc()}
    path = journal_path(CFG["runtime_root"], case, phase)
    atomic_json(path, record)
    for r in artifacts.values():
        os.replace(r["temporary_path"], r["path"])
    record.update(status="complete", completed_at_utc=utc())
    atomic_json(path, record)
    return record


def metric_summary(rows):
    result = methods.summarize_rows(rows)
    cases = collections.defaultdict(list)
    for row in rows:
        cases[row["case"]].append(row["dice"])
    result["case_wise_dice"] = math.fsum(math.fsum(v) / len(v) for v in cases.values()) / len(cases)
    result["cases"] = len(cases)
    return result


def baseline_gate(cfg, records):
    rows = [row for record in records for row in record["rows"]]
    actual = metric_summary(rows)
    ref = read(cfg["inputs"]["baseline_result"]["path"])["metrics"]
    require(actual["cases"] == 200 and actual["findings"] == ref["findings"] == 381, "baseline coverage")
    for a, b in [(actual, ref)] + [(actual["categories"][c], ref["categories"][c]) for c in methods.CATEGORIES]:
        require(a["findings"] == b["findings"] and a["hits"] == b["hits"], "baseline category/hit mismatch")
        require(a["dice"] is b["dice"] if a["dice"] is None or b["dice"] is None else abs(a["dice"] - b["dice"]) <= cfg["baseline"]["tolerance"], "baseline Dice mismatch")
    require(actual["hits"] == cfg["baseline"]["hits"] and abs(actual["dice"] - cfg["baseline"]["dice"]) <= cfg["baseline"]["tolerance"], "baseline anchor mismatch")
    return actual


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def run_phase(cfg, job, phase):
    runtime = Path(cfg["runtime_root"])
    records = []
    remaining = []
    for case in job["cases"]:
        existing = recover_case(cfg, job, case, phase)
        if existing is None:
            remaining.append(case)
        else:
            records.append(existing)
    if not remaining:
        return records
    started = time.monotonic()
    fresh = []
    total_elements = sum(math.prod(c["shape"]) for c in job["cases"])
    last_report = 0.0
    timing_path = runtime / (phase + "_timings.json")
    previous = read(timing_path) if timing_path.exists() else {}
    previous_wall = previous.get("wall_seconds", 0.0)
    attempts = previous.get("attempts", 0) + 1

    def persist_timing(status):
        atomic_json(timing_path, {"wall_seconds": previous_wall + time.monotonic() - started,
                    "status": status, "attempts": attempts,
                    "cases_reused_on_last_attempt": len(records) - len(fresh),
                    "cases_processed_on_last_attempt": len(fresh), "updated_at_utc": utc(),
                    "note": "Cumulative supervisor wall time; hard-kill recovery may omit up to one heartbeat interval."})

    def status():
        nonlocal last_report
        completed_elements = sum(r["elements"] for r in records)
        eta = None
        if len(fresh) >= 10:
            done = sum(r["elements"] for r in fresh)
            eta = (total_elements - completed_elements) / done * (time.monotonic() - started)
        value = {"run_id": cfg["run_id"], "job_sha256": job["job_sha256"], "status": "running",
                 "phase": phase, "cases_complete": len(records), "cases_total": 200,
                 "findings_complete": sum(len(r["rows"]) // (1 if phase == "baseline" else 4) for r in records),
                 "eta_seconds": eta, "updated_at_utc": utc(), "pid": os.getpid(), "workers": cfg["workers"],
                 "shared_free_bytes": shutil.disk_usage(runtime).free,
                 "phase_wall_seconds": time.monotonic() - started}
        atomic_json(runtime / "state.json", value)
        persist_timing("running")
        print(json.dumps(value), flush=True)
        last_report = time.monotonic()

    groups = [[c for c in remaining if c["name"] in job["smoke_cases"]],
              [c for c in remaining if c["name"] not in job["smoke_cases"]]]
    with cf.ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=mp.get_context("spawn"),
                                initializer=worker_init, initargs=(cfg, job)) as pool:
        for group in groups:
            tasks, pending = iter(group), set()
            exhausted = False
            while pending or not exhausted:
                while not STOP and len(pending) < cfg["workers"] and not exhausted:
                    try:
                        case = next(tasks)
                    except StopIteration:
                        exhausted = True
                        break
                    check_space(cfg)
                    pending.add(pool.submit(case_worker, (phase, case)))
                if STOP:
                    exhausted = True
                if time.monotonic() - last_report >= 60:
                    status()
                if pending:
                    done, pending = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        record = future.result()
                        records.append(record)
                        fresh.append(record)
                if STOP and not pending:
                    raise InterruptedError("stop requested; active cases completed and dispatch stopped")
    status()
    persist_timing("complete")
    return records


def compare_rows(rows, variant, baseline="d1"):
    indexed = {(r["variant"], r["case"], r["finding_index"]): r for r in rows}
    details = []
    for r in rows:
        if r["variant"] != variant:
            continue
        b = indexed[(baseline, r["case"], r["finding_index"])]
        delta = r["dice"] - b["dice"]
        tp = b["intersection_voxels"] - r["intersection_voxels"]
        fp = (b["pred_voxels"] - b["intersection_voxels"]) - (r["pred_voxels"] - r["intersection_voxels"])
        require(tp >= 0 and fp >= 0, "subset count invariant failed")
        details.append({"case": r["case"], "finding_index": r["finding_index"], "category": r["category"],
                        "prompt": r["prompt"], "variant": variant, "baseline": baseline,
                        "baseline_dice": b["dice"], "dice": r["dice"], "delta_dice": delta,
                        "change": "improved" if delta > 1e-12 else "decreased" if delta < -1e-12 else "unchanged",
                        "hit_gain": int(r["hit"] and not b["hit"]), "hit_loss": int(b["hit"] and not r["hit"]),
                        "tp_removed": tp, "fp_removed": fp,
                        "new_empty": int(b["pred_voxels"] > 0 and r["pred_voxels"] == 0)})

    def aggregate(selected):
        return {"findings": len(selected), "mean_delta_dice": math.fsum(r["delta_dice"] for r in selected) / len(selected) if selected else None,
                **{k: sum(r["change"] == k for r in selected) for k in ("improved", "decreased", "unchanged")},
                **{k: sum(r[k] for r in selected) for k in ("hit_gain", "hit_loss", "tp_removed", "fp_removed", "new_empty")}}
    summary = {"overall": aggregate(details), "categories": {c: aggregate([r for r in details if r["category"] == c]) for c in methods.CATEGORIES}}
    for r in [summary["overall"], *summary["categories"].values()]:
        require(r["improved"] + r["decreased"] + r["unchanged"] == r["findings"], "change-count invariant")
    return summary, details


def report(cfg, job, records=None):
    scientific()
    runtime = Path(cfg["runtime_root"])
    if records is None:
        records = []
        for phase in ("baseline", "postprocessing"):
            for c in job["cases"]:
                r = recover_case(cfg, job, c, phase)
                require(r is not None, "run is incomplete")
                records.append(r)
    rows = [dict(r) for record in records for r in record["rows"]]
    require(len(rows) == len({(r["variant"], r["case"], r["finding_index"]) for r in rows}) == 1905, "result coverage mismatch")
    baseline_gate(cfg, [r for r in records if r["phase"] == "baseline"])
    routing_rows = [{"case": rec["case"], "finding_index": r["finding_idx"], **r}
                    for rec in records for r in rec["routing"]]
    routing_index = {(r["case"], r["finding_index"]): r for r in routing_rows}
    for row in rows:
        variant = row["variant"]
        if variant == "d1":
            row["postprocessing_route"] = {"method": "raw_probability_ensemble"}
        else:
            route = routing_index[(row["case"], row["finding_index"])]
            row["postprocessing_route"] = route["ours"] if variant in {"d2", "d3"} else route["collaborator"]
    rows.sort(key=lambda r: (r["case"], r["finding_index"], methods.VARIANTS.index(r["variant"])))
    summaries = {v: metric_summary([r for r in rows if r["variant"] == v]) for v in methods.VARIANTS}
    changes, details = {}, []
    for v, base in [(v, "d1") for v in methods.VARIANTS[1:]] + [("d12", "d11")]:
        key = v + "_vs_" + base
        changes[key], data = compare_rows(rows, v, base)
        details.extend(data)
    inventories = {v: [] for v in methods.VARIANTS}
    for rec in records:
        for v, artifact in rec["artifacts"].items():
            pin(artifact)
            inventories[v].append(artifact)
    names = {c["name"] for c in job["cases"]}
    for v, files in inventories.items():
        require(len(files) == 200 and {Path(f["path"]).name for f in files} == names, "inventory case coverage")
        require({p.name for p in (runtime / "predictions" / v).glob("*.nii.gz")} == names, "unexpected prediction files")
        atomic_json(runtime / "manifests" / (v + ".json"), {"cases": 200, "findings": 381, "files": files})
    timings = {k: math.fsum(r["timings"][k] for r in records) for k in records[0]["timings"]}
    summary = {"status": "complete", "experiment": cfg["experiment"], "run_id": cfg["run_id"],
               "job_sha256": job["job_sha256"], "completed_at_utc": utc(), "metrics": summaries,
               "changes": changes, "timings_worker_seconds": timings,
               "phase_timings": {p: read(runtime / (p + "_timings.json")) for p in ("baseline", "postprocessing")},
               "output_bytes": sum(r["bytes"] for files in inventories.values() for r in files),
               "source_status": cfg["source_status"], "test_generation": cfg["test_generation"],
               "runtime_root": str(runtime), "interpretation": "Fixed full-val200 diagnostic; no method retuning or test-label evaluation."}
    reports = runtime / "reports"
    atomic_json(reports / "per_finding.json", rows)
    atomic_json(reports / "changes.json", details)
    atomic_json(reports / "routing.json", routing_rows)
    for filename, data in (("per_finding.csv", rows), ("changes.csv", details)):
        with (reports / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in data)
    fmt = lambda x: "NA" if x is None else f"{x:.9f}"
    lines = ["# SideExp005: frozen val200 postprocessing comparison", "",
             "200 CTs / 381 findings. All variants share the frozen top-four probability ensemble and official CT-RATE anatomy.", "",
             "| Variant | Finding Dice | Case Dice | Hits / 381 | HIT rate |", "|---|---:|---:|---:|---:|"]
    for v, m in summaries.items():
        lines.append(f"| {v} | {fmt(m['dice'])} | {fmt(m['case_wise_dice'])} | {m['hits']} | {fmt(m['hit_rate'])} |")
    lines += ["", "## All official categories", "", "Category Dice is the mean over findings in that category; HIT means finding Dice >= 0.1.", "",
              "| Category | Method | n | Dice | Hits | HIT rate |", "|---|---|---:|---:|---:|---:|"]
    for c in methods.CATEGORIES:
        for v in methods.VARIANTS:
            m = summaries[v]["categories"][c]
            lines.append(f"| {c} — {methods.CATEGORY_NAMES[c]} | {v} | {m['findings']} | {fmt(m['dice'])} | {m['hits']} | {fmt(m['hit_rate'])} |")
    for comparison, table in changes.items():
        lines += ["", "## " + comparison.replace("_", " "), "", "| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gained | HIT lost | TP removed | FP removed | New empty |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c, m in [("overall", table["overall"]), *table["categories"].items()]:
            lines.append(f"| {c} | {m['findings']} | {m['improved']} | {m['decreased']} | {m['unchanged']} | {fmt(m['mean_delta_dice'])} | {m['hit_gain']} | {m['hit_loss']} | {m['tp_removed']} | {m['fp_removed']} | {m['new_empty']} |")
    lines += ["", "## Provenance and timing", "", f"Run identity: `{job['job_sha256']}`. Image: `{cfg['image_id']}`.",
              "d2/d3 derive independently from d1; d11 is collaborator semantic-v1; d12 is d11 followed by frozen strict-v2.",
              f"Anatomy review status: `{cfg['source_status']}`. No test generation or upload is authorized by this completion.",
              f"Actual compressed prediction storage: {summary['output_bytes'] / 2**30:.3f} GiB.", "",
              "| Phase | Wall minutes |", "|---|---:|"]
    for p, t in summary["phase_timings"].items():
        lines.append(f"| {p} | {t['wall_seconds'] / 60:.2f} |")
    lines += ["", "Worker-phase seconds (summed across processes):", "", "```json", json.dumps(timings, indent=2), "```", "",
              f"Detailed private finding rows, routing and ranked changes: `{reports}`.",
              "Regenerate with the frozen `source/side_experiments/sideexp005_postprocessing_merge/runner.py report` in the recorded image.", ""]
    (reports / "report.md").write_text("\n".join(lines))
    private = ["# Finding improvements and regressions", ""]
    for comparison in changes:
        v, b = comparison.split("_vs_")
        selected = [r for r in details if r["variant"] == v and r["baseline"] == b]
        for label, reverse in (("Largest regressions", False), ("Largest improvements", True)):
            private += ["## " + comparison + ": " + label, "", "| Case | Finding | Category | ΔDice | Prompt |", "|---|---:|---|---:|---|"]
            changed = [r for r in selected if r["delta_dice"] > 1e-12] if reverse else [r for r in selected if r["delta_dice"] < -1e-12]
            for r in sorted(changed, key=lambda r: r["delta_dice"], reverse=reverse)[:20]:
                private.append(f"| {r['case']} | {r['finding_index']} | {r['category']} | {r['delta_dice']:+.9f} | {r['prompt'].replace('|', '/')} |")
            if not changed:
                private.append("| None | — | — | — | No findings in this direction |")
            private.append("")
    (reports / "ranked_changes.md").write_text("\n".join(private))
    atomic_json(reports / "summary.json", summary)
    atomic_json(runtime / "completion.json", {**summary, "files": 1000, "finding_evaluations": 1905,
                "report_sha256": sha(reports / "report.md"), "inventory_hashes": {v: sha(runtime / "manifests" / (v + ".json")) for v in methods.VARIANTS}})
    return summary


def run(cfg):
    scientific()
    runtime = Path(cfg["runtime_root"])
    with job_lock(runtime):
        job = read(runtime / "job_manifest.json")
        verify_job(cfg, job)
        require(read(runtime / "preflight.json")["job_sha256"] == job["job_sha256"], "preflight gate mismatch")
        if (runtime / "completion.json").exists():
            report(cfg, job)
            return
        check_space(cfg)
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        atomic_json(runtime / "owner.json", {"pid": os.getpid(), "hostname": os.uname().nodename,
                    "job_sha256": job["job_sha256"], "started_at_utc": utc(), "argv": sys.argv,
                    "python": sys.version, "environment": {k: os.environ.get(k) for k in cfg["environment"]}})
        try:
            baseline = run_phase(cfg, job, "baseline")
            metric = baseline_gate(cfg, baseline)
            atomic_json(runtime / "baseline_gate.json", {"status": "passed", "metrics": metric, "job_sha256": job["job_sha256"]})
            processed = run_phase(cfg, job, "postprocessing")
            result = report(cfg, job, baseline + processed)
            atomic_json(runtime / "state.json", {"status": "complete", "run_id": cfg["run_id"],
                        "job_sha256": job["job_sha256"], "cases_complete": 200, "files": 1000,
                        "finding_evaluations": 1905, "updated_at_utc": utc(), "metrics": result["metrics"]})
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed"
            atomic_json(runtime / "state.json", {"status": status, "run_id": cfg["run_id"], "job_sha256": job["job_sha256"],
                        "error": str(exc), "traceback": traceback.format_exc(), "updated_at_utc": utc()})
            raise


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "run", "watch", "report"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)
    cfg = read(args.config)
    runtime = Path(cfg["runtime_root"])
    if args.command == "preflight":
        job = preflight(cfg)
        print(json.dumps({"status": "passed", "job_sha256": job["job_sha256"], "cases": len(job["cases"])}))
    elif args.command == "run":
        logs = runtime / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "supervisor.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                run(cfg)
    elif args.command == "report":
        scientific()
        with job_lock(runtime):
            job = read(runtime / "job_manifest.json")
            verify_job(cfg, job)
            print(json.dumps(report(cfg, job)["metrics"], indent=2))
    else:
        while True:
            path = runtime / "state.json"
            state = read(path) if path.exists() else {"status": "not_started"}
            print(json.dumps(state, indent=2), flush=True)
            if args.once or state.get("status") in {"complete", "failed", "interrupted"}:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
