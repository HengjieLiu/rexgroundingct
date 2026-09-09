"""Resumable CPU rendering and packaging of the fixed val200 comparison."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import copy
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import textwrap
import time

import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
import numpy as np
from PIL import Image

import render_comparison as pilot

HERE, REPO = pilot.HERE, pilot.REPO
CATEGORIES = pilot.july.CATEGORY_NAMES
POLICIES = ("raw", "pp2_whole_lung20", "pp3_fine20_corrected_routes")
BACKGROUNDS = {"full_body": pilot.METHODS[0], "lung_only": pilot.METHODS[1]}
UPPER_RE = re.compile(r"\b(?:upper lobes of both lungs|both upper lobes|bilateral upper lobes)\b", re.I)
EXPECTED_COUNTS = dict(zip(CATEGORIES, (3, 11, 17, 6, 11, 4, 69, 49, 60, 132, 11, 0, 1, 7)))
ROLE_NAMES = {"overall_best": "Overall best", "category_best": "Category best", "iso07_best": "Iso07 best"}
WORKER_CONTEXT = None


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": pilot.july.sha256_file(path), "bytes": path.stat().st_size}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def category_dir(category):
    return "2f_empty" if category == "2f" else category


def corrected_route(original):
    """Only the explicitly approved phrase family may alter frozen PP3 routing."""
    effective = copy.deepcopy(original)
    corrected = bool(original["eligible"] and UPPER_RE.search(original["prompt"]))
    if corrected:
        effective["selected_labels"] = list(pilot.UPPER_LABELS)
        effective["fine_eligible"] = True
    return effective, corrected


def support_name(labels):
    names = {(10,): "left upper lobe", (11,): "left lower lobe", (12,): "right upper lobe",
             (13,): "right middle lobe", (14,): "right lower lobe", (10, 11): "left lung",
             (12, 13, 14): "right lung", (10, 12): "both upper lobes",
             pilot.WHOLE_LABELS: "whole lungs"}
    key = tuple(sorted(labels))
    return names.get(key, " + ".join(names[(label,)] for label in key))


def captions_for(route, corrected):
    pp2 = "PP2: whole lungs +20 mm" if route["eligible"] else "PP2: unchanged\nnot lung-eligible"
    if route["fine_eligible"]:
        pp3 = "PP3: " + support_name(route["selected_labels"]) + " +20 mm"
        pp3 = "\n".join(textwrap.wrap(pp3, width=35))
        if corrected:
            pp3 += "\ncorrected routing"
    else:
        pp3 = "PP3: unchanged\nno eligible fine support"
    return ("Raw", pp2, pp3)


def model_label(candidate):
    names = {
        "exp007_cont_e050_abs_e150": "Exp007 rel e50 / abs e150",
        "exp017_cont_e050_abs_e150": "Exp017 rel e50 / abs e150",
        "exp017_cont_e025_abs_e125": "Exp017 rel e25 / abs e125",
        "exp017_cont_e075_abs_e175": "Exp017 rel e75 / abs e175",
        "exp007_cont_e100_abs_e200": "Exp007 rel e100 / abs e200",
        "exp007_ddp_e025": "Exp007 phase 1 e25",
        "exp009_s3v2_e100": "Exp009 S3v2 e100",
        "exp011_e4d4_clip_linear_e100": "Exp011 e4/d4 linear-HU e100",
        "exp012_r02_category_2d_replay50_e100": "Exp012 2d replay50 e100",
        "exp013_category_2all_focal_target100_e100": "Exp013 2all focal e100",
    }
    return names.get(candidate, candidate)


def build_context(args, verify_checkpoints=True):
    root = args.runtime_root
    e24, e25 = root / pilot.EXP24, root / pilot.EXP25
    source_files = {
        "metadata": args.metadata_json, "fixed_val": args.val_json,
        "candidate_manifest": e24 / "config/candidate_manifest.json",
        "frozen_routes": e24 / "config/prompt_routing_val.json",
        "config24": e24 / "config/024_test_inference_anatomy_audit.json",
        "config25": REPO / "configs/experiments/025_iso07_best_anatomy_audit.json",
        "metrics24": e24 / "reports/b_validation_metrics.json",
        "metrics25": e25 / "reports/val_metrics.json",
        "anatomy_audit": root / pilot.EXP20 / "reports/private/audit_results.json",
    }
    data = {key: pilot.read_json(path) for key, path in source_files.items()}
    candidates = data["candidate_manifest"]["candidates"]
    entries = data["fixed_val"]["test"]
    metadata = {entry["name"]: entry for entry in data["metadata"]["val"]}
    if len(entries) != 200 or len(metadata) != 200 or {e["name"] for e in entries} != set(metadata):
        raise ValueError("Expected exactly the fixed 200 validation cases")
    route_rows = data["frozen_routes"]["rows"]
    routes = {(r["case"], int(r["finding_id"])): r for r in route_rows}
    expected_keys = {(name, int(fid)) for name, entry in metadata.items() for fid in entry["findings"]}
    if set(routes) != expected_keys or len(route_rows) != 381:
        raise ValueError("Frozen routes do not exactly cover 381 findings")
    counts = Counter(row["category"] for row in routes.values())
    if any(counts[c] != EXPECTED_COUNTS[c] for c in CATEGORIES):
        raise ValueError("Unexpected category census")
    corrections = []
    for key, route in routes.items():
        entry = metadata[key[0]]
        if route["prompt"] != entry["findings"][str(key[1])] or route["category"] != entry["categories"][str(key[1])]:
            raise ValueError("Metadata/frozen-route disagreement")
        effective, changed = corrected_route(route)
        if changed:
            corrections.append({"case": key[0], "finding_id": key[1], "original": route, "effective": effective})
    if len(corrections) != 15 or sum(not x["original"]["fine_eligible"] for x in corrections) != 7:
        raise ValueError("Expected exactly 15 corrections, including seven enabled fine routes")

    def path(value):
        p = Path(value)
        return root / p.relative_to(pilot.ROOT) if p.is_relative_to(pilot.ROOT) else p

    models = {key: {**value, "checkpoint": str(path(value["checkpoint"])),
                    "validation_prediction_dir": str(path(value["validation_prediction_dir"]))}
              for key, value in candidates.items()}
    iso_key = "exp017_cont_e050_abs_e150"
    iso_checkpoint = path(data["config25"]["checkpoint"])
    models[iso_key] = {"candidate_id": iso_key, "checkpoint": str(iso_checkpoint),
                       "checkpoint_sha256": data["config25"]["checkpoint_sha256"],
                       "validation_prediction_dir": str(iso_checkpoint.parents[2] / "eval_epoch050_val200/predictions")}
    overall_key = "exp007_cont_e050_abs_e150"
    audits = {r["volume_name"]: r for r in data["anatomy_audit"]["cases"]}
    jobs = []
    for index, entry in enumerate(entries):
        name = entry["name"]
        meta = metadata[name]
        audit = audits[name]
        if any(audit["geometry"][k] != "PASS" for k in ("index_grid_status", "world_header_status")):
            raise ValueError(f"{name}: anatomy geometry not verified")
        files = {"ct": pilot.ct_rate_path(name, args.ct_root), "gt": args.seg_dir / name,
                 "anatomy": path(audit["integrity"]["path"]),
                 "audit22": root / pilot.EXP22 / "cases" / (name + ".json"),
                 "overall_best_raw": Path(models[overall_key]["validation_prediction_dir"]) / name,
                 "category_best_raw": e24 / "validation/b1" / name,
                 "category_best_pp2": e24 / "validation/b2_prompt_lung20" / name,
                 "category_best_pp3_original": e24 / "validation/b3_prompt_fine20" / name,
                 "iso07_best_raw": e25 / "validation/baseline" / name,
                 "iso07_best_pp2": e25 / "validation/whole_lung20" / name,
                 "iso07_best_pp3_original": e25 / "validation/fine20" / name,
                 "iso07_source": Path(models[iso_key]["validation_prediction_dir"]) / name}
        case_models = {}
        for fid in sorted(map(int, meta["findings"])):
            cid = data["config24"]["oracle_categories"][meta["categories"][str(fid)]]
            case_models[fid] = {"overall_best": overall_key, "category_best": cid, "iso07_best": iso_key}
            files["category_source_" + cid] = Path(models[cid]["validation_prediction_dir"]) / name
        for p in files.values():
            if not p.is_file():
                raise FileNotFoundError(p)
        jobs.append({"name": name, "index": index, "entry": meta, "files": {k: str(p) for k, p in files.items()},
                     "models": case_models, "routes": {fid: routes[(name, fid)] for fid in case_models},
                     "anatomy_sha256": audit["integrity"]["observed_sha256"]})
    output = (args.output_dir or HERE / "results").resolve()
    runtime = (args.runtime_dir or pilot.july.visualization_dataset_root() / HERE.name / "full_val200_runtime").resolve()
    if runtime.is_relative_to(REPO.resolve()) or runtime == output:
        raise ValueError("Detailed runtime/derived masks must remain outside the repository and package")
    code_files = {name: HERE / name for name in ("full_gallery.py", "render_comparison.py")}
    code_files["july_renderer"] = Path(pilot.july.__file__)
    code_files["orientation_helper"] = HERE.parent / "rex_val20_coronal_viz.py"
    code_files["postprocessing"] = Path(pilot.post.__file__)
    shared = {k: file_record(p) for k, p in source_files.items()}
    code = {k: file_record(p) for k, p in code_files.items()}
    if shared["fixed_val"]["sha256"] != data["config24"]["val_manifest_sha256"]:
        raise ValueError("Fixed validation manifest hash differs from frozen Exp024 selection")
    if shared["metadata"]["sha256"] != data["config24"]["metadata_sha256"]:
        raise ValueError("Metadata hash differs from frozen Exp024 selection")
    used_models = sorted({cid for job in jobs for row in job["models"].values() for cid in row.values()})
    for cid in used_models:
        model = models[cid]
        if not Path(model["checkpoint"]).is_file():
            raise FileNotFoundError(model["checkpoint"])
        if verify_checkpoints:
            print(f"Verifying checkpoint {cid}", flush=True)
            if file_record(model["checkpoint"])["sha256"] != model["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {cid}")
    settings = {"dpi": args.dpi, "figsize": [20, 12], "window": [-600, 1500], "margin_mm": 20,
                "lung_labels": pilot.WHOLE_LABELS, "upper_labels": pilot.UPPER_LABELS,
                "correction_pattern": UPPER_RE.pattern, "background_only_restriction": True,
                "output": str(output), "runtime": str(runtime)}
    fingerprint = digest({"shared": shared, "code": code, "settings": settings,
                          "checkpoints": {cid: models[cid]["checkpoint_sha256"] for cid in used_models}})
    expected25 = {(policy, row["case"], int(row["finding_id"])): row["dice"]
                  for policy in ("baseline", "whole_lung20", "fine20") for row in data["metrics25"]["sets"][policy]["rows"]}
    ctx = {"output": str(output), "runtime": str(runtime), "settings": settings, "fingerprint": fingerprint,
           "shared": shared, "code": code, "models": {cid: models[cid] for cid in used_models},
           "corrections": corrections, "expected24": data["metrics24"]["per_case"], "expected25": expected25,
           "metadata": data["metadata"], "resume": args.resume, "checkpoints_verified": verify_checkpoints}
    return ctx, jobs


def init_worker(ctx):
    global WORKER_CONTEXT
    WORKER_CONTEXT = ctx


def resume_valid(record, fingerprint):
    if record.get("status") != "COMPLETE" or record.get("fingerprint") != fingerprint:
        return False
    findings = len(record.get("routes", {}))
    if (not findings or len(record.get("figures", [])) != findings * 2 or
            len(record.get("derived_masks", [])) != 4 or len(record.get("metrics", [])) != findings * 9):
        return False
    for item in record.get("figures", []) + record.get("derived_masks", []):
        p = Path(item["absolute_path"] if "absolute_path" in item else item["path"])
        if not p.is_file() or p.stat().st_size != item["bytes"] or pilot.july.sha256_file(p) != item["sha256"]:
            return False
    return bool(record.get("figures"))


def render_case(job):
    ctx = WORKER_CONTEXT
    started = time.monotonic()
    name, files, entry = job["name"], job["files"], job["entry"]
    case_token = f"val{job['index']:03d}_{name[:-7]}"
    runtime = Path(ctx["runtime"])
    complete_path = runtime / "cases" / (case_token + ".json")
    inputs = {key: file_record(path) for key, path in files.items()}
    fingerprint = digest({"shared": ctx["fingerprint"], "inputs": inputs, "routes": job["routes"]})
    if ctx["resume"] and complete_path.is_file():
        previous = pilot.read_json(complete_path)
        if resume_valid(previous, fingerprint):
            return {"record": str(complete_path), "case": name, "resumed": True, "seconds": time.monotonic() - started}
    audit = pilot.read_json(files["audit22"])
    for audit_key, key in (("ct", "ct"), ("gt", "gt"), ("anatomy", "anatomy"), ("pred", "overall_best_raw")):
        if inputs[key]["sha256"] != audit["hashes"][audit_key]:
            raise ValueError(f"{name}: audited input hash mismatch: {key}")
    if inputs["anatomy"]["sha256"] != job["anatomy_sha256"]:
        raise ValueError(f"{name}: official anatomy hash mismatch")
    ct = nib.load(files["ct"])
    finding_ids = sorted(map(int, entry["findings"]))
    expected_shape = (len(finding_ids), *ct.shape)
    header_records = {}
    for key, path in files.items():
        if key == "audit22":
            continue
        img = nib.load(path)
        if img.shape != (ct.shape if key in ("ct", "anatomy") else expected_shape):
            raise ValueError(f"{name}/{key}: spatial/finding shape mismatch")
        native = np.allclose(img.affine, ct.affine, atol=1e-5, rtol=0)
        legacy = np.allclose(img.affine, np.eye(4), atol=1e-5, rtol=0)
        if not native and not (key not in ("ct", "anatomy") and legacy):
            raise ValueError(f"{name}/{key}: unexpected affine")
        header_records[key] = "native CT" if native else "legacy finding-first identity; audited native index grid"
    gt = np.ascontiguousarray(pilot.load_gt(files["gt"]))
    gt_counts = {fid: int(gt[fid].sum()) for fid in finding_ids}
    anatomy = np.asarray(nib.load(files["anatomy"]).dataobj)
    transform = ornt_transform(io_orientation(ct.affine), axcodes2ornt(("R", "A", "S")))
    orient = lambda a: apply_orientation(a, transform)
    ct_data = np.asarray(ct.dataobj, dtype=np.float32)
    ct_ras = orient(ct_data)
    backgrounds = {"full_body": pilot.july.radiology_coronal_display(pilot.july.coronal_ct_projection_xz(ct_ras, "mean")),
                   "lung_only": pilot.july.radiology_coronal_display(pilot.lung_mean_projection(ct_ras, orient(np.isin(anatomy, pilot.WHOLE_LABELS))))}
    del ct_data, ct_ras
    support_cache = {}

    def support(labels):
        key = tuple(sorted(labels))
        if key not in support_cache:
            support_cache[key] = pilot.anatomy_support(anatomy, ct.affine, key)
        return support_cache[key]

    def supported(raw, route, policy):
        enabled = route["eligible"] if policy == "pp2" else route["fine_eligible"]
        if not enabled:
            return raw
        labels = pilot.WHOLE_LABELS if policy == "pp2" else route["selected_labels"]
        if not labels:
            raise ValueError(f"{name}: eligible policy with no support labels")
        return raw & support(labels)

    effective = {fid: corrected_route(job["routes"][fid]) for fid in finding_ids}
    scores, overlays = defaultdict(dict), defaultdict(dict)
    metrics, derived_records, original_pp3_metrics = [], [], []
    mask_dir = runtime / "derived_masks" / case_token
    mask_dir.mkdir(parents=True, exist_ok=True)
    for role in pilot.ROW_KEYS:
        raw = np.ascontiguousarray(pilot.load_binary(files[role + "_raw"]))
        saved2 = np.ascontiguousarray(pilot.load_binary(files[role + "_pp2"])) if role != "overall_best" else None
        saved3 = np.ascontiguousarray(pilot.load_binary(files[role + "_pp3_original"])) if role != "overall_best" else None
        if role == "category_best":
            for cid in sorted({job["models"][fid][role] for fid in finding_ids}):
                source = pilot.load_binary(files["category_source_" + cid])
                for fid in finding_ids:
                    if job["models"][fid][role] == cid and not np.array_equal(raw[fid], source[fid]):
                        raise ValueError(f"{name}/{fid}: category source-channel mismatch")
                del source
        if role == "iso07_best":
            if not np.array_equal(raw, pilot.load_binary(files["iso07_source"])):
                raise ValueError(f"{name}: iso07 source mismatch")
        derived2, derived3 = [], []
        for fid in finding_ids:
            original = job["routes"][fid]
            route, changed = effective[fid]
            pp2 = supported(raw[fid], original, "pp2")
            original3 = supported(raw[fid], original, "pp3")
            pp3 = supported(raw[fid], route, "pp3") if changed else original3
            if saved2 is not None:
                if not np.array_equal(saved2[fid], pp2) or not np.array_equal(saved3[fid], original3):
                    raise ValueError(f"{name}/{role}/{fid}: saved-policy reconstruction mismatch")
                pp2 = saved2[fid]
                if not changed:
                    pp3 = saved3[fid]
            predictions = (raw[fid], pp2, pp3)
            scores[role][fid] = []
            overlays[role][fid] = []
            for pi, prediction in enumerate(predictions):
                pred_count = int(np.count_nonzero(prediction))
                tp = int(np.count_nonzero(gt[fid] & prediction))
                dice = (2 * tp + 1e-6) / (gt_counts[fid] + pred_count + 1e-6)
                scores[role][fid].append(dice)
                overlays[role][fid].append(pilot.july.projected_error_rgba(orient(gt[fid]), orient(prediction)))
                metrics.append({"case": name, "val_index": job["index"], "finding_id": fid,
                                "category": entry["categories"][str(fid)], "row": role,
                                "model": job["models"][fid][role], "policy": POLICIES[pi],
                                "dice": dice, "hit": dice >= 0.1, "gt_voxels": gt_counts[fid],
                                "pred_voxels": pred_count, "tp": tp, "fp": pred_count - tp,
                                "fn": gt_counts[fid] - tp, "pp3_route_corrected": changed})
            original3_dice = pilot.dice_score(gt[fid], original3)
            original_pp3_metrics.append({"row": role, "finding_id": fid, "dice": original3_dice})
            if role == "overall_best":
                reference = next(r for r in audit["findings"] if int(r["finding_id"]) == fid)
                pilot.require_equal(scores[role][fid][0], reference["baseline"]["dice"], f"{name}/{fid}/overall baseline")
                if original["eligible"]:
                    pilot.require_equal(scores[role][fid][1], reference["candidates"]["all_lung:mask:20"]["dice"], f"{name}/{fid}/overall PP2")
                if name == "train_3026_a_2.nii.gz":
                    target = (0.37675890074900004, 0.24743654767197254)[fid]
                    pilot.require_equal(scores[role][fid][2], target, f"pilot corrected PP3/{fid}")
            elif role == "category_best":
                for observed, key in zip((scores[role][fid][0], scores[role][fid][1], original3_dice), ("b1", "b2_prompt_lung20", "b3_prompt_fine20")):
                    pilot.require_equal(observed, ctx["expected24"][key][name][fid], f"{name}/{fid}/{key}")
            else:
                for observed, key in zip((scores[role][fid][0], scores[role][fid][1], original3_dice), ("baseline", "whole_lung20", "fine20")):
                    pilot.require_equal(observed, ctx["expected25"][(key, name, fid)], f"{name}/{fid}/iso07/{key}")
            if role == "overall_best":
                derived2.append(pp2)
            derived3.append(pp3)
        exports = [(role + "_pp3_fine20", derived3)]
        if role == "overall_best":
            exports.append((role + "_pp2_whole_lung20", derived2))
        for label, masks in exports:
            destination = mask_dir / (label + ".nii.gz")
            temporary = destination.with_name("." + destination.name)
            pilot.save_masks(temporary, masks, ct)
            os.replace(temporary, destination)
            derived_records.append(file_record(destination) | {"finding_ids_by_channel": finding_ids})
        del raw, saved2, saved3, derived2, derived3, predictions, pp2, pp3, original3
    zooms = ct.header.get_zooms()[:3]
    spacing = [float(zooms[int(axis)]) for axis, _ in transform]
    output = Path(ctx["output"])
    figures = []
    for fid in finding_ids:
        category = entry["categories"][str(fid)]
        route, changed = effective[fid]
        rgba_gt = pilot.july.rgba_mask(pilot.july.radiology_coronal_display(pilot.july.coronal_mask_projection_xz(orient(gt[fid]))), pilot.july.GT_COLOR)
        grid = [overlays[role][fid] for role in pilot.ROW_KEYS]
        score_grid = [scores[role][fid] for role in pilot.ROW_KEYS]
        labels = [ROLE_NAMES[role] + (f" ({category})" if role == "category_best" else "") + " | " + model_label(job["models"][fid][role]) for role in pilot.ROW_KEYS]
        title = f"{case_token} · finding {fid} · {category}\n" + "\n".join(textwrap.wrap(entry["findings"][str(fid)], width=105))
        signature = None
        for background, method in BACKGROUNDS.items():
            fig, axes = pilot.create_figure(backgrounds[background], rgba_gt, grid, score_grid, labels,
                                            spacing[2] / spacing[0], title, method,
                                            captions=captions_for(route, changed),
                                            routing_note="PP3 bilateral upper-lobe correction" if changed else "Frozen prompt-selected anatomy support")
            for ri, ci in pilot.BLANK_CELLS:
                if axes[ri, ci].images or axes[ri, ci].texts or axes[ri, ci].get_title():
                    raise ValueError("Blank-cell layout violated")
            plotted = [np.asarray(axes[ri, pi + 2].images[1].get_array()) for ri in range(3) for pi in range(3)]
            for a, b in zip(plotted, [a for row in grid for a in row]):
                if not np.array_equal(a, b):
                    raise ValueError("Overlay changed during rendering")
            hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in plotted]
            if signature is not None and signature != hashes:
                raise ValueError("Background choice changed prediction overlays")
            signature = hashes
            relative = Path(category_dir(category)) / background / f"{case_token}_finding{fid}_{category}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name("." + destination.name)
            fig.savefig(temporary, dpi=ctx["settings"]["dpi"], facecolor="white", format="png")
            pilot.july.plt.close(fig)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
            info = file_record(destination)
            figures.append({**info, "path": relative.as_posix(), "absolute_path": str(destination),
                            "case": name, "val_index": job["index"], "finding_id": fid, "category": category,
                            "background": background, "prompt": entry["findings"][str(fid)], "dice": score_grid,
                            "prediction_overlay_sha256": hashes, "pp3_route_corrected": changed})
    record = {"status": "COMPLETE", "created_at_utc": now(), "case": name, "val_index": job["index"],
              "fingerprint": fingerprint, "shared_fingerprint": ctx["fingerprint"], "inputs": inputs,
              "native_shape": list(ct.shape), "native_affine": ct.affine.tolist(), "header_checks": header_records,
              "ras_transform": transform.tolist(), "spacing_ras_mm": spacing, "models": job["models"],
              "routes": {fid: {"original": job["routes"][fid], "effective": effective[fid][0], "corrected": effective[fid][1]} for fid in finding_ids},
              "metrics": metrics, "original_pp3_metrics": original_pp3_metrics, "figures": figures,
              "derived_masks": derived_records, "checks": "PASS", "elapsed_seconds": time.monotonic() - started}
    atomic_json(complete_path, record)
    return {"record": str(complete_path), "case": name, "resumed": False, "seconds": record["elapsed_seconds"]}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def metric_summary(rows, category=None):
    selected = [r for r in rows if category is None or r["category"] == category]
    result = []
    for role in pilot.ROW_KEYS:
        for policy in POLICIES:
            values = [r for r in selected if r["row"] == role and r["policy"] == policy]
            result.append({"row": role, "policy": policy, "findings": len(values),
                           "dice": sum(r["dice"] for r in values) / len(values) if values else None,
                           "hits": sum(r["hit"] for r in values)})
    return result


def summary_table(rows, category=None):
    labels = {POLICIES[0]: "Raw", POLICIES[1]: "PP2", POLICIES[2]: "PP3 (targeted corrections)"}
    return table(("Model row", "Policy", "Mean 3D Dice", "Hits / findings", "Hit rate"),
                 [(ROLE_NAMES[r["row"]], labels[r["policy"]], f"{r['dice']:.6f}" if r["dice"] is not None else "—",
                   f"{r['hits']}/{r['findings']}" if r["findings"] else "—",
                   f"{100*r['hits']/r['findings']:.2f}%" if r["findings"] else "—") for r in metric_summary(rows, category)])


COLOR_TEXT = """## Color definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP. Purple marks depth-disjoint FP and FN without voxelwise overlap.

| Color | Meaning |
| --- | --- |
| Green | Any voxelwise TP on the ray; takes priority |
| Red | FP only, without TP or FN |
| Blue | FN only, without TP or FP |
| Purple | Depth-disjoint FP + FN, without TP |
"""


def category_page(category, background, figures, rows, split_counts, split_totals):
    other = "lung_only" if background == "full_body" else "full_body"
    title = "Full-body CT mean" if background == "full_body" else "Lung-only CT mean"
    items = sorted([r for r in figures if r["category"] == category and r["background"] == background],
                   key=lambda r: (r["val_index"], r["finding_id"]))
    lines = [f"# {category} — {CATEGORIES[category]} — {title}", "",
             f"[← Results overview](../README.md) · [Other background]({other}.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)",
             "", "## Split distribution", "", table(("Split", "Finding count", "Within-split ratio"),
             [(split.title(), split_counts[split][category], f"{100*split_counts[split][category]/split_totals[split]:.2f}%") for split in ("train", "val", "test")]),
             "", "## Val200 model/policy summary", "", "A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.",
             "", summary_table(rows, category), "", COLOR_TEXT, "",
             "Lung restriction affects only the CT background. All GT/prediction voxels remain visible. PP2 follows frozen whole-lung eligibility; PP3 follows frozen side/lobe routing except the 15 approved bilateral upper-lobe corrections. Ineligible policies leave predictions unchanged.",
             "", f"## Figures ({len(items)})", ""]
    if not items:
        lines.append("There are no category-2f validation findings. This gallery intentionally contains zero PNGs.")
    for start in range(0, len(items), 10):
        batch = items[start:start + 10]
        lines += ["<details>", f"<summary>Figures {start+1}–{start+len(batch)} of {len(items)}</summary>", ""]
        for i, item in enumerate(batch, start + 1):
            stem = item["case"][:-7]
            caption = f"val{item['val_index']:03d}_{stem} · finding {item['finding_id']}"
            relative = Path(item["path"]).relative_to(category_dir(category)).as_posix()
            prompt = item["prompt"].replace("<", "&lt;").replace(">", "&gt;")
            lines += [f"### {i}. {caption}", "", prompt, "",
                      f"![{category} {caption}]({relative})", "", f"[Open full-resolution PNG]({relative})", ""]
        lines += ["</details>", ""]
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package(ctx, jobs):
    output, runtime = Path(ctx["output"]), Path(ctx["runtime"])
    case_records = [pilot.read_json(runtime / "cases" / f"val{job['index']:03d}_{job['name'][:-7]}.json") for job in jobs]
    if len(case_records) != 200 or any(r["status"] != "COMPLETE" or r["shared_fingerprint"] != ctx["fingerprint"] for r in case_records):
        raise ValueError("Cannot package an incomplete or mixed-provenance run")
    figures = sorted([r for case in case_records for r in case["figures"]], key=lambda r: (r["category"], r["val_index"], r["finding_id"], r["background"]))
    rows = sorted([r for case in case_records for r in case["metrics"]], key=lambda r: (r["val_index"], r["finding_id"], pilot.ROW_KEYS.index(r["row"]), POLICIES.index(r["policy"])))
    overall = {(r["row"], r["policy"]): r for r in metric_summary(rows)}
    targets = {("overall_best", "raw"): 0.3460234857111323,
               ("category_best", "raw"): 0.360278032575,
               ("iso07_best", "raw"): 0.33752638623854136,
               ("category_best", POLICIES[1]): 0.365156393439,
               ("iso07_best", POLICIES[1]): 0.346117179}
    for key, expected in targets.items():
        pilot.require_equal(overall[key]["dice"], expected, f"aggregate {key}", tolerance=1e-8)
    original_pp3 = [r for case in case_records for r in case["original_pp3_metrics"]]
    for role, target in (("category_best", 0.367795760985), ("iso07_best", 0.351435012)):
        values = [r["dice"] for r in original_pp3 if r["row"] == role]
        pilot.require_equal(sum(values) / len(values), target, f"original PP3 aggregate {role}")
    split_counts = {s: Counter(c for entry in ctx["metadata"][s] for c in entry["categories"].values()) for s in ("train", "val", "test")}
    split_totals = {s: sum(c.values()) for s, c in split_counts.items()}
    for category in CATEGORIES:
        folder = output / category_dir(category)
        folder.mkdir(parents=True, exist_ok=True)
        for background in BACKGROUNDS:
            (folder / (background + ".md")).write_text(category_page(category, background, figures, rows, split_counts, split_totals))
    write_csv(output / "finding_metrics.csv", rows, list(rows[0]))
    write_csv(output / "figure_index.csv", figures, ("val_index", "case", "finding_id", "category", "background", "path", "prompt", "pp3_route_corrected", "sha256", "bytes"))
    overview = ["# Full val200: whole-body and lung-only coronal comparison", "",
                "**200 cases · 381 findings · 762 figures.** Each finding has one 3×5 figure per CT background, with overall-best, category-best, and iso07-best model rows and raw/PP2/PP3 prediction columns.",
                "", "[Finding metrics](finding_metrics.csv) · [Figure index](figure_index.csv) · [Provenance](run_manifest.json) · [Renderer and usage](../README.md)",
                "", "## Val200 model/policy comparison", "", summary_table(rows), "",
                "The same full-volume 3D metrics apply to both backgrounds. A hit is Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections, not unbiased performance estimates.",
                "", "## Category galleries", "", table(("Category", "Findings", "Full body", "Lung only"),
                [(f"{c} — {CATEGORIES[c]}", EXPECTED_COUNTS[c], f"[Gallery]({category_dir(c)}/full_body.md)", f"[Gallery]({category_dir(c)}/lung_only.md)") for c in CATEGORIES]),
                "", COLOR_TEXT, "", "## Method and provenance", "",
                "Whole-body CT uses the July mean projection. Lung-only CT averages only exact lung voxels (labels 10–14), with black empty rays. Both use a −600/1500 HU window, superior-up radiology orientation, native physical aspect ratio, and unrestricted GT/prediction overlays.",
                "", "PP2 uses frozen whole-lung eligibility with a physical 20 mm margin. PP3 uses frozen fine-region routing with the approved correction for 15 bilateral upper-lobe findings, including seven newly fine-eligible findings. Corrected support is labels 10 and 12 plus 20 mm. All other routes and original experiment artifacts remain unchanged.",
                "", "PNG files are configured for Git LFS. Derived NIfTIs and detailed case records remain outside the repository. The official anatomy source retains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.",
                "", "## Regenerate or verify", "", "```bash",
                "python visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py --mode val200-by-category --workers 4 --resume",
                "python visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py --mode val200-by-category --verify-only",
                "```", ""]
    (output / "README.md").write_text("\n".join(overview))
    compact_figures = [{k: v for k, v in figure.items() if k != "absolute_path"} for figure in figures]
    manifest = {"schema_version": 1, "status": "COMPLETE", "created_at_utc": now(), "cases": 200, "findings": 381,
                "figures_count": len(figures), "shared_fingerprint": ctx["fingerprint"], "settings": ctx["settings"],
                "code": ctx["code"], "shared_inputs": ctx["shared"], "models": ctx["models"],
                "corrections": ctx["corrections"], "category_counts": EXPECTED_COUNTS,
                "figures": compact_figures, "overall_metrics": metric_summary(rows),
                "tables": {name: file_record(output / name) for name in ("finding_metrics.csv", "figure_index.csv")},
                "markdown_sha256": {str(p.relative_to(output)): pilot.july.sha256_file(p) for p in sorted(output.rglob("*.md"))},
                "anatomy_source_status": "PASS_PENDING_MANUAL_VISUAL_REVIEW", "detailed_runtime": str(runtime),
                "checks": {"checkpoint_hashes": "PASS", "source_geometry_and_hashes": "PASS", "source_composition": "PASS", "saved_policy_equivalence": "PASS", "original_metrics": "PASS", "pilot_corrected_pp3": "PASS", "overlay_invariance": "PASS", "blank_cells": "PASS", "derived_mask_roundtrip": "PASS"},
                "visual_review": "Pending deterministic sample inspection", "execution_environment": {"python": sys.version, "workers": ctx["workers"], "device": "CPU", "container_image": os.environ.get("COMPARISON_CONTAINER_IMAGE")}}
    atomic_json(output / "run_manifest.json", manifest)
    verification = verify_package(output)
    atomic_json(runtime / "completion.json", {"status": "COMPLETE", "created_at_utc": now(), "package": str(output), "verification": verification, "fingerprint": ctx["fingerprint"]})
    return verification


def verify_package(output):
    output = Path(output)
    manifest = pilot.read_json(output / "run_manifest.json")
    figures = manifest["figures"]
    if manifest["cases"] != 200 or manifest["findings"] != 381 or len(figures) != 762:
        raise ValueError("Incorrect package census")
    keys = {(r["case"], r["finding_id"], r["background"]) for r in figures}
    if len(keys) != 762 or len({r["case"] for r in figures}) != 200:
        raise ValueError("Duplicate or missing figure keys")
    expected_files = {r["path"] for r in figures}
    if expected_files != {p.relative_to(output).as_posix() for p in output.rglob("*.png")}:
        raise ValueError("PNG filesystem census differs from manifest")
    pairs = defaultdict(list)
    for figure in figures:
        p = output / figure["path"]
        if pilot.july.sha256_file(p) != figure["sha256"]:
            raise ValueError(f"PNG hash mismatch: {p}")
        with Image.open(p) as image:
            if image.size != (20 * manifest["settings"]["dpi"], 12 * manifest["settings"]["dpi"]):
                raise ValueError(f"Unexpected PNG dimensions: {p}")
            image.verify()
        pairs[(figure["case"], figure["finding_id"])].append(figure)
    for pair in pairs.values():
        if len(pair) != 2 or {r["background"] for r in pair} != set(BACKGROUNDS) or pair[0]["dice"] != pair[1]["dice"] or pair[0]["prediction_overlay_sha256"] != pair[1]["prediction_overlay_sha256"]:
            raise ValueError("Cross-background invariance/census failed")
    for c, count in EXPECTED_COUNTS.items():
        for background in BACKGROUNDS:
            if sum(r["category"] == c and r["background"] == background for r in figures) != count:
                raise ValueError(f"Category census failed: {c}/{background}")
            page = output / category_dir(c) / (background + ".md")
            content = page.read_text()
            embeds = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
            wanted = {Path(r["path"]).relative_to(category_dir(c)).as_posix() for r in figures if r["category"] == c and r["background"] == background}
            if len(embeds) != count or set(embeds) != wanted:
                raise ValueError(f"Gallery embeddings failed: {page}")
            for block in re.findall(r"<details>(.*?)</details>", content, re.S):
                if len(re.findall(r"!\[", block)) > 10:
                    raise ValueError("Gallery batch exceeds ten images")
    for relative, expected in manifest["markdown_sha256"].items():
        p = output / relative
        if pilot.july.sha256_file(p) != expected:
            raise ValueError(f"Markdown hash mismatch: {p}")
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if "://" not in link and not (p.parent / link.split("#")[0]).exists():
                raise ValueError(f"Broken relative link: {p}: {link}")
    for filename, record in manifest["tables"].items():
        if pilot.july.sha256_file(output / filename) != record["sha256"]:
            raise ValueError("Table hash mismatch")
    with (output / "finding_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metric_keys = {(r["case"], int(r["finding_id"]), r["row"], r["policy"]) for r in rows}
    wanted_keys = {(case, fid, role, policy) for case, fid in pairs for role in pilot.ROW_KEYS for policy in POLICIES}
    if len(rows) != 3429 or metric_keys != wanted_keys:
        raise ValueError("Metric census failed")
    return {"status": "PASS", "cases": 200, "findings": 381, "pngs": 762, "category_pages": 28, "metric_rows": 3429}


def main(args):
    if args.verify_only:
        print(json.dumps(verify_package(args.output_dir or HERE / "results"), indent=2))
        return
    ctx, jobs = build_context(args, verify_checkpoints=not args.dry_run)
    ctx["workers"] = args.workers
    if args.dry_run:
        print(json.dumps({"status": "PASS", "cases": len(jobs), "findings": 381, "pngs": 762,
                          "corrections": len(ctx["corrections"]), "new_fine_eligible": 7,
                          "checkpoints": len(ctx["models"]), "output": ctx["output"], "runtime": ctx["runtime"]}, indent=2))
        return
    runtime = Path(ctx["runtime"])
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".render.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(runtime / "run_config.json", {k: ctx[k] for k in ("fingerprint", "settings", "shared", "code", "models", "corrections")})
        print(f"Rendering {len(jobs)} cases on {args.workers} CPU workers", flush=True)
        completed = 0
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(ctx,)) as pool:
            futures = {pool.submit(render_case, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    print(f"[{completed}/200] {result['case']} {'resumed' if result['resumed'] else 'rendered'} ({result['seconds']:.1f}s)", flush=True)
                    atomic_json(runtime / "progress.json", {"completed": completed, "total": 200, "failures": failures, "updated_at_utc": now()})
                except Exception as error:
                    failures.append({"case": job["name"], "error": repr(error)})
                    print(f"FAILED {job['name']}: {error!r}", flush=True)
                    atomic_json(runtime / "failures.json", failures)
            if failures:
                raise RuntimeError(f"{len(failures)} cases failed; inspect {runtime / 'failures.json'} and resume after correction")
        print(json.dumps(package(ctx, jobs), indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use render_comparison.py --mode val200-by-category")
