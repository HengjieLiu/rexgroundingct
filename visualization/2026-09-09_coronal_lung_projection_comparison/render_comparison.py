#!/usr/bin/env python3
"""Render a frozen-model, single-case comparison with corrected upper-lobe PP3."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


july = import_file("july_coronal_projection", HERE.parent / "2026-07-31_visualdev_coronal_projection/coronal_projection.py")
post = import_file("comparison_exp024", REPO / "scripts/rexgroundingct/run_024_test_inference_anatomy.py")
from rex_val20_coronal_viz import ct_rate_path, dice_score  # noqa: E402

ROOT = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct")
EXP20 = "020_ct_rate_ts_total_rex_val200_audit"
EXP22 = "022_exp007_official_anatomy_val200_audit"
EXP24 = "024_test_inference_anatomy_audit"
EXP25 = "025_iso07_best_anatomy_audit"
WHOLE_LABELS = (10, 11, 12, 13, 14)
UPPER_LABELS = (10, 12)
BLANK_CELLS = ((1, 0), (1, 1), (2, 0), (2, 1))
POLICIES = ("raw", "pp2_whole_lung20", "pp3_upper_lobes20_corrected")
METHODS = ("whole_volume_mean", "lung_only_mean")
ROW_KEYS = ("overall_best", "category_best", "iso07_best")


def read_json(path):
    return json.loads(Path(path).read_text())


def lung_mean_projection(ct_ras, lung_ras):
    """Window first; average only eligible voxels, excluding empty rays."""
    if ct_ras.ndim != 3 or ct_ras.shape != lung_ras.shape:
        raise ValueError("CT and lung mask must share a 3D grid")
    windowed = july.normalized_lung_window(ct_ras)
    count = np.count_nonzero(lung_ras, axis=1)
    total = np.sum(windowed, axis=1, where=lung_ras.astype(bool))
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def anatomy_support(anatomy, affine, labels):
    return post.physical_dilation(np.isin(anatomy, labels), affine, 20.0)


def normalize_case(case):
    match = re.fullmatch(r"(?:val(\d+)_)?(train_\d+_[a-z]+_\d+)(?:\.nii\.gz)?", case)
    if not match:
        raise ValueError(f"Invalid case identifier: {case}")
    return match[2] + ".nii.gz", int(match[1]) if match[1] else None


def require_equal(actual, expected, label, tolerance=1e-8):
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise ValueError(f"{label}: observed {actual}, expected {expected}")


def prepare(args):
    name, requested_index = normalize_case(args.case)
    val = read_json(args.val_json)["test"]
    index = next(i for i, row in enumerate(val) if row["name"] == name)
    if requested_index is not None and requested_index != index:
        raise ValueError("Case name and fixed validation index disagree")
    entry = next(row for row in read_json(args.metadata_json)["val"] if row["name"] == name)
    finding_ids = args.findings if args.findings is not None else sorted(map(int, entry["findings"]))
    if not finding_ids or len(finding_ids) != len(set(finding_ids)):
        raise ValueError("Select distinct, existing finding IDs")
    for fid in finding_ids:
        prompt = entry["findings"].get(str(fid), "")
        if not re.search(r"upper lobes of both lungs", prompt, re.IGNORECASE):
            raise ValueError(f"Finding {fid} does not meet the explicit bilateral upper-lobe pilot contract")

    def path(value):
        p = Path(value)
        return args.runtime_root / p.relative_to(ROOT) if p.is_relative_to(ROOT) else p

    e24, e25 = args.runtime_root / EXP24, args.runtime_root / EXP25
    provenance = {
        "candidate_manifest": e24 / "config/candidate_manifest.json",
        "frozen_routes": e24 / "config/prompt_routing_val.json",
        "exp024_config": e24 / "config/024_test_inference_anatomy_audit.json",
        "exp025_config": REPO / "configs/experiments/025_iso07_best_anatomy_audit.json",
        "anatomy_audit": args.runtime_root / EXP20 / "reports/private/audit_results.json",
        "exp022_case": args.runtime_root / EXP22 / "cases" / (name + ".json"),
        "exp024_metrics": e24 / "reports/b_validation_metrics.json",
        "exp025_metrics": e25 / "reports/val_metrics.json",
        "metadata": args.metadata_json,
        "fixed_val": args.val_json,
    }
    candidates = read_json(provenance["candidate_manifest"])["candidates"]
    cfg24, cfg25 = read_json(provenance["exp024_config"]), read_json(provenance["exp025_config"])
    routes = {int(r["finding_id"]): r for r in read_json(provenance["frozen_routes"])["rows"] if r["case"] == name}
    for fid in finding_ids:
        route = routes[fid]
        if (not route["eligible"] or not route["fine_eligible"] or
                route["selected_labels"] != list(WHOLE_LABELS) or
                route["prompt"] != entry["findings"][str(fid)] or
                route["category"] != entry["categories"][str(fid)]):
            raise ValueError("Frozen routing differs from this pilot's documented original behavior")
    audit = next(row for row in read_json(provenance["anatomy_audit"])["cases"] if row["volume_name"] == name)
    for key in ("index_grid_status", "world_header_status"):
        if audit["geometry"][key] != "PASS":
            raise ValueError(f"Anatomy {key} failed")
    files = {
        "ct": ct_rate_path(name, args.ct_root), "gt": args.seg_dir / name,
        "anatomy": path(audit["integrity"]["path"]),
        "overall_best_raw": path(candidates["exp007_cont_e050_abs_e150"]["validation_prediction_dir"]) / name,
        "category_best_raw": e24 / "validation/b1" / name,
        "category_best_pp2": e24 / "validation/b2_prompt_lung20" / name,
        "category_best_original_pp3": e24 / "validation/b3_prompt_fine20" / name,
        "iso07_best_raw": e25 / "validation/baseline" / name,
        "iso07_best_pp2": e25 / "validation/whole_lung20" / name,
        "iso07_best_original_pp3": e25 / "validation/fine20" / name,
    }
    models = {}
    for fid in finding_ids:
        cat = entry["categories"][str(fid)]
        category_id = cfg24["oracle_categories"][cat]
        overall = candidates["exp007_cont_e050_abs_e150"]
        category = candidates[category_id]
        models[fid] = {
            "overall_best": {"candidate_id": overall["candidate_id"], "checkpoint": str(path(overall["checkpoint"])), "checkpoint_sha256": overall["checkpoint_sha256"], "label": "Overall best | Exp007 rel e50 / abs e150", "selection_dice": 0.3460234857111323, "selection_population": "full val200"},
            "category_best": {"candidate_id": category_id, "checkpoint": str(path(category["checkpoint"])), "checkpoint_sha256": category["checkpoint_sha256"], "label": f"Category best ({cat}) | " + ("Exp017 rel e75 / abs e175" if category_id == "exp017_cont_e075_abs_e175" else "Exp009 S3v2 e100" if category_id == "exp009_s3v2_e100" else category_id), "selection_dice": read_json(provenance["exp024_metrics"])["category_tables"]["b1"][cat]["dice"], "selection_population": f"val200 category {cat}"},
            "iso07_best": {"candidate_id": "exp017_cont_e050_abs_e150", "checkpoint": str(path(cfg25["checkpoint"])), "checkpoint_sha256": cfg25["checkpoint_sha256"], "label": "Iso07 best | Exp017 rel e50 / abs e150", "selection_dice": 0.33752638623854136, "selection_population": "full val200, iso07 candidates"},
        }
        files[f"category_source_{fid}"] = path(category["validation_prediction_dir"]) / name
    files["iso07_source"] = path(cfg25["checkpoint"]).parents[2] / "eval_epoch050_val200/predictions" / name
    # checkpoint parents[2] is the ddp_bs4 run root.
    for p in list(files.values()) + list(provenance.values()) + [Path(m["checkpoint"]) for row in models.values() for m in row.values()]:
        if not p.is_file():
            raise FileNotFoundError(p)
    ct = nib.load(files["ct"])
    count = len(entry["findings"])
    headers = {}
    for key, p in files.items():
        img = nib.load(p)
        expected = ct.shape if key in ("ct", "anatomy") else (count, *ct.shape)
        if img.shape != expected:
            raise ValueError(f"{key}: shape {img.shape} != {expected}")
        native = np.allclose(img.affine, ct.affine, atol=1e-5, rtol=0)
        legacy = np.allclose(img.affine, np.eye(4), atol=1e-5, rtol=0)
        if not native and not (key not in ("ct", "anatomy") and legacy):
            raise ValueError(f"{key}: unexpected affine")
        headers[key] = {"shape": list(img.shape), "affine": img.affine.tolist(), "geometry": "native CT" if native else "legacy finding-first identity; native index grid verified by audit/source equality"}
    token = f"val{index:03d}_{name[:-7]}"
    output = args.output_dir or july.visualization_dataset_root() / HERE.name / token
    return dict(name=name, index=index, entry=entry, finding_ids=finding_ids, files=files, provenance=provenance,
                models=models, routes=routes, audit=audit, headers=headers, output=output, token=token)


def load_binary(path):
    data = np.asarray(nib.load(path).dataobj)
    if not np.all((data == 0) | (data == 1)):
        raise ValueError(f"Nonbinary input: {path}")
    return data.astype(bool)


def load_gt(path):
    """Released GT channels use positive instance IDs; their union is target."""
    data = np.asarray(nib.load(path).dataobj)
    if not np.all(np.isfinite(data)) or np.any(data < 0) or not np.all(data == np.floor(data)):
        raise ValueError(f"Invalid GT instance labels: {path}")
    return data > 0


def create_figure(background, gt_overlay, overlays, scores, labels, aspect, title, method,
                  captions=None, routing_note="PP3 routing corrected for this pilot"):
    fig, axes = july.plt.subplots(3, 5, figsize=(20, 12), squeeze=False)
    title_shift = max(0, len(title.splitlines()) - 2) * 0.025
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.105, top=0.825 - title_shift, wspace=0.08, hspace=0.37)
    captions = captions or ("Raw", "PP2: whole lungs +20 mm", "PP3: upper lobes +20 mm\ncorrected routing")
    for ri in range(3):
        for ci in range(5):
            ax = axes[ri, ci]
            ax.set_axis_off()
            if (ri, ci) in BLANK_CELLS:
                continue
            ax.imshow(background, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
            if ci == 1:
                ax.imshow(gt_overlay, aspect=aspect, interpolation="nearest")
                ax.set_title("GT mask projection", fontsize=11)
            elif ci >= 2:
                ax.imshow(overlays[ri][ci - 2], aspect=aspect, interpolation="nearest")
                ax.set_title(captions[ci - 2], fontsize=10)
                ax.text(0.03, 0.055, f"3D Dice {scores[ri][ci - 2]:.4f}", transform=ax.transAxes,
                        color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))
            else:
                ax.set_title("CT mean" if method == METHODS[0] else "CT mean inside exact lungs", fontsize=11)
            july.add_orientation_markers(ax)
        axes[ri, 2].text(0, 1.28, labels[ri], transform=axes[ri, 2].transAxes,
                         fontsize=12, fontweight="bold", ha="left", clip_on=False)
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(0.5, 0.91 - title_shift, "Whole-volume CT mean" if method == METHODS[0] else "Exact-lung CT mean · background only; all GT/prediction voxels retained",
             ha="center", fontsize=12)
    legend = [("TP on ray (priority)", july.TP_COLOR), ("FP only", july.FP_COLOR),
              ("FN only", july.FN_COLOR), ("Depth-disjoint FP + FN; no TP", july.DEPTH_MISMATCH_COLOR)]
    fig.legend(handles=[july.Patch(facecolor=color, label=label) for label, color in legend],
               loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False, fontsize=11)
    fig.text(0.5, 0.018, f"Frozen validation-selected models · {routing_note} · Full-volume 3D Dice · Lung window −600 / 1500 HU",
             ha="center", fontsize=10)
    return fig, axes


def save_masks(path, masks, ct):
    header = ct.header.copy()
    header.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(np.stack(masks).astype(np.uint8), ct.affine, header)
    img.set_qform(ct.get_qform(), int(ct.header["qform_code"]))
    img.set_sform(ct.get_sform(), int(ct.header["sform_code"]))
    nib.save(img, path)
    check = nib.load(path)
    if check.shape != img.shape or check.get_data_dtype() != np.dtype("uint8") or not np.allclose(check.affine, ct.affine):
        raise ValueError("Derived mask export geometry failed")
    if not np.array_equal(np.asarray(check.dataobj), np.stack(masks)):
        raise ValueError("Derived mask round-trip failed")


def execute(plan, dpi):
    files, provenance = plan["files"], plan["provenance"]
    hashed = {}

    def file_record(p):
        p = Path(p)
        if str(p) not in hashed:
            hashed[str(p)] = july.sha256_file(p)
        return {"path": str(p), "sha256": hashed[str(p)], "bytes": p.stat().st_size}

    inputs = {k: file_record(p) for k, p in {**files, **provenance}.items()}
    audit22 = read_json(provenance["exp022_case"])
    for audit_key, input_key in (("ct", "ct"), ("gt", "gt"), ("anatomy", "anatomy"), ("pred", "overall_best_raw")):
        if inputs[input_key]["sha256"] != audit22["hashes"][audit_key]:
            raise ValueError(f"Exp022 audited source hash mismatch: {input_key}")
    if inputs["anatomy"]["sha256"] != plan["audit"]["integrity"]["observed_sha256"]:
        raise ValueError("Exp020 anatomy hash mismatch")
    for row in plan["models"].values():
        for model in row.values():
            if model["checkpoint"] not in hashed:
                print(f"Hashing checkpoint: {model['candidate_id']}", flush=True)
            if file_record(model["checkpoint"])["sha256"] != model["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {model['checkpoint']}")
    print("Source and checkpoint hashes verified", flush=True)
    ct = nib.load(files["ct"])
    ct_data = np.asarray(ct.dataobj, dtype=np.float32)
    gt = load_gt(files["gt"])
    anatomy = np.asarray(nib.load(files["anatomy"]).dataobj)
    whole = anatomy_support(anatomy, ct.affine, WHOLE_LABELS)
    upper = anatomy_support(anatomy, ct.affine, UPPER_LABELS)
    to_ras = ornt_transform(io_orientation(ct.affine), axcodes2ornt(("R", "A", "S")))
    orient = lambda a: apply_orientation(a, to_ras)
    ct_ras = orient(ct_data)
    backgrounds = {
        METHODS[0]: july.radiology_coronal_display(july.coronal_ct_projection_xz(ct_ras, "mean")),
        METHODS[1]: july.radiology_coronal_display(lung_mean_projection(ct_ras, orient(np.isin(anatomy, WHOLE_LABELS)))),
    }
    zooms = ct.header.get_zooms()[:3]
    spacing = [float(zooms[int(axis)]) for axis, _ in to_ras]
    aspect = spacing[2] / spacing[0]
    print("Native 20 mm supports and both CT backgrounds computed", flush=True)
    metrics24 = read_json(provenance["exp024_metrics"])
    metrics25 = read_json(provenance["exp025_metrics"])
    corrected, scores, overlays, records = {}, {}, {}, []
    for row_key in ROW_KEYS:
        raw = load_binary(files[row_key + "_raw"])
        pp2 = raw & whole[None]
        if row_key != "overall_best":
            saved = load_binary(files[row_key + "_pp2"])
            if not np.array_equal(saved, pp2):
                raise ValueError(f"{row_key}: saved PP2 differs from reconstruction")
            pp2 = saved
            if not np.array_equal(load_binary(files[row_key + "_original_pp3"]), pp2):
                raise ValueError(f"{row_key}: original PP3 no longer equals PP2")
        corrected[row_key] = [raw[fid] & upper for fid in plan["finding_ids"]]
        scores[row_key], overlays[row_key] = {}, {}
        for i, fid in enumerate(plan["finding_ids"]):
            if row_key == "category_best":
                if not np.array_equal(raw[fid], load_binary(files[f"category_source_{fid}"])[fid]):
                    raise ValueError("Category-composed mask differs from selected checkpoint's channel")
            if row_key == "iso07_best" and not np.array_equal(raw[fid], load_binary(files["iso07_source"])[fid]):
                raise ValueError("Iso07 baseline copy differs from original checkpoint export")
            preds = (raw[fid], pp2[fid], corrected[row_key][i])
            scores[row_key][fid] = [dice_score(gt[fid], p) for p in preds]
            overlays[row_key][fid] = [july.projected_error_rgba(orient(gt[fid]), orient(p)) for p in preds]
            if row_key == "overall_best":
                reference = next(f for f in audit22["findings"] if int(f["finding_id"]) == fid)
                expected = [reference["baseline"]["dice"], reference["candidates"]["prompt_lung:mask:20"]["dice"], reference["candidates"]["prompt_fine:mask:20"]["dice"]]
                if reference["candidates"]["prompt_fine:mask:20"]["labels"] != list(UPPER_LABELS):
                    raise ValueError("Exp022 reference does not describe corrected upper-lobe support")
            elif row_key == "category_best":
                expected = [metrics24["per_case"][k][plan["name"]][fid] for k in ("b1", "b2_prompt_lung20")]
            else:
                expected = [next(r["dice"] for r in metrics25["sets"][k]["rows"] if r["case"] == plan["name"] and int(r["finding_id"]) == fid) for k in ("baseline", "whole_lung20")]
            for pi, value in enumerate(expected):
                require_equal(scores[row_key][fid][pi], value, f"{row_key}/{fid}/{POLICIES[pi]}")
            for pi, pred in enumerate(preds):
                tp = int(np.count_nonzero(gt[fid] & pred))
                records.append(dict(case=plan["name"], finding_id=fid, category=plan["entry"]["categories"][str(fid)], row=row_key,
                                    model=plan["models"][fid][row_key]["candidate_id"], policy=POLICIES[pi], dice=scores[row_key][fid][pi],
                                    gt_voxels=int(gt[fid].sum()), pred_voxels=int(pred.sum()), tp=tp,
                                    fp=int(pred.sum()) - tp, fn=int(gt[fid].sum()) - tp))
        if row_key == "overall_best":
            overall_pp2 = [pp2[fid] for fid in plan["finding_ids"]]
        print(f"Verified {row_key}: " + str(scores[row_key]), flush=True)
    # Do not write any artifact until all scientific equivalence gates pass.
    output = plan["output"]
    if output.resolve().is_relative_to(REPO.resolve()):
        raise ValueError("Generated case artifacts must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    mask_dir = output / "derived_masks"
    mask_dir.mkdir(exist_ok=True)
    artifacts = []
    for key, masks in [("overall_best_pp2_whole_lung20", overall_pp2)] + [(k + "_pp3_upper_lobes20_corrected", v) for k, v in corrected.items()]:
        p = mask_dir / (key + ".nii.gz")
        save_masks(p, masks, ct)
        artifacts.append(file_record(p) | {"finding_ids_by_channel": plan["finding_ids"]})
    figures = []
    for fid in plan["finding_ids"]:
        overlay_grid = [overlays[k][fid] for k in ROW_KEYS]
        score_grid = [scores[k][fid] for k in ROW_KEYS]
        gt_overlay = july.rgba_mask(july.radiology_coronal_display(july.coronal_mask_projection_xz(orient(gt[fid]))), july.GT_COLOR)
        title = f"{plan['token']} · finding {fid} · {plan['entry']['categories'][str(fid)]}\n{plan['entry']['findings'][str(fid)]}"
        common_signature = None
        for method, background in backgrounds.items():
            fig, axes = create_figure(background, gt_overlay, overlay_grid, score_grid,
                                      [plan["models"][fid][k]["label"] for k in ROW_KEYS], aspect, title, method)
            for ri, ci in BLANK_CELLS:
                if axes[ri, ci].images or axes[ri, ci].texts or axes[ri, ci].get_title():
                    raise ValueError("Blank-cell layout violated")
            # Read back the plotted arrays, proving the background choice never clips overlays.
            signature = []
            for ri in range(3):
                for pi in range(3):
                    plotted = np.asarray(axes[ri, pi + 2].images[1].get_array())
                    if not np.array_equal(plotted, overlay_grid[ri][pi]):
                        raise ValueError("Plotted overlay changed")
                    signature.append(july.hashlib.sha256(plotted.tobytes()).hexdigest())
            if common_signature is not None and common_signature != signature:
                raise ValueError("Overlays differ between background methods")
            common_signature = signature
            p = output / f"{plan['token']}_finding{fid}_{plan['entry']['categories'][str(fid)]}_{method}.png"
            fig.savefig(p, dpi=dpi, facecolor="white")
            july.plt.close(fig)
            figures.append(file_record(p) | {"finding_id": fid, "method": method, "prediction_overlay_sha256": signature, "dice": score_grid, "blank_cells": BLANK_CELLS})
            print(f"Wrote {p}", flush=True)
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(), repo_commit=july.repo_commit(),
                    execution_environment={"python": sys.version, "device": "CPU", "container_image": os.environ.get("COMPARISON_CONTAINER_IMAGE"), "command": sys.argv},
                    case=plan["name"], fixed_val_index=plan["index"], finding_ids=plan["finding_ids"], inputs=inputs,
                    code={"renderer": file_record(Path(__file__)), "july_renderer": file_record(Path(july.__file__)),
                          "orientation_helper": file_record(HERE.parent / "rex_val20_coronal_viz.py"), "postprocessing": file_record(Path(post.__file__))},
                    models=plan["models"], native_headers=plan["headers"], ras_transform=to_ras.tolist(), spacing_ras_mm=spacing,
                    anatomy_source_status="PASS_PENDING_MANUAL_VISUAL_REVIEW", frozen_routes=plan["routes"],
                    corrected_route={"labels": UPPER_LABELS, "margin_mm": 20, "reason": "Explicit user-approved bilateral upper-lobe correction; frozen routes unchanged"},
                    pp2={"labels": WHOLE_LABELS, "margin_mm": 20}, ct_window={"center_hu": -600, "width_hu": 1500},
                    projection={"methods": METHODS, "lung_labels": WHOLE_LABELS, "lung_margin_mm": 0, "restriction": "CT background only", "empty_rays": "black", "overlay_rule": "3D TP priority; purple for depth-disjoint FP/FN without TP", "gt_binarization": "positive instance IDs > 0", "dice": "full-volume 3D"},
                    derived_masks=artifacts, figures=figures, metrics=file_record(metrics_path),
                    checks={"source_hashes": "PASS", "checkpoint_hashes": "PASS", "native_geometry": "PASS", "source_composition": "PASS", "saved_pp2": "PASS", "original_pp3_equals_pp2": "PASS", "published_metrics": "PASS", "exp022_corrected_pp3": "PASS", "overlay_invariance": "PASS", "blank_cells": "PASS", "derived_mask_roundtrip": "PASS"},
                    visual_review="Pending visual inspection of generated PNGs")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="val189_train_3026_a_2")
    p.add_argument("--mode", choices=("single-case", "val200-by-category"), default="single-case")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--runtime-dir", type=Path, help="External full-gallery derived masks and completion records")
    p.add_argument("--findings", nargs="+", type=int)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--runtime-root", type=Path, default=ROOT)
    p.add_argument("--ct-root", type=Path, default=post.CT_ROOT)
    p.add_argument("--seg-dir", type=Path, default=post.SEG_DIR)
    p.add_argument("--metadata-json", type=Path, default=post.META)
    p.add_argument("--val-json", type=Path, default=REPO / "configs/evaluation/rexgroundingct_val200_seed20260723.json")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dpi <= 0:
        p.error("--dpi must be positive")
    if args.workers <= 0:
        p.error("--workers must be positive")
    if args.mode == "val200-by-category":
        import full_gallery
        full_gallery.main(args)
        return
    if args.resume or args.verify_only or args.runtime_dir:
        p.error("--resume, --verify-only and --runtime-dir require --mode val200-by-category")
    plan = prepare(args)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "case": plan["name"], "index": plan["index"], "finding_ids": plan["finding_ids"],
                          "figure_count": len(plan["finding_ids"]) * 2, "output": str(plan["output"]),
                          "inputs": {k: str(v) for k, v in plan["files"].items()}, "models": plan["models"], "headers": plan["headers"]}, indent=2))
    else:
        print(f"Complete: {execute(plan, args.dpi)}")


if __name__ == "__main__":
    main()
