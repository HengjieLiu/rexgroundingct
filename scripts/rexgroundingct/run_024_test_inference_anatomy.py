#!/usr/bin/env python3
"""Resumable Exp024 test inference and validation-gated anatomy postprocessing.

The coordinator is intentionally conservative: prompt routing is frozen before
any prediction is inspected, all joins are by ``(case, finding index)``, and
the b-test stage refuses to start without a completed validation gate.  Heavy
VoxTell work is delegated to the existing inference entrypoint (normally in
the pinned VoxTell container); CPU-only routing, masking, export checks, and
validation reports remain reproducible here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/experiments/024_test_inference_anatomy_audit.json"
EXP_ROOT = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit")
META = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
CT_ROOT = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")
SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
SOURCE_VAL = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/020_ct_rate_ts_total_rex_val200_audit")
SOURCE_TEST = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/023_ct_rate_ts_total_rex_test300_audit")
ORACLE = REPO / "side_experiments/sideexp002_multimodel_ensemble_selection/outputs/all_methods_val200_cutoff030_category_oracle.json"
AUDIT_SOURCE = REPO / "experiments/018_voxtell_category2d_nodule_audit/nodule_method_audit.json"
VAL_JSON = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch050_val200/eval/val_as_test_dataset.json")

CATEGORIES = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
LABELS = {"LU": [10], "LL": [11], "RU": [12], "RM": [13], "RL": [14], "L": [10, 11], "R": [12, 13, 14], "B": [10, 11, 12, 13, 14], "W": [10, 11, 12, 13, 14], "N": []}
LOBE_RE = [("LU", r"(?:left|lingu(?:la|lar)|lt\.?)[ -]*(?:upper|sup(?:erior)?)[ -]*lobe"), ("LL", r"(?:left|lt\.?)[ -]*(?:lower|inferior)[ -]*lobe"), ("RU", r"(?:right|rt\.?)[ -]*(?:upper|sup(?:erior)?)[ -]*lobe"), ("RM", r"(?:right|rt\.?)[ -]*middle[ -]*lobe"), ("RL", r"(?:right|rt\.?)[ -]*(?:lower|inferior)[ -]*lobe")]
SIDE_RE = {"L": r"\b(?:left|lt\.?|lingula|lingular)\b", "R": r"\b(?:right|rt\.?)\b"}
RISK_RE = {"peripheral": r"peripheral|subpleural|pleural[ -]?based|juxtapleural|paraseptal", "pleural": r"pleur", "fissural": r"fissur", "airway": r"airway|bronch|tree[ -]?in[ -]?bud|centrilobular", "vascular": r"vascular|vessel|aorta|artery|vein", "pneumothorax": r"pneumothorax", "landmark": r"mediast|hilar|diaphragm|chest wall|fissur"}
NONPULMONARY_CATEGORIES = {"2e", "2g"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def entries(metadata: Path = META, split: str = "test") -> list[dict[str, Any]]:
    data = json.loads(metadata.read_text())
    out = []
    for i, row in enumerate(data[split]):
        x = dict(row); x["_split"] = split; x["_index"] = i; out.append(x)
    return out

def census(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in rows:
        if set(e.get("findings", {})) != set(e.get("categories", {})):
            raise ValueError(f"finding/category key mismatch: {e.get('name')}")
        for key, prompt in e["findings"].items():
            out.append({"case": e["name"], "finding_id": int(key), "prompt": str(prompt), "category": e["categories"][key]})
    return sorted(out, key=lambda r: (r["case"], r["finding_id"]))

def route_prompt(prompt: str, category: str) -> dict[str, Any]:
    """Outcome-blind conservative semantic routing (no GT or prediction access)."""
    text = prompt.lower()
    risks = [k for k, pat in RISK_RE.items() if re.search(pat, text)]
    explicit_lobes = []
    for code, pat in LOBE_RE:
        if re.search(pat, text): explicit_lobes.append(code)
    explicit_lobes = sorted(set(explicit_lobes))
    bilateral = bool(re.search(r"\bbilateral\b|\bboth\s+(?:lungs|sides)\b", text))
    sides = {code for code, pat in SIDE_RE.items() if re.search(pat, text)}
    if bilateral or ("L" in sides and "R" in sides): scope = "B"
    elif explicit_lobes:
        scope = "+".join(explicit_lobes) if len(explicit_lobes) > 1 else explicit_lobes[0]
    elif len(sides) == 1: scope = next(iter(sides))
    else: scope = "W"
    extent = "diffuse" if re.search(r"diffuse|widespread|extensive|throughout", text) else "multifocal" if re.search(r"multifocal|multiple|bilateral", text) else "focal" if re.search(r"focal|single|solitary|nodule|mass|opacity|focus", text) else "unclear"
    target_is_pleura = "pleural" in risks and not re.search(r"lung|pulmonary|parench|opacity|nodule|mass|consolid|ground.?glass", text)
    unsupported = category in NONPULMONARY_CATEGORIES or target_is_pleura or "pneumothorax" in risks or ("vascular" in risks and not re.search(r"lung|pulmonary|parench", text))
    pulmonary_target = not unsupported
    fine_eligible = pulmonary_target and bool(explicit_lobes or len(sides) == 1 or bilateral)
    eligible = pulmonary_target
    labels = sorted({v for token in scope.split("+") for v in LABELS[token]}) if eligible else []
    rationale = "eligible broad pulmonary support"
    if not eligible: rationale = "no hard restriction: target is pleural/extrapulmonary, pneumothorax, vascular, or unsupported"
    elif fine_eligible: rationale += "; fine support follows only explicitly named side/lobe and never a zone/segment"
    else: rationale += "; no explicit side/lobe support, so fine policy is unchanged"
    return {"category": category, "prompt": prompt, "extent": extent, "laterality": "bilateral" if scope == "B" else "left" if scope.startswith("L") else "right" if scope.startswith("R") else "unspecified", "explicit_lobes": explicit_lobes, "scope": scope, "risks": risks, "target_relation": "target itself" if not re.search(r"at the|in the|level|zone|lobe", text) else "target with anatomical landmark", "eligible": eligible, "fine_eligible": fine_eligible, "selected_labels": labels, "rationale": rationale, "uncertainty": bool(re.search(r"uncertain|indeterminate|nonspecific|cannot exclude", text)), "negation": bool(re.search(r"no |without |absence of", text))}

def freeze_routes(config: Mapping[str, Any], split: str, output: Path) -> dict[str, Any]:
    rows = census(entries(Path(config["metadata_path"]), split))
    expected = config["test_counts"] if split == "test" else config["val_counts"]
    if len(rows) != expected["findings"]: raise ValueError(f"{split}: expected {expected['findings']} findings, got {len(rows)}")
    frozen = []
    for idx, row in enumerate(rows):
        item = dict(row); item["route_id"] = idx; item.update(route_prompt(row["prompt"], row["category"])); frozen.append(item)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "frozen outcome-blind prompt routing", "split": split, "metadata_sha256": sha256(Path(config["metadata_path"])), "rows": frozen}
    if output.exists():
        old = json.loads(output.read_text())
        if old.get("rows") != frozen: raise RuntimeError(f"refusing to overwrite frozen routing: {output}")
        return old
    write_json(output, result); return result

def physical_dilation(mask: np.ndarray, affine: np.ndarray, margin_mm: float = 20.0) -> np.ndarray:
    """Binary support dilated in native physical space (anisotropic-safe)."""
    if mask.ndim != 3: raise ValueError("anatomy mask must be 3-D")
    from scipy.ndimage import distance_transform_edt
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    if np.any(spacing <= 0): raise ValueError("invalid affine spacing")
    if not np.any(mask): return np.zeros(mask.shape, dtype=bool)
    if float(margin_mm) <= 0:
        return mask.astype(bool, copy=True)
    return (mask.astype(bool) | (distance_transform_edt(~mask.astype(bool), sampling=spacing) <= float(margin_mm) + 1e-7))

def apply_support(prediction: np.ndarray, anatomy: np.ndarray, affine: np.ndarray, route: Mapping[str, Any], policy: str) -> np.ndarray:
    if prediction.ndim != 3: raise ValueError("finding prediction must be 3-D")
    if policy == "unchanged": return prediction.astype(np.uint8, copy=True)
    labels = LABELS["W"] if policy in {"whole_lung", "whole_lung_exact"} else route.get("selected_labels", [])
    if policy == "fine" and not route.get("fine_eligible"): return prediction.astype(np.uint8, copy=True)
    if policy == "whole_lung" and not route.get("eligible"): return prediction.astype(np.uint8, copy=True)
    support = physical_dilation(np.isin(anatomy, labels), affine, 0.0 if policy == "whole_lung_exact" else 20.0)
    return np.logical_and(prediction > 0, support).astype(np.uint8)

def source_mask_map(runtime: Path, split: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    report = runtime / ("reports/private/audit_results.json")
    if report.exists():
        for row in json.loads(report.read_text()).get("cases", []):
            name = row.get("volume_name"); path = row.get("integrity", {}).get("path")
            if name and path and Path(path).exists(): result[name] = Path(path)
    if not result:
        for p in runtime.rglob("*.nii.gz"):
            if p.name.endswith(".nii.gz") and ("sources" in str(p) or "masks" in str(p)): result.setdefault(p.name, p)
    return result

def compose_category_predictions(raw_root: Path, output: Path, metadata: Mapping[str, Any], candidate_manifest: Mapping[str, Any], oracle: Mapping[str, str], split: str) -> dict[str, Any]:
    """Compose b1 from complete per-checkpoint finding-first prediction files."""
    import nibabel as nib
    def pred_path(model: str, name: str) -> Path:
        local = raw_root / model / "predictions" / name
        if local.exists(): return local
        val_dir = candidate_manifest.get(model, {}).get("validation_prediction_dir")
        if val_dir: return Path(val_dir) / name
        return local
    # raw_root/<candidate>/predictions/<case> is the canonical private layout.
    output.mkdir(parents=True, exist_ok=True); records = []
    for entry in metadata[split]:
        name = entry["name"]; selected = None
        for model in sorted(set(oracle.get(str(c)) for c in entry["categories"].values() if oracle.get(str(c)))):
            path = pred_path(model, name)
            if path.exists():
                selected = nib.load(str(path)); break
        if selected is None:
            # Find one available model to establish the native reference.
            for model in candidate_manifest:
                path = pred_path(model, name)
                if path.exists(): selected = nib.load(str(path)); break
        if selected is None: raise FileNotFoundError(f"no raw candidate prediction for {name}")
        base = np.asanyarray(selected.dataobj)
        if base.ndim == 3: base = base[None]
        out = np.zeros_like(base, dtype=np.uint8)
        for key in entry["findings"]:
            model = oracle.get(str(entry["categories"][key]))
            if not model: continue
            path = pred_path(model, name)
            if not path.exists(): raise FileNotFoundError(path)
            data = np.asanyarray(nib.load(str(path)).dataobj)
            if data.ndim == 3: data = data[None]
            out[int(key)] = data[int(key)] > 0
        nib.save(nib.Nifti1Image(out, selected.affine, selected.header.copy()), str(output / name)); records.append({"name": name, "findings": int(out.shape[0])})
    return {"split": split, "cases": len(records), "findings": sum(r["findings"] for r in records), "source": str(raw_root)}

def evaluate_directory(pred_dir: Path, metadata: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Recompute finding Dice directly from released val GT (never used for test)."""
    import nibabel as nib
    rows = []; by_case: dict[str, list[float]] = {}
    for entry in metadata[split]:
        gt_path = SEG_DIR / entry["name"]; pred_path = pred_dir / entry["name"]
        if not gt_path.exists() or not pred_path.exists(): raise FileNotFoundError(str(gt_path if not gt_path.exists() else pred_path))
        g = np.asanyarray(nib.load(str(gt_path)).dataobj); p = np.asanyarray(nib.load(str(pred_path)).dataobj)
        if g.ndim == 3: g = g[None]
        if p.ndim == 3: p = p[None]
        if g.shape != p.shape or g.shape[0] != len(entry["findings"]): raise ValueError(f"{entry['name']}: evaluator finding-axis mismatch")
        vals = []
        for key in entry["findings"]:
            gb = g[int(key)] > 0; pb = p[int(key)] > 0; inter = int(np.logical_and(gb, pb).sum()); denom = int(gb.sum() + pb.sum()); dice = (2 * inter + 1e-6) / (denom + 1e-6) if denom else 1.0
            row = {"case": entry["name"], "finding_id": int(key), "category": entry["categories"][key], "dice": float(dice), "hit": bool(dice >= .1), "gt_voxels": int(gb.sum()), "pred_voxels": int(pb.sum()), "tp": inter, "fp": int(pb.sum()) - inter}; rows.append(row); vals.append(float(dice))
        by_case[entry["name"]] = vals
    return {"cases": len(by_case), "findings": len(rows), "hits": sum(int(r["hit"]) for r in rows), "dice": float(np.mean([r["dice"] for r in rows])), "case_dice": float(np.mean([np.mean(v) for v in by_case.values()])), "rows": rows, "by_case": by_case}

def bootstrap_delta(base: Mapping[str, Any], other: Mapping[str, Any], resamples: int = 2000, seed: int = 20260905) -> dict[str, float]:
    rng = np.random.default_rng(seed); names = sorted(base["by_case"]); a = np.asarray([base["by_case"][n] for n in names], dtype=object); b = np.asarray([other["by_case"][n] for n in names], dtype=object); vals = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, len(names), len(names)); vals[i] = float(np.mean(np.concatenate([b[j] for j in idx])) - np.mean(np.concatenate([a[j] for j in idx])))
    return {"mean_delta": float(other["dice"] - base["dice"]), "ci95_low": float(np.quantile(vals, .025)), "ci95_high": float(np.quantile(vals, .975)), "resamples": resamples, "seed": seed}

def write_validation_report(root: Path, metadata: Mapping[str, Any], dirs: Mapping[str, Path]) -> dict[str, Any]:
    results = {name: evaluate_directory(path, metadata, "val") for name, path in dirs.items()}; base = results["b1"]
    def category_table(result: Mapping[str, Any]) -> dict[str, Any]:
        grouped: dict[str, list[float]] = {}
        hits: dict[str, int] = {}
        for row in result["rows"]:
            grouped.setdefault(str(row["category"]), []).append(float(row["dice"]))
            hits[str(row["category"])] = hits.get(str(row["category"]), 0) + int(row["hit"])
        return {c: {"findings": len(grouped.get(c, [])), "hits": hits.get(c, 0), "dice": float(np.mean(grouped[c])) if grouped.get(c) else None} for c in CATEGORIES if grouped.get(c)}
    summary = {"oracle_qualification": "retrospective validation-selected and optimistic; not an unbiased test estimate", "baseline": {k: base[k] for k in ("cases", "findings", "hits", "dice", "case_dice")}, "category_tables": {"b1": category_table(base)}, "per_case": {"b1": base["by_case"]}, "policies": {}}
    for name, result in results.items():
        if name == "b1": continue
        paired = {(r["case"], r["finding_id"]): r for r in base["rows"]}
        tp_removed = sum(max(0, paired[(r["case"], r["finding_id"])] ["tp"] - r["tp"]) for r in result["rows"])
        fp_removed = sum(max(0, paired[(r["case"], r["finding_id"])] ["fp"] - r["fp"]) for r in result["rows"])
        empties = sum(int(r["pred_voxels"] == 0 and paired[(r["case"], r["finding_id"])] ["pred_voxels"] > 0) for r in result["rows"])
        summary["policies"][name] = {k: result[k] for k in ("cases", "findings", "hits", "dice", "case_dice")} | {"tp_voxels_removed": tp_removed, "fp_voxels_removed": fp_removed, "predictions_emptied": empties, "bootstrap": bootstrap_delta(base, result)}
        summary["category_tables"][name] = category_table(result); summary["per_case"][name] = result["by_case"]
    report = ["# Exp024 b validation gate", "", "The category oracle is retrospective, validation-selected, and optimistic; this is not an unbiased test estimate.", "", "| set | findings | hits | mean finding Dice | delta |", "|---|---:|---:|---:|---:|"]
    for name, result in results.items(): report.append(f"| {name} | {result['findings']} | {result['hits']} | {result['dice']:.9f} | {(result['dice']-base['dice']):+.9f} |" if name != "b1" else f"| b1 | {result['findings']} | {result['hits']} | {result['dice']:.9f} | +0.000000000 |")
    for name, result in results.items():
        if name == "b1": continue
        ci = summary["policies"][name]["bootstrap"]; report.extend(["", f"{name} paired CT bootstrap 95% CI: [{ci['ci95_low']:.9f}, {ci['ci95_high']:.9f}] (n={ci['resamples']}, seed={ci['seed']})."])
    report.extend(["", "## Category summaries", "", "| set | category | findings | hits | Dice |", "|---|---|---:|---:|---:|"])
    for name, table in summary["category_tables"].items():
        for category, metric in table.items(): report.append(f"| {name} | {category} | {metric['findings']} | {metric['hits']} | {metric['dice']:.9f} |")
    (root / "reports").mkdir(parents=True, exist_ok=True); (root / "reports/b_validation_report.md").write_text("\n".join(report) + "\n"); write_json(root / "reports/b_validation_metrics.json", summary); write_json(root / "b_validation_complete.json", {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": str(root / "reports/b_validation_metrics.json")}); return summary

def transform_case_dir(src: Path, dst: Path, routes: Mapping[tuple[str, int], Mapping[str, Any]], policy: str, masks: Mapping[str, Path], metadata: Mapping[str, Any], split: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
    import nibabel as nib
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_entries = metadata[split][start:end]
    for entry in selected_entries:
        name = entry["name"]; pred_path = src / name
        if not pred_path.exists(): raise FileNotFoundError(pred_path)
        pred_img = nib.load(str(pred_path)); pred = np.asanyarray(pred_img.dataobj)
        if pred.ndim == 3: pred = pred[None]
        if pred.ndim != 4 or pred.shape[0] != len(entry["findings"]): raise ValueError(f"{name}: finding-axis mismatch")
        ct = nib.load(str(CT_ROOT / (Path("dataset") / f"{name.split('_')[0]}_fixed" / '_'.join(name[:-7].split('_')[:2]) / '_'.join(name[:-7].split('_')[:3]) / name)))
        if tuple(pred.shape[1:]) != tuple(ct.shape): raise ValueError(f"{name}: prediction/CT shape mismatch {pred.shape} {ct.shape}")
        anatomy = None
        if name not in masks and policy != "unchanged":
            raise FileNotFoundError(f"no audited anatomy mask for {name}")
        if name in masks:
            anatomy = np.asanyarray(nib.load(str(masks[name])).dataobj)
            if anatomy.shape != ct.shape or not np.allclose(nib.load(str(masks[name])).affine, ct.affine, atol=1e-5, rtol=0): raise ValueError(f"{name}: anatomy/CT geometry mismatch")
        out = pred.astype(np.uint8, copy=True)
        support_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
        for key in entry["findings"]:
            rid = (name, int(key)); route = routes[rid]
            if anatomy is not None and policy != "unchanged":
                labels = tuple(LABELS["W"] if policy in {"whole_lung", "whole_lung_exact"} else route.get("selected_labels", []))
                eligible = policy == "whole_lung_exact" or (policy == "whole_lung" and route.get("eligible")) or (policy == "fine" and route.get("fine_eligible"))
                if eligible and labels:
                    cache_key = (policy, labels)
                    if cache_key not in support_cache:
                        support_cache[cache_key] = physical_dilation(np.isin(anatomy, labels), ct.affine, 0.0 if policy == "whole_lung_exact" else 20.0)
                    out[int(key)] = np.logical_and(out[int(key)] > 0, support_cache[cache_key]).astype(np.uint8)
        hdr = ct.header.copy(); hdr.set_data_dtype(np.uint8); nib.save(nib.Nifti1Image(out, ct.affine, hdr), str(dst / name))
        rows.append({"name": name, "shape": list(out.shape), "affine_sha256": hashlib.sha256(np.asarray(ct.affine, dtype=np.float64).tobytes()).hexdigest(), "findings": out.shape[0]})
    return {"cases": len(rows), "files": rows, "policy": policy}

def validate_outputs(root: Path, split: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    import nibabel as nib
    expected = metadata[split]; names = {e["name"] for e in expected}; files = {p.name for p in root.glob("*.nii.gz")}
    if files != names: raise ValueError(f"output set mismatch: expected {len(names)}, observed {len(files)}")
    records = []
    for e in expected:
        p = root / e["name"]; img = nib.load(str(p)); data = np.asanyarray(img.dataobj); ct_path = CT_ROOT / (Path("dataset") / f"{e['name'].split('_')[0]}_fixed" / '_'.join(e['name'][:-7].split('_')[:2]) / '_'.join(e['name'][:-7].split('_')[:3]) / e['name']); ct = nib.load(str(ct_path))
        if data.ndim != 4 or data.shape[0] != len(e["findings"]) or tuple(data.shape[1:]) != tuple(ct.shape): raise ValueError(f"{p}: native shape/finding count mismatch")
        if not np.allclose(img.affine, ct.affine, atol=1e-5, rtol=0): raise ValueError(f"{p}: affine mismatch")
        if img.get_data_dtype() != np.dtype("uint8") or not np.isin(data, [0, 1]).all(): raise ValueError(f"{p}: expected binary uint8")
        records.append({"name": e["name"], "shape": list(data.shape), "dtype": str(img.get_data_dtype()), "findings": int(data.shape[0])})
    return {"split": split, "cases": len(records), "findings": sum(r["findings"] for r in records), "files": records}

def resolve_candidates(config: Mapping[str, Any]) -> dict[str, Any]:
    if sha256(ORACLE) != config["oracle_sha256"]: raise ValueError("oracle hash mismatch")
    if sha256(AUDIT_SOURCE) != config["exp018_audit_source_sha256"]: raise ValueError("Exp018 source hash mismatch")
    oracle = json.loads(ORACLE.read_text()); source = json.loads(AUDIT_SOURCE.read_text()); rows = {r["candidate_id"]: r for r in source["primary_leaderboard"]}
    out = {}
    for category, candidate in config["oracle_categories"].items():
        if candidate is None: continue
        if candidate not in rows: raise ValueError(f"candidate missing from pinned source: {candidate}")
        r = rows[candidate]; ev = Path(r["evaluation_path"])
        model_dir = ev.parent.parent.parent / f"model_epoch{int(r['relative_epoch']):03d}"
        ckpt = model_dir / "fold_0/checkpoint_final.pth"
        if not ckpt.exists(): raise FileNotFoundError(ckpt)
        cache_id = "crop_clip1024_linear_iso07_v1" if candidate.startswith("exp017_") else "crop_clip1024_linear_native_v1" if candidate.startswith("exp011_") else "crop_zscore_native_v1"
        cache_path = (EXP_ROOT / "cache" / cache_id) if cache_id != "crop_clip1024_linear_iso07_v1" else Path(config["preprocessed_cache_root"]) / cache_id
        runtime_manifest = ev.parent.parent.parent / "run_manifest.json"
        # Some historical runs expose the network definition as nnUNet
        # ``plans.json`` rather than a separate model_spec.json.  Record that
        # immutable architecture file so every oracle checkpoint has an
        # auditable model-spec hash.
        model_spec = model_dir / "model_spec.json"
        if not model_spec.exists() and (model_dir / "plans.json").exists():
            model_spec = model_dir / "plans.json"
        val_pred_dir = ev.parent.parent / "predictions"
        out[candidate] = {"candidate_id": candidate, "categories": [], "evaluation_path": str(ev), "evaluation_sha256": sha256(ev), "validation_prediction_dir": str(val_pred_dir), "validation_prediction_reused": bool(val_pred_dir.is_dir() and len(list(val_pred_dir.glob("*.nii.gz"))) == 200), "validation_prediction_count": len(list(val_pred_dir.glob("*.nii.gz"))) if val_pred_dir.is_dir() else 0, "relative_epoch": int(r["relative_epoch"]), "absolute_epoch": int(r["absolute_epoch"]), "model_dir": str(model_dir), "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "model_spec": str(model_spec) if model_spec.exists() else None, "model_spec_sha256": sha256(model_spec) if model_spec.exists() else None, "architecture_spec": str(model_spec) if model_spec.exists() else None, "architecture_spec_sha256": sha256(model_spec) if model_spec.exists() else None, "runtime_manifest": str(runtime_manifest) if runtime_manifest.exists() else None, "runtime_manifest_sha256": sha256(runtime_manifest) if runtime_manifest.exists() else None, "cache": cache_id, "cache_root": str(cache_path), "cache_manifest": str(cache_path / "manifest.json"), "cache_manifest_sha256": sha256(cache_path / "manifest.json") if (cache_path / "manifest.json").exists() else None}
    for c, candidate in config["oracle_categories"].items():
        if candidate in out: out[candidate]["categories"].append(c)
    write_json(EXP_ROOT / "config/candidate_manifest.json", {"oracle_sha256": config["oracle_sha256"], "candidates": out, "validation_oracle_qualification": "retrospective validation-selected and optimistic; not an unbiased test estimate"})
    return out

def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=["preflight", "prompt-review", "resolve", "compose", "apply", "validate-b", "verify", "run-inference"]); p.add_argument("--split", choices=["val", "test"], default="test"); p.add_argument("--source"); p.add_argument("--output"); p.add_argument("--policy", choices=["unchanged", "whole_lung", "whole_lung_exact", "fine"]); p.add_argument("--input"); p.add_argument("--model-dir"); p.add_argument("--cache"); p.add_argument("--dataset"); p.add_argument("--gpu", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--shard-index", type=int, default=0); p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=None); p.add_argument("--image", default="rexgroundingct-voxtell:cu126"); p.add_argument("--no-docker", action="store_true"); args = p.parse_args(argv)
    cfg = json.loads(CONFIG.read_text()); EXP_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        if sha256(Path(cfg["metadata_path"])) != cfg["metadata_sha256"]: raise SystemExit("metadata hash mismatch")
        for split, expected in (("test", cfg["test_counts"]), ("val", cfg["val_counts"])):
            e = entries(Path(cfg["metadata_path"]), split); got = sum(len(x["findings"]) for x in e)
            if len(e) != expected["cases"] or got != expected["findings"]: raise SystemExit(f"{split} census mismatch")
        resolve_candidates(cfg); write_json(EXP_ROOT / "config/preflight.json", {"metadata_sha256": cfg["metadata_sha256"], "source_status": cfg["source_status"], "test": cfg["test_counts"], "val": cfg["val_counts"], "created_at_utc": datetime.now(timezone.utc).isoformat()}); print("PREFLIGHT_COMPLETE")
    elif args.stage == "prompt-review":
        dest = EXP_ROOT / "config" / f"prompt_routing_{args.split}.json"; freeze_routes(cfg, args.split, dest); print(dest)
    elif args.stage == "resolve": resolve_candidates(cfg); print("CANDIDATES_RESOLVED")
    elif args.stage == "compose":
        if not args.source or not args.output: raise SystemExit("compose requires --source and --output")
        meta = json.loads(Path(cfg["metadata_path"]).read_text()); manifest = json.loads((EXP_ROOT / "config/candidate_manifest.json").read_text()); result = compose_category_predictions(Path(args.source), Path(args.output), meta, manifest["candidates"], cfg["oracle_categories"], args.split); write_json(Path(args.output).parent / f"{Path(args.output).name}.manifest.json", result); print(json.dumps(result))
    elif args.stage == "apply":
        if not args.source or not args.output or not args.policy: raise SystemExit("apply requires --source --output --policy")
        meta = json.loads(Path(cfg["metadata_path"]).read_text()); frozen = json.loads((EXP_ROOT / "config" / f"prompt_routing_{args.split}.json").read_text()); routes = {(r["case"], int(r["finding_id"])): r for r in frozen["rows"]}; masks = source_mask_map(SOURCE_TEST if args.split == "test" else SOURCE_VAL, args.split); result = transform_case_dir(Path(args.source), Path(args.output), routes, args.policy, masks, meta, args.split, args.start, args.end); write_json(Path(args.output).parent / f"{Path(args.output).name}.manifest.json", result); print(json.dumps({k: result[k] for k in ("cases", "policy")}))
    elif args.stage == "verify":
        if not args.output: raise SystemExit("verify requires --output")
        meta = json.loads(Path(cfg["metadata_path"]).read_text()); result = validate_outputs(Path(args.output), args.split, meta); write_json(Path(args.output).parent / f"{Path(args.output).name}.verification.json", result); print("OUTPUTS_VALID", result["cases"], result["findings"])
    elif args.stage == "validate-b":
        meta = json.loads(Path(cfg["metadata_path"]).read_text()); root = EXP_ROOT / "validation"; dirs = {name: root / name for name in ("b1", "b_whole_mask0", "b2_prompt_lung20", "b3_prompt_fine20")}; write_validation_report(EXP_ROOT, meta, dirs); print("B_VALIDATION_COMPLETE")
    else:
        if not args.model_dir or not args.cache or not args.output:
            raise SystemExit("run-inference requires --model-dir --cache --output")
        dataset = args.dataset or (str(VAL_JSON) if args.split == "val" else str(cfg["metadata_path"]))
        status = str(Path(args.output).parent / (Path(args.output).name + f".shard{args.shard_index}.status.json"))
        if not (1 <= args.num_shards) or not (0 <= args.shard_index < args.num_shards): raise SystemExit("invalid shard arguments")
        inner = ["python", "/workspace/scripts/rexgroundingct/run_voxtell_val_inference.py", "--dataset-json", dataset, "--split", args.split, "--model-dir", args.model_dir, "--ct-root", str(cfg["ct_root"]), "--seg-dir", str(SEG_DIR), "--preprocessed-cache-dir", args.cache, "--output-dir", args.output, "--output-type", "mask", "--threshold", str(cfg["threshold"]), "--num-shards", str(args.num_shards), "--shard-index", str(args.shard_index), "--status-json", status]
        if args.no_docker:
            run_command([sys.executable, *inner[1:]], EXP_ROOT / "logs/inference.log")
        else:
            command = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--shm-size=32g", "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{REPO}:/workspace", "-v", "/data/hengjie:/data/hengjie", "-v", "/mnt/shengdata1:/mnt/shengdata1", "-e", "HOME=/tmp", "-e", "HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home", "-e", "HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub", "-e", "HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet", args.image, "bash", "-lc", " ".join(inner)]
            run_command(command, EXP_ROOT / "logs/inference.log")
        print("INFERENCE_COMPLETE", args.output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
