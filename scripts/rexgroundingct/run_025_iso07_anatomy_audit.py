#!/usr/bin/env python3
"""Run the approved iso07 checkpoint anatomy audit (validation then test).

This coordinator deliberately keeps the official TotalSegmentator masks in
native CT space.  The iso07 inference exporter restores predictions to native
CT geometry before this module applies clipping or physical-space dilation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "scripts/rexgroundingct/run_024_test_inference_anatomy.py"
RUNTIME = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/025_iso07_best_anatomy_audit")
CONFIG = REPO / "configs/experiments/025_iso07_best_anatomy_audit.json"
MODEL_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/"
    "runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/model_epoch050"
)
RAW_VAL = MODEL_DIR.parent / "eval_epoch050_val200/predictions"
CHECKPOINT = MODEL_DIR / "fold_0/checkpoint_final.pth"
CACHE = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1")
META = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
CT_ROOT = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")
SOURCE_VAL = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/020_ct_rate_ts_total_rex_val200_audit")
SOURCE_TEST = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/023_ct_rate_ts_total_rex_test300_audit")


def load_base():
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp024_audit", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXP_ROOT = RUNTIME
    module.CONFIG = CONFIG
    module.META = META
    module.CT_ROOT = CT_ROOT
    module.SEG_DIR = SEG_DIR
    module.SOURCE_VAL = SOURCE_VAL
    module.SOURCE_TEST = SOURCE_TEST
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def metadata() -> dict[str, Any]:
    return json.loads(META.read_text())


def ensure_routes(base: Any) -> dict[str, Any]:
    RUNTIME.joinpath("config").mkdir(parents=True, exist_ok=True)
    routes: dict[str, Any] = {}
    for split in ("val", "test"):
        dest = RUNTIME / "config" / f"prompt_routing_{split}.json"
        old = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/config") / dest.name
        if old.exists():
            if dest.exists() and dest.read_bytes() != old.read_bytes():
                raise RuntimeError(f"frozen route differs from Exp024: {dest}")
            if not dest.exists():
                shutil.copy2(old, dest)
            routes[split] = json.loads(dest.read_text())
        else:
            routes[split] = base.freeze_routes(json.loads(CONFIG.read_text()), split, dest)
    return routes


def copy_val_baseline() -> Path:
    dst = RUNTIME / "validation/baseline"
    dst.mkdir(parents=True, exist_ok=True)
    expected = {p.name for p in RAW_VAL.glob("*.nii.gz")}
    entries = metadata()["val"]
    names = {e["name"] for e in entries}
    if expected != names:
        raise RuntimeError(f"iso07 val predictions incomplete: {len(expected)} vs {len(names)}")
    for name in sorted(names):
        target = dst / name
        if not target.exists():
            shutil.copy2(RAW_VAL / name, target)
    return dst


def category_summary(rows: list[dict[str, Any]], categories: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cat in categories:
        subset = [r for r in rows if str(r["category"]) == cat]
        out[cat] = {
            "findings": len(subset),
            "hits": sum(int(r["hit"]) for r in subset),
            "dice": float(np.mean([r["dice"] for r in subset])) if subset else None,
        }
    return out


def paired_bootstrap(base_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]], cases: int = 2000, seed: int = 20260905) -> dict[str, Any]:
    by_base: dict[str, list[float]] = {}
    by_other: dict[str, list[float]] = {}
    for row in base_rows:
        by_base.setdefault(row["case"], []).append(float(row["dice"]))
    for row in other_rows:
        by_other.setdefault(row["case"], []).append(float(row["dice"]))
    names = sorted(set(by_base) & set(by_other))
    rng = np.random.default_rng(seed)
    deltas = np.empty(cases, dtype=np.float64)
    for i in range(cases):
        chosen = rng.integers(0, len(names), size=len(names))
        b = np.concatenate([by_base[names[j]] for j in chosen])
        o = np.concatenate([by_other[names[j]] for j in chosen])
        deltas[i] = float(o.mean() - b.mean())
    return {
        "mean_delta": float(np.mean([r["dice"] for r in other_rows]) - np.mean([r["dice"] for r in base_rows])),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "resamples": cases,
        "seed": seed,
        "ct_cases": len(names),
    }


def make_val_report(base: Any, dirs: Mapping[str, Path]) -> dict[str, Any]:
    meta = metadata()
    results = {name: base.evaluate_directory(path, meta, "val") for name, path in dirs.items()}
    reference = results["baseline"]
    expected_dice = 0.33752638623854136
    expected_hits = 290
    if abs(reference["dice"] - expected_dice) > 1e-7 or reference["hits"] != expected_hits:
        raise RuntimeError(f"iso07 baseline mismatch: {reference['dice']} / {reference['hits']}")
    categories = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
    summary: dict[str, Any] = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "cache_manifest_sha256": sha256(CACHE / "manifest.json"),
        "mask_source_status": "PASS_PENDING_MANUAL_VISUAL_REVIEW",
        "baseline_equivalence_target": {"dice": expected_dice, "hits": expected_hits, "tolerance": 1e-7},
        "sets": {},
    }
    base_rows = reference["rows"]
    for name, result in results.items():
        rows = result["rows"]
        entry: dict[str, Any] = {
            "cases": result["cases"], "findings": result["findings"], "hits": result["hits"],
            "dice": result["dice"], "case_dice": result["case_dice"],
            "category": category_summary(rows, categories),
            "rows": rows,
        }
        if name != "baseline":
            base_by_id = {(r["case"], int(r["finding_id"])): r for r in base_rows}
            improved = unchanged = worsened = gained = lost = emptied = tp_removed = fp_removed = 0
            for row in rows:
                old = base_by_id[(row["case"], int(row["finding_id"]))]
                delta = float(row["dice"] - old["dice"])
                improved += int(delta > 1e-12); unchanged += int(abs(delta) <= 1e-12); worsened += int(delta < -1e-12)
                gained += int(bool(row["hit"]) and not bool(old["hit"]))
                lost += int(bool(old["hit"]) and not bool(row["hit"]))
                emptied += int(row["pred_voxels"] == 0 and old["pred_voxels"] > 0)
                tp_removed += max(0, int(old["tp"]) - int(row["tp"]))
                fp_removed += max(0, int(old["fp"]) - int(row["fp"]))
            entry.update({"improved": improved, "unchanged": unchanged, "worsened": worsened,
                          "hit_gains": gained, "hit_losses": lost, "predictions_emptied": emptied,
                          "tp_voxels_removed": tp_removed, "fp_voxels_removed": fp_removed,
                          "bootstrap": paired_bootstrap(base_rows, rows)})
        summary["sets"][name] = entry
    out = RUNTIME / "reports"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "val_metrics.json", summary)
    with (out / "val_findings.csv").open("w", newline="") as f:
        fields = ["set", "case", "finding_id", "category", "dice", "hit", "gt_voxels", "pred_voxels", "tp", "fp"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, result in results.items():
            for row in result["rows"]:
                writer.writerow({"set": name, **{k: row[k] for k in fields if k != "set"}})
    lines = [
        "# Exp025 iso07 anatomy audit — val200", "",
        "Official masks are retained in native CT geometry; iso07 predictions are restored to native space before masking.",
        "Mask source status: PASS_PENDING_MANUAL_VISUAL_REVIEW.", "",
        "| set | findings | hits | mean finding Dice | delta vs baseline | improved | unchanged | worsened |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        s = summary["sets"][name]
        lines.append(f"| {name} | {result['findings']} | {result['hits']} | {result['dice']:.9f} | {(result['dice']-reference['dice']):+.9f} | {s.get('improved','—')} | {s.get('unchanged','—')} | {s.get('worsened','—')} |")
    lines += ["", "## Policy accounting", "", "| set | hit gains | hit losses | emptied | TP removed | FP removed | bootstrap 95% CI |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, s in summary["sets"].items():
        if name == "baseline": continue
        ci = s["bootstrap"]
        lines.append(f"| {name} | {s['hit_gains']} | {s['hit_losses']} | {s['predictions_emptied']} | {s['tp_voxels_removed']} | {s['fp_voxels_removed']} | [{ci['ci95_low']:+.9f}, {ci['ci95_high']:+.9f}] |")
    lines += ["", "## Category results", "", "| set | category | findings | hits | Dice | delta vs baseline |", "|---|---|---:|---:|---:|---:|"]
    base_cat = summary["sets"]["baseline"]["category"]
    for name, s in summary["sets"].items():
        for cat, m in s["category"].items():
            if m["findings"]:
                lines.append(f"| {name} | {cat} | {m['findings']} | {m['hits']} | {m['dice']:.9f} | {(m['dice']-base_cat[cat]['dice']):+.9f} |")
    lines += ["", "The four comparisons are diagnostic fixed policies, not a validation-selected production rule. Clipping can only remove predicted voxels; it cannot recover missed target voxels."]
    (out / "val_report.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "val", "merge-test", "test-apply", "verify-test", "finalize"])
    parser.add_argument("--policy", choices=["unchanged", "whole_lung", "whole_lung_exact", "fine"])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    base = load_base()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    if args.stage == "preflight":
        if sha256(META) != cfg["metadata_sha256"]: raise SystemExit("metadata hash mismatch")
        if sha256(CHECKPOINT) != cfg["checkpoint_sha256"]: raise SystemExit("checkpoint hash mismatch")
        if sha256(CACHE / "manifest.json") != cfg["cache_manifest_sha256"]: raise SystemExit("cache manifest hash mismatch")
        ensure_routes(base)
        copy_val_baseline()
        print("PREFLIGHT_COMPLETE")
        return 0
    routes = ensure_routes(base)
    if args.stage == "merge-test":
        entries = metadata()["test"]
        expected = {e["name"] for e in entries}
        merged = RUNTIME / "test_raw/iso07_e150/predictions"
        merged.mkdir(parents=True, exist_ok=True)
        observed: dict[str, Path] = {}
        for shard in sorted((RUNTIME / "test_raw/iso07_e150").glob("shard[0-9]*")):
            if not shard.is_dir():
                continue
            for path in shard.glob("*.nii.gz"):
                if path.name in observed:
                    raise RuntimeError(f"duplicate test raw prediction: {path.name}")
                observed[path.name] = path
        if set(observed) != expected:
            missing = sorted(expected - set(observed))[:5]
            extra = sorted(set(observed) - expected)[:5]
            raise RuntimeError(f"test raw census mismatch: expected {len(expected)}, observed {len(observed)}, missing={missing}, extra={extra}")
        for name, src in observed.items():
            dst = merged / name
            if not dst.exists():
                shutil.copy2(src, dst)
        write_json(RUNTIME / "test_raw/iso07_e150/merge_manifest.json", {"cases": len(observed), "findings": sum(len(e["findings"]) for e in entries), "sources": {k: str(v) for k, v in observed.items()}})
        print("TEST_RAW_MERGED", len(observed))
        return 0
    if args.stage == "val":
        baseline = copy_val_baseline()
        masks = base.source_mask_map(SOURCE_VAL, "val")
        frozen = routes["val"]; route_map = {(r["case"], int(r["finding_id"])): r for r in frozen["rows"]}
        dirs = {"baseline": baseline, "exact_lung": RUNTIME / "validation/exact_lung", "whole_lung20": RUNTIME / "validation/whole_lung20", "fine20": RUNTIME / "validation/fine20"}
        if args.start is not None or args.end is not None:
            start = 0 if args.start is None else args.start
            end = len(metadata()["val"]) if args.end is None else args.end
            if not (0 <= start < end <= len(metadata()["val"])):
                raise SystemExit(f"invalid val slice {start}:{end}")
            for policy, key in [("whole_lung_exact", "exact_lung"), ("whole_lung", "whole_lung20"), ("fine", "fine20")]:
                base.transform_case_dir(baseline, dirs[key], route_map, policy, masks, metadata(), "val", start=start, end=end)
            print("VAL_SLICE_COMPLETE", start, end)
            return 0
        for policy, key in [("whole_lung_exact", "exact_lung"), ("whole_lung", "whole_lung20"), ("fine", "fine20")]:
            expected_names = {e["name"] for e in metadata()["val"]}
            observed_names = {p.name for p in dirs[key].glob("*.nii.gz")}
            if observed_names != expected_names:
                base.transform_case_dir(baseline, dirs[key], route_map, policy, masks, metadata(), "val")
            (dirs[key] / ".complete").write_text("complete\n")
        summary = make_val_report(base, dirs)
        write_json(RUNTIME / "validation_complete.json", {"status": "PASS", "sets": list(summary["sets"]), "report": str(RUNTIME / "reports/val_report.md")})
        print("VAL_COMPLETE", summary["sets"]["baseline"]["dice"])
        return 0
    if args.stage == "test-apply":
        if not args.policy: raise SystemExit("test-apply requires --policy")
        src = RUNTIME / "test_raw/iso07_e150/predictions"
        out_name = {"unchanged": "baseline", "whole_lung": "whole_lung20", "fine": "fine20"}[args.policy]
        dst = RUNTIME / "outputs" / out_name
        masks = base.source_mask_map(SOURCE_TEST, "test")
        frozen = routes["test"]; route_map = {(r["case"], int(r["finding_id"])): r for r in frozen["rows"]}
        start = args.start
        end = args.end
        if start is not None or end is not None:
            start = 0 if start is None else start
            end = len(metadata()["test"]) if end is None else end
            if not (0 <= start < end <= len(metadata()["test"])):
                raise SystemExit(f"invalid test slice {start}:{end}")
            base.transform_case_dir(src, dst, route_map, args.policy, masks, metadata(), "test", start=start, end=end)
            print("TEST_POLICY_SLICE_COMPLETE", out_name, start, end)
            return 0
        expected_names = {e["name"] for e in metadata()["test"]}
        if {p.name for p in dst.glob("*.nii.gz")} != expected_names:
            base.transform_case_dir(src, dst, route_map, args.policy, masks, metadata(), "test")
        (dst / ".complete").write_text("complete\n")
        print("TEST_POLICY_COMPLETE", out_name)
        return 0
    if args.stage == "verify-test":
        for name in ("baseline", "whole_lung20", "fine20"):
            result = base.validate_outputs(RUNTIME / "outputs" / name, "test", metadata())
            write_json(RUNTIME / "outputs" / f"{name}.verification.json", result)
            print(name, result["cases"], result["findings"])
        return 0
    if args.stage == "finalize":
        expected = metadata()["test"]
        records = {}
        for name in ("baseline", "whole_lung20", "fine20"):
            verification = RUNTIME / "outputs" / f"{name}.verification.json"
            if not verification.exists():
                raise SystemExit(f"missing verification manifest: {verification}")
            data = json.loads(verification.read_text())
            if data.get("cases") != 300 or data.get("findings") != 582:
                raise SystemExit(f"verification census mismatch for {name}: {data.get('cases')} / {data.get('findings')}")
            records[name] = {"directory": str(RUNTIME / "outputs" / name), "cases": data["cases"], "findings": data["findings"], "verification": str(verification)}
        report = [
            "# Exp025 iso07 test300 outputs", "",
            "Checkpoint: Exp017 continuation relative e050 / absolute e150.",
            "All predictions were restored to native CT geometry before anatomy support. Test ground truth was not read or created.",
            "Official mask source status: PASS_PENDING_MANUAL_VISUAL_REVIEW.", "",
            "| set | cases | findings | directory |", "|---|---:|---:|---|",
        ]
        for name, row in records.items():
            report.append(f"| {name} | {row['cases']} | {row['findings']} | `{row['directory']}` |")
        report += ["", "Each directory passed native CT shape, finding-axis, affine, binary uint8, and value-domain checks."]
        (RUNTIME / "reports").mkdir(parents=True, exist_ok=True)
        (RUNTIME / "reports/test_report.md").write_text("\n".join(report) + "\n")
        write_json(RUNTIME / "completion.json", {"status": "COMPLETE", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "cache_manifest_sha256": sha256(CACHE / "manifest.json"), "val_report": str(RUNTIME / "reports/val_report.md"), "test_report": str(RUNTIME / "reports/test_report.md"), "test_sets": records, "source_status": "PASS_PENDING_MANUAL_VISUAL_REVIEW"})
        print("EXP025_COMPLETE")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
