#!/usr/bin/env python3
"""Evaluate prompt-only whole-lung gating for exp010."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP010_REPO_DIR = REPO_ROOT / "experiments/010_voxtell_public_anatomy_prior_fusion"
EXP010_RUNTIME_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion"
)
EXP006_RUN_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "006_voxtell_cached_native_v123_lr_ablation/runs/"
    "exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/"
    "eval_epoch100_val200"
)
DEFAULT_VAL_JSON = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
DEFAULT_METADATA_JSON = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
DEFAULT_ANATOMY_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_total_fast_3mm_v2_16_0/cases"
)
DEFAULT_OVERLAP_CSV = EXP010_RUNTIME_DIR / "reports/target_anatomy_overlap.csv"
DEFAULT_RUNTIME_REPORT_DIR = EXP010_RUNTIME_DIR / "reports"
DEFAULT_REPO_REPORT = EXP010_REPO_DIR / "lung_gating_prompt_only_val200.md"

LUNG_LABELS = (10, 11, 12, 13, 14)
HIT_THRESHOLD = 0.1

ORACLE_REFERENCE = {
    "baseline_no_gate": {"dice": 0.32413407768565705, "hit_rate": 0.7611548556430446, "hits": 290},
    "oracle_100": {"dice": 0.33060180268452954, "hit_rate": 0.7690288713910761, "hits": 293},
    "oracle_95": {"dice": 0.3339882301495826, "hit_rate": 0.7742782152230971, "hits": 295},
    "oracle_90": {"dice": 0.3337842381871777, "hit_rate": 0.7742782152230971, "hits": 295},
}


INCLUDE_PATTERNS = {
    "lung": r"\blungs?\b",
    "lobe": r"\blobes?\b",
    "lingula": r"\blingul(?:a|ar)\b",
    "pulmonary": r"\bpulmonary\b",
    "parenchymal": r"\bparenchym(?:a|al)\b",
    "centrilobular": r"\bcentrilobular\b",
    "segment": (
        r"\b(?:segment|segments|apical|apicoposterior|anterior|posterior|superior|inferior|"
        r"anterobasal|posterobasal|laterobasal|lateral basal|medial basal|mediobasal|basal)\b"
    ),
}

EXCLUDE_PATTERNS = {
    "pleural": r"\b(?:pleura|pleural|pleuroparenchymal|subpleural|juxtapleural|hemithorax)\b",
    "effusion": r"\beffusions?\b",
    "pneumothorax": r"\bpneumothorax\b",
    "peripheral": r"\bperipheral(?:ly)?\b",
    "fissural": r"\b(?:fissure|fissural|perifissural)\b",
    "airway": (
        r"\b(?:airway|bronchial|bronchiectasis|bronchiectatic|bronchus|bronchi|"
        r"bronchovascular|peribronchial|trachea|tracheal|tree in bud)\b"
    ),
    "central_hilar_mediastinal": (
        r"\b(?:central|hilar|perihilar|mediastinal|paramediastinal|paracardiac|"
        r"pericardiac|retrocardiac)\b"
    ),
    "diaphragm_chest_wall": (
        r"\b(?:diaphragm|diaphragmatic|chest wall|chestwall|subcutaneous|"
        r"paravertebral|costovertebral|parasternal|rib|ribs|sternal|sternum|costal)\b"
    ),
    "lymph_node": r"\b(?:lymph|lymphadenopathy|adenopathy|node|nodes)\b",
    "extrapulmonary": r"\b(?:abdominal|abdomen|liver|spleen|adrenal|soft tissue)\b",
}

POLICY_EXCLUDES = {
    "strict_clean_lung": (
        "pleural",
        "effusion",
        "pneumothorax",
        "peripheral",
        "fissural",
        "airway",
        "central_hilar_mediastinal",
        "diaphragm_chest_wall",
        "lymph_node",
        "extrapulmonary",
    ),
    "allow_peripheral": (
        "pleural",
        "effusion",
        "pneumothorax",
        "fissural",
        "airway",
        "central_hilar_mediastinal",
        "diaphragm_chest_wall",
        "lymph_node",
        "extrapulmonary",
    ),
    "broad_lung_prompt": (
        "pleural",
        "effusion",
        "pneumothorax",
        "fissural",
        "central_hilar_mediastinal",
        "diaphragm_chest_wall",
        "lymph_node",
    ),
}
POLICIES = tuple(POLICY_EXCLUDES)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalize_prompt(prompt: str) -> str:
    text = prompt.lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def matched(patterns: dict[str, str], text: str) -> list[str]:
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def route_prompt(prompt: str, policy: str) -> tuple[bool, list[str], list[str]]:
    text = normalize_prompt(prompt)
    include = matched(INCLUDE_PATTERNS, text)
    exclude_all = matched(EXCLUDE_PATTERNS, text)
    active_exclude = [name for name in exclude_all if name in POLICY_EXCLUDES[policy]]
    return bool(include) and not active_exclude, include, active_exclude


def risk_groups(prompt: str) -> list[str]:
    text = normalize_prompt(prompt)
    groups = matched(EXCLUDE_PATTERNS, text)
    if re.search(r"\b(?:diffuse|diffusely|widespread|bilateral|both|multiple|multifocal|scattered|several|throughout)\b", text):
        groups.append("diffuse_multifocal")
    if not groups:
        groups.append("clean_or_unspecified")
    return sorted(set(groups))


def route_entries(entries: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, entry in enumerate(entries):
        findings = entry.get("findings", {})
        categories = entry.get("categories", {})
        for finding_key in sorted(findings, key=lambda value: int(value)):
            finding_id = int(finding_key)
            prompt = findings[finding_key]
            row: dict[str, Any] = {
                "split": split,
                "case_order": case_index,
                "case": entry["name"],
                "finding_id": finding_id,
                "category": categories.get(finding_key, ""),
                "finding": prompt,
                "risk_groups": "|".join(risk_groups(prompt)),
            }
            for policy in POLICIES:
                should_gate, include, exclude = route_prompt(prompt, policy)
                row[f"{policy}_gate"] = should_gate
                row[f"{policy}_include_terms"] = "|".join(include)
                row[f"{policy}_exclude_terms"] = "|".join(exclude)
            rows.append(row)
    return rows


def parser_unit_checks() -> None:
    checks = [
        ("Pulmonary nodule in the right upper lobe", "strict_clean_lung", True),
        ("Bilateral pulmonary nodules in both lungs", "strict_clean_lung", True),
        ("Subpleural nodule in the right lower lobe", "strict_clean_lung", False),
        ("Peripheral consolidation in the right lower lobe", "strict_clean_lung", False),
        ("Peripheral consolidation in the right lower lobe", "allow_peripheral", True),
        ("Pleural effusion in the right hemithorax", "broad_lung_prompt", False),
        ("Right pneumothorax", "broad_lung_prompt", False),
        ("Bronchiectasis involving the right middle lobe", "strict_clean_lung", False),
        ("Diffuse bronchial wall thickening", "strict_clean_lung", False),
        ("Paramediastinal opacity in the right middle lobe", "broad_lung_prompt", False),
        ("Hilar lymph node", "broad_lung_prompt", False),
        ("Chest wall mass", "broad_lung_prompt", False),
        ("Costovertebral nodule in the right lower lobe", "broad_lung_prompt", False),
    ]
    failures = []
    for prompt, policy, expected in checks:
        observed, _, _ = route_prompt(prompt, policy)
        if observed is not expected:
            failures.append(
                f"{policy}: {prompt!r} expected {expected} observed {observed}"
            )
    if failures:
        raise AssertionError("Parser unit checks failed:\n" + "\n".join(failures))


def case_key(filename: str) -> str:
    return filename[: -len(".nii.gz")] if filename.endswith(".nii.gz") else Path(filename).stem


def dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    first = mask1 > 0
    second = mask2 > 0
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    denominator = first.sum(dtype=np.int64) + second.sum(dtype=np.int64)
    if denominator == 0:
        return 1.0
    return float((2 * intersection + 1e-6) / (denominator + 1e-6))


def evaluate_case_masks(args: tuple[str, list[int], str, str, str]) -> list[dict[str, Any]]:
    case_name, finding_ids, seg_dir, prediction_dir, anatomy_root = args
    gt_img = nib.load(str(Path(seg_dir) / case_name))
    pred_img = nib.load(str(Path(prediction_dir) / case_name))
    anatomy_img = nib.load(
        str(Path(anatomy_root) / case_key(case_name) / "total_labels.nii.gz")
    )
    if tuple(gt_img.shape) != tuple(pred_img.shape):
        raise ValueError(f"{case_name}: GT/prediction shape mismatch {gt_img.shape} vs {pred_img.shape}")
    if len(gt_img.shape) != 4 or tuple(gt_img.shape[1:]) != tuple(anatomy_img.shape):
        raise ValueError(
            f"{case_name}: GT/anatomy shape mismatch {gt_img.shape} vs {anatomy_img.shape}"
        )

    anatomy = np.asanyarray(anatomy_img.dataobj).astype(np.uint8, copy=False)
    whole_lung = np.isin(anatomy, LUNG_LABELS)
    rows = []
    for finding_id in finding_ids:
        gt = np.asanyarray(gt_img.dataobj[finding_id]).astype(np.uint8, copy=False)
        pred = np.asanyarray(pred_img.dataobj[finding_id]).astype(np.uint8, copy=False)
        original_dice = dice(gt, pred)
        gated_dice = dice(gt, np.logical_and(pred > 0, whole_lung))
        rows.append(
            {
                "case": case_name,
                "finding_id": finding_id,
                "recomputed_original_dice": original_dice,
                "whole_lung_gated_dice": gated_dice,
            }
        )
    return rows


def load_overlap(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["case"], int(row["finding_id"]))
            output[key] = {
                "target_voxels": int(row["target_voxels"]),
                "whole_lung_target_fraction": float(row["whole_lung_target_fraction"]),
                "best_overlapping_label": row["best_overlapping_label"],
                "best_label_target_fraction": float(row["best_label_target_fraction"]),
                "background_target_fraction": float(row["background_target_fraction"]),
            }
    return output


def load_original_eval(path: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for case in payload["cases"]:
        case_name = case["file"]
        for finding_name, metrics in case["findings"].items():
            finding_id = int(finding_name.split("_")[-1])
            rows[(case_name, finding_id)] = {
                "original_dice": float(metrics["global_dice"]),
                "original_hit": bool(metrics["global_hit"]),
            }
    return payload["summary"], rows


def materialize_mask_metrics(
    val_rows: list[dict[str, Any]],
    seg_dir: Path,
    prediction_dir: Path,
    anatomy_root: Path,
    workers: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    case_map: dict[str, list[int]] = defaultdict(list)
    for row in val_rows:
        case_map[row["case"]].append(int(row["finding_id"]))

    results: dict[tuple[str, int], dict[str, Any]] = {}
    tasks = [
        (case_name, sorted(finding_ids), str(seg_dir), str(prediction_dir), str(anatomy_root))
        for case_name, finding_ids in sorted(case_map.items())
    ]
    worker_count = max(1, min(workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(evaluate_case_masks, task) for task in tasks]
        for future in as_completed(futures):
            for row in future.result():
                results[(row["case"], int(row["finding_id"]))] = row
    return results


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_final_rows(rows: list[dict[str, Any]], dice_key: str, hit_key: str) -> dict[str, Any]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(float(row[dice_key]))
    return {
        "total_cases": len(by_case),
        "total_findings": len(rows),
        "total_hits": int(sum(bool(row[hit_key]) for row in rows)),
        "total_misses": int(sum(not bool(row[hit_key]) for row in rows)),
        "hit_rate": mean([1.0 if row[hit_key] else 0.0 for row in rows]),
        "mean_global_dice_per_finding": mean([float(row[dice_key]) for row in rows]),
        "mean_global_dice_per_case": mean([mean(values) for values in by_case.values()]),
    }


def stratum_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    if int(row["target_voxels"]) <= 64:
        flags.append("small_gt_le64vox")
    groups = str(row["risk_groups"]).split("|")
    if any(group in groups for group in ["pleural", "peripheral", "fissural", "diaphragm_chest_wall"]):
        flags.append("pleural_or_boundary_language")
    if "airway" in groups:
        flags.append("airway_language")
    if "diffuse_multifocal" in groups:
        flags.append("diffuse_multifocal_language")
    return flags or ["none"]


def stratum_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for flag in stratum_flags(row):
            grouped[flag].append(row)
    return {
        key: {
            "findings": len(items),
            "mean_original_dice": mean([float(item["original_dice"]) for item in items]),
            "mean_after_dice": mean([float(item["whole_lung_gated_dice"]) for item in items]),
            "mean_delta": mean(
                [
                    float(item["whole_lung_gated_dice"]) - float(item["original_dice"])
                    for item in items
                ]
            ),
            "hit_gains": int(
                sum((not bool(item["original_hit"])) and bool(item["whole_lung_gated_hit"]) for item in items)
            ),
            "hit_losses": int(
                sum(bool(item["original_hit"]) and (not bool(item["whole_lung_gated_hit"])) for item in items)
            ),
        }
        for key, items in sorted(grouped.items())
    }


def category_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    return {
        key: {
            "findings": len(items),
            "mean_original_dice": mean([float(item["original_dice"]) for item in items]),
            "mean_after_dice": mean([float(item["whole_lung_gated_dice"]) for item in items]),
            "mean_delta": mean(
                [
                    float(item["whole_lung_gated_dice"]) - float(item["original_dice"])
                    for item in items
                ]
            ),
            "hit_gains": int(
                sum((not bool(item["original_hit"])) and bool(item["whole_lung_gated_hit"]) for item in items)
            ),
            "hit_losses": int(
                sum(bool(item["original_hit"]) and (not bool(item["whole_lung_gated_hit"])) for item in items)
            ),
        }
        for key, items in sorted(grouped.items())
    }


def containment_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    target_voxels = [int(row["target_voxels"]) for row in selected]
    fractions = [float(row["whole_lung_target_fraction"]) for row in selected]
    aggregate = (
        float(
            sum(voxels * fraction for voxels, fraction in zip(target_voxels, fractions, strict=True))
            / sum(target_voxels)
        )
        if target_voxels and sum(target_voxels) > 0
        else 0.0
    )
    return {
        "selected_findings": len(selected),
        "aggregate_target_voxel_retention": aggregate,
        "mean_target_fraction": mean(fractions),
        "target_100": int(sum(fraction >= 1.0 for fraction in fractions)),
        "target_ge95": int(sum(fraction >= 0.95 for fraction in fractions)),
        "target_ge90": int(sum(fraction >= 0.90 for fraction in fractions)),
        "target_ge50": int(sum(fraction >= 0.50 for fraction in fractions)),
        "target_removed_completely": int(sum(fraction <= 0.0 for fraction in fractions)),
        "min_target_fraction": min(fractions) if fractions else 0.0,
    }


def evaluate_policies(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    original_summary: dict[str, Any],
) -> dict[str, Any]:
    policy_results: dict[str, Any] = {}
    baseline_rows = [
        {**row, "final_dice": float(row["original_dice"]), "final_hit": bool(row["original_hit"])}
        for row in val_rows
    ]
    baseline = summarize_final_rows(baseline_rows, "final_dice", "final_hit")

    for policy in POLICIES:
        selected = [row for row in val_rows if row[f"{policy}_gate"]]
        test_selected = [row for row in test_rows if row[f"{policy}_gate"]]
        final_rows = []
        for row in val_rows:
            gated = bool(row[f"{policy}_gate"])
            final_dice = (
                float(row["whole_lung_gated_dice"])
                if gated
                else float(row["original_dice"])
            )
            final_rows.append(
                {
                    **row,
                    "final_dice": final_dice,
                    "final_hit": final_dice >= HIT_THRESHOLD,
                    "selected_for_policy": gated,
                }
            )
        summary = summarize_final_rows(final_rows, "final_dice", "final_hit")
        improved = int(
            sum(float(row["whole_lung_gated_dice"]) > float(row["original_dice"]) + 1e-12 for row in selected)
        )
        worsened = int(
            sum(float(row["whole_lung_gated_dice"]) < float(row["original_dice"]) - 1e-12 for row in selected)
        )
        unchanged = len(selected) - improved - worsened
        hit_gain = int(
            sum((not bool(row["original_hit"])) and bool(row["whole_lung_gated_hit"]) for row in selected)
        )
        hit_loss = int(
            sum(bool(row["original_hit"]) and (not bool(row["whole_lung_gated_hit"])) for row in selected)
        )
        containment = containment_summary(selected)
        monitored = stratum_summary(selected)
        material_worsen = [
            name
            for name, item in monitored.items()
            if item["findings"] >= 5 and (item["hit_losses"] > 0 or item["mean_delta"] < -0.005)
        ]
        improves_metric = (
            summary["mean_global_dice_per_finding"] > baseline["mean_global_dice_per_finding"]
            or summary["total_hits"] > baseline["total_hits"]
        )
        passes_core_gate = (
            improves_metric
            and containment["target_removed_completely"] == 0
            and containment["aggregate_target_voxel_retention"] >= 0.99
            and not material_worsen
        )
        clipped_examples = [
            row
            for row in selected
            if float(row["whole_lung_target_fraction"]) < 0.95
        ]
        clipped_examples.sort(
            key=lambda row: (
                float(row["whole_lung_target_fraction"]),
                float(row["whole_lung_gated_dice"]) - float(row["original_dice"]),
            )
        )
        policy_results[policy] = {
            "val_selected": len(selected),
            "test_selected": len(test_selected),
            "val_total": len(val_rows),
            "test_total": len(test_rows),
            "summary": summary,
            "delta_vs_baseline": {
                "dice_per_finding": summary["mean_global_dice_per_finding"]
                - baseline["mean_global_dice_per_finding"],
                "dice_per_case": summary["mean_global_dice_per_case"]
                - baseline["mean_global_dice_per_case"],
                "hits": summary["total_hits"] - baseline["total_hits"],
            },
            "selected_subset": {
                "original_dice": mean([float(row["original_dice"]) for row in selected]),
                "after_dice": mean([float(row["whole_lung_gated_dice"]) for row in selected]),
                "original_hits": int(sum(bool(row["original_hit"]) for row in selected)),
                "after_hits": int(sum(bool(row["whole_lung_gated_hit"]) for row in selected)),
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "hit_gains": hit_gain,
                "hit_losses": hit_loss,
            },
            "containment": containment,
            "by_category_selected": category_summary(selected),
            "monitored_strata_selected": monitored,
            "material_worsen_strata": material_worsen,
            "passes_deployment_candidate_gate": passes_core_gate,
            "clipped_examples_lt95": [
                {
                    "case": row["case"],
                    "finding_id": row["finding_id"],
                    "category": row["category"],
                    "whole_lung_target_fraction": row["whole_lung_target_fraction"],
                    "original_dice": row["original_dice"],
                    "after_dice": row["whole_lung_gated_dice"],
                    "finding": row["finding"],
                    "risk_groups": row["risk_groups"],
                }
                for row in clipped_examples[:10]
            ],
        }
    candidates = [
        (policy, result)
        for policy, result in policy_results.items()
        if result["passes_deployment_candidate_gate"]
    ]
    candidates.sort(
        key=lambda item: item[1]["summary"]["mean_global_dice_per_finding"],
        reverse=True,
    )
    return {
        "baseline_recomputed": baseline,
        "baseline_saved": original_summary,
        "policies": policy_results,
        "selected_deployment_candidate": candidates[0][0] if candidates else None,
    }


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    baseline = payload["results"]["baseline_recomputed"]
    policies = payload["results"]["policies"]
    candidate = payload["results"]["selected_deployment_candidate"]
    lines = [
        "---",
        "created: 2026-07-27",
        "updated: 2026-07-27",
        "status: preliminary_result",
        "experiment_id: 010_voxtell_public_anatomy_prior_fusion",
        "source_prediction_experiment: 006_voxtell_cached_native_v123_lr_ablation",
        "---",
        "",
        "# Prompt-Only Whole-Lung Gating On Exp006 v123_cached_e5_d4 Val200",
        "",
        "## Summary",
        "",
        "This report evaluates whether deterministic prompt-only rules can choose",
        "which findings are safe for hard whole-lung post-processing. Routes are",
        "computed from the free-text finding string before any ground-truth",
        "containment or prediction metrics are read.",
        "",
        "The tested post-processing is:",
        "",
        "```text",
        "prediction := prediction AND TotalSegmentator whole_lung_mask",
        "```",
        "",
        "The primary conclusion is:",
        "",
    ]
    if candidate:
        lines.append(
            f"- Deployment-candidate gate selected: `{candidate}` based on the predeclared core checks."
        )
    else:
        lines.append(
            "- No prompt-only hard whole-lung policy passed the predeclared deployment-candidate checks."
        )
        lines.append(
            "- Treat hard whole-lung gating as a validation-only finding for now; prefer soft anatomy fusion or a learned/selective policy."
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Fixed validation JSON: `{payload['inputs']['val_json']}`.",
            f"- Test prompt source: `{payload['inputs']['metadata_json']}`.",
            f"- Prediction source: `{payload['inputs']['prediction_dir']}`.",
            f"- Anatomy cache: `{payload['inputs']['anatomy_root']}`.",
            "- Whole-lung labels: TotalSegmentator `{10, 11, 12, 13, 14}`.",
            f"- Frozen router JSON SHA256: `{payload['router']['router_json_sha256']}`.",
            "",
            "## Prompt-Only Policy Results",
            "",
            "| Policy | Val selected | Test selected | Dice/finding | Delta Dice | Hit rate | Hits | Target retained | Removed targets | Passes gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            (
                f"| Baseline, no gate | 0 / {baseline['total_findings']} | - | "
                f"{format_float(baseline['mean_global_dice_per_finding'])} | 0 | "
                f"{format_float(baseline['hit_rate'])} | {baseline['total_hits']} / {baseline['total_findings']} | "
                "- | - | - |"
            ),
        ]
    )
    for policy in POLICIES:
        result = policies[policy]
        summary = result["summary"]
        containment = result["containment"]
        delta = result["delta_vs_baseline"]
        lines.append(
            f"| `{policy}` | {result['val_selected']} / {result['val_total']} | "
            f"{result['test_selected']} / {result['test_total']} | "
            f"{format_float(summary['mean_global_dice_per_finding'])} | "
            f"{delta['dice_per_finding']:+.4f} | {format_float(summary['hit_rate'])} | "
            f"{summary['total_hits']} / {summary['total_findings']} | "
            f"{format_float(containment['aggregate_target_voxel_retention'])} | "
            f"{containment['target_removed_completely']} | "
            f"{'yes' if result['passes_deployment_candidate_gate'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Oracle Reference",
            "",
            "| Setting | Dice/finding | Hit rate | Hits |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Baseline, no gate | {format_float(ORACLE_REFERENCE['baseline_no_gate']['dice'])} | "
                f"{format_float(ORACLE_REFERENCE['baseline_no_gate']['hit_rate'])} | "
                f"{ORACLE_REFERENCE['baseline_no_gate']['hits']} / 381 |"
            ),
            (
                f"| Oracle 100% contained | {format_float(ORACLE_REFERENCE['oracle_100']['dice'])} | "
                f"{format_float(ORACLE_REFERENCE['oracle_100']['hit_rate'])} | "
                f"{ORACLE_REFERENCE['oracle_100']['hits']} / 381 |"
            ),
            (
                f"| Oracle >=95% contained | {format_float(ORACLE_REFERENCE['oracle_95']['dice'])} | "
                f"{format_float(ORACLE_REFERENCE['oracle_95']['hit_rate'])} | "
                f"{ORACLE_REFERENCE['oracle_95']['hits']} / 381 |"
            ),
            (
                f"| Oracle >=90% contained | {format_float(ORACLE_REFERENCE['oracle_90']['dice'])} | "
                f"{format_float(ORACLE_REFERENCE['oracle_90']['hit_rate'])} | "
                f"{ORACLE_REFERENCE['oracle_90']['hits']} / 381 |"
            ),
            "",
            "## Selected-Subset Behavior",
            "",
            "| Policy | Dice before -> after | Hits before -> after | Improved | Worsened | Hit gains/losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICIES:
        subset = policies[policy]["selected_subset"]
        lines.append(
            f"| `{policy}` | {format_float(subset['original_dice'])} -> "
            f"{format_float(subset['after_dice'])} | {subset['original_hits']} -> "
            f"{subset['after_hits']} | {subset['improved']} | {subset['worsened']} | "
            f"+{subset['hit_gains']} / -{subset['hit_losses']} |"
        )

    lines.extend(
        [
            "",
            "## Target Containment",
            "",
            "| Policy | Selected | 100% | >=95% | >=90% | >=50% | Min fraction | Mean fraction | Aggregate retained |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICIES:
        containment = policies[policy]["containment"]
        selected = containment["selected_findings"]
        lines.append(
            f"| `{policy}` | {selected} | {containment['target_100']} | "
            f"{containment['target_ge95']} | {containment['target_ge90']} | "
            f"{containment['target_ge50']} | {format_float(containment['min_target_fraction'])} | "
            f"{format_float(containment['mean_target_fraction'])} | "
            f"{format_float(containment['aggregate_target_voxel_retention'])} |"
        )

    lines.extend(
        [
            "",
            "## Clipped Examples",
            "",
            "Examples below are selected findings whose target had less than 95% whole-lung containment.",
            "",
        ]
    )
    for policy in POLICIES:
        examples = policies[policy]["clipped_examples_lt95"]
        lines.extend([f"### `{policy}`", ""])
        if not examples:
            lines.extend(["No selected examples below 95% containment.", ""])
            continue
        lines.extend(
            [
                "| Case | Finding | Category | Target in lung | Dice before -> after | Prompt |",
                "| --- | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for example in examples[:6]:
            prompt = str(example["finding"]).replace("|", "/")
            lines.append(
                f"| {example['case']} | {example['finding_id']} | {example['category']} | "
                f"{format_float(example['whole_lung_target_fraction'])} | "
                f"{format_float(example['original_dice'])} -> {format_float(example['after_dice'])} | "
                f"{prompt} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
        ]
    )
    if candidate:
        lines.append(
            f"`{candidate}` is the current prompt-only deployment candidate, but it should still be treated as preliminary."
        )
    else:
        lines.append(
            "Prompt-only hard whole-lung gating is not safe enough as a test-default policy under the predeclared checks."
        )
    lines.extend(
        [
            "",
            "Compared with the oracle gate, prompt-only routing has to trade off recall",
            "risk against false-positive removal without seeing target containment.",
            "The safest next direction is soft anatomy fusion or learned selective",
            "routing, with hard post-processing reserved for narrowly validated",
            "prompt cohorts.",
            "",
            "## Runtime Artifacts",
            "",
            f"- Frozen router JSON: `{payload['outputs']['router_json']}`.",
            f"- Frozen router CSV: `{payload['outputs']['router_csv']}`.",
            f"- Per-finding CSV: `{payload['outputs']['per_finding_csv']}`.",
            f"- Summary JSON: `{payload['outputs']['summary_json']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--eval-json", type=Path, default=EXP006_RUN_DIR / "eval/val_quick_global_eval.json")
    parser.add_argument("--prediction-dir", type=Path, default=EXP006_RUN_DIR / "predictions")
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--anatomy-root", type=Path, default=DEFAULT_ANATOMY_ROOT)
    parser.add_argument("--overlap-csv", type=Path, default=DEFAULT_OVERLAP_CSV)
    parser.add_argument("--runtime-report-dir", type=Path, default=DEFAULT_RUNTIME_REPORT_DIR)
    parser.add_argument("--repo-report-md", type=Path, default=DEFAULT_REPO_REPORT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite-router", action="store_true")
    args = parser.parse_args()

    parser_unit_checks()

    val_payload = json.loads(args.val_json.read_text())
    val_entries = val_payload["test"]
    metadata = json.loads(args.metadata_json.read_text())
    test_entries = metadata["test"]

    runtime_report_dir = args.runtime_report_dir
    router_json = runtime_report_dir / "prompt_only_lung_gate_router_frozen.json"
    router_csv = runtime_report_dir / "prompt_only_lung_gate_router_frozen.csv"
    summary_json = runtime_report_dir / "prompt_only_lung_gating_val200_summary.json"
    per_finding_csv = runtime_report_dir / "prompt_only_lung_gating_val200_per_finding.csv"

    if router_json.exists() and not args.overwrite_router:
        router_payload = json.loads(router_json.read_text())
        val_route_rows = router_payload["val_routes"]
        test_route_rows = router_payload["test_routes"]
    else:
        val_route_rows = route_entries(val_entries, "val200")
        test_route_rows = route_entries(test_entries, "test")
        router_payload = {
            "created_at_utc": utc_now(),
            "purpose": "prompt-only whole-lung hard-gate routing; no GT or prediction metrics",
            "policies": list(POLICIES),
            "policy_excludes": {key: list(value) for key, value in POLICY_EXCLUDES.items()},
            "include_patterns": INCLUDE_PATTERNS,
            "exclude_patterns": EXCLUDE_PATTERNS,
            "val_json": str(args.val_json),
            "val_json_sha256": sha256_file(args.val_json),
            "metadata_json": str(args.metadata_json),
            "metadata_json_sha256": sha256_file(args.metadata_json),
            "val_routes": val_route_rows,
            "test_routes": test_route_rows,
        }
        write_json_atomic(router_json, router_payload)
        route_fieldnames = list(val_route_rows[0].keys())
        write_csv(router_csv, val_route_rows + test_route_rows, route_fieldnames)

    router_sha = sha256_file(router_json)

    overlap = load_overlap(args.overlap_csv)
    original_summary, original_rows = load_original_eval(args.eval_json)
    mask_metrics = materialize_mask_metrics(
        val_route_rows,
        args.seg_dir,
        args.prediction_dir,
        args.anatomy_root,
        args.workers,
    )

    val_rows = []
    mismatch_count = 0
    for row in val_route_rows:
        key = (row["case"], int(row["finding_id"]))
        enriched = dict(row)
        enriched.update(overlap[key])
        enriched.update(original_rows[key])
        enriched.update(mask_metrics[key])
        enriched["whole_lung_gated_hit"] = (
            float(enriched["whole_lung_gated_dice"]) >= HIT_THRESHOLD
        )
        if abs(float(enriched["original_dice"]) - float(enriched["recomputed_original_dice"])) > 1e-7:
            mismatch_count += 1
        val_rows.append(enriched)
    if mismatch_count:
        raise RuntimeError(f"{mismatch_count} recomputed original Dice values differed from saved eval JSON")

    results = evaluate_policies(val_rows, test_route_rows, original_summary)
    saved = results["baseline_saved"]
    recomputed = results["baseline_recomputed"]
    if (
        abs(saved["mean_global_dice_per_finding"] - recomputed["mean_global_dice_per_finding"]) > 1e-7
        or saved["total_hits"] != recomputed["total_hits"]
    ):
        raise RuntimeError(
            "No-gate recomputation does not match saved exp006 summary: "
            f"saved={saved} recomputed={recomputed}"
        )

    payload = {
        "created_at_utc": utc_now(),
        "inputs": {
            "val_json": str(args.val_json),
            "metadata_json": str(args.metadata_json),
            "eval_json": str(args.eval_json),
            "prediction_dir": str(args.prediction_dir),
            "seg_dir": str(args.seg_dir),
            "anatomy_root": str(args.anatomy_root),
            "overlap_csv": str(args.overlap_csv),
        },
        "outputs": {
            "router_json": str(router_json),
            "router_csv": str(router_csv),
            "summary_json": str(summary_json),
            "per_finding_csv": str(per_finding_csv),
            "repo_report_md": str(args.repo_report_md),
        },
        "router": {
            "router_json_sha256": router_sha,
            "router_rows_val": len(val_route_rows),
            "router_rows_test": len(test_route_rows),
        },
        "oracle_reference": ORACLE_REFERENCE,
        "results": results,
    }
    write_json_atomic(summary_json, payload)

    per_finding_fieldnames = [
        "case",
        "finding_id",
        "category",
        "finding",
        "risk_groups",
        "target_voxels",
        "whole_lung_target_fraction",
        "original_dice",
        "whole_lung_gated_dice",
        "original_hit",
        "whole_lung_gated_hit",
        "strict_clean_lung_gate",
        "allow_peripheral_gate",
        "broad_lung_prompt_gate",
    ]
    write_csv(
        per_finding_csv,
        [
            {
                key: row[key]
                for key in per_finding_fieldnames
            }
            for row in val_rows
        ],
        per_finding_fieldnames,
    )

    report = render_report(payload)
    args.repo_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report_md.write_text(report)
    print(json.dumps(payload["results"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
