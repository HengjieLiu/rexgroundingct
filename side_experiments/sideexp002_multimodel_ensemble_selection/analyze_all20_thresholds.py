#!/usr/bin/env python3
"""Probe fixed thresholds for the uniform probability ensemble of all 20 models."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_candidates import (
    atomic_write_json,
    atomic_write_text,
    load_manifest,
    sha256_file,
)
from summarize_baseline_categories import (
    CATEGORY_LABELS,
    DIFFUSE_CATEGORIES,
    FOCAL_CATEGORIES,
    support_note,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "candidate_manifest.json"
DEFAULT_DATASET = (
    REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp002_multimodel_ensemble_selection"
)
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
DEFAULT_OUTPUT_JSON = (
    HERE / "outputs" / "all20_uniform_probability_threshold_summary.json"
)
DEFAULT_OUTPUT_MD = (
    HERE / "outputs" / "all20_uniform_probability_threshold_report.md"
)
ANALYSIS_ID = "sideexp002_all20_uniform_probability_threshold_sweep"
EXPECTED_MODELS = 20
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
EXPECTED_ARRAYS = EXPECTED_MODELS * EXPECTED_CASES
HIT_THRESHOLD = 0.1
DICE_EPSILON = 1e-6
RECOMPOSITION_TOLERANCE = 1e-12
THRESHOLDS = (
    0.10,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def threshold_key(value: float) -> str:
    return f"{value:.2f}"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_cases(
    dataset_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], str],
    dict[str, int],
]:
    data = read_json(dataset_path)
    cases = data.get("test")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"{dataset_path}: expected {EXPECTED_CASES} cases under 'test'"
        )
    category_by_key: dict[tuple[str, int], str] = {}
    supports = {code: 0 for code in CATEGORY_LABELS}
    seen_cases: set[str] = set()
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or name in seen_cases
        ):
            raise ValueError(f"{dataset_path}: duplicate or invalid case {name!r}")
        seen_cases.add(name)
        if (
            not isinstance(findings, dict)
            or not isinstance(categories, dict)
            or set(findings) != set(categories)
        ):
            raise ValueError(f"{dataset_path} {name}: finding/category mismatch")
        try:
            finding_keys = sorted(findings, key=int)
        except ValueError as exc:
            raise ValueError(
                f"{dataset_path} {name}: finding keys must be integers"
            ) from exc
        for finding_key in finding_keys:
            finding_index = int(finding_key)
            category = str(categories[finding_key])
            if category not in CATEGORY_LABELS:
                raise ValueError(
                    f"{dataset_path} {name} finding {finding_key}: "
                    f"unknown category {category!r}"
                )
            key = (name, finding_index)
            if key in category_by_key:
                raise ValueError(f"{dataset_path}: duplicate finding {key}")
            category_by_key[key] = category
            supports[category] += 1
    if len(category_by_key) != EXPECTED_FINDINGS:
        raise ValueError(
            f"{dataset_path}: expected {EXPECTED_FINDINGS} findings, "
            f"got {len(category_by_key)}"
        )
    if sum(supports.values()) != EXPECTED_FINDINGS:
        raise ValueError(f"{dataset_path}: category supports do not sum to 381")
    return cases, category_by_key, supports


def array_path(runtime_root: Path, candidate_id: str, case_name: str) -> Path:
    return runtime_root / "logits" / candidate_id / "cases" / f"{case_name}.npy"


def strict_gate_accepted(gate: dict[str, Any]) -> bool:
    return (
        gate.get("status") == "passed"
        and gate.get("storage_reproduction_status") == "passed"
        and gate.get("same_pass_mask_reproduction_status") == "passed"
        and gate.get("historical_reproduction_is_gate") is False
        and gate.get("array_hashes_verified") is True
    )


def gate_warning_reason(gate: dict[str, Any]) -> str:
    reasons = []
    for field in (
        "status",
        "storage_reproduction_status",
        "same_pass_mask_reproduction_status",
        "historical_reproduction_status",
    ):
        value = gate.get(field)
        if value not in {None, "passed"}:
            reasons.append(f"{field}={value}")
    reasons.extend(str(value) for value in gate.get("warnings", []))
    if not reasons:
        return "-"
    return "; ".join(dict.fromkeys(reasons))


def validate_array_record(
    path: Path,
    record: dict[str, Any],
    expected_findings: int,
) -> tuple[int, ...]:
    if record.get("status") != "complete":
        raise ValueError(f"{path}: incomplete case record")
    shape_value = record.get("shape")
    if (
        not isinstance(shape_value, list)
        or not shape_value
        or any(not isinstance(value, int) or value <= 0 for value in shape_value)
    ):
        raise ValueError(f"{path}: invalid recorded shape")
    shape = tuple(shape_value)
    if shape[0] != expected_findings:
        raise ValueError(
            f"{path}: expected {expected_findings} finding planes, got {shape[0]}"
        )
    dtype = str(record.get("dtype"))
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"{path}: unsupported recorded dtype {dtype}")
    if not isinstance(record.get("array_sha256"), str):
        raise ValueError(f"{path}: missing recorded array SHA256")
    if not path.is_file():
        raise FileNotFoundError(f"missing logit array: {path}")
    expected_bytes = int(record.get("npy_bytes", -1))
    if expected_bytes <= 0 or path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: file size differs from recorded npy_bytes {expected_bytes}"
        )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if tuple(array.shape) != shape:
            raise ValueError(
                f"{path}: array shape {tuple(array.shape)} != recorded {shape}"
            )
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"{path}: array dtype {array.dtype} != recorded {dtype}"
            )
    finally:
        del array
    return shape


def validate_exports(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    dataset_sha256: str,
    runtime_root: Path,
    allow_unaccepted: bool,
) -> dict[str, Any]:
    candidates = manifest["candidates"]
    if len(candidates) != EXPECTED_MODELS:
        raise ValueError(f"expected exactly {EXPECTED_MODELS} candidates")
    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    if len(set(candidate_ids)) != EXPECTED_MODELS:
        raise ValueError("candidate IDs are not unique")
    dataset_contract = manifest.get("dataset")
    if not isinstance(dataset_contract, dict):
        raise ValueError("candidate manifest has no dataset contract")
    if dataset_contract.get("sha256") != dataset_sha256:
        raise ValueError(
            "dataset SHA256 drifted from the immutable candidate manifest"
        )
    if (
        int(dataset_contract.get("cases", -1)) != EXPECTED_CASES
        or int(dataset_contract.get("findings", -1)) != EXPECTED_FINDINGS
    ):
        raise ValueError("candidate manifest has invalid dataset counts")

    case_by_name = {case["name"]: case for case in cases}
    expected_names = set(case_by_name)
    roster = []
    inventory = []
    case_records: dict[str, dict[str, dict[str, Any]]] = {}
    total_arrays = 0
    for candidate in sorted(candidates, key=lambda value: value["id"]):
        candidate_id = candidate["id"]
        root = runtime_root / "logits" / candidate_id
        export_path = root / "export_manifest.json"
        gate_path = root / "reproduction_validation.json"
        if not export_path.is_file() or not gate_path.is_file():
            raise FileNotFoundError(
                f"{candidate_id}: missing export or reproduction manifest"
            )
        export = read_json(export_path)
        gate = read_json(gate_path)
        if (
            export.get("status") != "complete"
            or export.get("candidate_id") != candidate_id
            or int(export.get("case_count", -1)) != EXPECTED_CASES
            or int(export.get("finding_count", -1)) != EXPECTED_FINDINGS
            or export.get("dataset_sha256") != dataset_sha256
        ):
            raise ValueError(f"{candidate_id}: incomplete or mismatched export")
        if export.get("candidate_source") != candidate:
            raise ValueError(
                f"{candidate_id}: exported candidate source differs from manifest"
            )
        if gate.get("candidate_id") != candidate_id:
            raise ValueError(f"{candidate_id}: reproduction manifest ID mismatch")
        if gate.get("array_hashes_verified") is not True:
            raise ValueError(
                f"{candidate_id}: recorded array-hash verification is not true"
            )
        accepted = strict_gate_accepted(gate)
        if not accepted and not allow_unaccepted:
            raise ValueError(
                f"{candidate_id}: strict gate failed; use --allow-unaccepted "
                "for the approved diagnostic override"
            )
        records = export.get("cases")
        if not isinstance(records, list) or len(records) != EXPECTED_CASES:
            raise ValueError(f"{candidate_id}: expected 200 exported case records")
        names = [record.get("name") for record in records]
        if len(set(names)) != EXPECTED_CASES or set(names) != expected_names:
            raise ValueError(f"{candidate_id}: duplicate or missing exported cases")
        dtype = str(export.get("dtype"))
        if dtype not in {"float16", "float32"}:
            raise ValueError(f"{candidate_id}: unsupported export dtype {dtype}")
        by_name = {}
        for record in records:
            name = record["name"]
            if str(record.get("dtype")) != dtype:
                raise ValueError(f"{candidate_id} {name}: mixed export dtype")
            path = array_path(runtime_root, candidate_id, name)
            shape = validate_array_record(
                path,
                record,
                len(case_by_name[name]["findings"]),
            )
            by_name[name] = record
            inventory.append(
                {
                    "candidate_id": candidate_id,
                    "case": name,
                    "array_sha256": record["array_sha256"],
                    "npy_bytes": int(record["npy_bytes"]),
                    "shape": list(shape),
                    "dtype": dtype,
                }
            )
            total_arrays += 1
        case_records[candidate_id] = by_name
        roster.append(
            {
                "candidate_id": candidate_id,
                "experiment_number": int(candidate["experiment_number"]),
                "experiment_name": candidate["experiment_name"],
                "model_id": candidate["model_id"],
                "epoch": int(candidate["epoch"]),
                "architecture": candidate["architecture"],
                "dtype": dtype,
                "weight": 1.0 / EXPECTED_MODELS,
                "strict_gate_accepted": accepted,
                "gate_status": gate.get("status"),
                "storage_reproduction_status": gate.get(
                    "storage_reproduction_status"
                ),
                "same_pass_mask_reproduction_status": gate.get(
                    "same_pass_mask_reproduction_status"
                ),
                "historical_reproduction_status": gate.get(
                    "historical_reproduction_status"
                ),
                "warning_reason": gate_warning_reason(gate),
                "checkpoint_sha256": candidate["checkpoint"]["sha256"],
                "export_manifest_sha256": sha256_file(export_path),
                "reproduction_manifest_sha256": sha256_file(gate_path),
            }
        )
    if total_arrays != EXPECTED_ARRAYS:
        raise ValueError(
            f"expected {EXPECTED_ARRAYS} arrays, validated {total_arrays}"
        )
    return {
        "roster": roster,
        "case_records": case_records,
        "total_arrays": total_arrays,
        "array_inventory_sha256": stable_sha256(inventory),
    }


def accumulate_probability_chunk(
    arrays: dict[str, np.ndarray],
    candidate_ids: tuple[str, ...],
    finding_index: int,
    start: int,
    stop: int,
) -> np.ndarray:
    ensemble = np.zeros(stop - start, dtype=np.float32)
    weight = np.float32(1.0 / len(candidate_ids))
    for candidate_id in candidate_ids:
        source = np.asarray(
            arrays[candidate_id][finding_index].reshape(-1)[start:stop],
            dtype=np.float32,
        )
        if not np.isfinite(source).all():
            raise ValueError(
                f"{candidate_id} finding {finding_index}: non-finite logits"
            )
        ensemble += weight * sigmoid(source)
    if not np.isfinite(ensemble).all():
        raise ValueError(f"finding {finding_index}: non-finite ensemble")
    return ensemble


def dice_from_counts(
    gt_voxels: int,
    predicted_voxels: int,
    intersection_voxels: int,
) -> float:
    denominator = gt_voxels + predicted_voxels
    if denominator == 0:
        return 1.0
    return (
        2 * intersection_voxels + DICE_EPSILON
    ) / (denominator + DICE_EPSILON)


def evaluate_case(
    case: dict[str, Any],
    candidate_ids: tuple[str, ...],
    runtime_root: Path,
    seg_dir: Path,
    chunk_elements: int,
    recipe_hash: str,
) -> dict[str, Any]:
    import nibabel as nib

    name = case["name"]
    arrays = {
        candidate_id: np.load(
            array_path(runtime_root, candidate_id, name),
            mmap_mode="r",
            allow_pickle=False,
        )
        for candidate_id in candidate_ids
    }
    shapes = {tuple(array.shape) for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"{name}: candidate logit geometry mismatch")
    shape = next(iter(shapes))
    gt_path = seg_dir / name
    if not gt_path.is_file():
        raise FileNotFoundError(f"{name}: missing ground truth {gt_path}")
    ground_truth = np.asanyarray(nib.load(str(gt_path)).dataobj)
    if tuple(ground_truth.shape) != shape:
        raise ValueError(f"{name}: logits {shape} != ground truth {ground_truth.shape}")
    rows = []
    for finding_key in sorted(case["findings"], key=int):
        finding_index = int(finding_key)
        truth = np.asarray(ground_truth[finding_index] > 0).reshape(-1)
        gt_voxels = int(np.count_nonzero(truth))
        pred_counts = np.zeros(len(THRESHOLDS), dtype=np.int64)
        intersections = np.zeros(len(THRESHOLDS), dtype=np.int64)
        for start in range(0, truth.size, chunk_elements):
            stop = min(start + chunk_elements, truth.size)
            ensemble = accumulate_probability_chunk(
                arrays,
                candidate_ids,
                finding_index,
                start,
                stop,
            )
            truth_chunk = truth[start:stop]
            for threshold_index, threshold in enumerate(THRESHOLDS):
                predicted = ensemble >= threshold
                pred_counts[threshold_index] += int(np.count_nonzero(predicted))
                intersections[threshold_index] += int(
                    np.count_nonzero(predicted & truth_chunk)
                )
        threshold_metrics = {}
        for threshold_index, threshold in enumerate(THRESHOLDS):
            dice = dice_from_counts(
                gt_voxels,
                int(pred_counts[threshold_index]),
                int(intersections[threshold_index]),
            )
            threshold_metrics[threshold_key(threshold)] = {
                "dice": dice,
                "hit": bool(dice >= HIT_THRESHOLD),
                "pred_voxels": int(pred_counts[threshold_index]),
                "intersection_voxels": int(intersections[threshold_index]),
            }
        rows.append(
            {
                "case": name,
                "finding_index": finding_index,
                "category": case["categories"][finding_key],
                "gt_voxels": gt_voxels,
                "thresholds": threshold_metrics,
            }
        )
    del ground_truth
    del arrays
    return {
        "schema_version": 1,
        "analysis": ANALYSIS_ID,
        "recipe_hash": recipe_hash,
        "case": name,
        "finding_count": len(rows),
        "thresholds": list(THRESHOLDS),
        "rows": rows,
    }


def validate_cached_case(
    value: dict[str, Any],
    case: dict[str, Any],
    recipe_hash: str,
) -> list[dict[str, Any]]:
    if (
        value.get("schema_version") != 1
        or value.get("analysis") != ANALYSIS_ID
        or value.get("recipe_hash") != recipe_hash
        or value.get("case") != case["name"]
        or value.get("thresholds") != list(THRESHOLDS)
    ):
        raise ValueError(f"{case['name']}: incompatible cached case result")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(case["findings"]):
        raise ValueError(f"{case['name']}: incomplete cached finding rows")
    expected = {
        (int(index), str(case["categories"][index]))
        for index in case["findings"]
    }
    observed = {
        (int(row.get("finding_index", -1)), str(row.get("category")))
        for row in rows
        if isinstance(row, dict)
    }
    if observed != expected:
        raise ValueError(f"{case['name']}: cached finding/category mismatch")
    expected_thresholds = {threshold_key(value) for value in THRESHOLDS}
    if any(set(row.get("thresholds", {})) != expected_thresholds for row in rows):
        raise ValueError(f"{case['name']}: cached threshold grid mismatch")
    return rows


def best_metric(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [dict(value) for value in metrics]
    if not values:
        raise ValueError("cannot choose from empty metrics")
    return min(
        values,
        key=lambda value: (
            -float(value["dice"]),
            -int(value["hits"]),
            float(value["threshold"]),
        ),
    )


def summarize_threshold(
    rows: list[dict[str, Any]],
    threshold: float,
    supports: dict[str, int],
) -> dict[str, Any]:
    key = threshold_key(threshold)
    dice_values = [float(row["thresholds"][key]["dice"]) for row in rows]
    hits = sum(bool(row["thresholds"][key]["hit"]) for row in rows)
    by_category: dict[str, list[float]] = collections.defaultdict(list)
    category_hits: collections.Counter[str] = collections.Counter()
    for row in rows:
        category = row["category"]
        by_category[category].append(float(row["thresholds"][key]["dice"]))
        category_hits[category] += int(row["thresholds"][key]["hit"])
    categories = {}
    for code in CATEGORY_LABELS:
        support = supports[code]
        values = by_category.get(code, [])
        if len(values) != support:
            raise ValueError(
                f"threshold {threshold:.2f} category {code}: "
                f"expected {support} findings, got {len(values)}"
            )
        if support == 0:
            categories[code] = {
                "available": False,
                "findings": 0,
                "dice": None,
                "hits": None,
                "hit_rate": None,
            }
            continue
        category_hit_count = int(category_hits[code])
        categories[code] = {
            "available": True,
            "findings": support,
            "dice": math.fsum(values) / support,
            "hits": category_hit_count,
            "hit_rate": category_hit_count / support,
        }
    supported_dice = [
        float(categories[code]["dice"])
        for code in CATEGORY_LABELS
        if supports[code] > 0
    ]
    result = {
        "threshold": threshold,
        "findings": len(rows),
        "dice": math.fsum(dice_values) / len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows),
        "macro_category_dice": math.fsum(supported_dice) / len(supported_dice),
        "minimum_category_dice": min(supported_dice),
        "categories": categories,
    }
    validate_recomposition(result, supports)
    return result


def validate_recomposition(
    metrics: dict[str, Any],
    supports: dict[str, int],
) -> None:
    recomposed_dice = math.fsum(
        float(metrics["categories"][code]["dice"]) * supports[code]
        for code in CATEGORY_LABELS
        if supports[code] > 0
    ) / EXPECTED_FINDINGS
    recomposed_hits = sum(
        int(metrics["categories"][code]["hits"])
        for code in CATEGORY_LABELS
        if supports[code] > 0
    )
    if not math.isclose(
        recomposed_dice,
        float(metrics["dice"]),
        rel_tol=0.0,
        abs_tol=RECOMPOSITION_TOLERANCE,
    ):
        raise ValueError(
            f"category Dice {recomposed_dice} != global Dice {metrics['dice']}"
        )
    if recomposed_hits != int(metrics["hits"]):
        raise ValueError(
            f"category hits {recomposed_hits} != global hits {metrics['hits']}"
        )
    if not math.isclose(
        recomposed_hits / EXPECTED_FINDINGS,
        float(metrics["hit_rate"]),
        rel_tol=0.0,
        abs_tol=RECOMPOSITION_TOLERANCE,
    ):
        raise ValueError("category hits do not recompose global hit rate")
    metrics["validation"] = {
        "category_dice_recomposed": recomposed_dice,
        "category_hits_recomposed": recomposed_hits,
        "category_hit_rate_recomposed": recomposed_hits / EXPECTED_FINDINGS,
    }


def category_oracle(
    rows: list[dict[str, Any]],
    category_best: dict[str, dict[str, Any]],
    supports: dict[str, int],
) -> dict[str, Any]:
    dice_values = []
    hits = 0
    for row in rows:
        threshold = float(category_best[row["category"]]["threshold"])
        metric = row["thresholds"][threshold_key(threshold)]
        dice_values.append(float(metric["dice"]))
        hits += int(metric["hit"])
    categories = {}
    for code in CATEGORY_LABELS:
        if supports[code] == 0:
            categories[code] = {
                "available": False,
                "findings": 0,
                "dice": None,
                "hits": None,
                "hit_rate": None,
                "threshold": None,
            }
            continue
        categories[code] = {
            "available": True,
            **category_best[code],
        }
    value = {
        "findings": len(rows),
        "dice": math.fsum(dice_values) / len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows),
        "categories": categories,
    }
    validate_recomposition(value, supports)
    return value


def build_summary(
    rows: list[dict[str, Any]],
    supports: dict[str, int],
    roster: list[dict[str, Any]],
    recipe_hash: str,
    manifest_path: Path,
    dataset_path: Path,
    runtime_root: Path,
    seg_dir: Path,
    inventory_sha256: str,
    reproduction_command: str,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_FINDINGS:
        raise ValueError(f"expected {EXPECTED_FINDINGS} rows, got {len(rows)}")
    keys = [(row["case"], int(row["finding_index"])) for row in rows]
    if len(set(keys)) != EXPECTED_FINDINGS:
        raise ValueError("duplicate finding rows in ensemble results")
    sweep = [
        summarize_threshold(rows, threshold, supports)
        for threshold in THRESHOLDS
    ]
    global_best = best_metric(sweep)
    category_best = {}
    for code in CATEGORY_LABELS:
        if supports[code] == 0:
            continue
        candidates = []
        for threshold_metrics in sweep:
            metric = threshold_metrics["categories"][code]
            candidates.append(
                {
                    "threshold": threshold_metrics["threshold"],
                    "findings": metric["findings"],
                    "dice": metric["dice"],
                    "hits": metric["hits"],
                    "hit_rate": metric["hit_rate"],
                }
            )
        category_best[code] = best_metric(candidates)
    supported_codes = [code for code in CATEGORY_LABELS if supports[code] > 0]
    for threshold_metrics in sweep:
        regrets = {
            code: float(category_best[code]["dice"])
            - float(threshold_metrics["categories"][code]["dice"])
            for code in supported_codes
        }
        threshold_metrics["categories_selecting_threshold"] = sum(
            math.isclose(
                float(category_best[code]["threshold"]),
                float(threshold_metrics["threshold"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for code in supported_codes
        )
        threshold_metrics["mean_category_regret"] = (
            math.fsum(regrets.values()) / len(regrets)
        )
        threshold_metrics["maximum_category_regret"] = max(regrets.values())
    global_threshold = float(global_best["threshold"])
    categories = {}
    for code, label in CATEGORY_LABELS.items():
        support = supports[code]
        if support == 0:
            categories[code] = {
                "label": label,
                "support": 0,
                "available": False,
                "global_threshold_metrics": None,
                "best_threshold_metrics": None,
                "dice_gain": None,
                "hit_gain": None,
            }
            continue
        global_metrics = global_best["categories"][code]
        best = category_best[code]
        categories[code] = {
            "label": label,
            "support": support,
            "available": True,
            "global_threshold_metrics": {
                "threshold": global_threshold,
                "findings": support,
                "dice": global_metrics["dice"],
                "hits": global_metrics["hits"],
                "hit_rate": global_metrics["hit_rate"],
            },
            "best_threshold_metrics": best,
            "dice_gain": float(best["dice"]) - float(global_metrics["dice"]),
            "hit_gain": int(best["hits"]) - int(global_metrics["hits"]),
        }
    oracle = category_oracle(rows, category_best, supports)
    oracle["dice_gain_over_global"] = (
        float(oracle["dice"]) - float(global_best["dice"])
    )
    oracle["hit_gain_over_global"] = (
        int(oracle["hits"]) - int(global_best["hits"])
    )
    matching = [
        code
        for code in supported_codes
        if math.isclose(
            float(category_best[code]["threshold"]),
            global_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    losses = {
        code: float(categories[code]["dice_gain"]) for code in supported_codes
    }
    worst_code = max(losses, key=lambda code: (losses[code], code))
    summary = {
        "schema_version": 1,
        "analysis": ANALYSIS_ID,
        "definitions": {
            "averaging": (
                "uniform mean of the 20 per-model sigmoid probabilities"
            ),
            "model_weight": 1.0 / EXPECTED_MODELS,
            "threshold_comparison": "ensemble probability >= threshold",
            "threshold_grid": list(THRESHOLDS),
            "hit_threshold": HIT_THRESHOLD,
            "dice_aggregation": "unweighted mean across findings",
            "global_selection": (
                "maximum full-precision overall Dice, then hits, then "
                "lower threshold"
            ),
            "category_selection": (
                "maximum full-precision category Dice, then category hits, "
                "then lower threshold"
            ),
            "acceptance_override": (
                "strict gate status is diagnostic and does not exclude a model"
            ),
        },
        "sources": {
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "dataset_json": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "runtime_root": str(runtime_root),
            "segmentation_dir": str(seg_dir),
            "array_inventory_sha256": inventory_sha256,
        },
        "recipe": {
            "name": "all20_uniform_probability",
            "recipe_hash": recipe_hash,
            "domain": "probability",
            "candidate_count": EXPECTED_MODELS,
            "candidate_ids": [row["candidate_id"] for row in roster],
            "weights": {
                row["candidate_id"]: row["weight"] for row in roster
            },
        },
        "counts": {
            "models": len(roster),
            "cases": EXPECTED_CASES,
            "findings": len(rows),
            "arrays": EXPECTED_ARRAYS,
            "official_categories": len(CATEGORY_LABELS),
            "represented_categories": len(supported_codes),
            "strict_gate_accepted_models": sum(
                bool(row["strict_gate_accepted"]) for row in roster
            ),
            "strict_gate_bypassed_models": sum(
                not bool(row["strict_gate_accepted"]) for row in roster
            ),
        },
        "candidates": roster,
        "category_support": {
            code: {
                "label": label,
                "support": supports[code],
                "available": supports[code] > 0,
            }
            for code, label in CATEGORY_LABELS.items()
        },
        "threshold_sweep": sweep,
        "global_best": global_best,
        "categories": categories,
        "category_oracle": oracle,
        "single_threshold_robustness": {
            "global_threshold": global_threshold,
            "represented_categories": len(supported_codes),
            "categories_selecting_global_threshold": matching,
            "category_count_selecting_global_threshold": len(matching),
            "mean_category_dice_loss": math.fsum(losses.values()) / len(losses),
            "maximum_category_dice_loss": losses[worst_code],
            "maximum_loss_category": worst_code,
        },
        "validation": {
            "candidate_count_is_20": len(roster) == EXPECTED_MODELS,
            "case_count_is_200": len({row["case"] for row in rows})
            == EXPECTED_CASES,
            "finding_count_is_381": len(rows) == EXPECTED_FINDINGS,
            "array_count_is_4000": EXPECTED_ARRAYS,
            "category_supports_sum_to_381": sum(supports.values())
            == EXPECTED_FINDINGS,
            "all_threshold_recompositions_passed": all(
                int(value["validation"]["category_hits_recomposed"])
                == int(value["hits"])
                and abs(
                    float(value["validation"]["category_dice_recomposed"])
                    - float(value["dice"])
                )
                <= RECOMPOSITION_TOLERANCE
                for value in sweep
            ),
            "category_oracle_recomposition_passed": (
                int(oracle["validation"]["category_hits_recomposed"])
                == int(oracle["hits"])
                and abs(
                    float(oracle["validation"]["category_dice_recomposed"])
                    - float(oracle["dice"])
                )
                <= RECOMPOSITION_TOLERANCE
            ),
        },
        "warnings": [
            (
                "Seven models have failed strict reproduction records and are "
                "included by explicit user-approved diagnostic override."
            ),
            (
                "Category 2f has no val200 findings, so its threshold and "
                "metrics are unavailable."
            ),
            (
                "Rare-category thresholds are descriptive and unstable, "
                "especially category 2g with one finding."
            ),
            (
                "Global and category thresholds are selected post hoc on "
                "val200 and do not establish test-set generalization."
            ),
        ],
        "reproduction_command": reproduction_command,
    }
    summary["deterministic_result_sha256"] = stable_sha256(summary)
    return summary


def candidate_reason_cell(value: str) -> str:
    return value.replace("|", "/")


def matrix_cell(
    summary: dict[str, Any],
    threshold_metrics: dict[str, Any],
    code: str,
) -> str:
    metric = threshold_metrics["categories"][code]
    if not metric["available"]:
        return "—"
    value = (
        f"{metric['dice']:.6f} "
        f"({metric['hits']}/{metric['findings']})"
    )
    selected = summary["categories"][code]["best_threshold_metrics"]
    if math.isclose(
        float(selected["threshold"]),
        float(threshold_metrics["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return f"**{value}**"
    return value


def build_matrix(
    summary: dict[str, Any],
    codes: tuple[str, ...],
    title: str,
) -> list[str]:
    header = "| Threshold | " + " | ".join(
        f"{code} (N={summary['category_support'][code]['support']})"
        for code in codes
    ) + " |"
    separator = "| ---: | " + " | ".join("---" for _ in codes) + " |"
    lines = [
        f"## {title}",
        "",
        "Each cell is `Dice (hits/N)`; the selected category threshold is bold.",
        "",
        header,
        separator,
    ]
    global_threshold = float(summary["global_best"]["threshold"])
    for metrics in summary["threshold_sweep"]:
        label = f"{metrics['threshold']:.2f}"
        if math.isclose(
            float(metrics["threshold"]),
            global_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            label = f"**{label} (global)**"
        cells = " | ".join(
            matrix_cell(summary, metrics, code) for code in codes
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    return lines


def build_report(summary: dict[str, Any]) -> str:
    global_best = summary["global_best"]
    oracle = summary["category_oracle"]
    robustness = summary["single_threshold_robustness"]
    lines = [
        "# Side Experiment 002 All-20 Probability Ensemble Threshold Report",
        "",
        "This post-hoc val200 probe applies sigmoid to each stored model logit, "
        "uniformly averages all 20 probabilities, and thresholds that mean.",
        "",
        "> **Acceptance override:** all 20 completed exports are used. Seven "
        "have failed strict reproduction records; their stored arrays passed "
        "recorded hash verification and structural checks, but their gate "
        "warnings remain visible below.",
        "",
        "> **Rare-category warning:** category `2g` has one finding, and `2f` "
        "has no findings. Category-specific winners are descriptive validation "
        "oracles, not generalizable threshold policies.",
        "",
        "## Executive Summary",
        "",
        f"- Best single global threshold: `{global_best['threshold']:.2f}`",
        f"- Global Dice: `{global_best['dice']:.6f}`",
        f"- Global hits: `{global_best['hits']} / {global_best['findings']}` "
        f"(`{global_best['hit_rate']:.6f}`)",
        f"- Per-category oracle Dice: `{oracle['dice']:.6f}` "
        f"(gain `{oracle['dice_gain_over_global']:+.6f}`)",
        f"- Per-category oracle hits: `{oracle['hits']} / {oracle['findings']}` "
        f"(gain `{oracle['hit_gain_over_global']:+d}`)",
        f"- Categories selecting the global threshold: "
        f"`{robustness['category_count_selecting_global_threshold']} / "
        f"{robustness['represented_categories']}`",
        f"- Mean category Dice loss from one global threshold: "
        f"`{robustness['mean_category_dice_loss']:.6f}`",
        f"- Maximum category Dice loss: "
        f"`{robustness['maximum_category_dice_loss']:.6f}` "
        f"for `{robustness['maximum_loss_category']}`",
        "",
        "## Ensemble Roster",
        "",
        "| Candidate | Source | Model / arm | Epoch | Architecture | Dtype | "
        "Weight | Strict gate | Warning reason |",
        "| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            f"| `{candidate['candidate_id']}` | "
            f"Exp{candidate['experiment_number']:03d} | "
            f"`{candidate['model_id']}` | {candidate['epoch']} | "
            f"{candidate['architecture']} | {candidate['dtype']} | "
            f"{candidate['weight']:.6f} | "
            f"{'accepted' if candidate['strict_gate_accepted'] else 'bypassed'} | "
            f"{candidate_reason_cell(candidate['warning_reason'])} |"
        )
    lines.extend(
        [
            "",
            "## Category Support",
            "",
            "| Code | Official category | Findings | Interpretation |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for code, value in summary["category_support"].items():
        lines.append(
            f"| {code} | {value['label']} | {value['support']} | "
            f"{support_note(int(value['support']))} |"
        )
    lines.extend(
        [
            "",
            "## Overall Threshold Sweep",
            "",
            "The global winner is selected by full-precision mean Dice, then "
            "hits, then the lower threshold.",
            "",
            "| Threshold | Dice | Hits | Hit rate | Macro-category Dice | "
            "Minimum category Dice | Categories choosing threshold | "
            "Mean category regret | Maximum category regret |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for value in summary["threshold_sweep"]:
        selected = math.isclose(
            float(value["threshold"]),
            float(global_best["threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        threshold = f"{value['threshold']:.2f}"
        dice = f"{value['dice']:.6f}"
        if selected:
            threshold = f"**{threshold}**"
            dice = f"**{dice}**"
        lines.append(
            f"| {threshold} | {dice} | "
            f"{value['hits']} / {value['findings']} | "
            f"{value['hit_rate']:.6f} | "
            f"{value['macro_category_dice']:.6f} | "
            f"{value['minimum_category_dice']:.6f} | "
            f"{value['categories_selecting_threshold']} / "
            f"{summary['counts']['represented_categories']} | "
            f"{value['mean_category_regret']:.6f} | "
            f"{value['maximum_category_regret']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Global Versus Category-Best Thresholds",
            "",
            "| Code | Category | N | Global threshold Dice | Global hits | "
            "Best threshold | Best Dice | Best hits | Dice gain | Hit gain |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: |",
        ]
    )
    for code, value in summary["categories"].items():
        if not value["available"]:
            lines.append(
                f"| {code} | {value['label']} | 0 | — | — | — | — | — | — | — |"
            )
            continue
        global_metrics = value["global_threshold_metrics"]
        best = value["best_threshold_metrics"]
        lines.append(
            f"| {code} | {value['label']} | {value['support']} | "
            f"{global_metrics['dice']:.6f} | "
            f"{global_metrics['hits']} / {global_metrics['findings']} | "
            f"**{best['threshold']:.2f}** | **{best['dice']:.6f}** | "
            f"**{best['hits']} / {best['findings']}** | "
            f"{value['dice_gain']:+.6f} | {value['hit_gain']:+d} |"
        )
    lines.append("")
    lines.extend(build_matrix(summary, DIFFUSE_CATEGORIES, "Diffuse Threshold Matrix"))
    lines.extend(build_matrix(summary, FOCAL_CATEGORIES, "Focal Threshold Matrix"))
    lines.extend(
        [
            "## Validation and Provenance",
            "",
            f"- Candidate manifest SHA256: "
            f"`{summary['sources']['candidate_manifest_sha256']}`",
            f"- Dataset SHA256: `{summary['sources']['dataset_sha256']}`",
            f"- Array inventory SHA256: "
            f"`{summary['sources']['array_inventory_sha256']}`",
            f"- Recipe hash: `{summary['recipe']['recipe_hash']}`",
            f"- Deterministic result SHA256: "
            f"`{summary['deterministic_result_sha256']}`",
            f"- Arrays checked: `{summary['counts']['arrays']}`",
            f"- Threshold recompositions passed: "
            f"`{summary['validation']['all_threshold_recompositions_passed']}`",
            f"- Category-oracle recomposition passed: "
            f"`{summary['validation']['category_oracle_recomposition_passed']}`",
            "",
            "Reproduction command:",
            "",
            "```bash",
            summary["reproduction_command"],
            "```",
            "",
            "Both the global winner and category-specific winners were selected "
            "on this same val200 set. The category oracle is a hindsight upper "
            "bound and must not be presented as expected test performance.",
            "",
        ]
    )
    return "\n".join(lines)


def reproduction_command(args: argparse.Namespace) -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 python "
        "side_experiments/sideexp002_multimodel_ensemble_selection/"
        "analyze_all20_thresholds.py "
        f"--manifest {args.manifest} "
        f"--dataset-json {args.dataset_json} "
        f"--runtime-root {args.runtime_root} "
        f"--seg-dir {args.seg_dir} "
        f"--output-json {args.output_json} "
        f"--output-markdown {args.output_markdown} "
        f"--chunk-elements {args.chunk_elements} "
        "--allow-unaccepted"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_elements < 1024:
        raise ValueError("--chunk-elements must be at least 1024")
    manifest = load_manifest(args.manifest)
    dataset_sha256 = sha256_file(args.dataset_json)
    cases, category_by_key, supports = load_cases(args.dataset_json)
    validation = validate_exports(
        manifest,
        cases,
        dataset_sha256,
        args.runtime_root,
        args.allow_unaccepted,
    )
    roster = validation["roster"]
    recipe_payload = {
        "analysis": ANALYSIS_ID,
        "domain": "probability",
        "thresholds": THRESHOLDS,
        "dataset_sha256": dataset_sha256,
        "array_inventory_sha256": validation["array_inventory_sha256"],
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "weight": row["weight"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in roster
        ],
    }
    recipe_hash = stable_sha256(recipe_payload)
    candidate_ids = tuple(row["candidate_id"] for row in roster)
    run_root = args.runtime_root / "all20_uniform_probability_threshold_sweep"
    cache_root = run_root / "cache" / recipe_hash / "cases"
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_manifest.json"
    progress_path = run_root / "progress.json"
    state = {
        "schema_version": 1,
        "analysis": ANALYSIS_ID,
        "status": "running",
        "recipe_hash": recipe_hash,
        "candidate_count": len(candidate_ids),
        "thresholds": list(THRESHOLDS),
        "allow_unaccepted": bool(args.allow_unaccepted),
        "gpu_inference_performed": False,
        "test_inference_performed": False,
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
    }
    atomic_write_json(state_path, state)
    all_rows = []
    cached_cases = 0
    evaluated_cases = 0
    started = time.monotonic()
    try:
        for case_index, case in enumerate(cases, 1):
            cache_path = cache_root / f"{case['name']}.json"
            if cache_path.is_file():
                rows = validate_cached_case(
                    read_json(cache_path),
                    case,
                    recipe_hash,
                )
                cached_cases += 1
            else:
                value = evaluate_case(
                    case,
                    candidate_ids,
                    args.runtime_root,
                    args.seg_dir,
                    args.chunk_elements,
                    recipe_hash,
                )
                atomic_write_json(cache_path, value)
                rows = value["rows"]
                evaluated_cases += 1
            all_rows.extend(rows)
            atomic_write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "analysis": ANALYSIS_ID,
                    "status": "running",
                    "recipe_hash": recipe_hash,
                    "completed_cases": case_index,
                    "total_cases": EXPECTED_CASES,
                    "completed_findings": len(all_rows),
                    "cached_cases": cached_cases,
                    "evaluated_cases": evaluated_cases,
                    "last_completed_case": case["name"],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        if {
            (row["case"], int(row["finding_index"])): row["category"]
            for row in all_rows
        } != category_by_key:
            raise ValueError("ensemble result rows do not match the dataset index")
        summary = build_summary(
            all_rows,
            supports,
            roster,
            recipe_hash,
            args.manifest,
            args.dataset_json,
            args.runtime_root,
            args.seg_dir,
            validation["array_inventory_sha256"],
            reproduction_command(args),
        )
        atomic_write_json(args.output_json, summary)
        atomic_write_text(args.output_markdown, build_report(summary))
        state.update(
            {
                "status": "complete",
                "completed_cases": EXPECTED_CASES,
                "completed_findings": EXPECTED_FINDINGS,
                "cached_cases": cached_cases,
                "evaluated_cases": evaluated_cases,
                "elapsed_seconds": time.monotonic() - started,
                "deterministic_result_sha256": summary[
                    "deterministic_result_sha256"
                ],
                "output_json_sha256": sha256_file(args.output_json),
                "output_markdown_sha256": sha256_file(args.output_markdown),
            }
        )
        atomic_write_json(state_path, state)
        atomic_write_json(
            progress_path,
            {
                "schema_version": 1,
                "analysis": ANALYSIS_ID,
                "status": "complete",
                "recipe_hash": recipe_hash,
                "completed_cases": EXPECTED_CASES,
                "total_cases": EXPECTED_CASES,
                "completed_findings": EXPECTED_FINDINGS,
                "cached_cases": cached_cases,
                "evaluated_cases": evaluated_cases,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return summary
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_write_json(state_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--chunk-elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--allow-unaccepted",
        action="store_true",
        help=(
            "Use structurally valid exports even when their strict reproduction "
            "gate failed."
        ),
    )
    args = parser.parse_args()
    summary = run(args)
    best = summary["global_best"]
    oracle = summary["category_oracle"]
    print(
        f"All-20 probability ensemble: Dice {best['dice']:.6f}, "
        f"{best['hits']}/{best['findings']} hits at threshold "
        f"{best['threshold']:.2f}; category oracle Dice {oracle['dice']:.6f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
