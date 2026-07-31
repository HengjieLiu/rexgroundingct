#!/usr/bin/env python3
"""Stream sideexp002 logits to select cross-validated ensemble recipes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_candidates import (
    atomic_write_json,
    atomic_write_text,
    flatten_eval,
    load_manifest,
    sha256_file,
)
from make_folds import DEFAULT_OUTPUT as DEFAULT_FOLDS


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
DEFAULT_OUTPUT_JSON = HERE / "outputs" / "ensemble_selection_summary.json"
DEFAULT_OUTPUT_MD = HERE / "outputs" / "ensemble_selection_report.md"
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
EVALUATOR = (
    REPO_ROOT
    / "external"
    / "RexGrounding_challenge2026"
    / "data"
    / "rexrank_eval.py"
)
COARSE_THRESHOLDS = tuple(
    sorted({0.02, 0.98, *(round(value * 0.05, 2) for value in range(1, 20))})
)
ALL_THRESHOLDS = tuple(round(value / 100, 2) for value in range(1, 100))
HIT_THRESHOLD = 0.1
DICE_EPSILON = 1e-6


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def threshold_key(value: float) -> str:
    return f"{value:.2f}"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights or any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("recipe weights must be finite and nonnegative")
    total = math.fsum(weights.values())
    if total <= 0:
        raise ValueError("recipe must have positive total weight")
    return {
        candidate_id: weights[candidate_id] / total
        for candidate_id in sorted(weights)
        if weights[candidate_id] > 0
    }


def make_recipe(
    name: str,
    domain: str,
    weights: dict[str, float],
    family: str,
) -> dict[str, Any]:
    if domain not in {"probability", "logit"}:
        raise ValueError(f"unsupported averaging domain: {domain}")
    return {
        "name": name,
        "domain": domain,
        "family": family,
        "weights": normalize_weights(weights),
    }


def recipe_hash(
    recipe: dict[str, Any],
    dataset_sha256: str,
    source_hashes: dict[str, str],
) -> str:
    payload = {
        "recipe": recipe,
        "dataset_sha256": dataset_sha256,
        "source_hashes": {
            candidate_id: source_hashes[candidate_id]
            for candidate_id in sorted(recipe["weights"])
        },
        "thresholds": ALL_THRESHOLDS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_result_hash(summary: dict[str, Any]) -> str:
    """Hash selection content while excluding measured runtime and output paths."""
    excluded = {
        "deterministic_result_sha256",
        "elapsed_seconds",
        "recipe_runtime_seconds",
        "mean_recipe_runtime_seconds",
        "materialized_verification",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in sorted(value.items())
                if key not in excluded
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    encoded = json.dumps(
        clean(summary),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    cases = data.get("test")
    if not isinstance(cases, list) or len(cases) != 200:
        raise ValueError(f"{path}: expected 200 cases under 'test'")
    seen: set[str] = set()
    count = 0
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"{path}: duplicate or invalid case name {name!r}")
        seen.add(name)
        if (
            not isinstance(findings, dict)
            or not isinstance(categories, dict)
            or set(findings) != set(categories)
        ):
            raise ValueError(f"{path} {name}: finding/category mismatch")
        count += len(findings)
    if count != 381:
        raise ValueError(f"{path}: expected 381 findings, got {count}")
    return cases


def validate_runtime_exports(
    candidates: list[dict[str, Any]],
    runtime_root: Path,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_names = {case["name"] for case in cases}
    export_manifests = {}
    for candidate in candidates:
        candidate_id = candidate["id"]
        root = runtime_root / "logits" / candidate_id
        export_path = root / "export_manifest.json"
        gate_path = root / "reproduction_validation.json"
        if not export_path.is_file() or not gate_path.is_file():
            raise FileNotFoundError(f"{candidate_id}: missing export/gate manifest")
        export = read_json(export_path)
        gate = read_json(gate_path)
        if (
            export.get("status") != "complete"
            or export.get("candidate_id") != candidate_id
            or int(export.get("case_count", -1)) != 200
            or int(export.get("finding_count", -1)) != 381
            or gate.get("status") != "passed"
            or gate.get("storage_reproduction_status") != "passed"
            or gate.get("same_pass_mask_reproduction_status") != "passed"
            or gate.get("historical_reproduction_is_gate") is not False
            or gate.get("array_hashes_verified") is not True
        ):
            raise ValueError(f"{candidate_id}: incomplete or failed logit export")
        if export.get("candidate_source") != candidate:
            raise ValueError(
                f"{candidate_id}: exported source metadata drifted from manifest"
            )
        names = [row.get("name") for row in export.get("cases", [])]
        if len(names) != 200 or set(names) != expected_names:
            raise ValueError(f"{candidate_id}: duplicate or missing exported cases")
        export_manifests[candidate_id] = export
    return export_manifests


def load_folds(path: Path, cases: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, Any]]:
    data = read_json(path)
    if data.get("num_folds") != 5 or data.get("seed") != 20260729:
        raise ValueError(f"{path}: expected five folds with seed 20260729")
    mapping: dict[str, int] = {}
    for fold in data.get("folds", []):
        fold_index = int(fold["fold"])
        for name in fold["cases"]:
            if name in mapping:
                raise ValueError(f"{path}: duplicate case {name}")
            mapping[name] = fold_index
    expected = {case["name"] for case in cases}
    if set(mapping) != expected:
        raise ValueError(f"{path}: fold cases do not match dataset")
    return mapping, data


def case_array_path(runtime_root: Path, candidate_id: str, name: str) -> Path:
    return runtime_root / "logits" / candidate_id / "cases" / f"{name}.npy"


def accumulate_ensemble_chunk(
    arrays: dict[str, np.ndarray],
    weights: dict[str, float],
    finding_index: int,
    start: int,
    stop: int,
    domain: str,
) -> np.ndarray:
    ensemble = np.zeros(stop - start, dtype=np.float32)
    for candidate_id, weight in weights.items():
        source = np.asarray(
            arrays[candidate_id][finding_index].reshape(-1)[start:stop],
            dtype=np.float32,
        )
        if domain == "probability":
            source = sigmoid(source)
        ensemble += np.float32(weight) * source
    if domain == "logit":
        ensemble = sigmoid(ensemble)
    return ensemble


def evaluate_recipe_uncached(
    recipe: dict[str, Any],
    cases: list[dict[str, Any]],
    runtime_root: Path,
    seg_dir: Path,
    chunk_elements: int,
) -> list[dict[str, Any]]:
    import nibabel as nib

    rows = []
    for case in cases:
        name = case["name"]
        arrays = {
            candidate_id: np.load(
                case_array_path(runtime_root, candidate_id, name),
                mmap_mode="r",
                allow_pickle=False,
            )
            for candidate_id in recipe["weights"]
        }
        shapes = {tuple(array.shape) for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"{name}: candidate logit geometry mismatch")
        ground_truth = np.asanyarray(nib.load(str(seg_dir / name)).dataobj)
        shape = next(iter(shapes))
        if tuple(ground_truth.shape) != shape:
            raise ValueError(f"{name}: logits {shape} != GT {ground_truth.shape}")
        for array in arrays.values():
            if array.dtype not in {np.dtype("float16"), np.dtype("float32")}:
                raise ValueError(f"{name}: unsupported logit dtype {array.dtype}")
        for finding_key in sorted(case["findings"], key=int):
            finding_index = int(finding_key)
            truth = np.asarray(ground_truth[finding_index] > 0).reshape(-1)
            gt_voxels = int(np.count_nonzero(truth))
            pred_counts = np.zeros(len(ALL_THRESHOLDS), dtype=np.int64)
            intersections = np.zeros(len(ALL_THRESHOLDS), dtype=np.int64)
            for start in range(0, truth.size, chunk_elements):
                stop = min(start + chunk_elements, truth.size)
                ensemble = accumulate_ensemble_chunk(
                    arrays,
                    recipe["weights"],
                    finding_index,
                    start,
                    stop,
                    recipe["domain"],
                )
                truth_chunk = truth[start:stop]
                if not np.isfinite(ensemble).all():
                    raise ValueError(f"{name} finding {finding_index}: non-finite ensemble")
                for threshold_index, threshold in enumerate(ALL_THRESHOLDS):
                    predicted = ensemble >= threshold
                    pred_counts[threshold_index] += int(np.count_nonzero(predicted))
                    intersections[threshold_index] += int(
                        np.count_nonzero(predicted & truth_chunk)
                    )
            metrics = {}
            for threshold_index, threshold in enumerate(ALL_THRESHOLDS):
                denominator = gt_voxels + int(pred_counts[threshold_index])
                dice = (
                    1.0
                    if denominator == 0
                    else (
                        2 * int(intersections[threshold_index]) + DICE_EPSILON
                    )
                    / (denominator + DICE_EPSILON)
                )
                metrics[threshold_key(threshold)] = {
                    "dice": float(dice),
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
                    "thresholds": metrics,
                }
            )
        del arrays
    if len(rows) != 381:
        raise ValueError(f"recipe {recipe['name']}: expected 381 rows, got {len(rows)}")
    return rows


def summarize_rows(
    rows: Iterable[dict[str, Any]],
    thresholds: Iterable[float] = ALL_THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    selected = list(rows)
    if not selected:
        raise ValueError("cannot summarize an empty finding set")
    output = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        dice = [float(row["thresholds"][key]["dice"]) for row in selected]
        hits = sum(bool(row["thresholds"][key]["hit"]) for row in selected)
        category_values: dict[str, list[float]] = collections.defaultdict(list)
        category_hits: collections.Counter[str] = collections.Counter()
        for row in selected:
            category = row["category"]
            category_values[category].append(float(row["thresholds"][key]["dice"]))
            category_hits[category] += int(row["thresholds"][key]["hit"])
        output[key] = {
            "threshold": threshold,
            "findings": len(selected),
            "dice": math.fsum(dice) / len(dice),
            "hits": hits,
            "hit_rate": hits / len(dice),
            "categories": {
                category: {
                    "findings": len(values),
                    "dice": math.fsum(values) / len(values),
                    "hits": category_hits[category],
                    "hit_rate": category_hits[category] / len(values),
                }
                for category, values in sorted(category_values.items())
            },
        }
    return output


def evaluate_recipe(
    recipe: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    digest = recipe_hash(
        recipe,
        context["dataset_sha256"],
        context["source_hashes"],
    )
    cache_path = context["cache_root"] / f"{digest}.json"
    if cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("recipe_hash") == digest and len(cached.get("rows", [])) == 381:
            return cached
        raise ValueError(f"corrupt or incompatible recipe cache: {cache_path}")
    started = time.monotonic()
    rows = evaluate_recipe_uncached(
        recipe,
        context["cases"],
        context["runtime_root"],
        context["seg_dir"],
        context["chunk_elements"],
    )
    result = {
        "schema_version": 1,
        "recipe_hash": digest,
        "recipe": recipe,
        "dataset_sha256": context["dataset_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    atomic_write_json(cache_path, result)
    return result


def best_threshold(
    result: dict[str, Any],
    selected_cases: set[str],
) -> dict[str, Any]:
    rows = [row for row in result["rows"] if row["case"] in selected_cases]
    coarse = summarize_rows(rows, COARSE_THRESHOLDS)
    coarse_best = sorted(
        coarse.values(),
        key=lambda row: (-row["dice"], -row["hit_rate"], row["threshold"]),
    )[0]
    center = float(coarse_best["threshold"])
    refinement = [
        threshold
        for threshold in ALL_THRESHOLDS
        if center - 0.0400001 <= threshold <= center + 0.0400001
    ]
    refined = summarize_rows(rows, refinement)
    return sorted(
        refined.values(),
        key=lambda row: (-row["dice"], -row["hit_rate"], row["threshold"]),
    )[0]


def result_at_threshold(
    result: dict[str, Any],
    selected_cases: set[str],
    threshold: float,
) -> dict[str, Any]:
    return summarize_rows(
        [row for row in result["rows"] if row["case"] in selected_cases],
        [threshold],
    )[threshold_key(threshold)]


def recipe_rank(
    metrics: dict[str, Any],
    recipe: dict[str, Any],
) -> tuple[Any, ...]:
    uniform_values = list(recipe["weights"].values())
    is_uniform = max(uniform_values) - min(uniform_values) <= 1e-12
    return (
        -float(metrics["dice"]),
        -float(metrics["hit_rate"]),
        len(recipe["weights"]),
        not is_uniform,
        recipe["name"],
    )


def choose_recipe_entry(
    entries: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> tuple[
    tuple[Any, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    best_dice = max(float(entry[3]["dice"]) for entry in entries)
    eligible = [
        entry
        for entry in entries
        if float(entry[3]["dice"]) >= best_dice - 0.001
    ]

    def key(entry: tuple[Any, ...]) -> tuple[Any, ...]:
        recipe = entry[1]
        metrics = entry[3]
        weights = list(recipe["weights"].values())
        uniform = max(weights) - min(weights) <= 1e-12
        return (
            -float(metrics["hit_rate"]),
            len(recipe["weights"]),
            not uniform,
            -float(metrics["dice"]),
            recipe["name"],
        )

    return min(eligible, key=key)


def grouped_weights(
    candidates: list[dict[str, Any]],
    field: str,
) -> dict[str, float]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for candidate in candidates:
        groups[str(candidate[field])].append(candidate["id"])
    weights = {}
    for values in groups.values():
        for candidate_id in values:
            weights[candidate_id] = 1.0 / len(groups) / len(values)
    return weights


def fixed_recipes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recipes = []
    for candidate in candidates:
        recipes.append(
            make_recipe(
                candidate["id"],
                "probability",
                {candidate["id"]: 1.0},
                "single",
            )
        )
    all_uniform = {candidate["id"]: 1.0 for candidate in candidates}
    for domain in ("probability", "logit"):
        recipes.extend(
            [
                make_recipe(
                    f"all20_{domain}",
                    domain,
                    all_uniform,
                    "all20",
                ),
                make_recipe(
                    f"equal_experiment_{domain}",
                    domain,
                    grouped_weights(candidates, "experiment_number"),
                    "equal_experiment",
                ),
                make_recipe(
                    f"equal_architecture_{domain}",
                    domain,
                    grouped_weights(candidates, "architecture"),
                    "equal_architecture",
                ),
            ]
        )
    groups = {
        "exp007_trajectory": [
            "exp007_ddp_e050",
            "exp007_ddp_e075",
            "exp007_ddp_e100",
        ],
        "exp008_epoch080": [
            candidate["id"]
            for candidate in candidates
            if candidate["id"].startswith("exp008_")
            and candidate["id"].endswith("e080")
        ],
        "exp008_epoch100": [
            candidate["id"]
            for candidate in candidates
            if candidate["id"].startswith("exp008_")
            and candidate["id"].endswith("e100")
        ],
        "exp009_all": [
            candidate["id"]
            for candidate in candidates
            if candidate["id"].startswith("exp009_")
        ],
        "exp011_normalization": [
            candidate["id"]
            for candidate in candidates
            if candidate["id"].startswith("exp011_")
        ],
    }
    for arm in ("shared", "dual", "precision", "joint"):
        groups[f"exp008_{arm}_snapshots"] = [
            f"exp008_{arm}_e080",
            f"exp008_{arm}_e100",
        ]
    for name, candidate_ids in groups.items():
        for domain in ("probability", "logit"):
            recipes.append(
                make_recipe(
                    f"{name}_{domain}",
                    domain,
                    {candidate_id: 1.0 for candidate_id in candidate_ids},
                    "trajectory",
                )
            )
    return recipes


def select_best_recipe(
    recipes: list[dict[str, Any]],
    context: dict[str, Any],
    train_cases: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ranked = []
    for recipe in recipes:
        result = evaluate_recipe(recipe, context)
        metrics = best_threshold(result, train_cases)
        ranked.append((recipe_rank(metrics, recipe), recipe, result, metrics))
    _, recipe, result, metrics = choose_recipe_entry(ranked)
    return recipe, result, metrics


def greedy_forward(
    domain: str,
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    train_cases: set[str],
    max_models: int,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    candidate_ids = sorted(candidate["id"] for candidate in candidates)
    selected: list[str] = []
    path = []
    while len(selected) < min(max_models, len(candidate_ids)):
        choices = []
        for candidate_id in candidate_ids:
            if candidate_id in selected:
                continue
            values = selected + [candidate_id]
            recipe = make_recipe(
                f"greedy_{domain}_{len(values):02d}_" + "_".join(values),
                domain,
                {value: 1.0 for value in values},
                "greedy",
            )
            result = evaluate_recipe(recipe, context)
            metrics = best_threshold(result, train_cases)
            choices.append((recipe_rank(metrics, recipe), recipe, result, metrics))
        _, recipe, result, metrics = choose_recipe_entry(choices)
        selected = list(recipe["weights"])
        path.append((recipe, result, metrics))
    return path


def coordinate_refine(
    recipe: dict[str, Any],
    context: dict[str, Any],
    train_cases: set[str],
    passes: int = 2,
    step: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = evaluate_recipe(recipe, context)
    metrics = best_threshold(result, train_cases)
    current = (recipe, result, metrics)
    ids = sorted(recipe["weights"])
    for pass_index in range(passes):
        improved = False
        for candidate_id in ids:
            trials = [current]
            for delta in (-step, step):
                weights = dict(current[0]["weights"])
                weights[candidate_id] = max(0.0, weights[candidate_id] + delta)
                if math.fsum(weights.values()) <= 0:
                    continue
                trial = make_recipe(
                    f"coordinate_p{pass_index}_{candidate_id}_{delta:+.2f}",
                    recipe["domain"],
                    weights,
                    "coordinate_refinement",
                )
                trial_result = evaluate_recipe(trial, context)
                trial_metrics = best_threshold(trial_result, train_cases)
                trials.append((trial, trial_result, trial_metrics))
            ranked_trials = [
                (recipe_rank(item[2], item[0]), item[0], item[1], item[2])
                for item in trials
            ]
            _, selected_recipe, selected_result, selected_metrics = (
                choose_recipe_entry(ranked_trials)
            )
            selected = (selected_recipe, selected_result, selected_metrics)
            if selected[0] != current[0]:
                improved = True
                current = selected
        if not improved:
            break
    return current


def held_out_record(
    strategy: str,
    fold: int,
    recipe: dict[str, Any],
    result: dict[str, Any],
    fit_metrics: dict[str, Any],
    held_cases: set[str],
) -> dict[str, Any]:
    held = result_at_threshold(result, held_cases, fit_metrics["threshold"])
    return {
        "strategy": strategy,
        "fold": fold,
        "recipe": recipe,
        "fit_metrics": fit_metrics,
        "held_out_metrics": held,
        "recipe_runtime_seconds": float(result.get("elapsed_seconds", 0.0)),
    }


def aggregate_oof(records: list[dict[str, Any]]) -> dict[str, Any]:
    findings = sum(row["held_out_metrics"]["findings"] for row in records)
    dice = math.fsum(
        row["held_out_metrics"]["dice"] * row["held_out_metrics"]["findings"]
        for row in records
    ) / findings
    hits = sum(row["held_out_metrics"]["hits"] for row in records)
    model_counts = [len(row["recipe"]["weights"]) for row in records]
    uniform_flags = []
    selection_frequency: collections.Counter[str] = collections.Counter()
    categories: dict[str, dict[str, float | int]] = {}
    for row in records:
        selection_frequency.update(row["recipe"]["weights"])
        weights = list(row["recipe"]["weights"].values())
        uniform_flags.append(max(weights) - min(weights) <= 1e-12)
        for category, metrics in row["held_out_metrics"]["categories"].items():
            value = categories.setdefault(
                category,
                {"findings": 0, "weighted_dice": 0.0, "hits": 0},
            )
            value["findings"] += int(metrics["findings"])
            value["weighted_dice"] += (
                float(metrics["dice"]) * int(metrics["findings"])
            )
            value["hits"] += int(metrics["hits"])
    return {
        "findings": findings,
        "dice": dice,
        "hits": hits,
        "hit_rate": hits / findings,
        "mean_model_count": math.fsum(model_counts) / len(model_counts),
        "uniform_fraction": sum(uniform_flags) / len(uniform_flags),
        "mean_recipe_runtime_seconds": math.fsum(
            float(row["recipe_runtime_seconds"]) for row in records
        )
        / len(records),
        "selection_frequency": dict(sorted(selection_frequency.items())),
        "categories": {
            category: {
                "findings": int(value["findings"]),
                "dice": float(value["weighted_dice"]) / int(value["findings"]),
                "hits": int(value["hits"]),
                "hit_rate": int(value["hits"]) / int(value["findings"]),
            }
            for category, value in sorted(categories.items())
        },
    }


def choose_strategy(
    strategies: dict[str, dict[str, Any]],
) -> str:
    best_dice = max(float(values["dice"]) for values in strategies.values())
    eligible = [
        name
        for name, values in strategies.items()
        if float(values["dice"]) >= best_dice - 0.001
    ]
    return min(
        eligible,
        key=lambda name: (
            -strategies[name]["hit_rate"],
            strategies[name]["mean_model_count"],
            -strategies[name]["uniform_fraction"],
            -strategies[name]["dice"],
            name,
        ),
    )


def select_strategy_alternatives(
    strategies: dict[str, dict[str, Any]],
    primary: str,
) -> tuple[str, str]:
    primary_dice = float(strategies[primary]["dice"])
    hit_candidates = [
        name
        for name, values in strategies.items()
        if float(values["dice"]) >= primary_dice - 0.002
    ]
    highest_hit = min(
        hit_candidates,
        key=lambda name: (
            -strategies[name]["hit_rate"],
            -strategies[name]["dice"],
            strategies[name]["mean_model_count"],
            name,
        ),
    )
    small_candidates = [
        name
        for name, values in strategies.items()
        if float(values["dice"]) >= primary_dice - 0.001
    ]
    smallest = min(
        small_candidates,
        key=lambda name: (
            strategies[name]["mean_model_count"],
            -strategies[name]["dice"],
            -strategies[name]["hit_rate"],
            name,
        ),
    )
    return highest_hit, smallest


def fit_strategy_all(
    strategy: str,
    fixed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_cases = {case["name"] for case in context["cases"]}
    if strategy == "fixed":
        return select_best_recipe(fixed, context, all_cases)
    domain = "probability" if strategy.startswith("greedy_probability") else "logit"
    path = greedy_forward(domain, candidates, context, all_cases, max_models=10)
    ranked_path = [
        (recipe_rank(item[2], item[0]), item[0], item[1], item[2])
        for item in path
    ]
    _, best_recipe, best_result, best_metrics = choose_recipe_entry(ranked_path)
    best = (best_recipe, best_result, best_metrics)
    if strategy.endswith("weighted"):
        return coordinate_refine(best[0], context, all_cases)
    return best


def category_probe(
    result: dict[str, Any],
    global_threshold: float,
) -> dict[str, Any]:
    output = {}
    categories = sorted({row["category"] for row in result["rows"]})
    for category in categories:
        rows = [row for row in result["rows"] if row["category"] == category]
        global_metrics = summarize_rows(rows, [global_threshold])[
            threshold_key(global_threshold)
        ]
        if len(rows) < 20:
            output[category] = {
                "findings": len(rows),
                "policy": "global_fallback",
                "threshold": global_threshold,
                "metrics": global_metrics,
            }
            continue
        coarse_refined = best_threshold(
            {"rows": rows},
            {row["case"] for row in rows},
        )
        output[category] = {
            "findings": len(rows),
            "policy": "post_hoc_category_probe",
            "threshold": coarse_refined["threshold"],
            "metrics": coarse_refined,
            "global_threshold_metrics": global_metrics,
        }
    return output


def hard_mask_hit_disagreement(
    recipe: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize finding-level hit diversity for the frozen model list."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    rows_by_id = {
        candidate_id: flatten_eval(by_id[candidate_id], verify_hash=True)[0]
        for candidate_id in recipe["weights"]
    }
    candidate_ids = sorted(rows_by_id)
    keys = sorted(next(iter(rows_by_id.values())))
    if any(set(rows) != set(keys) for rows in rows_by_id.values()):
        raise ValueError("frozen recipe evaluations do not share finding keys")
    hit_vectors = {
        candidate_id: np.asarray(
            [bool(rows_by_id[candidate_id][key]["hit"]) for key in keys],
            dtype=bool,
        )
        for candidate_id in candidate_ids
    }
    matrix = np.stack([hit_vectors[value] for value in candidate_ids], axis=0)
    pairwise = []
    for first, second in itertools.combinations(candidate_ids, 2):
        first_hits = hit_vectors[first]
        second_hits = hit_vectors[second]
        pairwise.append(
            {
                "first": first,
                "second": second,
                "hit_disagreements": int(np.count_nonzero(first_hits != second_hits)),
                "first_only_hits": int(
                    np.count_nonzero(first_hits & ~second_hits)
                ),
                "second_only_hits": int(
                    np.count_nonzero(second_hits & ~first_hits)
                ),
                "union_hits": int(np.count_nonzero(first_hits | second_hits)),
            }
        )
    return {
        "definition": (
            "Historical threshold-0.5 finding hit decisions; disagreement is "
            "descriptive and does not use held-out ensemble predictions."
        ),
        "models": candidate_ids,
        "findings": len(keys),
        "any_model_hit": int(np.count_nonzero(np.any(matrix, axis=0))),
        "all_models_hit": int(np.count_nonzero(np.all(matrix, axis=0))),
        "all_models_miss": int(np.count_nonzero(~np.any(matrix, axis=0))),
        "mixed_hit_decisions": int(
            np.count_nonzero(
                np.any(matrix, axis=0) & ~np.all(matrix, axis=0)
            )
        ),
        "pairwise": pairwise,
    }


def ensemble_case_probability(
    recipe: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    runtime_root: Path,
    chunk_elements: int,
) -> np.ndarray:
    arrays = {
        candidate_id: np.load(
            case_array_path(runtime_root, candidate_id, name),
            mmap_mode="r",
            allow_pickle=False,
        )
        for candidate_id in recipe["weights"]
    }
    output = np.empty(shape, dtype=np.float32)
    for finding_index in range(shape[0]):
        flat = output[finding_index].reshape(-1)
        for start in range(0, flat.size, chunk_elements):
            stop = min(start + chunk_elements, flat.size)
            flat[start:stop] = accumulate_ensemble_chunk(
                arrays,
                recipe["weights"],
                finding_index,
                start,
                stop,
                recipe["domain"],
            )
    return output


def materialize_and_verify(
    recipe: dict[str, Any],
    threshold: float,
    context: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    import nibabel as nib

    label = recipe_hash(
        recipe,
        context["dataset_sha256"],
        context["source_hashes"],
    )[:16]
    root = context["runtime_root"] / "finalists" / label
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in context["cases"]:
        name = case["name"]
        output_path = prediction_dir / name
        if output_path.is_file():
            continue
        gt_img = nib.load(str(context["seg_dir"] / name))
        probabilities = ensemble_case_probability(
            recipe,
            name,
            tuple(int(value) for value in gt_img.shape),
            context["runtime_root"],
            context["chunk_elements"],
        )
        mask = (probabilities >= threshold).astype(np.uint8, copy=False)
        header = gt_img.header.copy()
        header.set_data_dtype(np.uint8)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.nii.gz")
        nib.save(nib.Nifti1Image(mask, gt_img.affine, header), str(temporary))
        os.replace(temporary, output_path)
    eval_path = root / "eval" / "val_quick_global_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--gt_dir",
            str(context["seg_dir"]),
            "--pred_dir",
            str(prediction_dir),
            "--output_json",
            str(eval_path),
            "--dataset_json",
            str(context["dataset_json"]),
            "--num_workers",
            "16",
            "--global_only",
        ],
        check=True,
    )
    evaluated = read_json(eval_path)["summary"]
    if (
        abs(
            float(evaluated["mean_global_dice_per_finding"])
            - float(expected["dice"])
        )
        > 1e-6
        or int(evaluated["total_hits"]) != int(expected["hits"])
    ):
        raise RuntimeError("materialized evaluator metrics differ from streaming metrics")
    return {
        "recipe_id": label,
        "prediction_dir": str(prediction_dir),
        "evaluation_json": str(eval_path),
        "streaming_metrics": expected,
        "evaluator_metrics": evaluated,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Side Experiment 002 Ensemble Selection",
        "",
        "Probability averaging is the default domain; logit averaging is retained "
        "only when its five-fold out-of-fold result is better.",
        "",
        "## Cross-Validated Strategies",
        "",
        "| Strategy | OOF Dice | OOF hits | OOF hit rate | Mean models |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in sorted(
        summary["strategies"].items(),
        key=lambda item: (-item[1]["dice"], item[0]),
    ):
        marker = "**" if name == summary["primary_strategy"] else ""
        lines.append(
            f"| {marker}{name}{marker} | {values['dice']:.6f} | "
            f"{values['hits']} / {values['findings']} | "
            f"{values['hit_rate']:.6f} | {values['mean_model_count']:.2f} |"
        )
    final = summary["final_recipe"]
    lines.extend(
        [
            "",
            "## Frozen Val200 Recipe",
            "",
            f"- Strategy: `{summary['primary_strategy']}`",
            f"- Averaging domain: `{final['recipe']['domain']}`",
            f"- Global threshold: `{final['metrics']['threshold']:.2f}`",
            f"- Models: `{len(final['recipe']['weights'])}`",
            f"- Full-val Dice: `{final['metrics']['dice']:.6f}`",
            f"- Full-val hits: `{final['metrics']['hits']} / "
            f"{final['metrics']['findings']}`",
            f"- Highest-hit OOF strategy within 0.002 Dice: "
            f"`{summary['highest_hit_strategy']}`",
            f"- Smallest OOF strategy within 0.001 Dice: "
            f"`{summary['smallest_strategy']}`",
            "",
            "| Candidate | Weight |",
            "| --- | ---: |",
        ]
    )
    for candidate_id, weight in final["recipe"]["weights"].items():
        lines.append(f"| `{candidate_id}` | {weight:.8f} |")
    lines.extend(
        [
            "",
            f"- Recipe hash: `{final['recipe_hash']}`",
            f"- Deterministic result hash: "
            f"`{summary['deterministic_result_sha256']}`",
            f"- Candidate manifest hash: "
            f"`{summary['candidate_manifest_sha256']}`",
            f"- Fold hash: `{summary['folds_sha256']}`",
            f"- Dataset hash: `{summary['dataset_sha256']}`",
            "",
            "Reproduction command:",
            "",
            "```bash",
            summary["reproduction_command"],
            "```",
            "",
            "## Near-Primary Recipes Refit on All Val200",
            "",
            "| Role | OOF strategy | Domain | Models | Threshold | Full-val Dice | "
            "Full-val hits |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, value in summary["alternative_recipes"].items():
        lines.append(
            f"| {role} | `{value['strategy']}` | "
            f"{value['recipe']['domain']} | {len(value['recipe']['weights'])} | "
            f"{value['metrics']['threshold']:.2f} | "
            f"{value['metrics']['dice']:.6f} | "
            f"{value['metrics']['hits']} / {value['metrics']['findings']} |"
        )
    lines.extend(
        [
            "",
            "The frozen full-val recipe is a post-selection fit. Its out-of-fold "
            "strategy result is the generalization estimate; the full-val score "
            "must not be described as independent validation.",
            "",
            "## Primary Selection Frequency",
            "",
            "| Candidate | Selected folds |",
            "| --- | ---: |",
        ]
    )
    primary_frequency = summary["strategies"][
        summary["primary_strategy"]
    ]["selection_frequency"]
    for candidate_id, count in sorted(
        primary_frequency.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| `{candidate_id}` | {count} / 5 |")
    disagreement = summary["final_model_hit_disagreement"]
    lines.extend(
        [
            "",
            "## Frozen-Model Hard-Mask Disagreement",
            "",
            f"- Findings with mixed hit decisions: "
            f"`{disagreement['mixed_hit_decisions']} / {disagreement['findings']}`",
            f"- Hit by any frozen model: "
            f"`{disagreement['any_model_hit']} / {disagreement['findings']}`",
            f"- Hit by every frozen model: "
            f"`{disagreement['all_models_hit']} / {disagreement['findings']}`",
            "",
            "These are historical threshold-0.5 finding decisions, not an "
            "ensemble score.",
            "",
            "## Model-Count Pareto Curve",
            "",
            "| Domain | Models | OOF Dice | OOF hit rate | Mean cached runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, rows in summary["model_count_pareto"].items():
        for model_count, metrics in sorted(
            rows.items(),
            key=lambda item: int(item[0]),
        ):
            lines.append(
                f"| {domain} | {model_count} | {metrics['dice']:.6f} | "
                f"{metrics['hit_rate']:.6f} | "
                f"{metrics['mean_recipe_runtime_seconds']:.1f} |"
            )
    primary_categories = summary["strategies"][
        summary["primary_strategy"]
    ]["categories"]
    lines.extend(
        [
            "",
            "## Primary Out-of-Fold Categories",
            "",
            "| Category | Findings | OOF Dice | OOF hits | OOF hit rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, metrics in primary_categories.items():
        lines.append(
            f"| `{category}` | {metrics['findings']} | {metrics['dice']:.6f} | "
            f"{metrics['hits']} | {metrics['hit_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Category Threshold Probe",
            "",
            "| Category | Findings | Policy | Threshold | Dice | Hits |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for category, value in summary["category_probe"].items():
        metrics = value["metrics"]
        lines.append(
            f"| `{category}` | {value['findings']} | {value['policy']} | "
            f"{value['threshold']:.2f} | {metrics['dice']:.6f} | "
            f"{metrics['hits']} / {metrics['findings']} |"
        )
    lines.extend(
        [
            "",
            "Category-specific thresholds are post-hoc probes only. Categories "
            "with fewer than 20 findings use the global recipe.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--folds-json", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--chunk-elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-greedy-models", type=int, default=10)
    parser.add_argument("--skip-materialization", action="store_true")
    args = parser.parse_args()
    if args.chunk_elements < 1024:
        raise ValueError("--chunk-elements must be at least 1024")
    manifest = load_manifest(args.manifest)
    candidates = manifest["candidates"]
    cases = load_dataset(args.dataset_json)
    export_manifests = validate_runtime_exports(candidates, args.runtime_root, cases)
    fold_by_case, fold_data = load_folds(args.folds_json, cases)
    source_hashes = {
        candidate_id: export["candidate_source"]["checkpoint"]["sha256"]
        for candidate_id, export in export_manifests.items()
    }
    context = {
        "cases": cases,
        "runtime_root": args.runtime_root,
        "seg_dir": args.seg_dir,
        "dataset_json": args.dataset_json,
        "dataset_sha256": sha256_file(args.dataset_json),
        "source_hashes": source_hashes,
        "chunk_elements": args.chunk_elements,
        "cache_root": args.runtime_root / "search_cache",
    }
    context["cache_root"].mkdir(parents=True, exist_ok=True)
    fixed = fixed_recipes(candidates)
    records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    pareto_records: dict[str, dict[int, list[dict[str, Any]]]] = {
        "probability": collections.defaultdict(list),
        "logit": collections.defaultdict(list),
    }
    all_case_names = {case["name"] for case in cases}
    for fold in range(5):
        held = {name for name, value in fold_by_case.items() if value == fold}
        train = all_case_names - held
        fixed_recipe, fixed_result, fixed_metrics = select_best_recipe(
            fixed,
            context,
            train,
        )
        records["fixed"].append(
            held_out_record(
                "fixed",
                fold,
                fixed_recipe,
                fixed_result,
                fixed_metrics,
                held,
            )
        )
        for domain in ("probability", "logit"):
            path = greedy_forward(
                domain,
                candidates,
                context,
                train,
                max_models=args.max_greedy_models,
            )
            for path_recipe, path_result, path_metrics in path:
                pareto_records[domain][len(path_recipe["weights"])].append(
                    held_out_record(
                        f"greedy_{domain}_size{len(path_recipe['weights']):02d}",
                        fold,
                        path_recipe,
                        path_result,
                        path_metrics,
                        held,
                    )
                )
            ranked_path = [
                (recipe_rank(item[2], item[0]), item[0], item[1], item[2])
                for item in path
            ]
            _, best_recipe, best_result, best_metrics = choose_recipe_entry(
                ranked_path
            )
            best = (best_recipe, best_result, best_metrics)
            strategy = f"greedy_{domain}"
            records[strategy].append(
                held_out_record(
                    strategy,
                    fold,
                    best[0],
                    best[1],
                    best[2],
                    held,
                )
            )
            refined = coordinate_refine(best[0], context, train)
            weighted_strategy = f"{strategy}_weighted"
            records[weighted_strategy].append(
                held_out_record(
                    weighted_strategy,
                    fold,
                    refined[0],
                    refined[1],
                    refined[2],
                    held,
                )
            )
    strategies = {
        strategy: aggregate_oof(values)
        for strategy, values in sorted(records.items())
    }
    primary_strategy = choose_strategy(strategies)
    highest_hit_strategy, smallest_strategy = select_strategy_alternatives(
        strategies,
        primary_strategy,
    )
    model_count_pareto = {
        domain: {
            str(model_count): aggregate_oof(values)
            for model_count, values in sorted(by_count.items())
        }
        for domain, by_count in pareto_records.items()
    }
    final_recipe, final_result, final_metrics = fit_strategy_all(
        primary_strategy,
        fixed,
        candidates,
        context,
    )
    probe = category_probe(final_result, final_metrics["threshold"])
    alternative_recipes = {}
    for role, strategy in (
        ("primary", primary_strategy),
        ("highest_hit_within_0.002", highest_hit_strategy),
        ("smallest_within_0.001", smallest_strategy),
    ):
        recipe, result, metrics = fit_strategy_all(
            strategy,
            fixed,
            candidates,
            context,
        )
        alternative_recipes[role] = {
            "strategy": strategy,
            "recipe": recipe,
            "metrics": metrics,
            "recipe_hash": result["recipe_hash"],
        }
    reproduction_command = (
        "PYTHONDONTWRITEBYTECODE=1 python "
        "side_experiments/sideexp002_multimodel_ensemble_selection/"
        "search_ensembles.py "
        f"--manifest {args.manifest} "
        f"--dataset-json {args.dataset_json} "
        f"--folds-json {args.folds_json} "
        f"--runtime-root {args.runtime_root} "
        f"--seg-dir {args.seg_dir} "
        f"--output-json {args.output_json} "
        f"--output-markdown {args.output_markdown} "
        f"--chunk-elements {args.chunk_elements} "
        f"--max-greedy-models {args.max_greedy_models}"
    )
    if args.skip_materialization:
        reproduction_command += " --skip-materialization"
    final_summary = {
        "schema_version": 1,
        "side_experiment": "sideexp002_multimodel_ensemble_selection",
        "candidate_manifest": str(args.manifest),
        "candidate_manifest_sha256": sha256_file(args.manifest),
        "dataset_json": str(args.dataset_json),
        "dataset_sha256": context["dataset_sha256"],
        "folds_json": str(args.folds_json),
        "folds_sha256": sha256_file(args.folds_json),
        "selection_rules": {
            "primary": "highest OOF Dice; then hit rate, model count, uniformity",
            "default_domain": "probability",
            "threshold_search": {
                "coarse": COARSE_THRESHOLDS,
                "refinement": "within +/-0.04 at 0.01",
            },
        },
        "fold_records": records,
        "strategies": strategies,
        "primary_strategy": primary_strategy,
        "highest_hit_strategy": highest_hit_strategy,
        "smallest_strategy": smallest_strategy,
        "model_count_pareto": model_count_pareto,
        "final_recipe": {
            "recipe": final_recipe,
            "metrics": final_metrics,
            "recipe_hash": final_result["recipe_hash"],
            "source_checkpoint_sha256": {
                candidate_id: source_hashes[candidate_id]
                for candidate_id in final_recipe["weights"]
            },
        },
        "alternative_recipes": alternative_recipes,
        "final_model_hit_disagreement": hard_mask_hit_disagreement(
            final_recipe,
            candidates,
        ),
        "category_probe": probe,
        "reproduction_command": reproduction_command,
    }
    final_summary["deterministic_result_sha256"] = deterministic_result_hash(
        final_summary
    )
    if not args.skip_materialization:
        final_summary["materialized_verification"] = materialize_and_verify(
            final_recipe,
            final_metrics["threshold"],
            context,
            final_metrics,
        )
    atomic_write_json(args.output_json, final_summary)
    atomic_write_text(args.output_markdown, build_report(final_summary))
    print(
        f"Selected {primary_strategy}: OOF Dice "
        f"{strategies[primary_strategy]['dice']:.6f}; full-val recipe "
        f"{final_metrics['dice']:.6f} at threshold {final_metrics['threshold']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
