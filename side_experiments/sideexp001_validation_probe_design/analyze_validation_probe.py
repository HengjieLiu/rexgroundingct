#!/usr/bin/env python3
"""Design and audit a category-aware small ReXGroundingCT validation probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SIDE_EXPERIMENT_ID = "sideexp001_validation_probe_design"
RECONSTRUCTION_DICE_TOLERANCE = 1e-4
RATIO_LOW = 2.0 / 3.0
RATIO_HIGH = 1.5
ABSOLUTE_SHIFT_THRESHOLD = 0.02
Z_95 = 1.959963984540054
ACCEPTANCE_LIMITS = {
    "macro_dice_rmse": 0.015,
    "p90_dice_rmse": 0.030,
    "macro_hit_rmse": 0.025,
    "p90_hit_rmse": 0.060,
    "worst_dice_rmse": 0.040,
    "worst_hit_rmse": 0.100,
}


@dataclass(frozen=True)
class EvaluationSpec:
    experiment: str
    arm: str
    checkpoint: int
    role: str
    val200_path: Path
    val20_path: Path | None

    @property
    def row_id(self) -> str:
        return f"exp{self.experiment}:{self.arm}:epoch{self.checkpoint:03d}"


@dataclass
class EvaluationCollection:
    specs: list[EvaluationSpec]
    dice: np.ndarray
    hit: np.ndarray
    finding_dice: np.ndarray
    finding_hit: np.ndarray
    population_counts: np.ndarray

    @property
    def groups(self) -> list[str]:
        return [spec.experiment for spec in self.specs]

    @property
    def row_ids(self) -> list[str]:
        return [spec.row_id for spec in self.specs]


@dataclass
class StrategyResult:
    budget: int
    census_cutoff: int
    mandatory_cases: int
    candidate_count: int
    selection: tuple[int, ...]
    design_metrics: dict[str, float]
    crossval_metrics: dict[str, float]
    crossval_error: float
    distribution_score: float
    category_counts: list[int]
    finding_count: int
    fold_results: list[dict[str, Any]]
    frontier: bool = False
    knee: bool = False
    selected: bool = False


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def category_sort_key(value: str) -> tuple[int, str]:
    prefix = ""
    suffix = ""
    for char in value:
        if char.isdigit():
            prefix += char
        else:
            suffix += char
    return (int(prefix or 0), suffix)


def expand_evaluation_specs(manifest: dict[str, Any], repo_root: Path) -> list[EvaluationSpec]:
    specs: list[EvaluationSpec] = []
    for group in manifest["source_groups"]:
        root = resolve_path(group["root"], repo_root)
        for arm in group["arms"]:
            for checkpoint in group["checkpoints"]:
                val200_path = (
                    root
                    / arm
                    / f"eval_epoch{int(checkpoint):03d}_val200"
                    / "eval"
                    / "val_quick_global_eval.json"
                )
                val20_path = None
                if group["has_legacy_val20"]:
                    val20_path = (
                        root
                        / arm
                        / f"eval_epoch{int(checkpoint):03d}_val20"
                        / "eval"
                        / "val_quick_global_eval.json"
                    )
                specs.append(
                    EvaluationSpec(
                        experiment=str(group["experiment"]),
                        arm=arm,
                        checkpoint=int(checkpoint),
                        role=group["role"],
                        val200_path=val200_path,
                        val20_path=val20_path,
                    )
                )
    return specs


def evaluator_entries(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    entries = payload.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list under 'test'")
    return entries


def split_category_counts(
    metadata: dict[str, Any],
    split: str,
    categories: Sequence[str],
) -> tuple[int, int, dict[str, int]]:
    entries = metadata.get(split)
    if not isinstance(entries, list):
        raise ValueError(f"metadata has no list split {split!r}")
    counts = {category: 0 for category in categories}
    total = 0
    for entry in entries:
        for category in entry.get("categories", {}).values():
            if category not in counts:
                raise ValueError(f"{split}: unknown category {category!r}")
            counts[category] += 1
            total += 1
    return len(entries), total, counts


def _firestore_scalar(value: dict[str, Any]) -> Any:
    for key in ("integerValue", "doubleValue", "stringValue", "booleanValue"):
        if key in value:
            item = value[key]
            return int(item) if key == "integerValue" else item
    raise ValueError(f"unsupported Firestore scalar: {value}")


def extract_public_leaderboard_counts(
    path: Path,
    categories: Sequence[str],
) -> dict[str, Any]:
    payload = read_json(path)
    observations: dict[tuple[Any, ...], int] = {}
    excluded_legacy_unknown_rows = 0
    for document in payload.get("documents", []):
        fields = document.get("fields", {})
        if _firestore_scalar(fields["leaderboardSplit"]) != "public-50pct":
            continue
        per_category = fields["perCategory"]["mapValue"]["fields"]
        # One archived submission used a legacy "unknown" bucket for three
        # findings that the released taxonomy counts as category 2d. Preserve
        # that fact in the audit, but do not let a submission-specific legacy
        # label redefine the public-subset distribution.
        if "unknown" in per_category:
            excluded_legacy_unknown_rows += 1
            continue
        counts = {
            category: (
                int(_firestore_scalar(per_category[category]["mapValue"]["fields"]["n"]))
                if category in per_category
                else 0
            )
            for category in categories
        }
        observation = (
            int(_firestore_scalar(fields["totalCases"])),
            int(_firestore_scalar(fields["totalFindings"])),
            tuple(counts[category] for category in categories),
        )
        observations[observation] = observations.get(observation, 0) + 1
    if len(observations) != 1:
        raise ValueError(
            f"{path}: expected one consistent public-50pct count definition, "
            f"found {len(observations)}"
        )
    (cases, findings, values), supporting_rows = next(iter(observations.items()))
    if cases != 150 or findings != 302 or sum(values) != findings:
        raise ValueError(
            f"{path}: unexpected public subset totals cases={cases}, "
            f"findings={findings}, category_sum={sum(values)}"
        )
    return {
        "label": "official public 50% leaderboard subset (membership undisclosed)",
        "cases": cases,
        "findings": findings,
        "counts": dict(zip(categories, values)),
        "leaderboard_split": "public-50pct",
        "supporting_rows": supporting_rows,
        "excluded_legacy_unknown_rows": excluded_legacy_unknown_rows,
        "source": str(path),
        "sha256": sha256_file(path),
    }


def build_validation_layout(
    val_entries: Sequence[dict[str, Any]],
    categories: Sequence[str],
) -> dict[str, Any]:
    category_index = {category: index for index, category in enumerate(categories)}
    case_names = [entry["name"] for entry in val_entries]
    if len(case_names) != len(set(case_names)):
        raise ValueError("validation probe contains duplicate case names")
    case_index = {name: index for index, name in enumerate(case_names)}
    case_category_counts = np.zeros((len(val_entries), len(categories)), dtype=np.int64)
    findings: list[tuple[str, str, str]] = []
    finding_index: dict[tuple[str, str], int] = {}
    metadata_by_case: dict[str, dict[str, Any]] = {}
    for case_position, entry in enumerate(val_entries):
        metadata_by_case[entry["name"]] = entry
        finding_keys = set(entry.get("findings", {}))
        category_keys = set(entry.get("categories", {}))
        if finding_keys != category_keys:
            raise ValueError(f"{entry['name']}: finding/category keys do not match")
        for finding_key in sorted(finding_keys, key=int):
            category = entry["categories"][finding_key]
            category_position = category_index[category]
            case_category_counts[case_position, category_position] += 1
            finding_index[(entry["name"], finding_key)] = len(findings)
            findings.append((entry["name"], finding_key, category))
    return {
        "category_index": category_index,
        "case_names": case_names,
        "case_index": case_index,
        "case_category_counts": case_category_counts,
        "findings": findings,
        "finding_index": finding_index,
        "metadata_by_case": metadata_by_case,
    }


def load_evaluation_collection(
    specs: Sequence[EvaluationSpec],
    layout: dict[str, Any],
) -> tuple[EvaluationCollection, dict[str, str]]:
    case_count = len(layout["case_names"])
    category_count = len(layout["category_index"])
    finding_count = len(layout["findings"])
    dice_rows: list[np.ndarray] = []
    hit_rows: list[np.ndarray] = []
    finding_dice_rows: list[np.ndarray] = []
    finding_hit_rows: list[np.ndarray] = []
    input_hashes: dict[str, str] = {}
    expected_cases = set(layout["case_names"])
    for spec in specs:
        if not spec.val200_path.is_file():
            raise FileNotFoundError(spec.val200_path)
        payload = read_json(spec.val200_path)
        cases = payload.get("cases", [])
        observed_cases = {case["file"] for case in cases}
        if observed_cases != expected_cases or len(cases) != case_count:
            raise ValueError(
                f"{spec.val200_path}: case set differs from fixed val200 "
                f"(missing={sorted(expected_cases - observed_cases)}, "
                f"extra={sorted(observed_cases - expected_cases)})"
            )
        dice = np.zeros((case_count, category_count), dtype=np.float64)
        hit = np.zeros((case_count, category_count), dtype=np.float64)
        finding_dice = np.zeros(finding_count, dtype=np.float64)
        finding_hit = np.zeros(finding_count, dtype=np.float64)
        observed_findings = 0
        for case in cases:
            case_name = case["file"]
            entry = layout["metadata_by_case"][case_name]
            case_position = layout["case_index"][case_name]
            expected_keys = set(entry["findings"])
            metric_keys = {
                key.removeprefix("finding_") for key in case.get("findings", {})
            }
            if metric_keys != expected_keys:
                raise ValueError(
                    f"{spec.val200_path}: {case_name} finding keys differ "
                    f"(expected={sorted(expected_keys)}, observed={sorted(metric_keys)})"
                )
            for metric_key, metric in case["findings"].items():
                finding_key = metric_key.removeprefix("finding_")
                category = entry["categories"][finding_key]
                category_position = layout["category_index"][category]
                finding_position = layout["finding_index"][(case_name, finding_key)]
                value = float(metric["global_dice"])
                hit_value = float(bool(metric["global_hit"]))
                dice[case_position, category_position] += value
                hit[case_position, category_position] += hit_value
                finding_dice[finding_position] = value
                finding_hit[finding_position] = hit_value
                observed_findings += 1
        population_counts = layout["case_category_counts"].sum(axis=0)
        reconstructed_dice = float(dice.sum() / population_counts.sum())
        reconstructed_hit = float(hit.sum() / population_counts.sum())
        summary = payload["summary"]
        if not math.isclose(
            reconstructed_dice,
            float(summary["mean_global_dice_per_finding"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{spec.val200_path}: full Dice reconstruction failed")
        if not math.isclose(
            reconstructed_hit,
            float(summary["hit_rate"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{spec.val200_path}: full hit reconstruction failed")
        if observed_findings != int(population_counts.sum()):
            raise ValueError(f"{spec.val200_path}: finding count reconstruction failed")
        dice_rows.append(dice)
        hit_rows.append(hit)
        finding_dice_rows.append(finding_dice)
        finding_hit_rows.append(finding_hit)
        input_hashes[spec.row_id] = sha256_file(spec.val200_path)
    return (
        EvaluationCollection(
            specs=list(specs),
            dice=np.stack(dice_rows),
            hit=np.stack(hit_rows),
            finding_dice=np.stack(finding_dice_rows),
            finding_hit=np.stack(finding_hit_rows),
            population_counts=layout["case_category_counts"].sum(axis=0),
        ),
        input_hashes,
    )


def subset_category_predictions(
    collection: EvaluationCollection,
    selection: Sequence[int],
    case_category_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(selection, dtype=np.int64)
    counts = case_category_counts[positions].sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError("candidate does not cover every validation category")
    dice = collection.dice[:, positions, :].sum(axis=1) / counts
    hit = collection.hit[:, positions, :].sum(axis=1) / counts
    return dice, hit, counts


def full_category_predictions(
    collection: EvaluationCollection,
) -> tuple[np.ndarray, np.ndarray]:
    counts = collection.population_counts
    return collection.dice.sum(axis=1) / counts, collection.hit.sum(axis=1) / counts


def percentile90(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.9))


def pairwise_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    groups: Sequence[str],
    row_mask: np.ndarray,
) -> tuple[float, float]:
    squared_errors: list[float] = []
    agreements: list[float] = []
    for group in sorted(set(groups)):
        positions = [
            index
            for index, value in enumerate(groups)
            if value == group and bool(row_mask[index])
        ]
        for outer, left in enumerate(positions):
            for right in positions[outer + 1 :]:
                predicted_delta = predicted[left] - predicted[right]
                reference_delta = reference[left] - reference[right]
                squared_errors.extend(((predicted_delta - reference_delta) ** 2).tolist())
                nonzero = reference_delta != 0
                if np.any(nonzero):
                    agreements.extend(
                        (
                            np.sign(predicted_delta[nonzero])
                            == np.sign(reference_delta[nonzero])
                        )
                        .astype(float)
                        .tolist()
                    )
    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
    agreement = float(np.mean(agreements)) if agreements else 1.0
    return rmse, agreement


def diagnostic_metrics_from_predictions(
    collection: EvaluationCollection,
    predicted_dice: np.ndarray,
    predicted_hit: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, float]:
    full_dice, full_hit = full_category_predictions(collection)
    dice_error = np.sqrt(
        np.mean((predicted_dice[row_mask] - full_dice[row_mask]) ** 2, axis=0)
    )
    hit_error = np.sqrt(
        np.mean((predicted_hit[row_mask] - full_hit[row_mask]) ** 2, axis=0)
    )
    dice_delta_rmse, dice_sign = pairwise_metrics(
        predicted_dice, full_dice, collection.groups, row_mask
    )
    hit_delta_rmse, hit_sign = pairwise_metrics(
        predicted_hit, full_hit, collection.groups, row_mask
    )
    return {
        "macro_dice_rmse": float(np.mean(dice_error)),
        "p90_dice_rmse": percentile90(dice_error),
        "worst_dice_rmse": float(np.max(dice_error)),
        "macro_hit_rmse": float(np.mean(hit_error)),
        "p90_hit_rmse": percentile90(hit_error),
        "worst_hit_rmse": float(np.max(hit_error)),
        "dice_delta_rmse": dice_delta_rmse,
        "hit_delta_rmse": hit_delta_rmse,
        "dice_sign_agreement": dice_sign,
        "hit_sign_agreement": hit_sign,
    }


def diagnostic_loss(metrics: dict[str, float]) -> float:
    return float(
        np.mean(
            [
                metrics["macro_dice_rmse"],
                metrics["p90_dice_rmse"],
                metrics["macro_hit_rmse"],
                metrics["p90_hit_rmse"],
            ]
        )
    )


def candidate_score(
    metrics: dict[str, float],
    distribution_score: float,
    selection: tuple[int, ...],
) -> tuple[Any, ...]:
    category_error = sum(
        metrics[key]
        for key in (
            "macro_dice_rmse",
            "p90_dice_rmse",
            "worst_dice_rmse",
            "macro_hit_rmse",
            "p90_hit_rmse",
            "worst_hit_rmse",
        )
    )
    delta_error = metrics["dice_delta_rmse"] + metrics["hit_delta_rmse"]
    sign_penalty = 2.0 - (
        metrics["dice_sign_agreement"] + metrics["hit_sign_agreement"]
    )
    return (category_error, delta_error, sign_penalty, distribution_score, selection)


def aggregate_crossval_metrics(fold_metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in fold_metrics[0]:
        values = [metrics[key] for metrics in fold_metrics]
        output[key] = (
            float(max(values))
            if key.startswith("worst_")
            else float(np.mean(values))
        )
    return output


def build_difficulty_strata(
    design: EvaluationCollection,
    layout: dict[str, Any],
    categories: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    finding_difficulty = design.finding_dice.mean(axis=0)
    finding_hit = design.finding_hit.mean(axis=0)
    findings = layout["findings"]
    by_category: dict[str, list[int]] = {category: [] for category in categories}
    for finding_position, (_, _, category) in enumerate(findings):
        by_category[category].append(finding_position)
    strata_records: list[dict[str, Any]] = []
    finding_stratum = np.full(len(findings), -1, dtype=np.int64)
    for category in categories:
        positions = by_category[category]
        if len(positions) >= 40:
            bin_count = 3
        elif len(positions) >= 12:
            bin_count = 2
        else:
            bin_count = 1
        ordered = sorted(
            positions,
            key=lambda index: (
                float(finding_difficulty[index]),
                float(finding_hit[index]),
                findings[index][0],
                int(findings[index][1]),
            ),
        )
        for bin_index, members in enumerate(np.array_split(ordered, bin_count)):
            stratum_position = len(strata_records)
            for finding_position in members.tolist():
                finding_stratum[finding_position] = stratum_position
            strata_records.append(
                {
                    "category": category,
                    "difficulty_bin": int(bin_index),
                    "difficulty_bin_count": int(bin_count),
                    "finding_count": int(len(members)),
                    "mean_design_dice": float(
                        np.mean(finding_difficulty[members])
                    ),
                    "mean_design_hit_rate": float(np.mean(finding_hit[members])),
                }
            )
    if np.any(finding_stratum < 0):
        raise ValueError("not every validation finding received a difficulty stratum")
    case_stratum_counts = np.zeros(
        (len(layout["case_names"]), len(strata_records)), dtype=np.int64
    )
    for finding_position, (case_name, _, _) in enumerate(findings):
        case_position = layout["case_index"][case_name]
        case_stratum_counts[case_position, finding_stratum[finding_position]] += 1
    return case_stratum_counts, strata_records


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def distribution_distance(
    selection: Sequence[int],
    case_category_counts: np.ndarray,
    case_stratum_counts: np.ndarray,
) -> float:
    positions = np.asarray(selection, dtype=np.int64)
    selected_categories = case_category_counts[positions].sum(axis=0)
    full_categories = case_category_counts.sum(axis=0)
    selected_strata = case_stratum_counts[positions].sum(axis=0)
    full_strata = case_stratum_counts.sum(axis=0)
    selected_findings_per_case = case_category_counts[positions].sum(axis=1)
    full_findings_per_case = case_category_counts.sum(axis=1)
    max_findings = int(max(full_findings_per_case.max(), selected_findings_per_case.max()))
    selected_hist = np.bincount(
        selected_findings_per_case, minlength=max_findings + 1
    )[1:].astype(float)
    full_hist = np.bincount(
        full_findings_per_case, minlength=max_findings + 1
    )[1:].astype(float)
    distances = [
        total_variation(
            selected_categories / selected_categories.sum(),
            full_categories / full_categories.sum(),
        ),
        total_variation(
            selected_strata / selected_strata.sum(),
            full_strata / full_strata.sum(),
        ),
        total_variation(selected_hist / selected_hist.sum(), full_hist / full_hist.sum()),
    ]
    return float(np.mean(distances))


def mandatory_case_positions(
    census_cutoff: int,
    case_category_counts: np.ndarray,
) -> np.ndarray:
    population_counts = case_category_counts.sum(axis=0)
    census_categories = np.flatnonzero(population_counts <= census_cutoff)
    return np.flatnonzero((case_category_counts[:, census_categories] > 0).any(axis=1))


def generate_candidate(
    rng: np.random.Generator,
    budget: int,
    mandatory: np.ndarray,
    case_stratum_counts: np.ndarray,
) -> tuple[int, ...] | None:
    if len(mandatory) > budget:
        return None
    all_cases = np.arange(case_stratum_counts.shape[0])
    cases_by_stratum = [
        np.flatnonzero(case_stratum_counts[:, index] > 0)
        for index in range(case_stratum_counts.shape[1])
    ]
    for _ in range(64):
        selected = set(int(value) for value in mandatory)
        stratum_order = rng.permutation(len(cases_by_stratum))
        for stratum in stratum_order:
            if selected and np.any(
                case_stratum_counts[np.fromiter(selected, dtype=np.int64), stratum] > 0
            ):
                continue
            choices = [
                int(value)
                for value in cases_by_stratum[stratum]
                if int(value) not in selected
            ]
            if not choices:
                continue
            selected.add(int(rng.choice(choices)))
            if len(selected) > budget:
                break
        if len(selected) > budget:
            continue
        remaining = np.setdiff1d(
            all_cases, np.fromiter(selected, dtype=np.int64), assume_unique=False
        )
        needed = budget - len(selected)
        if needed > len(remaining):
            continue
        if needed:
            selected.update(int(value) for value in rng.choice(remaining, needed, replace=False))
        result = tuple(sorted(selected))
        if np.all(case_stratum_counts[np.asarray(result)].sum(axis=0) > 0):
            return result
    return None


def select_strategies(
    design: EvaluationCollection,
    layout: dict[str, Any],
    categories: Sequence[str],
    case_stratum_counts: np.ndarray,
    budgets: Sequence[int],
    census_cutoffs: Sequence[int],
    seed: int,
    candidate_draws: int,
) -> list[StrategyResult]:
    case_category_counts = layout["case_category_counts"]
    groups = sorted(set(design.groups))
    all_mask = np.ones(len(design.specs), dtype=bool)
    results: list[StrategyResult] = []
    for budget in budgets:
        for census_cutoff in census_cutoffs:
            mandatory = mandatory_case_positions(census_cutoff, case_category_counts)
            rng = np.random.default_rng(seed + budget * 1009 + census_cutoff * 9176)
            candidates: list[dict[str, Any]] = []
            seen: set[tuple[int, ...]] = set()
            attempts = 0
            maximum_attempts = max(candidate_draws * 20, 1000)
            while len(candidates) < candidate_draws and attempts < maximum_attempts:
                attempts += 1
                selection = generate_candidate(
                    rng, budget, mandatory, case_stratum_counts
                )
                if selection is None or selection in seen:
                    continue
                seen.add(selection)
                predicted_dice, predicted_hit, selected_counts = (
                    subset_category_predictions(
                        design, selection, case_category_counts
                    )
                )
                dist = distribution_distance(
                    selection, case_category_counts, case_stratum_counts
                )
                metrics_all = diagnostic_metrics_from_predictions(
                    design, predicted_dice, predicted_hit, all_mask
                )
                train_scores: dict[str, tuple[Any, ...]] = {}
                for heldout_group in groups:
                    train_mask = np.asarray(
                        [group != heldout_group for group in design.groups], dtype=bool
                    )
                    train_metrics = diagnostic_metrics_from_predictions(
                        design, predicted_dice, predicted_hit, train_mask
                    )
                    train_scores[heldout_group] = candidate_score(
                        train_metrics, dist, selection
                    )
                candidates.append(
                    {
                        "selection": selection,
                        "predicted_dice": predicted_dice,
                        "predicted_hit": predicted_hit,
                        "selected_counts": selected_counts,
                        "distribution_score": dist,
                        "metrics_all": metrics_all,
                        "score_all": candidate_score(metrics_all, dist, selection),
                        "train_scores": train_scores,
                    }
                )
            if not candidates:
                continue
            best_all = min(candidates, key=lambda candidate: candidate["score_all"])
            fold_results: list[dict[str, Any]] = []
            fold_metrics: list[dict[str, float]] = []
            for heldout_group in groups:
                best_fold = min(
                    candidates,
                    key=lambda candidate: candidate["train_scores"][heldout_group],
                )
                heldout_mask = np.asarray(
                    [group == heldout_group for group in design.groups], dtype=bool
                )
                metrics = diagnostic_metrics_from_predictions(
                    design,
                    best_fold["predicted_dice"],
                    best_fold["predicted_hit"],
                    heldout_mask,
                )
                fold_metrics.append(metrics)
                fold_results.append(
                    {
                        "heldout_experiment": heldout_group,
                        "selection_case_names": [
                            layout["case_names"][index]
                            for index in best_fold["selection"]
                        ],
                        "metrics": metrics,
                    }
                )
            crossval_metrics = aggregate_crossval_metrics(fold_metrics)
            results.append(
                StrategyResult(
                    budget=budget,
                    census_cutoff=census_cutoff,
                    mandatory_cases=len(mandatory),
                    candidate_count=len(candidates),
                    selection=best_all["selection"],
                    design_metrics=best_all["metrics_all"],
                    crossval_metrics=crossval_metrics,
                    crossval_error=diagnostic_loss(crossval_metrics),
                    distribution_score=float(best_all["distribution_score"]),
                    category_counts=[
                        int(value) for value in best_all["selected_counts"].tolist()
                    ],
                    finding_count=int(best_all["selected_counts"].sum()),
                    fold_results=fold_results,
                )
            )
    return results


def pareto_frontier(results: Sequence[StrategyResult]) -> list[StrategyResult]:
    frontier: list[StrategyResult] = []
    for candidate in results:
        dominated = any(
            other.budget <= candidate.budget
            and other.crossval_error <= candidate.crossval_error
            and (
                other.budget < candidate.budget
                or other.crossval_error < candidate.crossval_error
            )
            for other in results
        )
        candidate.frontier = not dominated
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.budget, item.crossval_error))


def choose_knee(frontier: Sequence[StrategyResult]) -> StrategyResult:
    if not frontier:
        raise ValueError("cannot choose a knee from an empty frontier")
    minimum_budget = min(item.budget for item in frontier)
    maximum_error = max(item.crossval_error for item in frontier)
    denominator_x = 200 - minimum_budget
    denominator_y = maximum_error if maximum_error > 0 else 1.0
    distances = []
    for item in frontier:
        x_value = (item.budget - minimum_budget) / denominator_x
        y_value = item.crossval_error / denominator_y
        distance = (1.0 - x_value) - y_value
        distances.append((distance, -item.budget, -item.census_cutoff, item))
    knee = max(distances, key=lambda value: value[:3])[-1]
    knee.knee = True
    return knee


def passes_acceptance(metrics: dict[str, float]) -> bool:
    return all(metrics[key] <= limit for key, limit in ACCEPTANCE_LIMITS.items())


def choose_accepted_candidate(
    frontier: Sequence[StrategyResult],
    knee: StrategyResult,
) -> StrategyResult | None:
    eligible = sorted(
        [item for item in frontier if item.budget >= knee.budget and item.budget <= 120],
        key=lambda item: (item.budget, item.crossval_error),
    )
    for item in eligible:
        if passes_acceptance(item.crossval_metrics):
            item.selected = True
            return item
    return None


def legacy_reconstruction_rows(
    specs: Sequence[EvaluationSpec],
    collection: EvaluationCollection,
    legacy_positions: Sequence[int],
    case_category_counts: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = np.asarray(legacy_positions, dtype=np.int64)
    finding_count = int(case_category_counts[selected].sum())
    for row_index, spec in enumerate(specs):
        dice = float(collection.dice[row_index, selected, :].sum() / finding_count)
        hits = int(round(float(collection.hit[row_index, selected, :].sum())))
        row = {
            "row_id": spec.row_id,
            "role": spec.role,
            "derived_dice": dice,
            "derived_hits": hits,
            "derived_hit_rate": hits / finding_count,
            "actual_val20_path": str(spec.val20_path) if spec.val20_path else "",
            "actual_dice": "",
            "actual_hits": "",
            "dice_abs_difference": "",
            "passed": "",
        }
        if spec.val20_path is not None:
            if not spec.val20_path.is_file():
                raise FileNotFoundError(spec.val20_path)
            actual = read_json(spec.val20_path)["summary"]
            actual_dice = float(actual["mean_global_dice_per_finding"])
            actual_hits = int(actual["total_hits"])
            difference = abs(dice - actual_dice)
            passed = (
                difference <= RECONSTRUCTION_DICE_TOLERANCE
                and hits == actual_hits
            )
            row.update(
                {
                    "actual_dice": actual_dice,
                    "actual_hits": actual_hits,
                    "dice_abs_difference": difference,
                    "passed": passed,
                }
            )
        rows.append(row)
    return rows


def legacy_bias_decomposition(
    collection: EvaluationCollection,
    legacy_positions: Sequence[int],
    case_category_counts: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(legacy_positions, dtype=np.int64)
    population_counts = case_category_counts.sum(axis=0)
    selected_counts = case_category_counts[selected].sum(axis=0)
    population_weights = population_counts / population_counts.sum()
    selected_weights = selected_counts / selected_counts.sum()
    full_dice, full_hit = full_category_predictions(collection)
    full_overall_dice = (full_dice * population_weights).sum(axis=1)
    full_overall_hit = (full_hit * population_weights).sum(axis=1)
    selected_raw_dice = (
        collection.dice[:, selected, :].sum(axis=(1, 2)) / selected_counts.sum()
    )
    selected_raw_hit = (
        collection.hit[:, selected, :].sum(axis=(1, 2)) / selected_counts.sum()
    )
    composition_dice = (full_dice * selected_weights).sum(axis=1)
    composition_hit = (full_hit * selected_weights).sum(axis=1)
    return {
        "dice": {
            "mean_total_bias": float(np.mean(selected_raw_dice - full_overall_dice)),
            "mean_composition_effect": float(
                np.mean(composition_dice - full_overall_dice)
            ),
            "mean_within_category_effect": float(
                np.mean(selected_raw_dice - composition_dice)
            ),
        },
        "hit_rate": {
            "mean_total_bias": float(np.mean(selected_raw_hit - full_overall_hit)),
            "mean_composition_effect": float(
                np.mean(composition_hit - full_overall_hit)
            ),
            "mean_within_category_effect": float(
                np.mean(selected_raw_hit - composition_hit)
            ),
        },
        "legacy_category_counts": [int(value) for value in selected_counts],
        "legacy_category_weights": [float(value) for value in selected_weights],
        "validation_category_weights": [float(value) for value in population_weights],
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2.0 * total)) / denominator
    half_width = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + Z_95**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def two_proportion_pvalue(
    left_successes: int,
    left_total: int,
    right_successes: int,
    right_total: int,
) -> float:
    pooled = (left_successes + right_successes) / (left_total + right_total)
    variance = pooled * (1.0 - pooled) * (
        1.0 / left_total + 1.0 / right_total
    )
    if variance <= 0:
        return 1.0
    z_value = (left_successes / left_total - right_successes / right_total) / math.sqrt(
        variance
    )
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = count - reverse_rank + 1
        running = min(running, pvalues[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def shift_flag(
    validation_count: int,
    validation_total: int,
    test_count: int,
    test_total: int,
) -> str:
    validation_ratio = validation_count / validation_total
    test_ratio = test_count / test_total
    absolute_shift = abs(test_ratio - validation_ratio)
    if validation_ratio == 0:
        ratio_trigger = test_ratio > 0
    else:
        ratio = test_ratio / validation_ratio
        ratio_trigger = ratio < RATIO_LOW or ratio > RATIO_HIGH
    if absolute_shift >= ABSOLUTE_SHIFT_THRESHOLD:
        return "VAL-TEST SHIFT"
    if ratio_trigger:
        if min(validation_count, test_count) < 5:
            return "LOW-COUNT SHIFT WARNING"
        return "VAL-TEST SHIFT"
    return ""


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    left_nonzero = left > 0
    right_nonzero = right > 0
    left_kl = float(
        np.sum(left[left_nonzero] * np.log2(left[left_nonzero] / midpoint[left_nonzero]))
    )
    right_kl = float(
        np.sum(
            right[right_nonzero]
            * np.log2(right[right_nonzero] / midpoint[right_nonzero])
        )
    )
    return 0.5 * (left_kl + right_kl)


def split_audit_rows(
    categories: Sequence[str],
    split_summaries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    validation = split_summaries["validation"]
    test = split_summaries["test"]
    pvalues = [
        two_proportion_pvalue(
            validation["counts"][category],
            validation["findings"],
            test["counts"][category],
            test["findings"],
        )
        for category in categories
    ]
    adjusted = benjamini_hochberg(pvalues)
    rows: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        validation_count = validation["counts"][category]
        test_count = test["counts"][category]
        validation_ratio = validation_count / validation["findings"]
        test_ratio = test_count / test["findings"]
        ratio = test_ratio / validation_ratio if validation_ratio else ""
        public_count = split_summaries["public_subset"]["counts"][category]
        public_ratio = (
            public_count / split_summaries["public_subset"]["findings"]
        )
        val_low, val_high = wilson_interval(validation_count, validation["findings"])
        test_low, test_high = wilson_interval(test_count, test["findings"])
        row: dict[str, Any] = {
            "category": category,
            "train_count": split_summaries["train"]["counts"][category],
            "train_ratio": split_summaries["train"]["counts"][category]
            / split_summaries["train"]["findings"],
            "validation_count": validation_count,
            "validation_ratio": validation_ratio,
            "validation_ci95_low": val_low,
            "validation_ci95_high": val_high,
            "legacy_val20_count": split_summaries["legacy_val20"]["counts"][category],
            "legacy_val20_ratio": split_summaries["legacy_val20"]["counts"][category]
            / split_summaries["legacy_val20"]["findings"],
            "candidate_count": split_summaries["candidate"]["counts"][category],
            "candidate_ratio": (
                split_summaries["candidate"]["counts"][category]
                / split_summaries["candidate"]["findings"]
                if split_summaries["candidate"]["findings"]
                else 0.0
            ),
            "test_count": test_count,
            "test_ratio": test_ratio,
            "test_ci95_low": test_low,
            "test_ci95_high": test_high,
            "test_minus_validation_percentage_points": 100.0
            * (test_ratio - validation_ratio),
            "test_to_validation_ratio": ratio,
            "two_proportion_pvalue": pvalues[index],
            "bh_adjusted_pvalue": adjusted[index],
            "statistically_significant_q05": adjusted[index] < 0.05,
            "shift_flag": shift_flag(
                validation_count,
                validation["findings"],
                test_count,
                test["findings"],
            ),
            "public_subset_count": public_count,
            "public_subset_ratio": public_ratio,
            "public_minus_full_test_percentage_points": 100.0
            * (public_ratio - test_ratio),
            "public_to_full_test_ratio": (
                public_ratio / test_ratio if test_ratio else ""
            ),
        }
        rows.append(row)
    validation_weights = np.asarray(
        [validation["counts"][category] for category in categories], dtype=float
    )
    test_weights = np.asarray(
        [test["counts"][category] for category in categories], dtype=float
    )
    public_weights = np.asarray(
        [
            split_summaries["public_subset"]["counts"][category]
            for category in categories
        ],
        dtype=float,
    )
    validation_weights /= validation_weights.sum()
    test_weights /= test_weights.sum()
    public_weights /= public_weights.sum()
    distances = {
        "validation_vs_full_test_total_variation": total_variation(
            validation_weights, test_weights
        ),
        "validation_vs_full_test_jensen_shannon_bits": jensen_shannon(
            validation_weights, test_weights
        ),
        "full_test_vs_public_subset_total_variation": total_variation(
            test_weights, public_weights
        ),
        "full_test_vs_public_subset_jensen_shannon_bits": jensen_shannon(
            test_weights, public_weights
        ),
    }
    return rows, distances


def all_row_aggregate_metrics(
    collection: EvaluationCollection,
    selection: Sequence[int],
    case_category_counts: np.ndarray,
    validation_weights: np.ndarray,
    test_weights: np.ndarray,
) -> list[dict[str, Any]]:
    positions = np.asarray(selection, dtype=np.int64)
    predicted_dice, predicted_hit, counts = subset_category_predictions(
        collection, positions, case_category_counts
    )
    raw_dice = collection.dice[:, positions, :].sum(axis=(1, 2)) / counts.sum()
    raw_hit = collection.hit[:, positions, :].sum(axis=(1, 2)) / counts.sum()
    rows = []
    for index, spec in enumerate(collection.specs):
        rows.append(
            {
                "row_id": spec.row_id,
                "role": spec.role,
                "raw_dice": float(raw_dice[index]),
                "raw_hit_rate": float(raw_hit[index]),
                "validation_poststratified_dice": float(
                    np.sum(predicted_dice[index] * validation_weights)
                ),
                "validation_poststratified_hit_rate": float(
                    np.sum(predicted_hit[index] * validation_weights)
                ),
                "test_reweighted_dice_extrapolation": float(
                    np.sum(predicted_dice[index] * test_weights)
                ),
                "test_reweighted_hit_rate_extrapolation": float(
                    np.sum(predicted_hit[index] * test_weights)
                ),
            }
        )
    return rows


def merge_collections(
    design: EvaluationCollection,
    holdout: EvaluationCollection,
) -> EvaluationCollection:
    return EvaluationCollection(
        specs=design.specs + holdout.specs,
        dice=np.concatenate([design.dice, holdout.dice], axis=0),
        hit=np.concatenate([design.hit, holdout.hit], axis=0),
        finding_dice=np.concatenate(
            [design.finding_dice, holdout.finding_dice], axis=0
        ),
        finding_hit=np.concatenate(
            [design.finding_hit, holdout.finding_hit], axis=0
        ),
        population_counts=design.population_counts,
    )


def category_error_rows(
    categories: Sequence[str],
    design: EvaluationCollection,
    holdout: EvaluationCollection,
    selection: Sequence[int],
    case_category_counts: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, collection in (("design", design), ("holdout", holdout)):
        predicted_dice, predicted_hit, selected_counts = subset_category_predictions(
            collection, selection, case_category_counts
        )
        full_dice, full_hit = full_category_predictions(collection)
        dice_rmse = np.sqrt(np.mean((predicted_dice - full_dice) ** 2, axis=0))
        hit_rmse = np.sqrt(np.mean((predicted_hit - full_hit) ** 2, axis=0))
        for index, category in enumerate(categories):
            rows.append(
                {
                    "cohort": label,
                    "category": category,
                    "validation_findings": int(collection.population_counts[index]),
                    "selected_findings": int(selected_counts[index]),
                    "dice_rmse": float(dice_rmse[index]),
                    "hit_rate_rmse": float(hit_rmse[index]),
                }
            )
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def build_report(
    summary: dict[str, Any],
    split_rows: Sequence[dict[str, Any]],
    strategy_rows: Sequence[dict[str, Any]],
) -> str:
    legacy = summary["legacy_bias_decomposition"]
    reconstruction = summary["legacy_reconstruction_audit"]
    selection = summary["selection"]
    lines = [
        "# SideExp001 Category-Aware Validation Probe Investigation",
        "",
        "## Outcome",
        "",
        f"- Recommendation status: **{selection['status']}**",
        f"- Selected case count: `{selection.get('case_count', 'none')}`",
        f"- Selected finding count: `{selection.get('finding_count', 'none')}`",
        f"- Census cutoff: `{selection.get('census_cutoff', 'none')}`",
        f"- Frozen pre-holdout selection SHA256: "
        f"`{selection.get('selection_sha256_pre_holdout', 'none')}`",
        "",
        "The selector used Exp006/007/008 only. Exp009/011 were loaded after",
        "the case set was frozen and were not used for retuning.",
        "",
        "## Legacy Val20 Finding",
        "",
        f"- Reconstruction check: `{reconstruction['passed_rows']}` of "
        f"`{reconstruction['paired_rows']}` paired evaluations met the absolute "
        f"Dice tolerance `{RECONSTRUCTION_DICE_TOLERANCE}` with identical hit counts.",
        f"- Maximum absolute Dice difference: "
        f"`{reconstruction['maximum_dice_abs_difference']:.8f}`.",
        f"- Mean Dice bias versus val200: "
        f"`{legacy['dice']['mean_total_bias']:+.4f}`.",
        f"- Dice category-composition component: "
        f"`{legacy['dice']['mean_composition_effect']:+.4f}`.",
        f"- Dice within-category case-selection component: "
        f"`{legacy['dice']['mean_within_category_effect']:+.4f}`.",
        f"- Mean hit-rate bias versus val200: "
        f"`{legacy['hit_rate']['mean_total_bias']:+.4f}`.",
        "",
        "## Validation Versus Full Test Distribution",
        "",
        "The full released test metadata contains 300 cases and 582 findings.",
        "The public live leaderboard uses an undisclosed 150-case subset with",
        "302 findings; its category counts are secondary reference only.",
        "",
        "| Category | Val n (%) | Full test n (%) | Test-val diff | Ratio | "
        "Flag | Public n (%) | Public-test diff |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in split_rows:
        lines.append(
            "| {category} | {val_n} ({val_pct:.2f}%) | {test_n} "
            "({test_pct:.2f}%) | {diff:+.3f} pp | {ratio} | {flag} | "
            "{public_n} ({public_pct:.2f}%) | {public_diff:+.3f} pp |".format(
                category=row["category"],
                val_n=row["validation_count"],
                val_pct=100.0 * row["validation_ratio"],
                test_n=row["test_count"],
                test_pct=100.0 * row["test_ratio"],
                diff=row["test_minus_validation_percentage_points"],
                ratio=(
                    format_float(row["test_to_validation_ratio"], 2)
                    if row["test_to_validation_ratio"] != ""
                    else "n/a"
                ),
                flag=row["shift_flag"] or "",
                public_n=row["public_subset_count"],
                public_pct=100.0 * row["public_subset_ratio"],
                public_diff=row["public_minus_full_test_percentage_points"],
            )
        )
    distances = summary["split_distances"]
    lines.extend(
        [
            "",
            f"- Validation/full-test total variation: "
            f"`{distances['validation_vs_full_test_total_variation']:.4f}`.",
            f"- Validation/full-test Jensen-Shannon divergence: "
            f"`{distances['validation_vs_full_test_jensen_shannon_bits']:.4f}` bits.",
            f"- Full-test/public-subset total variation: "
            f"`{distances['full_test_vs_public_subset_total_variation']:.4f}`.",
            "",
            "## Accuracy-Cost Frontier",
            "",
            "| Cases | Census cutoff | Mandatory cases | Findings | CV error | "
            "CV macro Dice | CV macro hit | Frontier | Knee | Selected |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in strategy_rows:
        if not row["frontier"] and not row["selected"]:
            continue
        lines.append(
            "| {budget} | {cutoff} | {mandatory} | {findings} | {error:.4f} | "
            "{dice:.4f} | {hit:.4f} | {frontier} | {knee} | {selected} |".format(
                budget=row["budget"],
                cutoff=row["census_cutoff"],
                mandatory=row["mandatory_cases"],
                findings=row["finding_count"],
                error=row["crossval_error"],
                dice=row["cv_macro_dice_rmse"],
                hit=row["cv_macro_hit_rmse"],
                frontier="yes" if row["frontier"] else "",
                knee="yes" if row["knee"] else "",
                selected="yes" if row["selected"] else "",
            )
        )
    if selection.get("holdout_metrics"):
        metrics = selection["holdout_metrics"]
        lines.extend(
            [
                "",
                "## Frozen Holdout Audit",
                "",
                f"- Holdout acceptance: "
                f"`{'passed' if selection['holdout_passed'] else 'failed'}`.",
                f"- Macro category Dice RMSE: "
                f"`{metrics['macro_dice_rmse']:.4f}`.",
                f"- 90th-percentile category Dice RMSE: "
                f"`{metrics['p90_dice_rmse']:.4f}`.",
                f"- Worst-category Dice RMSE: "
                f"`{metrics['worst_dice_rmse']:.4f}`.",
                f"- Macro category hit-rate RMSE: "
                f"`{metrics['macro_hit_rmse']:.4f}`.",
                f"- 90th-percentile category hit-rate RMSE: "
                f"`{metrics['p90_hit_rmse']:.4f}`.",
                f"- Worst-category hit-rate RMSE: "
                f"`{metrics['worst_hit_rmse']:.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Test-reweighted metrics are extrapolations from validation outcomes, "
            "not measured test performance.",
            "- The live leaderboard subset membership is undisclosed; its ratios "
            "cannot identify which test cases are evaluated.",
            "- Category `2f` has no validation or test findings and cannot be "
            "represented in a validation probe.",
            "",
        ]
    )
    return "\n".join(lines)


def strategy_rows(results: Sequence[StrategyResult]) -> list[dict[str, Any]]:
    rows = []
    for result in sorted(
        results, key=lambda item: (item.budget, item.census_cutoff)
    ):
        row = {
            "budget": result.budget,
            "census_cutoff": result.census_cutoff,
            "mandatory_cases": result.mandatory_cases,
            "candidate_count": result.candidate_count,
            "finding_count": result.finding_count,
            "crossval_error": result.crossval_error,
            "distribution_score": result.distribution_score,
            "frontier": result.frontier,
            "knee": result.knee,
            "selected": result.selected,
        }
        row.update(
            {f"cv_{key}": value for key, value in result.crossval_metrics.items()}
        )
        row.update(
            {
                f"design_{key}": value
                for key, value in result.design_metrics.items()
            }
        )
        rows.append(row)
    return rows


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = repository_root()
    manifest_path = resolve_path(args.manifest, repo_root)
    manifest = read_json(manifest_path)
    output_dir = resolve_path(args.output_dir, repo_root)
    metadata_path = resolve_path(manifest["metadata_json"], repo_root)
    val200_path = resolve_path(manifest["val200_json"], repo_root)
    legacy_val20_path = resolve_path(manifest["legacy_val20_json"], repo_root)
    leaderboard_path = resolve_path(
        manifest["public_leaderboard_snapshot"], repo_root
    )
    metadata = read_json(metadata_path)
    val200_entries = evaluator_entries(val200_path)
    legacy_entries = evaluator_entries(legacy_val20_path)
    metadata_categories = {
        category
        for split in ("train", "val", "test")
        for entry in metadata[split]
        for category in entry.get("categories", {}).values()
    }
    categories_all = sorted(metadata_categories, key=category_sort_key)
    val_categories = sorted(
        {
            category
            for entry in val200_entries
            for category in entry.get("categories", {}).values()
        },
        key=category_sort_key,
    )
    if "2f" in val_categories:
        raise ValueError("expected category 2f to be absent from fixed val200")
    layout = build_validation_layout(val200_entries, val_categories)
    metadata_val_names = {entry["name"] for entry in metadata["val"]}
    if set(layout["case_names"]) != metadata_val_names:
        raise ValueError("fixed val200 case set differs from metadata validation split")
    legacy_names = [entry["name"] for entry in legacy_entries]
    legacy_positions = [layout["case_index"][name] for name in legacy_names]
    all_specs = expand_evaluation_specs(manifest, repo_root)
    design_specs = [spec for spec in all_specs if spec.role == "design"]
    holdout_specs = [spec for spec in all_specs if spec.role == "holdout"]
    if len(design_specs) != 16 or len(holdout_specs) != 10:
        raise ValueError(
            f"expected 16 design and 10 holdout rows, found "
            f"{len(design_specs)} and {len(holdout_specs)}"
        )

    design, design_hashes = load_evaluation_collection(design_specs, layout)
    case_stratum_counts, strata_records = build_difficulty_strata(
        design, layout, val_categories
    )
    candidate_draws = int(
        args.candidate_draws
        if args.candidate_draws is not None
        else manifest["candidate_draws_per_strategy"]
    )
    results = select_strategies(
        design=design,
        layout=layout,
        categories=val_categories,
        case_stratum_counts=case_stratum_counts,
        budgets=[int(value) for value in manifest["budgets"]],
        census_cutoffs=[int(value) for value in manifest["census_cutoffs"]],
        seed=int(manifest["analysis_seed"]),
        candidate_draws=candidate_draws,
    )
    if not results:
        raise RuntimeError("no feasible candidate strategies were generated")
    frontier = pareto_frontier(results)
    knee = choose_knee(frontier)
    selected = choose_accepted_candidate(frontier, knee)
    selection_positions = selected.selection if selected is not None else knee.selection
    selection_status = (
        "design_accepted_pending_holdout"
        if selected is not None
        else "no_candidate_passed_crossvalidation"
    )
    freeze_payload = {
        "side_experiment_id": SIDE_EXPERIMENT_ID,
        "analysis_seed": int(manifest["analysis_seed"]),
        "candidate_draws_per_strategy": candidate_draws,
        "case_names": [
            layout["case_names"][index] for index in selection_positions
        ],
        "budget": selected.budget if selected is not None else knee.budget,
        "census_cutoff": (
            selected.census_cutoff if selected is not None else knee.census_cutoff
        ),
        "design_input_hashes": design_hashes,
        "metadata_sha256": sha256_file(metadata_path),
        "val200_json_sha256": sha256_file(val200_path),
    }
    selection_sha256_pre_holdout = sha256_bytes(canonical_json_bytes(freeze_payload))

    # Chronological holdout files are intentionally loaded only after freeze_payload
    # is complete and hashed.
    holdout, holdout_hashes = load_evaluation_collection(holdout_specs, layout)
    holdout_dice, holdout_hit, _ = subset_category_predictions(
        holdout, selection_positions, layout["case_category_counts"]
    )
    holdout_mask = np.ones(len(holdout_specs), dtype=bool)
    holdout_metrics = diagnostic_metrics_from_predictions(
        holdout, holdout_dice, holdout_hit, holdout_mask
    )
    holdout_passed = passes_acceptance(holdout_metrics)
    if selected is None:
        final_status = "rejected_crossvalidation"
    elif not holdout_passed:
        final_status = "rejected_frozen_holdout"
    else:
        final_status = "accepted"

    combined = merge_collections(design, holdout)
    reconstruction = legacy_reconstruction_rows(
        combined.specs,
        combined,
        legacy_positions,
        layout["case_category_counts"],
    )
    bias = legacy_bias_decomposition(
        combined, legacy_positions, layout["case_category_counts"]
    )
    paired_reconstruction = [
        row for row in reconstruction if row["actual_val20_path"]
    ]
    reconstruction_audit = {
        "paired_rows": len(paired_reconstruction),
        "passed_rows": sum(bool(row["passed"]) for row in paired_reconstruction),
        "all_passed": all(bool(row["passed"]) for row in paired_reconstruction),
        "dice_tolerance": RECONSTRUCTION_DICE_TOLERANCE,
        "all_hit_counts_identical": all(
            row["derived_hits"] == row["actual_hits"]
            for row in paired_reconstruction
        ),
        "maximum_dice_abs_difference": max(
            float(row["dice_abs_difference"]) for row in paired_reconstruction
        ),
        "failed_row_ids": [
            row["row_id"] for row in paired_reconstruction if not row["passed"]
        ],
    }

    public_subset = extract_public_leaderboard_counts(
        leaderboard_path, categories_all
    )
    split_summaries: dict[str, dict[str, Any]] = {}
    for key, split in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        cases, findings, counts = split_category_counts(
            metadata, split, categories_all
        )
        split_summaries[key] = {
            "cases": cases,
            "findings": findings,
            "counts": counts,
        }
    legacy_counts_array = layout["case_category_counts"][
        np.asarray(legacy_positions)
    ].sum(axis=0)
    candidate_counts_array = layout["case_category_counts"][
        np.asarray(selection_positions)
    ].sum(axis=0)
    split_summaries["legacy_val20"] = {
        "cases": len(legacy_positions),
        "findings": int(legacy_counts_array.sum()),
        "counts": {
            category: (
                int(legacy_counts_array[val_categories.index(category)])
                if category in val_categories
                else 0
            )
            for category in categories_all
        },
    }
    split_summaries["candidate"] = {
        "cases": len(selection_positions),
        "findings": int(candidate_counts_array.sum()),
        "counts": {
            category: (
                int(candidate_counts_array[val_categories.index(category)])
                if category in val_categories
                else 0
            )
            for category in categories_all
        },
    }
    split_summaries["public_subset"] = {
        "cases": public_subset["cases"],
        "findings": public_subset["findings"],
        "counts": public_subset["counts"],
    }
    split_rows, split_distances = split_audit_rows(
        categories_all, split_summaries
    )

    val_weights = np.asarray(
        [split_summaries["validation"]["counts"][category] for category in val_categories],
        dtype=float,
    )
    test_weights = np.asarray(
        [split_summaries["test"]["counts"][category] for category in val_categories],
        dtype=float,
    )
    val_weights /= val_weights.sum()
    test_weights /= test_weights.sum()
    aggregate_rows = all_row_aggregate_metrics(
        combined,
        selection_positions,
        layout["case_category_counts"],
        val_weights,
        test_weights,
    )
    error_rows = category_error_rows(
        val_categories,
        design,
        holdout,
        selection_positions,
        layout["case_category_counts"],
    )

    chosen_result = selected if selected is not None else knee
    selection_summary = {
        "status": final_status,
        "crossvalidation_status": selection_status,
        "case_count": len(selection_positions),
        "finding_count": int(candidate_counts_array.sum()),
        "budget": chosen_result.budget,
        "census_cutoff": chosen_result.census_cutoff,
        "selection_sha256_pre_holdout": selection_sha256_pre_holdout,
        "crossval_metrics": chosen_result.crossval_metrics,
        "design_metrics": chosen_result.design_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_passed": holdout_passed,
        "case_names": [
            layout["case_names"][index] for index in selection_positions
        ],
        "category_counts": {
            category: int(candidate_counts_array[index])
            for index, category in enumerate(val_categories)
        },
    }
    summary = {
        "side_experiment_id": SIDE_EXPERIMENT_ID,
        "status": final_status,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "split_summaries": {
                key: value
                for key, value in split_summaries.items()
                if key != "public_subset"
            },
        },
        "public_leaderboard_subset": public_subset,
        "val200_json": {
            "path": str(val200_path),
            "sha256": sha256_file(val200_path),
        },
        "legacy_val20_json": {
            "path": str(legacy_val20_path),
            "sha256": sha256_file(legacy_val20_path),
        },
        "evaluation_counts": {
            "design": len(design_specs),
            "holdout": len(holdout_specs),
            "total": len(all_specs),
            "paired_legacy_val20": sum(
                spec.val20_path is not None for spec in all_specs
            ),
        },
        "design_input_hashes": design_hashes,
        "holdout_input_hashes": holdout_hashes,
        "difficulty_strata": strata_records,
        "legacy_reconstruction_audit": reconstruction_audit,
        "legacy_bias_decomposition": bias,
        "split_distances": split_distances,
        "selection": selection_summary,
        "weights": {
            "validation": {
                category: float(val_weights[index])
                for index, category in enumerate(val_categories)
            },
            "full_test": {
                category: float(test_weights[index])
                for index, category in enumerate(val_categories)
            },
        },
        "aggregate_candidate_metrics": aggregate_rows,
        "acceptance_limits": ACCEPTANCE_LIMITS,
    }
    rows_strategy = strategy_rows(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis_summary.json", summary)
    write_csv(
        output_dir / "legacy_reconstruction.csv",
        reconstruction,
        [
            "row_id",
            "role",
            "derived_dice",
            "derived_hits",
            "derived_hit_rate",
            "actual_val20_path",
            "actual_dice",
            "actual_hits",
            "dice_abs_difference",
            "passed",
        ],
    )
    write_csv(
        output_dir / "split_category_audit.csv",
        split_rows,
        list(split_rows[0]),
    )
    write_csv(
        output_dir / "strategy_results.csv",
        rows_strategy,
        list(rows_strategy[0]),
    )
    write_csv(
        output_dir / "selected_category_errors.csv",
        error_rows,
        list(error_rows[0]),
    )
    report = build_report(summary, split_rows, rows_strategy)
    (output_dir / "investigation_report.md").write_text(report)

    candidate_manifest = {
        "side_experiment_id": SIDE_EXPERIMENT_ID,
        "status": final_status,
        "selection_sha256_pre_holdout": selection_sha256_pre_holdout,
        "selection": selection_summary,
        "weights": summary["weights"],
        "source": {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "val200_json": str(val200_path),
            "val200_json_sha256": sha256_file(val200_path),
            "design_input_hashes": design_hashes,
            "holdout_input_hashes": holdout_hashes,
        },
    }
    write_json(output_dir / "candidate_probe_manifest.json", candidate_manifest)
    if final_status == "accepted":
        selected_entries = [
            val200_entries[index] for index in selection_positions
        ]
        probe_filename = (
            f"rexgroundingct_val{len(selection_positions)}_"
            f"{SIDE_EXPERIMENT_ID}.json"
        )
        probe_path = output_dir / probe_filename
        write_json(probe_path, {"test": selected_entries})
        accepted_manifest = dict(candidate_manifest)
        accepted_manifest["probe_json"] = str(probe_path)
        accepted_manifest["probe_json_sha256"] = sha256_file(probe_path)
        write_json(
            output_dir / probe_filename.replace(".json", ".manifest.json"),
            accepted_manifest,
        )
    return summary


def parse_args() -> argparse.Namespace:
    side_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=side_dir / "historical_inputs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=side_dir / "outputs",
    )
    parser.add_argument(
        "--candidate-draws",
        type=int,
        default=None,
        help="Override the deterministic candidate count per budget/census strategy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_analysis(args)
    selection = summary["selection"]
    print(
        "status={status} cases={cases} findings={findings} "
        "holdout_passed={holdout}".format(
            status=selection["status"],
            cases=selection["case_count"],
            findings=selection["finding_count"],
            holdout=selection["holdout_passed"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
