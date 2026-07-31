#!/usr/bin/env python3
"""Create deterministic category-balanced val200 case folds for sideexp002."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analyze_candidates import atomic_write_json, sha256_file


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = (
    REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
)
DEFAULT_OUTPUT = HERE / "outputs" / "folds.json"
DEFAULT_SEED = 20260729
EXPECTED_CATEGORIES = {
    "1a",
    "1b",
    "1c",
    "1d",
    "1e",
    "1f",
    "2a",
    "2b",
    "2c",
    "2d",
    "2e",
    "2f",
    "2g",
    "2h",
}


def stable_unit(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    cases = data.get("test") if isinstance(data, dict) else None
    if not isinstance(cases, list) or len(cases) != 200:
        raise ValueError(f"{path}: expected 200 cases under 'test'")
    seen: set[str] = set()
    normalized = []
    total_findings = 0
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"{path}: duplicate or invalid case name {name!r}")
        seen.add(name)
        if not isinstance(findings, dict) or not isinstance(categories, dict):
            raise ValueError(f"{path} {name}: findings/categories must be objects")
        if set(findings) != set(categories):
            raise ValueError(f"{path} {name}: finding/category indices differ")
        unknown = set(categories.values()) - EXPECTED_CATEGORIES
        if unknown:
            raise ValueError(f"{path} {name}: unknown categories {sorted(unknown)}")
        counts = collections.Counter(str(value) for value in categories.values())
        total_findings += len(findings)
        normalized.append(
            {
                "name": name,
                "findings": len(findings),
                "category_counts": dict(sorted(counts.items())),
            }
        )
    if total_findings != 381:
        raise ValueError(f"{path}: expected 381 findings, got {total_findings}")
    return normalized


def assignment_score(
    fold: dict[str, Any],
    case: dict[str, Any],
    target_cases: float,
    target_findings: float,
    target_categories: dict[str, float],
) -> float:
    category_score = 0.0
    for category, target in target_categories.items():
        current = fold["category_counts"].get(category, 0)
        updated = fold["category_counts"].get(category, 0) + case[
            "category_counts"
        ].get(category, 0)
        category_score += (
            (updated - target) ** 2 - (current - target) ** 2
        ) / max(target, 1.0)
    current_findings = fold["findings"]
    updated_findings = fold["findings"] + case["findings"]
    current_cases = len(fold["cases"])
    updated_cases = len(fold["cases"]) + 1
    return (
        category_score
        + 0.35
        * (
            (updated_findings - target_findings) ** 2
            - (current_findings - target_findings) ** 2
        )
        / target_findings
        + 0.15
        * (
            (updated_cases - target_cases) ** 2
            - (current_cases - target_cases) ** 2
        )
        / target_cases
    )


def make_folds(
    cases: list[dict[str, Any]],
    num_folds: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    global_categories: collections.Counter[str] = collections.Counter()
    for case in cases:
        global_categories.update(case["category_counts"])
    target_categories = {
        category: count / num_folds
        for category, count in sorted(global_categories.items())
    }
    target_cases = len(cases) / num_folds
    target_findings = sum(case["findings"] for case in cases) / num_folds
    ordered = sorted(
        cases,
        key=lambda case: (
            -sum(
                count / global_categories[category]
                for category, count in case["category_counts"].items()
            ),
            -case["findings"],
            stable_unit(seed, case["name"]),
            case["name"],
        ),
    )
    folds = [
        {
            "fold": index,
            "cases": [],
            "findings": 0,
            "category_counts": collections.Counter(),
        }
        for index in range(num_folds)
    ]
    for case in ordered:
        scores = []
        for fold in folds:
            score = assignment_score(
                fold,
                case,
                target_cases,
                target_findings,
                target_categories,
            )
            tie = stable_unit(seed, f"{case['name']}:fold{fold['fold']}")
            scores.append(
                (
                    score,
                    len(fold["cases"]),
                    fold["findings"],
                    tie,
                    fold["fold"],
                )
            )
        selected_index = min(scores)[-1]
        selected = folds[selected_index]
        selected["cases"].append(case["name"])
        selected["findings"] += case["findings"]
        selected["category_counts"].update(case["category_counts"])
    for fold in folds:
        fold["cases"].sort()
        fold["category_counts"] = dict(sorted(fold["category_counts"].items()))
    return folds


def validate_folds(
    folds: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    expected = {case["name"] for case in cases}
    observed = [name for fold in folds for name in fold["cases"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("fold assignment has duplicate or missing cases")
    if sum(int(fold["findings"]) for fold in folds) != 381:
        raise ValueError("fold finding counts do not sum to 381")
    expected_categories: collections.Counter[str] = collections.Counter()
    for case in cases:
        expected_categories.update(case["category_counts"])
    observed_categories: collections.Counter[str] = collections.Counter()
    for fold in folds:
        observed_categories.update(fold["category_counts"])
    if observed_categories != expected_categories:
        raise ValueError("fold category counts do not reconstruct the dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()
    cases = load_cases(args.dataset_json)
    folds = make_folds(cases, num_folds=args.num_folds, seed=args.seed)
    validate_folds(folds, cases)
    category_totals: collections.Counter[str] = collections.Counter()
    for case in cases:
        category_totals.update(case["category_counts"])
    output = {
        "schema_version": 1,
        "side_experiment": "sideexp002_multimodel_ensemble_selection",
        "dataset_json": str(args.dataset_json),
        "dataset_sha256": sha256_file(args.dataset_json),
        "seed": args.seed,
        "num_folds": args.num_folds,
        "cases": len(cases),
        "findings": sum(case["findings"] for case in cases),
        "category_totals": dict(sorted(category_totals.items())),
        "folds": folds,
    }
    atomic_write_json(args.output_json, output)
    print(
        "Wrote folds: "
        + ", ".join(
            f"fold{fold['fold']}={len(fold['cases'])} cases/{fold['findings']} findings"
            for fold in folds
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
