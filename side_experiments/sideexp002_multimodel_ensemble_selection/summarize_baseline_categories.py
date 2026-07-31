#!/usr/bin/env python3
"""Rank the 20 fixed val200 baseline evaluations globally and by category."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from analyze_candidates import (
    DEFAULT_MANIFEST,
    atomic_write_json,
    atomic_write_text,
    flatten_eval,
    load_manifest,
    read_json,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = (
    REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
)
DEFAULT_OUTPUT_JSON = (
    HERE / "outputs" / "baseline_per_category_performance.json"
)
DEFAULT_OUTPUT_MD = (
    HERE / "outputs" / "baseline_per_category_performance.md"
)
EXPECTED_MODELS = 20
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
HIT_THRESHOLD = 0.1
MASK_THRESHOLD = 0.5
RECOMPOSITION_TOLERANCE = 1e-12
CATEGORY_LABELS = {
    "1a": "Bronchial wall thickening",
    "1b": "Bronchiectasis",
    "1c": "Emphysema",
    "1d": "Septal thickening / reticulation",
    "1e": "Micronodules / tree-in-bud",
    "1f": "Other diffuse lung/airway/pleural abnormality",
    "2a": "Linear opacity, scarring, fibrosis",
    "2b": "Atelectasis / consolidation",
    "2c": "Ground-glass opacity",
    "2d": "Pulmonary nodules / masses",
    "2e": "Pleural effusion / thickening",
    "2f": "Honeycombing",
    "2g": "Pneumothorax",
    "2h": "Other focal lung/airway/pleural finding",
}
DIFFUSE_CATEGORIES = tuple(code for code in CATEGORY_LABELS if code.startswith("1"))
FOCAL_CATEGORIES = tuple(code for code in CATEGORY_LABELS if code.startswith("2"))


def load_category_index(
    dataset_path: Path,
) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    data = read_json(dataset_path)
    cases = data.get("test")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"{dataset_path}: expected {EXPECTED_CASES} cases under 'test'"
        )
    category_by_key: dict[tuple[str, int], str] = {}
    counts = {code: 0 for code in CATEGORY_LABELS}
    seen_cases: set[str] = set()
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(name, str) or not name or name in seen_cases:
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
            index = int(finding_key)
            category = str(categories[finding_key])
            if category not in CATEGORY_LABELS:
                raise ValueError(
                    f"{dataset_path} {name} finding {finding_key}: "
                    f"unknown category {category!r}"
                )
            key = (name, index)
            if key in category_by_key:
                raise ValueError(f"{dataset_path}: duplicate finding {key}")
            category_by_key[key] = category
            counts[category] += 1
    if len(category_by_key) != EXPECTED_FINDINGS:
        raise ValueError(
            f"{dataset_path}: expected {EXPECTED_FINDINGS} findings, "
            f"got {len(category_by_key)}"
        )
    if sum(counts.values()) != EXPECTED_FINDINGS:
        raise ValueError(f"{dataset_path}: category counts do not sum to 381")
    return category_by_key, counts


def rank_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(record) for record in records),
        key=lambda record: (
            -float(record["dice"]),
            -int(record["hits"]),
            str(record["candidate_id"]),
        ),
    )
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
    return ranked


def observed_range(
    ranked_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ranked_records:
        raise ValueError("cannot calculate a range from no records")
    dice = [float(record["dice"]) for record in ranked_records]
    hits = [int(record["hits"]) for record in ranked_records]
    hit_rates = [float(record["hit_rate"]) for record in ranked_records]
    return {
        "best": {
            key: ranked_records[0][key]
            for key in (
                "candidate_id",
                "rank",
                "dice",
                "hits",
                "findings",
                "hit_rate",
            )
        },
        "worst": {
            key: ranked_records[-1][key]
            for key in (
                "candidate_id",
                "rank",
                "dice",
                "hits",
                "findings",
                "hit_rate",
            )
        },
        "dice": {
            "minimum": min(dice),
            "maximum": max(dice),
            "spread": max(dice) - min(dice),
        },
        "hits": {
            "minimum": min(hits),
            "maximum": max(hits),
            "spread": max(hits) - min(hits),
        },
        "hit_rate": {
            "minimum": min(hit_rates),
            "maximum": max(hit_rates),
            "spread": max(hit_rates) - min(hit_rates),
        },
    }


def aggregate_candidate(
    candidate: dict[str, Any],
    category_by_key: dict[tuple[str, int], str],
    category_counts: dict[str, int],
) -> dict[str, Any]:
    rows, overall = flatten_eval(candidate, verify_hash=True)
    if set(rows) != set(category_by_key):
        missing = sorted(set(category_by_key) - set(rows))
        extra = sorted(set(rows) - set(category_by_key))
        raise ValueError(
            f"{candidate['id']}: evaluation/dataset finding mismatch; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    dice_by_category: dict[str, list[float]] = defaultdict(list)
    hits_by_category: Counter[str] = Counter()
    for key, row in rows.items():
        category = category_by_key[key]
        dice_by_category[category].append(float(row["dice"]))
        hits_by_category[category] += int(row["hit"])
    categories: dict[str, dict[str, Any]] = {}
    for code in CATEGORY_LABELS:
        support = category_counts[code]
        values = dice_by_category.get(code, [])
        if len(values) != support:
            raise ValueError(
                f"{candidate['id']} category {code}: expected {support} "
                f"findings, got {len(values)}"
            )
        if support == 0:
            categories[code] = {
                "available": False,
                "findings": 0,
                "dice": None,
                "hits": None,
                "hit_rate": None,
                "rank": None,
            }
            continue
        hits = int(hits_by_category[code])
        categories[code] = {
            "available": True,
            "findings": support,
            "dice": math.fsum(values) / support,
            "hits": hits,
            "hit_rate": hits / support,
            "rank": None,
        }
    recomposed_dice = math.fsum(
        float(categories[code]["dice"]) * category_counts[code]
        for code in CATEGORY_LABELS
        if category_counts[code] > 0
    ) / EXPECTED_FINDINGS
    recomposed_hits = sum(
        int(categories[code]["hits"])
        for code in CATEGORY_LABELS
        if category_counts[code] > 0
    )
    if not math.isclose(
        recomposed_dice,
        float(overall["dice"]),
        rel_tol=0.0,
        abs_tol=RECOMPOSITION_TOLERANCE,
    ):
        raise ValueError(
            f"{candidate['id']}: category Dice {recomposed_dice} does not "
            f"recompose global Dice {overall['dice']}"
        )
    if recomposed_hits != int(overall["hits"]):
        raise ValueError(
            f"{candidate['id']}: category hits {recomposed_hits} do not "
            f"recompose global hits {overall['hits']}"
        )
    return {
        "candidate_id": candidate["id"],
        "experiment_number": int(candidate["experiment_number"]),
        "experiment_name": candidate["experiment_name"],
        "model_id": candidate["model_id"],
        "epoch": int(candidate["epoch"]),
        "architecture": candidate["architecture"],
        "normalization": candidate["normalization"],
        "evaluation_path": candidate["evaluation"]["path"],
        "evaluation_sha256": candidate["evaluation"]["sha256"],
        "overall": {
            "rank": None,
            "dice": float(overall["dice"]),
            "hits": int(overall["hits"]),
            "findings": int(overall["findings"]),
            "hit_rate": float(overall["hit_rate"]),
        },
        "categories": categories,
        "validation": {
            "category_dice_recomposed": recomposed_dice,
            "category_hits_recomposed": recomposed_hits,
            "global_dice_delta": recomposed_dice - float(overall["dice"]),
            "global_hits_delta": recomposed_hits - int(overall["hits"]),
        },
    }


def build_summary(
    manifest_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if len(manifest["candidates"]) != EXPECTED_MODELS:
        raise ValueError(f"{manifest_path}: expected exactly 20 candidates")
    dataset_contract = manifest.get("dataset")
    if not isinstance(dataset_contract, dict):
        raise ValueError(f"{manifest_path}: missing immutable dataset contract")
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_sha256 != dataset_contract.get("sha256"):
        raise ValueError(
            f"{dataset_path}: dataset SHA256 drifted; expected "
            f"{dataset_contract.get('sha256')}, got {dataset_sha256}"
        )
    if (
        int(dataset_contract.get("cases", -1)) != EXPECTED_CASES
        or int(dataset_contract.get("findings", -1)) != EXPECTED_FINDINGS
        or float(dataset_contract.get("threshold", math.nan)) != MASK_THRESHOLD
    ):
        raise ValueError(f"{manifest_path}: invalid dataset count/threshold contract")
    category_by_key, category_counts = load_category_index(dataset_path)
    models = [
        aggregate_candidate(candidate, category_by_key, category_counts)
        for candidate in manifest["candidates"]
    ]
    overall_records = rank_records(
        {
            "candidate_id": model["candidate_id"],
            **model["overall"],
        }
        for model in models
    )
    model_by_id = {model["candidate_id"]: model for model in models}
    for record in overall_records:
        model_by_id[record["candidate_id"]]["overall"]["rank"] = record["rank"]
    categories: dict[str, dict[str, Any]] = {}
    for code, label in CATEGORY_LABELS.items():
        support = category_counts[code]
        if support == 0:
            categories[code] = {
                "label": label,
                "support": 0,
                "available": False,
                "ranked_models": [],
                "observed_range": None,
            }
            continue
        category_records = rank_records(
            {
                "candidate_id": model["candidate_id"],
                **model["categories"][code],
            }
            for model in models
        )
        for record in category_records:
            model_by_id[record["candidate_id"]]["categories"][code]["rank"] = (
                record["rank"]
            )
        categories[code] = {
            "label": label,
            "support": support,
            "available": True,
            "ranked_models": category_records,
            "observed_range": observed_range(category_records),
        }
    ordered_models = sorted(
        models,
        key=lambda model: int(model["overall"]["rank"]),
    )
    summary = {
        "schema_version": 1,
        "analysis": "sideexp002_fixed_val200_baselines_by_category",
        "definitions": {
            "mask_threshold": MASK_THRESHOLD,
            "hit_threshold": HIT_THRESHOLD,
            "dice_aggregation": "unweighted mean across findings",
            "ranking": (
                "full-precision Dice descending, hits descending, "
                "candidate ID ascending"
            ),
            "range": (
                "observed minimum, maximum, and spread across the 20 models; "
                "not a confidence interval"
            ),
        },
        "sources": {
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "dataset_json": str(dataset_path),
            "dataset_sha256": dataset_sha256,
        },
        "counts": {
            "models": len(ordered_models),
            "cases_per_model": EXPECTED_CASES,
            "findings_per_model": EXPECTED_FINDINGS,
            "represented_categories": sum(
                support > 0 for support in category_counts.values()
            ),
            "official_categories": len(CATEGORY_LABELS),
        },
        "category_counts": category_counts,
        "overall_ranking": overall_records,
        "overall_observed_range": observed_range(overall_records),
        "categories": categories,
        "models": ordered_models,
        "validation": {
            "candidate_count_is_20": len(ordered_models) == EXPECTED_MODELS,
            "category_counts_sum_to_381": (
                sum(category_counts.values()) == EXPECTED_FINDINGS
            ),
            "all_model_category_recompositions_passed": all(
                abs(float(model["validation"]["global_dice_delta"]))
                <= RECOMPOSITION_TOLERANCE
                and int(model["validation"]["global_hits_delta"]) == 0
                for model in ordered_models
            ),
        },
        "warnings": [
            (
                "Category 2f has no val200 findings; all model metrics and "
                "ranks are unavailable."
            ),
            (
                "Rare-category rankings are descriptive and unstable, "
                "especially category 2g with one finding."
            ),
            (
                "These validation-set rankings and observed ranges do not "
                "establish test-set generalization."
            ),
        ],
    }
    return summary


def support_note(support: int) -> str:
    if support == 0:
        return "Unavailable"
    if support < 10:
        return "Very rare; descriptive only"
    if support < 20:
        return "Low support"
    return "Higher support"


def metric_cell(metric: dict[str, Any]) -> str:
    if not metric["available"]:
        return "—"
    value = (
        f"#{metric['rank']} {metric['dice']:.6f} "
        f"({metric['hits']}/{metric['findings']})"
    )
    return f"**{value}**" if int(metric["rank"]) == 1 else value


def build_matrix(
    summary: dict[str, Any],
    codes: tuple[str, ...],
    title: str,
) -> list[str]:
    header = "| Global rank | Candidate | " + " | ".join(
        f"{code} (N={summary['category_counts'][code]})" for code in codes
    ) + " |"
    separator = "| ---: | --- | " + " | ".join("---" for _ in codes) + " |"
    lines = [
        f"## {title}",
        "",
        "Each cell is `category rank · Dice (hits/N)`; the Dice-selected "
        "category winner is bold.",
        "",
        header,
        separator,
    ]
    for model in summary["models"]:
        cells = " | ".join(metric_cell(model["categories"][code]) for code in codes)
        lines.append(
            f"| {model['overall']['rank']} | `{model['candidate_id']}` | "
            f"{cells} |"
        )
    lines.append("")
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Side Experiment 002 Baseline Performance by ReX Category",
        "",
        "This report compares the immutable 20-model val200 baseline evaluations "
        "at mask threshold `0.5`. Dice is an unweighted mean across findings; "
        "a hit has Dice `>= 0.1`.",
        "",
        "Ranks use full-precision Dice, then hits, then candidate ID. Reported "
        "ranges are observed min–max spreads across these 20 validation models, "
        "not confidence intervals.",
        "",
        "> **Rare-category warning:** category `2g` has one finding. Its ranking "
        "does not support model-selection conclusions. Category `2f` has no "
        "val200 findings and is unavailable.",
        "",
        "## Category Support",
        "",
        "| Code | Official category | Findings | Interpretation |",
        "| --- | --- | ---: | --- |",
    ]
    for code, category in summary["categories"].items():
        lines.append(
            f"| {code} | {category['label']} | {category['support']} | "
            f"{support_note(int(category['support']))} |"
        )
    lines.extend(
        [
            "",
            "## Overall Val200 Leaderboard",
            "",
            "| Rank | Candidate | Source | Model / arm | Epoch | Architecture | "
            "Dice | Hits | Hit rate |",
            "| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    model_by_id = {
        model["candidate_id"]: model for model in summary["models"]
    }
    for record in summary["overall_ranking"]:
        model = model_by_id[record["candidate_id"]]
        lines.append(
            f"| {record['rank']} | `{model['candidate_id']}` | "
            f"Exp{model['experiment_number']:03d} | `{model['model_id']}` | "
            f"{model['epoch']} | {model['architecture']} | "
            f"{record['dice']:.6f} | {record['hits']} / "
            f"{record['findings']} | {record['hit_rate']:.6f} |"
        )
    overall_range = summary["overall_observed_range"]
    lines.extend(
        [
            "",
            f"Overall Dice range: `{overall_range['dice']['minimum']:.6f}`–"
            f"`{overall_range['dice']['maximum']:.6f}` "
            f"(spread `{overall_range['dice']['spread']:.6f}`). Overall hit "
            f"range: `{overall_range['hits']['minimum']}`–"
            f"`{overall_range['hits']['maximum']}` of 381.",
            "",
            "## Category Winners and Observed Ranges",
            "",
            "| Code | Category | N | Best model | Best Dice | Best hits | "
            "Worst model | Worst Dice | Worst hits | Dice min–max | "
            "Dice spread | Hit range |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )
    for code, category in summary["categories"].items():
        if not category["available"]:
            lines.append(
                f"| {code} | {category['label']} | 0 | — | — | — | — | — | "
                "— | — | — | — |"
            )
            continue
        value_range = category["observed_range"]
        best = value_range["best"]
        worst = value_range["worst"]
        lines.append(
            f"| {code} | {category['label']} | {category['support']} | "
            f"`{best['candidate_id']}` | {best['dice']:.6f} | "
            f"{best['hits']} / {best['findings']} | "
            f"`{worst['candidate_id']}` | {worst['dice']:.6f} | "
            f"{worst['hits']} / {worst['findings']} | "
            f"{value_range['dice']['minimum']:.6f}–"
            f"{value_range['dice']['maximum']:.6f} | "
            f"{value_range['dice']['spread']:.6f} | "
            f"{value_range['hits']['minimum']}–"
            f"{value_range['hits']['maximum']} |"
        )
    lines.append("")
    lines.extend(build_matrix(summary, DIFFUSE_CATEGORIES, "Diffuse Categories"))
    lines.extend(build_matrix(summary, FOCAL_CATEGORIES, "Focal Categories"))
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "- Category rankings use the same val200 set used to select the "
            "20-model pool and are descriptive only.",
            "- Low-support categories can change substantially from one finding; "
            "do not use them alone to select an ensemble.",
            "- `2f` is unavailable because val200 contains no honeycombing "
            "findings.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    manifest_path: Path,
    dataset_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    summary = build_summary(manifest_path, dataset_path)
    atomic_write_json(output_json, summary)
    atomic_write_text(output_markdown, build_markdown(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    summary = run(
        args.manifest,
        args.dataset_json,
        args.output_json,
        args.output_markdown,
    )
    print(
        f"Ranked {summary['counts']['models']} models across "
        f"{summary['counts']['official_categories']} official categories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
