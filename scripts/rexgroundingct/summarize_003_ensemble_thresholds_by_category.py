#!/usr/bin/env python3
"""Summarize an Exp003 ensemble threshold sweep by official ReX category."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


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
GLOBAL_HIT_THRESHOLD = 0.1
RECOMPOSITION_TOLERANCE = 1e-12
EXPECTED_CASE_COUNT = 200
EXPECTED_FINDING_COUNT = 381
EXPECTED_THRESHOLDS = (
    0.10,
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    0.90,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def threshold_label(threshold: float) -> str:
    return f"thr{int(round(threshold * 100)):03d}"


def sorted_finding_keys(entry: dict[str, Any]) -> list[str]:
    findings = entry.get("findings")
    categories = entry.get("categories")
    if not isinstance(findings, dict) or not findings:
        raise ValueError(f"{entry.get('name', '<unknown>')}: missing findings")
    if not isinstance(categories, dict):
        raise ValueError(f"{entry.get('name', '<unknown>')}: missing categories")
    if set(findings) != set(categories):
        raise ValueError(
            f"{entry.get('name', '<unknown>')}: finding/category keys do not match"
        )
    try:
        return sorted(findings, key=lambda value: int(value))
    except ValueError as error:
        raise ValueError(
            f"{entry.get('name', '<unknown>')}: finding keys must be integers"
        ) from error


def load_dataset(
    path: Path,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    data = read_json(path)
    entries = data.get("test") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty test list")

    cases: dict[str, list[tuple[str, str]]] = {}
    counts = {code: 0 for code in CATEGORY_LABELS}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every test entry must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: test entry is missing a name")
        if name in cases:
            raise ValueError(f"{path}: duplicate case name {name}")

        case_findings = []
        for position, finding_key in enumerate(sorted_finding_keys(entry)):
            category = str(entry["categories"][finding_key])
            if category not in CATEGORY_LABELS:
                raise ValueError(
                    f"{path}: unknown category {category!r} for {name} finding {finding_key}"
                )
            case_findings.append((f"finding_{position}", category))
            counts[category] += 1
        cases[name] = case_findings
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_CASE_COUNT} cases, found {len(cases)}"
        )
    if sum(counts.values()) != EXPECTED_FINDING_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_FINDING_COUNT} findings, "
            f"found {sum(counts.values())}"
        )
    return cases, counts


def load_threshold_index(path: Path) -> tuple[list[float], dict[str, dict[str, Any]]]:
    data = read_json(path)
    raw_thresholds = data.get("thresholds")
    raw_rows = data.get("rows")
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError(f"{path}: missing thresholds")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{path}: missing rows")

    thresholds = [float(value) for value in raw_thresholds]
    if len(thresholds) != len(set(thresholds)):
        raise ValueError(f"{path}: duplicate thresholds")
    if thresholds != sorted(thresholds):
        raise ValueError(f"{path}: thresholds must be sorted")
    if thresholds != list(EXPECTED_THRESHOLDS):
        raise ValueError(
            f"{path}: expected thresholds {list(EXPECTED_THRESHOLDS)}, "
            f"found {thresholds}"
        )

    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: threshold rows must be objects")
        label = str(row.get("threshold_label", ""))
        if not label:
            raise ValueError(f"{path}: threshold row is missing threshold_label")
        if label in rows:
            raise ValueError(f"{path}: duplicate threshold row {label}")
        if row.get("status") != "complete":
            raise ValueError(f"{path}: threshold row {label} is not complete")
        row_threshold = float(row.get("threshold", math.nan))
        if not math.isfinite(row_threshold) or threshold_label(row_threshold) != label:
            raise ValueError(f"{path}: threshold row {label} has inconsistent value")
        rows[label] = row

    expected_labels = {threshold_label(value) for value in thresholds}
    if set(rows) != expected_labels:
        missing = sorted(expected_labels - set(rows))
        extra = sorted(set(rows) - expected_labels)
        raise ValueError(f"{path}: threshold row mismatch; missing={missing}, extra={extra}")
    return thresholds, rows


def find_eval_json(ensemble_root: Path, label: str) -> Path:
    eval_json = (
        ensemble_root
        / f"probavg4_{label}"
        / "eval"
        / "val_quick_global_eval.json"
    )
    if not eval_json.is_file():
        raise FileNotFoundError(eval_json)
    return eval_json


def validate_summary_value(
    *,
    context: str,
    name: str,
    observed: float,
    expected: Any,
    tolerance: float = RECOMPOSITION_TOLERANCE,
) -> float:
    expected_float = float(expected)
    delta = abs(observed - expected_float)
    if delta > tolerance:
        raise ValueError(
            f"{context}: {name} mismatch; recomposed={observed}, "
            f"saved={expected_float}, delta={delta}"
        )
    return delta


def evaluate_threshold(
    *,
    threshold: float,
    index_row: dict[str, Any],
    eval_json: Path,
    dataset_cases: dict[str, list[tuple[str, str]]],
    category_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float]]:
    data = read_json(eval_json)
    raw_cases = data.get("cases") if isinstance(data, dict) else None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not isinstance(summary, dict):
        raise ValueError(f"{eval_json}: expected cases and summary")

    params = summary.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{eval_json}: summary params are missing")
    if params.get("global_only") is not True:
        raise ValueError(f"{eval_json}: expected global_only=true")
    if float(params.get("global_hit_thr", math.nan)) != GLOBAL_HIT_THRESHOLD:
        raise ValueError(
            f"{eval_json}: expected global hit threshold {GLOBAL_HIT_THRESHOLD}"
        )

    cases: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError(f"{eval_json}: case entries must be objects")
        name = case.get("file")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{eval_json}: case is missing file")
        if name in cases:
            raise ValueError(f"{eval_json}: duplicate case {name}")
        cases[name] = case

    dataset_names = set(dataset_cases)
    eval_names = set(cases)
    if dataset_names != eval_names:
        missing = sorted(dataset_names - eval_names)
        extra = sorted(eval_names - dataset_names)
        raise ValueError(
            f"{eval_json}: case mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    dice_by_category: dict[str, list[float]] = defaultdict(list)
    hits_by_category: dict[str, int] = defaultdict(int)
    all_dice: list[float] = []
    total_hits = 0
    for name, finding_specs in dataset_cases.items():
        findings = cases[name].get("findings")
        if not isinstance(findings, dict):
            raise ValueError(f"{eval_json}: {name} is missing findings")
        expected_keys = {finding_key for finding_key, _ in finding_specs}
        if set(findings) != expected_keys:
            missing = sorted(expected_keys - set(findings))
            extra = sorted(set(findings) - expected_keys)
            raise ValueError(
                f"{eval_json}: {name} finding mismatch; missing={missing}, extra={extra}"
            )
        for finding_key, category in finding_specs:
            metrics = findings[finding_key]
            if not isinstance(metrics, dict):
                raise ValueError(f"{eval_json}: {name} {finding_key} metrics are invalid")
            dice = float(metrics.get("global_dice", math.nan))
            if not math.isfinite(dice) or not 0.0 <= dice <= 1.0:
                raise ValueError(
                    f"{eval_json}: {name} {finding_key} has invalid Dice {dice}"
                )
            hit = metrics.get("global_hit")
            if not isinstance(hit, bool):
                raise ValueError(
                    f"{eval_json}: {name} {finding_key} has invalid hit value {hit!r}"
                )
            expected_hit = dice >= GLOBAL_HIT_THRESHOLD
            if hit != expected_hit:
                raise ValueError(
                    f"{eval_json}: {name} {finding_key} hit is inconsistent with Dice"
                )
            dice_by_category[category].append(dice)
            hits_by_category[category] += int(hit)
            all_dice.append(dice)
            total_hits += int(hit)

    total_findings = len(all_dice)
    if total_findings != sum(category_counts.values()):
        raise ValueError(
            f"{eval_json}: recomposed {total_findings} findings but dataset has "
            f"{sum(category_counts.values())}"
        )
    recomposed_dice = math.fsum(all_dice) / total_findings
    recomposed_hit_rate = total_hits / total_findings
    context = f"{eval_json} ({threshold:.2f})"

    if int(summary.get("total_cases", -1)) != len(dataset_cases):
        raise ValueError(f"{context}: total_cases mismatch")
    if int(summary.get("total_findings", -1)) != total_findings:
        raise ValueError(f"{context}: total_findings mismatch")
    if int(summary.get("total_hits", -1)) != total_hits:
        raise ValueError(f"{context}: total_hits mismatch")
    if int(summary.get("total_misses", -1)) != total_findings - total_hits:
        raise ValueError(f"{context}: total_misses mismatch")

    dice_delta = validate_summary_value(
        context=context,
        name="mean_global_dice_per_finding",
        observed=recomposed_dice,
        expected=summary.get("mean_global_dice_per_finding"),
    )
    hit_rate_delta = validate_summary_value(
        context=context,
        name="hit_rate",
        observed=recomposed_hit_rate,
        expected=summary.get("hit_rate"),
    )
    index_dice_delta = validate_summary_value(
        context=context,
        name="threshold index Dice",
        observed=recomposed_dice,
        expected=index_row.get("mean_global_dice_per_finding"),
    )
    index_hit_rate_delta = validate_summary_value(
        context=context,
        name="threshold index hit rate",
        observed=recomposed_hit_rate,
        expected=index_row.get("hit_rate"),
    )
    if int(index_row.get("total_hits", -1)) != total_hits:
        raise ValueError(f"{context}: threshold index total_hits mismatch")
    if int(index_row.get("total_findings", -1)) != total_findings:
        raise ValueError(f"{context}: threshold index total_findings mismatch")
    if int(index_row.get("total_cases", -1)) != len(dataset_cases):
        raise ValueError(f"{context}: threshold index total_cases mismatch")

    category_records: dict[str, dict[str, Any]] = {}
    for code in CATEGORY_LABELS:
        values = dice_by_category.get(code, [])
        expected_count = category_counts[code]
        if len(values) != expected_count:
            raise ValueError(
                f"{context}: category {code} has {len(values)} findings, "
                f"expected {expected_count}"
            )
        if not values:
            continue
        hits = hits_by_category[code]
        category_records[code] = {
            "threshold": threshold,
            "threshold_label": threshold_label(threshold),
            "mean_global_dice_per_finding": math.fsum(values) / len(values),
            "total_hits": hits,
            "total_findings": len(values),
            "hit_rate": hits / len(values),
            "_dice_sum": math.fsum(values),
        }

    overall = {
        "threshold": threshold,
        "threshold_label": threshold_label(threshold),
        "mean_global_dice_per_finding": recomposed_dice,
        "total_hits": total_hits,
        "total_findings": total_findings,
        "hit_rate": recomposed_hit_rate,
        "total_cases": len(dataset_cases),
        "eval_json": str(eval_json.resolve()),
    }
    validation = {
        "saved_summary_dice_delta": dice_delta,
        "saved_summary_hit_rate_delta": hit_rate_delta,
        "threshold_index_dice_delta": index_dice_delta,
        "threshold_index_hit_rate_delta": index_hit_rate_delta,
    }
    return overall, category_records, validation


def select_best(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot select a best threshold from an empty list")
    return min(
        records,
        key=lambda record: (
            -float(record["mean_global_dice_per_finding"]),
            -float(record["hit_rate"]),
            float(record["threshold"]),
        ),
    )


def public_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def build_summary(dataset_json: Path, ensemble_root: Path) -> dict[str, Any]:
    dataset_json = dataset_json.resolve()
    ensemble_root = ensemble_root.resolve()
    dataset_cases, category_counts = load_dataset(dataset_json)
    threshold_index_json = ensemble_root / "threshold_sweep_summary.json"
    thresholds, index_rows = load_threshold_index(threshold_index_json)

    overall_rows = []
    by_category: dict[str, list[dict[str, Any]]] = {
        code: [] for code in CATEGORY_LABELS
    }
    validation_rows = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        eval_json = find_eval_json(ensemble_root, label)
        overall, category_records, validation = evaluate_threshold(
            threshold=threshold,
            index_row=index_rows[label],
            eval_json=eval_json,
            dataset_cases=dataset_cases,
            category_counts=category_counts,
        )
        overall_rows.append(overall)
        validation_rows.append(
            {
                "threshold": threshold,
                "threshold_label": label,
                **validation,
            }
        )
        for code, record in category_records.items():
            by_category[code].append(record)

    global_best = select_best(overall_rows)
    if float(global_best["threshold"]) != 0.35:
        raise ValueError(
            "the recomposed global-best threshold is not the expected threshold 0.35"
        )
    reference = next(
        row for row in overall_rows if float(row["threshold"]) == 0.35
    )

    categories: dict[str, dict[str, Any]] = {}
    oracle_dice_sum = 0.0
    oracle_hits = 0
    for code, label in CATEGORY_LABELS.items():
        records = by_category[code]
        support = category_counts[code]
        if support == 0:
            categories[code] = {
                "label": label,
                "support": 0,
                "thresholds": [],
                "best": None,
                "reference_at_global_best_threshold": None,
                "delta_from_global_best_threshold": None,
            }
            continue
        if len(records) != len(thresholds):
            raise ValueError(
                f"category {code}: expected {len(thresholds)} threshold records, "
                f"found {len(records)}"
            )
        best = select_best(records)
        category_reference = next(
            record for record in records if float(record["threshold"]) == 0.35
        )
        oracle_dice_sum += float(best["_dice_sum"])
        oracle_hits += int(best["total_hits"])
        categories[code] = {
            "label": label,
            "support": support,
            "thresholds": [public_metric_record(record) for record in records],
            "best": public_metric_record(best),
            "reference_at_global_best_threshold": public_metric_record(
                category_reference
            ),
            "delta_from_global_best_threshold": {
                "mean_global_dice_per_finding": (
                    float(best["mean_global_dice_per_finding"])
                    - float(category_reference["mean_global_dice_per_finding"])
                ),
                "total_hits": int(best["total_hits"])
                - int(category_reference["total_hits"]),
                "hit_rate": float(best["hit_rate"])
                - float(category_reference["hit_rate"]),
            },
        }

    total_findings = sum(category_counts.values())
    oracle_dice = oracle_dice_sum / total_findings
    oracle_hit_rate = oracle_hits / total_findings
    validation_maxima = {
        key: max(float(row[key]) for row in validation_rows)
        for key in (
            "saved_summary_dice_delta",
            "saved_summary_hit_rate_delta",
            "threshold_index_dice_delta",
            "threshold_index_hit_rate_delta",
        )
    }
    return {
        "schema_version": 1,
        "experiment_id": "003_voxtell_rex_ft_rescue_ablation",
        "analysis": "epoch100_val200_four_model_probability_ensemble_by_category",
        "dataset_json": str(dataset_json),
        "ensemble_root": str(ensemble_root),
        "threshold_index_json": str(threshold_index_json.resolve()),
        "selection_rule": {
            "primary": "maximum category mean global Dice per finding",
            "first_tiebreaker": "higher category hit rate",
            "second_tiebreaker": "lower threshold",
            "global_hit_threshold": GLOBAL_HIT_THRESHOLD,
        },
        "thresholds": thresholds,
        "case_count": len(dataset_cases),
        "finding_count": total_findings,
        "category_counts": category_counts,
        "overall_threshold_sweep": overall_rows,
        "global_best": global_best,
        "categories": categories,
        "per_category_oracle": {
            "description": (
                "Validation-only post-hoc aggregate using each category's "
                "Dice-selected threshold on the same val200 findings."
            ),
            "mean_global_dice_per_finding": oracle_dice,
            "total_hits": oracle_hits,
            "total_findings": total_findings,
            "hit_rate": oracle_hit_rate,
            "comparison_threshold": 0.35,
            "delta_from_global_best_threshold": {
                "mean_global_dice_per_finding": (
                    oracle_dice
                    - float(reference["mean_global_dice_per_finding"])
                ),
                "total_hits": oracle_hits - int(reference["total_hits"]),
                "hit_rate": oracle_hit_rate - float(reference["hit_rate"]),
            },
        },
        "validation": {
            "recomposition_tolerance": RECOMPOSITION_TOLERANCE,
            "category_counts_sum_to_finding_count": (
                sum(category_counts.values()) == total_findings
            ),
            "thresholds_checked": len(validation_rows),
            "per_threshold": validation_rows,
            "maximum_absolute_deltas": validation_maxima,
        },
        "warnings": [
            (
                "Thresholds were selected and evaluated on the same val200 set; "
                "the per-category oracle is optimistic and is not a submission policy."
            ),
            (
                "Category 2f has no val200 findings, so no category-specific "
                "threshold can be estimated."
            ),
            (
                "Category 2g has one val200 finding; its selected threshold is "
                "descriptive and not reliable."
            ),
        ],
    }


def fmt_float(value: Any) -> str:
    return f"{float(value):.6f}"


def fmt_delta(value: Any) -> str:
    return f"{float(value):+.6f}"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    thresholds = [float(value) for value in summary["thresholds"]]
    categories = summary["categories"]
    global_best = summary["global_best"]
    oracle = summary["per_category_oracle"]
    reference_threshold = float(oracle["comparison_threshold"])
    overall_by_threshold = {
        float(row["threshold"]): row for row in summary["overall_threshold_sweep"]
    }
    reference = overall_by_threshold[reference_threshold]

    lines = [
        "# Experiment 003 Ensemble Threshold Sweep by ReX Category",
        "",
        "This report re-aggregates the completed epoch-100 four-model probability",
        "ensemble evaluations on the fixed val200 set. No model inference or",
        "evaluator run was repeated.",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Findings: `{summary['finding_count']}`",
        f"- Thresholds: `{len(thresholds)}`",
        f"- Hit definition: global Dice `>= {GLOBAL_HIT_THRESHOLD:.1f}`",
        (
            "- Best-threshold rule: maximum category mean global Dice per finding; "
            "ties use higher hit rate, then lower threshold."
        ),
        (
            f"- Global reference: threshold `{reference_threshold:.2f}`, Dice "
            f"`{fmt_float(reference['mean_global_dice_per_finding'])}`, hit rate "
            f"`{fmt_float(reference['hit_rate'])}` "
            f"(`{reference['total_hits']}/{reference['total_findings']}`)."
        ),
        "",
        "## Best Threshold Per Category",
        "",
        "| Category | Label | N | Best threshold | Best Dice | Hits / N | Hit rate | "
        f"Dice change vs {reference_threshold:.2f} |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for code, category in categories.items():
        best = category["best"]
        if best is None:
            lines.append(
                f"| {code} | {category['label']} | 0 | — | — | — | — | — |"
            )
            continue
        delta = category["delta_from_global_best_threshold"]
        lines.append(
            f"| {code} | {category['label']} | {category['support']} | "
            f"**{float(best['threshold']):.2f}** | "
            f"**{fmt_float(best['mean_global_dice_per_finding'])}** | "
            f"{best['total_hits']} / {best['total_findings']} | "
            f"{fmt_float(best['hit_rate'])} | "
            f"{fmt_delta(delta['mean_global_dice_per_finding'])} |"
        )

    lines.extend(
        [
            "",
            "## Full Category Threshold Matrix",
            "",
            "Each cell is `mean Dice (hits/N)`. The selected best threshold for each",
            "category is bold.",
            "",
            "| Category | Label | N | "
            + " | ".join(f"{threshold:.2f}" for threshold in thresholds)
            + " |",
            "| --- | --- | ---: | "
            + " | ".join("---:" for _ in thresholds)
            + " |",
        ]
    )
    for code, category in categories.items():
        best = category["best"]
        records = {
            float(record["threshold"]): record for record in category["thresholds"]
        }
        cells = []
        for threshold in thresholds:
            record = records.get(threshold)
            if record is None:
                cells.append("—")
                continue
            cell = (
                f"{fmt_float(record['mean_global_dice_per_finding'])} "
                f"({record['total_hits']}/{record['total_findings']})"
            )
            if best is not None and float(best["threshold"]) == threshold:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(
            f"| {code} | {category['label']} | {category['support']} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Overall Threshold Sweep",
            "",
            "| Threshold | Dice / finding | Hit rate | Hits / findings |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["overall_threshold_sweep"]:
        threshold = float(row["threshold"])
        values = (
            f"{threshold:.2f}",
            fmt_float(row["mean_global_dice_per_finding"]),
            fmt_float(row["hit_rate"]),
            f"{row['total_hits']} / {row['total_findings']}",
        )
        if threshold == float(global_best["threshold"]):
            values = tuple(f"**{value}**" for value in values)
        lines.append("| " + " | ".join(values) + " |")

    oracle_delta = oracle["delta_from_global_best_threshold"]
    lines.extend(
        [
            "",
            "## Validation-Only Per-Category Oracle",
            "",
            "| Policy | Dice / finding | Hit rate | Hits / findings | Dice change | "
            "Hit change |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| One global threshold ({reference_threshold:.2f}) | "
                f"{fmt_float(reference['mean_global_dice_per_finding'])} | "
                f"{fmt_float(reference['hit_rate'])} | "
                f"{reference['total_hits']} / {reference['total_findings']} | "
                "— | — |"
            ),
            (
                "| Per-category Dice-selected thresholds | "
                f"{fmt_float(oracle['mean_global_dice_per_finding'])} | "
                f"{fmt_float(oracle['hit_rate'])} | "
                f"{oracle['total_hits']} / {oracle['total_findings']} | "
                f"{fmt_delta(oracle_delta['mean_global_dice_per_finding'])} | "
                f"{int(oracle_delta['total_hits']):+d} |"
            ),
            "",
            "This oracle selects and evaluates thresholds on the same val200 set. It",
            "measures threshold sensitivity but is optimistic and must not be treated",
            "as a submission recipe or evidence of test-set generalization.",
            "",
            "## Support And Reliability",
            "",
        ]
    )
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    small_categories = [
        f"`{code}` ({category['support']})"
        for code, category in categories.items()
        if 0 < int(category["support"]) < 10
    ]
    lines.append(
        "- Additional low-support categories with fewer than 10 findings: "
        + ", ".join(small_categories)
        + "."
    )

    validation = summary["validation"]
    maximum_deltas = validation["maximum_absolute_deltas"]
    lines.extend(
        [
            "",
            "## Validation",
            "",
            (
                f"- All `{validation['thresholds_checked']}` threshold evaluations "
                "contain the same cases and finding alignment."
            ),
            (
                f"- Category support sums to `{summary['finding_count']}` findings "
                "at every threshold."
            ),
            (
                "- Reconstructed global Dice, hits, and hit rate match the saved "
                f"summaries within `{validation['recomposition_tolerance']:.0e}`."
            ),
            (
                "- Maximum saved-summary Dice delta: "
                f"`{maximum_deltas['saved_summary_dice_delta']:.3e}`."
            ),
            (
                "- Maximum threshold-index Dice delta: "
                f"`{maximum_deltas['threshold_index_dice_delta']:.3e}`."
            ),
            "",
            "## Provenance",
            "",
            f"- Dataset JSON: `{summary['dataset_json']}`",
            f"- Ensemble root: `{summary['ensemble_root']}`",
            f"- Threshold index: `{summary['threshold_index_json']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    summary = build_summary(args.dataset_json, args.ensemble_root)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(f"Category threshold summary JSON: {args.output_json}")
    print(f"Category threshold report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
