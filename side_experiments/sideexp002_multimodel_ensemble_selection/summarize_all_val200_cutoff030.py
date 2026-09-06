#!/usr/bin/env python3
"""Build the all-method fixed-val200 Dice/category-oracle report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
DEFAULT_AUDIT = REPO_ROOT / "experiments/018_voxtell_category2d_nodule_audit/nodule_method_audit.json"
DEFAULT_OUTPUT_JSON = HERE / "outputs/all_methods_val200_cutoff030_category_oracle.json"
DEFAULT_OUTPUT_MARKDOWN = HERE / "outputs/all_methods_val200_cutoff030_category_oracle.md"

EXPECTED_AUDIT_ROWS = 58
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
OVERALL_DICE_CUTOFF = 0.30
MASK_THRESHOLD = 0.5
HIT_DICE_THRESHOLD = 0.1
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


class ReportError(ValueError):
    """Raised when a source violates the fixed-val200 report contract."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReportError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if path.is_absolute():
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            return path
        return REPO_ROOT / relative
    return REPO_ROOT / path


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _category_items(categories: Any) -> Iterable[tuple[int, str]]:
    if isinstance(categories, dict):
        for index, category in categories.items():
            try:
                yield int(index), str(category)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"invalid category index {index!r}") from exc
        return
    if isinstance(categories, list):
        for index, category in enumerate(categories):
            yield index, str(category)
        return
    raise ReportError(f"unsupported category mapping type: {type(categories).__name__}")


def load_category_index(
    dataset_path: Path,
) -> tuple[dict[tuple[str, int], str], dict[str, int], set[str]]:
    data = read_json(dataset_path)
    cases = data.get("test")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ReportError(f"{dataset_path}: expected {EXPECTED_CASES} test cases")

    category_by_key: dict[tuple[str, int], str] = {}
    counts = {code: 0 for code in CATEGORY_LABELS}
    case_names: set[str] = set()
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(name, str) or not name or name in case_names:
            raise ReportError(f"{dataset_path}: duplicate or invalid case {name!r}")
        if not isinstance(findings, dict) or not isinstance(categories, dict):
            raise ReportError(f"{dataset_path} {name}: missing findings/categories")
        if set(findings) != set(categories):
            raise ReportError(f"{dataset_path} {name}: finding/category key mismatch")
        case_names.add(name)
        for index, category in _category_items(categories):
            if str(index) not in findings:
                raise ReportError(f"{dataset_path} {name}: missing finding {index}")
            category = str(category)
            if category not in CATEGORY_LABELS:
                raise ReportError(f"{dataset_path} {name}: unknown category {category!r}")
            key = (name, index)
            if key in category_by_key:
                raise ReportError(f"{dataset_path}: duplicate finding {key}")
            category_by_key[key] = category
            counts[category] += 1

    if len(category_by_key) != EXPECTED_FINDINGS:
        raise ReportError(
            f"{dataset_path}: expected {EXPECTED_FINDINGS} findings, got {len(category_by_key)}"
        )
    if sum(counts.values()) != EXPECTED_FINDINGS:
        raise ReportError("category supports do not sum to 381")
    return category_by_key, counts, case_names


def _finding_index(key: str) -> int:
    if not isinstance(key, str) or not key.startswith("finding_"):
        raise ReportError(f"invalid evaluator finding key {key!r}")
    try:
        return int(key.removeprefix("finding_"))
    except ValueError as exc:
        raise ReportError(f"invalid evaluator finding key {key!r}") from exc


def _validate_close(observed: float, expected: Any, label: str) -> None:
    try:
        expected_float = float(expected)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{label}: expected a numeric value") from exc
    if not math.isclose(
        observed,
        expected_float,
        rel_tol=0.0,
        abs_tol=RECOMPOSITION_TOLERANCE,
    ):
        raise ReportError(f"{label}: recomposed {observed} != source {expected_float}")


def aggregate_evaluation(
    evaluation: dict[str, Any],
    category_by_key: dict[tuple[str, int], str],
    category_counts: dict[str, int],
    *,
    expected_cases: int | None = EXPECTED_CASES,
    expected_findings: int | None = EXPECTED_FINDINGS,
) -> tuple[dict[tuple[str, int], tuple[float, bool]], dict[str, Any]]:
    """Validate and aggregate one raw evaluator JSON."""

    summary = evaluation.get("summary")
    cases = evaluation.get("cases")
    if not isinstance(summary, dict) or not isinstance(cases, list):
        raise ReportError("evaluator must contain summary and cases")
    params = summary.get("params")
    if not isinstance(params, dict):
        raise ReportError("evaluator summary is missing params")
    if params.get("global_only") is not True:
        raise ReportError("evaluator is not global-only")
    if float(params.get("global_hit_thr", math.nan)) != HIT_DICE_THRESHOLD:
        raise ReportError("evaluator does not use the hit Dice threshold 0.1")
    if expected_cases is not None and summary.get("total_cases") != expected_cases:
        raise ReportError(f"evaluator reports {summary.get('total_cases')} cases, expected {expected_cases}")
    if expected_findings is not None and summary.get("total_findings") != expected_findings:
        raise ReportError(
            f"evaluator reports {summary.get('total_findings')} findings, expected {expected_findings}"
        )

    observed: dict[tuple[str, int], tuple[float, bool]] = {}
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("file"), str):
            raise ReportError("evaluator case is missing its file name")
        case_name = case["file"]
        if case_name in seen_cases:
            raise ReportError(f"evaluator repeats case {case_name}")
        seen_cases.add(case_name)
        findings = case.get("findings")
        if not isinstance(findings, dict):
            raise ReportError(f"evaluator case {case_name} is missing findings")
        for finding_key, record in findings.items():
            index = _finding_index(finding_key)
            key = (case_name, index)
            if key not in category_by_key:
                raise ReportError(f"evaluator contains an unknown finding {key}")
            if key in observed:
                raise ReportError(f"evaluator repeats finding {key}")
            if not isinstance(record, dict):
                raise ReportError(f"malformed evaluator record {key}")
            dice = record.get("global_dice")
            hit = record.get("global_hit")
            if not isinstance(dice, (int, float)) or isinstance(dice, bool) or not math.isfinite(float(dice)):
                raise ReportError(f"non-finite Dice for {key}")
            if not isinstance(hit, bool):
                raise ReportError(f"non-boolean hit for {key}")
            dice = float(dice)
            if hit != (dice >= HIT_DICE_THRESHOLD):
                raise ReportError(f"hit disagreement for {key}")
            observed[key] = (dice, hit)

    if expected_cases is not None and len(seen_cases) != expected_cases:
        raise ReportError(f"evaluator contains {len(seen_cases)} cases, expected {expected_cases}")
    missing = set(category_by_key) - set(observed)
    extra = set(observed) - set(category_by_key)
    if missing or extra:
        raise ReportError(f"evaluator key mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    if expected_findings is not None and len(observed) != expected_findings:
        raise ReportError(f"evaluator contains {len(observed)} findings, expected {expected_findings}")

    overall_dice = math.fsum(dice for dice, _ in observed.values()) / len(observed)
    overall_hits = sum(int(hit) for _, hit in observed.values())
    _validate_close(overall_dice, summary.get("mean_global_dice_per_finding"), "overall Dice")
    if overall_hits != summary.get("total_hits"):
        raise ReportError(f"overall hits: recomposed {overall_hits} != source {summary.get('total_hits')}")

    category_values: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for key, metric in observed.items():
        category_values[category_by_key[key]].append(metric)
    categories: dict[str, dict[str, Any] | None] = {}
    for code in CATEGORY_LABELS:
        values = category_values.get(code, [])
        support = category_counts[code]
        if len(values) != support:
            raise ReportError(f"category {code}: got {len(values)} findings, expected {support}")
        if not values:
            categories[code] = None
            continue
        hits = sum(int(hit) for _, hit in values)
        categories[code] = {
            "available": True,
            "findings": support,
            "dice": math.fsum(dice for dice, _ in values) / support,
            "hits": hits,
            "hit_rate": hits / support,
        }

    recomposed_dice = math.fsum(
        float(categories[code]["dice"]) * category_counts[code]
        for code in CATEGORY_LABELS
        if category_counts[code]
    ) / len(observed)
    recomposed_hits = sum(
        int(categories[code]["hits"])
        for code in CATEGORY_LABELS
        if category_counts[code]
    )
    _validate_close(recomposed_dice, overall_dice, "category-to-overall Dice")
    if recomposed_hits != overall_hits:
        raise ReportError("category-to-overall hit recomposition failed")

    overall = {
        "findings": len(observed),
        "dice": overall_dice,
        "hits": overall_hits,
        "hit_rate": overall_hits / len(observed),
    }
    return observed, {"overall": overall, "categories": categories}


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
    return (
        -float(candidate["overall"]["dice"]),
        -int(candidate["overall"]["hits"]),
        str(candidate["candidate_id"]),
    )


def rank_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted((dict(candidate) for candidate in candidates), key=_candidate_sort_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def category_sort_key(candidate: dict[str, Any], code: str) -> tuple[float, int, float, str]:
    metric = candidate["categories"][code]
    if metric is None:
        raise ReportError(f"candidate {candidate['candidate_id']} has no category {code}")
    return (
        -float(metric["dice"]),
        -int(metric["hits"]),
        -float(candidate["overall"]["dice"]),
        str(candidate["candidate_id"]),
    )


def rank_categories(candidates: list[dict[str, Any]], category_counts: dict[str, int]) -> None:
    for code, support in category_counts.items():
        if not support:
            continue
        ranked = sorted(candidates, key=lambda candidate: category_sort_key(candidate, code))
        for rank, candidate in enumerate(ranked, start=1):
            candidate["categories"][code]["rank"] = rank


def select_category_winners(
    candidates: list[dict[str, Any]], category_counts: dict[str, int]
) -> dict[str, dict[str, Any] | None]:
    winners: dict[str, dict[str, Any] | None] = {}
    for code, support in category_counts.items():
        if not support:
            winners[code] = None
            continue
        winner = min(candidates, key=lambda candidate: category_sort_key(candidate, code))
        metric = winner["categories"][code]
        winners[code] = {
            "candidate_id": winner["candidate_id"],
            "dice": float(metric["dice"]),
            "hits": int(metric["hits"]),
            "findings": int(metric["findings"]),
            "hit_rate": float(metric["hit_rate"]),
            "category_rank": int(metric["rank"]),
        }
    return winners


def build_oracle(
    candidates: list[dict[str, Any]],
    category_counts: dict[str, int],
) -> dict[str, Any]:
    winners = select_category_winners(candidates, category_counts)
    finding_values: list[float] = []
    total_hits = 0
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for code, support in category_counts.items():
        if not support:
            continue
        winner = candidate_by_id[winners[code]["candidate_id"]]
        for key, (dice, hit) in winner["finding_metrics"].items():
            if winner["category_by_key"][key] == code:
                finding_values.append(float(dice))
                total_hits += int(hit)
    if len(finding_values) != EXPECTED_FINDINGS:
        raise ReportError(f"oracle recomposition contains {len(finding_values)} findings")
    dice = math.fsum(finding_values) / EXPECTED_FINDINGS
    weighted_dice = math.fsum(
        float(winners[code]["dice"]) * support
        for code, support in category_counts.items()
        if support
    ) / EXPECTED_FINDINGS
    _validate_close(dice, weighted_dice, "oracle category-weighted Dice")
    return {
        "findings": EXPECTED_FINDINGS,
        "dice": dice,
        "hits": total_hits,
        "hit_rate": total_hits / EXPECTED_FINDINGS,
        "categories": winners,
        "unique_winner_count": len({winner["candidate_id"] for winner in winners.values() if winner}),
    }


def _source_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ReportError("audit row is missing candidate_id")
    if row.get("evaluation_scope") != "fixed_val200":
        raise ReportError(f"{candidate_id}: evaluation scope is not fixed_val200")
    if float(row.get("mask_threshold", math.nan)) != MASK_THRESHOLD:
        raise ReportError(f"{candidate_id}: mask threshold is not 0.5")
    threshold_provenance = row.get("mask_threshold_provenance")
    if not isinstance(threshold_provenance, dict):
        raise ReportError(f"{candidate_id}: missing mask-threshold provenance")
    if float(threshold_provenance.get("value", math.nan)) != MASK_THRESHOLD:
        raise ReportError(f"{candidate_id}: mask-threshold provenance is not 0.5")
    if float(row.get("hit_dice_threshold", math.nan)) != HIT_DICE_THRESHOLD:
        raise ReportError(f"{candidate_id}: hit threshold is not 0.1")
    path_value = row.get("evaluation_path")
    expected_sha = row.get("evaluation_sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise ReportError(f"{candidate_id}: missing evaluator path/hash")
    return {
        "candidate_id": candidate_id,
        "experiment_id": str(row.get("experiment_id", "")),
        "method_family": str(row.get("method_family", "")),
        "model_id": str(row.get("model_id", "")),
        "phase": str(row.get("phase", "")),
        "relative_epoch": row.get("relative_epoch"),
        "absolute_epoch": row.get("absolute_epoch"),
        "method_scope": str(row.get("method_scope", "")),
        "evaluation_path": path_value,
        "evaluation_sha256": expected_sha,
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_provenance": threshold_provenance,
        "hit_dice_threshold": HIT_DICE_THRESHOLD,
    }


def load_candidates(
    audit_path: Path,
    category_by_key: dict[tuple[str, int], str],
    category_counts: dict[str, int],
    case_names: set[str],
) -> tuple[list[dict[str, Any]], str]:
    audit = read_json(audit_path)
    source_rows = audit.get("primary_leaderboard")
    if not isinstance(source_rows, list) or len(source_rows) != EXPECTED_AUDIT_ROWS:
        raise ReportError(f"{audit_path}: expected {EXPECTED_AUDIT_ROWS} primary rows")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in source_rows:
        if not isinstance(row, dict):
            raise ReportError("audit primary row is not an object")
        candidate = _source_candidate(row)
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen_ids:
            raise ReportError(f"duplicate candidate {candidate_id}")
        seen_ids.add(candidate_id)
        path = resolve_path(candidate["evaluation_path"])
        if str(path) in seen_paths:
            raise ReportError(f"duplicate evaluator path {path}")
        seen_paths.add(str(path))
        if not path.is_file():
            raise ReportError(f"{candidate_id}: evaluator is missing: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != candidate["evaluation_sha256"]:
            raise ReportError(
                f"{candidate_id}: evaluator SHA-256 mismatch; expected "
                f"{candidate['evaluation_sha256']}, got {observed_sha}"
            )
        finding_metrics, metrics = aggregate_evaluation(
            read_json(path), category_by_key, category_counts
        )
        candidate.update(metrics)
        candidate["finding_metrics"] = finding_metrics
        candidate["category_by_key"] = category_by_key
        if set(case_names) != {key[0] for key in finding_metrics}:
            raise ReportError(f"{candidate_id}: evaluator cases do not match fixed val200")
        candidates.append(candidate)
    return candidates, sha256_file(audit_path)


def _strip_internal(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key not in {"finding_metrics", "category_by_key"}}


def _display_epoch(candidate: dict[str, Any]) -> str:
    relative = candidate.get("relative_epoch")
    absolute = candidate.get("absolute_epoch")
    if relative is None and absolute is None:
        return "—"
    if relative == absolute or absolute is None:
        return f"e{int(relative):03d}"
    if relative is None:
        return f"abs e{int(absolute):03d}"
    return f"rel e{int(relative):03d} / abs e{int(absolute):03d}"


def _safe_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_cell(candidate: dict[str, Any], code: str) -> str:
    metric = candidate["categories"].get(code)
    if metric is None:
        return "—"
    value = f"#{metric['rank']} {metric['dice']:.6f} ({metric['hits']}/{metric['findings']})"
    return f"**{value}**" if int(metric["rank"]) == 1 else value


def build_markdown(
    *,
    dataset_path: Path,
    audit_path: Path,
    dataset_sha256: str,
    audit_sha256: str,
    category_counts: dict[str, int],
    all_ranked: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    winners: dict[str, dict[str, Any] | None],
    oracle: dict[str, Any],
) -> str:
    best = eligible[0]
    lines = [
        "# All Fixed-Val200 Methods at Dice Cutoff 0.30",
        "",
        "This supplemental Side Experiment 002 report compares the canonical "
        "fixed-val200 single-checkpoint roster. It uses the same global-only "
        "evaluator contract as the Exp018 audit: mask threshold `0.5`, hit "
        "threshold `global Dice >= 0.1`, and an unweighted mean across findings.",
        "",
        f"The inclusive overall Dice cutoff is `>= {OVERALL_DICE_CUTOFF:.2f}`. "
        f"Of {len(all_ranked)} audited methods, {len(eligible)} are eligible and "
        f"{len(excluded)} are excluded by the cutoff.",
        "",
        "> **Oracle warning:** The per-category result below is a retrospective "
        "ground-truth category oracle. It selects a different model for each "
        "known validation category, so it is an optimistic upper-bound analysis "
        "and is not a deployable single-model result.",
        "",
        "## Headline Results",
        "",
        "| Selection | Dice | Hits | Hit rate |",
        "| --- | ---: | ---: | ---: |",
        f"| Best eligible single model: `{best['candidate_id']}` | "
        f"{best['overall']['dice']:.12f} | {best['overall']['hits']}/{EXPECTED_FINDINGS} | "
        f"{best['overall']['hit_rate']:.6f} |",
        f"| Retrospective per-category oracle | {oracle['dice']:.12f} | "
        f"{oracle['hits']}/{EXPECTED_FINDINGS} | {oracle['hit_rate']:.6f} |",
        f"| Oracle gain over best single | +{oracle['dice'] - best['overall']['dice']:.12f} | "
        f"+{oracle['hits'] - best['overall']['hits']} | — |",
        "",
        "## Source and Metric Contract",
        "",
        f"- Dataset: `{dataset_path}` (SHA-256 `{dataset_sha256}`).",
        f"- Candidate roster: `{audit_path}` (SHA-256 `{audit_sha256}`), "
        f"{len(all_ranked)} primary single-checkpoint rows.",
        f"- Validation support: `{EXPECTED_CASES}` cases and `{EXPECTED_FINDINGS}` findings.",
        "- Ranks use full-precision Dice, then total hits, then candidate ID. "
        "Displayed values are rounded only for readability.",
        "- Category Dice is the unweighted mean over findings in that category; "
        "category hits use the same Dice `>= 0.1` rule.",
        "",
        "## Category Support",
        "",
        "| Code | Official category | Findings | Interpretation |",
        "| --- | --- | ---: | --- |",
    ]
    for code, label in CATEGORY_LABELS.items():
        support = category_counts[code]
        if support == 0:
            interpretation = "Unavailable in val200"
        elif support == 1:
            interpretation = "Single finding; descriptive only"
        elif support <= 7:
            interpretation = "Very low support; descriptive only"
        elif support <= 17:
            interpretation = "Low support"
        else:
            interpretation = "Higher support"
        lines.append(f"| {code} | {label} | {support} | {interpretation} |")

    lines.extend(
        [
            "",
            "## Eligible Overall Val200 Leaderboard",
            "",
            "| Rank | Candidate | Method family | Checkpoint | Overall Dice | Hits | Hit rate |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for candidate in eligible:
        overall = candidate["overall"]
        lines.append(
            f"| {candidate['rank']} | `{candidate['candidate_id']}` | "
            f"{_safe_cell(candidate['method_family'])} | {_display_epoch(candidate)} | "
            f"{overall['dice']:.6f} | {overall['hits']}/{EXPECTED_FINDINGS} | "
            f"{overall['hit_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Category Dice and Hit Matrix — Diffuse Categories",
            "",
            "Each cell is `category rank Dice (hits/N)`; bold marks the best "
            "eligible model for that category.",
            "",
            "| Overall rank | Candidate | "
            + " | ".join(f"{code} (N={category_counts[code]})" for code in DIFFUSE_CATEGORIES)
            + " |",
            "| ---: | --- | " + " | ".join("---" for _ in DIFFUSE_CATEGORIES) + " |",
        ]
    )
    for candidate in eligible:
        lines.append(
            f"| {candidate['rank']} | `{candidate['candidate_id']}` | "
            + " | ".join(_metric_cell(candidate, code) for code in DIFFUSE_CATEGORIES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Category Dice and Hit Matrix — Focal Categories",
            "",
            "Each cell is `category rank Dice (hits/N)`; bold marks the best "
            "eligible model for that category. Category `2f` is unavailable.",
            "",
            "| Overall rank | Candidate | "
            + " | ".join(f"{code} (N={category_counts[code]})" for code in FOCAL_CATEGORIES)
            + " |",
            "| ---: | --- | " + " | ".join("---" for _ in FOCAL_CATEGORIES) + " |",
        ]
    )
    for candidate in eligible:
        lines.append(
            f"| {candidate['rank']} | `{candidate['candidate_id']}` | "
            + " | ".join(_metric_cell(candidate, code) for code in FOCAL_CATEGORIES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Category Winners and Oracle Recomposition",
            "",
            "The winner for each supported category is selected by category Dice, "
            "then category hits, overall Dice, and candidate ID. The final oracle "
            "Dice is recomposed from the selected raw finding-level Dice values, "
            "not from rounded table values.",
            "",
            "| Code | Official category | N | Selected candidate | Category Dice | Hits | Hit rate |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for code, label in CATEGORY_LABELS.items():
        winner = winners[code]
        if winner is None:
            lines.append(f"| {code} | {label} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {code} | {label} | {winner['findings']} | `{winner['candidate_id']}` | "
            f"{winner['dice']:.6f} | {winner['hits']}/{winner['findings']} | "
            f"{winner['hit_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"The oracle uses {oracle['unique_winner_count']} unique model checkpoints "
            f"across the supported categories and yields Dice `{oracle['dice']:.12f}` "
            f"with `{oracle['hits']}/{EXPECTED_FINDINGS}` hits.",
            "",
            "## Excluded by Overall Dice Cutoff",
            "",
            "These canonical primary rows are retained here for auditability but "
            "are not used in the eligible leaderboard or category oracle.",
            "",
            "| Full-roster rank | Candidate | Overall Dice | Hits |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for candidate in excluded:
        lines.append(
            f"| {candidate['full_roster_rank']} | `{candidate['candidate_id']}` | "
            f"{candidate['overall']['dice']:.6f} | {candidate['overall']['hits']}/{EXPECTED_FINDINGS} |"
        )
    lines.extend(
        [
            "",
            "Exp021 e50 is also below the `0.30` overall-Dice cutoff and is not "
            "part of the frozen 58-row canonical audit roster, so it does not "
            "enter this selection analysis.",
            "",
            "## Interpretation Limits",
            "",
            "- The oracle uses the known validation category of each finding to route "
            "that finding to a selected checkpoint. It cannot be read as the result "
            "of one model or as a test-set generalization estimate.",
            "- Category `2g` has one finding and category `2f` has no findings; their "
            "category rankings are respectively unstable and unavailable.",
            "- This report does not include ensembles, threshold oracles, partial "
            "target-only evaluations, or incomplete fixed-val200 evidence.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python \\",
            "  side_experiments/sideexp002_multimodel_ensemble_selection/"
            "summarize_all_val200_cutoff030.py",
            "```",
        ]
    )
    return "\n".join(lines)


def build_summary(dataset_path: Path, audit_path: Path) -> dict[str, Any]:
    category_by_key, category_counts, case_names = load_category_index(dataset_path)
    candidates, audit_sha256 = load_candidates(
        audit_path, category_by_key, category_counts, case_names
    )
    all_ranked = rank_candidates(candidates)
    for full_rank, candidate in enumerate(all_ranked, start=1):
        candidate["full_roster_rank"] = full_rank
    eligible = [candidate for candidate in all_ranked if candidate["overall"]["dice"] >= OVERALL_DICE_CUTOFF]
    excluded = [candidate for candidate in all_ranked if candidate["overall"]["dice"] < OVERALL_DICE_CUTOFF]
    if len(eligible) != 50 or len(excluded) != 8:
        raise ReportError(
            f"cutoff contract changed: eligible={len(eligible)}, excluded={len(excluded)}"
        )
    rank_categories(eligible, category_counts)
    winners = select_category_winners(eligible, category_counts)
    oracle = build_oracle(eligible, category_counts)
    best = eligible[0]
    if best["candidate_id"] != "exp007_cont_e050_abs_e150":
        raise ReportError(f"unexpected best eligible candidate {best['candidate_id']}")
    _validate_close(best["overall"]["dice"], 0.3460234857111323, "best single Dice")
    if best["overall"]["hits"] != 296:
        raise ReportError(f"unexpected best single hits {best['overall']['hits']}")
    _validate_close(oracle["dice"], 0.3602780325749142, "oracle Dice")
    if oracle["hits"] != 298:
        raise ReportError(f"unexpected oracle hits {oracle['hits']}")

    dataset_sha256 = sha256_file(dataset_path)
    return {
        "schema_version": 1,
        "analysis": "sideexp002_all_fixed_val200_methods_cutoff030_category_oracle",
        "definitions": {
            "overall_dice_cutoff": OVERALL_DICE_CUTOFF,
            "cutoff_inclusive": True,
            "mask_threshold": MASK_THRESHOLD,
            "hit_dice_threshold": HIT_DICE_THRESHOLD,
            "dice_aggregation": "unweighted mean across findings",
            "global_ranking": "overall Dice, then total hits, then candidate ID",
            "category_ranking": "category Dice, then category hits, then overall Dice, then candidate ID",
            "oracle": "one eligible model selected per known validation category",
        },
        "sources": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "audit_path": str(audit_path),
            "audit_sha256": audit_sha256,
        },
        "validation": {
            "cases": EXPECTED_CASES,
            "findings": EXPECTED_FINDINGS,
            "audit_rows": len(all_ranked),
            "eligible_rows": len(eligible),
            "excluded_rows": len(excluded),
            "category_counts": category_counts,
        },
        "category_labels": CATEGORY_LABELS,
        "eligible_methods": [_strip_internal(candidate) for candidate in eligible],
        "excluded_methods": [_strip_internal(candidate) for candidate in excluded],
        "category_winners": winners,
        "oracle": oracle,
    }


def run(
    dataset_path: Path = DEFAULT_DATASET,
    audit_path: Path = DEFAULT_AUDIT,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_markdown: Path = DEFAULT_OUTPUT_MARKDOWN,
) -> dict[str, Any]:
    summary = build_summary(dataset_path, audit_path)
    atomic_write_json(output_json, summary)
    markdown = build_markdown(
        dataset_path=dataset_path,
        audit_path=audit_path,
        dataset_sha256=summary["sources"]["dataset_sha256"],
        audit_sha256=summary["sources"]["audit_sha256"],
        category_counts=summary["validation"]["category_counts"],
        all_ranked=[candidate for candidate in summary["eligible_methods"] + summary["excluded_methods"]],
        eligible=summary["eligible_methods"],
        excluded=summary["excluded_methods"],
        winners=summary["category_winners"],
        oracle=summary["oracle"],
    )
    atomic_write(output_markdown, markdown)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    args = parser.parse_args()
    run(
        resolve_path(args.dataset_json),
        resolve_path(args.audit_json),
        args.output_json,
        args.output_markdown,
    )
    print(f"wrote {args.output_markdown}")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
