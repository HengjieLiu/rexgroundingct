#!/usr/bin/env python3
"""Render provenance-aware val200 collaborator leaderboards for SideExp004.

The collaborator share bundle intentionally contains normalized metrics rather
than the original evaluator/config/checkpoint artifacts.  This script therefore
does not try to enroll the records in SideExp003's canonical checkpoint catalog.
It validates the supplied 24 selected result variants against the fixed val200
manifest, then renders a local, explicitly source-declared comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SOURCE_BUNDLE_RELATIVE = Path(
    "best_checkpoint_ensemble_share_20260907/outputs/"
    "best_checkpoint_ensemble_share_20260907"
)
VAL200_MANIFEST_RELATIVE = Path(
    "configs/evaluation/rexgroundingct_val200_seed20260723.json"
)
SIDEEXP003_CATALOG_RELATIVE = Path(
    "side_experiments/sideexp003_ensemble_method_hub/checkpoint_catalog.json"
)

OFFICIAL_CATEGORIES = (
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
)
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
HIT_DICE_THRESHOLD = 0.1
FLOAT_TOLERANCE = 1e-12

CATALOG_FILENAME = "collaborator_selected24_catalog.json"
CHECKPOINT_LEADERBOARD_FILENAME = "collaborator_val200_checkpoint_leaderboard.md"
SUBCATEGORY_LEADERBOARD_FILENAME = "collaborator_val200_subcategory_leaderboard.md"
COMPARISON_FILENAME = "my_vs_collaborator_val200.md"
OUTPUT_FILENAMES = (
    CATALOG_FILENAME,
    CHECKPOINT_LEADERBOARD_FILENAME,
    SUBCATEGORY_LEADERBOARD_FILENAME,
    COMPARISON_FILENAME,
)


class ValidationError(ValueError):
    """Raised when an input bundle cannot support the stated report contract."""


RecordT = TypeVar("RecordT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise ValidationError(f"Missing required input: {path}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid JSON in {path}: {error}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_close(actual: Any, expected: Any, message: str) -> None:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{message}: expected numeric values, got {actual!r} and {expected!r}"
        ) from error
    if not math.isfinite(actual_float) or not math.isfinite(expected_float):
        raise ValidationError(f"{message}: values must be finite")
    if not math.isclose(actual_float, expected_float, abs_tol=FLOAT_TOLERANCE, rel_tol=0.0):
        raise ValidationError(f"{message}: {actual_float:.17g} != {expected_float:.17g}")


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value: Any) -> str:
    return f"`{markdown_text(value)}`"


def format_dice(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def format_delta(value: float) -> str:
    return f"{value:+.6f}"


def format_hits(hits: int, findings: int) -> str:
    return f"{hits}/{findings}"


def source_type(record: Mapping[str, Any]) -> str:
    return "raw-like" if record["raw_like"] else "pipeline"


def checkpoint_status(record: Mapping[str, Any]) -> str:
    return "declared, not locally verifiable" if record["checkpoint_path"] else "not declared"


def rank_records(
    records: Iterable[RecordT],
    dice: Callable[[RecordT], float],
    hits: Callable[[RecordT], int],
    identifier: Callable[[RecordT], str],
) -> list[RecordT]:
    """Sort high Dice, then high hits, then ascending stable ID."""

    return sorted(
        records,
        key=lambda record: (-float(dice(record)), -int(hits(record)), str(identifier(record))),
    )


def expected_map_from_manifest(manifest_path: Path) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    manifest = read_json(manifest_path)
    cases = manifest.get("test")
    require(isinstance(cases, list), f"{manifest_path}: expected a test-case list")
    require(len(cases) == EXPECTED_CASES, f"{manifest_path}: expected {EXPECTED_CASES} cases")

    expected: dict[tuple[str, int], str] = {}
    names: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), f"{manifest_path}: each test case must be an object")
        name = case.get("name")
        categories = case.get("categories")
        findings = case.get("findings")
        require(isinstance(name, str) and name, f"{manifest_path}: case name is missing")
        require(name not in names, f"{manifest_path}: duplicate case name {name!r}")
        names.add(name)
        require(isinstance(categories, dict), f"{manifest_path}: {name}: categories are missing")
        require(isinstance(findings, dict), f"{manifest_path}: {name}: findings are missing")
        require(
            set(categories) == set(findings),
            f"{manifest_path}: {name}: finding/category keys disagree",
        )
        for raw_index, category in categories.items():
            try:
                finding_index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    f"{manifest_path}: {name}: invalid finding index {raw_index!r}"
                ) from error
            require(
                category in OFFICIAL_CATEGORIES,
                f"{manifest_path}: {name}: unknown category {category!r}",
            )
            key = (name, finding_index)
            require(key not in expected, f"{manifest_path}: duplicate finding key {key!r}")
            expected[key] = category

    require(
        len(expected) == EXPECTED_FINDINGS,
        f"{manifest_path}: expected {EXPECTED_FINDINGS} findings, found {len(expected)}",
    )
    category_counts = {category: 0 for category in OFFICIAL_CATEGORIES}
    for category in expected.values():
        category_counts[category] += 1
    return expected, category_counts


def _source_file_metadata(path: Path) -> dict[str, str]:
    return {"path": relative_to_repo(path), "sha256": sha256_file(path)}


def _finite_float(value: Any, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{context}: expected a number, got {value!r}") from error
    if not math.isfinite(numeric):
        raise ValidationError(f"{context}: expected a finite number, got {value!r}")
    return numeric


def _int_value(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{context}: expected an integer, got a boolean")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{context}: expected an integer, got {value!r}") from error
    if numeric != value:
        raise ValidationError(f"{context}: expected an integer, got {value!r}")
    return numeric


def _validate_selected_result_summary(
    selected: Mapping[str, Any], result: Mapping[str, Any], model_id: str
) -> None:
    require(
        selected == result,
        f"{model_id}: selected_models entry and result.json are not identical",
    )
    for field in ("model_id", "method_family", "result_dir", "checkpoint", "raw_like", "result_hash"):
        require(
            selected.get(field) == result.get(field),
            f"{model_id}: selected_models and result.json disagree on {field}",
        )
    for field in ("mean_dice", "hit_rate", "dice_2a", "dice_2b", "dice_2c", "dice_2d"):
        require_close(
            selected.get(field),
            result.get(field),
            f"{model_id}: selected_models and result.json disagree on {field}",
        )
    require(
        _int_value(selected.get("hits"), f"{model_id}: selected hits")
        == _int_value(result.get("hits"), f"{model_id}: result hits"),
        f"{model_id}: selected_models and result.json disagree on hits",
    )


def _recompose_collaborator_record(
    selected: Mapping[str, Any],
    result: Mapping[str, Any],
    per_category: Sequence[Mapping[str, Any]],
    per_finding: Sequence[Mapping[str, Any]],
    expected_map: Mapping[tuple[str, int], str],
    category_counts: Mapping[str, int],
    source_files: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    model_id = selected.get("model_id")
    require(isinstance(model_id, str) and model_id, "selected model is missing model_id")
    _validate_selected_result_summary(selected, result, model_id)
    require(isinstance(selected.get("raw_like"), bool), f"{model_id}: raw_like must be boolean")
    require(isinstance(per_finding, Sequence), f"{model_id}: per_finding must be an array")
    require(len(per_finding) == EXPECTED_FINDINGS, f"{model_id}: expected {EXPECTED_FINDINGS} per-finding rows")

    rows_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    grouped_dice: dict[str, list[float]] = defaultdict(list)
    grouped_hits: dict[str, int] = defaultdict(int)
    dice_values: list[float] = []
    total_hits = 0
    for row_number, row in enumerate(per_finding, start=1):
        require(isinstance(row, Mapping), f"{model_id}: row {row_number} is not an object")
        file_name = row.get("file")
        require(isinstance(file_name, str) and file_name, f"{model_id}: row {row_number}: missing file")
        finding_index = _int_value(row.get("finding_idx"), f"{model_id}: row {row_number}: finding_idx")
        key = (file_name, finding_index)
        require(key not in rows_by_key, f"{model_id}: duplicate per-finding key {key!r}")
        require(key in expected_map, f"{model_id}: unexpected per-finding key {key!r}")
        expected_category = expected_map[key]
        require(
            row.get("category") == expected_category,
            f"{model_id}: {key!r}: category {row.get('category')!r} != {expected_category!r}",
        )
        dice = _finite_float(row.get("dice"), f"{model_id}: {key!r}: dice")
        hit = row.get("hit")
        require(isinstance(hit, bool), f"{model_id}: {key!r}: hit must be boolean")
        require(
            hit == (dice >= HIT_DICE_THRESHOLD),
            f"{model_id}: {key!r}: hit must equal dice >= {HIT_DICE_THRESHOLD}",
        )
        rows_by_key[key] = row
        dice_values.append(dice)
        grouped_dice[expected_category].append(dice)
        grouped_hits[expected_category] += int(hit)
        total_hits += int(hit)

    require(
        set(rows_by_key) == set(expected_map),
        f"{model_id}: per-finding keys do not exactly match the fixed val200 manifest",
    )
    mean_dice = math.fsum(dice_values) / len(dice_values)
    require_close(mean_dice, result.get("mean_dice"), f"{model_id}: recomposed overall Dice")
    require(
        total_hits == _int_value(result.get("hits"), f"{model_id}: result hits"),
        f"{model_id}: recomposed hits {total_hits} != result hits {result.get('hits')}",
    )
    hit_rate = total_hits / EXPECTED_FINDINGS
    require_close(hit_rate, result.get("hit_rate"), f"{model_id}: recomposed overall hit rate")

    expected_available_categories = {
        category for category, findings in category_counts.items() if findings > 0
    }
    require(isinstance(per_category, Sequence), f"{model_id}: per_category must be an array")
    category_rows: dict[str, Mapping[str, Any]] = {}
    for row in per_category:
        require(isinstance(row, Mapping), f"{model_id}: per-category entry is not an object")
        category = row.get("category")
        require(isinstance(category, str), f"{model_id}: per-category entry is missing category")
        require(category not in category_rows, f"{model_id}: duplicate per-category row {category!r}")
        category_rows[category] = row
    require(
        set(category_rows) == expected_available_categories,
        f"{model_id}: per-category categories do not match fixed val200 availability",
    )

    categories: dict[str, dict[str, Any]] = {}
    for category in OFFICIAL_CATEGORIES:
        findings = category_counts[category]
        if findings == 0:
            categories[category] = {
                "available": False,
                "dice": None,
                "findings": 0,
                "hits": 0,
                "hit_rate": None,
            }
            continue
        category_dice = math.fsum(grouped_dice[category]) / findings
        category_hits = grouped_hits[category]
        category_hit_rate = category_hits / findings
        source_row = category_rows[category]
        require(
            _int_value(source_row.get("n"), f"{model_id}: {category}: source n") == findings,
            f"{model_id}: {category}: source category count disagrees with manifest",
        )
        require_close(
            category_dice,
            source_row.get("mean_dice"),
            f"{model_id}: {category}: recomposed category Dice",
        )
        require(
            category_hits == _int_value(source_row.get("hits"), f"{model_id}: {category}: source hits"),
            f"{model_id}: {category}: recomposed category hits disagree",
        )
        require_close(
            category_hit_rate,
            source_row.get("hit_rate"),
            f"{model_id}: {category}: recomposed category hit rate",
        )
        if "misses" in source_row:
            require(
                _int_value(source_row["misses"], f"{model_id}: {category}: source misses")
                == findings - category_hits,
                f"{model_id}: {category}: source misses disagree with recomposed hits",
            )
        result_field = f"dice_{category}"
        if result_field in result:
            require_close(
                category_dice,
                result[result_field],
                f"{model_id}: {category}: result.json category Dice",
            )
        categories[category] = {
            "available": True,
            "dice": category_dice,
            "findings": findings,
            "hits": category_hits,
            "hit_rate": category_hit_rate,
        }

    weighted_category_dice = math.fsum(
        categories[category]["dice"] * categories[category]["findings"]
        for category in OFFICIAL_CATEGORIES
        if categories[category]["available"]
    ) / EXPECTED_FINDINGS
    category_hits_total = sum(categories[category]["hits"] for category in OFFICIAL_CATEGORIES)
    require_close(
        weighted_category_dice,
        mean_dice,
        f"{model_id}: weighted category Dice does not recover overall Dice",
    )
    require(
        category_hits_total == total_hits,
        f"{model_id}: category hits do not recover overall hits",
    )

    raw_checkpoint_path = result.get("checkpoint")
    require(
        raw_checkpoint_path is None or isinstance(raw_checkpoint_path, str),
        f"{model_id}: checkpoint must be a string or null",
    )
    checkpoint_path = raw_checkpoint_path or ""
    return {
        "model_id": model_id,
        "method_family": result["method_family"],
        "result_dir": result["result_dir"],
        "result_hash": result["result_hash"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_path_status": (
            "declared_not_locally_verifiable" if checkpoint_path else "not_declared"
        ),
        "raw_like": result["raw_like"],
        "evaluation_type": (
            "source_declared_raw_like"
            if result["raw_like"]
            else "source_declared_pipeline"
        ),
        "provenance_status": "bundle_only_no_threshold_or_checkpoint_sha256",
        "metrics": {
            "overall": {
                "dice": mean_dice,
                "findings": EXPECTED_FINDINGS,
                "hits": total_hits,
                "hit_rate": hit_rate,
            },
            "categories": categories,
        },
        "source_files": source_files,
    }


def load_collaborator_records(
    source_bundle: Path,
    expected_map: Mapping[tuple[str, int], str],
    category_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_path = source_bundle / "selected_models.json"
    selected_models = read_json(selected_path)
    require(isinstance(selected_models, list), f"{selected_path}: expected an array")
    require(len(selected_models) == 24, f"{selected_path}: expected 24 selected models")

    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for selected in selected_models:
        require(isinstance(selected, Mapping), f"{selected_path}: each selection must be an object")
        model_id = selected.get("model_id")
        require(isinstance(model_id, str) and model_id, f"{selected_path}: missing model_id")
        require(model_id not in selected_by_id, f"{selected_path}: duplicate model_id {model_id!r}")
        selected_by_id[model_id] = selected

    models_dir = source_bundle / "models"
    require(models_dir.is_dir(), f"Missing models directory: {models_dir}")
    directory_ids = {path.name for path in models_dir.iterdir() if path.is_dir()}
    require(
        directory_ids == set(selected_by_id),
        "selected_models.json and models/ directories must have identical model IDs",
    )

    records: list[dict[str, Any]] = []
    for model_id in sorted(selected_by_id):
        model_dir = models_dir / model_id
        result_path = model_dir / "result.json"
        per_category_path = model_dir / "per_category.json"
        per_finding_path = model_dir / "per_finding.json"
        result = read_json(result_path)
        per_category = read_json(per_category_path)
        per_finding = read_json(per_finding_path)
        require(isinstance(result, Mapping), f"{result_path}: expected an object")
        source_files = {
            "selected_models": _source_file_metadata(selected_path),
            "result": _source_file_metadata(result_path),
            "per_category": _source_file_metadata(per_category_path),
            "per_finding": _source_file_metadata(per_finding_path),
        }
        records.append(
            _recompose_collaborator_record(
                selected_by_id[model_id],
                result,
                per_category,
                per_finding,
                expected_map,
                category_counts,
                source_files,
            )
        )

    raw_count = sum(record["raw_like"] for record in records)
    pipeline_count = len(records) - raw_count
    require(raw_count == 20, f"Expected 20 raw-like selected models, found {raw_count}")
    require(pipeline_count == 4, f"Expected 4 pipeline selected models, found {pipeline_count}")
    metadata = {
        "selected_models": _source_file_metadata(selected_path),
        "source_report": _source_file_metadata(source_bundle / "report.md"),
        "selected_model_count": len(records),
        "source_declared_raw_like_count": raw_count,
        "source_declared_pipeline_count": pipeline_count,
    }
    report_text = (source_bundle / "report.md").read_text(encoding="utf-8")
    commit_match = re.search(r"Repository commit:\s*`?([0-9a-f]{7,64})`?", report_text)
    metadata["source_reported_repository_commit"] = (
        commit_match.group(1) if commit_match else None
    )
    return records, metadata


def load_my_catalog(
    catalog_path: Path,
    manifest_sha256: str,
    category_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = read_json(catalog_path)
    require(isinstance(catalog, Mapping), f"{catalog_path}: expected an object")
    dataset = catalog.get("dataset")
    candidates = catalog.get("candidates")
    require(isinstance(dataset, Mapping), f"{catalog_path}: missing dataset metadata")
    require(isinstance(candidates, list) and candidates, f"{catalog_path}: missing candidates")
    require(dataset.get("sha256") == manifest_sha256, f"{catalog_path}: val200 SHA does not match")
    require(dataset.get("cases") == EXPECTED_CASES, f"{catalog_path}: val200 case count does not match")
    require(dataset.get("findings") == EXPECTED_FINDINGS, f"{catalog_path}: val200 finding count does not match")
    require(
        tuple(dataset.get("official_category_order", ())) == OFFICIAL_CATEGORIES,
        f"{catalog_path}: official category order does not match",
    )
    require(
        dataset.get("category_findings") == dict(category_counts),
        f"{catalog_path}: category counts do not match",
    )
    for candidate in candidates:
        require(isinstance(candidate, Mapping), f"{catalog_path}: candidate is not an object")
        candidate_id = candidate.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id, f"{catalog_path}: candidate ID is missing")
        metrics = candidate.get("metrics")
        require(isinstance(metrics, Mapping), f"{candidate_id}: metrics are missing")
        overall = metrics.get("overall")
        categories = metrics.get("categories")
        require(isinstance(overall, Mapping), f"{candidate_id}: overall metrics are missing")
        require(isinstance(categories, Mapping), f"{candidate_id}: category metrics are missing")
        require(overall.get("findings") == EXPECTED_FINDINGS, f"{candidate_id}: finding count mismatch")
        _finite_float(overall.get("dice"), f"{candidate_id}: overall Dice")
        _int_value(overall.get("hits"), f"{candidate_id}: overall hits")
        for category in OFFICIAL_CATEGORIES:
            category_metric = categories.get(category)
            require(isinstance(category_metric, Mapping), f"{candidate_id}: {category}: missing metric")
            expected_findings = category_counts[category]
            require(
                category_metric.get("findings") == expected_findings,
                f"{candidate_id}: {category}: finding count mismatch",
            )
            if expected_findings == 0:
                require(
                    category_metric.get("available") is False,
                    f"{candidate_id}: {category}: should be unavailable",
                )
            else:
                require(
                    category_metric.get("available") is True,
                    f"{candidate_id}: {category}: should be available",
                )
                _finite_float(category_metric.get("dice"), f"{candidate_id}: {category}: Dice")
                _int_value(category_metric.get("hits"), f"{candidate_id}: {category}: hits")
    return list(candidates), _source_file_metadata(catalog_path)


def build_state(
    *,
    source_bundle: Path | None = None,
    manifest_path: Path | None = None,
    my_catalog_path: Path | None = None,
) -> dict[str, Any]:
    source_bundle = source_bundle or BASE_DIR / SOURCE_BUNDLE_RELATIVE
    manifest_path = manifest_path or REPO_ROOT / VAL200_MANIFEST_RELATIVE
    my_catalog_path = my_catalog_path or REPO_ROOT / SIDEEXP003_CATALOG_RELATIVE
    require(source_bundle.is_dir(), f"Missing collaborator source bundle: {source_bundle}")

    expected_map, category_counts = expected_map_from_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    collaborator_records, collaborator_source = load_collaborator_records(
        source_bundle, expected_map, category_counts
    )
    my_candidates, my_catalog_source = load_my_catalog(
        my_catalog_path, manifest_sha256, category_counts
    )
    return {
        "dataset": {
            "manifest": _source_file_metadata(manifest_path),
            "cases": EXPECTED_CASES,
            "findings": EXPECTED_FINDINGS,
            "hit_dice_threshold": HIT_DICE_THRESHOLD,
            "official_category_order": list(OFFICIAL_CATEGORIES),
            "category_findings": dict(category_counts),
        },
        "collaborator_source": collaborator_source,
        "my_catalog_source": my_catalog_source,
        "collaborator_records": collaborator_records,
        "my_candidates": my_candidates,
    }


def collaborator_overall_rank(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return rank_records(
        records,
        lambda record: record["metrics"]["overall"]["dice"],
        lambda record: record["metrics"]["overall"]["hits"],
        lambda record: record["model_id"],
    )


def collaborator_category_rank(
    records: Iterable[Mapping[str, Any]], category: str
) -> list[Mapping[str, Any]]:
    return rank_records(
        records,
        lambda record: record["metrics"]["categories"][category]["dice"],
        lambda record: record["metrics"]["categories"][category]["hits"],
        lambda record: record["model_id"],
    )


def my_overall_rank(candidates: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return rank_records(
        candidates,
        lambda candidate: candidate["metrics"]["overall"]["dice"],
        lambda candidate: candidate["metrics"]["overall"]["hits"],
        lambda candidate: candidate["candidate_id"],
    )


def my_category_rank(
    candidates: Iterable[Mapping[str, Any]], category: str
) -> list[Mapping[str, Any]]:
    available = [
        candidate
        for candidate in candidates
        if candidate["metrics"]["categories"][category]["available"]
    ]
    return rank_records(
        available,
        lambda candidate: candidate["metrics"]["categories"][category]["dice"],
        lambda candidate: candidate["metrics"]["categories"][category]["hits"],
        lambda candidate: candidate["candidate_id"],
    )


def catalog_document(state: Mapping[str, Any]) -> dict[str, Any]:
    records = sorted(state["collaborator_records"], key=lambda record: record["model_id"])
    return {
        "schema_version": 1,
        "scope": {
            "cohort": "24 normalized selected collaborator result variants",
            "not_included": (
                "The larger 309-result bundle is excluded because it lacks full official "
                "category/per-finding data outside 2a-2d."
            ),
            "provenance_note": (
                "raw_like is source-declared only; the bundle does not supply threshold "
                "provenance or checkpoint SHA-256 values."
            ),
        },
        "dataset": state["dataset"],
        "sources": {
            "collaborator_bundle": state["collaborator_source"],
            "sideexp003_checkpoint_catalog": state["my_catalog_source"],
        },
        "records": records,
    }


def _append_ranked_result_table(
    lines: list[str], records: Sequence[Mapping[str, Any]]
) -> None:
    lines.extend(
        [
            "| Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance | Result hash |",
            "| ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, record in enumerate(records, start=1):
        overall = record["metrics"]["overall"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(rank),
                    format_dice(overall["dice"]),
                    format_hits(overall["hits"], overall["findings"]),
                    source_type(record),
                    code(record["method_family"]),
                    code(record["model_id"]),
                    checkpoint_status(record),
                    code(record["result_hash"][:12]),
                )
            )
            + " |"
        )


def render_checkpoint_leaderboard(state: Mapping[str, Any]) -> str:
    all_records = collaborator_overall_rank(state["collaborator_records"])
    raw_records = [record for record in all_records if record["raw_like"]]
    source = state["collaborator_source"]
    dataset = state["dataset"]
    lines = [
        "# SideExp004 Collaborator Val200 Result Leaderboard",
        "",
        "Generated deterministically by `build_collaborator_val200_leaderboards.py`; do not edit ranking rows by hand.",
        "",
        f"Fixed validation manifest: {code(dataset['manifest']['path'])} ({code(dataset['manifest']['sha256'])}), {dataset['cases']} cases / {dataset['findings']} findings.",
        f"Cohort: {source['selected_model_count']} normalized selected collaborator result variants ({source['source_declared_raw_like_count']} source-declared raw-like; {source['source_declared_pipeline_count']} pipeline).",
        "",
        "`raw-like` is copied from the supplied bundle; it is not a SideExp003-style eligibility or fixed-threshold provenance claim. The catalog records the source-file hashes and any reported checkpoint path.",
        "",
        "## All normalized collaborator result variants",
        "",
        "This result-level table retains distinct pipeline evaluations of a reported checkpoint so the supplied bundle remains auditable.",
        "",
    ]
    _append_ranked_result_table(lines, all_records)
    lines.extend(
        [
            "",
            "## Source-declared raw-like checkpoint-result subset",
            "",
            "This is the primary cohort for the closest available checkpoint-oriented comparison. It contains no pipeline rows, but still lacks source threshold and checkpoint-SHA provenance.",
            "",
        ]
    )
    _append_ranked_result_table(lines, raw_records)
    lines.extend(
        [
            "",
            "## Provenance boundary",
            "",
            f"The collaborator bundle reports repository commit {code(source['source_reported_repository_commit'] or 'not supplied')}. It was not inserted into SideExp003's canonical catalog.",
            "",
            "The bundle contains a larger 309-result aggregate index, but only these 24 normalized folders provide the full per-finding and all-category evidence required here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _category_cell(metric: Mapping[str, Any]) -> str:
    if not metric["available"]:
        return "—"
    return f"{metric['dice']:.6f} ({metric['hits']}/{metric['findings']})"


def _append_category_ranks(
    lines: list[str],
    records: Sequence[Mapping[str, Any]],
    category_counts: Mapping[str, int],
) -> None:
    lines.extend(
        [
            "| Category | Findings | Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for category in OFFICIAL_CATEGORIES:
        if category_counts[category] == 0:
            continue
        for rank, record in enumerate(collaborator_category_rank(records, category), start=1):
            metric = record["metrics"]["categories"][category]
            lines.append(
                "| "
                + " | ".join(
                    (
                        category,
                        str(category_counts[category]),
                        str(rank),
                        format_dice(metric["dice"]),
                        format_hits(metric["hits"], metric["findings"]),
                        source_type(record),
                        code(record["method_family"]),
                        code(record["model_id"]),
                        checkpoint_status(record),
                    )
                )
                + " |"
            )


def render_subcategory_leaderboard(state: Mapping[str, Any]) -> str:
    all_records = collaborator_overall_rank(state["collaborator_records"])
    raw_records = [record for record in all_records if record["raw_like"]]
    category_counts = state["dataset"]["category_findings"]
    header = [
        "Method family",
        "Model / result",
        "Type",
        "Overall Dice",
        *OFFICIAL_CATEGORIES,
    ]
    lines = [
        "# SideExp004 Collaborator Val200 Official-Category Leaderboard",
        "",
        "Generated deterministically by `build_collaborator_val200_leaderboards.py`; category metrics are recomposed from exact `(file, finding_idx)` alignment to the fixed val200 manifest.",
        "",
        "Each category cell is `Dice (hits/findings)`. `2f` is unavailable because the fixed val200 cohort has zero `2f` findings.",
        "",
        "## Wide table by normalized collaborator result",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for record in all_records:
        overall = record["metrics"]["overall"]
        cells = [
            code(record["method_family"]),
            code(record["model_id"]),
            source_type(record),
            f"{overall['dice']:.6f} ({overall['hits']}/{overall['findings']})",
        ]
        cells.extend(
            _category_cell(record["metrics"]["categories"][category])
            for category in OFFICIAL_CATEGORIES
        )
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Ranked within each official category — all normalized variants",
            "",
            "Pipeline rows are retained and labeled in this inclusive result-level view.",
            "",
        ]
    )
    _append_category_ranks(lines, all_records, category_counts)
    lines.extend(
        [
            "",
            "## Ranked within each official category — source-declared raw-like subset",
            "",
            "This is the primary source-declared checkpoint-result cohort used by the comparison report.",
            "",
        ]
    )
    _append_category_ranks(lines, raw_records, category_counts)
    return "\n".join(lines) + "\n"


def _my_metric(candidate: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    return candidate["metrics"]["overall"] if metric == "overall" else candidate["metrics"]["categories"][metric]


def _collaborator_metric(record: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    return record["metrics"]["overall"] if metric == "overall" else record["metrics"]["categories"][metric]


def _winner(delta: float) -> str:
    if math.isclose(delta, 0.0, abs_tol=FLOAT_TOLERANCE, rel_tol=0.0):
        return "tie"
    return "mine" if delta > 0 else "collaborator"


def _comparison_rows(
    state: Mapping[str, Any], collaborator_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_my = state["my_candidates"]
    category_counts = state["dataset"]["category_findings"]
    metrics = ("overall",) + tuple(
        category for category in OFFICIAL_CATEGORIES if category_counts[category] > 0
    )
    for metric in metrics:
        my_best = (
            my_overall_rank(all_my)[0]
            if metric == "overall"
            else my_category_rank(all_my, metric)[0]
        )
        collaborator_best = (
            collaborator_overall_rank(collaborator_records)[0]
            if metric == "overall"
            else collaborator_category_rank(collaborator_records, metric)[0]
        )
        my_value = _my_metric(my_best, metric)
        collaborator_value = _collaborator_metric(collaborator_best, metric)
        delta = my_value["dice"] - collaborator_value["dice"]
        rows.append(
            {
                "metric": "Overall" if metric == "overall" else metric,
                "findings": my_value["findings"],
                "my": my_best,
                "my_metric": my_value,
                "collaborator": collaborator_best,
                "collaborator_metric": collaborator_value,
                "delta": delta,
                "winner": _winner(delta),
            }
        )
    return rows


def _append_comparison_table(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "| Metric | Findings | My best candidate | My Dice | My hits | Collaborator best result | Collaborator Dice | Collaborator hits | Δ mine − collaborator | Higher Dice |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        my_candidate = row["my"]
        collaborator = row["collaborator"]
        my_metric = row["my_metric"]
        collaborator_metric = row["collaborator_metric"]
        my_label = f"{code(my_candidate['candidate_id'])} ({code(my_candidate['eligibility'])})"
        collaborator_label = code(collaborator["model_id"])
        lines.append(
            "| "
            + " | ".join(
                (
                    row["metric"],
                    str(row["findings"]),
                    my_label,
                    format_dice(my_metric["dice"]),
                    format_hits(my_metric["hits"], my_metric["findings"]),
                    collaborator_label,
                    format_dice(collaborator_metric["dice"]),
                    format_hits(
                        collaborator_metric["hits"], collaborator_metric["findings"]
                    ),
                    format_delta(row["delta"]),
                    row["winner"],
                )
            )
            + " |"
        )


def render_comparison(state: Mapping[str, Any]) -> str:
    all_records = collaborator_overall_rank(state["collaborator_records"])
    raw_records = [record for record in all_records if record["raw_like"]]
    raw_rows = _comparison_rows(state, raw_records)
    inclusive_rows = _comparison_rows(state, all_records)
    my_source = state["my_catalog_source"]
    collaborator_source = state["collaborator_source"]
    dataset = state["dataset"]
    lines = [
        "# SideExp004 My Best vs Collaborator Best on Val200",
        "",
        "This report compares independently selected best Dice values: the winner for a category need not be the same model as the overall-Dice winner.",
        "",
        f"Shared fixed map: {dataset['cases']} cases / {dataset['findings']} findings, manifest SHA {code(dataset['manifest']['sha256'])}.",
        f"My source: {code(my_source['path'])} ({code(my_source['sha256'])}). Collaborator selection source: {code(collaborator_source['selected_models']['path'])} ({code(collaborator_source['selected_models']['sha256'])}).",
        "",
        "## Primary comparison — source-declared raw-like collaborator results",
        "",
        "The primary collaborator cohort has 20 rows marked `raw_like` in the supplied share bundle. This is the closest available checkpoint-oriented comparison, not a strict fixed-threshold/checkpoint-SHA provenance equivalence claim.",
        "",
    ]
    _append_comparison_table(lines, raw_rows)
    lines.extend(
        [
            "",
            "## Pipeline-inclusive reference — all normalized collaborator variants",
            "",
            "This reference includes four supplied pipeline/post-processing rows. It is useful for observing achieved evaluation results, but is not directly comparable to SideExp003 checkpoint-only inference.",
            "",
        ]
    )
    _append_comparison_table(lines, inclusive_rows)
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- `2f` has zero findings in this fixed val200 split and is therefore unavailable.",
            "- My candidates use SideExp003's current catalog ordering (Dice, then hits, then candidate ID); each selected winner's eligibility appears in the table.",
            "- The collaborator bundle's per-finding keys and category labels exactly match the fixed manifest, but it does not provide evaluator configuration, threshold provenance, or checkpoint SHA-256 values.",
            "- Per-category maxima are same-validation selection summaries, not a held-out routed-ensemble estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_documents(state: Mapping[str, Any]) -> dict[str, str]:
    catalog_text = json.dumps(catalog_document(state), indent=2, sort_keys=True) + "\n"
    return {
        CATALOG_FILENAME: catalog_text,
        CHECKPOINT_LEADERBOARD_FILENAME: render_checkpoint_leaderboard(state),
        SUBCATEGORY_LEADERBOARD_FILENAME: render_subcategory_leaderboard(state),
        COMPARISON_FILENAME: render_comparison(state),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(text)
    os.replace(temporary_path, path)


def apply_documents(output_dir: Path, documents: Mapping[str, str]) -> None:
    for filename in OUTPUT_FILENAMES:
        atomic_write(output_dir / filename, documents[filename])


def check_documents(output_dir: Path, documents: Mapping[str, str]) -> list[str]:
    failures: list[str] = []
    for filename in OUTPUT_FILENAMES:
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing generated output: {path}")
        elif path.read_text(encoding="utf-8") != documents[filename]:
            failures.append(f"stale generated output: {path}")
    return failures


def _summary(state: Mapping[str, Any]) -> str:
    all_records = collaborator_overall_rank(state["collaborator_records"])
    raw_records = [record for record in all_records if record["raw_like"]]
    own_best = my_overall_rank(state["my_candidates"])[0]
    return (
        f"validated {len(all_records)} collaborator variants "
        f"({len(raw_records)} source-declared raw-like, {len(all_records) - len(raw_records)} pipeline); "
        f"all-result leader={all_records[0]['model_id']} ({all_records[0]['metrics']['overall']['dice']:.6f}); "
        f"raw-like leader={raw_records[0]['model_id']} ({raw_records[0]['metrics']['overall']['dice']:.6f}); "
        f"my leader={own_best['candidate_id']} ({own_best['metrics']['overall']['dice']:.6f})"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--dry-run", action="store_true", help="validate and render in memory without writing files (default)"
    )
    actions.add_argument("--apply", action="store_true", help="validate and atomically write generated files")
    actions.add_argument("--check", action="store_true", help="validate and require generated files to be current")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state = build_state()
        documents = render_documents(state)
        if args.apply:
            apply_documents(BASE_DIR, documents)
            print(f"Applied SideExp004 collaborator reports: {_summary(state)}")
        elif args.check:
            failures = check_documents(BASE_DIR, documents)
            if failures:
                for failure in failures:
                    print(f"ERROR: {failure}", file=sys.stderr)
                return 1
            print(f"Generated SideExp004 reports are current: {_summary(state)}")
        else:
            print(f"Dry run passed: {_summary(state)}")
        return 0
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
