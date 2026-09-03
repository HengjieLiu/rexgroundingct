#!/usr/bin/env python3
"""Generate the read-only Exp018 official-category-2d nodule audit.

The audit deliberately consumes already-produced evaluator JSON files.  It does
not load checkpoints, run inference, construct predictions, or launch training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "018_voxtell_category2d_nodule_audit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT_ID
DEFAULT_SOURCES_PATH = DEFAULT_OUTPUT_DIR / "audit_sources.json"
REPORT_NAME = "report.md"
SUMMARY_NAME = "nodule_method_audit.json"
HIT_DICE_THRESHOLD = 0.1
PRIMARY_MASK_THRESHOLD = 0.5

# These are lexical prompt-text cues, not anatomical-mask or image-derived
# localization labels. The first rule set gives every finding description one
# primary roll-up label; the second preserves every matching cue as overlapping
# evidence from the same text.
TEXTUAL_PRIMARY_LOCATION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Bilateral / both lungs",
        (
            r"\b(?:both lungs|bilateral(?:ly)?|both lung parenchyma|both upper lobes|both lower lobes)\b",
        ),
    ),
    (
        "Left lower lobe",
        (r"\bleft lower lobe\b", r"\blower lobe of the left lung\b"),
    ),
    (
        "Right lower lobe",
        (
            r"\bright lower lobe\b",
            r"\blower lobe of the right lung\b",
            r"\bsuperior right lower lobe\b",
        ),
    ),
    (
        "Right upper lobe",
        (r"\bright upper lobe\b", r"\bupper lobe of the right lung\b"),
    ),
    ("Right middle lobe", (r"\bright middle lobe\b",)),
    (
        "Left upper lobe",
        (r"\bleft upper lobe\b", r"\bupper lobe of the left lung\b"),
    ),
    ("Right lung, lobe not stated", (r"\bright lung\b",)),
    (
        "Right minor fissure / perifissural",
        (r"\bright minor fissure\b", r"\bperifissural\b"),
    ),
    ("Left lung fissure", (r"\bleft lung fissure\b",)),
    ("Laterobasal, side not stated", (r"\blaterobasal\b",)),
    ("Right apical (bleb wording)", (r"\bright apical\b",)),
)
TEXTUAL_PRIMARY_LOCATION_DEFAULT = "No location stated"

TEXTUAL_LOCATION_CUE_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Bilateral / both lungs",
        "laterality/distribution",
        TEXTUAL_PRIMARY_LOCATION_RULES[0][1],
    ),
    ("Left lower lobe", "lobe", TEXTUAL_PRIMARY_LOCATION_RULES[1][1]),
    ("Right lower lobe", "lobe", TEXTUAL_PRIMARY_LOCATION_RULES[2][1]),
    ("Right upper lobe", "lobe", TEXTUAL_PRIMARY_LOCATION_RULES[3][1]),
    ("Right middle lobe", "lobe", TEXTUAL_PRIMARY_LOCATION_RULES[4][1]),
    ("Left upper lobe", "lobe", TEXTUAL_PRIMARY_LOCATION_RULES[5][1]),
    ("Right-lung wording", "laterality", (r"\bright lung\b",)),
    ("Left-lung wording", "laterality", (r"\bleft lung\b",)),
    (
        "Basal / laterobasal / posterobasal / anterobasal / mediobasal",
        "within-lung qualifier",
        (r"\b(?:basal|laterobasal|posterobasal|anterobasal|mediobasal)\b",),
    ),
    ("Subpleural", "surface/relationship", (r"\bsubpleural\b",)),
    (
        "Lateral / laterobasal",
        "within-lung qualifier",
        (r"\blateral\b", r"\blaterobasal\b"),
    ),
    (
        "Posterior / posterobasal",
        "within-lung qualifier",
        (r"\bposterior\b", r"\bposterobasal\b"),
    ),
    ("Superior", "within-lung qualifier", (r"\bsuperior\b",)),
    (
        "Anterior / anterobasal",
        "within-lung qualifier",
        (r"\banterior\b", r"\banterobasal\b"),
    ),
    (
        "Fissural / perifissural",
        "surface/relationship",
        (r"\bfissure\b", r"\bperifissural\b"),
    ),
    ("Peripheral", "surface/relationship", (r"\bperipheral(?:ly)?\b",)),
    (
        "Diaphragmatic / juxtadiaphragmatic",
        "within-lung qualifier",
        (r"\b(?:diaphragmatic|juxtadiaphragmatic)\b",),
    ),
    ("Lingular", "within-lung qualifier", (r"\blingular\b",)),
    (
        "Medial / mediobasal",
        "within-lung qualifier",
        (r"\bmedial\b", r"\bmediobasal\b"),
    ),
    ("Pleural-based", "surface/relationship", (r"\bpleural[- ]based\b",)),
    ("Apical / apex", "within-lung qualifier", (r"\b(?:apical|apex)\b",)),
    ("Intrapulmonary", "relationship", (r"\bintrapulmonary\b",)),
)


class AuditError(RuntimeError):
    """Raised when the immutable audit contract cannot be reproduced."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AuditError(f"Missing audit source: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"Invalid JSON audit source: {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def display_path(value: str | Path, repo_root: Path = REPO_ROOT) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(path.as_posix())


def validate_source_hash(path: Path, expected_sha256: str, label: str) -> str:
    if not expected_sha256:
        raise AuditError(f"{label} has no expected SHA-256: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise AuditError(
            f"{label} SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return observed_sha256


def _json_value_at_path(data: Any, path: list[str], *, label: str) -> Any:
    """Read a scalar from a manifest-pinned JSON object without guessing paths."""

    value = data
    for component in path:
        if not isinstance(value, dict) or component not in value:
            rendered = ".".join(path)
            raise AuditError(f"{label} does not contain JSON path {rendered!r}")
        value = value[component]
    return value


def load_mask_threshold_evidence(
    manifest: dict[str, Any], *, repo_root: Path
) -> dict[str, dict[str, Any]]:
    """Validate the immutable 0.5 mask-threshold sources used by primary rows."""

    specs = manifest.get("mask_threshold_evidence")
    if not isinstance(specs, dict) or not specs:
        raise AuditError("Audit source manifest is missing mask_threshold_evidence")
    evidence: dict[str, dict[str, Any]] = {}
    for evidence_id, spec in specs.items():
        if not isinstance(evidence_id, str) or not evidence_id or not isinstance(spec, dict):
            raise AuditError("mask_threshold_evidence must map non-empty IDs to objects")
        source_value = spec.get("path")
        expected_sha256 = spec.get("expected_sha256")
        json_path = spec.get("json_path")
        if (
            not isinstance(source_value, str)
            or not isinstance(expected_sha256, str)
            or not isinstance(json_path, list)
            or not json_path
            or not all(isinstance(component, str) and component for component in json_path)
        ):
            raise AuditError(
                f"Mask-threshold evidence {evidence_id} needs path, expected_sha256, and json_path"
            )
        source_path = resolve_path(source_value, repo_root)
        observed_sha256 = validate_source_hash(
            source_path, expected_sha256, f"mask-threshold evidence {evidence_id}"
        )
        observed_value = _json_value_at_path(
            read_json(source_path), json_path, label=f"mask-threshold evidence {evidence_id}"
        )
        if not isinstance(observed_value, (int, float)) or float(observed_value) != PRIMARY_MASK_THRESHOLD:
            raise AuditError(
                f"Mask-threshold evidence {evidence_id} is not "
                f"{PRIMARY_MASK_THRESHOLD:g}: {observed_value!r}"
            )
        evidence[evidence_id] = {
            "evidence_id": evidence_id,
            "path": display_path(source_value, repo_root),
            "sha256": observed_sha256,
            "json_path": ".".join(json_path),
            "value": float(observed_value),
        }
    return evidence


def _mask_threshold_provenance(
    spec: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    *,
    label: str,
    fallback_evidence_id: str | None = None,
) -> dict[str, Any]:
    evidence_id = spec.get("mask_threshold_evidence_id", fallback_evidence_id)
    if not isinstance(evidence_id, str) or evidence_id not in evidence:
        raise AuditError(f"{label} has no valid mask_threshold_evidence_id")
    return dict(evidence[evidence_id])


def _category_items(categories: Any) -> Iterable[tuple[int, str]]:
    if isinstance(categories, dict):
        for index, category in categories.items():
            try:
                yield int(index), str(category)
            except (TypeError, ValueError) as exc:
                raise AuditError(f"Invalid category index {index!r}") from exc
        return
    if isinstance(categories, list):
        for index, category in enumerate(categories):
            yield index, str(category)
        return
    raise AuditError(f"Unsupported category mapping type: {type(categories).__name__}")


def category_positions(dataset: dict[str, Any], category_code: str) -> set[tuple[str, int]]:
    """Return canonical (case filename, finding index) records for one category."""

    test_rows = dataset.get("test")
    if not isinstance(test_rows, list):
        raise AuditError("Dataset JSON must contain a list-valued 'test' field")
    positions: set[tuple[str, int]] = set()
    for case in test_rows:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise AuditError("Dataset test entries must carry a string 'name'")
        for index, category in _category_items(case.get("categories", {})):
            if category != category_code:
                continue
            position = (case["name"], index)
            if position in positions:
                raise AuditError(f"Duplicate category position in dataset: {position}")
            positions.add(position)
    return positions


def category_finding_text_records(
    dataset: dict[str, Any], category_code: str
) -> list[dict[str, Any]]:
    """Return the category's canonical finding-description records.

    This deliberately reads only the dataset's ``findings`` text.  It does not
    inspect ``seg_path`` or load any masks, images, predictions, or metadata.
    """

    test_rows = dataset.get("test")
    if not isinstance(test_rows, list):
        raise AuditError("Dataset JSON must contain a list-valued 'test' field")
    records: list[dict[str, Any]] = []
    positions: set[tuple[str, int]] = set()
    for case in test_rows:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise AuditError("Dataset test entries must carry a string 'name'")
        findings = case.get("findings")
        if not isinstance(findings, dict):
            raise AuditError(f"Dataset test entry has no findings object: {case['name']}")
        for index, category in _category_items(case.get("categories", {})):
            if category != category_code:
                continue
            position = (case["name"], index)
            if position in positions:
                raise AuditError(f"Duplicate category position in dataset: {position}")
            text = findings.get(str(index))
            if not isinstance(text, str) or not text.strip():
                raise AuditError(f"Category finding has no non-empty text: {position}")
            positions.add(position)
            records.append(
                {
                    "case": case["name"],
                    "finding_index": index,
                    "finding_text": text,
                }
            )
    return records


def _matches_text_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def textual_location_audit(
    dataset: dict[str, Any],
    category_code: str,
    *,
    source: str = "fixed_val200_category_2d_findings_free_text",
    text_only_note: str | None = None,
) -> dict[str, Any]:
    """Produce a deterministic text-only location-description summary."""

    records = category_finding_text_records(dataset, category_code)
    primary_counts = {
        label: 0 for label, _patterns in TEXTUAL_PRIMARY_LOCATION_RULES
    }
    primary_counts[TEXTUAL_PRIMARY_LOCATION_DEFAULT] = 0
    cue_counts = {label: 0 for label, _dimension, _patterns in TEXTUAL_LOCATION_CUE_RULES}

    for record in records:
        text = str(record["finding_text"])
        primary_label = TEXTUAL_PRIMARY_LOCATION_DEFAULT
        for label, patterns in TEXTUAL_PRIMARY_LOCATION_RULES:
            if _matches_text_patterns(text, patterns):
                primary_label = label
                break
        primary_counts[primary_label] += 1
        for label, _dimension, patterns in TEXTUAL_LOCATION_CUE_RULES:
            if _matches_text_patterns(text, patterns):
                cue_counts[label] += 1

    total_findings = len(records)
    if total_findings == 0:
        raise AuditError(f"No finding text records found for category {category_code!r}")
    primary_rows = sorted(
        (
            {
                "location_description": label,
                "findings": count,
                "fraction_of_category_findings": count / total_findings,
            }
            for label, count in primary_counts.items()
            if count
        ),
        key=lambda row: (-int(row["findings"]), str(row["location_description"])),
    )
    for rank, row in enumerate(primary_rows, start=1):
        row["rank"] = rank
    cue_rows = sorted(
        (
            {
                "location_description": label,
                "dimension": dimension,
                "findings": cue_counts[label],
                "fraction_of_category_findings": cue_counts[label] / total_findings,
            }
            for label, dimension, _patterns in TEXTUAL_LOCATION_CUE_RULES
            if cue_counts[label]
        ),
        key=lambda row: (-int(row["findings"]), str(row["location_description"])),
    )
    for rank, row in enumerate(cue_rows, start=1):
        row["rank"] = rank

    return {
        "source": source,
        "unit": "one official category-2d finding description",
        "is_text_only": True,
        "text_only_note": text_only_note
        or (
            "This audit uses only the fixed-val200 findings free-text strings. "
            "It does not use actual lung or lobe masks, segmentation masks, CT images, "
            "lesion coordinates, predictions, or image-derived localization; it describes "
            "label wording rather than anatomical ground truth."
        ),
        "category_findings": total_findings,
        "category_cases": len({str(record["case"]) for record in records}),
        "primary_assignment": {
            "description": (
                "One text-derived location label per finding, using the first matching "
                "rule in the documented precedence order. The precedence is a reporting "
                "convention, not an anatomical hierarchy."
            ),
            "rules_in_precedence_order": [
                {
                    "precedence": precedence,
                    "location_description": label,
                    "text_patterns": list(patterns),
                }
                for precedence, (label, patterns) in enumerate(
                    TEXTUAL_PRIMARY_LOCATION_RULES, start=1
                )
            ],
            "default_location_description": TEXTUAL_PRIMARY_LOCATION_DEFAULT,
            "frequency_sorted": primary_rows,
            "findings_sum": sum(int(row["findings"]) for row in primary_rows),
        },
        "all_matching_location_cues": {
            "multi_label": True,
            "description": (
                "Every matching location cue in the same free-text description. Counts are "
                "non-exclusive and may exceed the category total."
            ),
            "rules": [
                {
                    "location_description": label,
                    "dimension": dimension,
                    "text_patterns": list(patterns),
                }
                for label, dimension, patterns in TEXTUAL_LOCATION_CUE_RULES
            ],
            "frequency_sorted": cue_rows,
        },
        "finding_descriptions_without_literal_nodule_wording": sum(
            not re.search(r"\bnodul", str(record["finding_text"]), flags=re.IGNORECASE)
            for record in records
        ),
    }


def _metadata_split_entries(
    metadata: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    entries = metadata.get(split)
    if not isinstance(entries, list) or not entries:
        raise AuditError(f"Raw metadata has no non-empty {split!r} split")
    if not all(isinstance(entry, dict) for entry in entries):
        raise AuditError(f"Raw metadata {split!r} split contains a non-object entry")
    return entries


def _category_entity_records(
    entries: list[dict[str, Any]], category_code: str, *, split: str
) -> list[dict[str, Any]]:
    """Read released entity-count metadata without loading segmentation masks."""

    records: list[dict[str, Any]] = []
    positions: set[tuple[str, int]] = set()
    for entry in entries:
        name = entry.get("name")
        findings = entry.get("findings")
        categories = entry.get("categories")
        entity_counts = entry.get("entity_counts")
        if not isinstance(name, str) or not name:
            raise AuditError(f"Raw metadata {split!r} entry has no string name")
        if not isinstance(findings, dict) or not isinstance(categories, dict):
            raise AuditError(
                f"Raw metadata {split!r} entry {name} lacks finding/category objects"
            )
        if not isinstance(entity_counts, dict):
            raise AuditError(
                f"Raw metadata {split!r} entry {name} lacks entity_counts metadata"
            )
        category_items = list(_category_items(categories))
        category_keys = {str(index) for index, _category in category_items}
        if set(findings) != category_keys or set(entity_counts) != category_keys:
            raise AuditError(
                f"Raw metadata {split!r} entry {name} has misaligned finding/category/entity keys"
            )
        for key, entity_count in entity_counts.items():
            if (
                isinstance(entity_count, bool)
                or not isinstance(entity_count, int)
                or entity_count <= 0
            ):
                raise AuditError(
                    f"Raw metadata {split!r} entry {name} has invalid entity count "
                    f"for finding {key!r}: {entity_count!r}"
                )
        for index, category in category_items:
            if category != category_code:
                continue
            position = (name, index)
            if position in positions:
                raise AuditError(
                    f"Duplicate raw metadata category position in {split!r}: {position}"
                )
            text = findings.get(str(index))
            if not isinstance(text, str) or not text.strip():
                raise AuditError(
                    f"Raw metadata {split!r} category finding has no text: {position}"
                )
            positions.add(position)
            records.append(
                {
                    "case": name,
                    "finding_index": index,
                    "entity_count": int(entity_counts[str(index)]),
                }
            )
    return records


def _entity_count_distribution(
    entity_counts: list[int], *, total_findings: int
) -> list[dict[str, Any]]:
    if not entity_counts or total_findings != len(entity_counts):
        raise AuditError("Entity-count distribution does not have one value per category finding")
    buckets = (
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4+", lambda value: value >= 4),
    )
    rows = []
    for bucket, predicate in buckets:
        findings = sum(predicate(value) for value in entity_counts)
        rows.append(
            {
                "bucket": bucket,
                "findings": findings,
                "fraction_of_category_findings": findings / total_findings,
            }
        )
    if sum(int(row["findings"]) for row in rows) != total_findings:
        raise AuditError("Entity-count distribution buckets do not recompose category findings")
    return rows


def _metadata_split_category_summary(
    entries: list[dict[str, Any]], category_code: str, *, split: str
) -> dict[str, Any]:
    """Summarize one raw split using text and released metadata only."""

    case_names: set[str] = set()
    findings_per_case: dict[str, int] = {}
    protocol_case_counts: Counter[str] = Counter()
    total_findings = 0
    for entry in entries:
        name = entry.get("name")
        findings = entry.get("findings")
        categories = entry.get("categories")
        protocol = entry.get("protocol")
        if not isinstance(name, str) or not name:
            raise AuditError(f"Raw metadata {split!r} entry has no string name")
        if name in case_names:
            raise AuditError(f"Raw metadata {split!r} repeats CT case {name}")
        if not isinstance(findings, dict) or not isinstance(categories, dict):
            raise AuditError(
                f"Raw metadata {split!r} entry {name} lacks finding/category objects"
            )
        if not isinstance(protocol, str) or not protocol:
            raise AuditError(f"Raw metadata {split!r} entry {name} has no annotation protocol")
        category_items = list(_category_items(categories))
        if set(findings) != {str(index) for index, _category in category_items}:
            raise AuditError(
                f"Raw metadata {split!r} entry {name} has misaligned finding/category keys"
            )
        case_names.add(name)
        total_findings += len(findings)
        findings_per_case[name] = sum(
            category == category_code for _index, category in category_items
        )
        protocol_case_counts[protocol] += 1

    entity_records = _category_entity_records(entries, category_code, split=split)
    category_findings = len(entity_records)
    category_cases = sum(count > 0 for count in findings_per_case.values())
    if category_findings != sum(findings_per_case.values()):
        raise AuditError(
            f"Raw metadata {split!r} category finding count disagrees with case summary"
        )
    if category_cases <= 0:
        raise AuditError(f"Raw metadata {split!r} has no category-{category_code} CT cases")
    entities_by_case: Counter[str] = Counter()
    entity_counts: list[int] = []
    for record in entity_records:
        entity_count = int(record["entity_count"])
        entities_by_case[str(record["case"])] += entity_count
        entity_counts.append(entity_count)
    category_entities = sum(entity_counts)
    entity_count_values = Counter(entity_counts)
    greater_than_three_records = sorted(
        (
            {
                "case": str(record["case"]),
                "finding_index": int(record["finding_index"]),
                "entity_count": int(record["entity_count"]),
            }
            for record in entity_records
            if int(record["entity_count"]) > 3
        ),
        key=lambda record: (
            str(record["case"]),
            int(record["finding_index"]),
            int(record["entity_count"]),
        ),
    )
    if any(entities_by_case[name] <= 0 for name, count in findings_per_case.items() if count):
        raise AuditError(f"Raw metadata {split!r} has a 2d finding without an entity count")
    location = textual_location_audit(
        {"test": entries},
        category_code,
        source=f"raw_metadata_{split}_category_{category_code}_findings_free_text",
        text_only_note=(
            f"This audit uses only the raw {split} split findings free-text strings for "
            f"official category {category_code}. It does not use actual lung or lobe masks, "
            "segmentation masks, CT images, lesion coordinates, predictions, or "
            "image-derived localization; it describes label wording rather than "
            "anatomical ground truth."
        ),
    )
    if location["category_findings"] != category_findings:
        raise AuditError(f"Raw metadata {split!r} text audit disagrees with entity records")

    case_count = len(entries)
    distribution = _entity_count_distribution(
        entity_counts, total_findings=category_findings
    )
    return {
        "split": split,
        "cases": case_count,
        "findings": total_findings,
        "category_2d_findings": category_findings,
        "category_2d_cases": category_cases,
        "category_2d_case_rate": category_cases / case_count,
        "category_2d_findings_per_ct_case": category_findings / case_count,
        "category_2d_findings_per_2d_positive_ct_case": category_findings
        / category_cases,
        "category_2d_entities": category_entities,
        "category_2d_entities_per_finding": category_entities / category_findings,
        "category_2d_entities_per_ct_case": category_entities / case_count,
        "category_2d_entities_per_2d_positive_ct_case": category_entities
        / category_cases,
        "entity_count_distribution": distribution,
        "entity_count_equal_three": entity_count_values[3],
        "entity_count_greater_than_three": len(greater_than_three_records),
        "entity_count_greater_than_three_breakdown": [
            {"entity_count": entity_count, "findings": entity_count_values[entity_count]}
            for entity_count in sorted(entity_count_values)
            if entity_count > 3
        ],
        "entity_count_greater_than_three_records": greater_than_three_records,
        "max_entity_count": max(entity_counts),
        "protocol_case_counts": dict(sorted(protocol_case_counts.items())),
        "textual_location_audit": location,
    }


def _validate_expected_split_summary(
    summary: dict[str, Any], expected: dict[str, Any], *, split: str
) -> None:
    expected_to_actual = {
        "expected_cases": "cases",
        "expected_findings": "findings",
        "expected_category_findings": "category_2d_findings",
        "expected_category_cases": "category_2d_cases",
        "expected_category_entities": "category_2d_entities",
        "expected_entity_count_equal_three": "entity_count_equal_three",
        "expected_entity_count_greater_than_three": "entity_count_greater_than_three",
        "expected_max_entity_count": "max_entity_count",
    }
    for expected_key, actual_key in expected_to_actual.items():
        expected_value = expected.get(expected_key)
        if isinstance(expected_value, bool) or not isinstance(expected_value, int):
            raise AuditError(
                f"Train/val comparison {split!r} has no integer {expected_key} contract"
            )
        actual_value = summary[actual_key]
        if actual_value != expected_value:
            raise AuditError(
                f"Train/val comparison {split!r} {actual_key} is {actual_value}, "
                f"expected {expected_value}"
            )


def _location_count_map(location_audit: dict[str, Any], section: str) -> dict[str, int]:
    if section == "primary":
        rows = location_audit["primary_assignment"]["frequency_sorted"]
    elif section == "cues":
        rows = location_audit["all_matching_location_cues"]["frequency_sorted"]
    else:
        raise AuditError(f"Unknown text-location section {section!r}")
    return {str(row["location_description"]): int(row["findings"]) for row in rows}


def _validate_raw_val_alignment(
    raw_val_entries: list[dict[str, Any]],
    reference_dataset: dict[str, Any],
    category_code: str,
) -> dict[str, bool]:
    reference_rows = reference_dataset.get("test")
    if not isinstance(reference_rows, list):
        raise AuditError("Fixed val200 reference dataset has no test records")
    raw_by_name = {str(entry.get("name")): entry for entry in raw_val_entries}
    reference_by_name = {
        str(entry.get("name")): entry
        for entry in reference_rows
        if isinstance(entry, dict)
    }
    if len(raw_by_name) != len(raw_val_entries) or len(reference_by_name) != len(reference_rows):
        raise AuditError("Raw validation or fixed val200 repeats a CT case name")
    if set(raw_by_name) != set(reference_by_name):
        raise AuditError("Raw validation CT names do not match fixed val200")
    for name in sorted(reference_by_name):
        raw_entry = raw_by_name[name]
        reference_entry = reference_by_name[name]
        if (
            raw_entry.get("findings") != reference_entry.get("findings")
            or raw_entry.get("categories") != reference_entry.get("categories")
        ):
            raise AuditError(
                f"Raw validation text/category content drifts from fixed val200: {name}"
            )
    raw_positions = category_positions({"test": raw_val_entries}, category_code)
    reference_positions = category_positions(reference_dataset, category_code)
    if raw_positions != reference_positions:
        raise AuditError("Raw validation category positions do not match fixed val200")
    return {
        "case_names_match_fixed_val200": True,
        "finding_texts_and_categories_match_fixed_val200": True,
        "category_positions_match_fixed_val200": True,
    }


def _location_comparison_rows(
    train_location_audit: dict[str, Any],
    val_location_audit: dict[str, Any],
    *,
    section: str,
) -> list[dict[str, Any]]:
    if section == "primary":
        train_rows = train_location_audit["primary_assignment"]["frequency_sorted"]
        val_rows = val_location_audit["primary_assignment"]["frequency_sorted"]
    elif section == "cues":
        train_rows = train_location_audit["all_matching_location_cues"]["frequency_sorted"]
        val_rows = val_location_audit["all_matching_location_cues"]["frequency_sorted"]
    else:
        raise AuditError(f"Unknown text-location comparison section {section!r}")
    train_by_label = {str(row["location_description"]): row for row in train_rows}
    val_by_label = {str(row["location_description"]): row for row in val_rows}
    if len(train_by_label) != len(train_rows) or len(val_by_label) != len(val_rows):
        raise AuditError("Text-location audit repeats a comparison label")
    rows: list[dict[str, Any]] = []
    for label in sorted(set(train_by_label).union(val_by_label)):
        train_row = train_by_label.get(label)
        val_row = val_by_label.get(label)
        train_findings = int(train_row["findings"]) if train_row else 0
        val_findings = int(val_row["findings"]) if val_row else 0
        train_fraction = (
            float(train_row["fraction_of_category_findings"]) if train_row else 0.0
        )
        val_fraction = (
            float(val_row["fraction_of_category_findings"]) if val_row else 0.0
        )
        row: dict[str, Any] = {
            "location_description": label,
            "train_findings": train_findings,
            "train_fraction_of_category_findings": train_fraction,
            "val_findings": val_findings,
            "val_fraction_of_category_findings": val_fraction,
            "val_minus_train_percentage_points": 100 * (val_fraction - train_fraction),
        }
        if section == "cues":
            dimension = (train_row or val_row).get("dimension")
            if not isinstance(dimension, str):
                raise AuditError(f"Text-location cue {label!r} has no dimension")
            row["dimension"] = dimension
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["train_fraction_of_category_findings"]),
            str(row["location_description"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _population_comparison_rows(
    train: dict[str, Any], val: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = (
        ("CT cases", "cases", "count"),
        ("2d-positive CT cases", "category_2d_cases", "count"),
        ("2d-positive CT-case share", "category_2d_case_rate", "fraction"),
        ("2d finding descriptions", "category_2d_findings", "count"),
        ("2d findings per CT case", "category_2d_findings_per_ct_case", "mean"),
        (
            "2d findings per 2d-positive CT case",
            "category_2d_findings_per_2d_positive_ct_case",
            "mean",
        ),
        ("Released annotated 2d entities", "category_2d_entities", "count"),
        (
            "Released entities per 2d finding",
            "category_2d_entities_per_finding",
            "mean",
        ),
        (
            "Released entities per CT case",
            "category_2d_entities_per_ct_case",
            "mean",
        ),
        (
            "Released entities per 2d-positive CT case",
            "category_2d_entities_per_2d_positive_ct_case",
            "mean",
        ),
    )
    return [
        {
            "metric": label,
            "unit": unit,
            "train_value": train[key],
            "val_value": val[key],
            "val_minus_train": val[key] - train[key],
        }
        for label, key, unit in metrics
    ]


def _entity_distribution_comparison_rows(
    train: dict[str, Any], val: dict[str, Any]
) -> list[dict[str, Any]]:
    train_by_bucket = {row["bucket"]: row for row in train["entity_count_distribution"]}
    val_by_bucket = {row["bucket"]: row for row in val["entity_count_distribution"]}
    buckets = ("1", "2", "3", "4+")
    if set(train_by_bucket) != set(buckets) or set(val_by_bucket) != set(buckets):
        raise AuditError("Entity-count distribution has unexpected buckets")
    return [
        {
            "bucket": bucket,
            "train_findings": int(train_by_bucket[bucket]["findings"]),
            "train_fraction_of_category_findings": float(
                train_by_bucket[bucket]["fraction_of_category_findings"]
            ),
            "val_findings": int(val_by_bucket[bucket]["findings"]),
            "val_fraction_of_category_findings": float(
                val_by_bucket[bucket]["fraction_of_category_findings"]
            ),
            "val_minus_train_percentage_points": 100
            * (
                float(val_by_bucket[bucket]["fraction_of_category_findings"])
                - float(train_by_bucket[bucket]["fraction_of_category_findings"])
            ),
        }
        for bucket in buckets
    ]


def build_training_validation_comparison(
    manifest: dict[str, Any],
    *,
    reference_dataset: dict[str, Any],
    category_code: str,
    reference_location_audit: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the hash-pinned raw-metadata train/validation comparison."""

    spec = manifest.get("training_validation_comparison")
    if not isinstance(spec, dict):
        raise AuditError("Audit source manifest is missing training_validation_comparison")
    if spec.get("category") != category_code:
        raise AuditError("Train/val comparison must use the official category-2d contract")
    metadata_value = spec.get("metadata_path")
    metadata_expected_sha256 = spec.get("metadata_expected_sha256")
    if not isinstance(metadata_value, str) or not isinstance(metadata_expected_sha256, str):
        raise AuditError("Train/val comparison requires metadata_path and metadata_expected_sha256")
    metadata_path = resolve_path(metadata_value, repo_root)
    metadata_sha256 = validate_source_hash(
        metadata_path, metadata_expected_sha256, "train/val raw challenge metadata"
    )
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise AuditError("Train/val raw challenge metadata must be an object")
    train_entries = _metadata_split_entries(metadata, "train")
    val_entries = _metadata_split_entries(metadata, "val")
    test_entries = _metadata_split_entries(metadata, "test")
    expected_test_cases = spec.get("expected_test_cases")
    if isinstance(expected_test_cases, bool) or not isinstance(expected_test_cases, int):
        raise AuditError("Train/val comparison has no integer expected_test_cases contract")
    if len(test_entries) != expected_test_cases:
        raise AuditError(
            f"Train/val comparison test split has {len(test_entries)} CT cases, "
            f"expected {expected_test_cases}"
        )
    if any("entity_counts" in entry for entry in test_entries):
        raise AuditError("Test unexpectedly exposes entity_counts; update Exp018 comparison scope")

    policy_spec = spec.get("annotation_policy_evidence")
    if not isinstance(policy_spec, dict):
        raise AuditError("Train/val comparison lacks annotation_policy_evidence")
    evidence_fields = (
        ("path", "expected_sha256", "annotation-policy evidence"),
        ("raw_archive_path", "raw_archive_expected_sha256", "annotation-policy raw archive"),
        (
            "entity_count_definition_raw_archive_path",
            "entity_count_definition_raw_archive_expected_sha256",
            "entity-count definition raw archive",
        ),
    )
    evidence: dict[str, dict[str, str]] = {}
    for path_key, sha_key, label in evidence_fields:
        source_value = policy_spec.get(path_key)
        expected_sha256 = policy_spec.get(sha_key)
        if not isinstance(source_value, str) or not isinstance(expected_sha256, str):
            raise AuditError(f"Train/val comparison {label} lacks path/SHA-256")
        source_path = resolve_path(source_value, repo_root)
        evidence[path_key] = {
            "path": display_path(source_value, repo_root),
            "sha256": validate_source_hash(source_path, expected_sha256, label),
        }
    policy_text = resolve_path(policy_spec["path"], repo_root).read_text()
    raw_policy_text = resolve_path(policy_spec["raw_archive_path"], repo_root).read_text()
    required_policy_phrases = (
        "Partial-instance (up to 3 instances per finding)",
        "Exhaustive (all instances segmented by radiologists)",
    )
    if not all(phrase in policy_text and phrase in raw_policy_text for phrase in required_policy_phrases):
        raise AuditError("Hash-pinned annotation-policy evidence lacks the split policy text")
    entity_definition_text = resolve_path(
        policy_spec["entity_count_definition_raw_archive_path"], repo_root
    ).read_text()
    if "Number of segmented entities for each finding" not in entity_definition_text:
        raise AuditError("Hash-pinned entity-count evidence lacks its field definition")

    train = _metadata_split_category_summary(train_entries, category_code, split="train")
    val = _metadata_split_category_summary(val_entries, category_code, split="val")
    expected_splits = spec.get("expected_splits")
    if not isinstance(expected_splits, dict):
        raise AuditError("Train/val comparison lacks expected_splits")
    for split, summary in (("train", train), ("val", val)):
        expected = expected_splits.get(split)
        if not isinstance(expected, dict):
            raise AuditError(f"Train/val comparison lacks expected {split!r} split totals")
        _validate_expected_split_summary(summary, expected, split=split)
    alignment = _validate_raw_val_alignment(val_entries, reference_dataset, category_code)
    if (
        _location_count_map(val["textual_location_audit"], "primary")
        != _location_count_map(reference_location_audit, "primary")
        or _location_count_map(val["textual_location_audit"], "cues")
        != _location_count_map(reference_location_audit, "cues")
    ):
        raise AuditError("Raw validation text-location audit does not match fixed val200")
    alignment["text_location_counts_match_fixed_val200"] = True

    return {
        "source": {
            "metadata": {
                "path": display_path(metadata_value, repo_root),
                "sha256": metadata_sha256,
            },
            "annotation_policy_evidence": {
                "rendered_snapshot": evidence["path"],
                "raw_archive": evidence["raw_archive_path"],
                "entity_count_definition_raw_archive": evidence[
                    "entity_count_definition_raw_archive_path"
                ],
            },
        },
        "scope": {
            "category": category_code,
            "unit": "released CT case; no reliable patient identifier is available",
            "location_analysis": (
                "Text only: locations are derived only from findings free text. It does not "
                "use actual lung or lobe masks, segmentation masks, CT images, coordinates, "
                "predictions, or image-derived localization."
            ),
            "entity_analysis": (
                "Released entity_counts metadata only; no segmentation masks are loaded. "
                "It reports segmented entities per finding, not an independently verified "
                "lesion census."
            ),
            "test_exclusion": (
                "Test is cited for its documented exhaustive policy but excluded from these "
                "entity tables because this metadata snapshot has no test entity_counts."
            ),
        },
        "annotation_policy": {
            "documented_training": "Partial-instance (up to 3 instances per finding)",
            "documented_validation_and_test": (
                "Exhaustive (all instances segmented by radiologists)"
            ),
            "interpretation": (
                "The policy concerns instances within an annotated finding; it does not imply "
                "fewer free-text finding descriptions per CT case. An entity_count of 3 is a "
                "censoring-risk indicator, not proof that additional visible entities exist."
            ),
            "metadata_exception_note": (
                "The released train metadata includes values above 3, so the documented "
                "up-to-3 policy is not enforced as a hard metadata invariant."
            ),
        },
        "validation_alignment": alignment,
        "split_summaries": {"train": train, "val": val},
        "population_and_annotation_metrics": _population_comparison_rows(train, val),
        "entity_count_distribution_comparison": _entity_distribution_comparison_rows(
            train, val
        ),
        "primary_text_location_comparison": _location_comparison_rows(
            train["textual_location_audit"],
            val["textual_location_audit"],
            section="primary",
        ),
        "all_matching_text_location_cue_comparison": _location_comparison_rows(
            train["textual_location_audit"],
            val["textual_location_audit"],
            section="cues",
        ),
        "test_cases": len(test_entries),
        "test_entity_counts_available": False,
    }


def _finding_index(key: str) -> int:
    prefix, separator, value = key.rpartition("_")
    if not prefix or separator != "_":
        raise AuditError(f"Evaluator finding key is not index-addressable: {key!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise AuditError(f"Evaluator finding key has non-integer suffix: {key!r}") from exc


def recompose_category_evaluation(
    evaluation: dict[str, Any],
    positions: set[tuple[str, int]],
    *,
    expected_cases: int | None = None,
    expected_findings: int | None = None,
) -> dict[str, Any]:
    """Recompose unweighted Dice and hits from raw evaluator finding records."""

    summary = evaluation.get("summary")
    cases = evaluation.get("cases")
    if not isinstance(summary, dict) or not isinstance(cases, list):
        raise AuditError("Evaluator JSON must contain object 'summary' and list 'cases'")
    params = summary.get("params")
    if not isinstance(params, dict):
        raise AuditError("Evaluator summary is missing params")
    if params.get("global_only") is not True:
        raise AuditError("Evaluator does not use global-only scoring")
    if float(params.get("global_hit_thr", math.nan)) != HIT_DICE_THRESHOLD:
        raise AuditError(
            "Evaluator global hit threshold is not the expected "
            f"{HIT_DICE_THRESHOLD:g}"
        )
    if expected_cases is not None and summary.get("total_cases") != expected_cases:
        raise AuditError(
            f"Evaluator reports {summary.get('total_cases')} cases, expected {expected_cases}"
        )
    if expected_findings is not None and summary.get("total_findings") != expected_findings:
        raise AuditError(
            "Evaluator reports "
            f"{summary.get('total_findings')} findings, expected {expected_findings}"
        )

    observed: dict[tuple[str, int], tuple[float, bool]] = {}
    case_names: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("file"), str):
            raise AuditError("Evaluator case is missing its string file name")
        case_name = case["file"]
        if case_name in case_names:
            raise AuditError(f"Evaluator contains a duplicate case: {case_name}")
        case_names.add(case_name)
        findings = case.get("findings")
        if not isinstance(findings, dict):
            raise AuditError(f"Evaluator case is missing findings: {case_name}")
        for key, result in findings.items():
            if not isinstance(key, str) or not isinstance(result, dict):
                raise AuditError(f"Malformed finding record in case {case_name}")
            position = (case_name, _finding_index(key))
            if position not in positions:
                continue
            if position in observed:
                raise AuditError(f"Evaluator repeats category finding: {position}")
            dice = result.get("global_dice")
            raw_hit = result.get("global_hit")
            if not isinstance(dice, (int, float)) or not math.isfinite(float(dice)):
                raise AuditError(f"Non-finite Dice for category finding: {position}")
            if not isinstance(raw_hit, bool):
                raise AuditError(f"Non-boolean global_hit for category finding: {position}")
            recomposed_hit = float(dice) >= HIT_DICE_THRESHOLD
            if raw_hit != recomposed_hit:
                raise AuditError(
                    f"global_hit disagreement for {position}: raw={raw_hit}, "
                    f"recomposed={recomposed_hit}"
                )
            observed[position] = (float(dice), raw_hit)

    missing = positions.difference(observed)
    if missing:
        preview = ", ".join(str(position) for position in sorted(missing)[:3])
        raise AuditError(
            f"Evaluator is missing {len(missing)} required category findings: {preview}"
        )
    if len(observed) != len(positions):
        raise AuditError(
            f"Evaluator category record count {len(observed)} does not match "
            f"dataset count {len(positions)}"
        )

    dices = [record[0] for record in observed.values()]
    hits = sum(record[1] for record in observed.values())
    return {
        "dice": sum(dices) / len(dices),
        "hits": int(hits),
        "findings": len(dices),
        "hit_rate": hits / len(dices),
        "cases_with_category_findings": len({case for case, _ in observed}),
    }


def sort_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the immutable Dice-first / hits-second / ID-third policy."""

    return sorted(
        rows,
        key=lambda row: (-float(row["dice"]), -int(row["hits"]), str(row["candidate_id"])),
    )


def validate_unique_primary_rows(rows: list[dict[str, Any]]) -> None:
    candidate_ids: set[str] = set()
    source_paths: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        source_path = str(row["evaluation_path"])
        if candidate_id in candidate_ids:
            raise AuditError(f"Duplicate primary candidate ID: {candidate_id}")
        if source_path in source_paths:
            raise AuditError(f"Duplicate primary evaluator source: {source_path}")
        candidate_ids.add(candidate_id)
        source_paths.add(source_path)


def _evaluate_source(
    spec: dict[str, Any],
    *,
    positions: set[tuple[str, int]],
    repo_root: Path,
    expected_cases: int | None,
    expected_findings: int | None,
) -> tuple[dict[str, Any], str, str]:
    source_value = spec.get("evaluation_path", spec.get("path"))
    if not isinstance(source_value, str):
        raise AuditError("Evaluation source is missing evaluation_path/path")
    source_path = resolve_path(source_value, repo_root)
    expected_sha256 = spec.get("expected_sha256")
    if not isinstance(expected_sha256, str):
        raise AuditError(f"Evaluation source has no SHA-256: {source_value}")
    observed_sha256 = validate_source_hash(
        source_path, expected_sha256, str(spec.get("candidate_id", source_value))
    )
    metrics = recompose_category_evaluation(
        read_json(source_path),
        positions,
        expected_cases=expected_cases,
        expected_findings=expected_findings,
    )
    return metrics, display_path(source_value, repo_root), observed_sha256


def _primary_row(
    spec: dict[str, Any],
    *,
    positions: set[tuple[str, int]],
    repo_root: Path,
    source_kind: str,
    mask_threshold_provenance: dict[str, Any],
    catalog_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if spec.get("evaluation_scope") != "fixed_val200":
        raise AuditError(
            f"Primary candidate {spec.get('candidate_id')} is not fixed_val200 evidence"
        )
    if float(spec.get("mask_threshold", math.nan)) != PRIMARY_MASK_THRESHOLD:
        raise AuditError(
            f"Primary candidate {spec.get('candidate_id')} does not declare "
            f"threshold {PRIMARY_MASK_THRESHOLD:g}"
        )
    candidate_id = spec.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise AuditError("Primary candidate is missing a candidate_id")
    if float(mask_threshold_provenance.get("value", math.nan)) != PRIMARY_MASK_THRESHOLD:
        raise AuditError(
            f"Primary candidate {candidate_id} lacks validated threshold "
            f"{PRIMARY_MASK_THRESHOLD:g} provenance"
        )
    metrics, source_path, source_sha256 = _evaluate_source(
        spec,
        positions=positions,
        repo_root=repo_root,
        expected_cases=200,
        expected_findings=381,
    )
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "experiment_id": str(spec["experiment_id"]),
        "method_family": str(spec["method_family"]),
        "model_id": str(spec["model_id"]),
        "phase": str(spec.get("phase", "single run")),
        "relative_epoch": spec.get("relative_epoch"),
        "absolute_epoch": spec.get("absolute_epoch"),
        "method_scope": str(spec.get("method_scope", "broad_single_model")),
        "evaluation_scope": "fixed_val200",
        "mask_threshold": PRIMARY_MASK_THRESHOLD,
        "mask_threshold_provenance": dict(mask_threshold_provenance),
        "hit_dice_threshold": HIT_DICE_THRESHOLD,
        "source_kind": source_kind,
        "evaluation_path": source_path,
        "evaluation_sha256": source_sha256,
        **metrics,
    }
    if catalog_provenance:
        row["catalog_path"] = catalog_provenance["path"]
        row["catalog_sha256"] = catalog_provenance["sha256"]
    return row


def _catalog_primary_rows(
    manifest: dict[str, Any],
    *,
    positions: set[tuple[str, int]],
    repo_root: Path,
    mask_threshold_evidence: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    catalog_spec = manifest.get("curated_baseline")
    if not isinstance(catalog_spec, dict):
        raise AuditError("Audit source manifest is missing curated_baseline")
    catalog_value = catalog_spec.get("path")
    catalog_sha256 = catalog_spec.get("expected_sha256")
    if not isinstance(catalog_value, str) or not isinstance(catalog_sha256, str):
        raise AuditError("curated_baseline requires path and expected_sha256")
    catalog_path = resolve_path(catalog_value, repo_root)
    threshold_provenance = _mask_threshold_provenance(
        catalog_spec,
        mask_threshold_evidence,
        label="curated baseline catalog",
    )
    observed_catalog_sha256 = validate_source_hash(
        catalog_path, catalog_sha256, "curated baseline catalog"
    )
    catalog = read_json(catalog_path)
    definitions = catalog.get("definitions")
    if not isinstance(definitions, dict):
        raise AuditError("Curated baseline catalog is missing definitions")
    if float(definitions.get("mask_threshold", math.nan)) != PRIMARY_MASK_THRESHOLD:
        raise AuditError("Curated baseline catalog does not declare mask threshold 0.5")
    if float(definitions.get("hit_threshold", math.nan)) != HIT_DICE_THRESHOLD:
        raise AuditError("Curated baseline catalog does not declare hit threshold 0.1")
    catalog_models = catalog.get("models")
    if not isinstance(catalog_models, list):
        raise AuditError("Curated baseline catalog is missing its model list")
    expected_rows = catalog_spec.get("expected_model_count")
    if expected_rows is not None and len(catalog_models) != int(expected_rows):
        raise AuditError(
            f"Curated baseline catalog has {len(catalog_models)} models, "
            f"expected {expected_rows}"
        )

    provenance = {
        "path": display_path(catalog_value, repo_root),
        "sha256": observed_catalog_sha256,
    }
    rows: list[dict[str, Any]] = []
    for model in catalog_models:
        if not isinstance(model, dict):
            raise AuditError("Curated baseline catalog contains a non-object model")
        category = model.get("categories", {}).get("2d")
        if not isinstance(category, dict) or category.get("available") is not True:
            raise AuditError(
                f"Curated model {model.get('candidate_id')} lacks available 2d metrics"
            )
        raw_spec = {
            "candidate_id": model.get("candidate_id"),
            "experiment_id": model.get("experiment_name"),
            "method_family": f"Exp{int(model['experiment_number']):03d} {model['model_id']}",
            "model_id": model.get("model_id"),
            "phase": "curated baseline",
            "relative_epoch": model.get("epoch"),
            "absolute_epoch": model.get("epoch"),
            "method_scope": "broad_single_model",
            "evaluation_scope": "fixed_val200",
            "mask_threshold": PRIMARY_MASK_THRESHOLD,
            "evaluation_path": model.get("evaluation_path"),
            "expected_sha256": model.get("evaluation_sha256"),
        }
        row = _primary_row(
            raw_spec,
            positions=positions,
            repo_root=repo_root,
            source_kind="curated_catalog_revalidated_raw_evaluator",
            mask_threshold_provenance=threshold_provenance,
            catalog_provenance=provenance,
        )
        if row["findings"] != int(category.get("findings", -1)):
            raise AuditError(f"Curated model count drift for {row['candidate_id']}")
        if row["hits"] != int(category.get("hits", -1)):
            raise AuditError(f"Curated model hit drift for {row['candidate_id']}")
        if not math.isclose(row["dice"], float(category.get("dice", math.nan)), abs_tol=1e-12):
            raise AuditError(f"Curated model Dice drift for {row['candidate_id']}")
        rows.append(row)
    return rows, provenance


def _evidence_row(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    category_code: str,
    mask_threshold_provenance: dict[str, Any],
) -> dict[str, Any]:
    dataset_value = spec.get("dataset_path")
    dataset_sha256 = spec.get("dataset_expected_sha256")
    if not isinstance(dataset_value, str) or not isinstance(dataset_sha256, str):
        raise AuditError("Fine-tune evidence requires dataset path and SHA-256")
    dataset_path = resolve_path(dataset_value, repo_root)
    observed_dataset_sha256 = validate_source_hash(
        dataset_path, dataset_sha256, "fine-tune evidence dataset"
    )
    positions = category_positions(read_json(dataset_path), category_code)
    expected_findings = int(spec.get("expected_category_findings", len(positions)))
    if len(positions) != expected_findings:
        raise AuditError(
            f"Fine-tune evidence has {len(positions)} category findings, "
            f"expected {expected_findings}"
        )
    metrics, evaluation_path, evaluation_sha256 = _evaluate_source(
        spec,
        positions=positions,
        repo_root=repo_root,
        expected_cases=None,
        expected_findings=None,
    )
    if metrics["findings"] != expected_findings:
        raise AuditError("Fine-tune evidence failed category-count validation")
    return {
        "label": str(spec["label"]),
        "evidence_scope": str(spec["evidence_scope"]),
        "evaluation_path": evaluation_path,
        "evaluation_sha256": evaluation_sha256,
        "dataset_path": display_path(dataset_value, repo_root),
        "dataset_sha256": observed_dataset_sha256,
        "category": category_code,
        "mask_threshold": PRIMARY_MASK_THRESHOLD,
        "mask_threshold_provenance": dict(mask_threshold_provenance),
        **metrics,
    }


def _unranked_runtime_row(
    spec: dict[str, Any], *, positions: set[tuple[str, int]], repo_root: Path
) -> dict[str, Any]:
    """Record a recomposed fixed-val200 evaluator that fails the ranking contract."""

    candidate_id = spec.get("candidate_id")
    reason = spec.get("exclusion_reason")
    if not isinstance(candidate_id, str) or not candidate_id or not isinstance(reason, str) or not reason:
        raise AuditError("Unranked runtime evidence requires candidate_id and exclusion_reason")
    metrics, source_path, source_sha256 = _evaluate_source(
        spec,
        positions=positions,
        repo_root=repo_root,
        expected_cases=200,
        expected_findings=381,
    )
    return {
        "candidate_id": candidate_id,
        "experiment_id": str(spec["experiment_id"]),
        "method_family": str(spec["method_family"]),
        "model_id": str(spec["model_id"]),
        "phase": str(spec.get("phase", "single run")),
        "relative_epoch": spec.get("relative_epoch"),
        "absolute_epoch": spec.get("absolute_epoch"),
        "method_scope": str(spec.get("method_scope", "broad_single_model")),
        "evaluation_scope": "fixed_val200_threshold_unverified",
        "mask_threshold": None,
        "hit_dice_threshold": HIT_DICE_THRESHOLD,
        "source_kind": "runtime_raw_evaluator_threshold_unverified",
        "exclusion_reason": reason,
        "evaluation_path": source_path,
        "evaluation_sha256": source_sha256,
        **metrics,
    }


def scan_runtime_evaluator_paths(
    manifest: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> list[str]:
    """Return the frozen runtime universe that must be ranked or explicitly excluded."""

    scan = manifest.get("runtime_evaluator_scan")
    if not isinstance(scan, dict):
        raise AuditError("Audit source manifest is missing runtime_evaluator_scan")
    root_value = scan.get("root")
    pattern = scan.get("glob")
    expected_count = scan.get("expected_evaluator_count")
    if (
        not isinstance(root_value, str)
        or not isinstance(pattern, str)
        or not pattern
        or not isinstance(expected_count, int)
    ):
        raise AuditError(
            "runtime_evaluator_scan requires root, glob, and expected_evaluator_count"
        )
    root = resolve_path(root_value, repo_root)
    if not root.is_dir():
        raise AuditError(f"Runtime evaluator scan root is unavailable: {root}")
    paths = sorted(str(path) for path in root.glob(pattern) if path.is_file())
    if len(paths) != expected_count:
        raise AuditError(
            f"Runtime evaluator scan found {len(paths)} files, expected {expected_count}"
        )
    if len(set(paths)) != len(paths):
        raise AuditError("Runtime evaluator scan contains duplicate paths")
    return paths


def _format_metric(row: dict[str, Any]) -> str:
    return f"{float(row['dice']):.6f}; {int(row['hits'])}/{int(row['findings'])}"


def _epoch_label(row: dict[str, Any]) -> str:
    relative = row.get("relative_epoch")
    absolute = row.get("absolute_epoch")
    if relative is None and absolute is None:
        return "—"
    if relative == absolute or absolute is None:
        return f"e{relative}"
    if relative is None:
        return f"absolute e{absolute}"
    return f"relative e{relative} / absolute e{absolute}"


def build_audit(
    manifest: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Build the deterministic audit snapshot in memory."""

    reference = manifest.get("reference_dataset")
    if not isinstance(reference, dict):
        raise AuditError("Audit source manifest is missing reference_dataset")
    category_code = reference.get("category")
    if category_code != "2d":
        raise AuditError("Exp018 is scoped strictly to official category '2d'")
    reference_value = reference.get("path")
    reference_sha256 = reference.get("expected_sha256")
    if not isinstance(reference_value, str) or not isinstance(reference_sha256, str):
        raise AuditError("reference_dataset requires path and expected_sha256")
    reference_path = resolve_path(reference_value, repo_root)
    observed_reference_sha256 = validate_source_hash(
        reference_path, reference_sha256, "fixed val200 category map"
    )
    reference_dataset = read_json(reference_path)
    positions = category_positions(reference_dataset, category_code)
    expected_category_findings = int(reference.get("expected_category_findings", 132))
    if len(positions) != expected_category_findings:
        raise AuditError(
            f"Fixed val200 has {len(positions)} category-2d findings, "
            f"expected {expected_category_findings}"
        )
    if len(reference_dataset.get("test", [])) != 200:
        raise AuditError("Fixed val200 category map does not contain 200 test cases")
    total_reference_findings = sum(
        len(case.get("findings", {})) for case in reference_dataset["test"]
    )
    if total_reference_findings != 381:
        raise AuditError(
            f"Fixed val200 has {total_reference_findings} findings, expected 381"
        )
    location_audit = textual_location_audit(reference_dataset, category_code)
    if location_audit["category_findings"] != len(positions):
        raise AuditError(
            "Text-only location audit finding count does not match the official "
            f"category map: {location_audit['category_findings']} != {len(positions)}"
        )
    if location_audit["primary_assignment"]["findings_sum"] != len(positions):
        raise AuditError("Text-only primary location assignment does not cover category 2d")
    training_validation_comparison = build_training_validation_comparison(
        manifest,
        reference_dataset=reference_dataset,
        category_code=category_code,
        reference_location_audit=location_audit,
        repo_root=repo_root,
    )

    mask_threshold_evidence = load_mask_threshold_evidence(
        manifest, repo_root=repo_root
    )
    catalog_rows, catalog_provenance = _catalog_primary_rows(
        manifest,
        positions=positions,
        repo_root=repo_root,
        mask_threshold_evidence=mask_threshold_evidence,
    )
    runtime_specs = manifest.get("runtime_evaluations")
    if not isinstance(runtime_specs, list):
        raise AuditError("Audit source manifest is missing runtime_evaluations")
    runtime_threshold_ids = manifest.get("runtime_mask_threshold_evidence")
    if not isinstance(runtime_threshold_ids, dict):
        raise AuditError("Audit source manifest is missing runtime_mask_threshold_evidence")
    runtime_rows = [
        _primary_row(
            spec,
            positions=positions,
            repo_root=repo_root,
            source_kind="runtime_raw_evaluator",
            mask_threshold_provenance=_mask_threshold_provenance(
                spec,
                mask_threshold_evidence,
                label=f"runtime candidate {spec.get('candidate_id')}",
                fallback_evidence_id=runtime_threshold_ids.get(spec.get("experiment_id")),
            ),
        )
        for spec in runtime_specs
    ]
    project_policy_specs = manifest.get("project_policy_runtime_evaluations", [])
    if not isinstance(project_policy_specs, list):
        raise AuditError(
            "Audit source manifest has a non-list project_policy_runtime_evaluations"
        )
    project_policy_rows = [
        _primary_row(
            spec,
            positions=positions,
            repo_root=repo_root,
            source_kind="runtime_raw_evaluator_project_fixed_threshold",
            mask_threshold_provenance=_mask_threshold_provenance(
                spec,
                mask_threshold_evidence,
                label=f"project-policy runtime candidate {spec.get('candidate_id')}",
            ),
        )
        for spec in project_policy_specs
    ]
    primary_rows = catalog_rows + runtime_rows + project_policy_rows
    validate_unique_primary_rows(primary_rows)
    unranked_specs = manifest.get("unranked_runtime_evaluations")
    if not isinstance(unranked_specs, list):
        raise AuditError("Audit source manifest is missing unranked_runtime_evaluations")
    unranked_rows = [
        _unranked_runtime_row(spec, positions=positions, repo_root=repo_root)
        for spec in unranked_specs
    ]
    all_candidate_ids = [row["candidate_id"] for row in primary_rows + unranked_rows]
    if len(set(all_candidate_ids)) != len(all_candidate_ids):
        raise AuditError("Primary and unranked runtime evidence repeat a candidate ID")
    all_evaluator_paths = [
        str(resolve_path(row["evaluation_path"], repo_root))
        for row in primary_rows + unranked_rows
    ]
    if len(set(all_evaluator_paths)) != len(all_evaluator_paths):
        raise AuditError("Primary and unranked runtime evidence repeat an evaluator source")
    scanned_runtime_paths = scan_runtime_evaluator_paths(manifest, repo_root=repo_root)
    scanned_runtime_path_set = set(scanned_runtime_paths)
    classified_runtime_path_set = set(all_evaluator_paths)
    if scanned_runtime_path_set != classified_runtime_path_set:
        missing = sorted(scanned_runtime_path_set.difference(classified_runtime_path_set))
        unexpected = sorted(classified_runtime_path_set.difference(scanned_runtime_path_set))
        detail = []
        if missing:
            detail.append(f"unclassified={missing[:3]}")
        if unexpected:
            detail.append(f"not_in_runtime_scan={unexpected[:3]}")
        raise AuditError("Runtime evaluator coverage is incomplete: " + "; ".join(detail))
    ranked_rows = sort_leaderboard(primary_rows)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    rows_by_id = {row["candidate_id"]: row for row in ranked_rows}
    inventory_spec = manifest.get("direct_2d_finetunes")
    if not isinstance(inventory_spec, list):
        raise AuditError("Audit source manifest is missing direct_2d_finetunes")
    direct_inventory: list[dict[str, Any]] = []
    direct_arms: set[str] = set()
    for item in inventory_spec:
        if not isinstance(item, dict):
            raise AuditError("Direct 2d fine-tune inventory contains a non-object")
        arm = item.get("arm")
        candidate_id = item.get("fixed_val200_candidate_id")
        if not isinstance(arm, str) or not isinstance(candidate_id, str):
            raise AuditError("Direct 2d inventory requires arm and fixed_val200_candidate_id")
        if candidate_id not in rows_by_id:
            raise AuditError(f"Missing primary row for direct 2d fine-tune: {candidate_id}")
        direct_arms.add(arm)
        target_threshold_provenance = _mask_threshold_provenance(
            item,
            mask_threshold_evidence,
            label=f"Direct 2d fine-tune {arm}",
        )
        target_series_spec = item.get("target_census_series")
        selected_target_label = item.get("selected_target_census_label")
        if not isinstance(target_series_spec, list) or not target_series_spec:
            raise AuditError(f"Direct 2d fine-tune {arm} lacks target-census evidence")
        if not isinstance(selected_target_label, str):
            raise AuditError(
                f"Direct 2d fine-tune {arm} has no selected target-census label"
            )
        target_series = [
            _evidence_row(
                evidence,
                repo_root=repo_root,
                category_code=category_code,
                mask_threshold_provenance=target_threshold_provenance,
            )
            for evidence in target_series_spec
        ]
        evidence_by_label = {evidence["label"]: evidence for evidence in target_series}
        if len(evidence_by_label) != len(target_series):
            raise AuditError(f"Direct 2d fine-tune {arm} repeats a target-census label")
        if selected_target_label not in evidence_by_label:
            raise AuditError(
                f"Direct 2d fine-tune {arm} selected target evidence is absent: "
                f"{selected_target_label}"
            )
        direct_inventory.append(
            {
                "experiment_id": str(item["experiment_id"]),
                "arm": arm,
                "training_contract": str(item["training_contract"]),
                "selection_note": str(item["selection_note"]),
                "target_census_evidence": evidence_by_label[selected_target_label],
                "target_census_series": target_series,
                "fixed_val200": rows_by_id[candidate_id],
            }
        )
    required_arms = {"category_2d_replay50", "category_2d_target100"}
    if not required_arms.issubset(direct_arms):
        missing = ", ".join(sorted(required_arms.difference(direct_arms)))
        raise AuditError(f"Direct 2d fine-tune inventory is incomplete: {missing}")

    broader_spec = manifest.get("broader_nodule_inclusive_finetunes")
    if not isinstance(broader_spec, list):
        raise AuditError("Audit source manifest is missing broader_nodule_inclusive_finetunes")
    broader_inventory: list[dict[str, Any]] = []
    for item in broader_spec:
        if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
            raise AuditError("Broader nodule-inclusive inventory has no candidate_id")
        candidate_id = item["candidate_id"]
        if candidate_id not in rows_by_id:
            raise AuditError(
                f"Missing primary row for broader nodule-inclusive method: {candidate_id}"
            )
        broader_inventory.append(
            {
                "experiment_id": str(item["experiment_id"]),
                "arm": str(item["arm"]),
                "target_categories": list(item["target_categories"]),
                "note": str(item["note"]),
                "fixed_val200": rows_by_id[candidate_id],
            }
        )

    headline_ids = manifest.get("headline_candidate_ids")
    if not isinstance(headline_ids, list) or len(headline_ids) != 2:
        raise AuditError("Audit source manifest needs exactly two headline_candidate_ids")
    headline = []
    for candidate_id in headline_ids:
        if candidate_id not in rows_by_id:
            raise AuditError(f"Headline candidate is absent from primary rows: {candidate_id}")
        headline.append(rows_by_id[candidate_id])

    direct_final = [item["fixed_val200"] for item in direct_inventory]
    best_direct = max(direct_final, key=lambda row: (row["dice"], row["hits"]))
    best_broad = headline[0]
    if best_direct["dice"] >= best_broad["dice"]:
        raise AuditError(
            "The configured conclusion is stale: a direct 2d fine-tune now matches or "
            "exceeds the broad reference"
        )

    provenance: list[dict[str, Any]] = []
    for row in ranked_rows:
        provenance.append(
            {
                "role": "primary_leaderboard",
                "row_id": row["candidate_id"],
                "path": row["evaluation_path"],
                "sha256": row["evaluation_sha256"],
            }
        )
    for row in unranked_rows:
        provenance.append(
            {
                "role": "unranked_runtime_evaluation",
                "row_id": row["candidate_id"],
                "path": row["evaluation_path"],
                "sha256": row["evaluation_sha256"],
            }
        )
    for item in direct_inventory:
        for evidence in item["target_census_series"]:
            provenance.append(
                {
                    "role": "direct_2d_target_census_evidence",
                    "row_id": f"{item['arm']}:{evidence['label']}",
                    "path": evidence["evaluation_path"],
                    "sha256": evidence["evaluation_sha256"],
                }
            )

    return {
        "schema_version": 3,
        "experiment_id": EXPERIMENT_ID,
        "audit_snapshot": str(manifest["audit_snapshot"]),
        "scope": {
            "official_category": category_code,
            "official_category_label": str(reference["category_label"]),
            "definition": "Pulmonary nodules/masses only; category 1e micronodules is excluded.",
            "primary_evaluation_scope": "fixed_val200",
            "primary_mask_threshold": PRIMARY_MASK_THRESHOLD,
            "hit_dice_threshold": HIT_DICE_THRESHOLD,
            "primary_sort": "Dice descending, hits descending, candidate ID ascending",
            "primary_population": "single-model checkpoints only",
        },
        "reference_dataset": {
            "path": display_path(reference_value, repo_root),
            "sha256": observed_reference_sha256,
            "cases": 200,
            "findings": total_reference_findings,
            "category_2d_findings": len(positions),
            "category_2d_cases": len({case for case, _ in positions}),
        },
        "textual_location_audit": location_audit,
        "training_validation_comparison": training_validation_comparison,
        "curated_baseline": {
            **catalog_provenance,
            "models": len(catalog_rows),
            "role": "historical candidate catalog; every included row is revalidated from its raw evaluator JSON",
        },
        "mask_threshold_evidence": [
            mask_threshold_evidence[evidence_id]
            for evidence_id in sorted(mask_threshold_evidence)
        ],
        "runtime_evaluator_coverage": {
            "scan_root": str(manifest["runtime_evaluator_scan"]["root"]),
            "glob": str(manifest["runtime_evaluator_scan"]["glob"]),
            "scanned_evaluators": len(scanned_runtime_paths),
            "ranked_primary_evaluators": len(ranked_rows),
            "unranked_evaluators": len(unranked_rows),
            "all_scanned_paths_classified": True,
        },
        "validation": {
            "category_2d_findings_exact": len(positions) == expected_category_findings,
            "text_only_location_findings_exact": (
                location_audit["category_findings"] == expected_category_findings
                and location_audit["primary_assignment"]["findings_sum"]
                == expected_category_findings
            ),
            "training_validation_source_hashes_validated": True,
            "raw_validation_matches_fixed_val200": all(
                training_validation_comparison["validation_alignment"].values()
            ),
            "training_validation_entity_counts_validated": True,
            "raw_evaluation_rows_recomposed": len(
                {source["path"] for source in provenance}
            ),
            "all_raw_evaluation_rows_recomposed": True,
            "all_source_hashes_validated": True,
            "all_primary_mask_threshold_evidence_validated": True,
            "primary_mask_threshold_evidence_ids": sorted(
                {
                    row["mask_threshold_provenance"]["evidence_id"]
                    for row in ranked_rows
                }
            ),
            "runtime_evaluator_coverage_complete": True,
            "duplicate_primary_candidate_ids_rejected": True,
            "duplicate_primary_evaluator_sources_rejected": True,
            "required_direct_2d_finetunes_present": sorted(required_arms),
        },
        "headline_pareto_references": headline,
        "primary_leaderboard": ranked_rows,
        "unranked_runtime_evaluations": unranked_rows,
        "direct_2d_finetune_inventory": direct_inventory,
        "broader_nodule_inclusive_finetunes": broader_inventory,
        "excluded_or_incomplete_evidence": manifest["excluded_or_incomplete_evidence"],
        "conclusion": {
            "best_broad_reference_candidate_id": best_broad["candidate_id"],
            "best_broad_reference": _format_metric(best_broad),
            "best_direct_2d_finetune_candidate_id": best_direct["candidate_id"],
            "best_direct_2d_finetune": _format_metric(best_direct),
            "statement": (
                "No direct official-category-2d fine-tune has exceeded the broad "
                "Exp007 phase-2 continuation on fixed val200 Dice."
            ),
        },
        "source_provenance": provenance,
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_comparison_value(value: Any, unit: str) -> str:
    if unit == "count":
        return f"{int(value):,}"
    if unit == "fraction":
        return f"{100 * float(value):.1f}%"
    if unit == "mean":
        return f"{float(value):.3f}"
    raise AuditError(f"Unknown comparison display unit {unit!r}")


def _format_comparison_delta(value: Any, unit: str) -> str:
    if unit == "count":
        return f"{int(value):+,}"
    if unit == "fraction":
        return f"{100 * float(value):+.1f} pp"
    if unit == "mean":
        return f"{float(value):+.3f}"
    raise AuditError(f"Unknown comparison delta unit {unit!r}")


def render_markdown(audit: dict[str, Any]) -> str:
    """Render the repo-local, deterministic human audit."""

    reference = audit["reference_dataset"]
    scope = audit["scope"]
    comparison = audit["training_validation_comparison"]
    train_summary = comparison["split_summaries"]["train"]
    val_summary = comparison["split_summaries"]["val"]
    comparison_sources = comparison["source"]
    lines = [
        "# Exp018: Official Category-2d Nodule Method Audit",
        "",
        f"- Status: `audit_complete` (source snapshot `{audit['audit_snapshot']}`)",
        "- Mode: read-only audit; no checkpoint production, inference, training, or GPU launch.",
        f"- Primary population: {scope['primary_population']} evaluated on fixed val200 at mask threshold `{scope['primary_mask_threshold']:.1f}`.",
        "",
        "## Scope and metric contract",
        "",
        f"Official category `2d` is **{scope['official_category_label']}**. It contains exactly `{reference['category_2d_findings']}` findings in `{reference['category_2d_cases']}` of the fixed `{reference['cases']}` val200 cases. `1e` micronodules/tree-in-bud and Exp004's legacy regex-based `nodule` stratum are not this primary population.",
        "",
        f"Dice is the unweighted mean of raw `global_dice` across the `{reference['category_2d_findings']}` official records. Hits recompute `global_dice >= {scope['hit_dice_threshold']:.1f}`. The deterministic primary order is {scope['primary_sort'].lower()}.",
        "",
    ]
    lines.extend(
        [
            "## Train–validation category-2d distribution",
            "",
            "The archived official challenge policy is correct at the **instance** level: training is `Partial-instance (up to 3 instances per finding)`, while validation and test are `Exhaustive (all instances segmented by radiologists)`. The hash-pinned rendered snapshot is `"
            f"{comparison_sources['annotation_policy_evidence']['rendered_snapshot']['path']}`; its raw archive is `"
            f"{comparison_sources['annotation_policy_evidence']['raw_archive']['path']}`.",
            "",
            "That policy does not imply fewer free-text finding descriptions per CT case. The released source has CT-case filenames rather than a reliable patient identifier, so every comparison below is per **CT case**, not per patient.",
            "",
            f"**Text-only location limitation:** {comparison['scope']['location_analysis']}",
            "",
            f"**Entity-count limitation:** {comparison['scope']['entity_analysis']} The field definition is hash-pinned from `"
            f"{comparison_sources['annotation_policy_evidence']['entity_count_definition_raw_archive']['path']}`.",
            "",
            "### Side-by-side population and annotation density",
            "",
            "| Metric | Train | Validation | Validation − train |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["population_and_annotation_metrics"]:
        lines.append(
            "| "
            f"{_markdown_escape(row['metric'])} | "
            f"{_format_comparison_value(row['train_value'], row['unit'])} | "
            f"{_format_comparison_value(row['val_value'], row['unit'])} | "
            f"{_format_comparison_delta(row['val_minus_train'], row['unit'])} |"
        )
    lines.extend(
        [
            "",
            "### Released entity-count distribution",
            "",
            "Each row is a released `entity_counts` value for one official 2d finding. This is metadata, not a new segmentation-mask audit.",
            "",
            "| Segmented entities per finding | Train | Validation | Validation − train |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["entity_count_distribution_comparison"]:
        lines.append(
            "| "
            f"{row['bucket']} | {row['train_findings']}/{train_summary['category_2d_findings']} "
            f"({100 * row['train_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_findings']}/{val_summary['category_2d_findings']} "
            f"({100 * row['val_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_minus_train_percentage_points']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            f"The expected density gap is visible: validation has `{val_summary['category_2d_entities_per_finding']:.3f}` released segmented entities per 2d finding versus `{train_summary['category_2d_entities_per_finding']:.3f}` in training. Do not treat an entity count of `3` as proof of omitted lesions: it is a censoring-risk indicator. The released train metadata also contains `{train_summary['entity_count_greater_than_three']}` 2d findings above three (maximum `{train_summary['max_entity_count']}`), so the documented up-to-three policy is not a hard metadata invariant.",
            "",
            "#### Observed training values above three",
            "",
            "| Released train entity count | Official 2d findings |",
            "| ---: | ---: |",
        ]
    )
    for row in train_summary["entity_count_greater_than_three_breakdown"]:
        lines.append(f"| {row['entity_count']} | {row['findings']} |")
    lines.extend(
        [
            "",
            "The machine-readable comparison block retains all sorted `(CT case, finding index, entity count)` records for these observed training values; they are reported as metadata exceptions, not capped or discarded.",
            "",
            f"{comparison['scope']['test_exclusion']}",
            "",
            "### Text-only location comparison",
            "",
            "The primary table assigns exactly one normalized text label per finding using fixed precedence. Rows are sorted by training share descending, then label; a finding description can cover multiple nodules.",
            "",
            f"| Rank | Normalized text location | Train (n / {train_summary['category_2d_findings']}) | Validation (n / {val_summary['category_2d_findings']}) | Validation − train |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["primary_text_location_comparison"]:
        lines.append(
            "| "
            f"{row['rank']} | {_markdown_escape(row['location_description'])} | "
            f"{row['train_findings']} ({100 * row['train_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_findings']} ({100 * row['val_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_minus_train_percentage_points']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "### Additional matching free-text location cues",
            "",
            "This table preserves every matching cue in a finding description, including a named lobe embedded in a bilateral description. It is multi-label and non-exclusive; counts do not sum to the split totals.",
            "",
            f"| Rank | Textual location cue | Dimension | Train (n / {train_summary['category_2d_findings']}) | Validation (n / {val_summary['category_2d_findings']}) | Validation − train |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["all_matching_text_location_cue_comparison"]:
        lines.append(
            "| "
            f"{row['rank']} | {_markdown_escape(row['location_description'])} | "
            f"{_markdown_escape(row['dimension'])} | "
            f"{row['train_findings']} ({100 * row['train_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_findings']} ({100 * row['val_fraction_of_category_findings']:.1f}%) | "
            f"{row['val_minus_train_percentage_points']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "Category membership, not a `nodul*` text regex, defines this population. The validation-only text audit is retained unchanged in `nodule_method_audit.json`; the comparison block records the same fixed rules and split-level counts.",
            "",
            "## Current Pareto references",
            "",
            "| Role | Candidate | Checkpoint | 2d Dice | Hits | Hit rate |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    roles = ["Dice-first reference", "Hit-rate guardrail"]
    for role, row in zip(roles, audit["headline_pareto_references"], strict=True):
        lines.append(
            "| "
            f"{role} | `{row['candidate_id']}` | {_epoch_label(row)} | "
            f"{row['dice']:.6f} | {row['hits']}/{row['findings']} | {row['hit_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The two rows express the current trade-off: Exp007's phase-2 continuation reaches the highest Dice, while Exp017 phase 1 epoch 75 preserves three more 2d hits.",
            "",
            "## Primary fixed-val200 single-checkpoint leaderboard",
            "",
            f"Every row below is a single model evaluated at the fixed `0.5` mask threshold. The frozen runtime scan found `{audit['runtime_evaluator_coverage']['scanned_evaluators']}` evaluator files; `{audit['runtime_evaluator_coverage']['ranked_primary_evaluators']}` meet the primary contract and `{audit['runtime_evaluator_coverage']['unranked_evaluators']}` are unranked. The legacy 20-model catalog supplies history, but its raw evaluator JSONs are hash-checked and recomposed here; newer runtime results are included as additional rows.",
            "",
            "| Rank | Candidate | Method family | Checkpoint | Scope | 2d Dice | Hits | Hit rate |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in audit["primary_leaderboard"]:
        lines.append(
            "| "
            f"{row['rank']} | `{row['candidate_id']}` | {_markdown_escape(row['method_family'])} | "
            f"{_epoch_label(row)} | `{row['method_scope']}` | {row['dice']:.6f} | "
            f"{row['hits']}/{row['findings']} | {row['hit_rate']:.3f} |"
        )

    if audit["unranked_runtime_evaluations"]:
        lines.extend(
            [
                "",
                "## Fixed-val200 rows not ranked",
                "",
                "These raw evaluator files are hash-pinned and recomposed, but they cannot be placed in the fixed-0.5 ranking without turning forensic reconstruction into a threshold claim.",
                "",
                "| Candidate | Method family | 2d Dice | Hits | Reason |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in audit["unranked_runtime_evaluations"]:
            lines.append(
                "| "
                f"`{row['candidate_id']}` | {_markdown_escape(row['method_family'])} | "
                f"{row['dice']:.6f} | {row['hits']}/{row['findings']} | "
                f"{_markdown_escape(row['exclusion_reason'])} |"
            )

    lines.extend(
        [
            "",
            "## Direct official-2d fine-tune inventory",
            "",
        "These are intentionally separate from the primary ranking's broad-model comparison. Target-census rows cover the official 132 `2d` records but only the 119 affected cases (or a union containing them), so they are partial evidence rather than full-val200 model selection evidence.",
            "",
            "| Method | Training contract | Target-census-only evidence | Full fixed-val200 checkpoint | Readout |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in audit["direct_2d_finetune_inventory"]:
        target_series = "; ".join(
            f"{evidence['label']}: {_format_metric(evidence)}"
            for evidence in item["target_census_series"]
        )
        full = item["fixed_val200"]
        lines.append(
            "| "
            f"`{item['arm']}` | {_markdown_escape(item['training_contract'])} | "
            f"{target_series} | "
            f"{_epoch_label(full)}: {_format_metric(full)} | "
            f"{_markdown_escape(item['selection_note'])} |"
        )

    lines.extend(
        [
            "",
            "## Broader nodule-inclusive fine-tunes",
            "",
            "These methods include `2d` in a broader target set; they are not direct 2d-only methods.",
            "",
            "| Method | Target categories | Fixed val200 2d | Note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in audit["broader_nodule_inclusive_finetunes"]:
        row = item["fixed_val200"]
        lines.append(
            "| "
            f"`{item['arm']}` | {', '.join(item['target_categories'])} | "
            f"{_format_metric(row)} | {_markdown_escape(item['note'])} |"
        )

    lines.extend(["", "## Exclusions and caveats", ""])
    for item in audit["excluded_or_incomplete_evidence"]:
        lines.append(f"- **{item['label']}:** {item['detail']}")

    lines.extend(
        [
            "",
            "## Threshold provenance",
            "",
            "Each primary row carries one of the hash-validated 0.5 sources below in the machine-readable snapshot. Raw evaluator JSON records the global hit threshold, so this additional provenance is required for the segmentation-mask threshold claim.",
            "",
            "| Evidence ID | Source | JSON field | Value | SHA-256 |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for evidence in audit["mask_threshold_evidence"]:
        lines.append(
            "| "
            f"`{evidence['evidence_id']}` | `{evidence['path']}` | "
            f"`{evidence['json_path']}` | {evidence['value']:.1f} | "
            f"`{evidence['sha256']}` |"
        )

    conclusion = audit["conclusion"]
    lines.extend(
        [
            "",
            "## Evidence-based conclusion",
            "",
            f"{conclusion['statement']} The broad reference is `{conclusion['best_broad_reference_candidate_id']}` at `{conclusion['best_broad_reference']}`; the best completed direct 2d fine-tune is `{conclusion['best_direct_2d_finetune_candidate_id']}` at `{conclusion['best_direct_2d_finetune']}`. A training matrix and GPU launch are intentionally outside Exp018.",
            "",
            "## Source provenance",
            "",
            "All evaluator source hashes below were validated before the metrics were recomposed. `primary_leaderboard` rows are raw fixed-val200 evaluator sources; unranked and target-census evidence is listed separately.",
            "",
            "| Role | Row | Evaluator path | SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for source in audit["source_provenance"]:
        lines.append(
            "| "
            f"{source['role']} | `{source['row_id']}` | `{source['path']}` | "
            f"`{source['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "The source catalog itself is `"
            f"{audit['curated_baseline']['path']}` with SHA-256 `"
            f"{audit['curated_baseline']['sha256']}`. The fixed category map is `"
            f"{reference['path']}` with SHA-256 `{reference['sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_json(audit: dict[str, Any]) -> str:
    return json.dumps(audit, indent=2, sort_keys=True) + "\n"


def _write_or_check(path: Path, content: str, *, check: bool) -> bool:
    if check:
        return path.is_file() and path.read_text() == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed Markdown and JSON render exactly without writing",
    )
    args = parser.parse_args()

    sources_path = args.sources if args.sources.is_absolute() else REPO_ROOT / args.sources
    manifest = read_json(sources_path)
    audit = build_audit(manifest)
    outputs = {
        args.output_dir / REPORT_NAME: render_markdown(audit),
        args.output_dir / SUMMARY_NAME: render_json(audit),
    }
    stale = [
        str(path)
        for path, content in outputs.items()
        if not _write_or_check(path, content, check=args.check)
    ]
    if stale:
        raise AuditError(
            "Generated Exp018 audit differs or is absent; rerun without --check: "
            + ", ".join(stale)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        raise SystemExit(f"audit error: {exc}")
