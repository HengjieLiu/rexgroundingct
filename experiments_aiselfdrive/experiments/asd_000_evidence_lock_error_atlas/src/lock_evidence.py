"""Deterministic, fail-closed helpers for locking local experiment evidence.

The functions in this module contain no experiment-specific scientific
constants.  Callers must provide every expected hash, count, threshold, and
metric through the protocol mapping (and the immutable control-plane report
declaration where needed).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_THRESHOLD_RE = re.compile(
    r"\bthreshold\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\b",
    flags=re.IGNORECASE,
)


class EvidenceLockError(RuntimeError):
    """Raised when evidence is missing, ambiguous, mutable, or inconsistent."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load a YAML protocol and require a mapping at its document root."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - repository dependency
        raise EvidenceLockError("PyYAML is required to load protocol.yaml") from exc

    protocol_path = _require_regular_file(path, label="protocol")
    try:
        with protocol_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceLockError(
            f"failed to load protocol YAML {protocol_path}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise EvidenceLockError(
            f"protocol root must be a mapping, got {type(loaded).__name__}"
        )
    return loaded


def sha256_file(
    path: str | Path,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return a file's SHA-256 digest while reading bounded binary chunks."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    file_path = _require_regular_file(path, label="evidence file")
    digest = hashlib.sha256()
    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    try:
        with file_path.open("rb", buffering=0) as stream:
            while True:
                read_count = stream.readinto(buffer)
                if not read_count:
                    break
                digest.update(view[:read_count])
    except OSError as exc:
        raise EvidenceLockError(f"failed to hash {file_path}: {exc}") from exc
    return digest.hexdigest()


def hash_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> dict[str, Any]:
    """Hash one file, detect mutation during the read, and verify expectations."""

    file_path = _require_regular_file(path, label="evidence file")
    expected_digest = (
        _validate_sha256(expected_sha256, label="expected_sha256")
        if expected_sha256 is not None
        else None
    )
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 0
    ):
        raise ValueError("expected_size_bytes must be a non-negative integer")

    before = _stable_stat(file_path)
    digest = sha256_file(file_path, chunk_size=chunk_size)
    after = _stable_stat(file_path)
    if before != after:
        raise EvidenceLockError(f"evidence changed while hashing: {file_path}")

    size_bytes = after[2]
    if expected_digest is not None and digest != expected_digest:
        raise EvidenceLockError(
            f"SHA-256 mismatch for {file_path}: expected {expected_digest}, got {digest}"
        )
    if expected_size_bytes is not None and size_bytes != expected_size_bytes:
        raise EvidenceLockError(
            f"size mismatch for {file_path}: expected {expected_size_bytes}, "
            f"got {size_bytes}"
        )

    return {
        "path": str(path),
        "resolved_path": str(file_path.resolve()),
        "sha256": digest,
        "size_bytes": size_bytes,
        "stable_read": True,
    }


def iter_files_sorted(root: str | Path) -> Iterator[Path]:
    """Yield regular files below ``root`` in sorted relative-path order.

    Symlinks are rejected rather than followed because a mutable link target
    would make a tree lock ambiguous.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise EvidenceLockError(f"evidence tree does not exist: {root_path}")
    if not root_path.is_dir():
        raise EvidenceLockError(f"evidence tree is not a directory: {root_path}")

    files: list[Path] = []
    try:
        for dir_path, dir_names, file_names in os.walk(
            root_path, topdown=True, followlinks=False
        ):
            current = Path(dir_path)
            for directory_name in dir_names:
                candidate = current / directory_name
                if candidate.is_symlink():
                    raise EvidenceLockError(
                        f"symlinked directory is not lockable: {candidate}"
                    )
            for file_name in file_names:
                candidate = current / file_name
                if candidate.is_symlink():
                    raise EvidenceLockError(f"symlinked file is not lockable: {candidate}")
                if not candidate.is_file():
                    raise EvidenceLockError(
                        f"non-regular entry is not lockable: {candidate}"
                    )
                files.append(candidate)
    except OSError as exc:
        raise EvidenceLockError(f"failed to traverse evidence tree {root_path}: {exc}") from exc

    files.sort(key=lambda item: item.relative_to(root_path).as_posix())
    yield from files


def hash_tree(
    root: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Hash a directory tree and return sorted, relative file records."""

    root_path = Path(root)
    initial_paths = list(iter_files_sorted(root_path))
    initial_relative = [
        item.relative_to(root_path).as_posix() for item in initial_paths
    ]
    records: list[dict[str, Any]] = []
    for file_path, relative_path in zip(initial_paths, initial_relative, strict=True):
        record = hash_file(file_path, chunk_size=chunk_size)
        records.append(
            {
                "path": relative_path,
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )

    final_relative = [
        item.relative_to(root_path).as_posix()
        for item in iter_files_sorted(root_path)
    ]
    if initial_relative != final_relative:
        raise EvidenceLockError(f"evidence tree changed while hashing: {root_path}")
    return records


def hash_tree_manifest(
    root: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> dict[str, Any]:
    """Return file records plus a deterministic aggregate tree digest."""

    records = hash_tree(root, chunk_size=chunk_size)
    canonical_records = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    root_path = Path(root)
    return {
        "path": str(root),
        "resolved_path": str(root_path.resolve()),
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "tree_sha256": hashlib.sha256(canonical_records).hexdigest(),
        "tree_hash_contract": "sha256(canonical-json-file-records-v1)",
        "files": records,
    }


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace ``path`` with deterministic, strict JSON."""

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not strict JSON: {exc}") from exc
    return _atomic_write_text(path, f"{text}\n")


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Atomically write mapping rows as deterministic RFC-style CSV text."""

    materialized = list(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"CSV row {index} must be a mapping")

    if fieldnames is None:
        if not materialized:
            raise ValueError("fieldnames are required when writing zero CSV rows")
        columns = sorted({str(key) for row in materialized for key in row})
    else:
        columns = [str(field) for field in fieldnames]
    if not columns:
        raise ValueError("CSV fieldnames must not be empty")
    if len(columns) != len(set(columns)):
        raise ValueError("CSV fieldnames must be unique")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    try:
        for row in materialized:
            writer.writerow(row)
    except (ValueError, csv.Error) as exc:
        raise ValueError(f"invalid CSV row: {exc}") from exc
    return _atomic_write_text(path, output.getvalue())


def extract_summary_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and internally validate the locked evaluator summary metrics."""

    if not isinstance(summary, Mapping):
        raise EvidenceLockError("evaluation summary must be a mapping")
    metric_source: Mapping[str, Any] = summary
    nested_summary = summary.get("summary")
    if nested_summary is not None:
        if not isinstance(nested_summary, Mapping):
            raise EvidenceLockError("evaluation summary['summary'] must be a mapping")
        metric_source = nested_summary

    dice = _finite_number(
        _required(metric_source, "mean_global_dice_per_finding", "evaluation summary"),
        label="mean_global_dice_per_finding",
    )
    hits = _non_negative_int(
        _required(metric_source, "total_hits", "evaluation summary"),
        label="total_hits",
    )
    findings = _positive_int(
        _required(metric_source, "total_findings", "evaluation summary"),
        label="total_findings",
    )
    cases_value = metric_source.get("total_cases")
    cases = (
        _positive_int(cases_value, label="total_cases")
        if cases_value is not None
        else None
    )
    if hits > findings:
        raise EvidenceLockError(
            f"evaluation summary has total_hits {hits} greater than "
            f"total_findings {findings}"
        )

    misses_value = metric_source.get("total_misses")
    if misses_value is not None:
        misses = _non_negative_int(misses_value, label="total_misses")
        if hits + misses != findings:
            raise EvidenceLockError(
                "evaluation summary counts are inconsistent: "
                f"{hits} hits + {misses} misses != {findings} findings"
            )

    hit_rate_value = metric_source.get("hit_rate")
    if hit_rate_value is not None:
        hit_rate = _finite_number(hit_rate_value, label="hit_rate")
        calculated_hit_rate = hits / findings
        if not math.isclose(
            hit_rate, calculated_hit_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise EvidenceLockError(
                "evaluation summary hit_rate is inconsistent: "
                f"reported {hit_rate}, calculated {calculated_hit_rate}"
            )
    else:
        hit_rate = hits / findings

    params = metric_source.get("params", {})
    if not isinstance(params, Mapping):
        raise EvidenceLockError("evaluation summary params must be a mapping")
    hit_threshold = _first_numeric(
        params,
        ("global_hit_thr", "hit_threshold", "global_hit_threshold"),
        label="hit threshold",
    )
    threshold = _extract_prediction_threshold(summary, metric_source, params)

    return {
        "dice_per_finding": dice,
        "hits": hits,
        "findings": findings,
        "cases": cases,
        "hit_rate": hit_rate,
        "threshold": threshold,
        "hit_threshold": hit_threshold,
    }


def verify_evaluation_summary(
    path: str | Path,
    *,
    expected_dice: float,
    dice_tolerance: float,
    expected_hits: int,
    expected_findings: int,
    expected_threshold: float | None = None,
    expected_hit_threshold: float | None = None,
    expected_cases: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a stored evaluator summary against a fully supplied contract."""

    expected_dice_value = _finite_number(expected_dice, label="expected_dice")
    tolerance = _finite_number(dice_tolerance, label="dice_tolerance")
    if tolerance < 0:
        raise ValueError("dice_tolerance must be non-negative")
    expected_hits_value = _non_negative_int(expected_hits, label="expected_hits")
    expected_findings_value = _positive_int(
        expected_findings, label="expected_findings"
    )
    expected_cases_value = (
        _positive_int(expected_cases, label="expected_cases")
        if expected_cases is not None
        else None
    )

    summary_path = _require_regular_file(path, label="evaluation summary")
    before = hash_file(summary_path, expected_sha256=expected_sha256)
    summary = _load_json_mapping(summary_path, label="evaluation summary")
    after = hash_file(summary_path, expected_sha256=expected_sha256)
    if before["sha256"] != after["sha256"]:
        raise EvidenceLockError(
            f"evaluation summary changed between reads: {summary_path}"
        )
    metrics = extract_summary_metrics(summary)

    dice_error = abs(metrics["dice_per_finding"] - expected_dice_value)
    if dice_error > tolerance:
        raise EvidenceLockError(
            f"Dice mismatch for {summary_path}: expected {expected_dice_value} "
            f"± {tolerance}, got {metrics['dice_per_finding']} "
            f"(absolute error {dice_error})"
        )
    if metrics["hits"] != expected_hits_value:
        raise EvidenceLockError(
            f"hit-count mismatch for {summary_path}: expected "
            f"{expected_hits_value}, got {metrics['hits']}"
        )
    if metrics["findings"] != expected_findings_value:
        raise EvidenceLockError(
            f"finding-count mismatch for {summary_path}: expected "
            f"{expected_findings_value}, got {metrics['findings']}"
        )
    if expected_cases_value is not None and metrics["cases"] != expected_cases_value:
        raise EvidenceLockError(
            f"case-count mismatch for {summary_path}: expected "
            f"{expected_cases_value}, got {metrics['cases']}"
        )

    expected_threshold_value = _verify_optional_threshold(
        actual=metrics["threshold"],
        expected=expected_threshold,
        label="prediction threshold",
        path=summary_path,
    )
    expected_hit_threshold_value = _verify_optional_threshold(
        actual=metrics["hit_threshold"],
        expected=expected_hit_threshold,
        label="hit threshold",
        path=summary_path,
    )
    return {
        **_declared_path_record(before, path),
        "metrics": metrics,
        "expected": {
            "dice_per_finding": expected_dice_value,
            "dice_tolerance_absolute": tolerance,
            "hits": expected_hits_value,
            "findings": expected_findings_value,
            "cases": expected_cases_value,
            "threshold": expected_threshold_value,
            "hit_threshold": expected_hit_threshold_value,
        },
        "dice_absolute_error": dice_error,
        "verified": True,
    }


def recompute_stored_mask_metrics(
    cohort_path: str | Path,
    prediction_root: str | Path,
    ground_truth_root: str | Path,
    *,
    prediction_threshold: float,
    hit_threshold: float,
    expected_cases: int,
    expected_findings: int,
    array_loader: Callable[[Path], Any],
    geometry_verifier: Callable[[Path, Path], Mapping[str, Any]],
    dice_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Recompute finding Dice and hits from stored masks and released GT.

    The stored prediction files are the scientific source for this check.  A
    separately written evaluator summary is deliberately not accepted as a
    substitute.  ``array_loader`` keeps the format-specific NIfTI reader in
    the command runner while allowing this calculation to be tested with
    small NumPy fixtures.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - repository dependency
        raise EvidenceLockError("NumPy is required to recompute mask metrics") from exc

    if not callable(array_loader):
        raise TypeError("array_loader must be callable")
    if not callable(geometry_verifier):
        raise TypeError("geometry_verifier must be callable")
    threshold = _finite_number(prediction_threshold, label="prediction_threshold")
    hit_cutoff = _finite_number(hit_threshold, label="hit_threshold")
    epsilon = _finite_number(dice_epsilon, label="dice_epsilon")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("prediction_threshold must be in [0, 1]")
    if not 0.0 <= hit_cutoff <= 1.0:
        raise ValueError("hit_threshold must be in [0, 1]")
    if epsilon <= 0.0:
        raise ValueError("dice_epsilon must be positive")
    case_limit = _positive_int(expected_cases, label="expected_cases")
    finding_limit = _positive_int(expected_findings, label="expected_findings")

    cohort_file = _require_regular_file(cohort_path, label="metric cohort")
    entries = _find_cohort_entries(_load_json(cohort_file, label="metric cohort"))
    if len(entries) != case_limit:
        raise EvidenceLockError(
            f"metric cohort case-count mismatch: expected {case_limit}, got {len(entries)}"
        )
    prediction_directory = Path(prediction_root)
    ground_truth_directory = Path(ground_truth_root)
    for directory, label in (
        (prediction_directory, "prediction root"),
        (ground_truth_directory, "ground-truth root"),
    ):
        if not directory.is_dir():
            raise EvidenceLockError(f"{label} is not a directory: {directory}")

    case_records: list[dict[str, Any]] = []
    finding_dice: list[float] = []
    hits = 0
    seen_names: set[str] = set()
    for case_index, entry in enumerate(entries):
        case_name = entry.get("name")
        segmentation_name = entry.get("seg_path")
        findings = entry.get("findings")
        if (
            not isinstance(case_name, str)
            or not case_name
            or Path(case_name).name != case_name
        ):
            raise EvidenceLockError(
                f"metric cohort entry {case_index} has an unsafe case name"
            )
        if case_name in seen_names:
            raise EvidenceLockError(f"metric cohort repeats case {case_name!r}")
        seen_names.add(case_name)
        if (
            not isinstance(segmentation_name, str)
            or not segmentation_name
            or Path(segmentation_name).name != segmentation_name
        ):
            raise EvidenceLockError(
                f"metric cohort entry {case_name!r} has an unsafe seg_path"
            )
        if isinstance(findings, Mapping):
            finding_keys = sorted(findings, key=lambda value: int(value))
            if finding_keys != [str(index) for index in range(len(findings))]:
                raise EvidenceLockError(
                    f"metric cohort findings are not contiguous for {case_name}"
                )
            finding_count = len(findings)
        elif isinstance(findings, list):
            finding_count = len(findings)
        else:
            raise EvidenceLockError(
                f"metric cohort entry {case_name!r} has malformed findings"
            )
        if finding_count <= 0:
            raise EvidenceLockError(f"metric cohort case has no findings: {case_name}")

        prediction_path = prediction_directory / case_name
        ground_truth_path = ground_truth_directory / segmentation_name
        prediction_lock = hash_file(prediction_path)
        ground_truth_lock = hash_file(ground_truth_path)
        try:
            geometry = dict(geometry_verifier(prediction_path, ground_truth_path))
        except Exception as exc:  # noqa: BLE001 - normalize verifier failures
            raise EvidenceLockError(
                f"prediction/GT geometry verification failed for {case_name}: {exc}"
            ) from exc
        if geometry.get("verified") is not True:
            raise EvidenceLockError(
                f"prediction/GT geometry was not verified for {case_name}"
            )
        prediction_stat = _stable_stat(prediction_path)
        ground_truth_stat = _stable_stat(ground_truth_path)
        try:
            prediction = np.asarray(array_loader(prediction_path))
            ground_truth = np.asarray(array_loader(ground_truth_path))
        except Exception as exc:  # noqa: BLE001 - normalize loader failures
            raise EvidenceLockError(
                f"failed to load masks for {case_name}: {exc}"
            ) from exc
        if prediction_stat != _stable_stat(prediction_path):
            raise EvidenceLockError(
                f"prediction changed during metric recomputation: {prediction_path}"
            )
        if ground_truth_stat != _stable_stat(ground_truth_path):
            raise EvidenceLockError(
                f"ground truth changed during metric recomputation: {ground_truth_path}"
            )
        if prediction.shape != ground_truth.shape:
            raise EvidenceLockError(
                f"prediction/GT shape mismatch for {case_name}: "
                f"{prediction.shape} != {ground_truth.shape}"
            )
        if prediction.ndim != 4 or prediction.shape[0] != finding_count:
            raise EvidenceLockError(
                f"stored masks for {case_name} must have shape (F,X,Y,Z) with "
                f"F={finding_count}, got {prediction.shape}"
            )
        if not np.issubdtype(prediction.dtype, np.number) or not np.issubdtype(
            ground_truth.dtype, np.number
        ):
            raise EvidenceLockError(f"stored masks are non-numeric for {case_name}")
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(ground_truth)):
            raise EvidenceLockError(f"stored masks contain non-finite values for {case_name}")
        prediction_values = np.unique(prediction)
        if not np.all(np.isin(prediction_values, np.asarray([0, 1]))):
            raise EvidenceLockError(
                f"stored prediction is not a binary threshold mask for {case_name}"
            )

        case_dice: list[float] = []
        for finding_index in range(finding_count):
            predicted = prediction[finding_index] > threshold
            positive = ground_truth[finding_index] > 0
            intersection = int(np.count_nonzero(predicted & positive))
            denominator = int(np.count_nonzero(predicted)) + int(
                np.count_nonzero(positive)
            )
            dice = float((2 * intersection + epsilon) / (denominator + epsilon))
            case_dice.append(dice)
            finding_dice.append(dice)
            hits += int(dice >= hit_cutoff)
        case_records.append(
            {
                "case_index": case_index,
                "case_name": case_name,
                "segmentation_name": segmentation_name,
                "finding_count": finding_count,
                "finding_dice": case_dice,
                "prediction_sha256": prediction_lock["sha256"],
                "prediction_size_bytes": prediction_lock["size_bytes"],
                "ground_truth_sha256": ground_truth_lock["sha256"],
                "ground_truth_size_bytes": ground_truth_lock["size_bytes"],
                "geometry": geometry,
            }
        )
        del prediction, ground_truth

    if len(finding_dice) != finding_limit:
        raise EvidenceLockError(
            f"recomputed finding-count mismatch: expected {finding_limit}, "
            f"got {len(finding_dice)}"
        )
    canonical_records = json.dumps(
        case_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    mean_dice = float(math.fsum(finding_dice) / len(finding_dice))
    return {
        "metric_source": "stored_prediction_masks_plus_released_ground_truth",
        "dice_contract": "(2*intersection+1e-6)/(predicted+positive+1e-6)",
        "prediction_contract": "stored_binary_mask_greater_than_threshold",
        "ground_truth_contract": "union_of_all_positive_instance_labels_gt_zero",
        "geometry_contract": (
            "exact_shape_affine_spacing_qform_sform_and_orientation_equality"
        ),
        "prediction_threshold": threshold,
        "hit_threshold": hit_cutoff,
        "dice_epsilon": epsilon,
        "cases": len(case_records),
        "findings": len(finding_dice),
        "hits": hits,
        "hit_rate": float(hits / len(finding_dice)),
        "dice_per_finding": mean_dice,
        "case_records_sha256": hashlib.sha256(canonical_records).hexdigest(),
        "case_records": case_records,
        "verified": True,
    }


def verify_cohort_json(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_cases: int,
    expected_findings: int,
) -> dict[str, Any]:
    """Verify the hash and aggregate counts of a declared cohort JSON."""

    cohort_path = _require_regular_file(path, label="cohort JSON")
    before = hash_file(cohort_path, expected_sha256=expected_sha256)
    payload = _load_json(cohort_path, label="cohort JSON")
    after = hash_file(cohort_path, expected_sha256=expected_sha256)
    if before["sha256"] != after["sha256"]:
        raise EvidenceLockError(f"cohort JSON changed between reads: {cohort_path}")

    entries = _find_cohort_entries(payload)
    expected_case_count = _positive_int(expected_cases, label="expected_cases")
    expected_finding_count = _non_negative_int(
        expected_findings, label="expected_findings"
    )
    names: list[str] = []
    finding_count = 0
    for index, entry in enumerate(entries):
        name = entry.get("name")
        findings = entry.get("findings")
        if not isinstance(name, str) or not name:
            raise EvidenceLockError(f"cohort entry {index} has no non-empty name")
        if not isinstance(findings, list):
            raise EvidenceLockError(
                f"cohort entry {name!r} has non-list findings"
            )
        names.append(name)
        finding_count += len(findings)
    if len(names) != len(set(names)):
        raise EvidenceLockError("cohort contains duplicate case names")
    if len(entries) != expected_case_count:
        raise EvidenceLockError(
            f"cohort case-count mismatch: expected {expected_case_count}, "
            f"got {len(entries)}"
        )
    if finding_count != expected_finding_count:
        raise EvidenceLockError(
            f"cohort finding-count mismatch: expected {expected_finding_count}, "
            f"got {finding_count}"
        )

    names_payload = json.dumps(
        names,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **_declared_path_record(before, path),
        "cases": len(entries),
        "findings": finding_count,
        "unique_case_names": True,
        "ordered_case_names_sha256": hashlib.sha256(names_payload).hexdigest(),
        "verified": True,
    }


def verify_embedding_bank(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_labels: int,
    expected_shape: Sequence[int],
    expected_dtype: str,
) -> dict[str, Any]:
    """Verify an NPZ embedding bank without assuming archive member names."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - repository dependency
        raise EvidenceLockError("NumPy is required to verify the embedding bank") from exc

    bank_path = _require_regular_file(path, label="embedding bank")
    before = hash_file(bank_path, expected_sha256=expected_sha256)
    label_count = _positive_int(expected_labels, label="expected_labels")
    shape = [
        _positive_int(dimension, label=f"expected_shape[{index}]")
        for index, dimension in enumerate(expected_shape)
    ]
    if not shape:
        raise ValueError("expected_shape must not be empty")
    dtype = str(expected_dtype)

    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in sorted(archive.files)}
    except (OSError, ValueError) as exc:
        raise EvidenceLockError(
            f"failed to read embedding bank {bank_path}: {exc}"
        ) from exc
    embedding_candidates = [
        name
        for name, array in arrays.items()
        if list(array.shape) == shape and str(array.dtype) == dtype
    ]
    if len(embedding_candidates) != 1:
        raise EvidenceLockError(
            "embedding bank must contain exactly one array with shape "
            f"{shape} and dtype {dtype}; candidates={embedding_candidates}"
        )
    embedding_key = embedding_candidates[0]
    label_candidates = [
        name
        for name, array in arrays.items()
        if name != embedding_key
        and array.ndim == 1
        and int(array.shape[0]) == label_count
    ]
    if len(label_candidates) != 1:
        raise EvidenceLockError(
            "embedding bank must contain exactly one one-dimensional label "
            f"array of length {label_count}; candidates={label_candidates}"
        )
    if shape[0] != label_count:
        raise EvidenceLockError(
            f"embedding rows {shape[0]} do not match label count {label_count}"
        )

    after = hash_file(bank_path, expected_sha256=expected_sha256)
    if before["sha256"] != after["sha256"]:
        raise EvidenceLockError(
            f"embedding bank changed between reads: {bank_path}"
        )
    return {
        **_declared_path_record(before, path),
        "archive_members": sorted(arrays),
        "embedding_member": embedding_key,
        "label_member": label_candidates[0],
        "labels": label_count,
        "shape": shape,
        "dtype": dtype,
        "verified": True,
    }


def build_lineage_evidence(
    protocol: Mapping[str, Any],
    repo_root: str | Path = Path("."),
    source_report: Mapping[str, Any] | None = None,
    *,
    mask_array_loader: Callable[[Path], Any] | None = None,
    mask_geometry_verifier: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify all protocol-declared local lineage inputs and return JSON data.

    ``source_report`` is accepted separately because it is immutable
    control-plane evidence declared by the portfolio, not experiment
    configuration.  If the protocol itself declares ``source_report``, the
    explicit mapping may be omitted.  Conflicting declarations are rejected.
    """

    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    root = Path(repo_root).resolve()
    protocol_report = protocol.get("source_report")
    if protocol_report is None and source_report is None:
        raise EvidenceLockError(
            "source_report declaration is required: pass the immutable "
            "control-plane source_report mapping because protocol.yaml does "
            "not declare one"
        )
    if protocol_report is not None and not isinstance(protocol_report, Mapping):
        raise EvidenceLockError("protocol source_report must be a mapping")
    if source_report is not None and not isinstance(source_report, Mapping):
        raise EvidenceLockError("source_report must be a mapping")
    if protocol_report is not None and source_report is not None:
        _require_matching_report_declarations(protocol_report, source_report)
    report_spec = source_report if source_report is not None else protocol_report
    assert report_spec is not None

    experiment_id = _required_string(protocol, "experiment_id", "protocol")
    schema_version = _required_string(protocol, "schema_version", "protocol")
    report_record = _verify_source_report(report_spec, root)

    cohorts = _required_mapping(protocol, "cohorts", "protocol")
    val200_spec = _required_mapping(cohorts, "val200", "protocol.cohorts")
    val200_path = _resolve_declared_path(
        root, _required_string(val200_spec, "path", "protocol.cohorts.val200")
    )
    val200 = verify_cohort_json(
        val200_path,
        expected_sha256=_required_string(
            val200_spec, "sha256", "protocol.cohorts.val200"
        ),
        expected_cases=_required(
            val200_spec, "cases", "protocol.cohorts.val200"
        ),
        expected_findings=_required(
            val200_spec, "findings", "protocol.cohorts.val200"
        ),
    )
    val200["path"] = str(val200_spec["path"])

    baseline_spec = _required_mapping(protocol, "baseline", "protocol")
    checkpoint_path = _resolve_declared_path(
        root, _required_string(baseline_spec, "checkpoint", "protocol.baseline")
    )
    checkpoint = hash_file(
        checkpoint_path,
        expected_sha256=_required_string(
            baseline_spec, "checkpoint_sha256", "protocol.baseline"
        ),
    )
    checkpoint["path"] = str(baseline_spec["checkpoint"])
    checkpoint["verified"] = True

    summary_path = _resolve_declared_path(
        root, _required_string(baseline_spec, "summary", "protocol.baseline")
    )
    summary = verify_evaluation_summary(
        summary_path,
        expected_dice=_required(
            baseline_spec, "expected_dice", "protocol.baseline"
        ),
        dice_tolerance=_required(
            baseline_spec, "dice_tolerance", "protocol.baseline"
        ),
        expected_hits=_required(
            baseline_spec, "expected_hits", "protocol.baseline"
        ),
        expected_findings=_required(
            baseline_spec, "expected_findings", "protocol.baseline"
        ),
        expected_cases=_required(val200_spec, "cases", "protocol.cohorts.val200"),
        expected_threshold=_required(
            baseline_spec, "threshold", "protocol.baseline"
        ),
        expected_hit_threshold=_required(
            baseline_spec, "hit_threshold", "protocol.baseline"
        ),
    )
    summary["path"] = str(baseline_spec["summary"])

    predictions_path = _resolve_declared_path(
        root, _required_string(baseline_spec, "predictions", "protocol.baseline")
    )
    predictions = hash_tree_manifest(predictions_path)
    predictions["path"] = str(baseline_spec["predictions"])
    expected_prediction_files = _positive_int(
        _required(val200_spec, "cases", "protocol.cohorts.val200"),
        label="protocol.cohorts.val200.cases",
    )
    if predictions["file_count"] != expected_prediction_files:
        raise EvidenceLockError(
            "baseline prediction count mismatch: expected "
            f"{expected_prediction_files}, got {predictions['file_count']}"
        )
    predictions["verified"] = True

    if mask_array_loader is None:
        raise EvidenceLockError(
            "mask_array_loader is required; evaluator-summary verification alone "
            "is not a baseline reproduction"
        )
    if mask_geometry_verifier is None:
        raise EvidenceLockError(
            "mask_geometry_verifier is required; array shape alone does not lock "
            "prediction/GT physical geometry"
        )
    data_spec = _required_mapping(protocol, "data", "protocol")
    segmentation_root = _resolve_declared_path(
        root,
        _required_string(data_spec, "segmentation_root", "protocol.data"),
    )
    recomputed = recompute_stored_mask_metrics(
        val200_path,
        predictions_path,
        segmentation_root,
        prediction_threshold=_required(
            baseline_spec, "threshold", "protocol.baseline"
        ),
        hit_threshold=_required(
            baseline_spec, "hit_threshold", "protocol.baseline"
        ),
        expected_cases=_required(val200_spec, "cases", "protocol.cohorts.val200"),
        expected_findings=_required(
            baseline_spec, "expected_findings", "protocol.baseline"
        ),
        array_loader=mask_array_loader,
        geometry_verifier=mask_geometry_verifier,
    )
    expected_dice = _finite_number(
        _required(baseline_spec, "expected_dice", "protocol.baseline"),
        label="protocol.baseline.expected_dice",
    )
    dice_tolerance = _finite_number(
        _required(baseline_spec, "dice_tolerance", "protocol.baseline"),
        label="protocol.baseline.dice_tolerance",
    )
    recomputed_error = abs(recomputed["dice_per_finding"] - expected_dice)
    if recomputed_error > dice_tolerance:
        raise EvidenceLockError(
            "recomputed Dice mismatch from stored predictions and GT: expected "
            f"{expected_dice} ± {dice_tolerance}, got "
            f"{recomputed['dice_per_finding']} (absolute error {recomputed_error})"
        )
    if recomputed["hits"] != int(baseline_spec["expected_hits"]):
        raise EvidenceLockError(
            "recomputed hit-count mismatch from stored predictions and GT: "
            f"expected {baseline_spec['expected_hits']}, got {recomputed['hits']}"
        )
    summary_metrics = summary["metrics"]
    if (
        abs(recomputed["dice_per_finding"] - summary_metrics["dice_per_finding"])
        > dice_tolerance
        or recomputed["hits"] != summary_metrics["hits"]
        or recomputed["findings"] != summary_metrics["findings"]
    ):
        raise EvidenceLockError(
            "stored evaluator summary does not agree with independently "
            "recomputed prediction/GT metrics"
        )
    recomputed["dice_absolute_error"] = recomputed_error
    recomputed["agrees_with_stored_summary"] = True

    embedding_spec = _required_mapping(protocol, "embedding_bank", "protocol")
    embedding_path = _resolve_declared_path(
        root, _required_string(embedding_spec, "path", "protocol.embedding_bank")
    )
    embedding_bank = verify_embedding_bank(
        embedding_path,
        expected_sha256=_required_string(
            embedding_spec, "sha256", "protocol.embedding_bank"
        ),
        expected_labels=_required(
            embedding_spec, "expected_labels", "protocol.embedding_bank"
        ),
        expected_shape=_required(
            embedding_spec, "expected_shape", "protocol.embedding_bank"
        ),
        expected_dtype=_required_string(
            embedding_spec, "expected_dtype", "protocol.embedding_bank"
        ),
    )
    embedding_bank["path"] = str(embedding_spec["path"])

    comparator: dict[str, Any] | None = None
    comparator_spec = protocol.get("s3_comparator")
    if comparator_spec is not None:
        if not isinstance(comparator_spec, Mapping):
            raise EvidenceLockError("protocol.s3_comparator must be a mapping")
        comparator_checkpoint_path = _resolve_declared_path(
            root,
            _required_string(
                comparator_spec, "checkpoint", "protocol.s3_comparator"
            ),
        )
        comparator_checkpoint = hash_file(
            comparator_checkpoint_path,
            expected_sha256=_required_string(
                comparator_spec,
                "checkpoint_sha256",
                "protocol.s3_comparator",
            ),
        )
        comparator_checkpoint["path"] = str(comparator_spec["checkpoint"])
        comparator_checkpoint["verified"] = True

        comparator_summary_path = _resolve_declared_path(
            root,
            _required_string(comparator_spec, "summary", "protocol.s3_comparator"),
        )
        comparator_summary = verify_evaluation_summary(
            comparator_summary_path,
            expected_dice=_required(
                comparator_spec, "expected_dice", "protocol.s3_comparator"
            ),
            dice_tolerance=0.0,
            expected_hits=_required(
                comparator_spec, "expected_hits", "protocol.s3_comparator"
            ),
            expected_findings=_required(
                val200_spec, "findings", "protocol.cohorts.val200"
            ),
            expected_cases=_required(
                val200_spec, "cases", "protocol.cohorts.val200"
            ),
            expected_threshold=_required(
                baseline_spec, "threshold", "protocol.baseline"
            ),
            expected_hit_threshold=_required(
                baseline_spec, "hit_threshold", "protocol.baseline"
            ),
        )
        comparator_summary["path"] = str(comparator_spec["summary"])
        comparator = {
            "checkpoint": comparator_checkpoint,
            "evaluation_summary": comparator_summary,
        }

    canonical_protocol = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    checks = {
        "source_report_hash_matches": True,
        "val200_hash_and_counts_match": True,
        "baseline_checkpoint_hash_matches": True,
        "baseline_prediction_tree_locked": True,
        "baseline_summary_matches_declared_values": True,
        "baseline_metrics_recomputed_from_predictions_and_gt": True,
        "baseline_recomputation_matches_stored_summary": True,
        "embedding_bank_hash_and_shape_match": True,
    }
    if comparator is not None:
        checks["comparator_hashes_and_metrics_match"] = True
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "experiment_id": experiment_id,
        "protocol_canonical_sha256": hashlib.sha256(canonical_protocol).hexdigest(),
        "source_report": report_record,
        "cohorts": {"val200": val200},
        "baseline": {
            "checkpoint": checkpoint,
            "predictions": predictions,
            "evaluation_summary": summary,
            "metric_recomputation": recomputed,
        },
        "embedding_bank": embedding_bank,
        "checks": checks,
        "verified": all(checks.values()),
    }
    if comparator is not None:
        result["s3_comparator"] = comparator
    # Fail here if a caller accidentally supplied a non-JSON-safe object.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _verify_source_report(
    report_spec: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    path_value = _required_string(report_spec, "path", "source_report")
    report_path = _resolve_declared_path(repo_root, path_value)
    record = hash_file(
        report_path,
        expected_sha256=_required_string(report_spec, "sha256", "source_report"),
    )
    record["path"] = path_value
    record["verified"] = True
    availability = report_spec.get("evidence_availability")
    if availability is not None:
        if not isinstance(availability, str) or not availability:
            raise EvidenceLockError(
                "source_report.evidence_availability must be a non-empty string"
            )
        record["evidence_availability"] = availability
    return record


def _require_matching_report_declarations(
    protocol_report: Mapping[str, Any],
    control_report: Mapping[str, Any],
) -> None:
    for key in ("path", "sha256"):
        protocol_value = _required(protocol_report, key, "protocol.source_report")
        control_value = _required(control_report, key, "source_report")
        if protocol_value != control_value:
            raise EvidenceLockError(
                f"conflicting source_report {key}: protocol declares "
                f"{protocol_value!r}, control plane declares {control_value!r}"
            )


def _find_cohort_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        candidates = [payload]
    elif isinstance(payload, Mapping):
        candidates = [
            value
            for value in payload.values()
            if isinstance(value, list)
            and all(
                isinstance(item, Mapping)
                and "name" in item
                and "findings" in item
                for item in value
            )
        ]
    else:
        candidates = []
    if len(candidates) != 1:
        raise EvidenceLockError(
            "cohort JSON must contain exactly one case-entry list; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _extract_prediction_threshold(
    original_summary: Mapping[str, Any],
    metric_source: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float | None:
    candidates: list[float] = []
    for mapping in (original_summary, metric_source, params):
        value = _first_numeric(
            mapping,
            ("threshold", "prediction_threshold", "global_threshold"),
            label="prediction threshold",
        )
        if value is not None:
            candidates.append(value)
    for mapping in (original_summary, metric_source):
        label = mapping.get("label")
        if isinstance(label, str):
            match = _THRESHOLD_RE.search(label)
            if match:
                candidates.append(float(match.group(1)))
    if not candidates:
        return None
    first = candidates[0]
    if any(
        not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-12)
        for value in candidates[1:]
    ):
        raise EvidenceLockError(
            f"evaluation summary has conflicting prediction thresholds: {candidates}"
        )
    return first


def _first_numeric(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> float | None:
    values = [
        _finite_number(mapping[key], label=f"{label} ({key})")
        for key in keys
        if key in mapping and mapping[key] is not None
    ]
    if not values:
        return None
    first = values[0]
    if any(
        not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-12)
        for value in values[1:]
    ):
        raise EvidenceLockError(f"conflicting {label} values: {values}")
    return first


def _verify_optional_threshold(
    *,
    actual: float | None,
    expected: float | None,
    label: str,
    path: Path,
) -> float | None:
    if expected is None:
        return None
    expected_value = _finite_number(expected, label=f"expected {label}")
    if actual is None:
        raise EvidenceLockError(
            f"{label} is not recorded unambiguously in {path}; "
            f"expected {expected_value}"
        )
    if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceLockError(
            f"{label} mismatch for {path}: expected {expected_value}, got {actual}"
        )
    return expected_value


def _atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceLockError(f"failed to load {label} {path}: {exc}") from exc


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, Mapping):
        raise EvidenceLockError(
            f"{label} {path} must contain a mapping, got "
            f"{type(payload).__name__}"
        )
    return payload


def _require_regular_file(path: str | Path, *, label: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise EvidenceLockError(f"{label} does not exist: {file_path}")
    if file_path.is_symlink():
        raise EvidenceLockError(f"{label} must not be a symlink: {file_path}")
    if not file_path.is_file():
        raise EvidenceLockError(f"{label} is not a regular file: {file_path}")
    return file_path


def _stable_stat(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise EvidenceLockError(f"failed to stat evidence {path}: {exc}") from exc
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _resolve_declared_path(repo_root: Path, value: str) -> Path:
    declared = Path(value)
    return declared if declared.is_absolute() else repo_root / declared


def _declared_path_record(
    record: Mapping[str, Any],
    declared_path: str | Path,
) -> dict[str, Any]:
    result = dict(record)
    result["path"] = str(declared_path)
    return result


def _required(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Any:
    if key not in mapping:
        raise EvidenceLockError(f"missing required {context}.{key}")
    return mapping[key]


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise EvidenceLockError(
            f"{context}.{key} must be a mapping, got {type(value).__name__}"
        )
    return value


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value:
        raise EvidenceLockError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvidenceLockError(
            f"{label} must be a 64-character lowercase SHA-256 digest"
        )
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceLockError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise EvidenceLockError(f"{label} must be finite")
    return converted


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceLockError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    converted = _non_negative_int(value, label=label)
    if converted == 0:
        raise EvidenceLockError(f"{label} must be positive")
    return converted
