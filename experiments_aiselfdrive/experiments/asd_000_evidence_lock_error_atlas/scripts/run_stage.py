#!/usr/bin/env python3
"""Fail-closed command entry point for the ASD-000 evidence-lock stages.

Scientific values come only from ``protocol.yaml``.  The immutable experiment
plan is read solely as control-plane metadata so the runner can enforce its
allowed outputs, acceptance-check identifiers, and completion contract.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml


EXPERIMENT_ID = "asd_000_evidence_lock_error_atlas"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SELFDRIVE_ROOT = EXPERIMENT_ROOT.parents[1]
REPO_ROOT = SELFDRIVE_ROOT.parent
SRC_ROOT = EXPERIMENT_ROOT / "src"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
SHARED_MANIFEST_ROOT = SELFDRIVE_ROOT / "shared" / "manifests"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cohorts import seal_cohorts  # noqa: E402
from error_atlas import (  # noqa: E402
    build_error_atlas,
    deterministic_case_cluster_bootstrap,
)
from lock_evidence import (  # noqa: E402
    EvidenceLockError,
    atomic_write_csv,
    atomic_write_json,
    build_lineage_evidence,
    hash_file,
    load_protocol,
    sha256_file,
)
from prompt_ontology import (  # noqa: E402
    left_right_axis_from_affine,
    parse_prompt,
    validate_ras_affine,
    write_ontology_manifest,
)


PHASE_TO_STAGE = {
    "lineage": "lock_local_lineage",
    "cohorts": "lock_cohorts_and_prompt_ontology",
    "logits": "audit_or_export_val80_logits",
    "atlas": "build_val80_error_atlas",
    "closeout": "closeout_evidence_lock",
}


class StageContractError(RuntimeError):
    """Raised when immutable control/config data cannot support a stage."""


_SHA256_HEX = frozenset("0123456789abcdef")
_NIFTI_DTYPES = {
    2: np.dtype("u1"),
    4: np.dtype("i2"),
    8: np.dtype("i4"),
    16: np.dtype("f4"),
    64: np.dtype("f8"),
    256: np.dtype("i1"),
    512: np.dtype("u2"),
    768: np.dtype("u4"),
}
_MAX_NIFTI_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class NiftiHeader:
    """Bounded NIfTI-1 single-file metadata needed by ASD-000."""

    path: Path
    endian: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    vox_offset: int
    affine: np.ndarray
    spacing: tuple[float, ...]
    scl_slope: float
    scl_inter: float

    @property
    def data_nbytes(self) -> int:
        return int(math.prod(self.shape) * self.dtype.itemsize)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace one small text artifact atomically in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        yaml.safe_dump(
            dict(payload), sort_keys=False, allow_unicode=True
        ),
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageContractError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageContractError(f"{label} must contain a mapping: {path}")
    return payload


_ANATOMY_METADATA_PATHS = frozenset(
    {
        ("output_sha256",),
        ("task",),
        ("case", "case_key"),
        ("case", "ct_path"),
        ("case", "finding_count"),
        ("case", "name"),
        ("case", "source_axcodes"),
        ("case", "source_shape_xyz"),
        ("case", "source_spacing_xyz_mm"),
        ("case", "val_order"),
        ("geometry", "output_affine"),
        ("geometry", "output_axcodes"),
        ("geometry", "output_shape_xyz"),
        ("geometry", "output_spacing_xyz_mm"),
        ("geometry", "source_affine"),
        ("geometry", "source_axcodes"),
        ("geometry", "source_shape_xyz"),
        ("geometry", "source_spacing_xyz_mm"),
    }
)


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _skip_json_value(text: str, index: int) -> int:
    """Skip one JSON value without constructing its arrays or mappings."""

    index = _skip_json_whitespace(text, index)
    if index >= len(text):
        raise StageContractError("truncated JSON value")
    if text[index] == '"':
        _, end = json.JSONDecoder().raw_decode(text, index)
        return end
    if text[index] in "[{":
        opening = text[index]
        closing = "]" if opening == "[" else "}"
        stack = [closing]
        index += 1
        in_string = False
        escaped = False
        while index < len(text) and stack:
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "[":
                stack.append("]")
            elif character == "{":
                stack.append("}")
            elif character in "]}":
                if character != stack[-1]:
                    raise StageContractError("malformed nested JSON value")
                stack.pop()
            index += 1
        if stack or in_string:
            raise StageContractError("truncated nested JSON value")
        return index
    end = index
    while end < len(text) and text[end] not in ",]} \t\r\n":
        end += 1
    if end == index:
        raise StageContractError("malformed scalar JSON value")
    token = text[index:end]
    try:
        json.loads(token)
    except json.JSONDecodeError as exc:
        raise StageContractError("malformed scalar JSON value") from exc
    return end


def _project_json_object(
    text: str,
    index: int,
    *,
    prefix: tuple[str, ...],
    selected_paths: frozenset[tuple[str, ...]],
) -> tuple[dict[str, Any], int]:
    index = _skip_json_whitespace(text, index)
    if index >= len(text) or text[index] != "{":
        raise StageContractError("projected JSON value must be an object")
    index += 1
    result: dict[str, Any] = {}
    decoder = json.JSONDecoder()
    while True:
        index = _skip_json_whitespace(text, index)
        if index >= len(text):
            raise StageContractError("truncated projected JSON object")
        if text[index] == "}":
            return result, index + 1
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise StageContractError("malformed projected JSON key") from exc
        if not isinstance(key, str):
            raise StageContractError("projected JSON object key is not a string")
        index = _skip_json_whitespace(text, index)
        if index >= len(text) or text[index] != ":":
            raise StageContractError("malformed projected JSON object")
        index = _skip_json_whitespace(text, index + 1)
        path = (*prefix, key)
        descendants = [candidate for candidate in selected_paths if candidate[: len(path)] == path]
        if path in selected_paths:
            try:
                value, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError as exc:
                raise StageContractError(f"malformed projected JSON field {path}") from exc
            result[key] = value
        elif descendants:
            value, index = _project_json_object(
                text,
                index,
                prefix=path,
                selected_paths=selected_paths,
            )
            result[key] = value
        else:
            index = _skip_json_value(text, index)
        index = _skip_json_whitespace(text, index)
        if index >= len(text) or text[index] not in ",}":
            raise StageContractError("malformed projected JSON separator")
        if text[index] == "}":
            return result, index + 1
        index += 1


def _load_anatomy_metadata_projection(path: Path) -> dict[str, Any]:
    """Load only geometry/identity lock fields, skipping label-derived metrics."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StageContractError(f"cannot read anatomy metadata {path}: {exc}") from exc
    payload, end = _project_json_object(
        text,
        0,
        prefix=(),
        selected_paths=_ANATOMY_METADATA_PATHS,
    )
    if _skip_json_whitespace(text, end) != len(text):
        raise StageContractError(f"trailing content in anatomy metadata: {path}")
    required_top = {"case", "geometry", "output_sha256", "task"}
    if set(payload) != required_top:
        raise StageContractError(
            f"anatomy metadata projection is missing required fields: {path}"
        )
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _open_nifti(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _qform_affine(
    header: bytes,
    endian: str,
    pixdim: tuple[float, ...],
) -> np.ndarray:
    b, c, d = struct.unpack_from(f"{endian}3f", header, 256)
    x, y, z = struct.unpack_from(f"{endian}3f", header, 268)
    squared = float(b * b + c * c + d * d)
    if squared > 1.0:
        scale = math.sqrt(squared)
        b, c, d = b / scale, c / scale, d / scale
        a = 0.0
    else:
        a = math.sqrt(max(0.0, 1.0 - squared))
    rotation = np.asarray(
        [
            [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
            [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
            [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
        ],
        dtype=np.float64,
    )
    spacing = np.asarray(pixdim[1:4], dtype=np.float64)
    if np.any(spacing <= 0):
        raise StageContractError("NIfTI qform has non-positive spatial spacing")
    if pixdim[0] < 0:
        spacing[2] *= -1.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = (x, y, z)
    return affine


def read_nifti_header(path: str | Path) -> NiftiHeader:
    """Read only a bounded NIfTI-1 header, never image voxels."""

    source = Path(path)
    if not source.is_file():
        raise StageContractError(f"NIfTI input is missing: {source}")
    try:
        with _open_nifti(source) as stream:
            header = stream.read(352)
    except (OSError, EOFError) as exc:
        raise StageContractError(f"cannot read NIfTI header {source}: {exc}") from exc
    if len(header) < 348:
        raise StageContractError(f"truncated NIfTI-1 header: {source}")
    little = struct.unpack_from("<i", header, 0)[0]
    big = struct.unpack_from(">i", header, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise StageContractError(f"not a NIfTI-1 header: {source}")
    if header[344:348] != b"n+1\x00":
        raise StageContractError(
            f"ASD-000 supports only single-file NIfTI-1 images: {source}"
        )
    dimensions = struct.unpack_from(f"{endian}8h", header, 40)
    ndim = int(dimensions[0])
    if ndim not in {3, 4}:
        raise StageContractError(
            f"NIfTI image must be 3-D or 4-D, got ndim={ndim}: {source}"
        )
    shape = tuple(int(value) for value in dimensions[1 : ndim + 1])
    if any(value <= 0 for value in shape):
        raise StageContractError(f"NIfTI image has invalid shape {shape}: {source}")
    datatype = int(struct.unpack_from(f"{endian}h", header, 70)[0])
    if datatype not in _NIFTI_DTYPES:
        raise StageContractError(
            f"unsupported NIfTI datatype code {datatype}: {source}"
        )
    native_dtype = _NIFTI_DTYPES[datatype]
    dtype = native_dtype.newbyteorder(endian) if native_dtype.itemsize > 1 else native_dtype
    bitpix = int(struct.unpack_from(f"{endian}h", header, 72)[0])
    if bitpix != dtype.itemsize * 8:
        raise StageContractError(f"NIfTI datatype/bitpix mismatch: {source}")
    pixdim = tuple(float(value) for value in struct.unpack_from(f"{endian}8f", header, 76))
    raw_offset = float(struct.unpack_from(f"{endian}f", header, 108)[0])
    if not math.isfinite(raw_offset) or raw_offset < 352 or not raw_offset.is_integer():
        raise StageContractError(f"invalid NIfTI vox_offset {raw_offset}: {source}")
    vox_offset = int(raw_offset)
    slope = float(struct.unpack_from(f"{endian}f", header, 112)[0])
    intercept = float(struct.unpack_from(f"{endian}f", header, 116)[0])
    qform_code, sform_code = struct.unpack_from(f"{endian}2h", header, 252)
    if sform_code > 0:
        affine = np.eye(4, dtype=np.float64)
        affine[0] = struct.unpack_from(f"{endian}4f", header, 280)
        affine[1] = struct.unpack_from(f"{endian}4f", header, 296)
        affine[2] = struct.unpack_from(f"{endian}4f", header, 312)
    elif qform_code > 0:
        affine = _qform_affine(header, endian, pixdim)
    else:
        raise StageContractError(f"NIfTI image has no qform or sform: {source}")
    if not np.all(np.isfinite(affine)):
        raise StageContractError(f"NIfTI affine is non-finite: {source}")
    result = NiftiHeader(
        path=source,
        endian=endian,
        shape=shape,
        dtype=dtype,
        vox_offset=vox_offset,
        affine=affine,
        spacing=tuple(abs(value) for value in pixdim[1 : ndim + 1]),
        scl_slope=slope,
        scl_inter=intercept,
    )
    if result.data_nbytes > _MAX_NIFTI_BYTES:
        raise StageContractError(
            f"NIfTI image exceeds the bounded reader limit: {source}"
        )
    return result


def read_nifti_data(header_or_path: NiftiHeader | str | Path) -> np.ndarray:
    """Read one already-bounded NIfTI-1 array in native XYZ/FXYZ order."""

    header = (
        header_or_path
        if isinstance(header_or_path, NiftiHeader)
        else read_nifti_header(header_or_path)
    )
    try:
        with _open_nifti(header.path) as stream:
            stream.seek(header.vox_offset)
            payload = stream.read(header.data_nbytes)
    except (OSError, EOFError) as exc:
        raise StageContractError(
            f"cannot read NIfTI data {header.path}: {exc}"
        ) from exc
    if len(payload) != header.data_nbytes:
        raise StageContractError(f"truncated NIfTI data: {header.path}")
    array = np.frombuffer(payload, dtype=header.dtype)
    return array.reshape(header.shape, order="F")


def _axis_codes(affine: Any) -> tuple[str, str, str]:
    matrix = np.asarray(validate_ras_affine(affine), dtype=np.float64)
    spatial = matrix[:3, :3]
    assignments = [int(np.argmax(np.abs(spatial[:, axis]))) for axis in range(3)]
    if len(set(assignments)) != 3:
        raise StageContractError("affine has ambiguous voxel-to-world axes")
    labels = (("L", "R"), ("P", "A"), ("I", "S"))
    return tuple(
        labels[world_axis][1 if spatial[world_axis, voxel_axis] > 0 else 0]
        for voxel_axis, world_axis in enumerate(assignments)
    )


def _validate_geometry_mapping(
    metadata: Mapping[str, Any],
    *,
    case_name: str,
    source_header: NiftiHeader | None,
    output_header: NiftiHeader | None,
) -> dict[str, Any]:
    case = _require_mapping(metadata, "case", label=f"anatomy metadata {case_name}")
    geometry = _require_mapping(
        metadata, "geometry", label=f"anatomy metadata {case_name}"
    )
    fields = ("shape_xyz", "affine", "spacing_xyz_mm", "axcodes")
    for suffix in fields:
        source_value = geometry.get(f"source_{suffix}")
        output_value = geometry.get(f"output_{suffix}")
        if source_value != output_value:
            raise StageContractError(
                f"{case_name} source/output {suffix} differ under exact geometry policy"
            )
    if case.get("name") != case_name:
        raise StageContractError(f"anatomy metadata case name mismatch for {case_name}")
    if case.get("source_shape_xyz") != geometry.get("source_shape_xyz"):
        raise StageContractError(f"duplicate source shape mismatch for {case_name}")
    if case.get("source_spacing_xyz_mm") != geometry.get("source_spacing_xyz_mm"):
        raise StageContractError(f"duplicate source spacing mismatch for {case_name}")
    if case.get("source_axcodes") != geometry.get("source_axcodes"):
        raise StageContractError(f"duplicate source axis-code mismatch for {case_name}")
    affine = np.asarray(geometry.get("source_affine"), dtype=np.float64)
    validate_ras_affine(affine)
    declared_codes = tuple(str(value) for value in geometry.get("source_axcodes", []))
    if len(declared_codes) != 3 or _axis_codes(affine) != declared_codes:
        raise StageContractError(f"source affine/axis-code mismatch for {case_name}")
    lr_axis = left_right_axis_from_affine(affine)
    expected_lr = "R" if lr_axis.positive_voxel_direction == "right" else "L"
    if declared_codes[lr_axis.voxel_axis] != expected_lr:
        raise StageContractError(f"left/right orientation mismatch for {case_name}")
    for header, prefix in ((source_header, "source"), (output_header, "output")):
        if header is None:
            continue
        if list(header.shape) != geometry.get(f"{prefix}_shape_xyz"):
            raise StageContractError(f"{case_name} {prefix} header shape mismatch")
        if header.affine.tolist() != geometry.get(f"{prefix}_affine"):
            raise StageContractError(f"{case_name} {prefix} header affine mismatch")
        if list(header.spacing[:3]) != geometry.get(f"{prefix}_spacing_xyz_mm"):
            raise StageContractError(f"{case_name} {prefix} header spacing mismatch")
        if _axis_codes(header.affine) != tuple(geometry.get(f"{prefix}_axcodes", [])):
            raise StageContractError(f"{case_name} {prefix} header axis-code mismatch")
    return {
        "shape_xyz": list(geometry["source_shape_xyz"]),
        "affine": geometry["source_affine"],
        "spacing_xyz_mm": list(geometry["source_spacing_xyz_mm"]),
        "axcodes": list(declared_codes),
        "left_right_axis": dict(lr_axis),
    }


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageContractError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise StageContractError(f"{label} must contain a mapping: {path}")
    return loaded


def _resolve_declared_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_mapping(
    parent: Mapping[str, Any], key: str, *, label: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise StageContractError(f"{label}.{key} must be declared as a mapping")
    return value


def _require_string(
    parent: Mapping[str, Any], key: str, *, label: str
) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise StageContractError(f"{label}.{key} must be a non-empty string")
    return value


def _plan_stage(plan: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    matches = [
        stage
        for stage in plan.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("id") == stage_id
    ]
    if len(matches) != 1:
        raise StageContractError(
            f"immutable plan must declare stage {stage_id!r} exactly once"
        )
    return matches[0]


def _output_path(output: Any) -> str | None:
    if isinstance(output, str):
        return output if "/" in output or Path(output).is_absolute() else None
    if isinstance(output, Mapping):
        value = output.get("path")
        return str(value) if value else None
    return None


def _completion_evidence_path(stage: Mapping[str, Any]) -> str:
    completion = _require_mapping(stage, "completion", label=str(stage.get("id")))
    return _require_string(
        completion, "evidence_file", label=f"{stage.get('id')}.completion"
    )


def _enforce_completion_contract(stage: Mapping[str, Any]) -> None:
    """Validate that the completion evidence resolves to an allowed local path.

    The controller owns the narrow exemption for an evidence file that is also
    listed as an output.  All other output files remain hash-verified.
    """

    evidence = _resolve_declared_path(_completion_evidence_path(stage)).resolve(
        strict=False
    )
    if not _is_within(evidence, SELFDRIVE_ROOT):
        raise StageContractError(
            f"completion evidence is outside experiments_aiselfdrive: {evidence}"
        )


def _acceptance_check_ids(stage: Mapping[str, Any]) -> list[str]:
    identifiers: list[str] = []
    for index, check in enumerate(stage.get("acceptance_checks", []), start=1):
        if isinstance(check, Mapping):
            identifiers.append(
                str(check.get("id", check.get("name", f"check_{index}")))
            )
        else:
            identifiers.append(str(check))
    return identifiers


def _artifact_record(identifier: str, declared_path: str) -> dict[str, Any]:
    path = _resolve_declared_path(declared_path)
    if not path.is_file():
        raise StageContractError(f"declared output is missing: {declared_path}")
    locked = hash_file(path)
    return {
        "id": identifier,
        "path": declared_path,
        "sha256": locked["sha256"],
        "size_bytes": locked["size_bytes"],
    }


def _write_passed_evidence(
    stage: Mapping[str, Any],
    *,
    summary: str,
    metrics: Mapping[str, Any],
    outcome: str | None = None,
    skipped_stage_ids: Sequence[str] = (),
) -> Path:
    evidence_declared = _completion_evidence_path(stage)
    evidence_resolved = _resolve_declared_path(evidence_declared).resolve(
        strict=False
    )
    artifacts: list[dict[str, Any]] = []
    for index, output in enumerate(stage.get("outputs", []), start=1):
        declared = _output_path(output)
        if declared is None:
            continue
        if _resolve_declared_path(declared).resolve(strict=False) == evidence_resolved:
            continue
        identifier = (
            str(output.get("id", f"output_{index}"))
            if isinstance(output, Mapping)
            else f"output_{index}"
        )
        artifacts.append(_artifact_record(identifier, declared))

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": str(stage["id"]),
        "status": "passed",
        "completed_at": _utc_now(),
        "summary": summary,
        "checks": [
            {"id": identifier, "status": "passed", "detail": summary}
            for identifier in _acceptance_check_ids(stage)
        ],
        "artifacts": artifacts,
        "metrics": dict(metrics),
        "error": None,
    }
    if outcome is not None:
        evidence["outcome"] = outcome
    if skipped_stage_ids:
        evidence["skipped_stage_ids"] = list(skipped_stage_ids)
    path = _resolve_declared_path(evidence_declared)
    atomic_write_json(path, evidence)
    return path


def _phase_lineage(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    portfolio = _load_mapping(
        SELFDRIVE_ROOT / "portfolio.yaml", label="portfolio control plane"
    )
    source_report = _require_mapping(
        portfolio, "source_report", label="portfolio"
    )
    lineage = build_lineage_evidence(
        protocol,
        repo_root=REPO_ROOT,
        source_report=source_report,
    )
    manifest_path = SHARED_MANIFEST_ROOT / "asd000_local_lineage.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "producer_stage_id": str(stage["id"]),
            "locked_at": _utc_now(),
            "lineage": lineage,
        },
    )
    return _write_passed_evidence(
        stage,
        summary="All declared local lineage inputs and baseline metrics verified.",
        metrics={"lineage": lineage},
    )


def _case_key(case_name: str) -> str:
    if not case_name.endswith(".nii.gz") or Path(case_name).name != case_name:
        raise StageContractError(f"invalid declared case name: {case_name!r}")
    return case_name[: -len(".nii.gz")]


def _require_binary_label_header(
    header: NiftiHeader,
    *,
    ndim: int,
    label: str,
) -> None:
    if len(header.shape) != ndim:
        raise StageContractError(
            f"{label} must be {ndim}-D, got shape {header.shape}"
        )
    if header.dtype != np.dtype("u1"):
        raise StageContractError(f"{label} must use uint8 storage")
    if header.scl_slope not in {0.0, 1.0} or header.scl_inter != 0.0:
        raise StageContractError(f"{label} uses unsupported NIfTI scaling")


def _audit_completion_marker(
    marker_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    *,
    output_path: Path,
    read_output_bytes: bool,
) -> dict[str, Any]:
    marker = _load_json_mapping(marker_path, label="anatomy completion marker")
    metadata_sha = sha256_file(metadata_path)
    if marker.get("metadata_sha256") != metadata_sha:
        raise StageContractError(f"metadata SHA-256 mismatch: {metadata_path}")
    output_sha = metadata.get("output_sha256")
    if not _is_sha256(output_sha) or marker.get("output_sha256") != output_sha:
        raise StageContractError(f"output SHA-256 contract mismatch: {output_path}")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise StageContractError(f"anatomy output is missing or empty: {output_path}")
    if read_output_bytes:
        actual = sha256_file(output_path)
        if actual != output_sha:
            raise StageContractError(f"anatomy output SHA-256 mismatch: {output_path}")
    return {
        "metadata_sha256": metadata_sha,
        "output_sha256": output_sha,
        "output_bytes_read": read_output_bytes,
        "output_size_bytes": output_path.stat().st_size,
    }


def _audit_anatomy_caches(
    protocol: Mapping[str, Any],
    *,
    case_names: Sequence[str],
    val80_names: set[str],
) -> dict[str, Any]:
    anatomy = _require_mapping(protocol, "anatomy", label="protocol")
    data = _require_mapping(protocol, "data", label="protocol")
    geometry_policy = _require_mapping(
        anatomy, "geometry_policy", label="protocol.anatomy"
    )
    if geometry_policy.get("comparison") != "exact":
        raise StageContractError("ASD-000 anatomy comparison policy must be exact")
    if geometry_policy.get("val120_output_access") != (
        "metadata_and_existence_only_no_label_bytes"
    ):
        raise StageContractError("val120 output access policy is not fail-closed")
    if len(case_names) != 200 or len(set(case_names)) != 200:
        raise StageContractError("anatomy audit requires exactly 200 unique cases")
    if len(val80_names) != 80 or not val80_names <= set(case_names):
        raise StageContractError("anatomy audit requires the exact val80 subset")
    ct_root = _resolve_declared_path(
        _require_string(data, "ct_root", label="protocol.data")
    )

    fast_root = _resolve_declared_path(
        _require_string(anatomy, "fast_root", label="protocol.anatomy")
    )
    fast_manifest_path = _resolve_declared_path(
        _require_string(anatomy, "fast_manifest", label="protocol.anatomy")
    )
    fast_manifest_sha = _require_string(
        anatomy, "fast_manifest_sha256", label="protocol.anatomy"
    )
    if sha256_file(fast_manifest_path) != fast_manifest_sha:
        raise StageContractError("fast anatomy global manifest SHA-256 mismatch")
    fast_manifest = _load_json_mapping(
        fast_manifest_path, label="fast anatomy manifest"
    )
    if fast_manifest.get("cases_expected") != 200 or fast_manifest.get(
        "cases_complete"
    ) != 200:
        raise StageContractError("fast anatomy manifest is not complete for 200 cases")
    fast_per_case = _require_mapping(
        anatomy, "fast_per_case", label="protocol.anatomy"
    )
    declared_contract = fast_per_case.get("contract")
    if declared_contract != fast_manifest.get("per_case_contract"):
        raise StageContractError("fast anatomy per-case contract drifted")
    if not isinstance(declared_contract, list) or not all(
        isinstance(item, str) and item for item in declared_contract
    ):
        raise StageContractError("fast anatomy per-case contract is malformed")
    cases_directory = _require_string(
        fast_per_case, "directory", label="protocol.anatomy.fast_per_case"
    )
    fast_metadata_name = _require_string(
        fast_per_case, "metadata", label="protocol.anatomy.fast_per_case"
    )
    fast_output_name = _require_string(
        fast_per_case, "output", label="protocol.anatomy.fast_per_case"
    )
    fast_completion_name = _require_string(
        fast_per_case, "completion", label="protocol.anatomy.fast_per_case"
    )

    fast_records: list[dict[str, Any]] = []
    source_headers: dict[str, NiftiHeader] = {}
    for val_order, case_name in enumerate(case_names):
        key = _case_key(case_name)
        case_root = fast_root / cases_directory / key
        for relative in declared_contract:
            if not (case_root / relative).is_file():
                raise StageContractError(
                    f"fast anatomy contract file is missing: {case_root / relative}"
                )
        metadata_path = case_root / fast_metadata_name
        output_path = case_root / fast_output_name
        marker_path = case_root / fast_completion_name
        metadata = _load_anatomy_metadata_projection(metadata_path)
        case = _require_mapping(metadata, "case", label=f"fast anatomy {case_name}")
        if case.get("val_order") != val_order:
            raise StageContractError(f"fast anatomy val_order mismatch for {case_name}")
        ct_path = Path(_require_string(case, "ct_path", label=f"fast anatomy {case_name}.case"))
        if not ct_path.is_absolute() or not _is_within(ct_path, ct_root):
            raise StageContractError(f"source CT escapes declared root for {case_name}")
        source_header = read_nifti_header(ct_path)
        source_headers[case_name] = source_header
        read_output = case_name in val80_names
        output_header = read_nifti_header(output_path) if read_output else None
        if output_header is not None:
            _require_binary_label_header(
                output_header, ndim=3, label=f"fast anatomy {case_name}"
            )
        lock = _audit_completion_marker(
            marker_path,
            metadata_path,
            metadata,
            output_path=output_path,
            read_output_bytes=read_output,
        )
        geometry = _validate_geometry_mapping(
            metadata,
            case_name=case_name,
            source_header=source_header,
            output_header=output_header,
        )
        fast_records.append(
            {
                "case_name": case_name,
                "val_order": val_order,
                "development_output_verified": read_output,
                "geometry_sha256": _canonical_sha256(geometry),
                **lock,
            }
        )

    highres_root = _resolve_declared_path(
        _require_string(anatomy, "highres_root", label="protocol.anatomy")
    )
    highres_manifest_path = _resolve_declared_path(
        _require_string(anatomy, "highres_manifest", label="protocol.anatomy")
    )
    highres_manifest_sha = _require_string(
        anatomy, "highres_manifest_sha256", label="protocol.anatomy"
    )
    if sha256_file(highres_manifest_path) != highres_manifest_sha:
        raise StageContractError("high-resolution anatomy manifest SHA-256 mismatch")
    highres_manifest = _load_json_mapping(
        highres_manifest_path, label="high-resolution anatomy manifest"
    )
    if highres_manifest.get("cases_expected") != 200:
        raise StageContractError("high-resolution anatomy manifest case count drifted")
    task_specs = _require_mapping(
        highres_manifest, "task_specs", label="high-resolution anatomy manifest"
    )
    tasks = _require_mapping(
        highres_manifest, "tasks", label="high-resolution anatomy manifest"
    )
    if set(task_specs) != {"body", "lung_vessels", "total_roi", "trunk_cavities"}:
        raise StageContractError("high-resolution anatomy task set drifted")
    highres_records: list[dict[str, Any]] = []
    for task_id in sorted(task_specs):
        task_summary = _require_mapping(tasks, task_id, label="high-resolution anatomy tasks")
        if task_summary.get("complete") != 200 or task_summary.get("failed") != 0:
            raise StageContractError(f"high-resolution anatomy task {task_id} is incomplete")
        spec = _require_mapping(
            task_specs, task_id, label="high-resolution anatomy task_specs"
        )
        metadata_name = _require_string(spec, "metadata", label=f"task_specs.{task_id}")
        output_name = _require_string(spec, "output", label=f"task_specs.{task_id}")
        completion_name = _require_string(spec, "complete", label=f"task_specs.{task_id}")
        for val_order, case_name in enumerate(case_names):
            key = _case_key(case_name)
            case_root = highres_root / "cases" / key
            metadata_path = case_root / metadata_name
            output_path = case_root / output_name
            marker_path = case_root / completion_name
            for required_path in (metadata_path, output_path, marker_path):
                if not required_path.is_file():
                    raise StageContractError(
                        f"high-resolution anatomy contract file missing: {required_path}"
                    )
            metadata = _load_anatomy_metadata_projection(metadata_path)
            case = _require_mapping(
                metadata, "case", label=f"high-resolution anatomy {task_id}"
            )
            if case.get("val_order") != val_order:
                raise StageContractError(
                    f"high-resolution anatomy val_order mismatch for {task_id}/{case_name}"
                )
            read_output = case_name in val80_names
            output_header = read_nifti_header(output_path) if read_output else None
            if output_header is not None:
                _require_binary_label_header(
                    output_header,
                    ndim=3,
                    label=f"high-resolution anatomy {task_id}/{case_name}",
                )
            lock = _audit_completion_marker(
                marker_path,
                metadata_path,
                metadata,
                output_path=output_path,
                read_output_bytes=read_output,
            )
            geometry = _validate_geometry_mapping(
                metadata,
                case_name=case_name,
                source_header=source_headers[case_name],
                output_header=output_header,
            )
            highres_records.append(
                {
                    "task_id": task_id,
                    "case_name": case_name,
                    "development_output_verified": read_output,
                    "geometry_sha256": _canonical_sha256(geometry),
                    **lock,
                }
            )

    return {
        "geometry_comparison": "exact",
        "val120_output_access": "metadata_and_existence_only_no_label_bytes",
        "fast": {
            "manifest_path": str(fast_manifest_path),
            "manifest_sha256": fast_manifest_sha,
            "case_count": len(fast_records),
            "val80_output_hash_count": sum(
                bool(record["development_output_verified"])
                for record in fast_records
            ),
            "val120_output_byte_read_count": 0,
            "case_records_sha256": _canonical_sha256(fast_records),
        },
        "highres": {
            "manifest_path": str(highres_manifest_path),
            "manifest_sha256": highres_manifest_sha,
            "task_count": len(task_specs),
            "case_task_count": len(highres_records),
            "val80_output_hash_count": sum(
                bool(record["development_output_verified"])
                for record in highres_records
            ),
            "val120_output_byte_read_count": 0,
            "case_task_records_sha256": _canonical_sha256(highres_records),
        },
    }


def _phase_cohorts(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    val200 = _require_mapping(cohorts, "val200", label="protocol.cohorts")
    val80 = _require_mapping(cohorts, "val80", label="protocol.cohorts")

    locked = seal_cohorts(
        _resolve_declared_path(
            _require_string(val200, "path", label="protocol.cohorts.val200")
        ),
        _resolve_declared_path(
            _require_string(val80, "path", label="protocol.cohorts.val80")
        ),
        SHARED_MANIFEST_ROOT / "val80_development_seed20260731.json",
        SHARED_MANIFEST_ROOT / "val120_complement_seed20260731.sealed.json",
        expected_val200_cases=int(val200["cases"]),
        expected_val80_cases=int(val80["cases"]),
        expected_val80_findings=int(val80["findings"]),
        expected_val120_cases=120,
        expected_val200_sha256=_require_string(
            val200, "sha256", label="protocol.cohorts.val200"
        ),
        expected_val80_sha256=_require_string(
            val80, "sha256", label="protocol.cohorts.val80"
        ),
        declared_val200_findings=int(val200["findings"]),
    )
    partition = _require_mapping(locked, "partition", label="sealed cohorts")
    parent_names = partition.get("parent_case_names")
    development_names = partition.get("development_case_names")
    if not isinstance(parent_names, list) or not all(
        isinstance(name, str) for name in parent_names
    ):
        raise StageContractError("sealed cohort parent identities are malformed")
    if not isinstance(development_names, list) or not all(
        isinstance(name, str) for name in development_names
    ):
        raise StageContractError("sealed cohort development identities are malformed")
    geometry = _audit_anatomy_caches(
        protocol,
        case_names=parent_names,
        val80_names=set(development_names),
    )
    ontology = write_ontology_manifest(
        SHARED_MANIFEST_ROOT / "prompt_ontology_v1.json"
    )
    return _write_passed_evidence(
        stage,
        summary="Development and sealed cohorts, geometry, and parser verified.",
        metrics={
            "cohorts": locked,
            "geometry": geometry,
            "prompt_ontology": ontology,
        },
    )


def _phase_logits(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    logits = _require_mapping(protocol, "logits", label="protocol")
    audit_path = _resolve_declared_path(
        _require_string(logits, "source_audit", label="protocol.logits")
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise StageContractError("protocol.logits.source_audit is not a mapping")
    candidate_id = _require_string(
        logits, "candidate_id", label="protocol.logits"
    )
    candidates = audit.get("candidates")
    if not isinstance(candidates, list):
        raise StageContractError("logit audit has no candidates list")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise StageContractError(
            f"logit audit must contain candidate {candidate_id!r} exactly once"
        )
    candidate = dict(matches[0])
    unavailable_policy = _require_string(
        logits, "unavailable_policy", label="protocol.logits"
    )
    if unavailable_policy != (
        "record_unavailable_and_omit_export_and_threshold_curves"
    ):
        raise StageContractError("unsupported logit availability policy")
    if candidate.get("accepted") is not False or candidate.get("status") != "not_started":
        raise StageContractError(
            f"declared logit candidate state drifted: {candidate!r}"
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "producer_stage_id": str(stage["id"]),
        "candidate_id": candidate_id,
        "status": "unavailable",
        "threshold_curves_available": False,
        "export_attempted": False,
        "gpu_access_attempted": False,
        "candidate": candidate,
        "source_audit_sha256": sha256_file(audit_path),
        "candidate_manifest_sha256": audit.get("candidate_manifest_sha256"),
    }
    return _write_passed_evidence(
        stage,
        summary="No verified logits are available; no export ran and threshold curves are omitted.",
        metrics={"logit_audit": result},
    )


@dataclass(frozen=True)
class _AtlasCaseSpec:
    case_name: str
    prediction_header: NiftiHeader
    segmentation_header: NiftiHeader
    anatomy_header: NiftiHeader


class _AtlasArrayCache:
    """Keep at most one val80 case's three source arrays resident."""

    def __init__(self, specs: Mapping[str, _AtlasCaseSpec]) -> None:
        self.specs = dict(specs)
        self._active_case: str | None = None
        self._arrays: dict[str, np.ndarray] = {}
        self._bounds: dict[
            tuple[str, int], tuple[tuple[slice, slice, slice], tuple[int, int, int]]
        ] = {}

    def _activate(self, case_name: str) -> None:
        if case_name not in self.specs:
            raise StageContractError(f"undeclared atlas case requested: {case_name}")
        if self._active_case != case_name:
            self._arrays.clear()
            self._active_case = case_name
            gc.collect()

    def full(self, case_name: str, source: str) -> np.ndarray:
        self._activate(case_name)
        if source not in self._arrays:
            spec = self.specs[case_name]
            header = {
                "prediction": spec.prediction_header,
                "segmentation": spec.segmentation_header,
                "anatomy": spec.anatomy_header,
            }.get(source)
            if header is None:
                raise StageContractError(f"unknown atlas array source: {source}")
            self._arrays[source] = read_nifti_data(header)
        return self._arrays[source]

    def bounds(
        self, case_name: str, finding_index: int
    ) -> tuple[tuple[slice, slice, slice], tuple[int, int, int]]:
        identity = (case_name, finding_index)
        if identity not in self._bounds:
            prediction = self.full(case_name, "prediction")[finding_index]
            positive = self.full(case_name, "segmentation")[finding_index]
            occupied = (prediction != 0) | (positive != 0)
            ranges: list[tuple[int, int]] = []
            for axis in range(3):
                other_axes = tuple(candidate for candidate in range(3) if candidate != axis)
                occupied_axis = np.flatnonzero(np.any(occupied, axis=other_axes))
                if occupied_axis.size:
                    ranges.append((int(occupied_axis[0]), int(occupied_axis[-1]) + 1))
                else:
                    ranges.append((0, 1))
            slices = tuple(slice(start, stop) for start, stop in ranges)
            origin = tuple(start for start, _ in ranges)
            self._bounds[identity] = (slices, origin)  # type: ignore[assignment]
        return self._bounds[identity]

    def cropped_affine(self, case_name: str, finding_index: int) -> np.ndarray:
        _, origin = self.bounds(case_name, finding_index)
        affine = self.specs[case_name].anatomy_header.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ np.asarray(origin, dtype=np.float64)
        return affine

    def full_supervision_counts(
        self,
        case_name: str,
        finding_index: int,
        *,
        target_lobe_ids: tuple[int, ...] | None,
    ) -> dict[str, int]:
        positive = self.full(case_name, "segmentation")[finding_index] != 0
        total = int(positive.size)
        positive_count = int(np.count_nonzero(positive))
        if target_lobe_ids is None:
            certified_count = 0
        else:
            anatomy = self.full(case_name, "anatomy")
            target = np.isin(anatomy, target_lobe_ids)
            certified_count = int(np.count_nonzero((~target) & (~positive)))
        return {
            "known_positive": positive_count,
            "certified_negative": certified_count,
            "unknown": total - positive_count - certified_count,
            "full_volume_voxels": total,
        }


@dataclass(frozen=True)
class _LazyFindingArray:
    cache: _AtlasArrayCache
    case_name: str
    finding_index: int
    mode: str
    lobe_ids: tuple[int, ...] = ()
    all_lobe_ids: tuple[int, ...] = ()
    subtract_positive: bool = False

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        slices, _ = self.cache.bounds(self.case_name, self.finding_index)
        positive = self.cache.full(self.case_name, "segmentation")[
            (self.finding_index, *slices)
        ] != 0
        if self.mode == "prediction":
            result = self.cache.full(self.case_name, "prediction")[
                (self.finding_index, *slices)
            ]
        elif self.mode == "positive":
            result = positive
        else:
            anatomy = self.cache.full(self.case_name, "anatomy")[slices]
            if self.mode == "lobes":
                result = np.isin(anatomy, self.lobe_ids)
            elif self.mode == "outside_lung":
                result = ~np.isin(anatomy, self.all_lobe_ids)
            else:
                raise StageContractError(f"unknown lazy atlas mask mode: {self.mode}")
            if self.subtract_positive:
                result = result & (~positive)
        array = np.asarray(result, dtype=dtype)
        return array.copy() if copy else array


@dataclass(frozen=True)
class _LazyFindingAffine:
    cache: _AtlasArrayCache
    case_name: str
    finding_index: int

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        result = np.asarray(
            self.cache.cropped_affine(self.case_name, self.finding_index),
            dtype=dtype,
        )
        return result.copy() if copy else result


def _atlas_lobe_contract(
    protocol: Mapping[str, Any],
    fast_manifest: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    anatomy = _require_mapping(protocol, "anatomy", label="protocol")
    declared = _require_mapping(anatomy, "lobe_ids", label="protocol.anatomy")
    expected_names = {
        "left_upper": "lung_upper_lobe_left",
        "left_lower": "lung_lower_lobe_left",
        "right_upper": "lung_upper_lobe_right",
        "right_middle": "lung_middle_lobe_right",
        "right_lower": "lung_lower_lobe_right",
    }
    if set(declared) != set(expected_names):
        raise StageContractError("protocol anatomy lobe-ID set drifted")
    class_map = _require_mapping(fast_manifest, "class_map", label="fast anatomy manifest")
    lobe_ids: dict[str, int] = {}
    for key, expected_name in expected_names.items():
        value = declared[key]
        if not isinstance(value, int) or class_map.get(str(value)) != expected_name:
            raise StageContractError(f"anatomy lobe ID mismatch for {key}")
        lobe_ids[key] = value
    atlas = _require_mapping(protocol, "atlas", label="protocol")
    certification = _require_mapping(
        atlas, "prompt_certification", label="protocol.atlas"
    )
    expected_certification = {
        "certifiable_groups": ["exact_lobe", "laterality_only"],
        "exact_lobe_target": "whole_lobe_safe_superset",
        "laterality_target": "whole_side_lung",
        "unsupported_groups": "unknown_without_certified_negative_regions",
        "certified_negative_overlap_policy": "subtract_released_known_positive",
    }
    if dict(certification) != expected_certification:
        raise StageContractError("prompt certification policy drifted")
    return lobe_ids, expected_certification


def _prepare_atlas_records(
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    _AtlasArrayCache,
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
]:
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    val80_config = _require_mapping(cohorts, "val80", label="protocol.cohorts")
    locked_path = SHARED_MANIFEST_ROOT / "val80_development_seed20260731.json"
    locked = _load_json_mapping(locked_path, label="locked val80 manifest")
    if locked.get("cohort_id") != "val80" or locked.get("sealed") is not True:
        raise StageContractError("development cohort manifest is not the sealed val80")
    if locked.get("case_count") != int(val80_config["cases"]) or locked.get(
        "finding_count"
    ) != int(val80_config["findings"]):
        raise StageContractError("locked val80 counts drifted")
    source = _require_mapping(locked, "source", label="locked val80 manifest")
    if source.get("path") != _require_string(
        val80_config, "path", label="protocol.cohorts.val80"
    ) or source.get("sha256") != _require_string(
        val80_config, "sha256", label="protocol.cohorts.val80"
    ):
        raise StageContractError("locked val80 source identity drifted")
    cases = locked.get("cases")
    if not isinstance(cases, list) or len(cases) != int(val80_config["cases"]):
        raise StageContractError("locked val80 case records are malformed")
    case_names = locked.get("case_names")
    if not isinstance(case_names, list) or [case.get("name") for case in cases] != case_names:
        raise StageContractError("locked val80 case identities/order drifted")
    if len(set(case_names)) != len(case_names):
        raise StageContractError("locked val80 case identities are not unique")

    baseline = _require_mapping(protocol, "baseline", label="protocol")
    predictions_root = _resolve_declared_path(
        _require_string(baseline, "predictions", label="protocol.baseline")
    )
    data = _require_mapping(protocol, "data", label="protocol")
    segmentation_root = _resolve_declared_path(
        _require_string(data, "segmentation_root", label="protocol.data")
    )
    anatomy = _require_mapping(protocol, "anatomy", label="protocol")
    fast_root = _resolve_declared_path(
        _require_string(anatomy, "fast_root", label="protocol.anatomy")
    )
    fast_manifest_path = _resolve_declared_path(
        _require_string(anatomy, "fast_manifest", label="protocol.anatomy")
    )
    if sha256_file(fast_manifest_path) != _require_string(
        anatomy, "fast_manifest_sha256", label="protocol.anatomy"
    ):
        raise StageContractError("fast anatomy manifest drifted before atlas")
    fast_manifest = _load_json_mapping(fast_manifest_path, label="fast anatomy manifest")
    lobe_ids, certification = _atlas_lobe_contract(protocol, fast_manifest)
    all_lobes = tuple(sorted(lobe_ids.values()))

    specs: dict[str, _AtlasCaseSpec] = {}
    prepared_cases: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    anatomy_records: list[dict[str, Any]] = []
    observed_findings = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise StageContractError("locked val80 contains a non-mapping case")
        case_name = _require_string(case, "name", label="locked val80 case")
        if case_name not in case_names:
            raise StageContractError(f"undeclared val80 case encountered: {case_name}")
        key = _case_key(case_name)
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(findings, Mapping) or not isinstance(categories, Mapping):
            raise StageContractError(f"val80 findings/categories malformed for {case_name}")
        finding_keys = sorted(findings, key=lambda value: int(value))
        if finding_keys != [str(index) for index in range(len(findings))]:
            raise StageContractError(f"non-contiguous finding indices for {case_name}")
        if set(categories) != set(findings):
            raise StageContractError(f"finding/category identities differ for {case_name}")
        finding_count = len(findings)
        observed_findings += finding_count
        seg_name = _require_string(case, "seg_path", label=f"locked val80 {case_name}")
        if Path(seg_name).name != seg_name:
            raise StageContractError(f"segmentation path escapes declared root: {seg_name}")
        prediction_path = predictions_root / case_name
        segmentation_path = segmentation_root / seg_name
        anatomy_root = fast_root / "cases" / key
        anatomy_path = anatomy_root / "total_labels.nii.gz"
        metadata_path = anatomy_root / "metadata.json"
        metadata = _load_anatomy_metadata_projection(metadata_path)
        completion = _load_json_mapping(
            anatomy_root / ".complete", label="fast anatomy completion marker"
        )
        if completion.get("metadata_sha256") != sha256_file(metadata_path):
            raise StageContractError(f"anatomy metadata hash drift for {case_name}")
        output_sha = metadata.get("output_sha256")
        if not _is_sha256(output_sha) or completion.get("output_sha256") != output_sha:
            raise StageContractError(f"anatomy output contract drift for {case_name}")
        if sha256_file(anatomy_path) != output_sha:
            raise StageContractError(f"anatomy output hash drift for {case_name}")

        prediction_header = read_nifti_header(prediction_path)
        segmentation_header = read_nifti_header(segmentation_path)
        anatomy_header = read_nifti_header(anatomy_path)
        _require_binary_label_header(
            prediction_header, ndim=4, label=f"prediction {case_name}"
        )
        _require_binary_label_header(
            segmentation_header, ndim=4, label=f"segmentation {case_name}"
        )
        _require_binary_label_header(
            anatomy_header, ndim=3, label=f"anatomy {case_name}"
        )
        expected_4d_shape = (finding_count, *anatomy_header.shape)
        if prediction_header.shape != expected_4d_shape or segmentation_header.shape != expected_4d_shape:
            raise StageContractError(
                f"finding channel/geometry mismatch for {case_name}: "
                f"prediction={prediction_header.shape}, segmentation={segmentation_header.shape}, "
                f"expected={expected_4d_shape}"
            )
        identity = np.eye(4, dtype=np.float64)
        if not np.array_equal(prediction_header.affine, identity) or not np.array_equal(
            segmentation_header.affine, identity
        ):
            raise StageContractError(f"prediction/segmentation header is not FXYZ identity for {case_name}")
        geometry = _validate_geometry_mapping(
            metadata,
            case_name=case_name,
            source_header=None,
            output_header=anatomy_header,
        )
        specs[case_name] = _AtlasCaseSpec(
            case_name=case_name,
            prediction_header=prediction_header,
            segmentation_header=segmentation_header,
            anatomy_header=anatomy_header,
        )
        prepared_cases.append((case, dict(findings)))
        anatomy_records.append(
            {
                "case_name": case_name,
                "output_sha256": output_sha,
                "geometry_sha256": _canonical_sha256(geometry),
                "finding_count": finding_count,
            }
        )
    if observed_findings != int(val80_config["findings"]):
        raise StageContractError("locked val80 finding count drifted during atlas preparation")

    cache = _AtlasArrayCache(specs)
    records: list[dict[str, Any]] = []
    supervision: dict[tuple[str, str], dict[str, Any]] = {}
    for case, findings in prepared_cases:
        case_name = str(case["name"])
        case_findings: list[dict[str, Any]] = []
        for finding_id in sorted(findings, key=lambda value: int(value)):
            finding_index = int(finding_id)
            parsed = parse_prompt(str(findings[finding_id]))
            target_ids: tuple[int, ...] | None = None
            ipsilateral_ids: tuple[int, ...] = ()
            contralateral_ids: tuple[int, ...] = ()
            target = parsed.restrictive_target or {}
            if parsed.group == "exact_lobe":
                lobe = target.get("lobe")
                if lobe not in lobe_ids:
                    raise StageContractError(f"parser emitted unknown lobe {lobe!r}")
                target_ids = (lobe_ids[lobe],)
                side = str(lobe).split("_", 1)[0]
                ipsilateral_ids = tuple(
                    value
                    for key, value in lobe_ids.items()
                    if key.startswith(f"{side}_") and value not in target_ids
                )
                other_side = "left" if side == "right" else "right"
                contralateral_ids = tuple(
                    value for key, value in lobe_ids.items() if key.startswith(f"{other_side}_")
                )
            elif parsed.group == "laterality_only":
                side = target.get("laterality")
                if side not in {"left", "right"}:
                    raise StageContractError("laterality-only parser output lacks a side")
                target_ids = tuple(
                    value for key, value in lobe_ids.items() if key.startswith(f"{side}_")
                )
                other_side = "left" if side == "right" else "right"
                contralateral_ids = tuple(
                    value for key, value in lobe_ids.items() if key.startswith(f"{other_side}_")
                )
            record: dict[str, Any] = {
                "finding_id": str(finding_id),
                "prediction_mask": _LazyFindingArray(
                    cache, case_name, finding_index, "prediction"
                ),
                "known_positive_mask": _LazyFindingArray(
                    cache, case_name, finding_index, "positive"
                ),
                "affine": _LazyFindingAffine(cache, case_name, finding_index),
            }
            status = "available" if target_ids is not None else "unavailable"
            if target_ids is not None:
                record.update(
                    {
                        "prompt_target_mask": _LazyFindingArray(
                            cache, case_name, finding_index, "lobes", target_ids
                        ),
                        "ipsilateral_nontarget_mask": _LazyFindingArray(
                            cache,
                            case_name,
                            finding_index,
                            "lobes",
                            ipsilateral_ids,
                            subtract_positive=True,
                        ),
                        "contralateral_mask": _LazyFindingArray(
                            cache,
                            case_name,
                            finding_index,
                            "lobes",
                            contralateral_ids,
                            subtract_positive=True,
                        ),
                        "outside_lung_mask": _LazyFindingArray(
                            cache,
                            case_name,
                            finding_index,
                            "outside_lung",
                            all_lobe_ids=all_lobes,
                            subtract_positive=True,
                        ),
                        "anatomy_compatible_mask": _LazyFindingArray(
                            cache, case_name, finding_index, "lobes", target_ids
                        ),
                    }
                )
            case_findings.append(record)
            supervision[(case_name, str(finding_id))] = {
                "status": status,
                "parser_group": parsed.group,
                "parser_reason_codes": list(parsed.reason_codes),
                "target_lobe_ids": target_ids,
            }
        records.append(
            {"case_id": case_name, "cohort": "val80", "findings": case_findings}
        )
    preparation = {
        "locked_val80_path": str(locked_path),
        "locked_val80_sha256": sha256_file(locked_path),
        "source_val80_sha256": source["sha256"],
        "case_count": len(records),
        "finding_count": observed_findings,
        "anatomy_records_sha256": _canonical_sha256(anatomy_records),
        "prompt_certification": certification,
        "val120_files_opened": 0,
    }
    return records, cache, supervision, preparation


def _mean_available(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(math.fsum(sorted(values)) / len(values)) if values else None


def _postprocess_atlas(
    report: dict[str, Any],
    *,
    cache: _AtlasArrayCache,
    supervision: Mapping[tuple[str, str], Mapping[str, Any]],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    rows = report.get("rows")
    components = report.get("component_rows")
    if not isinstance(rows, list) or not isinstance(components, list):
        raise StageContractError("error atlas rows/component rows are malformed")
    unavailable_fields = (
        "certified_precision",
        "certified_negative_fp_voxels",
        "prompt_off_location_fp_voxels",
        "prompt_off_location_fp_volume_mm3",
        "prompt_off_location_fp_fraction",
        "prompt_off_location_fp_components",
        "detached_off_location_component_count",
        "detached_off_location_component_fraction",
        "anatomy_overlap_fraction",
        "ipsilateral_nontarget_fp_voxels",
        "ipsilateral_nontarget_fp_volume_mm3",
        "ipsilateral_nontarget_fp_components",
        "contralateral_fp_voxels",
        "contralateral_fp_volume_mm3",
        "contralateral_fp_components",
        "outside_lung_fp_voxels",
        "outside_lung_fp_volume_mm3",
        "outside_lung_fp_components",
    )
    available_count = 0
    origins: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in rows:
        identity = (str(row["case_id"]), str(row["finding_id"]))
        contract = supervision.get(identity)
        if contract is None:
            raise StageContractError(f"atlas returned undeclared finding {identity}")
        finding_index = int(identity[1])
        _, origin = cache.bounds(identity[0], finding_index)
        origins[identity] = origin
        row.update(
            {
                "crop_origin_i": origin[0],
                "crop_origin_j": origin[1],
                "crop_origin_k": origin[2],
                "spatial_supervision_status": contract["status"],
                "prompt_parser_group": contract["parser_group"],
                "prompt_parser_reason_codes": contract["parser_reason_codes"],
            }
        )
        counts = cache.full_supervision_counts(
            identity[0],
            finding_index,
            target_lobe_ids=contract["target_lobe_ids"],
        )
        row["supervision_known_positive_voxels"] = counts["known_positive"]
        row["supervision_certified_negative_voxels"] = counts["certified_negative"]
        row["supervision_unknown_voxels"] = counts["unknown"]
        row["full_volume_voxels"] = counts["full_volume_voxels"]
        if contract["status"] == "available":
            available_count += 1
        else:
            for key in unavailable_fields:
                if key in row:
                    row[key] = None
    for component in components:
        identity = (str(component["case_id"]), str(component["finding_id"]))
        if identity not in origins:
            raise StageContractError(f"component returned for undeclared finding {identity}")
        origin = origins[identity]
        for key in ("centroid_voxel_ijk", "bbox_min_ijk", "bbox_max_exclusive_ijk"):
            values = component.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise StageContractError(f"component {key} is malformed for {identity}")
            component[key] = [values[index] + origin[index] for index in range(3)]
        contract = supervision[identity]
        component["crop_origin_ijk"] = list(origin)
        component["spatial_supervision_status"] = contract["status"]
        if contract["status"] != "available":
            component["off_location_voxels"] = None
            component["fully_certified_negative"] = None
            relations = component.get("relation_voxel_counts")
            if isinstance(relations, dict):
                for key in (
                    "ipsilateral_nontarget_fp",
                    "contralateral_fp",
                    "outside_lung_fp",
                    "other_certified_negative_fp",
                ):
                    relations[key] = None

    bootstrap_keys = (
        "raw_mask_dice",
        "raw_mask_hit",
        "prediction_empty",
        "known_positive_recall",
        "certified_precision",
        "predicted_volume_mm3",
        "prompt_off_location_fp_fraction",
        "detached_off_location_component_fraction",
        "candidate_recall",
        "anatomy_overlap_fraction",
    )
    totals_keys = (
        "prediction_voxels",
        "known_positive_voxels",
        "certified_negative_fp_voxels",
        "unknown_prediction_voxels",
        "prompt_off_location_fp_voxels",
        "prediction_component_count",
        "detached_component_count",
        "detached_off_location_component_count",
    )
    summary = _require_mapping(report, "summary", label="error atlas")
    summary["metric_means"] = {
        key: _mean_available(rows, key) for key in bootstrap_keys
    }
    summary["metric_totals"] = {
        key: int(sum(int(row[key]) for row in rows if row.get(key) is not None))
        for key in totals_keys
    }
    summary["bootstrap"] = deterministic_case_cluster_bootstrap(
        rows,
        bootstrap_keys,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    report["spatial_supervision_availability"] = {
        "available_finding_count": available_count,
        "unavailable_finding_count": len(rows) - available_count,
        "unavailable_metrics_are_null": True,
        "compatible_unlabeled_policy": "unknown",
    }
    return report


def _phase_atlas(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    atlas = _require_mapping(protocol, "atlas", label="protocol")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    if atlas.get("availability_policy") != (
        "verified_logits_only_otherwise_omit_threshold_curves"
    ):
        raise StageContractError("atlas availability policy drifted")
    records, cache, supervision, preparation = _prepare_atlas_records(protocol)
    report = build_error_atlas(
        records,
        cohort="val80",
        bootstrap_draws=int(atlas["bootstrap_draws"]),
        bootstrap_seed=int(atlas["bootstrap_seed"]),
        primary_threshold=float(baseline["threshold"]),
        hit_threshold=float(baseline["hit_threshold"]),
        connectivity=int(atlas["connectivity"]),
        thresholds=[float(value) for value in atlas["thresholds"]],
        expected_case_count=int(atlas["expected_cases"]),
        expected_finding_count=int(atlas["expected_findings"]),
    )
    report = _postprocess_atlas(
        report,
        cache=cache,
        supervision=supervision,
        bootstrap_seed=int(atlas["bootstrap_seed"]),
        bootstrap_draws=int(atlas["bootstrap_draws"]),
    )
    if report.get("case_count") != int(atlas["expected_cases"]) or report.get(
        "finding_count"
    ) != int(atlas["expected_findings"]):
        raise StageContractError("error atlas completeness check failed")
    if report.get("raw_mask_reconstruction_exact") is not True:
        raise StageContractError("error atlas raw-mask reconstruction is not exact")
    if report.get("logit_status", {}).get("threshold_curves_included") is not False:
        raise StageContractError("unverified threshold curves entered the atlas")
    payload_sha256 = _canonical_sha256(report)
    result_csv = RESULTS_ROOT / "error_atlas.csv"
    summary_md = RESULTS_ROOT / "error_atlas_summary.md"
    rows = report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise StageContractError("error atlas returned no finding rows")
    atomic_write_csv(result_csv, rows)
    atomic_write_json(
        SHARED_MANIFEST_ROOT / "asd000_error_atlas_contract.json",
        {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "producer_stage_id": str(stage["id"]),
            "atlas_payload_sha256": payload_sha256,
            "finding_count": len(rows),
            "cohort": "val80",
            "logit_status": report["logit_status"],
            "spatial_supervision_availability": report[
                "spatial_supervision_availability"
            ],
        },
    )
    means = report["summary"]["metric_means"]
    _atomic_write_text(
        summary_md,
        "# ASD-000 val80 error atlas\n\n"
        f"- Cases: {report.get('case_count')}\n"
        f"- Findings: {report.get('finding_count')}\n"
        f"- Mean raw-mask Dice: {means.get('raw_mask_dice')}\n"
        f"- Mean raw-mask hit: {means.get('raw_mask_hit')}\n"
        f"- Spatial supervision available: "
        f"{report['spatial_supervision_availability']['available_finding_count']}\n"
        f"- Verified logits: false\n"
        f"- Bootstrap draws: {atlas['bootstrap_draws']}\n",
    )
    return _write_passed_evidence(
        stage,
        summary="Every val80 finding was included in the deterministic error atlas.",
        metrics={
            "atlas": report,
            "atlas_payload_sha256": payload_sha256,
            "preparation": preparation,
        },
    )


def _validate_prior_evidence_contract(
    stage_plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    stage_id = str(stage_plan.get("id"))
    if (
        evidence.get("experiment_id") != EXPERIMENT_ID
        or evidence.get("stage_id") != stage_id
        or evidence.get("status") != "passed"
    ):
        raise StageContractError(f"invalid prior completion evidence for {stage_id}")
    checks = evidence.get("checks")
    if not isinstance(checks, list):
        raise StageContractError(f"prior evidence checks are malformed for {stage_id}")
    observed_checks = {
        str(check.get("id")): check.get("status")
        for check in checks
        if isinstance(check, Mapping)
    }
    expected_checks = set(_acceptance_check_ids(stage_plan))
    if set(observed_checks) != expected_checks or any(
        status != "passed" for status in observed_checks.values()
    ):
        raise StageContractError(f"prior acceptance checks failed for {stage_id}")
    if not isinstance(evidence.get("artifacts"), list):
        raise StageContractError(f"prior artifact evidence is malformed for {stage_id}")


def _phase_closeout(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    plan = _load_mapping(EXPERIMENT_ROOT / "experiment.yaml", label="plan")
    required_evidence = {
        "lock_local_lineage": RESULTS_ROOT / "lineage_lock.json",
        "lock_cohorts_and_prompt_ontology": RESULTS_ROOT
        / "cohort_and_parser_lock.json",
        "build_val80_error_atlas": RESULTS_ROOT / "error_atlas.json",
    }
    evidence_payloads: dict[str, dict[str, Any]] = {}
    for stage_id, evidence_path in required_evidence.items():
        stage_plan = _plan_stage(plan, stage_id)
        evidence = _load_json_mapping(
            evidence_path, label=f"{stage_id} completion evidence"
        )
        _validate_prior_evidence_contract(stage_plan, evidence)
        artifacts = evidence.get("artifacts")
        assert isinstance(artifacts, list)
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise StageContractError(f"prior artifact record is malformed for {stage_id}")
            declared = _require_string(artifact, "path", label=f"{stage_id}.artifact")
            artifact_path = _resolve_declared_path(declared)
            if not artifact_path.is_file():
                raise StageContractError(f"prior stage artifact is missing: {declared}")
            if artifact.get("sha256") != sha256_file(artifact_path) or artifact.get(
                "size_bytes"
            ) != artifact_path.stat().st_size:
                raise StageContractError(f"prior stage artifact lock drifted: {declared}")
        evidence_payloads[stage_id] = evidence

    lineage_metrics = _require_mapping(
        evidence_payloads["lock_local_lineage"], "metrics", label="lineage evidence"
    )
    lineage = _require_mapping(lineage_metrics, "lineage", label="lineage metrics")
    if lineage.get("verified") is not True or not all(
        value is True
        for value in _require_mapping(lineage, "checks", label="lineage").values()
    ):
        raise StageContractError("lineage gates are not all verified")
    cohort_metrics = _require_mapping(
        evidence_payloads["lock_cohorts_and_prompt_ontology"],
        "metrics",
        label="cohort evidence",
    )
    cohorts = _require_mapping(cohort_metrics, "cohorts", label="cohort metrics")
    partition = _require_mapping(cohorts, "partition", label="cohort metrics")
    if (
        partition.get("parent_count") != 200
        or partition.get("development_count") != 80
        or partition.get("confirmatory_count") != 120
    ):
        raise StageContractError("cohort partition gate failed")
    geometry = _require_mapping(cohort_metrics, "geometry", label="cohort metrics")
    if (
        geometry.get("geometry_comparison") != "exact"
        or geometry.get("val120_output_access")
        != "metadata_and_existence_only_no_label_bytes"
        or _require_mapping(geometry, "fast", label="geometry").get("case_count")
        != 200
        or _require_mapping(geometry, "highres", label="geometry").get(
            "case_task_count"
        )
        != 800
    ):
        raise StageContractError("geometry/orientation gate failed")
    atlas_metrics = _require_mapping(
        evidence_payloads["build_val80_error_atlas"],
        "metrics",
        label="atlas evidence",
    )
    atlas = _require_mapping(atlas_metrics, "atlas", label="atlas metrics")
    configured_atlas = _require_mapping(protocol, "atlas", label="protocol")
    if (
        atlas.get("case_count") != int(configured_atlas["expected_cases"])
        or atlas.get("finding_count") != int(configured_atlas["expected_findings"])
        or atlas.get("raw_mask_reconstruction_exact") is not True
    ):
        raise StageContractError("error-atlas completeness gate failed")
    if atlas_metrics.get("atlas_payload_sha256") != _canonical_sha256(atlas):
        raise StageContractError("error-atlas canonical payload hash drifted")

    artifact_path = EXPERIMENT_ROOT / "artifacts" / "manifest.json"
    artifact_manifest = _load_json_mapping(artifact_path, label="artifact manifest")
    artifact_rows = artifact_manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise StageContractError("artifact manifest rows are malformed")
    verified_ids: list[str] = []
    for artifact in artifact_rows:
        if not isinstance(artifact, dict):
            raise StageContractError("artifact manifest contains a malformed row")
        if not artifact.get("required"):
            continue
        declared = _require_string(artifact, "path", label="artifact manifest row")
        resolved = _resolve_declared_path(declared)
        if not resolved.is_file():
            raise StageContractError(f"required artifact is missing: {declared}")
        locked = hash_file(resolved)
        previous_sha = artifact.get("sha256")
        if artifact.get("status") == "verified" and previous_sha not in {
            None,
            locked["sha256"],
        }:
            raise StageContractError(f"required artifact hash drifted: {declared}")
        artifact["status"] = "verified"
        artifact["sha256"] = locked["sha256"]
        artifact["size_bytes"] = locked["size_bytes"]
        verified_ids.append(str(artifact["id"]))
    atomic_write_json(artifact_path, artifact_manifest)

    claims_path = EXPERIMENT_ROOT / "claims.yaml"
    claims = _load_mapping(claims_path, label="claims")
    claim_rows = claims.get("claims")
    if not isinstance(claim_rows, list):
        raise StageContractError("claims rows are malformed")
    decisions = {
        "canonical_lineage_reproducible": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/lineage_lock.json"
            ],
            "rationale": "The locked checkpoint, inputs, operating point, Dice tolerance, and exact hit count passed both lineage gates.",
        },
        "spatial_error_taxonomy_measurable": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/cohort_and_parser_lock.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/error_atlas.json",
            ],
            "rationale": "The conservative parser and exact-geometry val80 atlas passed with explicit per-finding spatial-supervision availability.",
        },
        "tri_state_policy_preserved": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/cohort_and_parser_lock.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/error_atlas.json",
            ],
            "rationale": "Known-positive, certified-negative, and compatible or unsupported unknown regions remain separately counted and evidenced.",
        },
    }
    if {str(claim.get("id")) for claim in claim_rows if isinstance(claim, Mapping)} != set(
        decisions
    ):
        raise StageContractError("predeclared claim identities drifted")
    decided_at = _utc_now()
    for claim in claim_rows:
        if not isinstance(claim, dict):
            raise StageContractError("claims contains a malformed row")
        decision = decisions[str(claim["id"])]
        if not claim.get("gate_refs"):
            raise StageContractError(f"claim {claim['id']} has no gate references")
        claim["verdict"] = "supported"
        claim["evidence_refs"] = decision["evidence_refs"]
        claim["decided_at"] = decided_at
        claim["rationale"] = decision["rationale"]
    _atomic_write_yaml(claims_path, claims)

    summary_path = RESULTS_ROOT / "summary.md"
    means = _require_mapping(
        _require_mapping(atlas, "summary", label="atlas"),
        "metric_means",
        label="atlas summary",
    )
    _atomic_write_text(
        summary_path,
        "# ASD-000 evidence-lock closeout\n\n"
        "Outcome: GO.\n\n"
        f"- Required artifacts verified: {len(verified_ids)}\n"
        f"- Val80 cases/findings: {atlas['case_count']}/{atlas['finding_count']}\n"
        f"- Val80 mean raw-mask Dice: {means.get('raw_mask_dice')}\n"
        f"- Val80 mean raw-mask hit: {means.get('raw_mask_hit')}\n"
        f"- Verified logits: {atlas.get('logit_status', {}).get('verified_complete')}\n"
        "- Confirmatory val120 labels, predictions, and metrics accessed: no\n",
    )
    closeout_metrics = {
        "outcome": "go",
        "completed_stage_ids": list(required_evidence),
        "verified_artifact_ids": sorted(verified_ids),
        "claims_supported": sorted(decisions),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "claims_sha256": sha256_file(claims_path),
        "atlas_payload_sha256": atlas_metrics["atlas_payload_sha256"],
        "val120_label_prediction_metric_access": False,
    }
    return _write_passed_evidence(
        stage,
        summary="All foundation gates, required artifacts, and predeclared claims verified.",
        metrics=closeout_metrics,
        outcome="go",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable ASD-000 evidence-lock phase"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_TO_STAGE))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        protocol = load_protocol(args.config)
        if protocol.get("experiment_id") != EXPERIMENT_ID:
            raise StageContractError(
                "protocol experiment_id does not match the owning experiment"
            )
        plan = _load_mapping(EXPERIMENT_ROOT / "experiment.yaml", label="plan")
        stage = _plan_stage(plan, PHASE_TO_STAGE[args.phase])
        _enforce_completion_contract(stage)

        handlers = {
            "lineage": _phase_lineage,
            "cohorts": _phase_cohorts,
            "logits": _phase_logits,
            "atlas": _phase_atlas,
            "closeout": _phase_closeout,
        }
        evidence_path = handlers[args.phase](protocol, stage)
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "phase": args.phase,
                    "evidence": str(evidence_path),
                    "ok": True,
                },
                sort_keys=True,
            )
        )
        return 0
    except (EvidenceLockError, StageContractError, OSError, ValueError) as exc:
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
