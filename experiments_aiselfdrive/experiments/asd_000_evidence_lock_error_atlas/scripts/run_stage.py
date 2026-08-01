#!/usr/bin/env python3
"""Fail-closed command entry point for the ASD-000 evidence-lock stages.

Scientific values come only from ``protocol.yaml``.  The immutable experiment
plan is read solely as control-plane metadata so the runner can enforce its
allowed outputs, acceptance-check identifiers, and completion contract.
"""

from __future__ import annotations

import argparse
import errno
import gc
import gzip
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import stat
import struct
import subprocess
import sys
import time
import unittest
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
    "selfcheck": "selfcheck_revision_2_implementation",
    "lineage": "lock_local_lineage",
    "cohorts": "lock_cohorts_and_prompt_ontology",
    "logits": "verify_or_export_val80_logits",
    "atlas": "build_val80_error_atlas",
    "closeout": "closeout_evidence_lock",
}


class StageContractError(RuntimeError):
    """Raised when immutable control/config data cannot support a stage."""


class AcceptedLogitSourceUnavailable(StageContractError):
    """Raised when a legacy accepted export lacks required local provenance."""


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
_GIB = 1024**3
_LOCAL_LOGIT_DURABLE_STORAGE_GIB = 64
_VAL80_FLOAT16_UNCOMPRESSED_BYTES = 37_666_291_712
_VAL80_MIN_CASE_FLOAT16_BYTES = 103_809_024
_VAL80_MAX_CASE_FLOAT16_BYTES = 1_274_019_840
_RESOURCE_CPU_SECONDS_STARTED = 0.0
_ACTIVE_RUNTIME_ROOT: Path | None = None
_ACTIVE_DURABLE_FILES: tuple[Path, ...] = ()
_DURABLE_STORAGE_HIGH_WATER_BYTES = 0
_RESOURCE_GPU_SECONDS = 0.0
_COMMAND_EXPERIMENT_ENV = "REXGROUNDINGCT_AISELFDRIVE_EXPERIMENT_ID"
_COMMAND_LEASE_ENV = "REXGROUNDINGCT_AISELFDRIVE_LEASE_ID"
_COMMAND_DOCKER_LABEL_ENV = "REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL"


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
    qform_code: int
    sform_code: int

    @property
    def data_nbytes(self) -> int:
        return int(math.prod(self.shape) * self.dtype.itemsize)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_tree_cpu_seconds() -> float:
    """Return CPU seconds consumed by this process and its reaped children."""

    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(
        own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
    )


_RESOURCE_CPU_SECONDS_STARTED = _process_tree_cpu_seconds()


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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_files(paths: Sequence[Path]) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for path in paths
    }


def _restore_file_snapshot(snapshot: Mapping[Path, bytes | None]) -> None:
    for path, payload in snapshot.items():
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(path, payload)


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
        qform_code=int(qform_code),
        sform_code=int(sform_code),
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


def verify_metric_mask_geometry(
    prediction_path: Path, ground_truth_path: Path
) -> dict[str, Any]:
    """Require exact prediction/GT NIfTI geometry for metric recomputation."""

    prediction = read_nifti_header(prediction_path)
    ground_truth = read_nifti_header(ground_truth_path)
    if prediction.shape != ground_truth.shape:
        raise StageContractError(
            f"prediction/GT header shape mismatch: {prediction.shape} != "
            f"{ground_truth.shape}"
        )
    if len(prediction.shape) != 4:
        raise StageContractError(
            f"metric masks must use F-first 4-D storage, got {prediction.shape}"
        )
    if not np.array_equal(prediction.affine, ground_truth.affine):
        raise StageContractError("prediction/GT affine mismatch")
    if prediction.qform_code != ground_truth.qform_code:
        raise StageContractError("prediction/GT qform-code mismatch")
    if prediction.sform_code != ground_truth.sform_code:
        raise StageContractError("prediction/GT sform-code mismatch")
    prediction_codes = _axis_codes(prediction.affine)
    ground_truth_codes = _axis_codes(ground_truth.affine)
    if prediction_codes != ground_truth_codes:
        raise StageContractError("prediction/GT orientation-code mismatch")
    if prediction.spacing != ground_truth.spacing:
        raise StageContractError("prediction/GT header spacing mismatch")
    return {
        "verified": True,
        "shape_fxyz": list(prediction.shape),
        "affine_sha256": _canonical_sha256(prediction.affine.tolist()),
        "axis_codes": list(prediction_codes),
        "spacing": list(prediction.spacing),
        "qform_code": prediction.qform_code,
        "sform_code": prediction.sform_code,
        "f_first": True,
    }


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
    allowed_paths = stage.get("allowed_write_paths")
    if not isinstance(allowed_paths, list) or not any(
        isinstance(value, str)
        and evidence == _resolve_declared_path(value).resolve(strict=False)
        for value in allowed_paths
    ):
        raise StageContractError(
            "completion evidence is not an exact declared allowed_write_path"
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


def _regular_tree_bytes(root: Path, *, excluded: frozenset[Path]) -> int:
    """Measure regular files below ``root`` without following symlinks."""

    resolved_root = root.resolve(strict=False)
    try:
        root_metadata = resolved_root.lstat()
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise StageContractError(
            f"cannot inspect durable-storage root {resolved_root}: {exc}"
        ) from exc
    if stat.S_ISLNK(root_metadata.st_mode):
        return 0
    if stat.S_ISREG(root_metadata.st_mode):
        return (
            0
            if resolved_root in excluded
            else int(root_metadata.st_size)
        )
    if not stat.S_ISDIR(root_metadata.st_mode):
        return 0

    total = 0

    def raise_walk_error(exc: OSError) -> None:
        raise StageContractError(
            f"cannot measure durable storage below {resolved_root}: {exc}"
        ) from exc

    for directory, directory_names, file_names in os.walk(
        resolved_root, topdown=True, onerror=raise_walk_error, followlinks=False
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = directory_path / name
            try:
                if not stat.S_ISLNK(candidate.lstat().st_mode):
                    retained_directories.append(name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise StageContractError(
                    f"cannot inspect durable-storage directory {candidate}: {exc}"
                ) from exc
        directory_names[:] = retained_directories
        for name in file_names:
            candidate = (directory_path / name).resolve(strict=False)
            if candidate in excluded:
                continue
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise StageContractError(
                    f"cannot inspect durable-storage file {candidate}: {exc}"
                ) from exc
            if stat.S_ISREG(metadata.st_mode):
                total += int(metadata.st_size)
    return total


def _durable_storage_bytes(*, exclude: Sequence[Path] = ()) -> int:
    """Measure ASD-000 durable outputs at their declared owning locations."""

    excluded = frozenset(path.resolve(strict=False) for path in exclude)
    roots = [EXPERIMENT_ROOT.resolve(strict=False)]
    if _ACTIVE_RUNTIME_ROOT is not None:
        runtime = _ACTIVE_RUNTIME_ROOT.resolve(strict=False)
        if not any(_is_within(runtime, root) for root in roots):
            roots.append(runtime)
    total = sum(_regular_tree_bytes(root, excluded=excluded) for root in roots)
    for path in _ACTIVE_DURABLE_FILES:
        resolved = path.resolve(strict=False)
        if resolved in excluded or any(_is_within(resolved, root) for root in roots):
            continue
        total += _regular_tree_bytes(resolved, excluded=excluded)
    return total


def _configure_resource_meter(
    protocol: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    """Start one stage-attempt meter and discover all ASD-000 durable outputs."""

    global _ACTIVE_DURABLE_FILES
    global _ACTIVE_RUNTIME_ROOT
    global _DURABLE_STORAGE_HIGH_WATER_BYTES
    global _RESOURCE_CPU_SECONDS_STARTED
    global _RESOURCE_GPU_SECONDS

    _RESOURCE_CPU_SECONDS_STARTED = _process_tree_cpu_seconds()
    _RESOURCE_GPU_SECONDS = 0.0
    _ACTIVE_RUNTIME_ROOT = Path(
        _require_string(protocol, "runtime_root", label="protocol")
    )
    durable_files: set[Path] = set()
    for stage in plan.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        for output in stage.get("outputs", []):
            declared = _output_path(output)
            if declared is not None:
                durable_files.add(_resolve_declared_path(declared))
        completion = stage.get("completion")
        if isinstance(completion, Mapping) and completion.get("evidence_file"):
            durable_files.add(
                _resolve_declared_path(str(completion["evidence_file"]))
            )
    _ACTIVE_DURABLE_FILES = tuple(
        sorted(durable_files, key=lambda item: str(item))
    )
    _DURABLE_STORAGE_HIGH_WATER_BYTES = _durable_storage_bytes()


def _strict_json_size_bytes(payload: Mapping[str, Any]) -> int:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return len(f"{text}\n".encode("utf-8"))


def _measured_resources(
    metrics: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    evidence_path: Path,
) -> dict[str, float]:
    """Return attempt increments plus experiment durable-storage high water."""

    global _DURABLE_STORAGE_HIGH_WATER_BYTES

    supplied = metrics.get("resources", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise StageContractError("metrics.resources must be a mapping")
    gpu_hours = supplied.get("gpu_hours", _RESOURCE_GPU_SECONDS / 3600.0)
    if (
        isinstance(gpu_hours, bool)
        or not isinstance(gpu_hours, (int, float))
        or not math.isfinite(float(gpu_hours))
        or float(gpu_hours) < 0
    ):
        raise StageContractError(
            "metrics.resources.gpu_hours must be finite and nonnegative"
        )
    cpu_seconds = _process_tree_cpu_seconds() - _RESOURCE_CPU_SECONDS_STARTED
    if not math.isfinite(cpu_seconds):
        raise StageContractError("measured CPU time is non-finite")
    cpu_hours = max(0.0, float(cpu_seconds)) / 3600.0

    base_bytes = _durable_storage_bytes(exclude=(evidence_path,))
    high_water = max(_DURABLE_STORAGE_HIGH_WATER_BYTES, base_bytes)
    measured = {
        "gpu_hours": max(
            float(gpu_hours), float(_RESOURCE_GPU_SECONDS / 3600.0)
        ),
        "cpu_hours": float(cpu_hours),
        "storage_gib": float(high_water / _GIB),
    }
    # The evidence file is itself durable. Iterate its deterministic serialized
    # size to a fixed point because the reported byte count affects that size.
    draft = dict(evidence)
    for _ in range(8):
        draft_metrics = dict(metrics)
        draft_metrics["resources"] = measured
        draft["metrics"] = draft_metrics
        candidate = max(high_water, base_bytes + _strict_json_size_bytes(draft))
        if math.isclose(
            measured["storage_gib"],
            candidate / _GIB,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            high_water = candidate
            break
        high_water = candidate
        measured = {**measured, "storage_gib": float(high_water / _GIB)}
    _DURABLE_STORAGE_HIGH_WATER_BYTES = max(
        _DURABLE_STORAGE_HIGH_WATER_BYTES, high_water
    )
    return measured


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

    evidence_metrics = dict(metrics)
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
        "metrics": evidence_metrics,
        "error": None,
    }
    if outcome is not None:
        evidence["outcome"] = outcome
    if skipped_stage_ids:
        evidence["skipped_stage_ids"] = list(skipped_stage_ids)
    path = _resolve_declared_path(evidence_declared)
    evidence_metrics["resources"] = _measured_resources(
        evidence_metrics,
        evidence=evidence,
        evidence_path=path,
    )
    atomic_write_json(path, evidence)
    return path


def _classify_stage_failure(exc: BaseException) -> str:
    """Map command failures to stable controller retry/non-retry classes."""

    message = f"{type(exc).__name__}: {exc}".casefold()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if isinstance(exc, OSError):
        if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}:
            return "resource_unavailable"
        if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
            return "write_scope_violation"
        return "transient_io"
    if "budget exceeded" in message or "budget expired" in message:
        return "budget_exhausted"
    if "checkpoint" in message and (
        "hash" in message or "does not match" in message or "drift" in message
    ):
        return "checkpoint_hash_drift"
    if "cohort" in message and ("hash" in message or "sha-256" in message):
        return "cohort_hash_drift"
    if any(
        token in message
        for token in (
            "dependency hash",
            "source sha-256",
            "config hash",
            "protocol hash",
        )
    ):
        return "config_hash_drift"
    if "hash" in message or "sha-256" in message:
        return "hash_mismatch"
    if "geometry" in message or "affine" in message:
        return "geometry_mismatch"
    if any(
        token in message
        for token in ("orientation", "left-right", "left/right", "axis code")
    ):
        return "orientation_mismatch"
    if "non-finite" in message or "nonfinite" in message:
        return "non_finite"
    if any(
        token in message
        for token in (
            "escapes runtime",
            "write scope",
            "source mount is writable",
            "write mount",
        )
    ):
        return "write_scope_violation"
    if any(
        token in message
        for token in (
            "docker executable is unavailable",
            "cuda is unavailable",
            "cuda device",
            "gpu has only",
            "fewer than",
            "cidfile already exists",
            "resource unavailable",
            "free memory",
        )
    ):
        return "resource_unavailable"
    if "subprocess failed with exit code" in message:
        return "command_runtime"
    if isinstance(exc, EvidenceLockError):
        return "lineage_mismatch"
    if isinstance(exc, StageContractError) and any(
        token in message
        for token in (
            "lineage",
            "candidate",
            "embedding",
            "prompt",
            "identity",
            "dtype",
            "shape",
            "count",
            "threshold-mask",
            "contract drift",
            "proof failed",
        )
    ):
        return "lineage_mismatch"
    if isinstance(exc, StageContractError):
        return "identity_control_failure"
    if isinstance(exc, (TypeError, ValueError)):
        return "identity_control_failure"
    return "command_runtime"


def _active_failure_identity(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Read the controller-owned state without mutating it."""

    state = _load_json_mapping(EXPERIMENT_ROOT / "state.json", label="state")
    if state.get("experiment_id") != EXPERIMENT_ID:
        raise StageContractError("state experiment identity drifted")
    plan_sha256 = state.get("plan_sha256")
    if not _is_sha256(plan_sha256):
        raise StageContractError("state plan SHA-256 is malformed")
    plan_path = EXPERIMENT_ROOT / "experiment.yaml"
    if not plan_path.is_file() or sha256_file(plan_path) != plan_sha256:
        raise StageContractError("active state/plan SHA-256 drifted")
    if (
        state.get("status") != "running"
        or state.get("current_stage_id") != stage.get("id")
    ):
        raise StageContractError("failure evidence stage is not active")
    matches = [
        row
        for row in state.get("stages", [])
        if isinstance(row, Mapping) and row.get("id") == stage.get("id")
    ]
    if len(matches) != 1:
        raise StageContractError("active stage identity is absent from state")
    stage_state = matches[0]
    attempt = stage_state.get("attempts")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise StageContractError("active stage attempt is not positive")
    if stage_state.get("status") != "running":
        raise StageContractError("failure evidence stage state is not running")
    return {
        "attempt": attempt,
        "plan_sha256": plan_sha256,
        "plan_revision": int(state.get("plan_revision", 1)),
        "state_revision": int(state.get("revision", 0)),
        "state_status": str(state.get("status", "running")),
    }


def _write_failed_evidence(
    stage: Mapping[str, Any], exc: BaseException
) -> Path:
    """Emit schema-valid, attempt-bound failure evidence for experimentctl."""

    _enforce_completion_contract(stage)
    identity = _active_failure_identity(stage)
    error_class = _classify_stage_failure(exc)
    message = str(exc).strip() or type(exc).__name__
    path = _resolve_declared_path(_completion_evidence_path(stage))
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "stage_id": str(stage["id"]),
        **identity,
        "status": "failed",
        "completed_at": _utc_now(),
        "summary": f"{error_class}: {message}",
        "checks": [
            {"id": identifier, "status": "failed", "detail": message}
            for identifier in _acceptance_check_ids(stage)
        ],
        "artifacts": [],
        "metrics": {},
        "error": {
            "class": error_class,
            "message": message,
            "exception_type": type(exc).__name__,
        },
    }
    evidence["metrics"]["resources"] = _measured_resources(
        {}, evidence=evidence, evidence_path=path
    )
    atomic_write_json(path, evidence)
    return path


def _iter_test_ids(suite: unittest.TestSuite) -> list[str]:
    identifiers: list[str] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            identifiers.extend(_iter_test_ids(item))
        else:
            identifiers.append(item.id())
    return identifiers


def _phase_selfcheck(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    """Compile and run every experiment-local synthetic check deterministically."""

    if protocol.get("plan_revision") != 2:
        raise StageContractError("revision-2 selfcheck requires plan_revision 2")
    python_paths = sorted(
        [*SRC_ROOT.glob("*.py"), *(EXPERIMENT_ROOT / "scripts").glob("*.py")]
        + list((EXPERIMENT_ROOT / "tests").glob("test_*.py")),
        key=lambda path: path.as_posix(),
    )
    if not python_paths:
        raise StageContractError("revision-2 selfcheck found no Python sources")
    compiled: list[dict[str, Any]] = []
    for path in python_paths:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec", dont_inherit=True)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise StageContractError(f"selfcheck compile failed for {path}: {exc}") from exc
        compiled.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
        )

    loader = unittest.TestLoader()
    suite = loader.discover(str(EXPERIMENT_ROOT / "tests"), pattern="test_*.py")
    test_ids = sorted(_iter_test_ids(suite))
    if loader.errors:
        raise StageContractError(
            "selfcheck test discovery failed: " + " | ".join(loader.errors)
        )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not result.wasSuccessful():
        failed = sorted(
            test.id() for test, _ in [*result.failures, *result.errors]
        )
        raise StageContractError(
            "experiment-local selfcheck tests failed: " + ", ".join(failed)
        )
    skipped_test_ids = sorted(test.id() for test, _ in result.skipped)
    passed_test_ids = sorted(set(test_ids) - set(skipped_test_ids))
    report_path = RESULTS_ROOT / "revision_2_selfcheck_report.json"
    report = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "status": "passed",
        "compiled_sources": compiled,
        "compiled_source_count": len(compiled),
        "test_ids": test_ids,
        "passed_test_ids": passed_test_ids,
        "skipped_test_ids": skipped_test_ids,
        "tests_run": result.testsRun,
        "failures": 0,
        "errors": 0,
        "skipped": len(result.skipped),
    }
    atomic_write_json(report_path, report)
    return _write_passed_evidence(
        stage,
        summary=(
            f"Compiled {len(compiled)} revision-2 sources and passed "
            f"{result.testsRun} experiment-local tests."
        ),
        metrics={"selfcheck": report},
    )


def _load_locked_prompt_cohort(
    protocol: Mapping[str, Any], cohort_id: str
) -> list[dict[str, Any]]:
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    cohort = _require_mapping(cohorts, cohort_id, label="protocol.cohorts")
    path = _resolve_declared_path(
        _require_string(cohort, "path", label=f"protocol.cohorts.{cohort_id}")
    )
    expected_sha256 = _require_string(
        cohort, "sha256", label=f"protocol.cohorts.{cohort_id}"
    )
    if sha256_file(path) != expected_sha256:
        raise StageContractError(f"{cohort_id} prompt source hash drifted")
    payload = _load_json_mapping(path, label=f"{cohort_id} prompt source")
    cases = payload.get("test")
    expected_cases = int(cohort["cases"])
    expected_findings = int(cohort["findings"])
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise StageContractError(f"{cohort_id} case count drifted")
    findings = 0
    names: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise StageContractError(f"{cohort_id} contains a malformed case")
        name = _require_string(case, "name", label=f"{cohort_id} case")
        if Path(name).name != name or name in names:
            raise StageContractError(f"{cohort_id} case identity drifted: {name}")
        prompt_map = _require_mapping(
            case, "findings", label=f"{cohort_id} {name}"
        )
        expected_keys = [str(index) for index in range(len(prompt_map))]
        try:
            observed_keys = sorted(prompt_map, key=int)
        except (TypeError, ValueError) as exc:
            raise StageContractError(
                f"{cohort_id} prompt indices are malformed for {name}"
            ) from exc
        if observed_keys != expected_keys or not all(
            isinstance(prompt_map[key], str) and prompt_map[key]
            for key in expected_keys
        ):
            raise StageContractError(
                f"{cohort_id} prompt order/text contract drifted for {name}"
            )
        names.add(name)
        findings += len(prompt_map)
        normalized_cases.append(dict(case))
    if findings != expected_findings:
        raise StageContractError(f"{cohort_id} finding count drifted")
    return normalized_cases


def _ordered_prompts(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(_require_mapping(case, "findings", label="cohort case")[str(index)])
        for case in cases
        for index in range(
            len(_require_mapping(case, "findings", label="cohort case"))
        )
    ]


def _audit_locked_embedding_bank(
    protocol: Mapping[str, Any]
) -> dict[str, Any]:
    bank = _require_mapping(protocol, "embedding_bank", label="protocol")
    path = Path(_require_string(bank, "path", label="protocol.embedding_bank"))
    expected_sha256 = _require_string(
        bank, "sha256", label="protocol.embedding_bank"
    )
    if not path.is_absolute() or sha256_file(path) != expected_sha256:
        raise StageContractError("locked embedding bank path/hash drifted")
    expected_shape = tuple(int(value) for value in bank.get("expected_shape", []))
    if (
        expected_shape != (6467, 2560)
        or bank.get("expected_labels") != 6467
        or bank.get("expected_dtype") != "float16"
        or bank.get("lookup_normalization") != "python_str_lower"
        or bank.get("missing_prompt_policy")
        != "fail_before_inference_no_qwen_fallback"
        or bank.get("expected_val80_prompt_occurrences") != 195
        or bank.get("expected_val200_prompt_occurrences") != 381
        or bank.get("expected_val80_exact_case_covered_occurrences") != 7
        or bank.get("expected_val200_exact_case_covered_occurrences") != 14
    ):
        raise StageContractError("locked embedding bank declaration drifted")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"labels", "embeddings"}:
                raise StageContractError("embedding bank members drifted")
            labels = np.asarray(archive["labels"])
            embeddings = np.asarray(archive["embeddings"])
    except (OSError, ValueError, KeyError) as exc:
        raise StageContractError(f"cannot load locked embedding bank: {exc}") from exc
    if (
        labels.shape != (6467,)
        or embeddings.shape != expected_shape
        or embeddings.dtype != np.float16
        or not np.all(np.isfinite(embeddings))
    ):
        raise StageContractError("locked embedding bank data contract drifted")
    label_values = [str(value) for value in labels.tolist()]
    label_set = set(label_values)
    if len(label_set) != 6467:
        raise StageContractError("locked embedding bank labels are not unique")

    coverage: dict[str, dict[str, int]] = {}
    for cohort_id in ("val80", "val200"):
        prompts = _ordered_prompts(_load_locked_prompt_cohort(protocol, cohort_id))
        normalized = [prompt.lower() for prompt in prompts]
        missing = [prompt for prompt in normalized if prompt not in label_set]
        if missing:
            raise StageContractError(
                f"{cohort_id} has {len(missing)} prompts absent after python str.lower"
            )
        coverage[cohort_id] = {
            "prompt_occurrences": len(prompts),
            "exact_case_covered_occurrences": sum(
                prompt in label_set for prompt in prompts
            ),
            "lowercase_covered_occurrences": len(prompts),
            "unique_lowercase_prompts": len(set(normalized)),
            "missing_after_lowercase": 0,
        }
    if (
        coverage["val80"]["prompt_occurrences"] != 195
        or coverage["val200"]["prompt_occurrences"] != 381
        or coverage["val80"]["exact_case_covered_occurrences"] != 7
        or coverage["val200"]["exact_case_covered_occurrences"] != 14
    ):
        raise StageContractError("embedding-bank prompt coverage count drifted")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "shape": list(expected_shape),
        "embedding_dimension": expected_shape[1],
        "dtype": "float16",
        "label_count": 6467,
        "lookup_normalization": "python_str_lower",
        "source_text_form": "original_finding_text_verbatim",
        "coverage": coverage,
        "qwen_fallback_permitted": False,
        "predictor_embedding_mode": (
            "explicit_local_bank_with_precomputed_autoload_disabled"
        ),
        "verified": True,
    }


def _derive_locked_val80_float16_layout(
    protocol: Mapping[str, Any]
) -> dict[str, Any]:
    cases = _load_locked_prompt_cohort(protocol, "val80")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    prediction_root = Path(
        _require_string(baseline, "predictions", label="protocol.baseline")
    )
    records: list[dict[str, Any]] = []
    total_bytes = 0
    finding_count = 0
    for case in cases:
        name = _require_string(case, "name", label="val80 case")
        prompt_map = _require_mapping(case, "findings", label=f"val80 {name}")
        header = read_nifti_header(prediction_root / name)
        if (
            len(header.shape) != 4
            or header.shape[0] != len(prompt_map)
            or header.dtype != np.dtype("u1")
        ):
            raise StageContractError(
                f"locked prediction header cannot define float16 layout: {name}"
            )
        logical_bytes = int(math.prod(header.shape) * np.dtype("f2").itemsize)
        total_bytes += logical_bytes
        finding_count += len(prompt_map)
        records.append(
            {
                "case_name": name,
                "shape_fxyz": list(header.shape),
                "finding_count": len(prompt_map),
                "logical_float16_bytes": logical_bytes,
            }
        )
    return {
        "derivation": (
            "sum(product(locked_prediction_header_shape_fxyz)*float16_itemsize)"
        ),
        "case_count": len(records),
        "finding_count": finding_count,
        "array_count": len(records),
        "dtype": "float16",
        "bytes_per_value": 2,
        "logical_uncompressed_bytes": total_bytes,
        "logical_uncompressed_gib": float(total_bytes / _GIB),
        "minimum_case_bytes": min(
            int(record["logical_float16_bytes"]) for record in records
        ),
        "maximum_case_bytes": max(
            int(record["logical_float16_bytes"]) for record in records
        ),
        "case_records_sha256": _canonical_sha256(records),
        "verified": True,
    }


def _build_evaluation_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    declared = _require_mapping(
        protocol, "evaluation_contract", label="protocol"
    )
    required_sections = {
        "prompt_form",
        "inference_window",
        "threshold_policy",
        "postprocessing",
    }
    if set(declared) != required_sections:
        raise StageContractError("evaluation-contract section set drifted")
    prompt_form = _require_mapping(declared, "prompt_form", label="evaluation_contract")
    inference = _require_mapping(
        declared, "inference_window", label="evaluation_contract"
    )
    threshold = _require_mapping(
        declared, "threshold_policy", label="evaluation_contract"
    )
    postprocessing = _require_mapping(
        declared, "postprocessing", label="evaluation_contract"
    )
    if (
        prompt_form.get("source") != "locked_cohort_case_findings"
        or prompt_form.get("finding_order") != "numeric_string_index_ascending"
        or prompt_form.get("text_form") != "original_finding_text_verbatim"
        or prompt_form.get("embedding_contract")
        != "frozen_precomputed_2560d_bank"
        or prompt_form.get("lookup_normalization") != "python_str_lower"
        or prompt_form.get("missing_prompt_policy")
        != "fail_before_inference_no_qwen_fallback"
    ):
        raise StageContractError("prompt-form evaluation contract drifted")
    if (
        inference.get("preprocessing_cache_id") != "crop_zscore_native_v1"
        or inference.get("spatial_resolution") != "native"
        or inference.get("patch_size_xyz") != [192, 192, 192]
        or inference.get("sliding_window_batch_size") != 1
        or inference.get("predictor_method")
        != "predict_sliding_window_return_logits"
        or inference.get("restoration")
        != "cached_native_crop_to_original_full_geometry"
    ):
        raise StageContractError("native inference-window contract drifted")
    for path_key, hash_key, label in (
        (
            "preprocessing_cache_manifest",
            "preprocessing_cache_manifest_sha256",
            "preprocessing cache manifest",
        ),
        ("model_plans", "model_plans_sha256", "model plans"),
    ):
        path = Path(_require_string(inference, path_key, label="inference_window"))
        expected = _require_string(inference, hash_key, label="inference_window")
        if not path.is_absolute() or sha256_file(path) != expected:
            raise StageContractError(f"{label} hash drifted")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    if (
        float(threshold.get("probability_threshold", -1))
        != float(baseline["threshold"])
        or float(threshold.get("logit_threshold", math.nan)) != 0.0
        or threshold.get("comparison") != "greater_than_or_equal"
        or float(threshold.get("hit_dice_threshold", -1))
        != float(baseline["hit_threshold"])
        or threshold.get("threshold_selection") != "locked_not_retuned"
    ):
        raise StageContractError("threshold-policy evaluation contract drifted")
    if dict(postprocessing) != {
        "method": "raw_thresholded_mask",
        "connected_component_filter": "none",
        "anatomy_filter": "none",
        "automatic_zoom": "none",
        "volume_filter": "none",
    }:
        raise StageContractError("post-processing evaluation contract drifted")
    canonical = json.loads(
        json.dumps(declared, sort_keys=True, allow_nan=False)
    )
    canonical["prompt_form"]["verified_embedding_bank"] = (
        _audit_locked_embedding_bank(protocol)
    )
    return {
        **canonical,
        "contract_sha256": _canonical_sha256(canonical),
        "hash_algorithm": "canonical_json_sha256_v1",
        "verified": True,
    }


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
        mask_array_loader=read_nifti_data,
        mask_geometry_verifier=verify_metric_mask_geometry,
    )
    evaluation_contract = _build_evaluation_contract(protocol)
    manifest_path = SHARED_MANIFEST_ROOT / "asd000_local_lineage_revision_2.json"
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "producer_stage_id": str(stage["id"]),
        "locked_at": _utc_now(),
        "lineage": lineage,
        "evaluation_contract": evaluation_contract,
    }
    atomic_write_json(manifest_path, payload)
    atomic_write_json(RESULTS_ROOT / "revision_2_lineage_lock.json", payload)
    return _write_passed_evidence(
        stage,
        summary=(
            "All declared local lineage inputs were locked and baseline Dice/hits "
            "were recomputed from stored predictions plus released ground truth."
        ),
        metrics={
            "lineage": lineage,
            "evaluation_contract": evaluation_contract,
        },
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
        SHARED_MANIFEST_ROOT / "val120_internal_replication_seed20260731.json",
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
    partition = _require_mapping(locked, "partition", label="locked cohorts")
    parent_names = partition.get("parent_case_names")
    development_names = partition.get("development_case_names")
    if not isinstance(parent_names, list) or not all(
        isinstance(name, str) for name in parent_names
    ):
        raise StageContractError("locked cohort parent identities are malformed")
    if not isinstance(development_names, list) or not all(
        isinstance(name, str) for name in development_names
    ):
        raise StageContractError("locked cohort development identities are malformed")
    geometry = _audit_anatomy_caches(
        protocol,
        case_names=parent_names,
        val80_names=set(development_names),
    )
    ontology = write_ontology_manifest(
        SHARED_MANIFEST_ROOT / "prompt_ontology_v1.json"
    )
    scientific_lock = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "producer_stage_id": str(stage["id"]),
        "cohorts": locked,
        "cohort_semantics": {
            "val80": "development",
            "val120": "historically_exposed_internal_held_out_replication",
            "val120_independence": "not_independent_and_not_blinded",
            "independent_confirmation": (
                "new_externally_unseen_custodied_data_required"
            ),
        },
        "geometry": geometry,
        "prompt_ontology": ontology,
    }
    atomic_write_json(
        RESULTS_ROOT / "revision_2_cohort_and_parser_lock.json",
        scientific_lock,
    )
    return _write_passed_evidence(
        stage,
        summary=(
            "Development and historically exposed internal-replication cohorts, "
            "geometry, and parser semantics verified."
        ),
        metrics={
            "cohorts": locked,
            "geometry": geometry,
            "prompt_ontology": ontology,
        },
    )


def _select_accepted_logit_candidate(
    audit: Mapping[str, Any], candidate_id: str
) -> dict[str, Any] | None:
    candidates = audit.get("candidates")
    if not isinstance(candidates, list):
        raise StageContractError("logit audit has no candidates list")
    matches = [
        dict(item)
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise StageContractError(
            f"logit audit must contain candidate {candidate_id!r} exactly once"
        )
    candidate = matches[0]
    if candidate.get("accepted") is not True:
        return None
    required = {
        "status": "accepted",
        "cases": 200,
        "findings": 381,
        "array_hashes_verified": True,
        "source_matches_manifest": True,
        "storage_reproduction_status": "passed",
        "same_pass_mask_reproduction_status": "passed",
    }
    drift = {
        key: (candidate.get(key), expected)
        for key, expected in required.items()
        if candidate.get(key) != expected
    }
    if drift:
        raise StageContractError(
            f"accepted sideexp002 candidate fails required contracts: {drift}"
        )
    return candidate


def _validate_candidate_manifest_lineage(
    protocol: Mapping[str, Any]
) -> dict[str, Any]:
    logits = _require_mapping(protocol, "logits", label="protocol")
    candidate_id = _require_string(logits, "candidate_id", label="protocol.logits")
    manifest_path = _resolve_declared_path(
        _require_string(logits, "candidate_manifest", label="protocol.logits")
    )
    manifest = _load_json_mapping(manifest_path, label="candidate manifest")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise StageContractError("candidate manifest has no candidates list")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("id") == candidate_id
    ]
    if len(matches) != 1:
        raise StageContractError("locked candidate identity is not unique")
    candidate = matches[0]
    checkpoint = candidate.get("checkpoint")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    expected_checkpoint_path = _require_string(
        baseline, "checkpoint", label="protocol.baseline"
    )
    expected_checkpoint_sha256 = _require_string(
        baseline, "checkpoint_sha256", label="protocol.baseline"
    )
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != expected_checkpoint_path
        or checkpoint.get("sha256") != expected_checkpoint_sha256
    ):
        raise StageContractError(
            "candidate checkpoint path/hash does not match protocol baseline"
        )
    checkpoint_path = Path(expected_checkpoint_path)
    if (
        not checkpoint_path.is_absolute()
        or not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != expected_checkpoint_sha256
    ):
        raise StageContractError("locked baseline checkpoint hash drifted")
    cache = candidate.get("cache")
    inference = _require_mapping(
        _require_mapping(protocol, "evaluation_contract", label="protocol"),
        "inference_window",
        label="protocol.evaluation_contract",
    )
    if (
        not isinstance(cache, Mapping)
        or cache.get("id") != inference.get("preprocessing_cache_id")
        or cache.get("root") != inference.get("preprocessing_cache_root")
        or cache.get("manifest_path")
        != inference.get("preprocessing_cache_manifest")
        or cache.get("manifest_sha256")
        != inference.get("preprocessing_cache_manifest_sha256")
    ):
        raise StageContractError(
            "candidate preprocessing-cache lineage does not match protocol"
        )
    return {
        "candidate_id": candidate_id,
        "checkpoint_path": expected_checkpoint_path,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "candidate_manifest_path": _require_string(
            logits, "candidate_manifest", label="protocol.logits"
        ),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "preprocessing_cache_id": inference["preprocessing_cache_id"],
        "preprocessing_cache_manifest_sha256": inference[
            "preprocessing_cache_manifest_sha256"
        ],
        "verified": True,
    }


def _validate_voxtell_submodule(protocol: Mapping[str, Any]) -> dict[str, Any]:
    logits = _require_mapping(protocol, "logits", label="protocol")
    declared = _require_mapping(
        logits, "voxtell_submodule", label="protocol.logits"
    )
    root = _resolve_declared_path(
        _require_string(declared, "root", label="protocol.logits.voxtell_submodule")
    ).resolve(strict=True)
    expected_root = (REPO_ROOT / "external" / "VoxTell").resolve(strict=True)
    expected_head = _require_string(
        declared,
        "expected_head",
        label="protocol.logits.voxtell_submodule",
    )
    if (
        root != expected_root
        or re.fullmatch(r"[0-9a-f]{40}", expected_head) is None
        or declared.get("require_clean") is not True
    ):
        raise StageContractError("VoxTell submodule declaration drifted")
    head = _run_checked(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        timeout=30,
        capture_output=True,
    ).stdout.strip()
    status = _run_checked(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        timeout=30,
        capture_output=True,
    ).stdout
    if head != expected_head or status.strip():
        raise StageContractError("VoxTell submodule HEAD/clean-state drifted")
    return {
        "root": str(root),
        "head": head,
        "clean": True,
        "verified": True,
    }


def _validate_logit_embedding_proof(
    protocol: Mapping[str, Any],
    container: Mapping[str, Any],
    *,
    label: str,
    unavailable_on_failure: bool = False,
) -> dict[str, Any]:
    failure_type = (
        AcceptedLogitSourceUnavailable
        if unavailable_on_failure
        else StageContractError
    )
    proof = container.get("embedding_bank")
    if not isinstance(proof, Mapping):
        raise failure_type(f"{label} lacks locked embedding-bank provenance")
    expected = _audit_locked_embedding_bank(protocol)
    required = {
        "path": expected["path"],
        "sha256": expected["sha256"],
        "label_count": expected["label_count"],
        "shape": expected["shape"],
        "embedding_dimension": 2560,
        "dtype": expected["dtype"],
        "lookup_normalization": expected["lookup_normalization"],
        "source_text_form": expected["source_text_form"],
        "coverage": expected["coverage"],
        "qwen_fallback_permitted": False,
        "predictor_embedding_mode": (
            "explicit_local_bank_with_precomputed_autoload_disabled"
        ),
        "verified": True,
    }
    drift = {
        key: (proof.get(key), value)
        for key, value in required.items()
        if proof.get(key) != value
    }
    if drift:
        raise failure_type(f"{label} embedding-bank provenance drifted: {drift}")
    return dict(proof)


def _validate_logit_candidate_proof(
    protocol: Mapping[str, Any],
    container: Mapping[str, Any],
    *,
    label: str,
    unavailable_on_failure: bool = False,
) -> dict[str, Any]:
    failure_type = (
        AcceptedLogitSourceUnavailable
        if unavailable_on_failure
        else StageContractError
    )
    proof = container.get("candidate_lineage")
    if not isinstance(proof, Mapping):
        raise failure_type(f"{label} lacks candidate-lineage provenance")
    expected = _validate_candidate_manifest_lineage(protocol)
    drift = {
        key: (proof.get(key), value)
        for key, value in expected.items()
        if proof.get(key) != value
    }
    if drift:
        raise failure_type(f"{label} candidate lineage drifted: {drift}")
    return dict(proof)


def _validate_runtime_durability_proof(
    protocol: Mapping[str, Any], container: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    proof = container.get("runtime_durability_probe")
    expected_root = str(
        Path(_require_string(protocol, "runtime_root", label="protocol")).resolve(
            strict=False
        )
    )
    required = {
        "status": "passed",
        "runtime_root": expected_root,
        "probe_contract": (
            "atomic_replace_file_fsync_directory_fsync_read_delete_v1"
        ),
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync_after_replace": True,
        "readback_exact": True,
        "delete_verified": True,
        "directory_fsync_after_delete": True,
        "completed_before_cuda_and_predictor_initialization": True,
    }
    if not isinstance(proof, Mapping) or any(
        proof.get(key) != value for key, value in required.items()
    ):
        raise StageContractError(f"{label} runtime durability proof failed")
    return dict(proof)


def _validate_offline_container_proof(
    container: Mapping[str, Any], *, label: str
) -> None:
    if (
        container.get("network_mode") != "none"
        or container.get("offline_environment")
        != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    ):
        raise StageContractError(f"{label} offline/no-network proof failed")


def _stop_owned_docker_container(cidfile: Path) -> str:
    """Stop only a timed-out container carrying this exact controller lease."""

    try:
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "cidfile_not_created"
    except OSError as exc:
        return f"cidfile_unreadable:{type(exc).__name__}"
    if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
        return "cidfile_invalid"
    controller = _controller_docker_contract()
    try:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_id,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        labels = json.loads(inspected.stdout.strip())
        if not isinstance(labels, Mapping) or (
            labels.get("rexgroundingct.aiselfdrive.lease")
            != controller["lease_id"]
            or labels.get("rexgroundingct.aiselfdrive.experiment")
            != EXPERIMENT_ID
        ):
            return "ownership_labels_mismatch_not_stopped"
        subprocess.run(
            ["docker", "stop", "--time", "30", container_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return "owned_container_stopped"
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as exc:
        return f"owned_container_stop_failed:{type(exc).__name__}"


def _run_checked(
    argv: Sequence[str],
    *,
    timeout: float,
    capture_output: bool,
    docker_cidfile: Path | None = None,
    account_gpu_time: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one argument-array command and translate all failures fail-closed."""

    command = [str(value) for value in argv]
    if not command or any(not value for value in command):
        raise StageContractError("subprocess command contains an empty argument")
    global _RESOURCE_GPU_SECONDS

    cidfile_existed = docker_cidfile is not None and docker_cidfile.exists()
    started = time.monotonic()
    try:
        run_kwargs: dict[str, Any] = {
            "check": True,
            "text": True,
            "timeout": timeout,
        }
        if capture_output:
            run_kwargs["capture_output"] = True
        else:
            # Preserve a bounded diagnostic class on failure while allowing
            # normal stdout to stream to the controller.
            run_kwargs["stderr"] = subprocess.PIPE
        return subprocess.run(command, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        cleanup = (
            _stop_owned_docker_container(docker_cidfile)
            if docker_cidfile is not None and not cidfile_existed
            else "cidfile_preexisted_not_touched"
        )
        raise StageContractError(
            f"subprocess timed out after {timeout:.0f}s: {command[0]}; "
            f"container_cleanup={cleanup}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr or exc.stdout or ""
        diagnostic = " ".join(str(diagnostic).split())[-2000:]
        suffix = f"; stderr_tail={diagnostic}" if diagnostic else ""
        raise StageContractError(
            f"subprocess failed with exit code {exc.returncode}: "
            f"{command[0]}{suffix}"
        ) from exc
    finally:
        if account_gpu_time:
            _RESOURCE_GPU_SECONDS += max(0.0, time.monotonic() - started)
        if docker_cidfile is not None and not cidfile_existed:
            try:
                docker_cidfile.unlink(missing_ok=True)
            except OSError:
                pass


def _verify_docker_image_id(image: str, expected_id: str) -> dict[str, str]:
    if not expected_id.startswith("sha256:") or len(expected_id) != 71:
        raise StageContractError("pinned Docker image ID is malformed")
    inspected = _run_checked(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        timeout=60,
        capture_output=True,
    )
    observed_id = inspected.stdout.strip()
    if observed_id != expected_id:
        raise StageContractError(
            f"Docker image ID mismatch for {image}: {observed_id!r}"
        )
    return {"image": image, "expected_id": expected_id, "observed_id": observed_id}


def _controller_docker_contract() -> dict[str, Any]:
    """Require the controller lease identity before starting nested Docker."""

    experiment_id = os.environ.get(_COMMAND_EXPERIMENT_ENV)
    lease_id = os.environ.get(_COMMAND_LEASE_ENV)
    declared_label = os.environ.get(_COMMAND_DOCKER_LABEL_ENV)
    if experiment_id != EXPERIMENT_ID:
        raise StageContractError(
            f"{_COMMAND_EXPERIMENT_ENV} must exactly identify {EXPERIMENT_ID}"
        )
    if lease_id is None or not lease_id.strip() or lease_id != lease_id.strip():
        raise StageContractError(
            f"{_COMMAND_LEASE_ENV} must contain a nonempty lease ID"
        )
    lease_label = f"rexgroundingct.aiselfdrive.lease={lease_id}"
    if declared_label != lease_label:
        raise StageContractError(
            f"{_COMMAND_DOCKER_LABEL_ENV} must exactly match the active "
            "controller lease label"
        )
    experiment_label = (
        f"rexgroundingct.aiselfdrive.experiment={experiment_id}"
    )
    return {
        "experiment_id": experiment_id,
        "lease_id": lease_id,
        "lease_label": lease_label,
        "experiment_label": experiment_label,
        "environment": {
            _COMMAND_EXPERIMENT_ENV: experiment_id,
            _COMMAND_LEASE_ENV: lease_id,
            _COMMAND_DOCKER_LABEL_ENV: declared_label,
        },
    }


def _docker_cidfile_path(
    protocol: Mapping[str, Any], *, preflight_only: bool
) -> Path:
    contract = _controller_docker_contract()
    lease_digest = hashlib.sha256(
        str(contract["lease_id"]).encode("utf-8")
    ).hexdigest()[:24]
    mode = "preflight" if preflight_only else "export"
    runtime_root = Path(_require_string(protocol, "runtime_root", label="protocol"))
    return runtime_root / ".controller" / f"{lease_digest}-{mode}.cid"


def _validate_local_export_policy(
    protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Pin the nested-Docker write, network, bank, and storage declaration."""

    logits = _require_mapping(protocol, "logits", label="protocol")
    local = _require_mapping(logits, "local_export", label="protocol.logits")
    runtime_root = _require_string(protocol, "runtime_root", label="protocol")
    expected_runtime = (
        "/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/"
        f"{EXPERIMENT_ID}"
    )
    expected_values: dict[str, Any] = {
        "max_durable_storage_gib": _LOCAL_LOGIT_DURABLE_STORAGE_GIB,
        "disk_preflight_free_gib_min": _LOCAL_LOGIT_DURABLE_STORAGE_GIB,
        "full_layout_float16_uncompressed_bytes": (
            _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        ),
        "full_layout_float16_uncompressed_gib": (
            _VAL80_FLOAT16_UNCOMPRESSED_BYTES / _GIB
        ),
        "minimum_case_float16_bytes": _VAL80_MIN_CASE_FLOAT16_BYTES,
        "maximum_case_float16_bytes": _VAL80_MAX_CASE_FLOAT16_BYTES,
        "storage_headroom_bytes": (
            _LOCAL_LOGIT_DURABLE_STORAGE_GIB * _GIB
            - _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        ),
        "storage_headroom_gib": (
            _LOCAL_LOGIT_DURABLE_STORAGE_GIB
            - _VAL80_FLOAT16_UNCOMPRESSED_BYTES / _GIB
        ),
        "storage_capacity_multiple": (
            _LOCAL_LOGIT_DURABLE_STORAGE_GIB
            * _GIB
            / _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        ),
        "storage_basis": (
            "sum_product_locked_prediction_header_shape_fxyz_times_"
            "float16_itemsize"
        ),
        "storage_cap_rationale": (
            "64_GiB_bounds_the_export_with_28_92_GiB_headroom_over_exact_"
            "uncompressed_layout"
        ),
        "source_parent_mount": "/mnt/shengdata1:/mnt/shengdata1:ro",
        "runtime_overlay_mount": f"{expected_runtime}:{expected_runtime}:rw",
        "network_mode": "none",
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "runtime_probe_contract": (
            "atomic_replace_file_fsync_directory_fsync_read_delete_v1"
        ),
        "runtime_probe_ordering": (
            "completed_before_cuda_and_predictor_initialization"
        ),
        "host_launch_attestation": (
            f"{expected_runtime}/val80_logits_revision_2/"
            "host_launch_attestation.json"
        ),
        "cached_export_policy": (
            "require_host_launch_attestation_bound_to_manifest_sha256_or_"
            "reexport"
        ),
        "child_failure_policy": (
            "bounded_stderr_tail_preserves_hash_geometry_lineage_resource_or_"
            "timeout_class"
        ),
        "embedding_mode": (
            "explicit_local_bank_with_precomputed_autoload_disabled"
        ),
        "accepted_source_embedding_proof_policy": (
            "require_exact_bank_path_hash_shape_normalization_and_val80_"
            "val200_coverage_or_use_local_export"
        ),
        "checkpoint_binding_policy": (
            "candidate_checkpoint_path_and_sha256_equal_protocol_baseline"
        ),
    }
    if runtime_root != expected_runtime:
        raise StageContractError("ASD-000 local-export runtime root drifted")
    drift: dict[str, tuple[Any, Any]] = {}
    for key, expected in expected_values.items():
        actual = local.get(key)
        if isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                drift[key] = (actual, expected)
        elif actual != expected:
            drift[key] = (actual, expected)
    if drift:
        raise StageContractError(f"local-export policy drifted: {drift}")
    return dict(local)


def _project_accepted_sideexp_val80(
    protocol: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    """Project a fully accepted 200-case sideexp export into locked val80 order."""

    logits = _require_mapping(protocol, "logits", label="protocol")
    candidate_id = _require_string(logits, "candidate_id", label="protocol.logits")
    sideexp_root = Path(
        _require_string(logits, "sideexp_runtime_root", label="protocol.logits")
    )
    candidate_root = sideexp_root / "logits" / candidate_id
    export_path = Path(
        str(candidate.get("export_manifest_path") or candidate_root / "export_manifest.json")
    )
    reproduction_path = Path(
        str(
            candidate.get("reproduction_validation_path")
            or candidate_root / "reproduction_validation.json"
        )
    )
    if not _is_within(export_path, sideexp_root) or not _is_within(
        reproduction_path, sideexp_root
    ):
        raise StageContractError("accepted sideexp002 manifest path escapes its runtime")
    export = _load_json_mapping(export_path, label="accepted sideexp002 export")
    reproduction = _load_json_mapping(
        reproduction_path, label="accepted sideexp002 reproduction"
    )
    _validate_logit_embedding_proof(
        protocol,
        export,
        label="accepted sideexp002 export",
        unavailable_on_failure=True,
    )
    _validate_logit_embedding_proof(
        protocol,
        reproduction,
        label="accepted sideexp002 reproduction",
        unavailable_on_failure=True,
    )
    _validate_logit_candidate_proof(
        protocol,
        export,
        label="accepted sideexp002 export",
        unavailable_on_failure=True,
    )
    _validate_logit_candidate_proof(
        protocol,
        reproduction,
        label="accepted sideexp002 reproduction",
        unavailable_on_failure=True,
    )
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    val200 = _require_mapping(cohorts, "val200", label="protocol.cohorts")
    export_records = export.get("cases")
    if (
        export.get("candidate_id") != candidate_id
        or export.get("status") != "complete"
        or export.get("case_count") != 200
        or export.get("finding_count") != 381
        or export.get("dataset_sha256") != val200.get("sha256")
        or export.get("dtype") != "float16"
        or not isinstance(export_records, list)
        or len(export_records) != 200
    ):
        raise StageContractError("accepted sideexp002 export manifest is incomplete")
    reproduction_records = reproduction.get("case_results")
    if (
        reproduction.get("candidate_id") != candidate_id
        or reproduction.get("status") != "passed"
        or reproduction.get("findings") != 381
        or reproduction.get("array_hashes_verified") is not True
        or reproduction.get("storage_reproduction_status") != "passed"
        or reproduction.get("same_pass_mask_reproduction_status") != "passed"
        or not isinstance(reproduction_records, list)
        or len(reproduction_records) != 200
    ):
        raise StageContractError(
            "accepted sideexp002 reproduction manifest is incomplete"
        )
    by_name = {
        record.get("name"): record
        for record in export_records
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    reproduction_by_name = {
        record.get("name"): record
        for record in reproduction_records
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    if len(by_name) != 200 or len(reproduction_by_name) != 200:
        raise StageContractError("accepted sideexp002 case identities are not unique")

    val80 = _require_mapping(cohorts, "val80", label="protocol.cohorts")
    val80_path = _resolve_declared_path(
        _require_string(val80, "path", label="protocol.cohorts.val80")
    )
    val80_cases = _load_json_mapping(val80_path, label="val80 source").get("test")
    if not isinstance(val80_cases, list) or len(val80_cases) != 80:
        raise StageContractError("val80 source does not contain exactly 80 cases")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    val200 = _require_mapping(cohorts, "val200", label="protocol.cohorts")
    embedding_bank = _require_mapping(protocol, "embedding_bank", label="protocol")
    prediction_root = _resolve_declared_path(
        _require_string(baseline, "predictions", label="protocol.baseline")
    )
    data = _require_mapping(protocol, "data", label="protocol")
    segmentation_root = _resolve_declared_path(
        _require_string(data, "segmentation_root", label="protocol.data")
    )
    projected: list[dict[str, Any]] = []
    for order, case in enumerate(val80_cases):
        if not isinstance(case, Mapping):
            raise StageContractError("val80 source contains a malformed case")
        name = _require_string(case, "name", label="val80 case")
        findings = _require_mapping(case, "findings", label=f"val80 {name}")
        export_record = by_name.get(name)
        reproduction_record = reproduction_by_name.get(name)
        if not isinstance(export_record, Mapping) or not isinstance(
            reproduction_record, Mapping
        ):
            raise StageContractError(f"accepted sideexp002 export omits val80 case {name}")
        same_pass = reproduction_record.get("same_pass_mask_reproduction")
        if (
            export_record.get("status") != "complete"
            or export_record.get("dtype") != "float16"
            or export_record.get("same_pass_mask_comparison")
            != "exact_threshold_zero_voxels"
            or export_record.get("same_pass_mask_mismatch_voxels") != 0
            or not isinstance(same_pass, Mapping)
            or same_pass.get("passed") is not True
            or same_pass.get("mismatch_voxels") != 0
        ):
            raise StageContractError(
                f"accepted sideexp002 same-pass proof failed for {name}"
            )
        array_path = Path(
            _require_string(export_record, "array_path", label=f"sideexp002 {name}")
        )
        if not array_path.is_absolute() or not _is_within(array_path, sideexp_root):
            raise StageContractError(f"accepted sideexp002 array escapes runtime: {name}")
        expected_hash = _require_string(
            export_record, "array_sha256", label=f"sideexp002 {name}"
        )
        if not _is_sha256(expected_hash) or sha256_file(array_path) != expected_hash:
            raise StageContractError(f"accepted sideexp002 array hash failed for {name}")
        array = _load_logit_array(array_path)
        expected_shape = tuple(int(value) for value in export_record.get("shape", []))
        if (
            array.dtype != np.float16
            or array.ndim != 4
            or array.shape[0] != len(findings)
            or tuple(array.shape) != expected_shape
            or not np.all(np.isfinite(array))
        ):
            raise StageContractError(f"accepted sideexp002 array contract failed for {name}")
        seg_name = _require_string(case, "seg_path", label=f"val80 {name}")
        geometry = verify_metric_mask_geometry(
            prediction_root / name,
            segmentation_root / seg_name,
        )
        if tuple(array.shape) != tuple(geometry["shape_fxyz"]):
            raise StageContractError(f"accepted sideexp002 geometry mismatch for {name}")
        projected.append(
            {
                "status": "complete",
                "case_order": order,
                "case_name": name,
                "finding_count": len(findings),
                "array_path": str(array_path),
                "file_sha256": expected_hash,
                "size_bytes": array_path.stat().st_size,
                "logical_array_sha256": export_record.get("array_sha256"),
                "dtype": "float16",
                "shape_fxyz": list(array.shape),
                "finite": True,
                "embedding_bank_sha256": export["embedding_bank"]["sha256"],
                "checkpoint_sha256": export["candidate_lineage"][
                    "checkpoint_sha256"
                ],
                "geometry": {**geometry, "verified": True},
                "sideexp_export_record_sha256": _canonical_sha256(export_record),
                "sideexp_reproduction_record_sha256": _canonical_sha256(
                    reproduction_record
                ),
            }
        )
        del array
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "status": "verified_complete",
        "candidate_id": candidate_id,
        "source": "accepted_sideexp002_export",
        "candidate_lineage": dict(export["candidate_lineage"]),
        "embedding_bank": dict(export["embedding_bank"]),
        "source_export_manifest": str(export_path),
        "source_export_manifest_sha256": sha256_file(export_path),
        "source_reproduction_manifest": str(reproduction_path),
        "source_reproduction_manifest_sha256": sha256_file(reproduction_path),
        "val80_path": str(val80_path),
        "val80_sha256": val80["sha256"],
        "case_count": len(projected),
        "finding_count": sum(int(record["finding_count"]) for record in projected),
        "dtype": "float16",
        "logical_uncompressed_bytes": sum(
            int(math.prod(record["shape_fxyz"]) * np.dtype("f2").itemsize)
            for record in projected
        ),
        "logical_array_count": len(projected),
        "threshold_probability": 0.5,
        "threshold_logit": 0.0,
        "threshold_mask_equivalence": "exact",
        "cases": projected,
    }
    return manifest, export_path, reproduction_path


def _try_accepted_sideexp_projection(
    protocol: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], Path, Path] | None, str | None]:
    """Convert any unusable legacy export into an explicit local-fallback reason."""

    try:
        return _project_accepted_sideexp_val80(protocol, candidate), None
    except StageContractError as exc:
        return None, str(exc)


def _docker_val80_export_argv(
    protocol: Mapping[str, Any], *, preflight_only: bool
) -> list[str]:
    logits = _require_mapping(protocol, "logits", label="protocol")
    local = _require_mapping(logits, "local_export", label="protocol.logits")
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    val80 = _require_mapping(cohorts, "val80", label="protocol.cohorts")
    val200 = _require_mapping(cohorts, "val200", label="protocol.cohorts")
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    data = _require_mapping(protocol, "data", label="protocol")
    embedding_bank = _require_mapping(
        protocol, "embedding_bank", label="protocol"
    )
    image = _require_string(local, "docker_image", label="protocol.logits.local_export")
    gpu = int(local.get("gpu", 0))
    adapter = _require_string(local, "adapter", label="protocol.logits.local_export")
    controller = _controller_docker_contract()
    runtime_root = _require_string(protocol, "runtime_root", label="protocol")
    expected_runtime = (
        "/mnt/shengdata1/hengjie/experiments/rexgroundingct/aiselfdrive/"
        f"{EXPERIMENT_ID}"
    )
    if runtime_root != expected_runtime:
        raise StageContractError("ASD-000 Docker runtime root drifted")
    cidfile = _docker_cidfile_path(protocol, preflight_only=preflight_only)
    repo = str(REPO_ROOT)
    argv = [
        "docker",
        "run",
        "--rm",
        "--cidfile",
        str(cidfile),
        "--network",
        "none",
        "--label",
        str(controller["lease_label"]),
        "--label",
        str(controller["experiment_label"]),
        "--env",
        f"{_COMMAND_EXPERIMENT_ENV}={controller['experiment_id']}",
        "--env",
        f"{_COMMAND_LEASE_ENV}={controller['lease_id']}",
        "--env",
        f"{_COMMAND_DOCKER_LABEL_ENV}={controller['lease_label']}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--gpus",
        "all" if preflight_only else f"device={gpu}",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--volume",
        f"{repo}:/workspace:ro",
        "--volume",
        f"{repo}:{repo}:ro",
        "--volume",
        "/mnt/shengdata1:/mnt/shengdata1:ro",
        "--volume",
        f"{runtime_root}:{runtime_root}:rw",
        "--volume",
        "/data/hengjie:/data/hengjie:ro",
        "--workdir",
        "/workspace",
        image,
        "python",
        f"/workspace/{adapter}",
        "--candidate-manifest",
        f"/workspace/{_require_string(logits, 'candidate_manifest', label='protocol.logits')}",
        "--candidate-id",
        _require_string(logits, "candidate_id", label="protocol.logits"),
        "--val80-json",
        f"/workspace/{_require_string(val80, 'path', label='protocol.cohorts.val80')}",
        "--val80-sha256",
        _require_string(val80, "sha256", label="protocol.cohorts.val80"),
        "--val200-json",
        f"/workspace/{_require_string(val200, 'path', label='protocol.cohorts.val200')}",
        "--val200-sha256",
        _require_string(val200, "sha256", label="protocol.cohorts.val200"),
        "--runtime-root",
        _require_string(protocol, "runtime_root", label="protocol"),
        "--prediction-root",
        _require_string(baseline, "predictions", label="protocol.baseline"),
        "--segmentation-root",
        _require_string(data, "segmentation_root", label="protocol.data"),
        "--embeddings",
        _require_string(embedding_bank, "path", label="protocol.embedding_bank"),
        "--embeddings-sha256",
        _require_string(
            embedding_bank, "sha256", label="protocol.embedding_bank"
        ),
        "--expected-embedding-labels",
        str(int(embedding_bank.get("expected_labels", 0))),
        "--expected-embedding-dimension",
        str(int(embedding_bank.get("expected_shape", [0, 0])[1])),
        "--embedding-lookup-normalization",
        _require_string(
            embedding_bank,
            "lookup_normalization",
            label="protocol.embedding_bank",
        ),
        "--expected-checkpoint-path",
        _require_string(baseline, "checkpoint", label="protocol.baseline"),
        "--expected-checkpoint-sha256",
        _require_string(
            baseline, "checkpoint_sha256", label="protocol.baseline"
        ),
        "--gpu",
        str(gpu),
        "--expected-device-count",
        str(int(local.get("expected_cuda_device_count", 0))),
        "--expected-device-name",
        _require_string(
            local,
            "expected_cuda_device_name",
            label="protocol.logits.local_export",
        ),
        "--max-durable-storage-gib",
        str(int(local.get("max_durable_storage_gib", 0))),
    ]
    if preflight_only:
        argv.append("--preflight-only")
    return argv


def _docker_launch_contract_proof(
    protocol: Mapping[str, Any], *, preflight_only: bool
) -> dict[str, Any]:
    """Summarize security-relevant facts from the exact Docker argv."""

    argv = _docker_val80_export_argv(
        protocol, preflight_only=preflight_only
    )
    values_after = lambda flag: [  # noqa: E731 - compact argument parser
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == flag
    ]
    volumes = values_after("--volume")
    environments = values_after("--env")
    labels = values_after("--label")
    network_values = values_after("--network")
    cidfiles = values_after("--cidfile")
    runtime_root = _require_string(protocol, "runtime_root", label="protocol")
    expected_rw = f"{runtime_root}:{runtime_root}:rw"
    rw_volumes = [volume for volume in volumes if volume.endswith(":rw")]
    environment_names = sorted(value.split("=", 1)[0] for value in environments)
    required_environment_names = sorted(
        [
            _COMMAND_DOCKER_LABEL_ENV,
            _COMMAND_EXPERIMENT_ENV,
            _COMMAND_LEASE_ENV,
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ]
    )
    cidfile_path = Path(cidfiles[0]) if len(cidfiles) == 1 else Path(".")
    verified = (
        network_values == ["none"]
        and "/mnt/shengdata1:/mnt/shengdata1:ro" in volumes
        and expected_rw in volumes
        and rw_volumes == [expected_rw]
        and all(
            volume.endswith(":ro") or volume == expected_rw
            for volume in volumes
        )
        and environment_names == required_environment_names
        and "HF_HUB_OFFLINE=1" in environments
        and "TRANSFORMERS_OFFLINE=1" in environments
        and len(labels) == 2
        and any(
            value.startswith("rexgroundingct.aiselfdrive.lease=")
            for value in labels
        )
        and f"rexgroundingct.aiselfdrive.experiment={EXPERIMENT_ID}" in labels
        and len(cidfiles) == 1
        and _is_within(cidfile_path, Path(runtime_root) / ".controller")
    )
    if not verified:
        raise StageContractError("nested Docker launch/write-scope contract failed")
    return {
        "network_mode": "none",
        "source_parent_mount": "/mnt/shengdata1:/mnt/shengdata1:ro",
        "only_writable_mount": expected_rw,
        "source_mounts_read_only": True,
        "controller_environment_names": required_environment_names,
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "exact_lease_and_experiment_labels": True,
        "cidfile_below_runtime_controller_root": True,
        "preflight_only": preflight_only,
        "verified": True,
    }


def _local_launch_attestation_path(protocol: Mapping[str, Any]) -> Path:
    runtime_root = Path(_require_string(protocol, "runtime_root", label="protocol"))
    expected = (
        runtime_root
        / "val80_logits_revision_2"
        / "host_launch_attestation.json"
    )
    local = _require_mapping(
        _require_mapping(protocol, "logits", label="protocol"),
        "local_export",
        label="protocol.logits",
    )
    declared = Path(
        _require_string(
            local, "host_launch_attestation", label="protocol.logits.local_export"
        )
    )
    if declared != expected:
        raise StageContractError("host launch-attestation path drifted")
    return expected


def _write_local_launch_attestation(
    protocol: Mapping[str, Any],
    manifest_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a local export to the host wrapper's actual launch contracts."""

    if not manifest_path.is_file():
        raise StageContractError("local export manifest is missing for attestation")
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "status": "verified",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_result_sha256": _canonical_sha256(preflight),
        "preflight_launch_contract": _docker_launch_contract_proof(
            protocol, preflight_only=True
        ),
        "export_launch_contract": _docker_launch_contract_proof(
            protocol, preflight_only=False
        ),
        "verified": True,
    }
    attestation_path = _local_launch_attestation_path(protocol)
    atomic_write_json(attestation_path, payload)
    return {
        **payload,
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256_file(attestation_path),
    }


def _validate_local_launch_attestation(
    protocol: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """Reject legacy local caches that lack host-side mount attestation."""

    attestation_path = _local_launch_attestation_path(protocol)
    attestation = _load_json_mapping(
        attestation_path, label="local Docker launch attestation"
    )
    required = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "status": "verified",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_launch_contract": _docker_launch_contract_proof(
            protocol, preflight_only=True
        ),
        "export_launch_contract": _docker_launch_contract_proof(
            protocol, preflight_only=False
        ),
        "verified": True,
    }
    if not _is_sha256(attestation.get("preflight_result_sha256")) or any(
        attestation.get(key) != value for key, value in required.items()
    ):
        raise StageContractError("local Docker launch attestation drifted")
    return {
        **dict(attestation),
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256_file(attestation_path),
    }


def _load_logit_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["logits"]:
                raise StageContractError(f"logit archive contract drifted: {path}")
            return np.asarray(archive["logits"])
    raise StageContractError(f"unsupported logit array format: {path}")


def _validate_val80_logit_manifest(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    cohorts = _require_mapping(protocol, "cohorts", label="protocol")
    val80 = _require_mapping(cohorts, "val80", label="protocol.cohorts")
    val80_path = _resolve_declared_path(
        _require_string(val80, "path", label="protocol.cohorts.val80")
    )
    val80_payload = _load_json_mapping(val80_path, label="val80 source")
    cases = val80_payload.get("test")
    if not isinstance(cases, list) or len(cases) != 80:
        raise StageContractError("val80 source does not contain exactly 80 cases")
    records = manifest.get("cases")
    if (
        manifest.get("status") != "verified_complete"
        or manifest.get("val80_sha256") != val80["sha256"]
        or manifest.get("case_count") != 80
        or manifest.get("finding_count") != 195
        or manifest.get("threshold_mask_equivalence") != "exact"
        or not isinstance(records, list)
        or len(records) != 80
    ):
        raise StageContractError("val80 logit manifest is incomplete or unverified")
    _validate_logit_embedding_proof(
        protocol, manifest, label="val80 logit manifest"
    )
    _validate_logit_candidate_proof(
        protocol, manifest, label="val80 logit manifest"
    )
    layout = _derive_locked_val80_float16_layout(protocol)
    if (
        manifest.get("logical_uncompressed_bytes")
        != layout["logical_uncompressed_bytes"]
        or manifest.get("logical_array_count") != layout["array_count"]
    ):
        raise StageContractError("val80 logit logical-layout proof drifted")
    if manifest.get("source") == "local_docker_export":
        _validate_runtime_durability_proof(
            protocol, manifest, label="local val80 logit manifest"
        )
        _validate_offline_container_proof(
            manifest, label="local val80 logit manifest"
        )
    baseline = _require_mapping(protocol, "baseline", label="protocol")
    prediction_root = _resolve_declared_path(
        _require_string(baseline, "predictions", label="protocol.baseline")
    )
    total_bytes = 0
    verified_records: list[dict[str, Any]] = []
    for order, (case, record) in enumerate(zip(cases, records, strict=True)):
        if not isinstance(case, Mapping) or not isinstance(record, Mapping):
            raise StageContractError("val80 logit case record is malformed")
        name = _require_string(case, "name", label="val80 case")
        findings = _require_mapping(case, "findings", label=f"val80 {name}")
        if (
            record.get("case_order") != order
            or record.get("case_name") != name
            or record.get("finding_count") != len(findings)
            or record.get("finite") is not True
        ):
            raise StageContractError(f"val80 logit identity/count drift for {name}")
        if (
            record.get("embedding_bank_sha256")
            != _require_mapping(protocol, "embedding_bank", label="protocol").get(
                "sha256"
            )
            or record.get("checkpoint_sha256")
            != _require_mapping(protocol, "baseline", label="protocol").get(
                "checkpoint_sha256"
            )
        ):
            raise StageContractError(f"val80 logit lineage drift for {name}")
        array_path = Path(_require_string(record, "array_path", label=f"logits {name}"))
        if not array_path.is_absolute() or not array_path.is_file():
            raise StageContractError(f"val80 logit array is missing: {array_path}")
        if manifest.get("source") == "local_docker_export" and not _is_within(
            array_path,
            Path(_require_string(protocol, "runtime_root", label="protocol")),
        ):
            raise StageContractError(f"local logit array escapes ASD-000 runtime: {array_path}")
        file_sha256 = record.get("file_sha256")
        if not _is_sha256(file_sha256) or file_sha256 != sha256_file(array_path):
            raise StageContractError(f"logit file hash mismatch for {name}")
        array = _load_logit_array(array_path)
        if array.dtype != np.float16 or array.ndim != 4 or array.shape[0] != len(findings):
            raise StageContractError(f"logit dtype/shape mismatch for {name}: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise StageContractError(f"non-finite logits for {name}")
        prediction = read_nifti_data(prediction_root / name)
        if array.shape != prediction.shape or np.any((array >= 0.0) != (prediction > 0)):
            raise StageContractError(f"logit threshold-mask equivalence failed for {name}")
        geometry = record.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("verified") is not True:
            raise StageContractError(f"logit geometry proof is missing for {name}")
        total_bytes += array_path.stat().st_size
        verified_records.append(dict(record))
        del array, prediction
    if manifest.get("source") == "local_docker_export":
        local_export = _require_mapping(
            _require_mapping(protocol, "logits", label="protocol"),
            "local_export",
            label="protocol.logits",
        )
        storage_cap_gib = local_export.get("max_durable_storage_gib")
        if (
            isinstance(storage_cap_gib, bool)
            or not isinstance(storage_cap_gib, int)
            or storage_cap_gib != _LOCAL_LOGIT_DURABLE_STORAGE_GIB
        ):
            raise StageContractError(
                "local val80 logit durable-storage contract drifted"
            )
        if manifest.get("durable_storage_cap_gib") != storage_cap_gib:
            raise StageContractError(
                "local val80 logit manifest storage cap drifted"
            )
        if total_bytes > storage_cap_gib * _GIB:
            raise StageContractError(
                f"val80 logit storage exceeds {storage_cap_gib} GiB"
            )
    return {
        **dict(manifest),
        "cases": verified_records,
        "total_size_bytes": total_bytes,
        "host_revalidation": {
            "complete": True,
            "case_count": 80,
            "finding_count": 195,
            "finite_array_count": 80,
            "hash_verified_array_count": 80,
            "exact_threshold_mask_case_count": 80,
        },
    }


def _phase_logits(
    protocol: Mapping[str, Any], stage: Mapping[str, Any]
) -> Path:
    phase_started = time.monotonic()
    logits = _require_mapping(protocol, "logits", label="protocol")
    local_export = _validate_local_export_policy(protocol)
    stage_resources = _require_mapping(stage, "resources", label=str(stage.get("id")))
    for key in (
        "resource_class",
        "docker_image",
        "docker_image_id",
        "expected_cuda_device_count",
        "expected_cuda_device_name",
    ):
        if stage_resources.get(key) != local_export.get(key):
            raise StageContractError(
                f"stage/protocol local-export resource contract drifted for {key}"
            )
    storage_cap_gib = local_export.get("max_durable_storage_gib")
    disk_preflight_gib = local_export.get("disk_preflight_free_gib_min")
    uncompressed_bytes = local_export.get(
        "full_layout_float16_uncompressed_bytes"
    )
    uncompressed_gib = local_export.get("full_layout_float16_uncompressed_gib")
    locked_layout = _derive_locked_val80_float16_layout(protocol)
    if (
        isinstance(storage_cap_gib, bool)
        or not isinstance(storage_cap_gib, int)
        or storage_cap_gib != _LOCAL_LOGIT_DURABLE_STORAGE_GIB
        or stage_resources.get("durable_storage_gib_max") != storage_cap_gib
        or disk_preflight_gib != storage_cap_gib
        or stage_resources.get("disk_preflight_free_gib_min")
        != disk_preflight_gib
        or stage_resources.get("training") != "prohibited"
        or uncompressed_bytes != _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        or locked_layout["logical_uncompressed_bytes"]
        != _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        or locked_layout["minimum_case_bytes"]
        != _VAL80_MIN_CASE_FLOAT16_BYTES
        or locked_layout["maximum_case_bytes"]
        != _VAL80_MAX_CASE_FLOAT16_BYTES
        or storage_cap_gib * _GIB <= _VAL80_FLOAT16_UNCOMPRESSED_BYTES
        or isinstance(uncompressed_gib, bool)
        or not isinstance(uncompressed_gib, (int, float))
        or not math.isclose(
            float(uncompressed_gib),
            _VAL80_FLOAT16_UNCOMPRESSED_BYTES / _GIB,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise StageContractError("local-export durable-storage contract drifted")
    if (
        stage_resources.get("gpu_optional") is not True
        or int(stage_resources.get("gpu_count", -1)) != 0
        or int(stage_resources.get("gpu_count_max", -1)) != 1
    ):
        raise StageContractError("conditional GPU resource contract drifted")
    dependency_hashes = _require_mapping(
        logits, "dependency_hashes", label="protocol.logits"
    )
    for declared, expected in dependency_hashes.items():
        if sha256_file(_resolve_declared_path(str(declared))) != expected:
            raise StageContractError(f"logit-export dependency hash drifted: {declared}")
    voxtell_submodule = _validate_voxtell_submodule(protocol)
    embedding_proof = _audit_locked_embedding_bank(protocol)
    candidate_lineage = _validate_candidate_manifest_lineage(protocol)
    audit_path = _resolve_declared_path(
        _require_string(logits, "source_audit", label="protocol.logits")
    )
    expected_audit_sha256 = _require_string(
        logits, "source_audit_sha256", label="protocol.logits"
    )
    if sha256_file(audit_path) != expected_audit_sha256:
        raise StageContractError("sideexp002 logit audit hash drifted")
    audit = _load_json_mapping(audit_path, label="sideexp002 logit audit")
    candidate_id = _require_string(logits, "candidate_id", label="protocol.logits")
    accepted = _select_accepted_logit_candidate(audit, candidate_id)
    accepted_declared = accepted is not None
    accepted_unavailable_reason: str | None = None
    local_root = Path(_require_string(protocol, "runtime_root", label="protocol"))
    local_manifest_path = local_root / "val80_logits_revision_2" / "export_manifest.json"
    export_attempted = False
    preflight: dict[str, Any] | None = None
    reproduction_path: Path | None = None
    if accepted is not None:
        projected, accepted_unavailable_reason = (
            _try_accepted_sideexp_projection(protocol, accepted)
        )
        if projected is None:
            accepted = None
        else:
            manifest, manifest_path, reproduction_path = projected
    if accepted is None:
        source = "local_docker_export"
        manifest_path = local_manifest_path
        local_manifest_usable = False
        if local_manifest_path.is_file():
            try:
                cached_manifest = _load_json_mapping(
                    local_manifest_path,
                    label="cached local val80 logit manifest",
                )
                _validate_logit_embedding_proof(
                    protocol,
                    cached_manifest,
                    label="cached local val80 logit manifest",
                )
                _validate_logit_candidate_proof(
                    protocol,
                    cached_manifest,
                    label="cached local val80 logit manifest",
                )
                _validate_runtime_durability_proof(
                    protocol,
                    cached_manifest,
                    label="cached local val80 logit manifest",
                )
                _validate_offline_container_proof(
                    cached_manifest,
                    label="cached local val80 logit manifest",
                )
                _validate_local_launch_attestation(
                    protocol, local_manifest_path
                )
                _validate_val80_logit_manifest(protocol, cached_manifest)
                local_manifest_usable = True
            except StageContractError:
                local_manifest_usable = False
        if not local_manifest_usable:
            if shutil.which("docker") is None:
                raise StageContractError(
                    "Docker executable is unavailable for required logit export"
                )
            image = _require_string(
                local_export, "docker_image", label="protocol.logits.local_export"
            )
            image_id = _require_string(
                local_export,
                "docker_image_id",
                label="protocol.logits.local_export",
            )
            gpu_deadline = time.monotonic() + 2 * 60 * 60
            image_verification = _verify_docker_image_id(image, image_id)
            local_root.mkdir(parents=True, exist_ok=True)
            preflight_cidfile = _docker_cidfile_path(
                protocol, preflight_only=True
            )
            export_cidfile = _docker_cidfile_path(
                protocol, preflight_only=False
            )
            preflight_cidfile.parent.mkdir(parents=True, exist_ok=True)
            if preflight_cidfile.exists() or export_cidfile.exists():
                raise StageContractError(
                    "nested Docker cidfile already exists; controller reclaim required"
                )
            free_bytes = shutil.disk_usage(local_root).free
            if free_bytes < int(disk_preflight_gib) * _GIB:
                raise StageContractError(
                    f"fewer than {disk_preflight_gib} GiB are free at the "
                    "ASD-000 runtime"
                )
            preflight_timeout = min(3600.0, gpu_deadline - time.monotonic())
            if preflight_timeout <= 0:
                raise StageContractError("two-hour GPU budget expired before preflight")
            preflight_run = _run_checked(
                _docker_val80_export_argv(protocol, preflight_only=True),
                timeout=preflight_timeout,
                capture_output=True,
                docker_cidfile=preflight_cidfile,
                account_gpu_time=True,
            )
            try:
                preflight = json.loads(preflight_run.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise StageContractError("Docker preflight emitted no valid JSON") from exc
            expected_device_count = int(local_export["expected_cuda_device_count"])
            expected_device_name = _require_string(
                local_export,
                "expected_cuda_device_name",
                label="protocol.logits.local_export",
            )
            if (
                preflight.get("status") != "passed"
                or preflight.get("cuda_inventory_exact") is not True
                or preflight.get("visible_cuda_device_count")
                != expected_device_count
                or preflight.get("visible_cuda_device_names")
                != [expected_device_name] * expected_device_count
            ):
                raise StageContractError("Docker GPU/checkpoint/one-case preflight failed")
            _validate_logit_embedding_proof(
                protocol, preflight, label="Docker preflight"
            )
            _validate_logit_candidate_proof(
                protocol, preflight, label="Docker preflight"
            )
            _validate_runtime_durability_proof(
                protocol, preflight, label="Docker preflight"
            )
            _validate_offline_container_proof(preflight, label="Docker preflight")
            preflight["docker_image"] = image_verification
            export_timeout = gpu_deadline - time.monotonic()
            if export_timeout <= 0:
                raise StageContractError("two-hour GPU budget expired after preflight")
            _run_checked(
                _docker_val80_export_argv(protocol, preflight_only=False),
                timeout=export_timeout,
                capture_output=False,
                docker_cidfile=export_cidfile,
                account_gpu_time=True,
            )
            assert preflight is not None
            _write_local_launch_attestation(
                protocol, local_manifest_path, preflight
            )
            if _RESOURCE_GPU_SECONDS > 2 * 60 * 60:
                raise StageContractError("two-hour GPU budget exceeded")
            export_attempted = True
        manifest = _load_json_mapping(
            local_manifest_path, label="local val80 logit manifest"
        )
    else:
        source = "accepted_sideexp002_export"
    verified = _validate_val80_logit_manifest(protocol, manifest)
    launch_attestation = (
        _validate_local_launch_attestation(protocol, manifest_path)
        if source == "local_docker_export"
        else None
    )
    docker_contract = (
        _docker_launch_contract_proof(protocol, preflight_only=False)
        if source == "local_docker_export"
        else None
    )
    result_path = RESULTS_ROOT / "revision_2_logit_source.json"
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "producer_stage_id": str(stage["id"]),
        "status": "verified_complete",
        "source": source,
        "source_audit_path": str(audit_path),
        "source_audit_sha256": sha256_file(audit_path),
        "candidate_id": candidate_id,
        "accepted_sideexp002_declared": accepted_declared,
        "accepted_sideexp002_available": accepted is not None,
        "accepted_sideexp002_unavailable_reason": accepted_unavailable_reason,
        "export_attempted": export_attempted,
        "preflight": preflight,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reproduction_manifest_path": (
            str(reproduction_path) if reproduction_path is not None else None
        ),
        "reproduction_manifest_sha256": (
            sha256_file(reproduction_path)
            if reproduction_path is not None
            else None
        ),
        "manifest": verified,
        "logical_float16_layout": locked_layout,
        "embedding_bank": embedding_proof,
        "candidate_lineage": candidate_lineage,
        "voxtell_submodule": voxtell_submodule,
        "local_docker_contract": docker_contract,
        "local_launch_attestation": launch_attestation,
    }
    resources = {
        "gpu_hours": float(_RESOURCE_GPU_SECONDS / 3600.0),
        "cpu_hours": float((time.monotonic() - phase_started) / 3600.0),
        "storage_gib": (
            float(verified["total_size_bytes"] / 1024**3)
            if source == "local_docker_export"
            else 0.0
        ),
    }
    result["resources"] = resources
    atomic_write_json(result_path, result)
    return _write_passed_evidence(
        stage,
        summary="Verified complete val80 logits with exact locked-mask equivalence.",
        metrics={"logit_source": result, "resources": resources},
    )


@dataclass(frozen=True)
class _AtlasCaseSpec:
    case_name: str
    prediction_header: NiftiHeader
    segmentation_header: NiftiHeader
    anatomy_header: NiftiHeader
    logit_path: Path


class _AtlasArrayCache:
    """Keep at most one val80 case's four source arrays resident."""

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
            if source == "logits":
                self._arrays[source] = _load_logit_array(spec.logit_path)
                return self._arrays[source]
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
            # Threshold curves down to p=0.05 require every score voxel.  A
            # threshold-0.5 mask/GT bounding box would silently discard
            # low-confidence false positives, so revision 2 uses full FXYZ.
            spatial_shape = self.specs[case_name].anatomy_header.shape
            slices = tuple(slice(0, stop) for stop in spatial_shape)
            origin = (0, 0, 0)
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
        certified_lobe_ids: tuple[int, ...] | None,
        all_lobe_ids: tuple[int, ...],
        certify_outside_lung: bool,
    ) -> dict[str, int]:
        positive = self.full(case_name, "segmentation")[finding_index] != 0
        total = int(positive.size)
        positive_count = int(np.count_nonzero(positive))
        if certified_lobe_ids is None:
            certified_count = 0
        else:
            anatomy = self.full(case_name, "anatomy")
            certified = np.isin(anatomy, certified_lobe_ids)
            if certify_outside_lung:
                certified |= ~np.isin(anatomy, all_lobe_ids)
            certified_count = int(np.count_nonzero(certified & (~positive)))
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
        elif self.mode == "logits":
            result = self.cache.full(self.case_name, "logits")[
                (self.finding_index, *slices)
            ]
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
        "laterality_only_outside_lung_policy": "unknown_never_certified_negative",
    }
    if dict(certification) != expected_certification:
        raise StageContractError("prompt certification policy drifted")
    return lobe_ids, expected_certification


def _prompt_lobe_certification(
    parsed: Any, lobe_ids: Mapping[str, int]
) -> dict[str, Any]:
    """Build conservative prompt partitions without over-certifying laterality."""

    target_ids: tuple[int, ...] | None = None
    ipsilateral_ids: tuple[int, ...] = ()
    contralateral_ids: tuple[int, ...] = ()
    certify_outside_lung = False
    target = parsed.restrictive_target or {}
    if parsed.group == "exact_lobe":
        lobe = target.get("lobe")
        if lobe not in lobe_ids:
            raise StageContractError(f"parser emitted unknown lobe {lobe!r}")
        target_ids = (int(lobe_ids[lobe]),)
        side = str(lobe).split("_", 1)[0]
        ipsilateral_ids = tuple(
            int(value)
            for key, value in lobe_ids.items()
            if key.startswith(f"{side}_") and int(value) not in target_ids
        )
        other_side = "left" if side == "right" else "right"
        contralateral_ids = tuple(
            int(value)
            for key, value in lobe_ids.items()
            if key.startswith(f"{other_side}_")
        )
        certify_outside_lung = True
    elif parsed.group == "laterality_only":
        side = target.get("laterality")
        if side not in {"left", "right"}:
            raise StageContractError("laterality-only parser output lacks a side")
        target_ids = tuple(
            int(value)
            for key, value in lobe_ids.items()
            if key.startswith(f"{side}_")
        )
        other_side = "left" if side == "right" else "right"
        contralateral_ids = tuple(
            int(value)
            for key, value in lobe_ids.items()
            if key.startswith(f"{other_side}_")
        )
        # Side-only findings may extend through pleura or chest wall. Only the
        # opposite lung is certified incompatible; extrathoracic tissue stays
        # unknown and cannot become a synthetic negative.
        certify_outside_lung = False
    return {
        "target_lobe_ids": target_ids,
        "ipsilateral_nontarget_lobe_ids": ipsilateral_ids,
        "contralateral_lobe_ids": contralateral_ids,
        "certified_lobe_ids": tuple(sorted((*ipsilateral_ids, *contralateral_ids))),
        "certify_outside_lung": certify_outside_lung,
    }


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

    logit_source_path = RESULTS_ROOT / "revision_2_logit_source.json"
    logit_source = _load_json_mapping(
        logit_source_path, label="revision-2 val80 logit source"
    )
    manifest_path = Path(
        _require_string(logit_source, "manifest_path", label="val80 logit source")
    )
    if (
        logit_source.get("status") != "verified_complete"
        or not manifest_path.is_file()
        or logit_source.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise StageContractError("revision-2 val80 logit source lock drifted")
    reproduction_path_value = logit_source.get("reproduction_manifest_path")
    if reproduction_path_value is not None:
        reproduction_path = Path(str(reproduction_path_value))
        if (
            not reproduction_path.is_file()
            or logit_source.get("reproduction_manifest_sha256")
            != sha256_file(reproduction_path)
        ):
            raise StageContractError("accepted sideexp002 reproduction lock drifted")
    manifest_payload = _require_mapping(
        logit_source, "manifest", label="val80 logit source"
    )
    verified_logit_manifest = _validate_val80_logit_manifest(
        protocol, manifest_payload
    )
    logit_records = verified_logit_manifest.get("cases")
    if not isinstance(logit_records, list) or len(logit_records) != 80:
        raise StageContractError("verified val80 logit records are incomplete")
    logit_by_case = {
        record.get("case_name"): record
        for record in logit_records
        if isinstance(record, Mapping)
    }
    if list(logit_by_case) != case_names:
        raise StageContractError("verified val80 logit order/identity drifted")

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
        logit_record = logit_by_case.get(case_name)
        if not isinstance(logit_record, Mapping):
            raise StageContractError(f"verified logits omit val80 case {case_name}")
        logit_path = Path(
            _require_string(logit_record, "array_path", label=f"logits {case_name}")
        )
        specs[case_name] = _AtlasCaseSpec(
            case_name=case_name,
            prediction_header=prediction_header,
            segmentation_header=segmentation_header,
            anatomy_header=anatomy_header,
            logit_path=logit_path,
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
            spatial = _prompt_lobe_certification(parsed, lobe_ids)
            target_ids = spatial["target_lobe_ids"]
            ipsilateral_ids = spatial["ipsilateral_nontarget_lobe_ids"]
            contralateral_ids = spatial["contralateral_lobe_ids"]
            record: dict[str, Any] = {
                "finding_id": str(finding_id),
                "prediction_mask": _LazyFindingArray(
                    cache, case_name, finding_index, "prediction"
                ),
                "known_positive_mask": _LazyFindingArray(
                    cache, case_name, finding_index, "positive"
                ),
                "affine": _LazyFindingAffine(cache, case_name, finding_index),
                "logits": _LazyFindingArray(
                    cache, case_name, finding_index, "logits"
                ),
                "logits_verified": True,
                "score_kind": "logit",
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
                        "anatomy_compatible_mask": _LazyFindingArray(
                            cache, case_name, finding_index, "lobes", target_ids
                        ),
                    }
                )
                if spatial["certify_outside_lung"]:
                    record["outside_lung_mask"] = _LazyFindingArray(
                        cache,
                        case_name,
                        finding_index,
                        "outside_lung",
                        all_lobe_ids=all_lobes,
                        subtract_positive=True,
                    )
            case_findings.append(record)
            supervision[(case_name, str(finding_id))] = {
                "status": status,
                "parser_group": parsed.group,
                "parser_reason_codes": list(parsed.reason_codes),
                "target_lobe_ids": target_ids,
                "certified_lobe_ids": spatial["certified_lobe_ids"],
                "certify_outside_lung": spatial["certify_outside_lung"],
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
        "all_lobe_ids": list(all_lobes),
        "logit_source_path": str(logit_source_path),
        "logit_source_sha256": sha256_file(logit_source_path),
        "verified_logit_records_sha256": _canonical_sha256(logit_records),
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
    all_lobe_ids: tuple[int, ...],
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
            certified_lobe_ids=(
                contract["certified_lobe_ids"]
                if contract["status"] == "available"
                else None
            ),
            all_lobe_ids=all_lobe_ids,
            certify_outside_lung=bool(contract["certify_outside_lung"]),
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
    summary["metric_observation_counts"] = {
        key: sum(row.get(key) is not None for row in rows)
        for key in bootstrap_keys
    }
    total_values = {
        key: [int(row[key]) for row in rows if row.get(key) is not None]
        for key in totals_keys
    }
    summary["metric_totals"] = {
        key: int(sum(values)) if values else None
        for key, values in total_values.items()
    }
    summary["metric_total_observation_counts"] = {
        key: len(values) for key, values in total_values.items()
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
        "verified_complete_val80_logits_required_with_threshold_curves"
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
        all_lobe_ids=tuple(int(value) for value in preparation["all_lobe_ids"]),
        bootstrap_seed=int(atlas["bootstrap_seed"]),
        bootstrap_draws=int(atlas["bootstrap_draws"]),
    )
    report["prompt_certification_policy"] = dict(
        preparation["prompt_certification"]
    )
    report["val120_files_opened"] = int(preparation["val120_files_opened"])
    if report.get("case_count") != int(atlas["expected_cases"]) or report.get(
        "finding_count"
    ) != int(atlas["expected_findings"]):
        raise StageContractError("error atlas completeness check failed")
    if report.get("raw_mask_reconstruction_exact") is not True:
        raise StageContractError("error atlas raw-mask reconstruction is not exact")
    if (
        report.get("logit_status", {}).get("verified_complete") is not True
        or report.get("logit_status", {}).get("threshold_curves_included") is not True
    ):
        raise StageContractError("verified complete threshold curves are missing")
    payload_sha256 = _canonical_sha256(report)
    result_json = RESULTS_ROOT / "revision_2_error_atlas.json"
    result_csv = RESULTS_ROOT / "revision_2_error_atlas.csv"
    summary_md = RESULTS_ROOT / "revision_2_error_atlas_summary.md"
    rows = report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise StageContractError("error atlas returned no finding rows")
    atomic_write_json(result_json, report)
    atomic_write_csv(result_csv, rows)
    atomic_write_json(
        SHARED_MANIFEST_ROOT / "asd000_error_atlas_contract_revision_2.json",
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
        "# ASD-000 revision-2 val80 error atlas\n\n"
        f"- Cases: {report.get('case_count')}\n"
        f"- Findings: {report.get('finding_count')}\n"
        f"- Mean raw-mask Dice: {means.get('raw_mask_dice')}\n"
        f"- Mean raw-mask hit: {means.get('raw_mask_hit')}\n"
        f"- Spatial supervision available: "
        f"{report['spatial_supervision_availability']['available_finding_count']}\n"
        f"- Verified logits: true\n"
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
        "selfcheck_revision_2_implementation": RESULTS_ROOT
        / "revision_2_selfcheck_evidence.json",
        "lock_local_lineage": RESULTS_ROOT / "revision_2_lineage_evidence.json",
        "lock_cohorts_and_prompt_ontology": RESULTS_ROOT
        / "revision_2_cohort_evidence.json",
        "verify_or_export_val80_logits": RESULTS_ROOT
        / "revision_2_logit_evidence.json",
        "build_val80_error_atlas": RESULTS_ROOT
        / "revision_2_atlas_evidence.json",
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
    locked_evaluation_contract = _require_mapping(
        lineage_metrics, "evaluation_contract", label="lineage metrics"
    )
    if dict(locked_evaluation_contract) != _build_evaluation_contract(protocol):
        raise StageContractError("locked evaluation contract drifted")
    recomputation = _require_mapping(
        _require_mapping(lineage, "baseline", label="lineage"),
        "metric_recomputation",
        label="lineage.baseline",
    )
    configured_baseline = _require_mapping(protocol, "baseline", label="protocol")
    if (
        recomputation.get("metric_source")
        != "stored_prediction_masks_plus_released_ground_truth"
        or recomputation.get("verified") is not True
        or recomputation.get("findings") != 381
        or recomputation.get("hits") != 288
        or abs(
            float(recomputation.get("dice_per_finding"))
            - float(configured_baseline["expected_dice"])
        )
        > float(configured_baseline["dice_tolerance"])
        or recomputation.get("agrees_with_stored_summary") is not True
    ):
        raise StageContractError("direct prediction/GT baseline gate failed")
    cohort_metrics = _require_mapping(
        evidence_payloads["lock_cohorts_and_prompt_ontology"],
        "metrics",
        label="cohort evidence",
    )
    cohorts = _require_mapping(cohort_metrics, "cohorts", label="cohort metrics")
    partition = _require_mapping(cohorts, "partition", label="cohort metrics")
    parent_case_names = partition.get("parent_case_names")
    development_case_names = partition.get("development_case_names")
    replication_case_names = partition.get("internal_replication_case_names")
    if (
        partition.get("parent_count") != 200
        or partition.get("development_count") != 80
        or partition.get("internal_replication_count") != 120
        or partition.get("disjoint") is not True
        or partition.get("exhaustive") is not True
        or not isinstance(parent_case_names, list)
        or not isinstance(development_case_names, list)
        or not isinstance(replication_case_names, list)
        or set(development_case_names) & set(replication_case_names)
        or set(development_case_names) | set(replication_case_names)
        != set(parent_case_names)
    ):
        raise StageContractError("cohort partition gate failed")
    val120 = _require_mapping(cohorts, "val120", label="cohort metrics")
    if (
        val120.get("role")
        != "historically_exposed_internal_held_out_replication"
        or val120.get("independence_status")
        != "not_independent_and_not_blinded"
        or val120.get("identity_only") is not True
    ):
        raise StageContractError("val120 role/independence contract failed")
    geometry = _require_mapping(cohort_metrics, "geometry", label="cohort metrics")
    prompt_ontology = _require_mapping(
        cohort_metrics, "prompt_ontology", label="cohort metrics"
    )
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
    logit_metrics = _require_mapping(
        evidence_payloads["verify_or_export_val80_logits"],
        "metrics",
        label="logit evidence",
    )
    logit_source = _require_mapping(
        logit_metrics, "logit_source", label="logit evidence"
    )
    verified_logit_manifest = _require_mapping(
        logit_source, "manifest", label="logit source"
    )
    host_revalidation = _require_mapping(
        verified_logit_manifest,
        "host_revalidation",
        label="verified logit manifest",
    )
    if (
        logit_source.get("status") != "verified_complete"
        or verified_logit_manifest.get("case_count") != 80
        or verified_logit_manifest.get("finding_count") != 195
        or verified_logit_manifest.get("threshold_mask_equivalence") != "exact"
        or host_revalidation.get("complete") is not True
        or host_revalidation.get("hash_verified_array_count") != 80
        or host_revalidation.get("exact_threshold_mask_case_count") != 80
    ):
        raise StageContractError("required val80 logit gate failed")
    observed_layout = _require_mapping(
        logit_source, "logical_float16_layout", label="logit source"
    )
    locked_layout = _derive_locked_val80_float16_layout(protocol)
    if dict(observed_layout) != locked_layout:
        raise StageContractError("locked val80 logical-layout evidence drifted")
    observed_embedding_bank = _require_mapping(
        logit_source, "embedding_bank", label="logit source"
    )
    locked_embedding_bank = _audit_locked_embedding_bank(protocol)
    if dict(observed_embedding_bank) != locked_embedding_bank:
        raise StageContractError("locked embedding-bank evidence drifted")
    observed_candidate = _require_mapping(
        logit_source, "candidate_lineage", label="logit source"
    )
    locked_candidate = _validate_candidate_manifest_lineage(protocol)
    if dict(observed_candidate) != locked_candidate:
        raise StageContractError("locked candidate-lineage evidence drifted")
    _validate_logit_embedding_proof(
        protocol, verified_logit_manifest, label="closeout logit manifest"
    )
    _validate_logit_candidate_proof(
        protocol, verified_logit_manifest, label="closeout logit manifest"
    )
    local_export_policy = _validate_local_export_policy(protocol)
    logit_source_kind = logit_source.get("source")
    if logit_source_kind not in {
        "local_docker_export",
        "accepted_sideexp002_export",
    }:
        raise StageContractError("logit source kind drifted")
    if logit_source_kind == "local_docker_export":
        _validate_runtime_durability_proof(
            protocol,
            verified_logit_manifest,
            label="closeout local logit manifest",
        )
        _validate_offline_container_proof(
            verified_logit_manifest,
            label="closeout local logit manifest",
        )
        docker_contract = _require_mapping(
            logit_source, "local_docker_contract", label="logit source"
        )
        expected_docker_contract = _docker_launch_contract_proof(
            protocol, preflight_only=False
        )
        if dict(docker_contract) != expected_docker_contract:
            raise StageContractError("recorded nested Docker contract drifted")
        recorded_attestation = _require_mapping(
            logit_source, "local_launch_attestation", label="logit source"
        )
        current_attestation = _validate_local_launch_attestation(
            protocol,
            Path(
                _require_string(
                    logit_source, "manifest_path", label="logit source"
                )
            ),
        )
        if dict(recorded_attestation) != current_attestation:
            raise StageContractError("local launch attestation evidence drifted")
    else:
        if logit_source.get("local_docker_contract") is not None:
            raise StageContractError("accepted logit source claims a local Docker run")
        if logit_source.get("local_launch_attestation") is not None:
            raise StageContractError(
                "accepted logit source claims a local launch attestation"
            )

    selfcheck_metrics = _require_mapping(
        evidence_payloads["selfcheck_revision_2_implementation"],
        "metrics",
        label="selfcheck evidence",
    )
    selfcheck = _require_mapping(
        selfcheck_metrics, "selfcheck", label="selfcheck metrics"
    )
    selfcheck_test_ids = selfcheck.get("passed_test_ids")
    if not isinstance(selfcheck_test_ids, list) or not all(
        isinstance(value, str) for value in selfcheck_test_ids
    ):
        raise StageContractError("selfcheck test identity evidence is missing")

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
        or _require_mapping(atlas, "logit_status", label="atlas").get(
            "verified_complete"
        )
        is not True
        or _require_mapping(atlas, "logit_status", label="atlas").get(
            "threshold_curves_included"
        )
        is not True
    ):
        raise StageContractError("error-atlas completeness gate failed")
    if atlas_metrics.get("atlas_payload_sha256") != _canonical_sha256(atlas):
        raise StageContractError("error-atlas canonical payload hash drifted")

    atlas_summary = _require_mapping(atlas, "summary", label="atlas")
    observation_counts = _require_mapping(
        atlas_summary, "metric_observation_counts", label="atlas.summary"
    )
    total_observation_counts = _require_mapping(
        atlas_summary, "metric_total_observation_counts", label="atlas.summary"
    )
    if not observation_counts or not total_observation_counts:
        raise StageContractError("atlas null/observation-count contract is missing")
    spatial_availability = _require_mapping(
        atlas, "spatial_supervision_availability", label="atlas"
    )
    certification_policy = _require_mapping(
        atlas, "prompt_certification_policy", label="atlas"
    )
    metric_means_for_gates = _require_mapping(
        atlas_summary, "metric_means", label="atlas.summary"
    )
    metric_totals_for_gates = _require_mapping(
        atlas_summary, "metric_totals", label="atlas.summary"
    )

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
    claims_path = EXPERIMENT_ROOT / "claims.yaml"
    claims = _load_mapping(claims_path, label="claims")
    claim_rows = claims.get("claims")
    if not isinstance(claim_rows, list):
        raise StageContractError("claims rows are malformed")
    declared_gate_ids = {
        str(value) for value in plan.get("gate_ids", [])
    }
    for plan_stage in plan.get("stages", []):
        if isinstance(plan_stage, Mapping):
            declared_gate_ids.update(_acceptance_check_ids(plan_stage))
    expected_gate_ids = {
        "baseline_dice_recomputed_from_prediction_and_gt_within_0_000005",
        "baseline_hits_recomputed_from_prediction_and_gt_equal_288_of_381",
        "prompt_parser_safety_checks_pass",
        "val80_error_atlas_complete",
        "noninterventional_scope_enforced",
        "independent_confirmation_not_claimed_without_external_custodied_data",
        "val80_val120_disjoint_exhaustive_and_roles_honest",
        "geometry_and_orientation_checks_pass",
        "unavailable_spatial_metrics_are_null_with_observation_counts",
        "local_logit_storage_capacity_covers_locked_full_layout_float16",
        "locked_val80_layout_is_derived_from_80_prediction_headers",
        "explicit_embedding_bank_covers_all_locked_prompts_without_qwen_fallback",
        "candidate_checkpoint_matches_locked_baseline",
        "accepted_logit_source_without_exact_bank_proof_forces_local_export",
        "local_docker_runtime_write_scope_and_offline_contract_pass",
        "external_runtime_atomic_probe_passes_before_local_gpu_inference",
    }
    if not expected_gate_ids <= declared_gate_ids:
        raise StageContractError("revision-2 claim gate identifiers drifted")
    accepted_declared = logit_source.get("accepted_sideexp002_declared") is True
    accepted_available = (
        logit_source.get("accepted_sideexp002_available") is True
    )
    accepted_unavailable_reason = logit_source.get(
        "accepted_sideexp002_unavailable_reason"
    )
    accepted_fallback_tested = any(
        value.endswith(
            "AcceptedLogitSourceTests."
            "test_missing_embedding_proof_forces_local_fallback"
        )
        for value in selfcheck_test_ids
    )
    runtime_probe_tested = any(
        value.endswith(
            "RuntimeDurabilityTests."
            "test_probe_performs_durable_round_trip_and_leaves_no_file"
        )
        for value in selfcheck_test_ids
    )
    docker_scope_tested = any(
        value.endswith(
            "DockerFallbackContractTests."
            "test_fallback_is_an_argument_array_scoped_to_asd000_runtime"
        )
        for value in selfcheck_test_ids
    )
    parser_safety_tested = all(
        any(value.endswith(suffix) for value in selfcheck_test_ids)
        for suffix in (
            "ConservativeParserTests.test_negation_and_ambiguity_are_unknown",
            "ConservativeParserTests."
            "test_bilateral_multifocal_and_diffuse_are_nonrestrictive",
            "ConservativeParserTests."
            "test_conflicting_literal_spans_are_unknown",
        )
    )
    tri_state_safety_tested = all(
        any(value.endswith(suffix) for value in selfcheck_test_ids)
        for suffix in (
            "ConservativeSupervisionTests."
            "test_prompt_partitions_keep_compatible_predictions_unknown",
            "RunnerContractTests."
            "test_laterality_only_never_certifies_outside_lung",
        )
    )
    prompt_payload = {
        key: value
        for key, value in prompt_ontology.items()
        if key != "payload_sha256"
    }
    parser_contract_passed = (
        prompt_ontology.get("default_group") == "unknown"
        and prompt_ontology.get("restrictive_target_policy")
        == "literal_unambiguous_nonnegated_unilateral_nondiffuse_matches_only"
        and prompt_ontology.get("unsafe_precedence")
        == [
            "negation",
            "ambiguity",
            "bilateral_or_multifocal",
            "diffuse",
            "conflicting_locations",
        ]
        and prompt_ontology.get("payload_sha256")
        == _canonical_sha256(prompt_payload)
        and parser_safety_tested
    )
    null_observation_contract_passed = (
        all(
            int(count) > 0 or metric_means_for_gates.get(key) is None
            for key, count in observation_counts.items()
        )
        and all(
            int(count) > 0 or metric_totals_for_gates.get(key) is None
            for key, count in total_observation_counts.items()
        )
        and spatial_availability.get("unavailable_metrics_are_null") is True
        and spatial_availability.get("compatible_unlabeled_policy") == "unknown"
        and certification_policy.get("laterality_only_outside_lung_policy")
        == "unknown_never_certified_negative"
        and certification_policy.get("unsupported_groups")
        == "unknown_without_certified_negative_regions"
        and atlas.get("val120_files_opened") == 0
        and tri_state_safety_tested
    )
    gate_evidence = {
        "baseline_dice_recomputed_from_prediction_and_gt_within_0_000005": True,
        "baseline_hits_recomputed_from_prediction_and_gt_equal_288_of_381": True,
        "prompt_parser_safety_checks_pass": parser_contract_passed,
        "val80_error_atlas_complete": True,
        "noninterventional_scope_enforced": (
            _require_mapping(plan, "resources", label="plan").get("training")
            == "prohibited"
        ),
        "independent_confirmation_not_claimed_without_external_custodied_data": (
            _require_mapping(plan, "evaluation", label="plan").get(
                "independent_confirmation_cohort"
            )
            == "new_externally_unseen_custodied_data_required"
        ),
        "val80_val120_disjoint_exhaustive_and_roles_honest": (
            partition.get("disjoint") is True
            and partition.get("exhaustive") is True
            and val120.get("identity_only") is True
            and val120.get("independence_status")
            == "not_independent_and_not_blinded"
        ),
        "geometry_and_orientation_checks_pass": (
            geometry.get("geometry_comparison") == "exact"
            and geometry.get("val120_output_access")
            == "metadata_and_existence_only_no_label_bytes"
            and _require_mapping(geometry, "fast", label="geometry").get(
                "val120_output_byte_read_count"
            )
            == 0
            and _require_mapping(geometry, "highres", label="geometry").get(
                "val120_output_byte_read_count"
            )
            == 0
        ),
        "unavailable_spatial_metrics_are_null_with_observation_counts": (
            null_observation_contract_passed
        ),
        "local_logit_storage_capacity_covers_locked_full_layout_float16": (
            local_export_policy.get("max_durable_storage_gib")
            == _LOCAL_LOGIT_DURABLE_STORAGE_GIB
            and locked_layout["logical_uncompressed_bytes"]
            == _VAL80_FLOAT16_UNCOMPRESSED_BYTES
            and locked_layout["logical_uncompressed_bytes"]
            < _LOCAL_LOGIT_DURABLE_STORAGE_GIB * _GIB
        ),
        "locked_val80_layout_is_derived_from_80_prediction_headers": (
            locked_layout["verified"] is True
            and locked_layout["case_count"] == 80
            and locked_layout["finding_count"] == 195
            and locked_layout["array_count"] == 80
            and locked_layout["minimum_case_bytes"]
            == _VAL80_MIN_CASE_FLOAT16_BYTES
            and locked_layout["maximum_case_bytes"]
            == _VAL80_MAX_CASE_FLOAT16_BYTES
        ),
        "explicit_embedding_bank_covers_all_locked_prompts_without_qwen_fallback": (
            locked_embedding_bank["sha256"]
            == "a7351992fa57e019acb485f6b1f37045e7efcd879118cbac972b1b7ed94d5575"
            and locked_embedding_bank["shape"] == [6467, 2560]
            and locked_embedding_bank["lookup_normalization"]
            == "python_str_lower"
            and locked_embedding_bank["coverage"]["val80"]
            ["lowercase_covered_occurrences"]
            == 195
            and locked_embedding_bank["coverage"]["val200"]
            ["lowercase_covered_occurrences"]
            == 381
            and locked_embedding_bank["coverage"]["val80"]
            ["exact_case_covered_occurrences"]
            == 7
            and locked_embedding_bank["coverage"]["val200"]
            ["exact_case_covered_occurrences"]
            == 14
            and locked_embedding_bank["qwen_fallback_permitted"] is False
        ),
        "candidate_checkpoint_matches_locked_baseline": (
            locked_candidate["checkpoint_path"]
            == configured_baseline["checkpoint"]
            and locked_candidate["checkpoint_sha256"]
            == configured_baseline["checkpoint_sha256"]
            and locked_candidate["verified"] is True
        ),
        "accepted_logit_source_without_exact_bank_proof_forces_local_export": (
            accepted_fallback_tested
            and (
                (
                    logit_source_kind == "accepted_sideexp002_export"
                    and accepted_declared
                    and accepted_available
                )
                or (
                    logit_source_kind == "local_docker_export"
                    and not accepted_available
                    and (
                        not accepted_declared
                        or isinstance(accepted_unavailable_reason, str)
                        and bool(accepted_unavailable_reason.strip())
                    )
                )
            )
        ),
        "local_docker_runtime_write_scope_and_offline_contract_pass": (
            docker_scope_tested
            and (
                logit_source_kind == "accepted_sideexp002_export"
                or _require_mapping(
                    logit_source, "local_docker_contract", label="logit source"
                ).get("verified")
                is True
            )
        ),
        "external_runtime_atomic_probe_passes_before_local_gpu_inference": (
            runtime_probe_tested
            and (
                logit_source_kind == "accepted_sideexp002_export"
                or _require_mapping(
                    verified_logit_manifest,
                    "runtime_durability_probe",
                    label="verified logit manifest",
                ).get("status")
                == "passed"
            )
        ),
    }
    if not all(gate_evidence.values()):
        raise StageContractError("one or more revision-2 claim gates failed")
    decisions = {
        "canonical_lineage_reproducible": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_lineage_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_logit_evidence.json",
            ],
            "rationale": "Stored prediction masks and released GT directly reproduced the locked Dice and 288/381 hit count; the evaluator summary was only a cross-check. Direct parsing of all 80 locked val80 prediction headers gives a 37,666,291,712-byte (35.0794677734375-GiB) float16 layout; the bounded 64-GiB external-runtime cap retains 28.9205322265625 GiB of headroom.",
        },
        "spatial_error_taxonomy_measurable": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_cohort_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_logit_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_atlas_evidence.json",
            ],
            "rationale": "The conservative parser, verified full-volume logits, and exact-geometry val80 atlas passed with explicit per-finding supervision availability and observation counts.",
        },
        "noninterventional_scope_boundary_enforced": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_selfcheck_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_atlas_evidence.json",
            ],
            "rationale": "ASD-000 performs provenance locking and diagnostics only; no training intervention or causal effect estimate is present.",
        },
        "baseline_continuity_without_independent_confirmation": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_lineage_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_cohort_evidence.json",
            ],
            "rationale": "The direct recomputation supports historical baseline continuity only. Val120 is explicitly historically exposed internal replication; independent confirmation requires new externally unseen custodied data.",
        },
        "tri_state_policy_preserved": {
            "evidence_refs": [
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_selfcheck_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_cohort_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_logit_evidence.json",
                "experiments_aiselfdrive/experiments/asd_000_evidence_lock_error_atlas/results/revision_2_atlas_evidence.json",
            ],
            "rationale": "Known-positive, certified-negative, and compatible or unsupported unknown regions remain separately counted; laterality-only outside-lung tissue stays unknown.",
        },
    }
    if {str(claim.get("id")) for claim in claim_rows if isinstance(claim, Mapping)} != set(
        decisions
    ):
        raise StageContractError("predeclared claim identities drifted")
    expected_claim_types = {
        "canonical_lineage_reproducible": "feasibility",
        "spatial_error_taxonomy_measurable": "mechanistic",
        "noninterventional_scope_boundary_enforced": "causal",
        "baseline_continuity_without_independent_confirmation": "performance",
        "tri_state_policy_preserved": "safety",
    }
    observed_claim_types = {
        str(claim.get("id")): claim.get("type")
        for claim in claim_rows
        if isinstance(claim, Mapping)
    }
    if observed_claim_types != expected_claim_types:
        raise StageContractError("predeclared claim type mapping drifted")
    decided_at = _utc_now()
    for claim in claim_rows:
        if not isinstance(claim, dict):
            raise StageContractError("claims contains a malformed row")
        decision = decisions[str(claim["id"])]
        gate_refs = claim.get("gate_refs")
        if not isinstance(gate_refs, list) or not gate_refs:
            raise StageContractError(f"claim {claim['id']} has no gate references")
        unresolved = [
            str(gate_ref)
            for gate_ref in gate_refs
            if str(gate_ref) not in declared_gate_ids
            or gate_evidence.get(str(gate_ref)) is not True
        ]
        if unresolved:
            raise StageContractError(
                f"claim {claim['id']} has unresolved gates: {unresolved}"
            )
        claim["verdict"] = "supported"
        claim["evidence_refs"] = decision["evidence_refs"]
        claim["decided_at"] = decided_at
        claim["rationale"] = decision["rationale"]
    summary_path = RESULTS_ROOT / "revision_2_summary.md"
    means = _require_mapping(
        _require_mapping(atlas, "summary", label="atlas"),
        "metric_means",
        label="atlas summary",
    )
    summary_text = (
        "# ASD-000 revision-2 evidence-lock closeout\n\n"
        "Outcome: GO.\n\n"
        f"- Required artifacts verified: {len(verified_ids)}\n"
        f"- Val80 cases/findings: {atlas['case_count']}/{atlas['finding_count']}\n"
        f"- Val80 mean raw-mask Dice: {means.get('raw_mask_dice')}\n"
        f"- Val80 mean raw-mask hit: {means.get('raw_mask_hit')}\n"
        f"- Verified logits: {atlas.get('logit_status', {}).get('verified_complete')}\n"
        "- Val120 role: historically exposed internal held-out replication; not independent or blinded\n"
        "- Independent confirmation: new externally unseen custodied data required\n"
    )
    artifact_manifest_bytes = (
        json.dumps(
            artifact_manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    claims_bytes = yaml.safe_dump(
        dict(claims), sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    closeout_metrics = {
        "outcome": "go",
        "completed_stage_ids": list(required_evidence),
        "verified_artifact_ids": sorted(verified_ids),
        "claims_supported": sorted(decisions),
        "artifact_manifest_sha256": hashlib.sha256(
            artifact_manifest_bytes
        ).hexdigest(),
        "claims_sha256": hashlib.sha256(claims_bytes).hexdigest(),
        "atlas_payload_sha256": atlas_metrics["atlas_payload_sha256"],
        "val120_role": "historically_exposed_internal_held_out_replication",
        "val120_independence_status": "not_independent_and_not_blinded",
        "independent_confirmation_requirement": (
            "new_externally_unseen_custodied_data_required"
        ),
        "gate_evidence": gate_evidence,
    }
    closeout_path = RESULTS_ROOT / "revision_2_closeout.json"
    evidence_path = _resolve_declared_path(_completion_evidence_path(stage))
    closeout_payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "plan_revision": 2,
        "producer_stage_id": str(stage["id"]),
        **closeout_metrics,
    }
    snapshot = _snapshot_files(
        [artifact_path, claims_path, summary_path, closeout_path, evidence_path]
    )
    try:
        atomic_write_json(artifact_path, artifact_manifest)
        _atomic_write_yaml(claims_path, claims)
        _atomic_write_text(summary_path, summary_text)
        atomic_write_json(closeout_path, closeout_payload)
        return _write_passed_evidence(
            stage,
            summary=(
                "All foundation gates, required artifacts, and predeclared "
                "claims verified."
            ),
            metrics=closeout_metrics,
            outcome="go",
        )
    except Exception:
        _restore_file_snapshot(snapshot)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable ASD-000 evidence-lock phase"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_TO_STAGE))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stage: Mapping[str, Any] | None = None
    completion_safe = False
    try:
        plan = _load_mapping(EXPERIMENT_ROOT / "experiment.yaml", label="plan")
        stage = _plan_stage(plan, PHASE_TO_STAGE[args.phase])
        _enforce_completion_contract(stage)
        completion_safe = True
        protocol = load_protocol(args.config)
        if protocol.get("experiment_id") != EXPERIMENT_ID:
            raise StageContractError(
                "protocol experiment_id does not match the owning experiment"
            )
        _configure_resource_meter(protocol, plan)

        handlers = {
            "selfcheck": _phase_selfcheck,
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
    except Exception as exc:
        failure_evidence: Path | None = None
        failure_evidence_error: BaseException | None = None
        if stage is not None and completion_safe:
            try:
                failure_evidence = _write_failed_evidence(stage, exc)
            except Exception as write_exc:
                failure_evidence_error = write_exc
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if failure_evidence is not None:
            print(
                f"failed_evidence={failure_evidence}",
                file=sys.stderr,
            )
        elif failure_evidence_error is not None:
            print(
                "failed_evidence_write_error="
                f"{type(failure_evidence_error).__name__}: "
                f"{failure_evidence_error}",
                file=sys.stderr,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
