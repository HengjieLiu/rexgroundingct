#!/usr/bin/env python3
"""Acquire and audit the official CT-RATE ``ts_total`` ReX val200 subset.

This program intentionally has explicit stages.  It never resamples, rewrites,
or otherwise changes an input CT or official segmentation.  Case-level
manifests, source files, QC images, and detailed results are written only below
the externally managed runtime root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import Counter
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import urlopen

import numpy as np


EXPERIMENT_ID = "020_ct_rate_ts_total_rex_val200_audit"
SCRIPT_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "020_ct_rate_ts_total_rex_val200_audit"
)

REX_REPO_ID = "rajpurkarlab/ReXGroundingCT"
CT_RATE_REPO_ID = "ibrahimhamamci/CT-RATE"
REX_METADATA_REMOTE_PATH = "MICCAI_challenge_dataset.json"
CT_RATE_TS_README_PATH = "dataset/ts_seg/README.md"
CT_RATE_TS_REQUIREMENTS_PATH = "dataset/ts_seg/requirements.txt"
TOTALSEGMENTATOR_VERSION = "2.7.0"
TOTALSEGMENTATOR_MAP_URL = (
    "https://raw.githubusercontent.com/wasserth/TotalSegmentator/"
    "v2.7.0/totalsegmentator/map_to_binary.py"
)
TOTALSEGMENTATOR_MAP_SHA256 = (
    "40443a4ad56438b5ab55c9c02310f92f64345ebd6c8159af620dd1b61775ac7a"
)
TOTAL_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}
LUNG_LABELS = {
    "left": (10, 11),
    "right": (12, 13, 14),
}
HEADER_FORM_QC_ALLOWED_STATUSES = frozenset({"PASS", "SFORM_UNSET_QFORM_ALIGNED"})
KNOWN_UPSTREAM_MISSING = {
    "train_1267_a_4.nii.gz",
    "train_11755_a_3.nii.gz",
    "train_11755_a_4.nii.gz",
}
VOLUME_NAME_PATTERN = re.compile(
    r"^(?P<split>train|valid)_(?P<patient>[^_]+)_(?P<scan>[^_]+)_(?P<reconstruction>[^_]+)\.nii\.gz$"
)


def utc_now() -> str:
    """Return an ISO-8601 timestamp with an explicit UTC offset."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return Git's blob object SHA-1 for a local regular file."""
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}, got {type(payload).__name__}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def runtime_paths(runtime_root: Path, ct_rate_revision: str | None = None) -> dict[str, Path]:
    paths = {
        "root": runtime_root,
        "config": runtime_root / "config",
        "manifests": runtime_root / "manifests" / "private",
        "reports": runtime_root / "reports",
        "private_reports": runtime_root / "reports" / "private",
        "staging": runtime_root / "staging",
        "derived_lungs": runtime_root / "derived" / "lungs_native",
        "visual_qc": runtime_root / "visual_qc",
        "logs": runtime_root / "logs",
    }
    if ct_rate_revision:
        paths["raw_masks"] = runtime_root / "sources" / ct_rate_revision / "raw"
    return paths


def make_runtime_directories(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def source_lock_contract(source_lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable portion of a source lock (excluding its timestamp/hash)."""
    return {
        key: value
        for key, value in source_lock.items()
        if key not in {"created_at", "source_lock_sha256"}
    }


def source_lock_sha256(source_lock: Mapping[str, Any]) -> str:
    return canonical_json_sha256(source_lock_contract(source_lock))


def _archive_generated_file(path: Path, archive_directory: Path, label: str) -> Path:
    """Preserve stale generated state rather than silently overwriting it."""
    archive_directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = archive_directory / f"{path.stem}.{label}.{stamp}.{os.getpid()}{path.suffix}"
    os.replace(path, archived)
    return archived


def parse_volume_name(volume_name: str) -> dict[str, str]:
    match = VOLUME_NAME_PATTERN.fullmatch(volume_name)
    if not match:
        raise ValueError(f"Unexpected CT-RATE VolumeName: {volume_name!r}")
    fields = match.groupdict()
    original_split = fields["split"]
    patient = f"{original_split}_{fields['patient']}"
    scan = f"{patient}_{fields['scan']}"
    return {
        "volume_name": volume_name,
        "ct_rate_original_split": original_split,
        "ct_rate_fixed_split": f"{original_split}_fixed",
        "patient": patient,
        "scan": scan,
    }


def ct_rate_ct_remote_path(volume_name: str) -> str:
    parsed = parse_volume_name(volume_name)
    return (
        f"dataset/{parsed['ct_rate_fixed_split']}/{parsed['patient']}/"
        f"{parsed['scan']}/{volume_name}"
    )


def ct_rate_ts_total_remote_path(volume_name: str) -> str:
    parsed = parse_volume_name(volume_name)
    return (
        f"dataset/ts_seg/ts_total/{parsed['ct_rate_fixed_split']}/"
        f"{parsed['patient']}/{parsed['scan']}/{volume_name}"
    )


def local_ct_sidecar_path(ct_root: Path, remote_ct_path: str) -> Path:
    return ct_root / ".cache" / "huggingface" / "download" / f"{remote_ct_path}.metadata"


def parse_local_ct_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "SIDECAR_MISSING", "path": str(path)}
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return {"status": "SIDECAR_INVALID", "path": str(path)}
    return {
        "status": "SIDECAR_PRESENT",
        "path": str(path),
        "revision": lines[0].strip(),
        "etag": lines[1].strip(),
        "downloaded_at_epoch": lines[2].strip() if len(lines) > 2 else None,
    }


def node_to_record(node: Any) -> dict[str, Any]:
    """Convert a Hub tree node into a JSON-stable, credential-free record."""
    lfs = getattr(node, "lfs", None)
    if isinstance(lfs, Mapping):
        lfs_sha256 = lfs.get("sha256")
        lfs_size = lfs.get("size")
    else:
        lfs_sha256 = getattr(lfs, "sha256", None)
        lfs_size = getattr(lfs, "size", None)
    return {
        "path": getattr(node, "path", None),
        "size": getattr(node, "size", None),
        "blob_id": getattr(node, "blob_id", None),
        "lfs_sha256": lfs_sha256,
        "lfs_size": lfs_size,
        "xet_hash": getattr(node, "xet_hash", None),
    }


def _path_records(nodes: Iterable[Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for node in nodes:
        record = node_to_record(node)
        path = record["path"]
        if path:
            output[str(path)] = record
    return output


def load_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Config {config_path} is for {config.get('experiment')!r}, "
            f"not {EXPERIMENT_ID!r}"
        )
    return config


def _resolve_repo_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def load_population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    population = config["population"]
    metadata_path = Path(str(population["metadata_path"]))
    expected_metadata_hash = str(population["metadata_sha256"])
    observed_metadata_hash = sha256_file(metadata_path)
    if observed_metadata_hash != expected_metadata_hash:
        raise ValueError(
            "Local ReX metadata hash mismatch: "
            f"expected={expected_metadata_hash} observed={observed_metadata_hash}"
        )
    metadata = read_json(metadata_path)
    split = str(population["rex_split"])
    source_rows = metadata.get(split)
    if not isinstance(source_rows, list):
        raise ValueError(f"Expected list metadata[{split!r}] in {metadata_path}")

    names: list[str] = []
    for row in source_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ValueError(f"Malformed row in metadata[{split!r}]: {row!r}")
        names.append(str(row["name"]))
    expected_cases = int(population["expected_unique_cases"])
    if len(names) != expected_cases or len(set(names)) != expected_cases:
        raise ValueError(
            f"Expected exactly {expected_cases} unique {split} cases, "
            f"found rows={len(names)} unique={len(set(names))}"
        )

    evaluation_path = _resolve_repo_relative(str(population["evaluation_manifest"]))
    expected_eval_hash = str(population["evaluation_manifest_sha256"])
    observed_eval_hash = sha256_file(evaluation_path)
    if observed_eval_hash != expected_eval_hash:
        raise ValueError(
            "Evaluation manifest hash mismatch: "
            f"expected={expected_eval_hash} observed={observed_eval_hash}"
        )
    evaluation = read_json(evaluation_path)
    evaluation_rows = evaluation.get("test")
    if not isinstance(evaluation_rows, list):
        raise ValueError(f"Expected a test list in {evaluation_path}")
    evaluation_names = [str(row["name"]) for row in evaluation_rows]
    if len(evaluation_names) != expected_cases or set(evaluation_names) != set(names):
        raise ValueError("ReX metadata val population and fixed evaluation set differ")

    output: list[dict[str, Any]] = []
    for metadata_index, name in enumerate(names):
        parsed = parse_volume_name(name)
        output.append(
            {
                "metadata_index": metadata_index,
                "volume_name": name,
                "rex_split": split,
                "ct_rate_original_split": parsed["ct_rate_original_split"],
                "ct_rate_fixed_split": parsed["ct_rate_fixed_split"],
                "remote_ct_path": ct_rate_ct_remote_path(name),
                "remote_ts_total_path": ct_rate_ts_total_remote_path(name),
                "known_upstream_missing": name in KNOWN_UPSTREAM_MISSING,
            }
        )
    expected_splits = dict(population["expected_ct_rate_split_counts"])
    observed_splits = Counter(row["ct_rate_fixed_split"] for row in output)
    normalized_observed_splits = {
        split: int(observed_splits.get(split, 0)) for split in expected_splits
    }
    unexpected_splits = sorted(set(observed_splits) - set(expected_splits))
    if normalized_observed_splits != expected_splits or unexpected_splits:
        raise ValueError(
            f"Unexpected CT-RATE fixed split counts: expected={expected_splits} "
            f"observed={dict(observed_splits)}"
        )
    return output


def classify_hub_exception(exc: BaseException) -> str:
    """Classify Hub errors without treating access failures as absent files."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return "auth_error"
    if status_code == 404:
        return "upstream_absent"
    name = type(exc).__name__.lower()
    if "gated" in name or "auth" in name or "forbidden" in name:
        return "auth_error"
    if "entrynotfound" in name or "repositorynotfound" in name:
        return "upstream_absent"
    return "transport_error"


def validate_lut_provenance(
    *,
    ct_rate_readme_bytes: bytes | None,
    map_source_bytes: bytes | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Validate the CT-RATE producer version and its immutable official map."""
    expected_lines = (
        b'10: "lung_upper_lobe_left"',
        b'11: "lung_lower_lobe_left"',
        b'12: "lung_upper_lobe_right"',
        b'13: "lung_middle_lobe_right"',
        b'14: "lung_lower_lobe_right"',
    )
    record: dict[str, Any] = {
        "status": "LUT_PROVENANCE_BLOCKED",
        "ct_rate_producer_version": TOTALSEGMENTATOR_VERSION,
        "ct_rate_readme_sha256": (
            sha256_bytes(ct_rate_readme_bytes) if ct_rate_readme_bytes is not None else None
        ),
        "official_map_url": TOTALSEGMENTATOR_MAP_URL,
        "official_map_sha256": (
            sha256_bytes(map_source_bytes) if map_source_bytes is not None else None
        ),
        "expected_official_map_sha256": TOTALSEGMENTATOR_MAP_SHA256,
        "total_task_label_map": {str(key): value for key, value in TOTAL_LABELS.items()},
        "lung_lobe_label_map": {
            str(key): TOTAL_LABELS[key]
            for key in (*LUNG_LABELS["left"], *LUNG_LABELS["right"])
        },
        "error": error,
    }
    if ct_rate_readme_bytes is None or map_source_bytes is None:
        return record
    readme_text = ct_rate_readme_bytes.decode("utf-8", errors="replace")
    if not re.search(
        rf"TotalSegmentator version:\s*{re.escape(TOTALSEGMENTATOR_VERSION)}\b",
        readme_text,
    ):
        record["error"] = "CT-RATE producer README does not declare the expected version"
        return record
    if record["official_map_sha256"] != TOTALSEGMENTATOR_MAP_SHA256:
        record["error"] = "Official TotalSegmentator map checksum differs from the pinned source"
        return record
    if not all(line in map_source_bytes for line in expected_lines):
        record["error"] = "Pinned official map does not contain the expected v2 lobe mapping"
        return record
    record["status"] = "PASS"
    record["error"] = None
    return record


def _fetch_lut_evidence(
    *,
    ct_rate_revision: str,
    ct_rate_readme_node: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fetch only small documentation/source files needed for LUT provenance."""
    try:
        from huggingface_hub import hf_hub_download

        readme_path = Path(
            hf_hub_download(
                repo_id=CT_RATE_REPO_ID,
                repo_type="dataset",
                filename=CT_RATE_TS_README_PATH,
                revision=ct_rate_revision,
                token=True,
            )
        )
        readme_bytes = readme_path.read_bytes()
        with urlopen(TOTALSEGMENTATOR_MAP_URL, timeout=30) as response:
            map_source_bytes = response.read()
        evidence = validate_lut_provenance(
            ct_rate_readme_bytes=readme_bytes,
            map_source_bytes=map_source_bytes,
        )
    except Exception as exc:  # The audit can still download/audit geometry when this is blocked.
        evidence = validate_lut_provenance(
            ct_rate_readme_bytes=None,
            map_source_bytes=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    evidence["ct_rate_readme_remote_node"] = dict(ct_rate_readme_node or {})
    return evidence


def preflight(config: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    """Resolve the precise immutable source object set without downloading masks."""
    from huggingface_hub import HfApi, hf_hub_download

    rows = load_population(config)
    pins = config["repository_pins"]
    rex_pin = str(pins["rexgroundingct"]["revision"])
    ct_rate_pin = str(pins["ct_rate"]["revision"])
    config_canonical_sha256 = canonical_json_sha256(config)
    script_sha256 = sha256_file(Path(__file__))
    ct_root = Path(str(config["inputs"]["local_ct_root"]))
    paths = runtime_paths(runtime_root, ct_rate_pin)
    make_runtime_directories(paths)

    api = HfApi()
    # An authenticated exact-file query, rather than repo_info alone, proves gated access.
    rex_info = api.repo_info(REX_REPO_ID, repo_type="dataset", revision=rex_pin, token=True)
    ct_info = api.repo_info(CT_RATE_REPO_ID, repo_type="dataset", revision=ct_rate_pin, token=True)
    if rex_info.sha != rex_pin or ct_info.sha != ct_rate_pin:
        raise RuntimeError(
            "Pinned source revision resolution changed unexpectedly: "
            f"ReX={rex_info.sha}, CT-RATE={ct_info.sha}"
        )
    rex_nodes = _path_records(
        api.get_paths_info(
            REX_REPO_ID,
            [REX_METADATA_REMOTE_PATH],
            repo_type="dataset",
            revision=rex_pin,
            token=True,
        )
    )
    if REX_METADATA_REMOTE_PATH not in rex_nodes:
        raise RuntimeError("Authenticated ReX metadata lookup returned no target object")
    metadata_path = Path(str(config["population"]["metadata_path"]))
    local_git_blob = git_blob_sha1(metadata_path)
    remote_metadata = rex_nodes[REX_METADATA_REMOTE_PATH]
    # Hub ``blob_id`` is an opaque API object identifier here, not necessarily
    # Git's computed blob SHA-1.  Compare actual pinned bytes instead.
    remote_metadata_path = Path(
        hf_hub_download(
            repo_id=REX_REPO_ID,
            repo_type="dataset",
            filename=REX_METADATA_REMOTE_PATH,
            revision=rex_pin,
            token=True,
        )
    )
    local_metadata_sha256 = sha256_file(metadata_path)
    remote_metadata_sha256 = sha256_file(remote_metadata_path)
    if (
        local_metadata_sha256 != remote_metadata_sha256
        or metadata_path.stat().st_size != remote_metadata_path.stat().st_size
    ):
        raise RuntimeError(
            "Local ReX metadata bytes do not match the pinned remote metadata: "
            f"local_sha256={local_metadata_sha256} remote_sha256={remote_metadata_sha256}"
        )

    seg_paths = [str(row["remote_ts_total_path"]) for row in rows]
    ct_paths = [str(row["remote_ct_path"]) for row in rows]
    seg_nodes = _path_records(
        api.get_paths_info(
            CT_RATE_REPO_ID,
            seg_paths,
            repo_type="dataset",
            revision=ct_rate_pin,
            token=True,
        )
    )
    ct_nodes = _path_records(
        api.get_paths_info(
            CT_RATE_REPO_ID,
            ct_paths,
            repo_type="dataset",
            revision=ct_rate_pin,
            token=True,
        )
    )
    doc_nodes = _path_records(
        api.get_paths_info(
            CT_RATE_REPO_ID,
            [CT_RATE_TS_README_PATH, CT_RATE_TS_REQUIREMENTS_PATH],
            repo_type="dataset",
            revision=ct_rate_pin,
            token=True,
        )
    )
    missing_seg_paths = sorted(set(seg_paths) - set(seg_nodes))
    missing_ct_paths = sorted(set(ct_paths) - set(ct_nodes))
    if missing_seg_paths or missing_ct_paths:
        raise RuntimeError(
            "Pinned remote source is incomplete: "
            f"seg_missing={len(missing_seg_paths)} ct_missing={len(missing_ct_paths)}"
        )

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        remote_ct_path = str(row["remote_ct_path"])
        remote_seg_path = str(row["remote_ts_total_path"])
        ct_node = ct_nodes[remote_ct_path]
        seg_node = seg_nodes[remote_seg_path]
        local_ct_path = ct_root / remote_ct_path
        sidecar = parse_local_ct_sidecar(local_ct_sidecar_path(ct_root, remote_ct_path))
        sidecar_matches = (
            sidecar.get("status") == "SIDECAR_PRESENT"
            and sidecar.get("revision") == ct_rate_pin
            and sidecar.get("etag") == ct_node.get("lfs_sha256")
        )
        local_size = local_ct_path.stat().st_size if local_ct_path.exists() else None
        record = dict(row)
        record.update(
            {
                "local_ct_path": str(local_ct_path),
                "local_ct_exists": local_ct_path.exists(),
                "local_ct_size": local_size,
                "local_ct_sidecar": sidecar,
                "local_ct_sidecar_status": "SIDECAR_MATCH" if sidecar_matches else "SIDECAR_MISMATCH",
                "remote_ct": ct_node,
                "remote_ts_total": seg_node,
                "local_ts_total_path": str(paths["raw_masks"] / remote_seg_path),
            }
        )
        manifest_rows.append(record)

    manifest_contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "metadata_sha256": config["population"]["metadata_sha256"],
        "rex_revision": rex_pin,
        "ct_rate_revision": ct_rate_pin,
        "cases": manifest_rows,
    }
    manifest_sha256 = canonical_json_sha256(manifest_contract)
    manifest_payload = {
        **manifest_contract,
        "manifest_sha256": manifest_sha256,
        "created_at": utc_now(),
    }
    lut_evidence = _fetch_lut_evidence(
        ct_rate_revision=ct_rate_pin,
        ct_rate_readme_node=doc_nodes.get(CT_RATE_TS_README_PATH),
    )
    source_lock = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256,
        "config_canonical_sha256": config_canonical_sha256,
        "created_at": utc_now(),
        "manifest_sha256": manifest_sha256,
        "repositories": {
            "rexgroundingct": {
                "repo_id": REX_REPO_ID,
                "revision": rex_pin,
                "metadata_remote_node": remote_metadata,
                "local_metadata_git_blob_sha1": local_git_blob,
                "local_metadata_sha256": local_metadata_sha256,
                "remote_metadata_sha256": remote_metadata_sha256,
            },
            "ct_rate": {
                "repo_id": CT_RATE_REPO_ID,
                "revision": ct_rate_pin,
                "ts_readme_remote_node": doc_nodes.get(CT_RATE_TS_README_PATH),
                "requirements_remote_node": doc_nodes.get(CT_RATE_TS_REQUIREMENTS_PATH),
            },
        },
        "lut_provenance": lut_evidence,
        "summary": {
            "requested_masks": len(manifest_rows),
            "remote_ts_total_bytes": sum(
                int(row["remote_ts_total"].get("size") or 0) for row in manifest_rows
            ),
            "split_counts": dict(Counter(row["ct_rate_fixed_split"] for row in manifest_rows)),
            "known_upstream_missing_in_population": sum(
                bool(row["known_upstream_missing"]) for row in manifest_rows
            ),
            "local_ct_present": sum(bool(row["local_ct_exists"]) for row in manifest_rows),
            "local_ct_sidecar_matches": sum(
                row["local_ct_sidecar_status"] == "SIDECAR_MATCH" for row in manifest_rows
            ),
        },
    }
    source_lock["source_lock_sha256"] = source_lock_sha256(source_lock)
    write_json_atomic(paths["config"] / "canonical_config_snapshot.json", config)
    write_json_atomic(paths["manifests"] / "remote_manifest.json", manifest_payload)
    write_json_atomic(paths["config"] / "source_lock.json", source_lock)
    return {"manifest": manifest_payload, "source_lock": source_lock, "paths": paths}


def _require_preflight(runtime_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    source_lock_path = runtime_root / "config" / "source_lock.json"
    manifest_path = runtime_root / "manifests" / "private" / "remote_manifest.json"
    if not source_lock_path.exists() or not manifest_path.exists():
        raise RuntimeError("No preflight source lock/manifest found; run --preflight first")
    source_lock = read_json(source_lock_path)
    manifest = read_json(manifest_path)
    expected_source_lock_hash = source_lock.get("source_lock_sha256")
    if not isinstance(expected_source_lock_hash, str) or source_lock_sha256(source_lock) != expected_source_lock_hash:
        raise RuntimeError("Source lock integrity check failed; rerun --preflight")
    if source_lock.get("script_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Audit script differs from the sealed source lock; rerun --preflight")
    snapshot_path = runtime_root / "config" / "canonical_config_snapshot.json"
    if not snapshot_path.exists():
        raise RuntimeError("Source-lock config snapshot is missing; rerun --preflight")
    snapshot = read_json(snapshot_path)
    if canonical_json_sha256(snapshot) != source_lock.get("config_canonical_sha256"):
        raise RuntimeError("Source-lock config snapshot integrity check failed; rerun --preflight")
    expected_manifest_hash = source_lock.get("manifest_sha256")
    contract = {
        key: manifest[key]
        for key in ("schema_version", "experiment", "metadata_sha256", "rex_revision", "ct_rate_revision", "cases")
    }
    observed_manifest_hash = canonical_json_sha256(contract)
    if observed_manifest_hash != expected_manifest_hash or manifest.get("manifest_sha256") != expected_manifest_hash:
        raise RuntimeError("Runtime manifest is not the immutable preflight-approved manifest")
    paths = runtime_paths(runtime_root, str(manifest["ct_rate_revision"]))
    make_runtime_directories(paths)
    return source_lock, manifest, paths


def _import_nibabel() -> Any:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nibabel is required for this stage; run it in the documented "
            "TotalSegmentator audit container"
        ) from exc
    return nib


def execution_environment_record(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture and, when configured, enforce the immutable audit container identity."""
    expected = dict(config.get("execution_environment", {}))
    image_ref = os.environ.get("AUDIT_CONTAINER_IMAGE_REF")
    image_id = os.environ.get("AUDIT_CONTAINER_IMAGE_ID")
    if expected.get("container_image_ref") and image_ref != expected["container_image_ref"]:
        raise RuntimeError("Audit container image reference does not match the sealed config")
    if expected.get("container_image_id") and image_id != expected["container_image_id"]:
        raise RuntimeError("Audit container image ID does not match the sealed config")
    packages: dict[str, str | None] = {}
    for name in ("nibabel", "scipy", "matplotlib", "huggingface_hub", "TotalSegmentator"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at": utc_now(),
        "python_version": sys.version,
        "container_image_ref": image_ref,
        "container_image_id": image_id,
        "packages": packages,
    }


def validate_mask_file(path: Path, expected_size: int | None, expected_sha256: str | None) -> dict[str, Any]:
    """Full-read, content-addressed NIfTI validation for a source mask."""
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    observed_size = path.stat().st_size
    if expected_size is not None and observed_size != int(expected_size):
        return {
            "status": "SIZE_MISMATCH",
            "path": str(path),
            "observed_size": observed_size,
            "expected_size": expected_size,
        }
    observed_sha256 = sha256_file(path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        return {
            "status": "SHA256_MISMATCH",
            "path": str(path),
            "observed_sha256": observed_sha256,
            "expected_sha256": expected_sha256,
        }
    try:
        nib = _import_nibabel()
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)  # Deliberately forces a complete gzip/NIfTI read.
        if data.ndim != 3:
            return {"status": "NOT_3D", "path": str(path), "shape": list(data.shape)}
        if not np.isfinite(data).all():
            return {"status": "NONFINITE", "path": str(path), "shape": list(data.shape)}
        if not np.array_equal(data, np.rint(data)):
            return {"status": "NONINTEGER", "path": str(path), "shape": list(data.shape)}
    except Exception as exc:
        return {"status": "NIFTI_INVALID", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "VALID",
        "path": str(path),
        "observed_size": observed_size,
        "observed_sha256": observed_sha256,
        "shape": [int(value) for value in data.shape],
    }


def _atomic_copy_validate(
    source: Path,
    destination: Path,
    staging_root: Path,
    expected_size: int | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Copy a cached Hub object through staging and promote only after full validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_name = destination.name
    # Keep a NIfTI-recognized suffix so the mandatory staged full-read is real.
    temporary = staging_root / f".{relative_name}.{os.getpid()}.part.nii.gz"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    validation = validate_mask_file(temporary, expected_size, expected_sha256)
    if validation["status"] != "VALID":
        return validation
    if destination.exists():
        existing = validate_mask_file(destination, expected_size, expected_sha256)
        if existing["status"] == "VALID":
            temporary.unlink(missing_ok=True)
            return {**existing, "promotion": "already_valid"}
        quarantine_directory = staging_root / "quarantine"
        quarantine_directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantined = quarantine_directory / f"{destination.name}.{stamp}.{os.getpid()}.invalid.nii.gz"
        os.replace(destination, quarantined)
        os.replace(temporary, destination)
        return {
            **validation,
            "path": str(destination),
            "promotion": "replaced_quarantined_invalid_destination",
            "quarantined_destination": str(quarantined),
            "existing_validation": existing,
        }
    os.replace(temporary, destination)
    return {**validation, "path": str(destination), "promotion": "downloaded"}


def download(
    config: Mapping[str, Any],
    runtime_root: Path,
    approval_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Targeted, resumable, revision-pinned source acquisition."""
    from huggingface_hub import hf_hub_download

    source_lock, manifest, paths = _require_preflight(runtime_root)
    manifest_sha256 = str(manifest["manifest_sha256"])
    if approval_manifest_sha256 != manifest_sha256:
        raise RuntimeError(
            "Download requires the exact manifest approval hash printed by --preflight; "
            f"expected {manifest_sha256}"
        )
    ct_rate_revision = str(manifest["ct_rate_revision"])
    download_path = paths["manifests"] / "download_manifest.json"
    if download_path.exists():
        previous = read_json(download_path)
        if (
            previous.get("manifest_sha256") != manifest_sha256
            or previous.get("source_lock_sha256") != source_lock.get("source_lock_sha256")
        ):
            archived = _archive_generated_file(
                download_path, paths["manifests"] / "archive", "stale"
            )
            print(f"Archived stale download manifest: {archived}", flush=True)
            previous = {"cases": {}}
    else:
        previous = {"cases": {}}
    result_by_name: dict[str, dict[str, Any]] = dict(previous.get("cases", {}))
    cases = list(manifest["cases"])
    for index, row in enumerate(cases, start=1):
        volume_name = str(row["volume_name"])
        remote = dict(row["remote_ts_total"])
        destination = Path(str(row["local_ts_total_path"]))
        expected_size = remote.get("size")
        expected_sha256 = remote.get("lfs_sha256")
        existing = validate_mask_file(destination, expected_size, expected_sha256)
        if existing["status"] == "VALID":
            result = {
                "status": "already_valid",
                "file_size": existing["observed_size"],
                "sha256": existing["observed_sha256"],
                "error": None,
            }
        else:
            try:
                cached = Path(
                    hf_hub_download(
                        repo_id=CT_RATE_REPO_ID,
                        repo_type="dataset",
                        filename=str(row["remote_ts_total_path"]),
                        revision=ct_rate_revision,
                        token=True,
                    )
                )
                staged = _atomic_copy_validate(
                    cached,
                    destination,
                    paths["staging"],
                    expected_size,
                    expected_sha256,
                )
                if staged["status"] == "VALID":
                    result = {
                        "status": str(staged["promotion"]),
                        "file_size": staged["observed_size"],
                        "sha256": staged["observed_sha256"],
                        "error": None,
                    }
                else:
                    result = {
                        "status": "integrity_error",
                        "file_size": None,
                        "sha256": None,
                        "error": json.dumps(staged, sort_keys=True),
                    }
            except Exception as exc:
                result = {
                    "status": classify_hub_exception(exc),
                    "file_size": None,
                    "sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        result_by_name[volume_name] = result
        payload = {
            "schema_version": 1,
            "experiment": EXPERIMENT_ID,
            "manifest_sha256": manifest_sha256,
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "updated_at": utc_now(),
            "progress": {"completed": index, "requested": len(cases)},
            "cases": result_by_name,
        }
        write_json_atomic(download_path, payload)
        print(f"[{index}/{len(cases)}] {volume_name}: {result['status']}", flush=True)
    summary = Counter(result["status"] for result in result_by_name.values())
    return {"download_manifest": read_json(download_path), "summary": dict(summary), "source_lock": source_lock}


def affine_geometry(
    ct_affine: np.ndarray,
    seg_affine: np.ndarray,
    ct_shape: Sequence[int],
    seg_shape: Sequence[int],
    index_tolerance_vox: float,
    corner_tolerance_mm: float,
) -> dict[str, Any]:
    """Measure native grid compatibility with explicit voxel/world thresholds."""
    ct_affine = np.asarray(ct_affine, dtype=np.float64)
    seg_affine = np.asarray(seg_affine, dtype=np.float64)
    output: dict[str, Any] = {
        "ct_shape": [int(value) for value in ct_shape[:3]],
        "seg_shape": [int(value) for value in seg_shape[:3]],
        "shape_identical": tuple(ct_shape[:3]) == tuple(seg_shape[:3]),
        "ct_affine": ct_affine.tolist(),
        "seg_affine": seg_affine.tolist(),
        "index_tolerance_vox": float(index_tolerance_vox),
        "corner_tolerance_mm": float(corner_tolerance_mm),
    }
    try:
        transform = np.linalg.inv(ct_affine) @ seg_affine
    except np.linalg.LinAlgError:
        output.update(
            {
                "index_grid_status": "INVALID_CT_AFFINE",
                "world_header_status": "INVALID_CT_AFFINE",
            }
        )
        return output
    identity_residual = transform - np.eye(4)
    output["ct_index_from_seg_index_transform"] = transform.tolist()
    output["max_index_transform_identity_error_vox"] = float(np.max(np.abs(identity_residual)))
    output["ct_axis_scales_mm"] = [float(value) for value in np.linalg.norm(ct_affine[:3, :3], axis=0)]
    output["seg_axis_scales_mm"] = [float(value) for value in np.linalg.norm(seg_affine[:3, :3], axis=0)]
    output["ct_determinant_mm3"] = float(np.linalg.det(ct_affine[:3, :3]))
    output["seg_determinant_mm3"] = float(np.linalg.det(seg_affine[:3, :3]))
    output["ct_handedness"] = "right" if output["ct_determinant_mm3"] > 0 else "left"
    output["seg_handedness"] = "right" if output["seg_determinant_mm3"] > 0 else "left"
    if not output["shape_identical"]:
        output.update(
            {
                "index_grid_status": "SHAPE_MISMATCH",
                "world_header_status": "SHAPE_MISMATCH",
                "corner_residuals_mm": [],
                "max_corner_residual_mm": None,
            }
        )
        return output
    corners = list(itertools.product(*[(0, int(size) - 1) for size in ct_shape[:3]]))
    residuals: list[dict[str, Any]] = []
    for corner in corners:
        homogeneous = np.array([*corner, 1.0], dtype=np.float64)
        residual = float(np.linalg.norm((ct_affine @ homogeneous)[:3] - (seg_affine @ homogeneous)[:3]))
        residuals.append({"index": [int(value) for value in corner], "residual_mm": residual})
    max_corner = max(item["residual_mm"] for item in residuals)
    output["corner_residuals_mm"] = residuals
    output["max_corner_residual_mm"] = float(max_corner)
    output["index_grid_status"] = (
        "PASS"
        if output["max_index_transform_identity_error_vox"] <= index_tolerance_vox
        else "INDEX_TRANSFORM_MISMATCH"
    )
    output["world_header_status"] = (
        "PASS" if max_corner <= corner_tolerance_mm else "WORLD_CORNER_MISMATCH"
    )
    return output


def _matrix_record(matrix: np.ndarray | None, code: int | None) -> dict[str, Any]:
    return {
        "code": int(code) if code is not None else None,
        "matrix": np.asarray(matrix, dtype=np.float64).tolist() if matrix is not None else None,
    }


def label_qc(data: np.ndarray, lut_status: str) -> dict[str, Any]:
    """Perform data-level validation without silently accepting unexpected labels."""
    if data.ndim != 3:
        return {"status": "NOT_3D", "shape": [int(value) for value in data.shape]}
    if not np.isfinite(data).all():
        return {"status": "NONFINITE"}
    if not np.array_equal(data, np.rint(data)):
        return {"status": "NONINTEGER"}
    labels = np.unique(data.astype(np.int64, copy=False))
    nonzero = int(np.count_nonzero(data))
    output: dict[str, Any] = {
        "status": "LUT_PROVENANCE_BLOCKED" if lut_status != "PASS" else "PASS",
        "unique_label_ids": [int(value) for value in labels.tolist()],
        "nonzero_voxels": nonzero,
        "foreground_fraction": float(nonzero / data.size),
        "unexpected_label_ids": [],
    }
    if nonzero == 0:
        output["status"] = "EMPTY_SEGMENTATION"
        return output
    if lut_status == "PASS":
        unexpected = sorted(set(int(value) for value in labels) - ({0} | set(TOTAL_LABELS)))
        output["unexpected_label_ids"] = unexpected
        if unexpected:
            output["status"] = "UNEXPECTED_LABEL_IDS"
    return output


def local_ct_provenance(
    row: Mapping[str, Any],
    hash_cache: dict[str, Any],
    *,
    hash_local_ct: bool,
) -> dict[str, Any]:
    """Verify a local CT against its pinned CT-RATE LFS object when possible."""
    local_path = Path(str(row["local_ct_path"]))
    remote = dict(row["remote_ct"])
    expected_size = remote.get("size")
    expected_sha256 = remote.get("lfs_sha256")
    if not local_path.is_file():
        return {"status": "LOCAL_CT_MISSING", "path": str(local_path)}
    stat = local_path.stat()
    base: dict[str, Any] = {
        "path": str(local_path),
        "size": stat.st_size,
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "sidecar_status": row.get("local_ct_sidecar_status"),
    }
    if expected_size is not None and stat.st_size != int(expected_size):
        return {**base, "status": "LOCAL_CT_SIZE_MISMATCH"}
    if not expected_sha256:
        return {**base, "status": "REMOTE_CT_CHECKSUM_UNAVAILABLE"}
    cache_key = str(row["remote_ct_path"])
    if hash_local_ct:
        observed_sha256 = sha256_file(local_path)
        hash_cache[cache_key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": observed_sha256,
            "hashed_at": utc_now(),
        }
        hash_origin = "new_full_hash"
    else:
        observed_sha256 = None
        hash_origin = "not_requested"
    if observed_sha256 is None:
        status = "SIDECAR_MATCH_UNHASHED" if row.get("local_ct_sidecar_status") == "SIDECAR_MATCH" else "UNHASHED"
    else:
        status = "FULL_HASH_MATCH" if observed_sha256 == expected_sha256 else "FULL_HASH_MISMATCH"
    return {
        **base,
        "status": status,
        "observed_sha256": observed_sha256,
        "hash_origin": hash_origin,
    }


def _nifti_header_record(image: Any, nib: Any) -> dict[str, Any]:
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    zooms = image.header.get_zooms()[:3]
    units = image.header.get_xyzt_units()
    return {
        "best_affine": np.asarray(image.affine, dtype=np.float64).tolist(),
        "shape": [int(value) for value in image.shape[:3]],
        "zooms_mm": [float(value) for value in zooms],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "units": [str(value) for value in units],
        "qform": _matrix_record(qform, qform_code),
        "sform": _matrix_record(sform, sform_code),
    }


def _form_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_matrix = left.get("matrix")
    right_matrix = right.get("matrix")
    if left_matrix is None or right_matrix is None:
        return None
    return float(np.max(np.abs(np.asarray(left_matrix) - np.asarray(right_matrix))))


def _form_status(
    left: Mapping[str, Any], right: Mapping[str, Any], tolerance_mm: float
) -> str:
    """Assess qform/sform presence, code, and matrix compatibility separately."""
    left_matrix = left.get("matrix")
    right_matrix = right.get("matrix")
    left_code = left.get("code")
    right_code = right.get("code")
    if left_matrix is None and right_matrix is None:
        return "BOTH_UNSET" if left_code == right_code == 0 else "FORM_MISSING"
    if left_matrix is None or right_matrix is None:
        return "FORM_PRESENCE_MISMATCH"
    if left_code != right_code:
        return "FORM_CODE_MISMATCH"
    difference = _form_difference(left, right)
    if difference is None or difference > tolerance_mm:
        return "FORM_MATRIX_MISMATCH"
    return "PASS"


def _form_vs_best_affine_status(form: Mapping[str, Any], best_affine: Sequence[Sequence[float]], tolerance_mm: float) -> str:
    """Ensure a coded qform/sform does not contradict its image's best affine."""
    matrix = form.get("matrix")
    code = form.get("code")
    if matrix is None:
        return "UNSET" if code == 0 else "FORM_MISSING"
    difference = float(np.max(np.abs(np.asarray(matrix) - np.asarray(best_affine))))
    return "PASS" if difference <= tolerance_mm else "FORM_BEST_AFFINE_MISMATCH"


def _form_is_coded(form: Mapping[str, Any]) -> bool:
    """Return whether a form has both a matrix and an authoritative code."""
    return form.get("matrix") is not None and int(form.get("code") or 0) > 0


def _sform_unset_qform_aligned(
    ct_header: Mapping[str, Any],
    seg_header: Mapping[str, Any],
    *,
    qform_status: str,
    sform_status: str,
    ct_qform_best_status: str,
    ct_sform_best_status: str,
    seg_qform_best_status: str,
    seg_sform_best_status: str,
) -> bool:
    """Recognize the one safe non-identical form convention for QC-only masks.

    CT-RATE ``ts_total`` masks may retain the exact coded qform but leave the
    redundant sform unset.  This is not a second, competing spatial transform
    when the CT's coded qform and sform both agree with their best affine and
    the segmentation's coded qform agrees with its best affine.  Keep this
    allowance deliberately asymmetric and narrow: an unset CT sform, a missing
    qform, or any contradictory matrix/code remains a blocking mismatch.
    """
    return (
        qform_status == "PASS"
        and sform_status == "FORM_PRESENCE_MISMATCH"
        and ct_qform_best_status == "PASS"
        and ct_sform_best_status == "PASS"
        and seg_qform_best_status == "PASS"
        and seg_sform_best_status == "UNSET"
        and _form_is_coded(ct_header["qform"])
        and _form_is_coded(seg_header["qform"])
        and _form_is_coded(ct_header["sform"])
        and seg_header["sform"].get("matrix") is None
        and int(seg_header["sform"].get("code") or 0) == 0
    )


def header_form_status_allows_native_qc(status: object) -> bool:
    """Return whether a recorded header-form status permits QC-only derivation."""
    return str(status) in HEADER_FORM_QC_ALLOWED_STATUSES


def _header_comparison(
    ct_header: Mapping[str, Any], seg_header: Mapping[str, Any], tolerance_mm: float
) -> dict[str, Any]:
    ct_zooms = np.asarray(ct_header["zooms_mm"], dtype=np.float64)
    seg_zooms = np.asarray(seg_header["zooms_mm"], dtype=np.float64)
    qform_status = _form_status(ct_header["qform"], seg_header["qform"], tolerance_mm)
    sform_status = _form_status(ct_header["sform"], seg_header["sform"], tolerance_mm)
    ct_qform_best_status = _form_vs_best_affine_status(
        ct_header["qform"], ct_header["best_affine"], tolerance_mm
    )
    ct_sform_best_status = _form_vs_best_affine_status(
        ct_header["sform"], ct_header["best_affine"], tolerance_mm
    )
    seg_qform_best_status = _form_vs_best_affine_status(
        seg_header["qform"], seg_header["best_affine"], tolerance_mm
    )
    seg_sform_best_status = _form_vs_best_affine_status(
        seg_header["sform"], seg_header["best_affine"], tolerance_mm
    )
    ct_spatial_unit = str(ct_header["units"][0])
    seg_spatial_unit = str(seg_header["units"][0])
    if ct_spatial_unit == seg_spatial_unit == "mm":
        units_status = "PASS"
    elif ct_spatial_unit != seg_spatial_unit:
        units_status = "UNIT_MISMATCH"
    else:
        units_status = "UNSUPPORTED_SPATIAL_UNIT"
    forms_exactly_compatible = qform_status in {"PASS", "BOTH_UNSET"} and sform_status in {
        "PASS",
        "BOTH_UNSET",
    }
    has_coded_spatial_form = qform_status == "PASS" or sform_status == "PASS"
    input_forms_consistent = all(
        status in {"PASS", "UNSET"}
        for status in (
            ct_qform_best_status,
            ct_sform_best_status,
            seg_qform_best_status,
            seg_sform_best_status,
        )
    )
    sform_unset_qform_aligned = _sform_unset_qform_aligned(
        ct_header,
        seg_header,
        qform_status=qform_status,
        sform_status=sform_status,
        ct_qform_best_status=ct_qform_best_status,
        ct_sform_best_status=ct_sform_best_status,
        seg_qform_best_status=seg_qform_best_status,
        seg_sform_best_status=seg_sform_best_status,
    )
    if units_status != "PASS":
        header_form_status = units_status
    elif forms_exactly_compatible and not input_forms_consistent:
        header_form_status = "INPUT_FORM_BEST_AFFINE_MISMATCH"
    elif forms_exactly_compatible and not has_coded_spatial_form:
        header_form_status = "NO_CODED_SPATIAL_FORM"
    elif forms_exactly_compatible:
        header_form_status = "PASS"
    elif sform_unset_qform_aligned:
        header_form_status = "SFORM_UNSET_QFORM_ALIGNED"
    else:
        header_form_status = "QFORM_SFORM_MISMATCH"
    return {
        "orientation_identical": ct_header["orientation"] == seg_header["orientation"],
        "units_identical": ct_header["units"] == seg_header["units"],
        "ct_spatial_unit": ct_spatial_unit,
        "seg_spatial_unit": seg_spatial_unit,
        "units_status": units_status,
        "max_absolute_spacing_difference_mm": float(np.max(np.abs(ct_zooms - seg_zooms))),
        "spacing_status": (
            "PASS" if np.max(np.abs(ct_zooms - seg_zooms)) <= tolerance_mm else "SPACING_MISMATCH"
        ),
        "qform_code_identical": ct_header["qform"]["code"] == seg_header["qform"]["code"],
        "sform_code_identical": ct_header["sform"]["code"] == seg_header["sform"]["code"],
        "qform_max_absolute_difference": _form_difference(ct_header["qform"], seg_header["qform"]),
        "sform_max_absolute_difference": _form_difference(ct_header["sform"], seg_header["sform"]),
        "qform_status": qform_status,
        "sform_status": sform_status,
        "ct_qform_best_affine_status": ct_qform_best_status,
        "ct_sform_best_affine_status": ct_sform_best_status,
        "seg_qform_best_affine_status": seg_qform_best_status,
        "seg_sform_best_affine_status": seg_sform_best_status,
        "header_form_status": header_form_status,
        "header_metadata_status": (
            "EXACT_FORMS" if header_form_status == "PASS"
            else "SEGMENTATION_SFORM_UNSET" if sform_unset_qform_aligned
            else "MISMATCH_OR_UNSUPPORTED"
        ),
    }


def _save_binary_lung(nib: Any, mask: np.ndarray, template: Any, output_path: Path) -> None:
    """Save a derived binary mask without changing its source-grid geometry."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = template.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask.astype(np.uint8, copy=False), template.affine, header)
    qform, qform_code = template.get_qform(coded=True)
    sform, sform_code = template.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qform_code))
    if sform is not None:
        image.set_sform(sform, int(sform_code))
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(image, str(temporary))
    os.replace(temporary, output_path)


def lung_metrics(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Calculate native-space lung QC metrics using 26-connectivity."""
    voxels = int(np.count_nonzero(mask))
    voxel_volume_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000.0
    output: dict[str, Any] = {
        "voxel_count": voxels,
        "physical_volume_ml": float(voxels * voxel_volume_ml),
        "voxel_volume_ml": voxel_volume_ml,
        "connected_components_26": 0,
        "largest_component_fraction": 0.0,
        "flags": [],
    }
    if voxels == 0:
        output["flags"].append("empty")
        return output
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for 26-connected-component lung QC") from exc
    labels, components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:]
    largest = int(counts.max()) if counts.size else 0
    output["connected_components_26"] = int(components)
    output["largest_component_fraction"] = float(largest / voxels)
    # These are review flags rather than exclusions: a partial CT can have a small lung volume.
    if output["physical_volume_ml"] < 250.0:
        output["flags"].append("very_small_native_volume")
    if components > 5 or output["largest_component_fraction"] < 0.95:
        output["flags"].append("fragmented")
    return output


def add_lung_outlier_flags(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Add deterministic robust native-volume and left/right ratio review flags."""
    policy: dict[str, Any] = {
        "method": "median_absolute_deviation_modified_zscore",
        "threshold": 3.5,
        "sides": {},
        "left_right_ratio_threshold": 0.25,
    }
    for side in ("left", "right"):
        eligible = [
            (result, float(result["lungs"][side]["physical_volume_ml"]))
            for result in results
            if result.get("lungs", {}).get(side, {}).get("voxel_count", 0) > 0
        ]
        values = np.asarray([value for _, value in eligible], dtype=np.float64)
        if values.size < 4:
            policy["sides"][side] = {"eligible_cases": int(values.size), "outliers": 0}
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        policy["sides"][side] = {"eligible_cases": int(values.size), "median_ml": median, "mad_ml": mad}
        if mad == 0.0:
            continue
        modified_z = 0.6745 * np.abs(values - median) / mad
        outliers = 0
        for (result, _), score in zip(eligible, modified_z, strict=True):
            if score > 3.5:
                result["lungs"][side]["flags"].append("native_volume_outlier")
                result["lungs"][side]["modified_zscore"] = float(score)
                outliers += 1
        policy["sides"][side]["outliers"] = outliers
    for result in results:
        lungs = result.get("lungs", {})
        left = lungs.get("left", {})
        right = lungs.get("right", {})
        left_volume = float(left.get("physical_volume_ml", 0.0))
        right_volume = float(right.get("physical_volume_ml", 0.0))
        if left_volume > 0 and right_volume > 0:
            ratio = min(left_volume, right_volume) / max(left_volume, right_volume)
            result["lungs"]["left_right_volume_ratio"] = float(ratio)
            if ratio < 0.25:
                left["flags"].append("left_right_volume_imbalance")
                right["flags"].append("left_right_volume_imbalance")
    return policy


def _case_result_csv_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        geometry = result.get("geometry", {})
        labels = result.get("labels", {})
        lungs = result.get("lungs", {})
        rows.append(
            {
                "volume_name": result.get("volume_name"),
                "download_status": result.get("download_status"),
                "source_provenance_status": result.get("source_provenance", {}).get("status"),
                "integrity_status": result.get("integrity", {}).get("status"),
                "shape_identical": geometry.get("shape_identical"),
                "index_grid_status": geometry.get("index_grid_status"),
                "world_header_status": geometry.get("world_header_status"),
                "max_corner_residual_mm": geometry.get("max_corner_residual_mm"),
                "spacing_status": geometry.get("header_comparison", {}).get("spacing_status"),
                "header_form_status": geometry.get("header_comparison", {}).get("header_form_status"),
                "header_metadata_status": geometry.get("header_comparison", {}).get("header_metadata_status"),
                "label_status": labels.get("status"),
                "lung_left_volume_ml": lungs.get("left", {}).get("physical_volume_ml"),
                "lung_right_volume_ml": lungs.get("right", {}).get("physical_volume_ml"),
                "lung_left_flags": ";".join(lungs.get("left", {}).get("flags", [])),
                "lung_right_flags": ";".join(lungs.get("right", {}).get("flags", [])),
                "visual_qc_status": result.get("visual_qc_status"),
                "error": result.get("error"),
            }
        )
    return rows


def verified_audit_results(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], source_lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Reject stale, mixed, duplicated, or foreign per-case audit results."""
    if payload.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError("Audit-results manifest hash does not match the source manifest")
    if payload.get("source_lock_sha256") != source_lock.get("source_lock_sha256"):
        raise RuntimeError("Audit-results source-lock hash does not match the sealed source lock")
    if payload.get("config_canonical_sha256") != source_lock.get("config_canonical_sha256"):
        raise RuntimeError("Audit-results config hash does not match the sealed config")
    if payload.get("script_sha256") != source_lock.get("script_sha256"):
        raise RuntimeError("Audit-results script hash does not match the sealed script")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("Audit results have no case list")
    expected_by_name = {
        str(row["volume_name"]): int(row["metadata_index"]) for row in manifest["cases"]
    }
    observed_names = [str(row.get("volume_name")) for row in cases if isinstance(row, Mapping)]
    if len(cases) != len(expected_by_name) or len(observed_names) != len(cases):
        raise RuntimeError("Audit-results case count is malformed")
    if len(set(observed_names)) != len(observed_names) or set(observed_names) != set(expected_by_name):
        raise RuntimeError("Audit-results cases are not the exact unique preflight population")
    for row in cases:
        if int(row.get("metadata_index", -1)) != expected_by_name[str(row["volume_name"])]:
            raise RuntimeError("Audit-results metadata ordering does not match the locked manifest")
    return [dict(row) for row in cases]


def _audit_exception_reasons(result: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    provenance = result.get("source_provenance", {}).get("status")
    if provenance != "FULL_HASH_MATCH":
        reasons.append(f"source_provenance:{provenance}")
    integrity = result.get("integrity", {}).get("status")
    if integrity != "VALID":
        reasons.append(f"integrity:{integrity}")
    geometry = result.get("geometry", {})
    if geometry.get("index_grid_status") != "PASS":
        reasons.append(f"index_grid:{geometry.get('index_grid_status')}")
    if geometry.get("world_header_status") != "PASS":
        reasons.append(f"world_header:{geometry.get('world_header_status')}")
    header_comparison = geometry.get("header_comparison", {})
    if header_comparison.get("spacing_status") != "PASS":
        reasons.append(f"spacing:{header_comparison.get('spacing_status')}")
    if not header_form_status_allows_native_qc(header_comparison.get("header_form_status")):
        reasons.append(f"header_forms:{header_comparison.get('header_form_status')}")
    labels = result.get("labels", {})
    if labels.get("status") != "PASS":
        reasons.append(f"labels:{labels.get('status')}")
    for side in ("left", "right"):
        flags = result.get("lungs", {}).get(side, {}).get("flags", [])
        reasons.extend(f"lung_{side}:{flag}" for flag in flags)
    if result.get("error"):
        reasons.append("processing_error")
    return reasons


def _aggregate_report(
    source_lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    visual_review: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Return a redacted aggregate report and its explicit conclusion."""
    requested = len(manifest["cases"])
    download_statuses = Counter(str(result.get("download_status")) for result in results)
    provenance_statuses = Counter(
        str(result.get("source_provenance", {}).get("status")) for result in results
    )
    integrity_statuses = Counter(
        str(result.get("integrity", {}).get("status")) for result in results
    )
    index_statuses = Counter(
        str(result.get("geometry", {}).get("index_grid_status")) for result in results
    )
    world_statuses = Counter(
        str(result.get("geometry", {}).get("world_header_status")) for result in results
    )
    spacing_statuses = Counter(
        str(result.get("geometry", {}).get("header_comparison", {}).get("spacing_status"))
        for result in results
    )
    header_form_statuses = Counter(
        str(result.get("geometry", {}).get("header_comparison", {}).get("header_form_status"))
        for result in results
    )
    header_metadata_statuses = Counter(
        str(result.get("geometry", {}).get("header_comparison", {}).get("header_metadata_status"))
        for result in results
    )
    label_statuses = Counter(str(result.get("labels", {}).get("status")) for result in results)
    exception_count = sum(bool(_audit_exception_reasons(result)) for result in results)
    visual_generated = sum(result.get("visual_qc_status") == "GENERATED_PENDING_MANUAL_REVIEW" for result in results)
    all_downloaded = sum(download_statuses[key] for key in ("downloaded", "already_valid")) == requested
    all_provenance = provenance_statuses.get("FULL_HASH_MATCH", 0) == requested
    all_geometry = index_statuses.get("PASS", 0) == requested and world_statuses.get("PASS", 0) == requested
    all_spacing = spacing_statuses.get("PASS", 0) == requested
    all_header_forms = all(
        header_form_status_allows_native_qc(
            result.get("geometry", {}).get("header_comparison", {}).get("header_form_status")
        )
        for result in results
    )
    all_labels = label_statuses.get("PASS", 0) == requested
    lut_status = str(source_lock.get("lut_provenance", {}).get("status"))
    review_passed = bool(visual_review and visual_review.get("passed") is True)
    if not all_downloaded:
        conclusion = "PARTIAL"
    elif not all_provenance:
        conclusion = "PARTIAL_SOURCE_PROVENANCE"
    elif not all_geometry:
        conclusion = "FAIL"
    elif not all_spacing or not all_header_forms:
        conclusion = "PARTIAL_HEADER_COMPATIBILITY"
    elif lut_status != "PASS" or not all_labels:
        conclusion = "LUT_PROVENANCE_BLOCKED"
    elif not review_passed:
        conclusion = "PASS_PENDING_MANUAL_VISUAL_REVIEW"
    else:
        conclusion = "PASS"

    def counts_line(counter: Mapping[str, int]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"

    lines = [
        "# CT-RATE `ts_total` / ReX validation audit",
        "",
        "This redacted report intentionally contains aggregate evidence only; detailed ",
        "case-level manifests, source objects, and overlays remain in the restricted runtime root.",
        "",
        "## Conclusion",
        "",
        f"**{conclusion}**",
        "",
        "A geometry mismatch is quarantined evidence and never authorizes automatic resampling.",
        "",
        "## Pinned sources",
        "",
        f"- ReXGroundingCT revision: `{manifest['rex_revision']}`",
        f"- CT-RATE revision: `{manifest['ct_rate_revision']}`",
        f"- Approved remote-manifest SHA-256: `{manifest['manifest_sha256']}`",
        f"- Producer LUT status: `{lut_status}`",
        "",
        "## Coverage and provenance",
        "",
        f"- Requested masks: {requested}",
        f"- Download statuses: {counts_line(download_statuses)}",
        f"- Local CT source-provenance statuses: {counts_line(provenance_statuses)}",
        f"- Full gzip/NIfTI integrity statuses: {counts_line(integrity_statuses)}",
        "",
        "## Native geometry",
        "",
        f"- Index-grid statuses: {counts_line(index_statuses)}",
        f"- World/header statuses: {counts_line(world_statuses)}",
        f"- Spacing statuses: {counts_line(spacing_statuses)}",
        f"- qform/sform/unit compatibility statuses: {counts_line(header_form_statuses)}",
        f"- Header metadata statuses: {counts_line(header_metadata_statuses)}",
        f"- Label-QC statuses: {counts_line(label_statuses)}",
        f"- Cases selected as exceptions for visual QC: {exception_count}",
        f"- Visual QC images generated: {visual_generated}",
        "",
        "## Review state",
        "",
        (
            f"- Manual visual review: {'passed' if review_passed else 'not yet recorded'}"
            + (f" (reviewer: {visual_review.get('reviewer')})" if review_passed else "")
        ),
        "",
        "No model, preprocessing, inference, or anatomy-prior integration was modified by this audit.",
    ]
    return "\n".join(lines) + "\n", conclusion


def write_aggregate_report(runtime_root: Path) -> str:
    source_lock, manifest, paths = _require_preflight(runtime_root)
    results_path = paths["private_reports"] / "audit_results.json"
    if not results_path.exists():
        raise RuntimeError("No audit results found; run --audit first")
    results_payload = read_json(results_path)
    results = verified_audit_results(results_payload, manifest, source_lock)
    visual_review = _verified_manual_visual_review(
        paths, manifest, source_lock, results_path, results
    )
    report, conclusion = _aggregate_report(
        source_lock,
        manifest,
        results,
        visual_review,
    )
    report_path = paths["reports"] / "ct_rate_ts_total_rex_val200_audit_report.md"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(report)
    os.replace(temporary, report_path)
    return conclusion


def audit(
    config: Mapping[str, Any],
    runtime_root: Path,
    *,
    derive_lungs: bool,
    hash_local_ct: bool,
) -> dict[str, Any]:
    """Perform source provenance, NIfTI, geometry, label, and optional lung audit."""
    source_lock, manifest, paths = _require_preflight(runtime_root)
    if canonical_json_sha256(config) != source_lock.get("config_canonical_sha256"):
        raise RuntimeError("Live config differs from the sealed preflight config; rerun --preflight")
    sealed_config = read_json(paths["config"] / "canonical_config_snapshot.json")
    if canonical_json_sha256(sealed_config) != source_lock.get("config_canonical_sha256"):
        raise RuntimeError("Sealed config snapshot is invalid; rerun --preflight")
    config = sealed_config
    download_path = paths["manifests"] / "download_manifest.json"
    if not download_path.exists():
        raise RuntimeError("No download manifest found; run --download first")
    download_manifest = read_json(download_path)
    if (
        download_manifest.get("manifest_sha256") != manifest.get("manifest_sha256")
        or download_manifest.get("source_lock_sha256") != source_lock.get("source_lock_sha256")
    ):
        raise RuntimeError("Download manifest does not belong to this sealed source lock")
    environment = execution_environment_record(config)
    write_json_atomic(paths["config"] / "execution_environment.json", environment)
    nib = _import_nibabel()
    geometry_config = config["geometry_audit"]
    lut_status = str(source_lock.get("lut_provenance", {}).get("status"))
    hash_path = paths["private_reports"] / "local_ct_hashes.json"
    hash_payload = read_json(hash_path) if hash_path.exists() else {"cases": {}}
    hash_cache: dict[str, Any] = dict(hash_payload.get("cases", {}))
    results: list[dict[str, Any]] = []
    cases = list(manifest["cases"])
    for index, row_raw in enumerate(cases, start=1):
        row = dict(row_raw)
        volume_name = str(row["volume_name"])
        download_result = dict(download_manifest.get("cases", {}).get(volume_name, {}))
        result: dict[str, Any] = {
            "metadata_index": row["metadata_index"],
            "volume_name": volume_name,
            "download_status": download_result.get("status", "not_attempted"),
            "source_provenance": {},
            "integrity": {},
            "geometry": {},
            "labels": {},
            "lungs": {},
            "visual_qc_status": "NOT_GENERATED",
            "error": None,
        }
        try:
            result["source_provenance"] = local_ct_provenance(
                row, hash_cache, hash_local_ct=hash_local_ct
            )
        except Exception as exc:
            result["source_provenance"] = {
                "status": "LOCAL_CT_PROVENANCE_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            seg_path = Path(str(row["local_ts_total_path"]))
            remote_seg = dict(row["remote_ts_total"])
            integrity = validate_mask_file(
                seg_path, remote_seg.get("size"), remote_seg.get("lfs_sha256")
            )
            result["integrity"] = integrity
            if integrity.get("status") != "VALID":
                result["error"] = "Segmentation failed integrity validation; geometry is quarantined"
                results.append(result)
                continue
            ct_path = Path(str(row["local_ct_path"]))
            if not ct_path.is_file():
                result["error"] = "Local CT is absent; geometry is unavailable"
                results.append(result)
                continue
            ct_image = nib.load(str(ct_path))
            seg_image = nib.load(str(seg_path))
            if len(ct_image.shape) != 3 or len(seg_image.shape) != 3:
                raise ValueError(
                    f"Native geometry audit requires 3-D inputs: ct={ct_image.shape} seg={seg_image.shape}"
                )
            ct_header = _nifti_header_record(ct_image, nib)
            seg_header = _nifti_header_record(seg_image, nib)
            geometry = affine_geometry(
                ct_image.affine,
                seg_image.affine,
                ct_image.shape,
                seg_image.shape,
                float(geometry_config["index_tolerance_vox"]),
                float(geometry_config["corner_tolerance_mm"]),
            )
            geometry["ct_header"] = ct_header
            geometry["seg_header"] = seg_header
            geometry["header_comparison"] = _header_comparison(
                ct_header,
                seg_header,
                float(geometry_config["corner_tolerance_mm"]),
            )
            result["geometry"] = geometry
            seg_data = np.asanyarray(seg_image.dataobj)
            result["labels"] = label_qc(seg_data, lut_status)
            if (
                derive_lungs
                and lut_status == "PASS"
                and result["labels"].get("status") == "PASS"
                and result["source_provenance"].get("status") == "FULL_HASH_MATCH"
                and geometry.get("index_grid_status") == "PASS"
                and geometry.get("world_header_status") == "PASS"
                and geometry["header_comparison"].get("spacing_status") == "PASS"
                and header_form_status_allows_native_qc(
                    geometry["header_comparison"].get("header_form_status")
                )
            ):
                for side, label_ids in LUNG_LABELS.items():
                    mask = np.isin(seg_data, label_ids)
                    metrics = lung_metrics(mask, seg_image.affine)
                    result["lungs"][side] = metrics
                    destination = paths["derived_lungs"] / side / str(row["remote_ts_total_path"])
                    _save_binary_lung(nib, mask, seg_image, destination)
                    metrics["derived_path"] = str(destination)
            elif derive_lungs:
                result["lungs"] = {"status": "SKIPPED_PROVENANCE_OR_GEOMETRY_OR_LABEL_GATE"}
        except Exception as exc:  # Record every case-level failure; do not silently omit it.
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc(limit=3)
        results.append(result)
        if index % 10 == 0 or result.get("error"):
            print(f"[{index}/{len(cases)}] audit progress", flush=True)
        write_json_atomic(
            hash_path,
            {"updated_at": utc_now(), "cases": hash_cache},
        )

    if derive_lungs and lut_status == "PASS":
        outlier_policy = add_lung_outlier_flags(results)
    else:
        outlier_policy = {"status": "SKIPPED", "reason": f"lut_status={lut_status}"}
    results_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "config_canonical_sha256": source_lock["config_canonical_sha256"],
        "script_sha256": source_lock["script_sha256"],
        "execution_environment": environment,
        "generated_at": utc_now(),
        "lut_status": lut_status,
        "local_ct_hashing": "enabled" if hash_local_ct else "disabled",
        "lung_outlier_policy": outlier_policy,
        "cases": results,
    }
    write_json_atomic(hash_path, {"updated_at": utc_now(), "cases": hash_cache})
    results_path = paths["private_reports"] / "audit_results.json"
    write_json_atomic(results_path, results_payload)
    write_csv_atomic(
        paths["private_reports"] / "case_audit.csv",
        _case_result_csv_rows(results),
        (
            "volume_name",
            "download_status",
            "source_provenance_status",
            "integrity_status",
            "shape_identical",
            "index_grid_status",
            "world_header_status",
            "max_corner_residual_mm",
            "spacing_status",
            "header_form_status",
            "header_metadata_status",
            "label_status",
            "lung_left_volume_ml",
            "lung_right_volume_ml",
            "lung_left_flags",
            "lung_right_flags",
            "visual_qc_status",
            "error",
        ),
    )
    conclusion = write_aggregate_report(runtime_root)
    return {"results": results_payload, "conclusion": conclusion}


def _deterministic_visual_selection(
    results: Sequence[Mapping[str, Any]], representative_count: int
) -> list[tuple[Mapping[str, Any], list[str]]]:
    """Select all exceptions plus a stable hash-ordered representative subset."""
    if representative_count < 0:
        raise ValueError("representative_count must be nonnegative")
    exception_by_name = {
        str(result["volume_name"]): _audit_exception_reasons(result)
        for result in results
        if _audit_exception_reasons(result)
    }
    ordered = sorted(
        results,
        key=lambda result: hashlib.sha256(str(result["volume_name"]).encode()).hexdigest(),
    )
    selected_names = set(exception_by_name)
    selected_names.update(str(result["volume_name"]) for result in ordered[:representative_count])
    output: list[tuple[Mapping[str, Any], list[str]]] = []
    for result in sorted(results, key=lambda item: int(item["metadata_index"])):
        name = str(result["volume_name"])
        if name in selected_names:
            reasons = exception_by_name.get(name, ["deterministic_representative_sample"])
            output.append((result, reasons))
    return output


def _central_lung_index(left: np.ndarray, right: np.ndarray, shape: Sequence[int]) -> tuple[int, int, int]:
    coordinates = np.argwhere(np.logical_or(left, right))
    if coordinates.size == 0:
        return tuple(int(size // 2) for size in shape[:3])
    median = np.median(coordinates, axis=0).astype(np.int64)
    return tuple(int(value) for value in median)


def _draw_slice(
    axis: Any,
    ct_slice: np.ndarray,
    left_slice: np.ndarray,
    right_slice: np.ndarray,
    title: str,
    horizontal_axis: str,
    vertical_axis: str,
    vmin: float,
    vmax: float,
) -> None:
    # Inputs are RAS-canonical arrays.  Transpose makes the horizontal/vertical
    # directions explicit instead of relying on an unlabelled display rotation.
    axis.imshow(ct_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
    if np.any(left_slice):
        axis.contour(left_slice.T.astype(np.uint8), levels=[0.5], colors=["#18a8ff"], linewidths=0.7)
    if np.any(right_slice):
        axis.contour(right_slice.T.astype(np.uint8), levels=[0.5], colors=["#ff4f70"], linewidths=0.7)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(horizontal_axis, fontsize=8)
    axis.set_ylabel(vertical_axis, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def render_visual_qc(
    ct_image: Any,
    left: np.ndarray,
    right: np.ndarray,
    output_path: Path,
) -> None:
    """Render RAS-canonical axial/coronal/sagittal overlays without a case identifier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct_data = np.asanyarray(ct_image.dataobj)
    if ct_data.ndim != 3:
        raise ValueError(f"Expected 3-D CT image, got {ct_data.shape}")
    finite = ct_data[np.isfinite(ct_data)]
    if finite.size == 0:
        raise ValueError("CT contains no finite voxels for visual QC")
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if vmin == vmax:
        vmin, vmax = float(vmin) - 1.0, float(vmax) + 1.0
    x, y, z = _central_lung_index(left, right, ct_data.shape)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _draw_slice(
        axes[0], ct_data[:, :, z], left[:, :, z], right[:, :, z], "axial (RAS)", "R →", "A →", float(vmin), float(vmax)
    )
    _draw_slice(
        axes[1], ct_data[:, y, :], left[:, y, :], right[:, y, :], "coronal (RAS)", "R →", "S →", float(vmin), float(vmax)
    )
    _draw_slice(
        axes[2], ct_data[x, :, :], left[x, :, :], right[x, :, :], "sagittal (RAS)", "A →", "S →", float(vmin), float(vmax)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output_path)


def visual_qc(runtime_root: Path, representative_count: int) -> dict[str, Any]:
    """Generate overlays for a deterministic sample and every audit exception."""
    source_lock, manifest, paths = _require_preflight(runtime_root)
    if source_lock.get("lut_provenance", {}).get("status") != "PASS":
        raise RuntimeError("Visual lung-overlay QC is blocked because LUT provenance did not pass")
    results_path = paths["private_reports"] / "audit_results.json"
    if not results_path.exists():
        raise RuntimeError("No audit results found; run --audit --derive-lungs first")
    payload = read_json(results_path)
    results = verified_audit_results(payload, manifest, source_lock)
    nib = _import_nibabel()
    selection = _deterministic_visual_selection(results, representative_count)
    case_by_name = {str(row["volume_name"]): row for row in manifest["cases"]}
    selected_manifest: list[dict[str, Any]] = []
    for index, (result, reasons) in enumerate(selection, start=1):
        name = str(result["volume_name"])
        row = case_by_name[name]
        left_path = paths["derived_lungs"] / "left" / str(row["remote_ts_total_path"])
        right_path = paths["derived_lungs"] / "right" / str(row["remote_ts_total_path"])
        try:
            ct_native = nib.load(str(row["local_ct_path"]))
            if left_path.is_file() and right_path.is_file():
                left_native = nib.load(str(left_path))
                right_native = nib.load(str(right_path))
                overlay_source = "saved_native_derivation"
            else:
                # For quarantined exceptions, build an in-memory display-only
                # lobe union.  It is never saved or used downstream.
                seg_native = nib.load(str(row["local_ts_total_path"]))
                seg_data = np.asanyarray(seg_native.dataobj)
                if seg_data.ndim != 3 or seg_data.shape != ct_native.shape[:3]:
                    raise ValueError(
                        "No same-index display overlay is possible for a source grid shape mismatch"
                    )
                header = seg_native.header.copy()
                header.set_data_dtype(np.uint8)
                left_native = nib.Nifti1Image(
                    np.isin(seg_data, LUNG_LABELS["left"]).astype(np.uint8),
                    seg_native.affine,
                    header,
                )
                right_native = nib.Nifti1Image(
                    np.isin(seg_data, LUNG_LABELS["right"]).astype(np.uint8),
                    seg_native.affine,
                    header,
                )
                overlay_source = "transient_exception_display_only"
            ct_image = nib.as_closest_canonical(ct_native)
            left_image = nib.as_closest_canonical(left_native)
            right_image = nib.as_closest_canonical(right_native)
            left = np.asanyarray(left_image.dataobj).astype(bool, copy=False)
            right = np.asanyarray(right_image.dataobj).astype(bool, copy=False)
            if left.shape != ct_image.shape[:3] or right.shape != ct_image.shape[:3]:
                raise ValueError(
                    f"Derived mask/CT shape mismatch: ct={ct_image.shape} left={left.shape} right={right.shape}"
                )
            output_path = paths["visual_qc"] / f"{Path(name).stem}.png"
            render_visual_qc(ct_image, left, right, output_path)
            result["visual_qc_status"] = "GENERATED_PENDING_MANUAL_REVIEW"
            result["visual_qc_path"] = str(output_path)
            result.pop("visual_qc_error", None)
            selected_manifest.append(
                {
                    "volume_name": name,
                    "status": result["visual_qc_status"],
                    "path": str(output_path),
                    "overlay_source": overlay_source,
                    "reasons": reasons,
                }
            )
        except Exception as exc:
            result["visual_qc_status"] = "GENERATE_ERROR"
            result["visual_qc_error"] = f"{type(exc).__name__}: {exc}"
            selected_manifest.append(
                {
                    "volume_name": name,
                    "status": result["visual_qc_status"],
                    "reasons": reasons,
                    "error": result["visual_qc_error"],
                }
            )
        if index % 10 == 0 or index == len(selection):
            print(f"[{index}/{len(selection)}] visual-QC progress", flush=True)
    payload["visual_qc_generated_at"] = utc_now()
    payload["visual_qc_representative_count"] = representative_count
    payload["cases"] = results
    write_json_atomic(results_path, payload)
    audit_results_sha256 = sha256_file(results_path)
    write_csv_atomic(
        paths["private_reports"] / "case_audit.csv",
        _case_result_csv_rows(results),
        (
            "volume_name",
            "download_status",
            "source_provenance_status",
            "integrity_status",
            "shape_identical",
            "index_grid_status",
            "world_header_status",
            "max_corner_residual_mm",
            "spacing_status",
            "header_form_status",
            "header_metadata_status",
            "label_status",
            "lung_left_volume_ml",
            "lung_right_volume_ml",
            "lung_left_flags",
            "lung_right_flags",
            "visual_qc_status",
            "error",
        ),
    )
    write_json_atomic(
        paths["private_reports"] / "visual_qc_manifest.json",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "config_canonical_sha256": source_lock["config_canonical_sha256"],
            "script_sha256": source_lock["script_sha256"],
            "audit_results_sha256": audit_results_sha256,
            "generated_at": utc_now(),
            "representative_count": representative_count,
            "selected": selected_manifest,
        },
    )
    template_path = paths["reports"] / "manual_visual_review_template.md"
    if not template_path.exists():
        template_path.write_text(
            "# Manual visual-QC review\n\n"
            "Review the external `visual_qc/` images and then record the result with "
            "`--record-visual-review --reviewer <name> --review-notes <summary> ` "
            "and either `--manual-review-passed` or `--manual-review-failed`.\n"
        )
    conclusion = write_aggregate_report(runtime_root)
    return {"selected": len(selection), "conclusion": conclusion}


def _validated_visual_qc_manifest(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    results_path: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    require_rendered_outputs: bool,
) -> dict[str, Any]:
    """Bind visual-review inputs to the exact sealed results and selected cases."""
    visual_path = paths["private_reports"] / "visual_qc_manifest.json"
    if not visual_path.exists():
        raise RuntimeError("No visual-QC manifest found; run --visual-qc first")
    visual = read_json(visual_path)
    required_hashes = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "config_canonical_sha256": source_lock["config_canonical_sha256"],
        "script_sha256": source_lock["script_sha256"],
        "audit_results_sha256": sha256_file(results_path),
    }
    for key, expected in required_hashes.items():
        if visual.get(key) != expected:
            raise RuntimeError(f"Visual-QC manifest {key} does not match the sealed audit state")
    count = int(visual.get("representative_count", -1))
    expected_selection = _deterministic_visual_selection(results, count)
    expected_names = [str(result["volume_name"]) for result, _ in expected_selection]
    selected = visual.get("selected")
    if not isinstance(selected, list):
        raise RuntimeError("Visual-QC manifest has no selected-case list")
    selected_names = [str(item.get("volume_name")) for item in selected if isinstance(item, Mapping)]
    if selected_names != expected_names or len(set(selected_names)) != len(selected_names):
        raise RuntimeError("Visual-QC selection does not match the deterministic audit selection")
    if require_rendered_outputs:
        visual_root = paths["visual_qc"].resolve()
        for item in selected:
            if item.get("status") != "GENERATED_PENDING_MANUAL_REVIEW":
                raise RuntimeError("A selected visual-QC case lacks a usable generated overlay")
            output = item.get("path")
            if not isinstance(output, str) or not Path(output).is_file():
                raise RuntimeError("A selected visual-QC overlay file is missing")
            try:
                Path(output).resolve().relative_to(visual_root)
            except ValueError as exc:
                raise RuntimeError("Visual-QC output lies outside the audit visual-QC root") from exc
    return visual


def _verified_manual_visual_review(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    results_path: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    review_path = paths["reports"] / "manual_visual_review.json"
    if not review_path.exists():
        return None
    review = read_json(review_path)
    visual = _validated_visual_qc_manifest(
        paths,
        manifest,
        source_lock,
        results_path,
        results,
        require_rendered_outputs=bool(review.get("passed") is True),
    )
    required_hashes = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "config_canonical_sha256": source_lock["config_canonical_sha256"],
        "script_sha256": source_lock["script_sha256"],
        "audit_results_sha256": sha256_file(results_path),
        "visual_qc_manifest_sha256": sha256_file(paths["private_reports"] / "visual_qc_manifest.json"),
    }
    for key, expected in required_hashes.items():
        if review.get(key) != expected:
            raise RuntimeError(f"Manual visual review {key} does not match the sealed review inputs")
    if review.get("visual_qc_selection_count") != len(visual["selected"]):
        raise RuntimeError("Manual visual review selection count does not match visual-QC manifest")
    return review


def record_visual_review(
    runtime_root: Path,
    reviewer: str | None,
    notes: str | None,
    passed: bool | None,
) -> str:
    """Record a human review state; this cannot be inferred from generated images."""
    if not reviewer or not notes or passed is None:
        raise ValueError(
            "A reviewer, review notes, and an explicit passed/failed result are required"
        )
    source_lock, manifest, paths = _require_preflight(runtime_root)
    results_path = paths["private_reports"] / "audit_results.json"
    if not results_path.exists():
        raise RuntimeError("No audit results found; run --audit and --visual-qc first")
    results = verified_audit_results(read_json(results_path), manifest, source_lock)
    visual_manifest = _validated_visual_qc_manifest(
        paths,
        manifest,
        source_lock,
        results_path,
        results,
        require_rendered_outputs=bool(passed),
    )
    visual_manifest_path = paths["private_reports"] / "visual_qc_manifest.json"
    write_json_atomic(
        paths["reports"] / "manual_visual_review.json",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "config_canonical_sha256": source_lock["config_canonical_sha256"],
            "script_sha256": source_lock["script_sha256"],
            "audit_results_sha256": sha256_file(results_path),
            "visual_qc_manifest_sha256": sha256_file(visual_manifest_path),
            "visual_qc_selection_count": len(visual_manifest["selected"]),
            "recorded_at": utc_now(),
            "reviewer": reviewer,
            "notes": notes,
            "passed": passed,
        },
    )
    return write_aggregate_report(runtime_root)


def _print_preflight_summary(preflight_result: Mapping[str, Any]) -> None:
    manifest = preflight_result["manifest"]
    source_lock = preflight_result["source_lock"]
    cases = list(manifest["cases"])
    summary = source_lock["summary"]
    print("Preflight complete (no masks downloaded).")
    print(f"Requested: {summary['requested_masks']}")
    print(f"Remote ts_total bytes: {summary['remote_ts_total_bytes']}")
    print(f"CT-RATE split counts: {summary['split_counts']}")
    print(f"LUT provenance: {source_lock['lut_provenance']['status']}")
    print(f"Manifest approval SHA-256: {manifest['manifest_sha256']}")
    print("First 10 resolved rows:")
    for row in cases[:10]:
        print(
            " | ".join(
                (
                    str(row["volume_name"]),
                    str(row["ct_rate_fixed_split"]),
                    str(row["local_ct_path"]),
                    str(row["remote_ts_total_path"]),
                    "exists=yes",
                )
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    stages = parser.add_argument_group("exactly one stage is required")
    stages.add_argument("--preflight", action="store_true", help="Resolve and lock sources; do not download masks.")
    stages.add_argument("--download", action="store_true", help="Download only the preflight-approved masks.")
    stages.add_argument("--audit", action="store_true", help="Audit downloaded masks against native local CTs.")
    stages.add_argument("--visual-qc", action="store_true", help="Generate deterministic native-grid lung overlays.")
    stages.add_argument("--record-visual-review", action="store_true", help="Record a human visual-review outcome.")
    parser.add_argument(
        "--approve-manifest-sha256",
        help="Required exact preflight manifest SHA-256 for --download.",
    )
    parser.add_argument(
        "--derive-lungs",
        action="store_true",
        help="With --audit, derive native left/right lungs only after LUT evidence passes.",
    )
    parser.add_argument(
        "--skip-local-ct-hash",
        action="store_true",
        help="Do not perform the default full local CT content hashes (not suitable for final PASS).",
    )
    parser.add_argument("--representative-count", type=int, default=20)
    parser.add_argument("--reviewer")
    parser.add_argument("--review-notes")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--manual-review-passed", action="store_true")
    review_group.add_argument("--manual-review-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_stages = sum(
        bool(value)
        for value in (
            args.preflight,
            args.download,
            args.audit,
            args.visual_qc,
            args.record_visual_review,
        )
    )
    if selected_stages != 1:
        parser.error("Choose exactly one stage")
    if args.derive_lungs and not args.audit:
        parser.error("--derive-lungs is valid only with --audit")
    if args.skip_local_ct_hash and not args.audit:
        parser.error("--skip-local-ct-hash is valid only with --audit")
    if (args.manual_review_passed or args.manual_review_failed) and not args.record_visual_review:
        parser.error("Manual review outcome flags require --record-visual-review")
    config = load_config(args.config)
    runtime_root = args.runtime_root
    try:
        if args.preflight:
            _print_preflight_summary(preflight(config, runtime_root))
        elif args.download:
            outcome = download(config, runtime_root, args.approve_manifest_sha256)
            print(f"Download status summary: {outcome['summary']}")
        elif args.audit:
            outcome = audit(
                config,
                runtime_root,
                derive_lungs=args.derive_lungs,
                hash_local_ct=not args.skip_local_ct_hash,
            )
            print(f"Audit conclusion: {outcome['conclusion']}")
        elif args.visual_qc:
            outcome = visual_qc(runtime_root, args.representative_count)
            print(f"Visual QC selected: {outcome['selected']}; conclusion: {outcome['conclusion']}")
        else:
            passed: bool | None
            if args.manual_review_passed:
                passed = True
            elif args.manual_review_failed:
                passed = False
            else:
                passed = None
            conclusion = record_visual_review(runtime_root, args.reviewer, args.review_notes, passed)
            print(f"Recorded manual visual review; conclusion: {conclusion}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
