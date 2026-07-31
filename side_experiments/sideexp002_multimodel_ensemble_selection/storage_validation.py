"""Storage-only validation helpers without model or medical-image dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def case_mask_storage_proof(
    logits: np.ndarray,
    case_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify or conservatively reconstruct exact same-pass mask preservation."""
    recorded = case_manifest.get("same_pass_mask_mismatch_voxels")
    if recorded is not None:
        mismatches = int(recorded)
        return {
            "passed": mismatches == 0,
            "mismatch_voxels": mismatches,
            "method": "export_time_exact_threshold_zero_comparison",
        }

    # Older exports predate the explicit comparison. A finite cast/clamp can
    # only change a threshold-zero mask when a negative value rounds to signed
    # zero. If the stored array contains no zeros, its signs are therefore an
    # exact proof that the same-pass threshold mask was preserved.
    zero_voxels = sum(
        int(np.count_nonzero(logits[index] == 0))
        for index in range(logits.shape[0])
    )
    return {
        "passed": zero_voxels == 0,
        "mismatch_voxels": None,
        "method": (
            "legacy_zero_free_sign_preservation"
            if zero_voxels == 0
            else "legacy_unproven_due_to_stored_zero_voxels"
        ),
        "stored_zero_voxels": zero_voxels,
    }


def validate_existing_case(
    array_path: Path,
    metadata_path: Path,
    expected_dtype: str,
    expected_findings: int,
) -> dict[str, Any] | None:
    """Return complete case metadata only when a resumable array is intact."""
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("status") != "complete"
        or metadata.get("dtype") != expected_dtype
        or str(array.dtype) != expected_dtype
        or array.ndim != 4
        or array.shape[0] != expected_findings
        or list(array.shape) != metadata.get("shape")
        or int(array_path.stat().st_size) != int(metadata.get("npy_bytes", -1))
    ):
        return None
    return metadata
