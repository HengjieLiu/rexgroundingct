#!/usr/bin/env python3
"""Storage-publication tests; skipped outside the GPU image dependencies."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import fresh_export
except ModuleNotFoundError as exc:  # Host control plane intentionally has no torch/nibabel.
    fresh_export = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(fresh_export is None, f"inference dependencies unavailable: {IMPORT_ERROR}")
class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "cache"
        self.progress_path = Path(self.temporary.name) / "progress.json"
        self.case = {"name": "case.nii.gz", "findings": {"0": "finding"}}
        self.progress = {"candidate_id": "candidate", "status": "exporting"}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def stage(self, array: np.ndarray) -> None:
        array_path, metadata_path = fresh_export._stage_paths(self.root, self.case["name"])
        fresh_export.atomic_save_npy(array_path, array.astype(np.float32))
        fresh_export.atomic_write_json(
            metadata_path,
            {
                "name": self.case["name"],
                "status": "complete",
                "array_path": str(self.root / "cases" / f"{self.case['name']}.npy"),
                "npy_bytes": array_path.stat().st_size,
                "array_sha256": fresh_export.array_sha256(array.astype(np.float32)),
                "shape": list(array.shape),
                "dtype": "float32",
                "same_pass_reference": {
                    "finding_dice": [1.0],
                    "hits": 1,
                    "findings": 1,
                },
            },
        )

    def test_exact_sign_cache_publishes_float16(self) -> None:
        self.stage(np.asarray([[[[-1.0, 0.25]]]], dtype=np.float32))
        dtype, records, _progress = fresh_export._publish(
            self.root, [self.case], self.progress_path, self.progress
        )
        self.assertEqual(dtype, "float16")
        self.assertEqual(records[0]["same_pass_mask_mismatch_voxels"], 0)
        published = np.load(self.root / "cases/case.nii.gz.npy", allow_pickle=False)
        self.assertEqual(str(published.dtype), "float16")

    def test_sign_loss_keeps_float32_without_reinference(self) -> None:
        self.stage(np.asarray([[[[-1e-12, 0.25]]]], dtype=np.float32))
        dtype, records, _progress = fresh_export._publish(
            self.root, [self.case], self.progress_path, self.progress
        )
        self.assertEqual(dtype, "float32")
        self.assertEqual(records[0]["same_pass_mask_mismatch_voxels"], 0)
        published = np.load(self.root / "cases/case.nii.gz.npy", allow_pickle=False)
        self.assertEqual(str(published.dtype), "float32")
        self.assertLess(float(published.flat[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
