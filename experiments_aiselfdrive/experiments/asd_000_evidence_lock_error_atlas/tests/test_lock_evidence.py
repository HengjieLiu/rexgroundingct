from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from lock_evidence import (  # noqa: E402
    EvidenceLockError,
    atomic_write_csv,
    atomic_write_json,
    build_lineage_evidence,
    hash_file,
    hash_tree,
    hash_tree_manifest,
    iter_files_sorted,
    load_protocol,
    recompute_stored_mask_metrics,
    sha256_file,
    verify_evaluation_summary,
)


class HashAndAtomicWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_sha256_file_streams_and_verifies_expected_metadata(self) -> None:
        evidence = self.root / "evidence.bin"
        content = bytes(range(251)) * 37
        evidence.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()

        self.assertEqual(sha256_file(evidence, chunk_size=17), expected)
        record = hash_file(
            evidence,
            expected_sha256=expected,
            expected_size_bytes=len(content),
            chunk_size=31,
        )
        self.assertEqual(record["sha256"], expected)
        self.assertEqual(record["size_bytes"], len(content))
        self.assertTrue(record["stable_read"])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            sha256_file(evidence, chunk_size=0)
        with self.assertRaisesRegex(EvidenceLockError, "SHA-256 mismatch"):
            hash_file(evidence, expected_sha256="0" * 64)

    def test_tree_traversal_and_manifest_are_deterministic(self) -> None:
        (self.root / "z").mkdir()
        (self.root / "a").mkdir()
        (self.root / "z" / "b.bin").write_bytes(b"b")
        (self.root / "a" / "z.bin").write_bytes(b"z")
        (self.root / "a" / "a.bin").write_bytes(b"a")
        (self.root / "middle.bin").write_bytes(b"m")

        relative = [
            path.relative_to(self.root).as_posix()
            for path in iter_files_sorted(self.root)
        ]
        self.assertEqual(
            relative,
            ["a/a.bin", "a/z.bin", "middle.bin", "z/b.bin"],
        )
        first = hash_tree(self.root, chunk_size=1)
        second = hash_tree(self.root, chunk_size=3)
        self.assertEqual(first, second)
        manifest = hash_tree_manifest(self.root, chunk_size=2)
        self.assertEqual(manifest["file_count"], 4)
        self.assertEqual(
            manifest["tree_sha256"],
            hashlib.sha256(
                json.dumps(
                    first,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def test_atomic_json_is_deterministic_and_failure_preserves_target(self) -> None:
        destination = self.root / "nested" / "evidence.json"
        atomic_write_json(destination, {"z": [2, 1], "a": "value"})
        expected = '{\n  "a": "value",\n  "z": [\n    2,\n    1\n  ]\n}\n'
        self.assertEqual(destination.read_text(encoding="utf-8"), expected)

        with self.assertRaisesRegex(ValueError, "strict JSON"):
            atomic_write_json(destination, {"invalid": float("nan")})
        self.assertEqual(destination.read_text(encoding="utf-8"), expected)
        self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_atomic_csv_has_stable_columns_and_preserves_target_on_error(self) -> None:
        destination = self.root / "evidence.csv"
        atomic_write_csv(
            destination,
            [{"b": 2, "a": "first"}, {"a": "second", "b": 3}],
        )
        with destination.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, [["a", "b"], ["first", "2"], ["second", "3"]])
        original = destination.read_bytes()

        with self.assertRaisesRegex(ValueError, "invalid CSV row"):
            atomic_write_csv(
                destination,
                [{"a": 1, "unexpected": 2}],
                fieldnames=["a"],
            )
        self.assertEqual(destination.read_bytes(), original)


class SummaryVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.summary_path = self.root / "summary.json"
        self.summary = {
            "label": "synthetic continuity evaluation threshold 0.5",
            "mean_global_dice_per_finding": 0.25,
            "hit_rate": 0.75,
            "total_cases": 2,
            "total_findings": 4,
            "total_hits": 3,
            "total_misses": 1,
            "params": {"global_hit_thr": 0.1, "global_only": True},
        }
        self.summary_path.write_text(
            json.dumps(self.summary), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_summary_metrics_and_operating_point_are_verified(self) -> None:
        result = verify_evaluation_summary(
            self.summary_path,
            expected_dice=0.250004,
            dice_tolerance=0.000005,
            expected_hits=3,
            expected_findings=4,
            expected_cases=2,
            expected_threshold=0.5,
            expected_hit_threshold=0.1,
        )
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["dice_absolute_error"], 0.000004)
        self.assertEqual(result["metrics"]["hits"], 3)
        self.assertEqual(result["metrics"]["threshold"], 0.5)

    def test_summary_metric_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(EvidenceLockError, "Dice mismatch"):
            verify_evaluation_summary(
                self.summary_path,
                expected_dice=0.3,
                dice_tolerance=0.001,
                expected_hits=3,
                expected_findings=4,
            )
        with self.assertRaisesRegex(EvidenceLockError, "hit-count mismatch"):
            verify_evaluation_summary(
                self.summary_path,
                expected_dice=0.25,
                dice_tolerance=0.0,
                expected_hits=2,
                expected_findings=4,
            )
        with self.assertRaisesRegex(EvidenceLockError, "prediction threshold mismatch"):
            verify_evaluation_summary(
                self.summary_path,
                expected_dice=0.25,
                dice_tolerance=0.0,
                expected_hits=3,
                expected_findings=4,
                expected_threshold=0.4,
            )

    def test_missing_operating_point_is_not_inferred(self) -> None:
        self.summary.pop("label")
        self.summary_path.write_text(
            json.dumps(self.summary), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            EvidenceLockError, "prediction threshold is not recorded unambiguously"
        ):
            verify_evaluation_summary(
                self.summary_path,
                expected_dice=0.25,
                dice_tolerance=0.0,
                expected_hits=3,
                expected_findings=4,
                expected_threshold=0.5,
            )


class ProtocolAndLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_synthetic_lineage(self) -> tuple[dict, dict]:
        report = self.root / "technical_report.md"
        report.write_text("historical evidence\n", encoding="utf-8")

        cohort = self.root / "continuity.json"
        cohort_payload = {
            "test": [
                {
                    "name": "case_b.npy",
                    "findings": ["one"],
                    "seg_path": "case_b.npy",
                },
                {
                    "name": "case_a.npy",
                    "findings": ["two", "three"],
                    "seg_path": "case_a.npy",
                },
            ]
        }
        cohort.write_text(json.dumps(cohort_payload), encoding="utf-8")

        checkpoint = self.root / "baseline.pth"
        checkpoint.write_bytes(b"baseline checkpoint")
        comparator_checkpoint = self.root / "comparator.pth"
        comparator_checkpoint.write_bytes(b"comparator checkpoint")

        predictions = self.root / "predictions"
        predictions.mkdir()
        segmentations = self.root / "segmentations"
        segmentations.mkdir()
        prediction_arrays = {
            "case_b.npy": np.asarray([[[[1]], [[0]]]], dtype=np.uint8),
            "case_a.npy": np.asarray(
                [[[[1]], [[0]]], [[[0]], [[0]]]], dtype=np.uint8
            ),
        }
        ground_truth_arrays = {
            "case_b.npy": np.asarray([[[[1]], [[0]]]], dtype=np.uint8),
            "case_a.npy": np.asarray(
                [[[[0]], [[1]]], [[[0]], [[0]]]], dtype=np.uint8
            ),
        }
        for name, array in prediction_arrays.items():
            np.save(predictions / name, array, allow_pickle=False)
        for name, array in ground_truth_arrays.items():
            np.save(segmentations / name, array, allow_pickle=False)
        mismatch_dice = 1e-6 / (2 + 1e-6)
        expected_baseline_dice = (1.0 + mismatch_dice + 1.0) / 3

        def write_summary(path: Path, dice: float, hits: int) -> None:
            payload = {
                "label": "synthetic continuity evaluation threshold 0.5",
                "mean_global_dice_per_finding": dice,
                "hit_rate": hits / 3,
                "total_cases": 2,
                "total_findings": 3,
                "total_hits": hits,
                "total_misses": 3 - hits,
                "params": {"global_hit_thr": 0.1},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

        summary = self.root / "baseline_summary.json"
        write_summary(summary, expected_baseline_dice, 2)
        comparator_summary = self.root / "comparator_summary.json"
        write_summary(comparator_summary, 0.5, 3)

        bank = self.root / "embeddings.npz"
        np.savez(
            bank,
            labels=np.asarray(["first", "second"], dtype="U6"),
            vectors=np.arange(6, dtype=np.float16).reshape(2, 3),
        )

        protocol = {
            "schema_version": "test-1",
            "experiment_id": "synthetic_lineage",
            "baseline": {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions": predictions.name,
                "summary": summary.name,
                "expected_dice": expected_baseline_dice,
                "dice_tolerance": 0.000001,
                "expected_hits": 2,
                "expected_findings": 3,
                "threshold": 0.5,
                "hit_threshold": 0.1,
            },
            "s3_comparator": {
                "checkpoint": comparator_checkpoint.name,
                "checkpoint_sha256": sha256_file(comparator_checkpoint),
                "summary": comparator_summary.name,
                "expected_dice": 0.5,
                "expected_hits": 3,
            },
            "embedding_bank": {
                "path": bank.name,
                "expected_labels": 2,
                "expected_shape": [2, 3],
                "expected_dtype": "float16",
                "sha256": sha256_file(bank),
            },
            "cohorts": {
                "val200": {
                    "path": cohort.name,
                    "sha256": sha256_file(cohort),
                    "cases": 2,
                    "findings": 3,
                },
                # This sealed path is deliberately nonexistent. Lineage locking
                # must neither open nor derive metrics from it.
                "val120_output": "sealed/not-readable.json",
            },
            "data": {"segmentation_root": segmentations.name},
        }
        source_report = {
            "path": report.name,
            "sha256": sha256_file(report),
            "evidence_availability": "historical_only",
        }
        return protocol, source_report

    def test_load_protocol_requires_a_yaml_mapping(self) -> None:
        config = self.root / "protocol.yaml"
        config.write_text(
            "experiment_id: synthetic\nnested:\n  enabled: true\n",
            encoding="utf-8",
        )
        self.assertEqual(
            load_protocol(config),
            {"experiment_id": "synthetic", "nested": {"enabled": True}},
        )
        config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceLockError, "root must be a mapping"):
            load_protocol(config)

    def test_control_plane_report_declaration_is_required_precisely(self) -> None:
        with self.assertRaisesRegex(
            EvidenceLockError,
            "source_report declaration is required",
        ):
            build_lineage_evidence(
                {"schema_version": "test", "experiment_id": "synthetic"},
                self.root,
            )

    def test_build_lineage_evidence_verifies_declared_local_inputs(self) -> None:
        protocol, source_report = self._write_synthetic_lineage()
        first = build_lineage_evidence(
            protocol,
            repo_root=self.root,
            source_report=source_report,
            mask_array_loader=lambda path: np.load(path, allow_pickle=False),
            mask_geometry_verifier=lambda prediction, ground_truth: {
                "verified": True,
                "shape_fxyz": list(np.load(prediction).shape),
                "affine_sha256": "0" * 64,
                "axis_codes": ["R", "A", "S"],
                "qform_code": 0,
                "sform_code": 1,
            },
        )
        second = build_lineage_evidence(
            protocol,
            repo_root=self.root,
            source_report=source_report,
            mask_array_loader=lambda path: np.load(path, allow_pickle=False),
            mask_geometry_verifier=lambda prediction, ground_truth: {
                "verified": True,
                "shape_fxyz": list(np.load(prediction).shape),
                "affine_sha256": "0" * 64,
                "axis_codes": ["R", "A", "S"],
                "qform_code": 0,
                "sform_code": 1,
            },
        )

        self.assertEqual(first, second)
        self.assertTrue(first["verified"])
        self.assertTrue(all(first["checks"].values()))
        self.assertEqual(first["cohorts"]["val200"]["cases"], 2)
        self.assertEqual(first["baseline"]["predictions"]["file_count"], 2)
        self.assertEqual(first["baseline"]["evaluation_summary"]["metrics"]["hits"], 2)
        self.assertEqual(first["baseline"]["metric_recomputation"]["hits"], 2)
        self.assertEqual(first["baseline"]["metric_recomputation"]["findings"], 3)
        self.assertTrue(
            first["checks"]["baseline_metrics_recomputed_from_predictions_and_gt"]
        )
        self.assertEqual(first["embedding_bank"]["shape"], [2, 3])
        self.assertEqual(
            first["source_report"]["evidence_availability"],
            "historical_only",
        )
        json.dumps(first, allow_nan=False, sort_keys=True)

    def test_build_lineage_evidence_rejects_declared_hash_drift(self) -> None:
        protocol, source_report = self._write_synthetic_lineage()
        protocol["baseline"]["checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(EvidenceLockError, "SHA-256 mismatch"):
            build_lineage_evidence(
                protocol,
                repo_root=self.root,
                source_report=source_report,
                mask_array_loader=lambda path: np.load(path, allow_pickle=False),
                mask_geometry_verifier=lambda prediction, ground_truth: {
                    "verified": True
                },
            )

    def test_build_lineage_rejects_stale_summary_when_masks_disagree(self) -> None:
        protocol, source_report = self._write_synthetic_lineage()
        np.save(
            self.root / "predictions" / "case_b.npy",
            np.zeros((1, 2, 1, 1), dtype=np.uint8),
            allow_pickle=False,
        )
        with self.assertRaisesRegex(EvidenceLockError, "recomputed Dice mismatch"):
            build_lineage_evidence(
                protocol,
                repo_root=self.root,
                source_report=source_report,
                mask_array_loader=lambda path: np.load(path, allow_pickle=False),
                mask_geometry_verifier=lambda prediction, ground_truth: {
                    "verified": True
                },
            )

    def test_mask_metric_recomputation_rejects_nonbinary_predictions(self) -> None:
        protocol, _ = self._write_synthetic_lineage()
        np.save(
            self.root / "predictions" / "case_b.npy",
            np.asarray([[[[2]], [[0]]]], dtype=np.uint8),
            allow_pickle=False,
        )
        with self.assertRaisesRegex(EvidenceLockError, "not a binary threshold mask"):
            recompute_stored_mask_metrics(
                self.root / protocol["cohorts"]["val200"]["path"],
                self.root / protocol["baseline"]["predictions"],
                self.root / protocol["data"]["segmentation_root"],
                prediction_threshold=0.5,
                hit_threshold=0.1,
                expected_cases=2,
                expected_findings=3,
                array_loader=lambda path: np.load(path, allow_pickle=False),
                geometry_verifier=lambda prediction, ground_truth: {
                    "verified": True
                },
            )


if __name__ == "__main__":
    unittest.main()
