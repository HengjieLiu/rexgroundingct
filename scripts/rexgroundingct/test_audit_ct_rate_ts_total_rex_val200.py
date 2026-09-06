#!/usr/bin/env python3
"""Network-free focused tests for the CT-RATE ``ts_total`` audit helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_ct_rate_ts_total_rex_val200 as audit  # noqa: E402


class MappingTests(unittest.TestCase):
    def test_filename_maps_to_the_correct_fixed_split_and_hierarchy(self) -> None:
        name = "train_1741_b_2.nii.gz"
        self.assertEqual(
            audit.ct_rate_ct_remote_path(name),
            "dataset/train_fixed/train_1741/train_1741_b/train_1741_b_2.nii.gz",
        )
        self.assertEqual(
            audit.ct_rate_ts_total_remote_path(name),
            "dataset/ts_seg/ts_total/train_fixed/train_1741/train_1741_b/"
            "train_1741_b_2.nii.gz",
        )

    def test_validation_prefix_is_not_inferred_from_rex_split(self) -> None:
        self.assertIn("valid_fixed", audit.ct_rate_ct_remote_path("valid_53_a_1.nii.gz"))

    def test_malformed_filename_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit.parse_volume_name("not_a_ct_rate_volume.nii.gz")

    def test_canonical_json_hash_is_order_stable(self) -> None:
        self.assertEqual(
            audit.canonical_json_sha256({"b": 2, "a": [1, 3]}),
            audit.canonical_json_sha256({"a": [1, 3], "b": 2}),
        )


class HubStatusTests(unittest.TestCase):
    def test_auth_error_is_not_classified_as_absent(self) -> None:
        class Response:
            status_code = 401

        class Error(Exception):
            response = Response()

        self.assertEqual(audit.classify_hub_exception(Error()), "auth_error")

    def test_not_found_is_classified_as_upstream_absent(self) -> None:
        class Response:
            status_code = 404

        class Error(Exception):
            response = Response()

        self.assertEqual(audit.classify_hub_exception(Error()), "upstream_absent")

    def test_transport_error_stays_distinct(self) -> None:
        self.assertEqual(audit.classify_hub_exception(OSError("network unavailable")), "transport_error")


class GeometryTests(unittest.TestCase):
    def test_identical_grid_passes_index_and_world_checks(self) -> None:
        affine = np.array(
            [[0.7, 0.0, 0.0, 10.0], [0.0, 0.7, 0.0, -5.0], [0.0, 0.0, 1.2, 2.0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        result = audit.affine_geometry(affine, affine, (8, 9, 10), (8, 9, 10), 1e-5, 0.01)
        self.assertEqual(result["index_grid_status"], "PASS")
        self.assertEqual(result["world_header_status"], "PASS")
        self.assertEqual(len(result["corner_residuals_mm"]), 8)
        self.assertEqual(result["max_corner_residual_mm"], 0.0)

    def test_same_shape_translation_fails_explicit_world_and_index_checks(self) -> None:
        ct = np.eye(4)
        seg = np.eye(4)
        seg[0, 3] = 2.0
        result = audit.affine_geometry(ct, seg, (6, 7, 8), (6, 7, 8), 1e-5, 0.01)
        self.assertTrue(result["shape_identical"])
        self.assertEqual(result["index_grid_status"], "INDEX_TRANSFORM_MISMATCH")
        self.assertEqual(result["world_header_status"], "WORLD_CORNER_MISMATCH")
        self.assertAlmostEqual(result["max_corner_residual_mm"], 2.0)

    def test_same_affine_but_different_shape_is_not_accepted(self) -> None:
        result = audit.affine_geometry(np.eye(4), np.eye(4), (5, 6, 7), (5, 6, 8), 1e-5, 0.01)
        self.assertEqual(result["index_grid_status"], "SHAPE_MISMATCH")
        self.assertEqual(result["world_header_status"], "SHAPE_MISMATCH")
        self.assertEqual(result["corner_residuals_mm"], [])

    def test_scale_change_is_detected(self) -> None:
        ct = np.eye(4)
        seg = np.eye(4)
        seg[2, 2] = 1.1
        result = audit.affine_geometry(ct, seg, (10, 10, 10), (10, 10, 10), 1e-5, 0.01)
        self.assertEqual(result["index_grid_status"], "INDEX_TRANSFORM_MISMATCH")
        self.assertGreater(result["max_corner_residual_mm"], 0.01)

    def test_coded_forms_and_millimetre_units_have_an_explicit_gate(self) -> None:
        matrix = np.eye(4).tolist()
        header = {
            "best_affine": matrix,
            "orientation": ["R", "A", "S"],
            "units": ["mm", "sec"],
            "zooms_mm": [1.0, 1.0, 1.0],
            "qform": {"code": 1, "matrix": matrix},
            "sform": {"code": 1, "matrix": matrix},
        }
        passing = audit._header_comparison(header, header, 0.01)
        self.assertEqual(passing["header_form_status"], "PASS")
        mismatched = dict(header)
        mismatched["qform"] = {"code": 2, "matrix": matrix}
        self.assertEqual(
            audit._header_comparison(header, mismatched, 0.01)["header_form_status"],
            "QFORM_SFORM_MISMATCH",
        )
        internally_mismatched = dict(header)
        internally_mismatched["qform"] = {"code": 1, "matrix": np.diag([2, 1, 1, 1]).tolist()}
        self.assertEqual(
            audit._header_comparison(internally_mismatched, internally_mismatched, 0.01)[
                "header_form_status"
            ],
            "INPUT_FORM_BEST_AFFINE_MISMATCH",
        )

    def test_unset_segmentation_sform_is_qc_compatible_only_with_aligned_qforms(self) -> None:
        matrix = np.eye(4).tolist()
        ct_header = {
            "best_affine": matrix,
            "orientation": ["R", "A", "S"],
            "units": ["mm", "sec"],
            "zooms_mm": [1.0, 1.0, 1.0],
            "qform": {"code": 1, "matrix": matrix},
            "sform": {"code": 1, "matrix": matrix},
        }
        seg_header = dict(ct_header)
        seg_header["sform"] = {"code": 0, "matrix": None}
        comparison = audit._header_comparison(ct_header, seg_header, 0.01)
        self.assertEqual(comparison["sform_status"], "FORM_PRESENCE_MISMATCH")
        self.assertEqual(comparison["header_form_status"], "SFORM_UNSET_QFORM_ALIGNED")
        self.assertEqual(comparison["header_metadata_status"], "SEGMENTATION_SFORM_UNSET")
        self.assertTrue(audit.header_form_status_allows_native_qc(comparison["header_form_status"]))

        missing_seg_qform = dict(seg_header)
        missing_seg_qform["qform"] = {"code": 0, "matrix": None}
        blocked = audit._header_comparison(ct_header, missing_seg_qform, 0.01)
        self.assertEqual(blocked["header_form_status"], "QFORM_SFORM_MISMATCH")
        self.assertFalse(audit.header_form_status_allows_native_qc(blocked["header_form_status"]))

        missing_ct_sform = dict(ct_header)
        missing_ct_sform["sform"] = {"code": 0, "matrix": None}
        inverse_pattern = audit._header_comparison(missing_ct_sform, ct_header, 0.01)
        self.assertEqual(inverse_pattern["header_form_status"], "QFORM_SFORM_MISMATCH")
        self.assertFalse(audit.header_form_status_allows_native_qc(inverse_pattern["header_form_status"]))

        contradictory_ct_sform = dict(ct_header)
        contradictory_ct_sform["sform"] = {"code": 1, "matrix": np.diag([2, 1, 1, 1]).tolist()}
        contradictory = audit._header_comparison(contradictory_ct_sform, seg_header, 0.01)
        self.assertEqual(contradictory["header_form_status"], "QFORM_SFORM_MISMATCH")
        self.assertFalse(audit.header_form_status_allows_native_qc(contradictory["header_form_status"]))


class LabelTests(unittest.TestCase):
    def test_valid_v2_lung_labels_pass_with_a_verified_lut(self) -> None:
        data = np.array([[[0, 10], [11, 12]], [[13, 14], [51, 117]]], dtype=np.int16)
        result = audit.label_qc(data, "PASS")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["unexpected_label_ids"], [])

    def test_lut_block_is_preserved_without_relabeling(self) -> None:
        result = audit.label_qc(np.array([[[10]]], dtype=np.int16), "LUT_PROVENANCE_BLOCKED")
        self.assertEqual(result["status"], "LUT_PROVENANCE_BLOCKED")

    def test_unexpected_label_is_not_silently_accepted(self) -> None:
        result = audit.label_qc(np.array([[[0, 118]]], dtype=np.int16), "PASS")
        self.assertEqual(result["status"], "UNEXPECTED_LABEL_IDS")
        self.assertEqual(result["unexpected_label_ids"], [118])

    def test_empty_and_noninteger_masks_fail(self) -> None:
        self.assertEqual(audit.label_qc(np.zeros((2, 2, 2), dtype=np.int16), "PASS")["status"], "EMPTY_SEGMENTATION")
        self.assertEqual(audit.label_qc(np.array([[[1.5]]]), "PASS")["status"], "NONINTEGER")


class ProvenanceTests(unittest.TestCase):
    def test_git_blob_sha1_uses_a_nul_header_separator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "payload"
            path.write_bytes(b"abc")
            self.assertEqual(audit.git_blob_sha1(path), "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f")

    def test_sidecar_parser_requires_revision_and_etag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "file.metadata"
            self.assertEqual(audit.parse_local_ct_sidecar(path)["status"], "SIDECAR_MISSING")
            path.write_text("only-one-line\n")
            self.assertEqual(audit.parse_local_ct_sidecar(path)["status"], "SIDECAR_INVALID")
            path.write_text("revision\netag\n123.0\n")
            parsed = audit.parse_local_ct_sidecar(path)
            self.assertEqual(parsed["revision"], "revision")
            self.assertEqual(parsed["etag"], "etag")

    def test_lut_requires_both_producer_documentation_and_exact_map_hash(self) -> None:
        result = audit.validate_lut_provenance(ct_rate_readme_bytes=None, map_source_bytes=None)
        self.assertEqual(result["status"], "LUT_PROVENANCE_BLOCKED")
        result = audit.validate_lut_provenance(
            ct_rate_readme_bytes=b"TotalSegmentator version: 2.7.0",
            map_source_bytes=b'10: "lung_upper_lobe_left"',
        )
        self.assertEqual(result["status"], "LUT_PROVENANCE_BLOCKED")


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _result(
        name: str,
        index: int,
        world_status: str = "PASS",
        header_form_status: str = "PASS",
    ) -> dict[str, object]:
        return {
            "volume_name": name,
            "metadata_index": index,
            "source_provenance": {"status": "FULL_HASH_MATCH"},
            "integrity": {"status": "VALID"},
            "geometry": {
                "index_grid_status": "PASS",
                "world_header_status": world_status,
                "header_comparison": {
                    "spacing_status": "PASS",
                    "header_form_status": header_form_status,
                },
            },
            "labels": {"status": "PASS"},
            "lungs": {"left": {"flags": []}, "right": {"flags": []}},
            "error": None,
        }

    def test_selection_is_deterministic_and_includes_every_exception(self) -> None:
        results = [
            self._result("train_1_a_1.nii.gz", 0),
            self._result("train_2_a_1.nii.gz", 1, "WORLD_CORNER_MISMATCH"),
            self._result("train_3_a_1.nii.gz", 2),
        ]
        first = audit._deterministic_visual_selection(results, 1)
        second = audit._deterministic_visual_selection(results, 1)
        self.assertEqual(
            [(item[0]["volume_name"], item[1]) for item in first],
            [(item[0]["volume_name"], item[1]) for item in second],
        )
        self.assertIn("train_2_a_1.nii.gz", [item[0]["volume_name"] for item in first])

    def test_safe_sform_metadata_difference_is_not_an_exception(self) -> None:
        result = self._result(
            "train_1_a_1.nii.gz",
            0,
            header_form_status="SFORM_UNSET_QFORM_ALIGNED",
        )
        self.assertEqual(audit._audit_exception_reasons(result), [])


class ConclusionGateTests(unittest.TestCase):
    def test_full_hash_mismatch_cannot_receive_a_pass(self) -> None:
        result = {
            "volume_name": "train_1_a_1.nii.gz",
            "download_status": "downloaded",
            "source_provenance": {"status": "FULL_HASH_MISMATCH"},
            "integrity": {"status": "VALID"},
            "geometry": {
                "index_grid_status": "PASS",
                "world_header_status": "PASS",
                "header_comparison": {"spacing_status": "PASS", "header_form_status": "PASS"},
            },
            "labels": {"status": "PASS"},
            "lungs": {"left": {"flags": []}, "right": {"flags": []}},
            "visual_qc_status": "GENERATED_PENDING_MANUAL_REVIEW",
            "error": None,
        }
        _, conclusion = audit._aggregate_report(
            {"lut_provenance": {"status": "PASS"}},
            {
                "cases": [{}],
                "rex_revision": "r",
                "ct_rate_revision": "c",
                "manifest_sha256": "m",
            },
            [result],
            {"passed": True, "reviewer": "reviewer"},
        )
        self.assertEqual(conclusion, "PARTIAL_SOURCE_PROVENANCE")

    def test_safe_sform_metadata_difference_does_not_block_the_review_gate(self) -> None:
        result = {
            "volume_name": "train_1_a_1.nii.gz",
            "download_status": "downloaded",
            "source_provenance": {"status": "FULL_HASH_MATCH"},
            "integrity": {"status": "VALID"},
            "geometry": {
                "index_grid_status": "PASS",
                "world_header_status": "PASS",
                "header_comparison": {
                    "spacing_status": "PASS",
                    "header_form_status": "SFORM_UNSET_QFORM_ALIGNED",
                    "header_metadata_status": "SEGMENTATION_SFORM_UNSET",
                },
            },
            "labels": {"status": "PASS"},
            "lungs": {"left": {"flags": []}, "right": {"flags": []}},
            "visual_qc_status": "GENERATED_PENDING_MANUAL_REVIEW",
            "error": None,
        }
        _, conclusion = audit._aggregate_report(
            {"lut_provenance": {"status": "PASS"}},
            {
                "cases": [{}],
                "rex_revision": "r",
                "ct_rate_revision": "c",
                "manifest_sha256": "m",
            },
            [result],
            None,
        )
        self.assertEqual(conclusion, "PASS_PENDING_MANUAL_VISUAL_REVIEW")


class SealedArtifactTests(unittest.TestCase):
    def test_source_lock_hash_detects_mutation(self) -> None:
        source_lock = {
            "schema_version": 1,
            "experiment": audit.EXPERIMENT_ID,
            "created_at": "time-is-excluded",
            "lut_provenance": {"status": "PASS"},
        }
        source_lock["source_lock_sha256"] = audit.source_lock_sha256(source_lock)
        self.assertEqual(audit.source_lock_sha256(source_lock), source_lock["source_lock_sha256"])
        source_lock["lut_provenance"] = {"status": "LUT_PROVENANCE_BLOCKED"}
        self.assertNotEqual(audit.source_lock_sha256(source_lock), source_lock["source_lock_sha256"])

    def test_audit_results_must_have_exact_unique_locked_population(self) -> None:
        manifest = {
            "manifest_sha256": "manifest",
            "cases": [
                {"volume_name": "train_1_a_1.nii.gz", "metadata_index": 0},
                {"volume_name": "train_2_a_1.nii.gz", "metadata_index": 1},
            ],
        }
        source_lock = {
            "source_lock_sha256": "lock",
            "config_canonical_sha256": "config",
            "script_sha256": "script",
        }
        payload = {
            "manifest_sha256": "manifest",
            "source_lock_sha256": "lock",
            "config_canonical_sha256": "config",
            "script_sha256": "script",
            "cases": [
                {"volume_name": "train_1_a_1.nii.gz", "metadata_index": 0},
                {"volume_name": "train_1_a_1.nii.gz", "metadata_index": 0},
            ],
        }
        with self.assertRaises(RuntimeError):
            audit.verified_audit_results(payload, manifest, source_lock)


if __name__ == "__main__":
    unittest.main()
