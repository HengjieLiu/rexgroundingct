#!/usr/bin/env python3
"""Network-free focused tests for the CT-RATE ``ts_total`` test300 audit."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_ct_rate_ts_total_rex_test300 as audit  # noqa: E402


def _test_rows() -> list[dict[str, str]]:
    """Build a private-free, valid 300-case ReX test population."""
    return [{"name": f"train_{index}_a_1.nii.gz"} for index in range(300)]


class PopulationLockTests(unittest.TestCase):
    def test_config_locks_test300_without_a_val_evaluation_manifest(self) -> None:
        config = audit.load_config(audit.DEFAULT_CONFIG)
        population = config["population"]
        self.assertEqual(config["experiment"], audit.EXPERIMENT_ID)
        self.assertEqual(population["rex_split"], "test")
        self.assertEqual(population["expected_unique_cases"], 300)
        self.assertEqual(population["expected_ct_rate_split_counts"], {"train_fixed": 300, "valid_fixed": 0})
        self.assertNotIn("evaluation_manifest", population)
        self.assertNotIn("evaluation_manifest_sha256", population)

    def test_load_population_uses_only_the_test_split_and_locks_all_300_names(self) -> None:
        metadata = {
            "test": _test_rows(),
            # Deliberately distinct: test300 must never depend on a validation
            # evaluation manifest or accidentally select this split.
            "val": [{"name": "valid_53_a_1.nii.gz"}],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            metadata_path = Path(temporary_dir) / "MICCAI_challenge_dataset.json"
            metadata_path.write_text(json.dumps(metadata))
            config = {
                "population": {
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    "rex_split": "test",
                    "expected_unique_cases": 300,
                    "expected_ct_rate_split_counts": {"train_fixed": 300, "valid_fixed": 0},
                }
            }
            population = audit.load_population(config)

        self.assertEqual(len(population), 300)
        self.assertEqual([row["metadata_index"] for row in population], list(range(300)))
        self.assertEqual({row["ct_rate_fixed_split"] for row in population}, {"train_fixed"})
        self.assertEqual(population[0]["rex_split"], "test")

    def test_duplicate_test_name_is_rejected_before_any_remote_work(self) -> None:
        metadata = {"test": _test_rows()}
        metadata["test"][-1] = dict(metadata["test"][0])
        with tempfile.TemporaryDirectory() as temporary_dir:
            metadata_path = Path(temporary_dir) / "MICCAI_challenge_dataset.json"
            metadata_path.write_text(json.dumps(metadata))
            config = {
                "population": {
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    "rex_split": "test",
                    "expected_unique_cases": 300,
                    "expected_ct_rate_split_counts": {"train_fixed": 300, "valid_fixed": 0},
                }
            }
            with self.assertRaises(ValueError):
                audit.load_population(config)


class MappingTests(unittest.TestCase):
    def test_train_and_valid_filename_prefixes_keep_their_original_fixed_split(self) -> None:
        self.assertEqual(
            audit.ct_rate_ct_remote_path("train_1741_b_2.nii.gz"),
            "dataset/train_fixed/train_1741/train_1741_b/train_1741_b_2.nii.gz",
        )
        self.assertEqual(
            audit.ct_rate_ts_total_remote_path("valid_53_a_1.nii.gz"),
            "dataset/ts_seg/ts_total/valid_fixed/valid_53/valid_53_a/valid_53_a_1.nii.gz",
        )


class PreflightContractTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "population": {"expected_ct_rate_split_counts": {"train_fixed": 300, "valid_fixed": 0}},
            "preflight": {
                "requested_masks": 300,
                "expected_remote_bytes": 626_543_520,
                "expected_known_upstream_missing_in_population": 0,
            },
        }

    @staticmethod
    def _manifest() -> dict[str, object]:
        return {"cases": [{"rex_split": "test"} for _ in range(300)]}

    @staticmethod
    def _source_lock() -> dict[str, object]:
        return {
            "summary": {
                "requested_masks": 300,
                "remote_ts_total_bytes": 626_543_520,
                # The Hub-derived Counter omits a zero-valued fixed split.
                "split_counts": {"train_fixed": 300},
                "known_upstream_missing_in_population": 0,
                "local_ct_present": 300,
                "local_ct_sidecar_matches": 300,
            }
        }

    def test_preflight_contract_requires_all_locked_aggregate_facts(self) -> None:
        audit._assert_preflight_summary(self._config(), self._manifest(), self._source_lock())

        wrong_bytes = self._source_lock()
        wrong_bytes["summary"]["remote_ts_total_bytes"] = 1  # type: ignore[index]
        with self.assertRaises(RuntimeError):
            audit._assert_preflight_summary(self._config(), self._manifest(), wrong_bytes)

        unexpected_split = self._source_lock()
        unexpected_split["summary"]["split_counts"] = {"train_fixed": 300, "other_fixed": 1}  # type: ignore[index]
        with self.assertRaises(RuntimeError):
            audit._assert_preflight_summary(self._config(), self._manifest(), unexpected_split)


class SourceLockTests(unittest.TestCase):
    def _sealed_lock(self, config: dict[str, object]) -> dict[str, object]:
        source_lock: dict[str, object] = {
            "schema_version": 1,
            "experiment": audit.EXPERIMENT_ID,
            "config_canonical_sha256": audit.base.canonical_json_sha256(config),
            "created_at": "excluded-from-hash",
            "test300_entrypoint": {
                "entrypoint_sha256": audit.base.sha256_file(Path(audit.__file__)),
                "parent_audit_script_sha256": audit.base.sha256_file(Path(audit.base.__file__)),
            },
        }
        source_lock["source_lock_sha256"] = audit.source_lock_sha256(source_lock)
        return source_lock

    def test_source_lock_hash_seals_both_wrapper_and_parent_dependency(self) -> None:
        source_lock = self._sealed_lock({"population": {"rex_split": "test"}})
        self.assertEqual(audit.source_lock_sha256(source_lock), source_lock["source_lock_sha256"])

        wrapper_changed = copy.deepcopy(source_lock)
        wrapper_changed["test300_entrypoint"]["entrypoint_sha256"] = "0" * 64  # type: ignore[index]
        self.assertNotEqual(audit.source_lock_sha256(wrapper_changed), source_lock["source_lock_sha256"])

        parent_changed = copy.deepcopy(source_lock)
        parent_changed["test300_entrypoint"]["parent_audit_script_sha256"] = "1" * 64  # type: ignore[index]
        self.assertNotEqual(audit.source_lock_sha256(parent_changed), source_lock["source_lock_sha256"])

    def test_verify_rejects_foreign_experiment_or_parent_dependency(self) -> None:
        config = {"population": {"rex_split": "test"}}
        foreign_lock = self._sealed_lock(config)
        foreign_lock["experiment"] = audit.base.EXPERIMENT_ID
        foreign_lock["source_lock_sha256"] = audit.source_lock_sha256(foreign_lock)

        parent_changed = self._sealed_lock(config)
        parent_changed["test300_entrypoint"]["parent_audit_script_sha256"] = "1" * 64  # type: ignore[index]
        parent_changed["source_lock_sha256"] = audit.source_lock_sha256(parent_changed)

        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime_root = Path(temporary_dir)
            with mock.patch.object(audit.base, "_require_preflight", return_value=(foreign_lock, {}, {})):
                with self.assertRaises(RuntimeError):
                    audit.verify_test_source_lock(runtime_root, config)
            with mock.patch.object(audit.base, "_require_preflight", return_value=(parent_changed, {}, {})):
                with self.assertRaises(RuntimeError):
                    audit.verify_test_source_lock(runtime_root, config)


class ReportTests(unittest.TestCase):
    def test_aggregate_report_is_test_named_and_never_emits_case_identifiers(self) -> None:
        case_name = "train_123_a_1.nii.gz"
        result = {
            "volume_name": case_name,
            "download_status": "downloaded",
            "source_provenance": {"status": "FULL_HASH_MATCH"},
            "integrity": {"status": "VALID"},
            "geometry": {
                "index_grid_status": "PASS",
                "world_header_status": "PASS",
                "header_comparison": {
                    "spacing_status": "PASS",
                    "header_form_status": "PASS",
                    "header_metadata_status": "PASS",
                },
            },
            "labels": {"status": "PASS"},
            "lungs": {"left": {"flags": []}, "right": {"flags": []}},
            "visual_qc_status": "GENERATED_PENDING_MANUAL_REVIEW",
            "error": None,
        }
        report, conclusion = audit._test_aggregate_report(
            {"lut_provenance": {"status": "PASS"}},
            {
                "cases": [{}],
                "rex_revision": "rex-pin",
                "ct_rate_revision": "ct-pin",
                "manifest_sha256": "manifest-pin",
            },
            [result],
            None,
        )
        self.assertEqual(conclusion, "PASS_PENDING_MANUAL_VISUAL_REVIEW")
        self.assertIn("ReX test300 audit", report)
        self.assertNotIn("ReX validation audit", report)
        self.assertNotIn(case_name, report)


class InheritedSafetyTests(unittest.TestCase):
    def test_auth_and_missing_source_statuses_remain_distinct(self) -> None:
        class Response:
            status_code = 401

        class Error(Exception):
            response = Response()

        self.assertEqual(audit.base.classify_hub_exception(Error()), "auth_error")
        self.assertEqual(audit.base.classify_hub_exception(OSError("network unavailable")), "transport_error")

    def test_inherited_geometry_and_label_gates_remain_fail_closed(self) -> None:
        ct = np.eye(4)
        segmentation = np.eye(4)
        segmentation[0, 3] = 2.0
        geometry = audit.base.affine_geometry(ct, segmentation, (6, 7, 8), (6, 7, 8), 1e-5, 0.01)
        self.assertEqual(geometry["index_grid_status"], "INDEX_TRANSFORM_MISMATCH")
        self.assertEqual(geometry["world_header_status"], "WORLD_CORNER_MISMATCH")

        labels = audit.base.label_qc(np.array([[[0, 118]]], dtype=np.int16), "PASS")
        self.assertEqual(labels["status"], "UNEXPECTED_LABEL_IDS")


if __name__ == "__main__":
    unittest.main()
