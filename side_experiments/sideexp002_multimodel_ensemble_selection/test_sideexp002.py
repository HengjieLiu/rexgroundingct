from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_candidates
import analyze_all20_thresholds
import audit_logit_exports
import finalize_when_complete
import launch_logit_exports
import make_folds
import revalidate_completed_exports
import search_ensembles
import storage_validation
import summarize_baseline_categories
import wait_then_launch


class CandidateAnalysisTests(unittest.TestCase):
    def test_manifest_has_exact_selected_pool(self) -> None:
        manifest = analyze_candidates.load_manifest(HERE / "candidate_manifest.json")
        self.assertEqual(len(manifest["candidates"]), 20)
        self.assertEqual(
            {candidate["experiment_number"] for candidate in manifest["candidates"]},
            {6, 7, 8, 9, 11},
        )
        self.assertTrue(
            all(
                candidate["performance"]["dice"] >= 0.32
                for candidate in manifest["candidates"]
            )
        )

    def test_exp007_frozen_config_preserves_original_hash(self) -> None:
        manifest = analyze_candidates.load_manifest(HERE / "candidate_manifest.json")
        candidate = next(
            value
            for value in manifest["candidates"]
            if value["id"] == "exp007_ddp_e050"
        )
        resolved = analyze_candidates.require_candidate_config(candidate)
        self.assertEqual(
            analyze_candidates.sha256_file(resolved),
            candidate["config"]["sha256"],
        )


class BaselineCategoryTests(unittest.TestCase):
    def test_rank_tie_breaking_and_observed_range(self) -> None:
        records = [
            {
                "candidate_id": "z",
                "dice": 0.4,
                "hits": 3,
                "findings": 5,
                "hit_rate": 0.6,
            },
            {
                "candidate_id": "a",
                "dice": 0.4,
                "hits": 3,
                "findings": 5,
                "hit_rate": 0.6,
            },
            {
                "candidate_id": "b",
                "dice": 0.4,
                "hits": 4,
                "findings": 5,
                "hit_rate": 0.8,
            },
            {
                "candidate_id": "c",
                "dice": 0.2,
                "hits": 2,
                "findings": 5,
                "hit_rate": 0.4,
            },
        ]
        ranked = summarize_baseline_categories.rank_records(records)
        self.assertEqual(
            [record["candidate_id"] for record in ranked],
            ["b", "a", "z", "c"],
        )
        value_range = summarize_baseline_categories.observed_range(ranked)
        self.assertAlmostEqual(value_range["dice"]["spread"], 0.2)
        self.assertEqual(value_range["hits"]["minimum"], 2)
        self.assertEqual(value_range["hits"]["maximum"], 4)

    def test_real_report_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_json = root / "first.json"
            first_md = root / "first.md"
            second_json = root / "second.json"
            second_md = root / "second.md"
            first = summarize_baseline_categories.run(
                HERE / "candidate_manifest.json",
                summarize_baseline_categories.DEFAULT_DATASET,
                first_json,
                first_md,
            )
            second = summarize_baseline_categories.run(
                HERE / "candidate_manifest.json",
                summarize_baseline_categories.DEFAULT_DATASET,
                second_json,
                second_md,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            self.assertEqual(first["counts"]["models"], 20)
            self.assertEqual(first["counts"]["represented_categories"], 13)
            self.assertEqual(sum(first["category_counts"].values()), 381)
            self.assertFalse(first["categories"]["2f"]["available"])
            self.assertEqual(first["categories"]["2f"]["support"], 0)
            self.assertEqual(first["categories"]["2g"]["support"], 1)
            self.assertEqual(
                len(first["categories"]["2g"]["ranked_models"]),
                20,
            )
            self.assertTrue(
                first["validation"][
                    "all_model_category_recompositions_passed"
                ]
            )
            for category in first["categories"].values():
                if category["available"]:
                    self.assertEqual(
                        [row["rank"] for row in category["ranked_models"]],
                        list(range(1, 21)),
                    )
            report = first_md.read_text()
            self.assertIn("Rare-category warning", report)
            self.assertIn("## Diffuse Categories", report)
            self.assertIn("## Focal Categories", report)
            self.assertIn("**#1 ", report)

    def test_dataset_source_hash_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            value = json.loads(
                summarize_baseline_categories.DEFAULT_DATASET.read_text()
            )
            value["test"][0]["categories"]["0"] = "2h"
            changed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "SHA256 drifted"):
                summarize_baseline_categories.build_summary(
                    HERE / "candidate_manifest.json",
                    changed,
                )


class CandidateAnalysisAdditionalTests(unittest.TestCase):
    def test_historical_mismatch_is_warning_only(self) -> None:
        decision = analyze_candidates.reproduction_decision(
            storage_passed=True,
            historical_passed=False,
        )
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["historical_reproduction_is_gate"])
        self.assertEqual(
            decision["warnings"],
            ["historical_reproduction_mismatch"],
        )

    def test_pair_comparison_uses_keys_not_insertion_order(self) -> None:
        first = {
            ("case_a", 0): {"dice": 0.2, "hit": True},
            ("case_b", 0): {"dice": 0.0, "hit": False},
            ("case_c", 0): {"dice": 0.6, "hit": True},
        }
        second = {
            ("case_c", 0): {"dice": 0.5, "hit": True},
            ("case_a", 0): {"dice": 0.0, "hit": False},
            ("case_b", 0): {"dice": 0.3, "hit": True},
        }
        summary = {
            "first": {"dice": 0.8 / 3},
            "second": {"dice": 0.8 / 3},
        }
        result = analyze_candidates.compare_pair(
            "first",
            "second",
            {"first": first, "second": second},
            summary,
        )
        self.assertEqual(result["hit_disagreements"], 2)
        self.assertEqual(result["first_only_hits"], 1)
        self.assertEqual(result["second_only_hits"], 1)
        self.assertEqual(result["union_hits"], 3)
        self.assertAlmostEqual(result["oracle_dice"], (0.2 + 0.3 + 0.6) / 3)

    def test_generated_analysis_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = analyze_candidates.run(
                HERE / "candidate_manifest.json",
                root / "models_1.md",
                root / "pairs_1.md",
                root / "pairs_1.json",
                verify_large_hashes=False,
            )
            second = analyze_candidates.run(
                HERE / "candidate_manifest.json",
                root / "models_2.md",
                root / "pairs_2.md",
                root / "pairs_2.json",
                verify_large_hashes=False,
            )
            self.assertEqual(
                (root / "models_1.md").read_bytes(),
                (root / "models_2.md").read_bytes(),
            )
            self.assertEqual(
                (root / "pairs_1.md").read_bytes(),
                (root / "pairs_2.md").read_bytes(),
            )
            first.pop("source_manifest_sha256")
            second.pop("source_manifest_sha256")
            self.assertEqual(first, second)


class FoldTests(unittest.TestCase):
    def test_real_fold_assignment_is_deterministic_and_balanced(self) -> None:
        cases = make_folds.load_cases(make_folds.DEFAULT_DATASET)
        first = make_folds.make_folds(cases)
        second = make_folds.make_folds(cases)
        self.assertEqual(first, second)
        make_folds.validate_folds(first, cases)
        self.assertLessEqual(
            max(len(fold["cases"]) for fold in first)
            - min(len(fold["cases"]) for fold in first),
            2,
        )
        self.assertLessEqual(
            max(fold["findings"] for fold in first)
            - min(fold["findings"] for fold in first),
            3,
        )


class EnsembleCoreTests(unittest.TestCase):
    def test_probability_and_logit_averaging_are_distinct(self) -> None:
        arrays = {
            "a": np.asarray([[[-4.0, 4.0]]], dtype=np.float32),
            "b": np.asarray([[[0.0, 0.0]]], dtype=np.float32),
        }
        weights = {"a": 0.5, "b": 0.5}
        probability = search_ensembles.accumulate_ensemble_chunk(
            arrays,
            weights,
            finding_index=0,
            start=0,
            stop=2,
            domain="probability",
        )
        logits = search_ensembles.accumulate_ensemble_chunk(
            arrays,
            weights,
            finding_index=0,
            start=0,
            stop=2,
            domain="logit",
        )
        np.testing.assert_allclose(probability, [0.25899312, 0.7410069], rtol=1e-6)
        np.testing.assert_allclose(logits, [0.11920292, 0.880797], rtol=1e-6)

    def test_weight_normalization_and_recipe_hash_are_stable(self) -> None:
        recipe = search_ensembles.make_recipe(
            "test",
            "probability",
            {"b": 2.0, "a": 1.0},
            "unit",
        )
        self.assertEqual(list(recipe["weights"]), ["a", "b"])
        self.assertAlmostEqual(sum(recipe["weights"].values()), 1.0)
        first = search_ensembles.recipe_hash(
            recipe,
            "dataset",
            {"a": "hash-a", "b": "hash-b"},
        )
        second = search_ensembles.recipe_hash(
            recipe,
            "dataset",
            {"b": "hash-b", "a": "hash-a"},
        )
        self.assertEqual(first, second)

    def test_result_hash_ignores_runtime_but_not_selection(self) -> None:
        first = {
            "primary_strategy": "greedy_probability",
            "dice": 0.34,
            "mean_recipe_runtime_seconds": 12.0,
            "fold_records": [{"recipe_runtime_seconds": 3.0}],
        }
        second = {
            "primary_strategy": "greedy_probability",
            "dice": 0.34,
            "mean_recipe_runtime_seconds": 99.0,
            "fold_records": [{"recipe_runtime_seconds": 17.0}],
        }
        self.assertEqual(
            search_ensembles.deterministic_result_hash(first),
            search_ensembles.deterministic_result_hash(second),
        )
        second["dice"] = 0.35
        self.assertNotEqual(
            search_ensembles.deterministic_result_hash(first),
            search_ensembles.deterministic_result_hash(second),
        )

    def test_two_stage_threshold_tie_prefers_lower_threshold(self) -> None:
        rows = []
        thresholds = {
            search_ensembles.threshold_key(value): {
                "dice": 0.2,
                "hit": True,
                "pred_voxels": 1,
                "intersection_voxels": 1,
            }
            for value in search_ensembles.ALL_THRESHOLDS
        }
        rows.append(
            {
                "case": "case",
                "finding_index": 0,
                "category": "2d",
                "gt_voxels": 1,
                "thresholds": thresholds,
            }
        )
        best = search_ensembles.best_threshold({"rows": rows}, {"case"})
        self.assertEqual(best["threshold"], 0.01)


class All20ThresholdAnalysisTests(unittest.TestCase):
    @staticmethod
    def make_rows() -> tuple[list[dict[str, object]], dict[str, int]]:
        supports = {
            code: 0 for code in analyze_all20_thresholds.CATEGORY_LABELS
        }
        supports["2d"] = 380
        supports["2g"] = 1
        rows = []
        for finding_number in range(381):
            category = "2g" if finding_number == 380 else "2d"
            case_number = min(finding_number, 199)
            finding_index = finding_number if case_number == 199 else 0
            metrics = {}
            for threshold in analyze_all20_thresholds.THRESHOLDS:
                if category == "2d":
                    dice = 0.4 if threshold == 0.35 else 0.3
                else:
                    dice = 0.9 if threshold == 0.70 else 0.1
                metrics[analyze_all20_thresholds.threshold_key(threshold)] = {
                    "dice": dice,
                    "hit": dice >= 0.1,
                }
            rows.append(
                {
                    "case": f"case_{case_number:03d}.nii.gz",
                    "finding_index": finding_index,
                    "category": category,
                    "gt_voxels": 1,
                    "thresholds": metrics,
                }
            )
        return rows, supports

    def test_requested_threshold_grid_is_exact(self) -> None:
        self.assertEqual(
            analyze_all20_thresholds.THRESHOLDS,
            (
                0.10,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.90,
            ),
        )

    def test_uniform_probability_averaging(self) -> None:
        arrays = {
            "a": np.asarray([[[-4.0, 4.0]]], dtype=np.float32),
            "b": np.asarray([[[0.0, 0.0]]], dtype=np.float32),
        }
        probability = analyze_all20_thresholds.accumulate_probability_chunk(
            arrays,
            ("a", "b"),
            finding_index=0,
            start=0,
            stop=2,
        )
        np.testing.assert_allclose(
            probability,
            [0.25899312, 0.7410069],
            rtol=1e-6,
        )

    def test_gate_override_preserves_warning(self) -> None:
        gate = {
            "status": "failed",
            "storage_reproduction_status": "failed",
            "same_pass_mask_reproduction_status": "failed",
            "historical_reproduction_status": "warning",
            "historical_reproduction_is_gate": False,
            "array_hashes_verified": True,
            "warnings": ["historical_reproduction_mismatch"],
        }
        self.assertFalse(
            analyze_all20_thresholds.strict_gate_accepted(gate)
        )
        reason = analyze_all20_thresholds.gate_warning_reason(gate)
        self.assertIn("storage_reproduction_status=failed", reason)
        self.assertIn("historical_reproduction_mismatch", reason)

    def test_array_record_rejects_structural_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.npy"
            np.save(path, np.zeros((1, 2, 3, 4), dtype=np.float16))
            record = {
                "status": "complete",
                "shape": [1, 2, 3, 4],
                "dtype": "float16",
                "array_sha256": "recorded",
                "npy_bytes": path.stat().st_size,
            }
            self.assertEqual(
                analyze_all20_thresholds.validate_array_record(
                    path,
                    record,
                    expected_findings=1,
                ),
                (1, 2, 3, 4),
            )
            record["npy_bytes"] += 1
            with self.assertRaisesRegex(ValueError, "file size differs"):
                analyze_all20_thresholds.validate_array_record(
                    path,
                    record,
                    expected_findings=1,
                )

    def test_global_and_category_ties_prefer_lower_threshold(self) -> None:
        values = [
            {
                "threshold": 0.35,
                "dice": 0.4,
                "hits": 3,
            },
            {
                "threshold": 0.30,
                "dice": 0.4,
                "hits": 3,
            },
            {
                "threshold": 0.25,
                "dice": 0.4,
                "hits": 2,
            },
        ]
        best = analyze_all20_thresholds.best_metric(values)
        self.assertEqual(best["threshold"], 0.30)

    def test_zero_support_rare_category_and_oracle_recompose(self) -> None:
        rows, supports = self.make_rows()
        at_035 = analyze_all20_thresholds.summarize_threshold(
            rows,
            0.35,
            supports,
        )
        self.assertFalse(at_035["categories"]["2f"]["available"])
        self.assertEqual(at_035["categories"]["2g"]["findings"], 1)
        category_best = {
            "2d": {
                "threshold": 0.35,
                "findings": 380,
                "dice": 0.4,
                "hits": 380,
                "hit_rate": 1.0,
            },
            "2g": {
                "threshold": 0.70,
                "findings": 1,
                "dice": 0.9,
                "hits": 1,
                "hit_rate": 1.0,
            },
        }
        oracle = analyze_all20_thresholds.category_oracle(
            rows,
            category_best,
            supports,
        )
        self.assertAlmostEqual(oracle["dice"], (380 * 0.4 + 0.9) / 381)
        self.assertEqual(oracle["hits"], 381)
        self.assertEqual(
            oracle["validation"]["category_hits_recomposed"],
            381,
        )

    def test_cached_case_requires_matching_recipe(self) -> None:
        case = {
            "name": "case.nii.gz",
            "findings": {"0": "finding"},
            "categories": {"0": "2d"},
        }
        thresholds = {
            analyze_all20_thresholds.threshold_key(threshold): {
                "dice": 0.2,
                "hit": True,
            }
            for threshold in analyze_all20_thresholds.THRESHOLDS
        }
        cached = {
            "schema_version": 1,
            "analysis": analyze_all20_thresholds.ANALYSIS_ID,
            "recipe_hash": "recipe",
            "case": case["name"],
            "thresholds": list(analyze_all20_thresholds.THRESHOLDS),
            "rows": [
                {
                    "case": case["name"],
                    "finding_index": 0,
                    "category": "2d",
                    "thresholds": thresholds,
                }
            ],
        }
        self.assertEqual(
            len(
                analyze_all20_thresholds.validate_cached_case(
                    cached,
                    case,
                    "recipe",
                )
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            analyze_all20_thresholds.validate_cached_case(
                cached,
                case,
                "different",
            )

    def test_summary_and_markdown_are_deterministic(self) -> None:
        rows, supports = self.make_rows()
        roster = [
            {
                "candidate_id": f"candidate_{index:02d}",
                "experiment_number": 6,
                "experiment_name": "experiment",
                "model_id": f"arm_{index:02d}",
                "epoch": 100,
                "architecture": "standard",
                "dtype": "float16",
                "weight": 0.05,
                "strict_gate_accepted": index >= 7,
                "gate_status": "passed" if index >= 7 else "failed",
                "storage_reproduction_status": (
                    "passed" if index >= 7 else "failed"
                ),
                "same_pass_mask_reproduction_status": (
                    "passed" if index >= 7 else "failed"
                ),
                "historical_reproduction_status": "passed",
                "warning_reason": "-" if index >= 7 else "status=failed",
                "checkpoint_sha256": f"checkpoint-{index}",
                "export_manifest_sha256": f"export-{index}",
                "reproduction_manifest_sha256": f"gate-{index}",
            }
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            dataset = root / "dataset.json"
            manifest.write_text("{}")
            dataset.write_text("{}")
            first = analyze_all20_thresholds.build_summary(
                rows,
                supports,
                roster,
                "recipe",
                manifest,
                dataset,
                root,
                root,
                "inventory",
                "reproduce",
            )
            second = analyze_all20_thresholds.build_summary(
                rows,
                supports,
                roster,
                "recipe",
                manifest,
                dataset,
                root,
                root,
                "inventory",
                "reproduce",
            )
            self.assertEqual(first, second)
            self.assertEqual(
                analyze_all20_thresholds.build_report(first),
                analyze_all20_thresholds.build_report(second),
            )
            self.assertEqual(
                first["counts"]["strict_gate_bypassed_models"],
                7,
            )
            self.assertFalse(first["categories"]["2f"]["available"])
            self.assertEqual(first["categories"]["2g"]["support"], 1)


class LauncherTests(unittest.TestCase):
    def test_no_memory_wait_disables_gate(self) -> None:
        args = SimpleNamespace(no_memory_wait=True)
        self.assertFalse(launch_logit_exports.memory_wait_enabled(args))

    def test_storage_retry_requires_float16_storage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "logits" / "candidate" / "reproduction_validation.json"
            gate.parent.mkdir(parents=True)
            gate.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "dtype": "float16",
                        "storage_reproduction_status": "passed",
                        "historical_reproduction_status": "warning",
                    }
                )
            )
            self.assertFalse(
                launch_logit_exports.storage_retry_required(root, "candidate")
            )
            value = json.loads(gate.read_text())
            value["storage_reproduction_status"] = "failed"
            gate.write_text(json.dumps(value))
            self.assertTrue(
                launch_logit_exports.storage_retry_required(root, "candidate")
            )

    def test_worker_records_failure_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                runtime_root=Path(temporary),
                smoke=False,
            )
            completed = {
                "candidate_id": "second",
                "gpu": 0,
                "status": "complete_float16",
            }
            with mock.patch.object(
                launch_logit_exports,
                "run_candidate",
                side_effect=[RuntimeError("first failed"), completed],
            ):
                results = launch_logit_exports.worker(
                    0,
                    ["first", "second"],
                    args,
                )
            self.assertEqual(results[0]["status"], "failed")
            self.assertEqual(results[1], completed)
            first_record = (
                Path(temporary) / "queue_results" / "first.json"
            )
            second_record = (
                Path(temporary) / "queue_results" / "second.json"
            )
            self.assertTrue(first_record.is_file())
            self.assertTrue(second_record.is_file())

    def test_gpu_lock_rejects_duplicate_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with launch_logit_exports.gpu_export_lock(root, 2):
                with self.assertRaises(RuntimeError):
                    with launch_logit_exports.gpu_export_lock(root, 2):
                        pass

    def test_followup_waits_for_complete_export_and_terminal_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "logits" / "candidate"
            candidate.mkdir(parents=True)
            (candidate / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "case_count": 200,
                        "finding_count": 381,
                        "dtype": "float16",
                    }
                )
            )
            (candidate / "reproduction_validation.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "dtype": "float16",
                        "cases": 200,
                        "findings": 381,
                    }
                )
            )
            self.assertTrue(
                wait_then_launch.candidate_finished(root, "candidate")
            )
            (candidate / ".export.lock").write_text("active\n")
            self.assertFalse(
                wait_then_launch.candidate_finished(root, "candidate")
            )

    def test_followup_rejects_a_stale_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "logits" / "candidate"
            candidate.mkdir(parents=True)
            (candidate / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "case_count": 200,
                        "finding_count": 381,
                        "dtype": "float32",
                    }
                )
            )
            (candidate / "reproduction_validation.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "dtype": "float16",
                        "cases": 200,
                        "findings": 381,
                    }
                )
            )
            self.assertFalse(
                wait_then_launch.candidate_finished(root, "candidate")
            )


class ValidationTests(unittest.TestCase):
    def test_complete_case_is_resumable_and_dtype_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array_path = root / "case.npy"
            metadata_path = root / "case.json"
            np.save(
                array_path,
                np.zeros((2, 3, 4, 5), dtype=np.float16),
                allow_pickle=False,
            )
            metadata = {
                "status": "complete",
                "dtype": "float16",
                "shape": [2, 3, 4, 5],
                "npy_bytes": array_path.stat().st_size,
            }
            metadata_path.write_text(json.dumps(metadata))
            self.assertIsNotNone(
                storage_validation.validate_existing_case(
                    array_path,
                    metadata_path,
                    "float16",
                    2,
                )
            )
            self.assertIsNone(
                storage_validation.validate_existing_case(
                    array_path,
                    metadata_path,
                    "float32",
                    2,
                )
            )

    def test_explicit_same_pass_mask_mismatch_is_a_storage_failure(self) -> None:
        logits = np.asarray([[[[-1.0, 1.0]]]], dtype=np.float16)
        passed = storage_validation.case_mask_storage_proof(
            logits,
            {"same_pass_mask_mismatch_voxels": 0},
        )
        failed = storage_validation.case_mask_storage_proof(
            logits,
            {"same_pass_mask_mismatch_voxels": 1},
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

    def test_legacy_zero_free_sign_proof_is_conservative(self) -> None:
        zero_free = np.asarray([[[[-1.0, 1.0]]]], dtype=np.float16)
        with_zero = np.asarray([[[[-0.0, 1.0]]]], dtype=np.float16)
        self.assertTrue(
            storage_validation.case_mask_storage_proof(
                zero_free,
                {},
            )["passed"]
        )
        self.assertFalse(
            storage_validation.case_mask_storage_proof(
                with_zero,
                {},
            )["passed"]
        )

    def test_completed_revalidation_queue_is_fail_soft(self) -> None:
        def fake_validator(candidate_id: str, **_kwargs):
            if candidate_id == "first":
                raise RuntimeError("broken")
            return {
                "dtype": "float16",
                "historical_reproduction_status": "warning",
                "warnings": ["historical_reproduction_mismatch"],
            }

        results = revalidate_completed_exports.revalidate_candidates(
            ["first", "second"],
            fake_validator,
            {},
        )
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["status"], "passed")


class ExportAuditTests(unittest.TestCase):
    def test_incomplete_audit_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_json = root / "first.json"
            first_md = root / "first.md"
            second_json = root / "second.json"
            second_md = root / "second.md"
            first = audit_logit_exports.run(
                HERE / "candidate_manifest.json",
                root / "runtime",
                first_json,
                first_md,
            )
            second = audit_logit_exports.run(
                HERE / "candidate_manifest.json",
                root / "runtime",
                second_json,
                second_md,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            self.assertEqual(first["accepted_candidates"], 0)

    def test_finalizer_snapshot_lists_incomplete_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = finalize_when_complete.acceptance_snapshot(
                HERE / "candidate_manifest.json",
                Path(temporary),
            )
            self.assertEqual(snapshot["accepted"], 0)
            self.assertEqual(snapshot["total"], 20)
            self.assertEqual(len(snapshot["incomplete_candidate_ids"]), 20)


if __name__ == "__main__":
    unittest.main()
