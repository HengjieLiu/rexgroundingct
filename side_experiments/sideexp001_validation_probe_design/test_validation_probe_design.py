from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SIDEEXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDEEXP_DIR))

import analyze_validation_probe as analysis
import summarize_probe_eval as summarizer


class DistributionAuditTests(unittest.TestCase):
    def test_full_test_metadata_counts_and_hash(self) -> None:
        manifest = analysis.read_json(SIDEEXP_DIR / "historical_inputs.json")
        metadata_path = analysis.resolve_path(
            manifest["metadata_json"], analysis.repository_root()
        )
        if not metadata_path.is_file():
            self.skipTest("external challenge metadata is unavailable")
        metadata = analysis.read_json(metadata_path)
        categories = sorted(
            {
                category
                for split in ("train", "val", "test")
                for entry in metadata[split]
                for category in entry.get("categories", {}).values()
            },
            key=analysis.category_sort_key,
        )
        cases, findings, counts = analysis.split_category_counts(
            metadata, "test", categories
        )
        self.assertEqual((cases, findings), (300, 582))
        self.assertEqual(
            counts,
            {
                "1a": 6,
                "1b": 11,
                "1c": 27,
                "1d": 9,
                "1e": 16,
                "1f": 4,
                "2a": 113,
                "2b": 89,
                "2c": 87,
                "2d": 190,
                "2e": 27,
                "2f": 0,
                "2g": 1,
                "2h": 2,
            },
        )
        self.assertEqual(
            analysis.sha256_file(metadata_path),
            "3a66087608d5d0177a30c0238845a7e53518fcd82f8f41283a7d14fc02b37e6a",
        )

    def test_public_subset_is_labeled_and_hashed_as_150_case_subset(self) -> None:
        path = (
            analysis.repository_root()
            / "challenge_info/snapshots/2026-07-22T001343Z/raw/"
            "firestore.googleapis.com/rexgrounding-challenge/leaderboard.json"
        )
        categories = [
            "1a",
            "1b",
            "1c",
            "1d",
            "1e",
            "1f",
            "2a",
            "2b",
            "2c",
            "2d",
            "2e",
            "2f",
            "2g",
            "2h",
        ]
        result = analysis.extract_public_leaderboard_counts(path, categories)
        self.assertEqual((result["cases"], result["findings"]), (150, 302))
        self.assertEqual(result["leaderboard_split"], "public-50pct")
        self.assertIn("membership undisclosed", result["label"])
        self.assertEqual(result["supporting_rows"], 16)
        self.assertEqual(result["excluded_legacy_unknown_rows"], 1)
        self.assertEqual(sum(result["counts"].values()), 302)
        self.assertEqual(result["counts"]["2d"], 97)
        self.assertEqual(
            result["sha256"],
            "d33ccef8f34dfb2e242eb48f648f88fad8d40948db972f37c4748bcb2de47f7a",
        )

    def test_shift_threshold_and_low_count_rules(self) -> None:
        self.assertEqual(
            analysis.shift_flag(10, 100, 13, 100), "VAL-TEST SHIFT"
        )
        self.assertEqual(
            analysis.shift_flag(10, 1000, 20, 1000), "VAL-TEST SHIFT"
        )
        self.assertEqual(
            analysis.shift_flag(2, 1000, 4, 1000),
            "LOW-COUNT SHIFT WARNING",
        )
        self.assertEqual(analysis.shift_flag(10, 1000, 12, 1000), "")

    def test_benjamini_hochberg_correction(self) -> None:
        adjusted = analysis.benjamini_hochberg([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])


class SelectionTests(unittest.TestCase):
    def test_census_constraints_include_whole_rare_category_cases(self) -> None:
        case_category_counts = np.asarray(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 3, 0],
                [0, 2, 1],
                [0, 2, 1],
            ]
        )
        mandatory = analysis.mandatory_case_positions(2, case_category_counts)
        np.testing.assert_array_equal(mandatory, [0, 1, 3, 4])

    def test_candidate_generation_is_deterministic_and_covers_strata(self) -> None:
        case_stratum_counts = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )
        left = analysis.generate_candidate(
            np.random.default_rng(71), 4, np.asarray([0]), case_stratum_counts
        )
        right = analysis.generate_candidate(
            np.random.default_rng(71), 4, np.asarray([0]), case_stratum_counts
        )
        self.assertEqual(left, right)
        self.assertEqual(len(left), 4)
        self.assertIn(0, left)
        self.assertTrue(
            np.all(case_stratum_counts[np.asarray(left)].sum(axis=0) > 0)
        )

    def test_metric_reconstruction_checks_dice_and_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actual_path = Path(directory) / "actual.json"
            actual_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "mean_global_dice_per_finding": 0.5,
                            "total_hits": 1,
                        }
                    }
                )
            )
            spec = analysis.EvaluationSpec(
                experiment="fixture",
                arm="arm",
                checkpoint=1,
                role="design",
                val200_path=Path("unused"),
                val20_path=actual_path,
            )
            collection = analysis.EvaluationCollection(
                specs=[spec],
                dice=np.asarray([[[0.25, 0.0], [0.0, 0.75]]]),
                hit=np.asarray([[[1.0, 0.0], [0.0, 0.0]]]),
                finding_dice=np.asarray([[0.25, 0.75]]),
                finding_hit=np.asarray([[1.0, 0.0]]),
                population_counts=np.asarray([1, 1]),
            )
            rows = analysis.legacy_reconstruction_rows(
                [spec], collection, [0, 1], np.asarray([[1, 0], [0, 1]])
            )
            self.assertTrue(rows[0]["passed"])
            self.assertEqual(rows[0]["derived_hits"], 1)

    def test_holdout_loading_occurs_after_selection_hash_freeze(self) -> None:
        source = inspect.getsource(analysis.run_analysis)
        freeze_position = source.index("selection_sha256_pre_holdout =")
        holdout_position = source.index(
            "holdout, holdout_hashes = load_evaluation_collection"
        )
        self.assertLess(freeze_position, holdout_position)
        freeze_block = source[source.index("freeze_payload =") : freeze_position]
        self.assertNotIn("holdout", freeze_block)


class SummarizerTests(unittest.TestCase):
    def test_raw_poststratified_and_test_extrapolation_weights(self) -> None:
        probe = {
            "test": [
                {
                    "name": "case_a",
                    "findings": {"0": "a"},
                    "categories": {"0": "a"},
                },
                {
                    "name": "case_b",
                    "findings": {"0": "b"},
                    "categories": {"0": "b"},
                },
            ]
        }
        manifest = {
            "side_experiment_id": analysis.SIDE_EXPERIMENT_ID,
            "status": "accepted",
            "selection_sha256_pre_holdout": "fixture",
            "selection": {"category_counts": {"a": 1, "b": 1}},
            "weights": {
                "validation": {"a": 0.25, "b": 0.75},
                "full_test": {"a": 0.75, "b": 0.25},
            },
        }
        evaluation = {
            "cases": [
                {
                    "file": "case_a",
                    "findings": {
                        "finding_0": {"global_dice": 0.2, "global_hit": False}
                    },
                },
                {
                    "file": "case_b",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True}
                    },
                },
            ]
        }
        result = summarizer.summarize(probe, manifest, evaluation)
        self.assertAlmostEqual(
            result["metrics"]["raw_evaluator"]["mean_global_dice_per_finding"],
            0.5,
        )
        self.assertAlmostEqual(
            result["metrics"]["validation_post_stratified"][
                "mean_global_dice_per_finding"
            ],
            0.65,
        )
        extrapolation = result["metrics"]["full_test_reweighted_extrapolation"]
        self.assertAlmostEqual(extrapolation["mean_global_dice_per_finding"], 0.35)
        self.assertIn("not measured test performance", extrapolation["label"])


if __name__ == "__main__":
    unittest.main()
