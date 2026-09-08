"""Regression tests for the SideExp004 collaborator leaderboard generator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_collaborator_val200_leaderboards.py")
MODULE_SPEC = importlib.util.spec_from_file_location("sideexp004_leaderboards", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
leaderboards = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(leaderboards)


class CollaboratorLeaderboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = leaderboards.build_state()

    def test_fixed_val200_coverage_and_cohort_counts(self) -> None:
        dataset = self.state["dataset"]
        source = self.state["collaborator_source"]
        self.assertEqual(dataset["manifest"]["sha256"], "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897")
        self.assertEqual(dataset["cases"], 200)
        self.assertEqual(dataset["findings"], 381)
        self.assertEqual(source["selected_model_count"], 24)
        self.assertEqual(source["source_declared_raw_like_count"], 20)
        self.assertEqual(source["source_declared_pipeline_count"], 4)
        self.assertEqual(
            dataset["category_findings"],
            {
                "1a": 3,
                "1b": 11,
                "1c": 17,
                "1d": 6,
                "1e": 11,
                "1f": 4,
                "2a": 69,
                "2b": 49,
                "2c": 60,
                "2d": 132,
                "2e": 11,
                "2f": 0,
                "2g": 1,
                "2h": 7,
            },
        )
        for record in self.state["collaborator_records"]:
            self.assertEqual(record["metrics"]["overall"]["findings"], 381)
            self.assertFalse(record["metrics"]["categories"]["2f"]["available"])

    def test_current_overall_leaders(self) -> None:
        all_records = leaderboards.collaborator_overall_rank(self.state["collaborator_records"])
        raw_records = [record for record in all_records if record["raw_like"]]
        mine = leaderboards.my_overall_rank(self.state["my_candidates"])[0]
        self.assertEqual(
            all_records[0]["model_id"],
            "lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b",
        )
        self.assertAlmostEqual(all_records[0]["metrics"]["overall"]["dice"], 0.35363191436506036)
        self.assertEqual(
            raw_records[0]["model_id"],
            "resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e",
        )
        self.assertAlmostEqual(raw_records[0]["metrics"]["overall"]["dice"], 0.34508090072683734)
        self.assertEqual(mine["candidate_id"], "exp007_ddp_bs4_e050_a499ad1c")
        self.assertEqual(mine["eligibility"], "eligible")
        self.assertAlmostEqual(mine["metrics"]["overall"]["dice"], 0.3460234857111323)

    def test_ranking_tie_break_is_deterministic(self) -> None:
        records = [
            {"id": "z", "dice": 0.3, "hits": 8},
            {"id": "b", "dice": 0.4, "hits": 2},
            {"id": "a", "dice": 0.4, "hits": 2},
            {"id": "c", "dice": 0.4, "hits": 3},
        ]
        ranked = leaderboards.rank_records(
            records,
            lambda record: record["dice"],
            lambda record: record["hits"],
            lambda record: record["id"],
        )
        self.assertEqual([record["id"] for record in ranked], ["c", "a", "b", "z"])

    def test_documents_are_deterministic_and_checkable(self) -> None:
        documents = leaderboards.render_documents(self.state)
        self.assertEqual(set(documents), set(leaderboards.OUTPUT_FILENAMES))
        self.assertIn("source-declared raw-like", documents[leaderboards.CHECKPOINT_LEADERBOARD_FILENAME])
        self.assertIn("`2f` is unavailable", documents[leaderboards.SUBCATEGORY_LEADERBOARD_FILENAME])
        self.assertIn("Pipeline-inclusive reference", documents[leaderboards.COMPARISON_FILENAME])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            leaderboards.apply_documents(output_dir, documents)
            self.assertEqual(leaderboards.check_documents(output_dir, documents), [])
            (output_dir / leaderboards.COMPARISON_FILENAME).write_text("stale\n", encoding="utf-8")
            self.assertEqual(
                leaderboards.check_documents(output_dir, documents),
                [f"stale generated output: {output_dir / leaderboards.COMPARISON_FILENAME}"],
            )


if __name__ == "__main__":
    unittest.main()
