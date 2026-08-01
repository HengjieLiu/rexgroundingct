#!/usr/bin/env python3
"""Tests for the committed val200 results packager."""

from __future__ import annotations

import unittest
from pathlib import Path

from package_results import (
    CATEGORY_CODES,
    CATEGORY_DIRECTORIES,
    CATEGORY_FIGURE_COUNTS,
    CATEGORY_FINDING_COUNTS,
    DEFAULT_ANALYSIS_SUMMARY,
    DEFAULT_SOURCE,
    DEFAULT_SPLIT_AUDIT,
    MODEL_KEYS,
    PackageContext,
    batch_items,
    build_category_readme,
    category_directory,
    load_context,
    summarize_metrics,
    val_index,
)


class PackageResultsTests(unittest.TestCase):
    def test_category_contract_includes_documented_empty_2f_directory(self) -> None:
        self.assertEqual(
            CATEGORY_DIRECTORIES,
            (
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
                "2f_empty",
                "2g",
                "2h",
            ),
        )
        self.assertEqual(category_directory("2f"), "2f_empty")
        self.assertEqual(CATEGORY_FIGURE_COUNTS["2f"], 0)
        self.assertEqual(sum(CATEGORY_FIGURE_COUNTS.values()), 342)
        self.assertEqual(sum(CATEGORY_FINDING_COUNTS.values()), 381)

    def test_batches_never_exceed_ten_figures(self) -> None:
        items = [Path(f"val{index:03d}_case.png") for index in range(23)]
        batches = batch_items(items)
        self.assertEqual([len(batch) for batch in batches], [10, 10, 3])
        self.assertEqual([item for batch in batches for item in batch], items)

    def test_val_first_filename_parser(self) -> None:
        self.assertEqual(val_index(Path("val000_train_19753_a_2.png")), 0)
        self.assertEqual(val_index(Path("val199_train_999_a_1.png")), 199)
        with self.assertRaises(ValueError):
            val_index(Path("train_19753_a_2_val000.png"))

    def test_metric_summary_preserves_model_order_and_hit_threshold(self) -> None:
        rows = []
        for code in CATEGORY_CODES:
            if code == "2f":
                continue
            row = {"category": code}
            for model_index, key in enumerate(MODEL_KEYS):
                dice = 0.1 if model_index % 2 == 0 else 0.099999
                row[f"{key}_dice"] = str(dice)
                row[f"{key}_hit"] = str(dice >= 0.1)
            rows.append(row)
        overall, by_category = summarize_metrics(rows)
        self.assertEqual([row["model_key"] for row in overall], list(MODEL_KEYS))
        self.assertEqual([row["hit_count"] for row in overall], [13, 0, 13, 0])
        two_f = [row for row in by_category if row["category"] == "2f"]
        self.assertEqual(len(two_f), 4)
        self.assertTrue(all(row["mean_dice"] is None for row in two_f))

    def test_empty_category_readme_explains_zero_pngs(self) -> None:
        split_rows = {
            code: {
                "train_count": "16" if code == "2f" else "1",
                "train_ratio": "0.002",
                "validation_count": "0" if code == "2f" else "1",
                "validation_ratio": "0.0" if code == "2f" else "0.01",
                "test_count": "0" if code == "2f" else "1",
                "test_ratio": "0.0" if code == "2f" else "0.01",
            }
            for code in CATEGORY_CODES
        }
        finding_rows = []
        for code in CATEGORY_CODES:
            if code == "2f":
                continue
            row = {"category": code}
            for key in MODEL_KEYS:
                row[f"{key}_dice"] = "0.2"
                row[f"{key}_hit"] = "True"
            finding_rows.append(row)
        overall, by_category = summarize_metrics(finding_rows)
        context = PackageContext(
            source=Path("/source"),
            manifest={},
            finding_rows=finding_rows,
            category_index_rows=[],
            split_rows=split_rows,
            split_summaries={},
            source_pngs={code: tuple() for code in CATEGORY_CODES},
            overall_rows=overall,
            category_metric_rows=by_category,
        )
        readme = build_category_readme(context, "2f")
        self.assertIn("gallery intentionally contains zero PNGs", readme)
        self.assertIn("training split has 16", readme)
        self.assertNotIn("<details>", readme)
        self.assertNotIn("![", readme)

    @unittest.skipUnless(
        DEFAULT_SOURCE.is_dir()
        and DEFAULT_SPLIT_AUDIT.is_file()
        and DEFAULT_ANALYSIS_SUMMARY.is_file(),
        "external full-val200 gallery is not mounted",
    )
    def test_default_external_gallery_satisfies_full_contract(self) -> None:
        context = load_context(DEFAULT_SOURCE, DEFAULT_SPLIT_AUDIT, DEFAULT_ANALYSIS_SUMMARY)
        self.assertEqual(len(context.finding_rows), 381)
        self.assertEqual(sum(len(paths) for paths in context.source_pngs.values()), 342)
        self.assertEqual(
            [row["hit_count"] for row in context.overall_rows],
            [204, 285, 288, 296],
        )


if __name__ == "__main__":
    unittest.main()
