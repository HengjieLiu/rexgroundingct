#!/usr/bin/env python3
"""Focused verification for the Exp018 official-category-2d nodule audit."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if __package__:
    from .audit_category2d_nodule_methods import (
        AuditError,
        DEFAULT_SOURCES_PATH,
        REPO_ROOT,
        build_audit,
        category_positions,
        read_json,
        recompose_category_evaluation,
        scan_runtime_evaluator_paths,
        sha256_file,
        sort_leaderboard,
        validate_source_hash,
        validate_unique_primary_rows,
    )
else:
    from audit_category2d_nodule_methods import (
        AuditError,
        DEFAULT_SOURCES_PATH,
        REPO_ROOT,
        build_audit,
        category_positions,
        read_json,
        recompose_category_evaluation,
        scan_runtime_evaluator_paths,
        sha256_file,
        sort_leaderboard,
        validate_source_hash,
        validate_unique_primary_rows,
    )


class Category2dAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(DEFAULT_SOURCES_PATH)
        cls.reference_dataset = read_json(
            REPO_ROOT / cls.manifest["reference_dataset"]["path"]
        )
        cls.audit = build_audit(cls.manifest)

    def test_fixed_val200_category_map_is_exactly_132_2d_findings(self) -> None:
        positions = category_positions(self.reference_dataset, "2d")
        self.assertEqual(len(self.reference_dataset["test"]), 200)
        self.assertEqual(sum(len(row["findings"]) for row in self.reference_dataset["test"]), 381)
        self.assertEqual(len(positions), 132)
        self.assertEqual(len({case for case, _ in positions}), 119)

    def test_raw_evaluator_recomposes_headline_dice_and_hits(self) -> None:
        positions = category_positions(self.reference_dataset, "2d")
        source = next(
            row
            for row in self.manifest["runtime_evaluations"]
            if row["candidate_id"] == "exp007_cont_e050_abs_e150"
        )
        metrics = recompose_category_evaluation(
            read_json(Path(source["evaluation_path"])),
            positions,
            expected_cases=200,
            expected_findings=381,
        )
        self.assertAlmostEqual(metrics["dice"], 0.3949357063551517)
        self.assertEqual(metrics["hits"], 115)
        self.assertEqual(metrics["findings"], 132)

    def test_full_audit_has_required_direct_methods_and_current_headlines(self) -> None:
        audit = self.audit
        self.assertTrue(audit["validation"]["category_2d_findings_exact"])
        self.assertTrue(audit["validation"]["text_only_location_findings_exact"])
        self.assertTrue(audit["validation"]["training_validation_source_hashes_validated"])
        self.assertTrue(audit["validation"]["raw_validation_matches_fixed_val200"])
        self.assertTrue(audit["validation"]["training_validation_entity_counts_validated"])
        self.assertTrue(audit["validation"]["all_raw_evaluation_rows_recomposed"])
        self.assertTrue(audit["validation"]["all_source_hashes_validated"])
        self.assertTrue(audit["validation"]["all_primary_mask_threshold_evidence_validated"])
        self.assertTrue(audit["validation"]["runtime_evaluator_coverage_complete"])
        self.assertEqual(audit["validation"]["raw_evaluation_rows_recomposed"], 64)
        self.assertEqual(len(audit["primary_leaderboard"]), 58)
        self.assertEqual(len(audit["unranked_runtime_evaluations"]), 0)
        self.assertEqual(audit["runtime_evaluator_coverage"]["scanned_evaluators"], 58)
        self.assertEqual(audit["runtime_evaluator_coverage"]["ranked_primary_evaluators"], 58)
        self.assertEqual(audit["runtime_evaluator_coverage"]["unranked_evaluators"], 0)
        classified = {
            row["evaluation_path"]
            for row in audit["primary_leaderboard"]
            + audit["unranked_runtime_evaluations"]
        }
        self.assertEqual(classified, set(scan_runtime_evaluator_paths(self.manifest)))
        self.assertEqual(
            [row["candidate_id"] for row in audit["headline_pareto_references"]],
            ["exp007_cont_e050_abs_e150", "exp017_phase1_e075"],
        )
        direct_arms = {
            item["arm"] for item in audit["direct_2d_finetune_inventory"]
        }
        self.assertEqual(
            direct_arms, {"category_2d_replay50", "category_2d_target100"}
        )
        exp012 = next(
            item
            for item in audit["direct_2d_finetune_inventory"]
            if item["arm"] == "category_2d_replay50"
        )
        self.assertAlmostEqual(exp012["target_census_evidence"]["dice"], 0.37845167509335165)
        self.assertEqual(exp012["target_census_evidence"]["hits"], 113)
        self.assertAlmostEqual(exp012["fixed_val200"]["dice"], 0.36379250249120765)
        self.assertEqual(exp012["fixed_val200"]["hits"], 110)
        exp013 = next(
            item
            for item in audit["direct_2d_finetune_inventory"]
            if item["arm"] == "category_2d_target100"
        )
        self.assertEqual(len(exp013["target_census_series"]), 5)
        self.assertEqual(
            [item["hits"] for item in exp013["target_census_series"]],
            [70, 103, 107, 105, 105],
        )

    def test_train_validation_category_2d_comparison_is_complete(self) -> None:
        comparison = self.audit["training_validation_comparison"]
        self.assertEqual(
            comparison["source"]["metadata"]["sha256"],
            "3a66087608d5d0177a30c0238845a7e53518fcd82f8f41283a7d14fc02b37e6a",
        )
        self.assertEqual(comparison["test_cases"], 300)
        self.assertFalse(comparison["test_entity_counts_available"])
        self.assertEqual(
            comparison["validation_alignment"],
            {
                "case_names_match_fixed_val200": True,
                "finding_texts_and_categories_match_fixed_val200": True,
                "category_positions_match_fixed_val200": True,
                "text_location_counts_match_fixed_val200": True,
            },
        )
        self.assertIn(
            "censoring-risk indicator, not proof",
            comparison["annotation_policy"]["interpretation"],
        )
        self.assertIn(
            "no segmentation masks are loaded",
            comparison["scope"]["entity_analysis"],
        )
        self.assertIn(
            "Text only",
            comparison["scope"]["location_analysis"],
        )

        expected = {
            "train": {
                "cases": 2992,
                "category_2d_findings": 1743,
                "category_2d_cases": 1305,
                "category_2d_entities": 3185,
                "category_2d_findings_per_ct_case": 0.5825534759358288,
                "category_2d_findings_per_2d_positive_ct_case": 1.335632183908046,
                "category_2d_entities_per_finding": 1.8273092369477912,
                "category_2d_entities_per_2d_positive_ct_case": 2.4406130268199235,
                "entity_count_equal_three": 504,
                "entity_count_greater_than_three": 36,
                "max_entity_count": 5,
                "distribution": {"1": 887, "2": 316, "3": 504, "4+": 36},
            },
            "val": {
                "cases": 200,
                "category_2d_findings": 132,
                "category_2d_cases": 119,
                "category_2d_entities": 466,
                "category_2d_findings_per_ct_case": 0.66,
                "category_2d_findings_per_2d_positive_ct_case": 1.1092436974789917,
                "category_2d_entities_per_finding": 3.5303030303030303,
                "category_2d_entities_per_2d_positive_ct_case": 3.9159663865546217,
                "entity_count_equal_three": 14,
                "entity_count_greater_than_three": 46,
                "max_entity_count": 22,
                "distribution": {"1": 51, "2": 21, "3": 14, "4+": 46},
            },
        }
        for split, split_expected in expected.items():
            summary = comparison["split_summaries"][split]
            for key in (
                "cases",
                "category_2d_findings",
                "category_2d_cases",
                "category_2d_entities",
                "entity_count_equal_three",
                "entity_count_greater_than_three",
                "max_entity_count",
            ):
                self.assertEqual(summary[key], split_expected[key])
            for key in (
                "category_2d_findings_per_ct_case",
                "category_2d_findings_per_2d_positive_ct_case",
                "category_2d_entities_per_finding",
                "category_2d_entities_per_2d_positive_ct_case",
            ):
                self.assertAlmostEqual(summary[key], split_expected[key])
            self.assertEqual(
                {
                    row["bucket"]: row["findings"]
                    for row in summary["entity_count_distribution"]
                },
                split_expected["distribution"],
            )
            self.assertTrue(summary["textual_location_audit"]["is_text_only"])
            self.assertIn(
                "does not use actual lung or lobe masks",
                summary["textual_location_audit"]["text_only_note"],
            )

        train_summary = comparison["split_summaries"]["train"]
        exceptions = train_summary["entity_count_greater_than_three_records"]
        self.assertEqual(len(exceptions), 36)
        self.assertEqual(
            train_summary["entity_count_greater_than_three_breakdown"],
            [
                {"entity_count": 4, "findings": 26},
                {"entity_count": 5, "findings": 10},
            ],
        )
        self.assertEqual(
            exceptions,
            sorted(
                exceptions,
                key=lambda row: (
                    row["case"], row["finding_index"], row["entity_count"]
                ),
            ),
        )
        self.assertEqual(
            sum(row["entity_count"] for row in exceptions),
            26 * 4 + 10 * 5,
        )
        self.assertEqual(len({(row["case"], row["finding_index"]) for row in exceptions}), 36)

    def test_train_validation_location_union_tables_are_deterministic(self) -> None:
        comparison = self.audit["training_validation_comparison"]
        for rows, expected_length in (
            (comparison["primary_text_location_comparison"], 12),
            (comparison["all_matching_text_location_cue_comparison"], 22),
        ):
            self.assertEqual(len(rows), expected_length)
            self.assertEqual(
                [
                    (-row["train_fraction_of_category_findings"], row["location_description"])
                    for row in rows
                ],
                sorted(
                    (-row["train_fraction_of_category_findings"], row["location_description"])
                    for row in rows
                ),
            )
            self.assertEqual(
                [row["rank"] for row in rows], list(range(1, len(rows) + 1)),
            )
            for row in rows:
                self.assertAlmostEqual(
                    row["val_minus_train_percentage_points"],
                    100
                    * (
                        row["val_findings"] / 132
                        - row["train_findings"] / 1743
                    ),
                )

        primary = {
            row["location_description"]: row
            for row in comparison["primary_text_location_comparison"]
        }
        self.assertEqual(primary["Bilateral / both lungs"]["train_findings"], 669)
        self.assertEqual(primary["Bilateral / both lungs"]["val_findings"], 61)
        self.assertAlmostEqual(
            primary["Bilateral / both lungs"]["val_minus_train_percentage_points"],
            7.8300213842382504,
        )
        self.assertEqual(primary["No location stated"]["train_findings"], 177)
        self.assertEqual(primary["No location stated"]["val_findings"], 3)
        self.assertAlmostEqual(
            primary["No location stated"]["val_minus_train_percentage_points"],
            -7.882178062900954,
        )
        self.assertEqual(
            {
                label
                for label, row in primary.items()
                if row["train_findings"] == 0
            },
            {"Left lung fissure", "Right minor fissure / perifissural"},
        )
        cues = {
            row["location_description"]: row
            for row in comparison["all_matching_text_location_cue_comparison"]
        }
        self.assertEqual(cues["Subpleural"]["train_findings"], 172)
        self.assertEqual(cues["Subpleural"]["val_findings"], 19)

    def test_text_only_location_audit_is_complete_and_frequency_sorted(self) -> None:
        location_audit = self.audit["textual_location_audit"]
        self.assertTrue(location_audit["is_text_only"])
        self.assertIn("does not use actual lung or lobe masks", location_audit["text_only_note"])
        self.assertEqual(location_audit["category_findings"], 132)
        self.assertEqual(location_audit["category_cases"], 119)
        self.assertEqual(
            location_audit["finding_descriptions_without_literal_nodule_wording"], 3
        )

        primary = location_audit["primary_assignment"]
        self.assertEqual(primary["findings_sum"], 132)
        primary_counts = {
            row["location_description"]: row["findings"]
            for row in primary["frequency_sorted"]
        }
        self.assertEqual(
            primary_counts,
            {
                "Bilateral / both lungs": 61,
                "Left lower lobe": 16,
                "Right lower lobe": 14,
                "Right upper lobe": 11,
                "Right middle lobe": 9,
                "Left upper lobe": 8,
                "Right lung, lobe not stated": 5,
                "No location stated": 3,
                "Right minor fissure / perifissural": 2,
                "Laterobasal, side not stated": 1,
                "Left lung fissure": 1,
                "Right apical (bleb wording)": 1,
            },
        )
        self.assertEqual(
            [(-row["findings"], row["location_description"]) for row in primary["frequency_sorted"]],
            sorted(
                (-row["findings"], row["location_description"])
                for row in primary["frequency_sorted"]
            ),
        )
        self.assertEqual(
            [row["rank"] for row in primary["frequency_sorted"]],
            list(range(1, len(primary["frequency_sorted"]) + 1)),
        )

        cues = location_audit["all_matching_location_cues"]
        self.assertTrue(cues["multi_label"])
        cue_counts = {
            row["location_description"]: row["findings"]
            for row in cues["frequency_sorted"]
        }
        self.assertEqual(cue_counts["Bilateral / both lungs"], 61)
        self.assertEqual(
            cue_counts["Basal / laterobasal / posterobasal / anterobasal / mediobasal"],
            20,
        )
        self.assertEqual(cue_counts["Left lower lobe"], 19)
        self.assertEqual(cue_counts["Subpleural"], 19)
        self.assertEqual(cue_counts["Right lower lobe"], 16)
        self.assertEqual(cue_counts["Right upper lobe"], 12)
        self.assertEqual(
            [(-row["findings"], row["location_description"]) for row in cues["frequency_sorted"]],
            sorted(
                (-row["findings"], row["location_description"])
                for row in cues["frequency_sorted"]
            ),
        )

    def test_sort_order_is_dice_then_hits_then_candidate_id(self) -> None:
        rows = [
            {"candidate_id": "z", "dice": 0.4, "hits": 100},
            {"candidate_id": "b", "dice": 0.4, "hits": 101},
            {"candidate_id": "a", "dice": 0.4, "hits": 101},
            {"candidate_id": "c", "dice": 0.41, "hits": 1},
        ]
        self.assertEqual(
            [row["candidate_id"] for row in sort_leaderboard(rows)],
            ["c", "a", "b", "z"],
        )

    def test_duplicate_primary_rows_are_rejected(self) -> None:
        rows = [
            {"candidate_id": "same", "evaluation_path": "/source/a.json"},
            {"candidate_id": "same", "evaluation_path": "/source/b.json"},
        ]
        with self.assertRaisesRegex(AuditError, "Duplicate primary candidate ID"):
            validate_unique_primary_rows(rows)
        rows[1] = {"candidate_id": "different", "evaluation_path": "/source/a.json"}
        with self.assertRaisesRegex(AuditError, "Duplicate primary evaluator source"):
            validate_unique_primary_rows(rows)

    def test_source_hash_validation_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_text('{"fixture": true}\n')
            expected = sha256_file(source)
            self.assertEqual(validate_source_hash(source, expected, "fixture"), expected)
            with self.assertRaisesRegex(AuditError, "SHA-256 mismatch"):
                validate_source_hash(source, "0" * 64, "fixture")

    def test_generator_regenerates_deterministically_and_check_detects_stale_output(self) -> None:
        script = REPO_ROOT / "scripts/rexgroundingct/audit_category2d_nodule_methods.py"
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "audit"
            command = [sys.executable, str(script), "--output-dir", str(output_dir)]
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
            first_report = (output_dir / "report.md").read_bytes()
            first_json = (output_dir / "nodule_method_audit.json").read_bytes()
            rendered_report = first_report.decode("utf-8")
            for required_text in (
                "Partial-instance (up to 3 instances per finding)",
                "Exhaustive (all instances segmented by radiologists)",
                "This is metadata, not a new segmentation-mask audit.",
                "censoring-risk indicator",
                "Text-only location limitation",
                "does not use actual lung or lobe masks",
                "excluded from these entity tables because this metadata snapshot has no test entity_counts",
            ):
                self.assertIn(required_text, rendered_report)
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
            self.assertEqual((output_dir / "report.md").read_bytes(), first_report)
            self.assertEqual((output_dir / "nodule_method_audit.json").read_bytes(), first_json)
            subprocess.run(
                [*command, "--check"], cwd=REPO_ROOT, env=environment, check=True
            )
            (output_dir / "report.md").write_text("stale\n")
            stale = subprocess.run(
                [*command, "--check"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("Generated Exp018 audit differs", stale.stderr)

    def test_rejects_missing_direct_inventory_arm(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["direct_2d_finetunes"] = manifest["direct_2d_finetunes"][:1]
        with self.assertRaisesRegex(AuditError, "Direct 2d fine-tune inventory is incomplete"):
            build_audit(manifest)

    def test_rejects_train_validation_metadata_hash_drift(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["training_validation_comparison"]["metadata_expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(AuditError, "SHA-256 mismatch"):
            build_audit(manifest)


if __name__ == "__main__":
    unittest.main()
