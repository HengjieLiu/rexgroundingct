from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_collaborator_attention_variance as comparison


class StatisticalHelperTests(unittest.TestCase):
    def test_variance_range_iqr_and_mad(self) -> None:
        result = comparison.describe_values([1.0, 2.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_variance"], 1.0)
        self.assertEqual(result["sample_sd"], 1.0)
        self.assertEqual(result["minimum"], 1.0)
        self.assertEqual(result["maximum"], 3.0)
        self.assertEqual(result["range"], 2.0)
        self.assertEqual(result["q1"], 1.5)
        self.assertEqual(result["q3"], 2.5)
        self.assertEqual(result["iqr"], 1.0)
        self.assertEqual(result["mad"], 1.0)

    def test_percentile_uses_tie_aware_midrank(self) -> None:
        reference = [1.0, 1.0, 2.0, 3.0]
        tied = comparison.empirical_midrank_percentile(1.0, reference)
        middle = comparison.empirical_midrank_percentile(2.0, reference)
        self.assertEqual(tied["less"], 0)
        self.assertEqual(tied["equal"], 2)
        self.assertEqual(tied["percentile"], 25.0)
        self.assertEqual(middle["less"], 2)
        self.assertEqual(middle["equal"], 1)
        self.assertEqual(middle["percentile"], 62.5)

    def test_evidence_rule_requires_support_direction_and_percentiles(self) -> None:
        self.assertEqual(
            comparison.evidence_label(49, 0.01, 0.02, 98.0, 96.0, 99.0),
            "attention_associated_outlier",
        )
        self.assertEqual(
            comparison.evidence_label(19, 0.01, 0.02, 100.0, 100.0, 100.0),
            "low_support_no_attribution",
        )
        self.assertEqual(
            comparison.evidence_label(49, 0.01, -0.02, 100.0, 100.0, 100.0),
            "mixed_direction_not_outlier",
        )
        self.assertEqual(
            comparison.evidence_label(132, -0.01, -0.02, 50.0, 99.0, 99.0),
            "possible_attention_tradeoff_not_outlier",
        )


class CollaboratorAttentionComparisonTests(unittest.TestCase):
    def test_real_summary_reconstructs_requested_comparison(self) -> None:
        summary = comparison.build_summary(
            comparison.DEFAULT_COLLABORATOR_MANIFEST,
            comparison.DEFAULT_BASELINE_SUMMARY,
        )
        self.assertEqual(summary["counts"]["collaborator_variants"], 3)
        self.assertEqual(summary["counts"]["twenty_model_candidates"], 20)
        self.assertEqual(summary["counts"]["represented_categories"], 13)
        self.assertEqual(summary["counts"]["matched_triplets_per_category"], 1140)
        self.assertEqual(summary["counts"]["model_pairs"], 190)
        self.assertEqual(sum(summary["category_supports"].values()), 381)
        self.assertFalse(summary["categories"]["2f"]["available"])
        self.assertEqual(summary["categories"]["2g"]["support"], 1)
        self.assertEqual(
            summary["categories"]["2b"]["evidence_label"],
            "attention_associated_outlier",
        )
        self.assertEqual(
            summary["categories"]["2d"]["evidence_label"],
            "possible_attention_tradeoff_not_outlier",
        )
        self.assertEqual(
            summary["categories"]["1f"]["evidence_label"],
            "low_support_no_attribution",
        )
        self.assertAlmostEqual(
            summary["categories"]["2b"][
                "matched_triplet_spread_reference"
            ]["position"]["percentile"],
            98.42105263157895,
        )
        self.assertTrue(
            summary["validation"]["triplet_distribution_count_is_1140"]
        )
        self.assertTrue(summary["validation"]["pair_distribution_count_is_190"])

    def test_transcription_and_overall_recomposition(self) -> None:
        summary = comparison.build_summary(
            comparison.DEFAULT_COLLABORATOR_MANIFEST,
            comparison.DEFAULT_BASELINE_SUMMARY,
        )
        rows = {row["code"]: row for row in summary["transcription"]["rows"]}
        self.assertEqual(rows["2b"]["scores"]["baseline"], "0.3418")
        self.assertEqual(rows["2b"]["scores"]["all_category"], "0.3578")
        self.assertEqual(rows["2b"]["scores"]["strict_2b2c"], "0.3625")
        self.assertEqual(rows["2b"]["support"], 49)
        self.assertFalse(summary["transcription"]["flagged_delta_discrepancies"])
        self.assertTrue(
            summary["transcription"]["delta_checks"]["1b"]["all_category"][
                "within_rounding_tolerance"
            ]
        )
        overall = summary["collaborator_recomposed_overall"]
        self.assertAlmostEqual(
            overall["baseline"]["support_weighted_dice"],
            0.33443858267716536,
        )
        self.assertAlmostEqual(
            overall["all_category"]["support_weighted_dice"],
            0.3355160104986877,
        )
        self.assertAlmostEqual(
            overall["strict_2b2c"]["support_weighted_dice"],
            0.33085695538057747,
        )

    def test_delta_beyond_rounding_tolerance_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest = json.loads(
                comparison.DEFAULT_COLLABORATOR_MANIFEST.read_text()
            )
            manifest["categories"][0]["reported_deltas"]["all_category"] = (
                "+0.9000"
            )
            manifest_path.write_text(json.dumps(manifest))
            summary = comparison.build_summary(
                manifest_path,
                comparison.DEFAULT_BASELINE_SUMMARY,
            )
            self.assertFalse(
                summary["validation"][
                    "all_reported_deltas_within_rounding_tolerance"
                ]
            )
            self.assertEqual(
                summary["transcription"]["flagged_delta_discrepancies"][0][
                    "category"
                ],
                "1a",
            )

    def test_baseline_source_hash_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            changed_path = Path(temporary) / "changed_baseline.json"
            changed = json.loads(comparison.DEFAULT_BASELINE_SUMMARY.read_text())
            changed["warnings"].append("drift")
            changed_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "SHA256 drifted"):
                comparison.build_summary(
                    comparison.DEFAULT_COLLABORATOR_MANIFEST,
                    changed_path,
                )

    def test_markdown_and_json_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = comparison.run(
                comparison.DEFAULT_COLLABORATOR_MANIFEST,
                comparison.DEFAULT_BASELINE_SUMMARY,
                root / "first.json",
                root / "first.md",
            )
            second = comparison.run(
                comparison.DEFAULT_COLLABORATOR_MANIFEST,
                comparison.DEFAULT_BASELINE_SUMMARY,
                root / "second.json",
                root / "second.md",
            )
            self.assertEqual(first, second)
            self.assertEqual(
                (root / "first.json").read_bytes(),
                (root / "second.json").read_bytes(),
            )
            self.assertEqual(
                (root / "first.md").read_bytes(),
                (root / "second.md").read_bytes(),
            )
            self.assertRegex(first["result_sha256"], r"^[0-9a-f]{64}$")
            report = (root / "first.md").read_text()
            self.assertIn("## Take-Home Message", report)
            self.assertIn("redistribute category performance", report)
            self.assertIn("## Concise Evidence Assessment", report)
            self.assertIn("Attention-associated outlier", report)
            self.assertIn("## Secondary Context: Exp009 Attention Ablation", report)
            self.assertIn("## Secondary Context: Optimization Trajectories", report)
            self.assertIn("random-seed variance estimate", report)

    def test_cross_category_agreement_and_magnitudes(self) -> None:
        summary = comparison.build_summary(
            comparison.DEFAULT_COLLABORATOR_MANIFEST,
            comparison.DEFAULT_BASELINE_SUMMARY,
        )
        agreement = summary["cross_category"]["attention_variant_agreement"]
        self.assertEqual(
            agreement["all_represented_categories"]["same_nonzero_direction"],
            6,
        )
        self.assertEqual(
            agreement["high_support_categories"]["codes"],
            ["2a", "2b", "2c", "2d"],
        )
        self.assertEqual(
            agreement["high_support_categories"]["same_nonzero_direction"],
            3,
        )
        for variant in comparison.ATTENTION_VARIANTS:
            record = summary["cross_category"]["variants"][variant]
            self.assertTrue(math.isfinite(record["support_weighted_rms"]["value"]))
            self.assertTrue(math.isfinite(record["macro_rms"]["value"]))
            self.assertEqual(
                record["support_weighted_rms"]["position"]["reference_count"],
                190,
            )


if __name__ == "__main__":
    unittest.main()
