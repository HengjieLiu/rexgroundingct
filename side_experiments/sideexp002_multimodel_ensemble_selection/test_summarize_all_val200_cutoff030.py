#!/usr/bin/env python3
"""Tests for the all-method fixed-val200 category report."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_all_val200_cutoff030 import (
    CATEGORY_LABELS,
    DEFAULT_AUDIT,
    DEFAULT_DATASET,
    EXPECTED_FINDINGS,
    EXPECTED_CASES,
    HIT_DICE_THRESHOLD,
    OVERALL_DICE_CUTOFF,
    aggregate_evaluation,
    build_oracle,
    build_summary,
    rank_candidates,
    rank_categories,
    select_category_winners,
)


def _candidate(candidate_id: str, dice: float, hits: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "overall": {
            "dice": dice,
            "hits": hits,
        },
        "categories": {
            code: {
                "rank": 1,
                "findings": 1,
                "dice": dice,
                "hits": int(dice >= HIT_DICE_THRESHOLD),
                "hit_rate": int(dice >= HIT_DICE_THRESHOLD),
            }
            for code in CATEGORY_LABELS
        },
    }


class TestAllVal200Report(unittest.TestCase):
    def test_aggregate_evaluation_recomposes_categories(self) -> None:
        category_by_key = {
            ("case_a", 0): "1a",
            ("case_a", 1): "2d",
            ("case_b", 0): "1a",
            ("case_b", 1): "2d",
        }
        category_counts = {code: 0 for code in CATEGORY_LABELS}
        category_counts["1a"] = 2
        category_counts["2d"] = 2
        evaluation = {
            "summary": {
                "params": {"global_only": True, "global_hit_thr": HIT_DICE_THRESHOLD},
                "total_cases": 2,
                "total_findings": 4,
                "total_hits": 3,
                "mean_global_dice_per_finding": (0.2 + 0.4 + 0.8 + 0.05) / 4,
            },
            "cases": [
                {
                    "file": "case_a",
                    "findings": {
                        "finding_0": {"global_dice": 0.2, "global_hit": True},
                        "finding_1": {"global_dice": 0.4, "global_hit": True},
                    },
                },
                {
                    "file": "case_b",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True},
                        "finding_1": {"global_dice": 0.05, "global_hit": False},
                    },
                },
            ],
        }
        _, metrics = aggregate_evaluation(
            evaluation,
            category_by_key,
            category_counts,
            expected_cases=2,
            expected_findings=4,
        )
        self.assertAlmostEqual(metrics["overall"]["dice"], 0.3625)
        self.assertEqual(metrics["overall"]["hits"], 3)
        self.assertAlmostEqual(metrics["categories"]["1a"]["dice"], 0.5)
        self.assertEqual(metrics["categories"]["2d"]["hits"], 1)

    def test_cutoff_and_global_ranking(self) -> None:
        candidates = [
            _candidate("low", 0.299999, 381),
            _candidate("high_fewer_hits", 0.35, 290),
            _candidate("high_more_hits", 0.35, 291),
        ]
        ranked = rank_candidates(candidates)
        self.assertEqual([candidate["candidate_id"] for candidate in ranked], [
            "high_more_hits",
            "high_fewer_hits",
            "low",
        ])
        eligible = [candidate for candidate in ranked if candidate["overall"]["dice"] >= OVERALL_DICE_CUTOFF]
        self.assertEqual([candidate["candidate_id"] for candidate in eligible], [
            "high_more_hits",
            "high_fewer_hits",
        ])

    def test_category_tie_break_uses_hits_then_overall_then_id(self) -> None:
        counts = {code: 0 for code in CATEGORY_LABELS}
        counts["2d"] = 1
        candidates = [
            _candidate("z_id", 0.40, 1),
            _candidate("a_id", 0.45, 1),
            _candidate("b_id", 0.50, 1),
        ]
        for candidate, hits in zip(candidates, (2, 3, 3)):
            candidate["categories"]["2d"].update({
                "dice": 0.5,
                "hits": hits,
                "hit_rate": hits,
            })
        rank_categories(candidates, counts)
        winner = select_category_winners(candidates, counts)["2d"]
        self.assertEqual(winner["candidate_id"], "b_id")
        self.assertEqual(winner["hits"], 3)

        candidates[2]["overall"]["dice"] = 0.35
        winner = select_category_winners(candidates, counts)["2d"]
        self.assertEqual(winner["candidate_id"], "a_id")

    def test_oracle_recomposes_selected_raw_findings(self) -> None:
        counts = {code: 0 for code in CATEGORY_LABELS}
        counts["1a"] = 3
        counts["2d"] = 132
        counts["2h"] = EXPECTED_FINDINGS - 135
        category_by_key = {}
        cursor = 0
        for code, support in counts.items():
            for offset in range(support):
                category_by_key[(f"case_{cursor // 3}", cursor % 3)] = code
                cursor += 1

        candidates = []
        for candidate_id, special_code in (("model_a", "1a"), ("model_b", "2d")):
            finding_metrics = {}
            categories = {}
            for code, support in counts.items():
                if not support:
                    categories[code] = None
                    continue
                dice = 0.4 if code == special_code else 0.2
                hits = support
                categories[code] = {
                    "findings": support,
                    "dice": dice,
                    "hits": hits,
                    "hit_rate": 1.0,
                    "rank": 1,
                }
                for key, actual_code in category_by_key.items():
                    if actual_code == code:
                        finding_metrics[key] = (dice, True)
            candidates.append({
                "candidate_id": candidate_id,
                "overall": {"dice": 0.2, "hits": EXPECTED_FINDINGS},
                "categories": categories,
                "finding_metrics": finding_metrics,
                "category_by_key": category_by_key,
            })
        rank_categories(candidates, counts)
        oracle = build_oracle(candidates, counts)
        expected_dice = (0.4 * 135 + 0.2 * (EXPECTED_FINDINGS - 135)) / EXPECTED_FINDINGS
        self.assertTrue(math.isclose(oracle["dice"], expected_dice, abs_tol=1e-12))
        self.assertEqual(oracle["hits"], EXPECTED_FINDINGS)
        self.assertEqual(oracle["categories"]["1a"]["candidate_id"], "model_a")
        self.assertEqual(oracle["categories"]["2d"]["candidate_id"], "model_b")

    def test_real_fixed_val200_integration_contract(self) -> None:
        summary = build_summary(DEFAULT_DATASET, DEFAULT_AUDIT)
        validation = summary["validation"]
        self.assertEqual(validation["cases"], EXPECTED_CASES)
        self.assertEqual(validation["findings"], EXPECTED_FINDINGS)
        self.assertEqual(validation["audit_rows"], 58)
        self.assertEqual(validation["eligible_rows"], 50)
        self.assertEqual(validation["excluded_rows"], 8)
        self.assertEqual(summary["eligible_methods"][0]["candidate_id"], "exp007_cont_e050_abs_e150")
        self.assertAlmostEqual(summary["eligible_methods"][0]["overall"]["dice"], 0.3460234857111323)
        self.assertEqual(summary["eligible_methods"][0]["overall"]["hits"], 296)
        self.assertAlmostEqual(summary["oracle"]["dice"], 0.3602780325749142)
        self.assertEqual(summary["oracle"]["hits"], 298)


if __name__ == "__main__":
    unittest.main()
