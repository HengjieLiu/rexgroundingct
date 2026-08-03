#!/usr/bin/env python3
"""Focused tests for Exp012 schedules, subsets, and checkpoint selection."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_012_category_specialists import (
    DEFAULT_VAL200,
    DEFAULT_VAL80,
    DIFFUSE_TARGET_COUNTS,
    MASTER_SEED,
    REX_METADATA,
    audit_schedule,
    make_target_event,
    prepare_evaluation_subsets,
    target_category_sequence,
)
from summarize_012_results import select_checkpoint, summarize_evaluation


class Exp012ScheduleTest(unittest.TestCase):
    def test_target_event_guarantees_requested_anchor(self) -> None:
        entry = {
            "name": "train_synthetic.nii.gz",
            "_split": "train",
            "_index": 7,
            "findings": {"0": "Target finding", "1": "Partner finding"},
            "categories": {"0": "2a", "1": "2b"},
        }
        options = {
            "spatial_shape": [256, 256, 256],
            "points": {0: [100, 100, 100], 1: [110, 110, 110]},
            "compatible": {0: [1], 1: [0]},
        }
        event = make_target_event(
            entry,
            target_index=0,
            target_category="2a",
            options=options,
            negative_pool=["unrelated negative"],
            rng=np.random.default_rng(3),
        )
        self.assertEqual(event["anchor_target_index"], 0)
        self.assertEqual(event["requested_target_category"], "2a")
        self.assertEqual(len(event["prompt_slots"]), 3)
        self.assertTrue(
            any(
                slot["is_positive"] and slot["target_index"] == 0
                for slot in event["prompt_slots"]
            )
        )
        self.assertIn(0, {item["target_index"] for item in event["positive_crop_points"]})

    def test_diffuse_allocation_is_exact(self) -> None:
        sequence = target_category_sequence(
            "category_1alldiffuse_replay50", np.random.default_rng(MASTER_SEED)
        )
        observed = {category: sequence.count(category) for category in DIFFUSE_TARGET_COUNTS}
        self.assertEqual(observed, DIFFUSE_TARGET_COUNTS)
        self.assertEqual(len(sequence), 5000)

    def test_full_schedule_audit_enforces_each_epoch_ratio(self) -> None:
        categories = [
            category
            for category, count in DIFFUSE_TARGET_COUNTS.items()
            for _ in range(count)
        ]
        events = []
        target_cursor = 0
        replay_cursor = 20_000
        for epoch in range(100):
            for position in range(100):
                if position < 50:
                    category = categories[target_cursor]
                    event = {
                        "event_index": len(events),
                        "sampling_source": "targeted",
                        "requested_target_category": category,
                        "requested_target_index": 0,
                        "anchor_target_index": 0,
                        "prompt_slots": [
                            {"prompt": "target", "target_index": 0, "is_positive": True},
                            {"prompt": "negative 1", "target_index": None, "is_positive": False},
                            {"prompt": "negative 2", "target_index": None, "is_positive": False},
                        ],
                        "positive_crop_points": [{"target_index": 0, "slot_index": 0, "point": [0, 0, 0]}],
                    }
                    target_cursor += 1
                else:
                    event = {
                        "event_index": len(events),
                        "sampling_source": "replay",
                        "replay_source_event_index": replay_cursor,
                        "case_name": f"case_{replay_cursor}",
                        "prompt_slots": [],
                        "patch_starts": [0, 0, 0],
                    }
                    replay_cursor += 1
                events.append(event)
        result = audit_schedule(events, "category_1alldiffuse_replay50")
        self.assertEqual(result["sampling_source_counts"], {"targeted": 5000, "replay": 5000})
        self.assertEqual(result["target_anchor_failures"], 0)


class Exp012EvaluationTest(unittest.TestCase):
    def test_reporter_filters_target_non_target_and_sentinel(self) -> None:
        dataset = {
            "test": [
                {
                    "name": "case_a.nii.gz",
                    "findings": {"0": "target", "1": "other"},
                    "categories": {"0": "2a", "1": "2b"},
                },
                {
                    "name": "case_b.nii.gz",
                    "findings": {"0": "other"},
                    "categories": {"0": "2b"},
                },
            ]
        }
        evaluation = {
            "summary": {
                "total_cases": 2,
                "total_findings": 3,
                "total_hits": 2,
                "mean_global_dice_per_finding": 0.5,
                "hit_rate": 2 / 3,
            },
            "cases": [
                {
                    "file": "case_a.nii.gz",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True},
                        "finding_1": {"global_dice": 0.2, "global_hit": False},
                    },
                },
                {
                    "file": "case_b.nii.gz",
                    "findings": {
                        "finding_0": {"global_dice": 0.5, "global_hit": True}
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            eval_path = Path(tmp) / "evaluation.json"
            dataset_path.write_text(json.dumps(dataset))
            eval_path.write_text(json.dumps(evaluation))
            result = summarize_evaluation(
                eval_path,
                dataset_path,
                {"2a"},
                {("case_a.nii.gz", "0"), ("case_a.nii.gz", "1")},
            )
        self.assertAlmostEqual(result["target"]["mean_global_dice_per_finding"], 0.8)
        self.assertEqual(result["non_target"]["findings"], 2)
        self.assertEqual(result["val80_total"]["findings"], 2)
        self.assertEqual(result["val80_total"]["hits"], 1)
        self.assertAlmostEqual(result["val80_total"]["mean_global_dice_per_finding"], 0.5)
        self.assertEqual(result["sentinel_non_target"]["findings"], 1)
        self.assertAlmostEqual(
            result["sentinel_non_target"]["mean_global_dice_per_finding"], 0.2
        )

    def test_reporter_val80_uses_exact_finding_identity(self) -> None:
        dataset = {
            "test": [
                {
                    "name": "case_a.nii.gz",
                    "findings": {"0": "target", "1": "other"},
                    "categories": {"0": "2a", "1": "2b"},
                }
            ]
        }
        evaluation = {
            "summary": {
                "total_cases": 1,
                "total_findings": 2,
                "total_hits": 1,
                "mean_global_dice_per_finding": 0.5,
                "hit_rate": 0.5,
            },
            "cases": [
                {
                    "file": "case_a.nii.gz",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True},
                        "finding_1": {"global_dice": 0.2, "global_hit": False},
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            eval_path = Path(tmp) / "evaluation.json"
            dataset_path.write_text(json.dumps(dataset))
            eval_path.write_text(json.dumps(evaluation))
            result = summarize_evaluation(
                eval_path,
                dataset_path,
                {"2a"},
                {("case_a.nii.gz", "1")},
            )
        self.assertEqual(result["val80_total"]["findings"], 1)
        self.assertAlmostEqual(result["val80_total"]["mean_global_dice_per_finding"], 0.2)

    def test_reporter_rejects_missing_val80_finding(self) -> None:
        dataset = {
            "test": [
                {
                    "name": "case_a.nii.gz",
                    "findings": {"0": "target"},
                    "categories": {"0": "2a"},
                }
            ]
        }
        evaluation = {
            "summary": {
                "total_cases": 1,
                "total_findings": 1,
                "total_hits": 1,
                "mean_global_dice_per_finding": 0.8,
                "hit_rate": 1.0,
            },
            "cases": [
                {
                    "file": "case_a.nii.gz",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True}
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            eval_path = Path(tmp) / "evaluation.json"
            dataset_path.write_text(json.dumps(dataset))
            eval_path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, "missing 1 fixed-val80 finding"):
                summarize_evaluation(
                    eval_path,
                    dataset_path,
                    {"2a"},
                    {("case_a.nii.gz", "0"), ("case_a.nii.gz", "1")},
                )

    def test_target_only_evaluation_leaves_val80_metrics_pending(self) -> None:
        dataset = {
            "test": [
                {
                    "name": "case_a.nii.gz",
                    "findings": {"0": "target"},
                    "categories": {"0": "2a"},
                }
            ]
        }
        evaluation = {
            "summary": {
                "total_cases": 1,
                "total_findings": 1,
                "total_hits": 1,
                "mean_global_dice_per_finding": 0.8,
                "hit_rate": 1.0,
            },
            "cases": [
                {
                    "file": "case_a.nii.gz",
                    "findings": {
                        "finding_0": {"global_dice": 0.8, "global_hit": True}
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            eval_path = Path(tmp) / "evaluation.json"
            dataset_path.write_text(json.dumps(dataset))
            eval_path.write_text(json.dumps(evaluation))
            result = summarize_evaluation(
                eval_path,
                dataset_path,
                {"2a"},
                {("not_evaluated.nii.gz", "0")},
                include_val80_metrics=False,
            )
        self.assertIsNone(result["val80_total"])
        self.assertIsNone(result["sentinel_non_target"])

    def test_real_validation_subsets_reproduce_audited_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                metadata=REX_METADATA,
                val200_json=DEFAULT_VAL200,
                val80_json=DEFAULT_VAL80,
                evaluation_output_dir=Path(tmp),
            )
            manifest = prepare_evaluation_subsets(args)
        observed = {
            arm: (
                record["target"]["cases"],
                record["target"]["target_findings"],
                record["union"]["cases"],
            )
            for arm, record in manifest["arms"].items()
        }
        self.assertEqual(
            observed,
            {
                "category_1alldiffuse_replay50": (42, 52, 80),
                "category_2a_replay50": (63, 69, 114),
                "category_2b_replay50": (40, 49, 98),
                "category_2c_replay50": (53, 60, 110),
            },
        )

    def test_selection_uses_target_dice_then_hit_then_earlier_epoch(self) -> None:
        milestones = {}
        for epoch, dice, hit in [(0, 0.30, 0.7), (5, 0.35, 0.7), (20, 0.3500005, 0.8)]:
            milestones[str(epoch)] = {
                "evaluation": {
                    "target": {
                        "mean_global_dice_per_finding": dice,
                        "hit_rate": hit,
                        "hits": 1,
                        "findings": 1,
                    }
                }
            }
        with tempfile.TemporaryDirectory() as tmp:
            selection = select_checkpoint(
                Path(tmp), Path("/source/checkpoint.pth"), "category_2a_replay50", milestones
            )
        self.assertIsNotNone(selection)
        self.assertEqual(selection["epoch"], 20)


if __name__ == "__main__":
    unittest.main()
