#!/usr/bin/env python3
"""Focused verification for Exp012 profile r02_2d_diffuse_replay_ratio."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_012_r02_replay_ratio import (
    ARM_SPECS,
    DEFAULT_VAL200,
    DEFAULT_VAL80,
    DIFFUSE_TARGET_COUNTS_BY_ARM,
    MASTER_SEED,
    REX_METADATA,
    audit_schedule,
    prepare_evaluation_subsets,
    target_category_sequence,
)
from summarize_012_r02_results import atomic_write_json, build_summary, select_checkpoint
from update_012_run_index import build_index


REPO_ROOT = Path(__file__).resolve().parents[2]


class Exp012R02ScheduleTest(unittest.TestCase):
    def test_ratios_and_diffuse_allocations_are_exact(self) -> None:
        expected_per_epoch = {
            "category_2d_replay50": (50, 50),
            "category_1alldiffuse_replay25": (75, 25),
            "category_1alldiffuse_replay10": (90, 10),
            "category_1alldiffuse_replay00": (100, 0),
        }
        for offset, (arm, spec) in enumerate(ARM_SPECS.items()):
            self.assertEqual(
                (spec["target_per_epoch"], spec["replay_per_epoch"]),
                expected_per_epoch[arm],
            )
            sequence = target_category_sequence(
                arm, np.random.default_rng(MASTER_SEED + offset)
            )
            self.assertEqual(len(sequence), 100 * expected_per_epoch[arm][0])
            if arm in DIFFUSE_TARGET_COUNTS_BY_ARM:
                self.assertEqual(
                    dict(Counter(sequence)), DIFFUSE_TARGET_COUNTS_BY_ARM[arm]
                )
            else:
                self.assertEqual(Counter(sequence), {"2d": 5000})

    def test_replay_ranges_are_disjoint_and_have_exact_lengths(self) -> None:
        ranges = [
            tuple(spec["replay_range"])
            for spec in ARM_SPECS.values()
            if spec["replay_range"] is not None
        ]
        for index, (start, stop) in enumerate(ranges):
            self.assertEqual(stop - start, [5000, 2500, 1000][index])
            for other_start, other_stop in ranges[index + 1 :]:
                self.assertTrue(stop <= other_start or other_stop <= start)

    def test_audit_enforces_every_arm_ratio_and_anchor(self) -> None:
        for arm, spec in ARM_SPECS.items():
            categories = target_category_sequence(arm, np.random.default_rng(7))
            target_cursor = 0
            replay_cursor = (spec["replay_range"] or (0, 0))[0]
            events = []
            for _epoch in range(100):
                for position in range(100):
                    if position < spec["target_per_epoch"]:
                        category = categories[target_cursor]
                        event = {
                            "event_index": len(events),
                            "sampling_source": "targeted",
                            "requested_target_category": category,
                            "requested_target_index": 0,
                            "anchor_target_index": 0,
                            "prompt_slots": [
                                {
                                    "prompt": "target",
                                    "target_index": 0,
                                    "is_positive": True,
                                }
                            ],
                            "positive_crop_points": [
                                {"target_index": 0, "slot_index": 0, "point": [0, 0, 0]}
                            ],
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
            result = audit_schedule(events, arm)
            self.assertEqual(result["events"], 10000)
            self.assertEqual(result["target_anchor_failures"], 0)
            self.assertTrue(
                all(
                    row
                    == {
                        "targeted": spec["target_per_epoch"],
                        "replay": spec["replay_per_epoch"],
                    }
                    for row in result["per_epoch"]
                )
            )


class Exp012R02EvaluationTest(unittest.TestCase):
    def test_real_subsets_reproduce_audited_counts_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                metadata=REX_METADATA,
                val200_json=DEFAULT_VAL200,
                val80_json=DEFAULT_VAL80,
                evaluation_output_dir=Path(temporary),
            )
            manifest = prepare_evaluation_subsets(args)
        observed = {
            arm: (
                record["target"]["cases"],
                record["target"]["target_findings"],
                record["union"]["cases"],
                record["union"]["all_findings"],
            )
            for arm, record in manifest["arms"].items()
        }
        self.assertEqual(observed["category_2d_replay50"], (119, 132, 160, 321))
        for arm in DIFFUSE_TARGET_COUNTS_BY_ARM:
            self.assertEqual(observed[arm], (42, 52, 80, 195))
        self.assertTrue(manifest["zero_train_validation_case_overlap"])

    def test_selection_remains_target_only(self) -> None:
        milestones = {
            "0": {
                "evaluation": {
                    "target": {"mean_global_dice_per_finding": 0.3, "hit_rate": 0.9},
                    "sentinel_non_target": {"mean_global_dice_per_finding": 0.5},
                }
            },
            "20": {
                "evaluation": {
                    "target": {"mean_global_dice_per_finding": 0.4, "hit_rate": 0.8},
                    "sentinel_non_target": {"mean_global_dice_per_finding": 0.0},
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = select_checkpoint(
                Path(temporary),
                Path("/source/checkpoint.pth"),
                "category_2d_replay50",
                milestones,
            )
        self.assertEqual(result["epoch"], 20)

    def test_queue_dry_run_has_all_gate_states(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts/rexgroundingct/queue_012_r02_after_r01.sh"),
            ],
            cwd=REPO_ROOT,
            env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1"},
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            result.stdout.splitlines()[:4],
            ["waiting_run1", "waiting_gpus", "launching", "running"],
        )

    def test_partial_failed_report_is_atomic_and_keeps_null_selections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subset = root / "subsets.json"
            val80 = root / "val80.json"
            val200 = root / "val200.json"
            subset.write_text(json.dumps({"arms": {}}))
            val80.write_text(json.dumps({"test": []}))
            val200.write_text(json.dumps({"test": []}))
            args = argparse.Namespace(
                group_dir=root / "group",
                source_checkpoint=Path("/source/checkpoint.pth"),
                subset_manifest=subset,
                val80_json=val80,
                val200_json=val200,
                run1_reference_json=None,
                failure="synthetic arm failure",
                current_stage="smoke",
            )
            summary = build_summary(args)
            output = root / "progress.json"
            atomic_write_json(output, {"old": True})
            atomic_write_json(output, summary)
            loaded = json.loads(output.read_text())
        self.assertEqual(loaded["status"], "failed")
        self.assertTrue(all(arm["selection"] is None for arm in loaded["arms"].values()))

    def test_run_index_preserves_r01_and_r02_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            r01 = experiment / "runs" / "exp012_category_specialists_20260801T000000Z"
            r02 = experiment / "runs" / "exp012_category_specialists_r02_replay_ratio_20260802T000000Z"
            (r01 / "reports").mkdir(parents=True)
            (r02 / "reports").mkdir(parents=True)
            (r01 / "run_group_manifest.json").write_text(json.dumps({"experiment": "012"}))
            (r02 / "run_group_manifest.json").write_text(
                json.dumps({"profile": "r02_2d_diffuse_replay_ratio"})
            )
            (r01 / ".selection_ready").touch()
            index = build_index(experiment)
        profiles = {record["profile"]: record for record in index["runs"]}
        self.assertTrue(profiles["r01_core_replay50"]["selection_ready"])
        self.assertEqual(profiles["r02_2d_diffuse_replay_ratio"]["status"], "created")
        self.assertEqual(index["experiment_status"], "active")


if __name__ == "__main__":
    unittest.main()
