#!/usr/bin/env python3
"""Tests for the training-dynamics metric loader and summaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_training_dynamics import (
    finite_stats,
    load_arm_history,
    rolling_mean,
    trim_history,
)


def update(index: int, loss: float, **extra: float) -> dict[str, float | int]:
    return {"global_update": index, "loss": loss, **extra}


class TrainingDynamicsTest(unittest.TestCase):
    def write_metrics(
        self,
        arm_dir: Path,
        name: str,
        updates: list[dict[str, float | int]],
    ) -> Path:
        path = arm_dir / "reports" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"updates": updates}) + "\n")
        return path

    def load(self, arm_dir: Path):
        return load_arm_history(
            experiment="test_experiment",
            profile="test_profile",
            arm="test_arm",
            label="Test arm",
            arm_dir=arm_dir,
        )

    def test_merges_segments_and_identical_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "group" / "arm"
            third = update(3, 0.3, component=0.2)
            self.write_metrics(
                arm_dir,
                "training_metrics_segment_epoch001.json",
                [update(1, 0.5), update(2, 0.4), third],
            )
            self.write_metrics(
                arm_dir,
                "training_metrics_recovery_epoch002.json",
                [third, update(4, 0.2), update(5, 0.1)],
            )
            self.write_metrics(
                arm_dir,
                "training_metrics.json",
                [update(4, 0.2), update(5, 0.1)],
            )
            history = self.load(arm_dir)
            self.assertEqual(history.last_update, 5)
            self.assertEqual(history.global_updates.tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(len(history.source_files), 3)

    def test_rejects_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "group" / "arm"
            self.write_metrics(
                arm_dir,
                "training_metrics_segment_epoch001.json",
                [update(1, 0.5), update(2, 0.4)],
            )
            self.write_metrics(
                arm_dir,
                "training_metrics.json",
                [update(2, 0.9)],
            )
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                self.load(arm_dir)

    def test_rejects_noncontiguous_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "group" / "arm"
            self.write_metrics(
                arm_dir,
                "training_metrics.json",
                [update(1, 0.5), update(3, 0.3)],
            )
            with self.assertRaisesRegex(ValueError, "not a contiguous prefix"):
                self.load(arm_dir)

    def test_missing_optional_fields_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "group" / "arm"
            self.write_metrics(
                arm_dir,
                "training_metrics.json",
                [
                    update(1, 0.5),
                    update(2, float("nan")),
                    update(3, float("inf")),
                    update(4, 0.25),
                ],
            )
            history = self.load(arm_dir)
            missing = history.series("missing_component")
            self.assertTrue(np.isnan(missing).all())
            stats = finite_stats(history.series("loss"))
            self.assertEqual(stats["finite_observations"], 2)
            self.assertEqual(stats["nonfinite_observations"], 2)
            self.assertAlmostEqual(stats["first100_mean"], 0.375)

    def test_trim_history_stops_at_completed_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "group" / "arm"
            self.write_metrics(
                arm_dir,
                "training_metrics.json",
                [update(index, 1.0 / index) for index in range(1, 7)],
            )
            trimmed = trim_history(self.load(arm_dir), 4)
            self.assertEqual(trimmed.last_update, 4)
            self.assertEqual(len(trimmed.updates), 4)

    def test_rolling_mean_ignores_nonfinite_values(self) -> None:
        values = np.asarray([1.0, np.nan, 3.0, 5.0], dtype=np.float64)
        result = rolling_mean(values, window=3)
        np.testing.assert_allclose(result, [1.0, 1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
