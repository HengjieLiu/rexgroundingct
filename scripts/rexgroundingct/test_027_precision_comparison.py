"""CPU tests for the authorized sequential precision comparison and launch gates."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp027_common import ARMS, DEFAULT_CONFIG, atomic_json, lock, read_json, sha256
import run_027_fp32_comparison as comparison


def configs(root):
    fp16 = read_json(comparison.DEFAULT_CONFIG)
    fp16["experiment_dir"] = str(root)
    fp32 = copy.deepcopy(fp16)
    fp32["experiment_dir"] = str(root / "precision_fp32")
    fp32["training"]["amp"] = False
    return fp16, fp32


def completed_fixture(root):
    atomic_json(root / "status.json", {"status": "awaiting_schedule_decision"})
    records = [{"key": f"case::{i}"} for i in range(69)]
    atomic_json(root / "prepared.json", {"val": records})
    timing = {"status": "awaiting_schedule_decision", "arms": {}}
    for arm in ARMS:
        folder = root / "benchmark" / arm
        history = [{"update": i, "loss": .4, "grad_norm": .2, "patch_dice": .5,
                    "residual_mean_abs": .1, "amp_overflow_retries": 0} for i in range(1, 101)]
        atomic_json(folder / "training.json", {"updates": history, "initial_weights_sha256": "common"})
        timing["arms"][arm] = {"training": {"updates_completed": 100, "started_at_update": 0}}
        checkpoint = folder / "checkpoints/update_0000100.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint fixture")
        atomic_json(folder / "evaluations/update_0000100/summary.json",
                    {"findings": records, "metrics": {"full": {"findings": 69, "dice": .4, "hit_rate": .8}},
                     "checkpoint_sha256": sha256(checkpoint)})
    atomic_json(root / "timing_report.json", timing)


class PrecisionComparisonTests(unittest.TestCase):
    def test_configuration_changes_only_precision_and_output(self):
        a, b = configs(Path("/tmp/fixture"))
        comparison.validate_pair(a, b)
        b["optimizer"]["lr"] = .002
        with self.assertRaisesRegex(ValueError, "differ only"):
            comparison.validate_pair(a, b)
        b = copy.deepcopy(a)
        with self.assertRaisesRegex(ValueError, "AMP-disabled"):
            comparison.validate_pair(a, b)

    def test_pending_and_failed_prerequisite_never_counts_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(comparison.completion(root))
            for status in ("training", "evaluating", "launching"):
                atomic_json(root / "status.json", {"status": status})
                self.assertFalse(comparison.completion(root))
            atomic_json(root / "status.json", {"status": "failed", "error": "fixture failure"})
            with self.assertRaisesRegex(RuntimeError, "Benchmark failed"):
                comparison.completion(root)

    def test_complete_prerequisite_verifies_updates_findings_and_checkpoint(self):
        for corruption in (None, "updates", "duplicate_findings", "checkpoint", "initial_weights"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                completed_fixture(root)
                folder = root / "benchmark" / ARMS[0]
                if corruption in ("updates", "initial_weights"):
                    p = folder / "training.json"
                    value = read_json(p)
                    if corruption == "updates":
                        value["updates"].pop()
                    else:
                        value["initial_weights_sha256"] = "different"
                    atomic_json(p, value)
                elif corruption == "duplicate_findings":
                    p = folder / "evaluations/update_0000100/summary.json"
                    value = read_json(p)
                    value["findings"][-1] = value["findings"][0]
                    atomic_json(p, value)
                elif corruption == "checkpoint":
                    (folder / "checkpoints/update_0000100.pth").write_bytes(b"wrong checkpoint")
                if corruption:
                    with self.assertRaises(ValueError):
                        comparison.completion(root)
                else:
                    self.assertTrue(comparison.completion(root))

    def test_running_orchestrator_lock_prevents_gpu_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with lock(root / ".orchestrator.lock"):
                self.assertFalse(comparison.execution_released(root))
            self.assertTrue(comparison.execution_released(root))

    def test_docker_uses_fp32_policy_and_same_four_gpus(self):
        command = comparison.docker_command(Path("/tmp/config.json"), Path("/tmp/fp32"), "sha256:fixture")
        self.assertIn("NVIDIA_TF32_OVERRIDE=0", command)
        self.assertIn("EXP027_CONTAINER_IMAGE_ID=sha256:fixture", command)
        self.assertIn("--allow-gpu", command)
        self.assertEqual(command[-5:], ["--gpus", "0", "1", "2", "3"])

    def test_report_pending_partial_and_failures_in_fixed_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            followup = root / "precision_fp32"
            comparison.report(root, followup, {"status": "waiting_for_fp16"})
            value = read_json(root / "reports/precision_comparison.json")
            self.assertEqual([r["arm"] for r in value["rows"]], [arm for arm in ARMS for _ in range(2)])
            self.assertTrue(all(r["training_seconds"] is None for r in value["rows"]))
            atomic_json(followup / "benchmark" / ARMS[1] / "status.json", {"status": "failed", "error": "OOM fixture"})
            comparison.report(root, followup, {"status": "failed", "error": "Worker failure"})
            text = (root / "reports/precision_comparison.md").read_text()
            self.assertIn("OOM fixture", text)
            self.assertIn("pending", text)

    def test_dry_run_has_no_writes_or_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = configs(root)
            first, second = root / "fp16.json", root / "fp32.json"
            atomic_json(first, a)
            atomic_json(second, b)
            atomic_json(root / "run_manifest.json", {"image_id": "sha256:fixture"})
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            with patch.object(comparison, "DEFAULT_CONFIG", first), patch.object(comparison, "initialize") as init, \
                    patch.object(comparison.subprocess, "Popen") as launch, redirect_stdout(io.StringIO()):
                self.assertEqual(comparison.main(["--config", str(second)]), 0)
                init.assert_not_called()
                launch.assert_not_called()
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_executing_controller_waits_for_fp16_without_gpu_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = configs(root)
            first, second = root / "fp16.json", root / "fp32.json"
            atomic_json(first, a)
            atomic_json(second, b)
            atomic_json(root / "run_manifest.json", {"image_id": "sha256:fixture"})
            atomic_json(root / "status.json", {"status": "evaluating"})
            with patch.object(comparison, "DEFAULT_CONFIG", first), patch.object(comparison, "initialize"), \
                    patch("exp027_report.report_command"), patch.object(comparison.subprocess, "Popen") as launch, \
                    patch.object(comparison, "idle_gpus") as idle, \
                    patch.object(comparison.time, "sleep", side_effect=KeyboardInterrupt("fixture stop")):
                with self.assertRaises(KeyboardInterrupt):
                    comparison.main(["--config", str(second), "--execute"])
                launch.assert_not_called()
                idle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
