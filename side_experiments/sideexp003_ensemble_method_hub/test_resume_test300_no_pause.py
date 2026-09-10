"""Recovery must never pause or unpause the concurrent CPU search."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fresh_cache as fc
import resume_test300_no_pause as recovery
import test300_runner as tr


class NoPauseTests(unittest.TestCase):
    def runner(self):
        runner = recovery.NoPauseRunner.__new__(recovery.NoPauseRunner)
        runner.paused = None
        runner.paused_id = None
        return runner

    def test_sustained_slowdown_never_controls_cpu(self):
        runner = self.runner()
        records = {"candidate": {
            "status": "exporting",
            "recent_case_costs": [{"seconds": 100, "elements": 1}] * 10,
        }}
        with patch.object(tr, "command") as command, patch.object(tr, "docker_state") as inspect:
            for _ in range(10):
                self.assertIsNone(runner.contention(records))
            self.assertIsNone(runner.contention({}))
            runner.release_pause()
            command.assert_not_called()
            inspect.assert_not_called()

    def test_unresolved_old_pause_is_rejected_without_unpausing(self):
        runner = self.runner()
        runner.paused = "rex-resume-production"
        runner.paused_id = "immutable-id"
        with patch.object(tr, "command") as command, patch.object(tr, "docker_state") as inspect:
            with self.assertRaisesRegex(fc.FreshCacheError, "unresolved prior CPU pause"):
                runner.release_pause()
            command.assert_not_called()
            inspect.assert_not_called()

    def test_original_gpu_and_disk_gates_are_still_called(self):
        runner = self.runner()
        with patch.object(tr.Runner, "check", side_effect=fc.FreshCacheError("disk gate")) as gate:
            with self.assertRaisesRegex(fc.FreshCacheError, "disk gate"):
                runner.check()
            gate.assert_called_once_with()

    def test_original_worker_and_stop_paths_are_inherited(self):
        for name in ("run", "launch", "stop", "collect", "cpu_ready"):
            self.assertIs(getattr(recovery.NoPauseRunner, name), getattr(tr.Runner, name))

    def test_recovery_source_drift_is_rejected(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "runner.py"
            source.write_text("original")
            runner.recovery_sources = {str(source): fc.sha256_file(source)}
            runner.recorded = True
            source.write_text("changed")
            with patch.object(tr.Runner, "check"):
                with self.assertRaisesRegex(fc.FreshCacheError, "recovery source drift"):
                    runner.check()


if __name__ == "__main__":
    unittest.main()
