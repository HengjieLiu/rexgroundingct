#!/usr/bin/env python3
"""Resume frozen test300 inference without controlling the CPU search."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import sys

import fresh_cache as fc
import test300_cache as tc
import test300_runner as tr

POLICY = "leave_val200_cpu_search_uninterrupted"
SPEC = Path(__file__).with_name("test300_no_pause_execution_spec.md")


class NoPauseRunner(tr.Runner):
    def __init__(self, path, session_id):
        tc.require(re.fullmatch(r"[a-zA-Z0-9_-]+", session_id), "unsafe session ID")
        super().__init__(path)
        self.session = self.root / "recovery_sessions" / session_id
        self.recovery_sources = {
            str(p.resolve()): fc.sha256_file(p)
            for p in (Path(__file__), SPEC)
        }
        self.recorded = False

    def contention(self, records):
        # User accepts slowdown. Do not invoke the original pause state machine.
        return None

    def release_pause(self):
        # Also overrides the inherited startup recovery and finally paths.
        tc.require(self.paused is None,
                   "unresolved prior CPU pause; recovery will not control that container")

    def check(self):
        super().check()
        for path, expected in self.recovery_sources.items():
            tc.require(fc.sha256_file(Path(path)) == expected,
                       "recovery source drift: " + path)
        if not self.recorded:
            # The inherited run() calls check() while holding launch.lock,
            # before changing worker progress or launching containers.
            self.session.mkdir(parents=True, exist_ok=False)
            prior = self.session / "prior"
            prior.mkdir()
            for name in ("state.json", "supervisor.log", "cpu_pause.json", "launcher.json"):
                source = self.root / name
                if source.exists():
                    shutil.copy2(source, prior / name)
            if (self.root / "progress").exists():
                shutil.copytree(self.root / "progress", prior / "progress")
            for path in self.recovery_sources:
                shutil.copy2(path, self.session / Path(path).name)
            fc.atomic_write_json(self.session / "execution_manifest.json", {
                "created_at_utc": fc.utc_now(),
                "pid": os.getpid(),
                "job_id": self.job["job_id"],
                "job_spec_sha256": self.job["job_spec_sha256"],
                "inference_source_bundle_sha256": self.job["source_bundle"]["sha256"],
                "coordinator_sources": self.recovery_sources,
                "cpu_policy": POLICY,
                "supervisor_log": str(self.root / "logs" / (self.session.name + ".supervisor.log")),
                "argv": [sys.executable, *sys.argv],
                "cpu_state_before_launch": fc.read_json(tr.CPU_STATE),
                "prior_evidence": str(prior),
            })
            self.recorded = True

    def state(self, status, **kw):
        super().state(status, cpu_policy=POLICY,
                      recovery_session=str(self.session), **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    runner = NoPauseRunner(args.job, args.session_id)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(runner, "interrupted", True))
    runner.run()


if __name__ == "__main__":
    main()
