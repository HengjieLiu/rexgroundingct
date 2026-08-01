from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT / "experiments_aiselfdrive"
MODULE_PATH = WORKSPACE_ROOT / "tools" / "experimentctl.py"
SPEC = importlib.util.spec_from_file_location("aiselfdrive_experimentctl", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
experimentctl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experimentctl
SPEC.loader.exec_module(experimentctl)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def event_hash(event: dict[str, object]) -> str:
    body = dict(event)
    body.pop("hash", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class WorkspaceFixture:
    experiment_id = "asd_000_fixture"

    def __init__(self, parent: Path, *, command_stage: bool = False) -> None:
        self.repo_root = parent / "repo"
        self.workspace = self.repo_root / "experiments_aiselfdrive"
        self.external = parent / "runtime"
        self.experiment = self.workspace / "experiments" / self.experiment_id
        self.schemas = self.workspace / "schemas"
        self.schemas.mkdir(parents=True)
        for source in (WORKSPACE_ROOT / "schemas").glob("*.schema.json"):
            shutil.copy2(source, self.schemas / source.name)
        (self.workspace / "shared").mkdir()
        (self.workspace / "shared" / "hardware_profile.yaml").write_text(
            'schema_version: "1.0"\n', encoding="utf-8"
        )
        self.report = self.workspace / "report.md"
        self.report.write_text("# Historical report\n", encoding="utf-8")
        self.experiment.mkdir(parents=True)
        (self.experiment / "README.md").write_text(
            "# Fixture experiment\n", encoding="utf-8"
        )
        (self.experiment / "results").mkdir()
        (self.experiment / "artifacts").mkdir()
        self.plan_path = self.experiment / "experiment.yaml"
        self.state_path = self.experiment / "state.json"
        self.events_path = self.experiment / "events.jsonl"
        self.claims_path = self.experiment / "claims.yaml"
        self.manifest_path = self.experiment / "artifacts" / "manifest.json"
        self.output_path = self.experiment / "results" / "implementation.txt"
        self.evidence_path = self.experiment / "results" / "implement.json"
        self._write_plan(command_stage=command_stage)
        self._write_package_state()
        self._write_portfolio()

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.repo_root))

    def _write_plan(self, *, command_stage: bool) -> None:
        implementation: dict[str, object] = {
            "id": "implement",
            "kind": "implementation",
            "execution_mode": "agent",
            "needs": [],
            "required": True,
            "description": "Create the bounded fixture output.",
            "instructions": ["Create only the declared output.", "Record evidence."],
            "allowed_write_paths": [
                self.rel(self.output_path),
                self.rel(self.evidence_path),
            ],
            "inputs": [],
            "outputs": [self.rel(self.output_path)],
            "completion": {"evidence_file": self.rel(self.evidence_path)},
            "resources": {"gpu_count": 0},
            "timeout_seconds": 60,
            "retry": {"max_attempts": 1},
            "stop_conditions": ["write_scope_violation"],
            "acceptance_checks": ["fixture_check"],
        }
        if command_stage:
            implementation["execution_mode"] = "command"
            implementation.pop("instructions")
            implementation["outputs"] = []
            implementation["completion"] = {
                "predicate": "exit zero",
                "process_exit_only": True,
            }
            implementation["command"] = {
                "argv": ["fake-command", "argument with spaces"],
                "cwd": str(self.repo_root),
                "env": {"FIXTURE_ENV": "yes"},
            }
        closeout = {
            "id": "closeout",
            "kind": "closeout",
            "execution_mode": "agent",
            "needs": ["implement"],
            "required": True,
            "description": "Close the fixture.",
            "instructions": "Decide the predeclared fixture claim.",
            "allowed_write_paths": [
                self.rel(self.evidence_path),
                self.rel(self.claims_path),
                self.rel(self.manifest_path),
            ],
            "inputs": [],
            "outputs": [],
            "completion": {"predicate": "claims and artifacts decided"},
            "resources": {"gpu_count": 0},
            "timeout_seconds": 60,
            "retry": {"max_attempts": 1},
            "stop_conditions": [],
            "acceptance_checks": ["closeout_check"],
        }
        self.plan = {
            "schema_version": "1.0",
            "id": self.experiment_id,
            "title": "Control-plane fixture",
            "plan_revision": 1,
            "created_at": "2026-07-31T00:00:00Z",
            "updated_at": "2026-07-31T00:00:00Z",
            "priority": 1,
            "idea": "Exercise the control plane without science jobs.",
            "problem": "Mutable orchestration needs deterministic tests.",
            "novelty_vs_report": "This is a control-plane fixture only.",
            "hypotheses": {"primary": "State transitions remain consistent.", "null": None},
            "goals": ["Validate orchestration."],
            "non_goals": ["Run training."],
            "evidence_references": [],
            "lineage": {"availability": "verified_local"},
            "data": {"cohort": "synthetic_fixture"},
            "method": {"kind": "control_plane_test"},
            "resources": {"gpu_count": 0},
            "stages": [implementation, closeout],
            "evaluation": {"gates": {"valid": True}},
            "paths": {
                "workspace": self.rel(self.experiment),
                "runtime_root": str(self.external / self.experiment_id),
            },
            "budget": {"gpu_hours": 0},
            "retention": {"durable": ["events"]},
            "prohibited_directions": ["real_training"],
        }
        self.plan_path.write_text(
            yaml.safe_dump(self.plan, sort_keys=False), encoding="utf-8"
        )

    def _write_package_state(self) -> None:
        plan_sha = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.state = {
            "schema_version": "1.0",
            "experiment_id": self.experiment_id,
            "plan_sha256": plan_sha,
            "plan_revision": 1,
            "revision": 0,
            "last_event_seq": 1,
            "phase": "foundation",
            "status": "ready",
            "outcome": None,
            "created_at": "2026-07-31T00:00:00Z",
            "updated_at": "2026-07-31T00:00:00Z",
            "current_stage_id": "implement",
            "executor": None,
            "lease": None,
            "blockers": [],
            "stages": [
                {
                    "id": "implement",
                    "status": "ready",
                    "attempts": 0,
                    "started_at": None,
                    "completed_at": None,
                    "last_error": None,
                    "evidence_refs": [],
                },
                {
                    "id": "closeout",
                    "status": "pending",
                    "attempts": 0,
                    "started_at": None,
                    "completed_at": None,
                    "last_error": None,
                    "evidence_refs": [],
                },
            ],
            "artifacts_verified": False,
            "closeout_passed": False,
            "budget_consumed": {"gpu_hours": 0},
            "verified_artifact_ids": ["experiment_plan"],
        }
        self.state_path.write_text(
            json.dumps(self.state, indent=2) + "\n", encoding="utf-8"
        )
        claims = {
            "schema_version": "1.0",
            "experiment_id": self.experiment_id,
            "claims": [
                {
                    "id": "control_plane_valid",
                    "type": "feasibility",
                    "statement": "The fixture follows the control-plane contract.",
                    "verdict": "pending",
                    "gate_refs": ["evaluation.gates.valid"],
                    "evidence_refs": [],
                    "decided_at": None,
                    "rationale": None,
                }
            ],
        }
        self.claims_path.write_text(
            yaml.safe_dump(claims, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "schema_version": "1.0",
            "experiment_id": self.experiment_id,
            "artifacts": [
                {
                    "id": "experiment_plan",
                    "path": self.rel(self.plan_path),
                    "role": "manifest",
                    "ownership": "workspace",
                    "status": "verified",
                    "required": True,
                    "sha256": plan_sha,
                    "size_bytes": self.plan_path.stat().st_size,
                    "producer_stage_id": None,
                    "retention": "durable",
                },
                {
                    "id": "implementation_output",
                    "path": self.rel(self.output_path),
                    "role": "output",
                    "ownership": "workspace",
                    "status": "planned",
                    "required": False,
                    "sha256": None,
                    "size_bytes": None,
                    "producer_stage_id": "implement",
                    "retention": "durable",
                },
            ],
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        event = {
            "schema_version": "1.0",
            "sequence": 1,
            "event_id": f"{self.experiment_id}-000001",
            "experiment_id": self.experiment_id,
            "timestamp": "2026-07-31T00:00:00Z",
            "event_type": "experiment_initialized",
            "actor": "test_fixture",
            "previous_hash": None,
            "payload": {
                "status": "ready",
                "outcome": None,
                "revision": 0,
                "plan_sha256": plan_sha,
            },
        }
        event["hash"] = event_hash(event)
        self.events_path.write_text(canonical_json(event) + "\n", encoding="utf-8")

    def _write_portfolio(self) -> None:
        portfolio = {
            "schema_version": "1.0",
            "workspace_id": "control_plane_fixture",
            "created_at": "2026-07-31T00:00:00Z",
            "source_report": {
                "path": self.rel(self.report),
                "sha256": hashlib.sha256(self.report.read_bytes()).hexdigest(),
                "evidence_availability": "historical_only",
            },
            "allowed_write_roots": [self.rel(self.workspace), str(self.external)],
            "runtime_root": str(self.external),
            "hardware_profile_ref": self.rel(
                self.workspace / "shared" / "hardware_profile.yaml"
            ),
            "selection_policy": {
                "ordering": ["priority_ascending", "id_ascending"],
                "max_active_training_experiments": 1,
                "max_parallel_arms_within_experiment": 1,
                "allow_automatic_second_experiment": False,
                "unsatisfied_terminal_dependency_action": "finish_cancelled",
            },
            "lease_policy": {
                "heartbeat_seconds": 60,
                "ttl_seconds": 900,
                "stale_reclaim_requires_liveness_check": True,
                "stale_archive_path": self.rel(
                    self.workspace / ".runtime" / "stale_leases"
                ),
            },
            "retry_defaults": {},
            "promotion_policy": {},
            "experiments": [
                {
                    "id": self.experiment_id,
                    "path": self.rel(self.experiment),
                    "plan_path": self.rel(self.plan_path),
                    "state_path": self.rel(self.state_path),
                    "priority": 1,
                    "depends_on": [],
                }
            ],
        }
        (self.workspace / "portfolio.yaml").write_text(
            yaml.safe_dump(portfolio, sort_keys=False), encoding="utf-8"
        )

    def refresh_plan_hashes(self) -> None:
        self.plan_path.write_text(
            yaml.safe_dump(self.plan, sort_keys=False), encoding="utf-8"
        )
        plan_sha = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["plan_sha256"] = plan_sha
        self.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["sha256"] = plan_sha
        manifest["artifacts"][0]["size_bytes"] = self.plan_path.stat().st_size
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        event = json.loads(self.events_path.read_text(encoding="utf-8"))
        event["payload"]["plan_sha256"] = plan_sha
        event["hash"] = event_hash(event)
        self.events_path.write_text(canonical_json(event) + "\n", encoding="utf-8")


class ExperimentCtlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = WorkspaceFixture(Path(self.temporary.name))
        self.controller = experimentctl.ControlPlane(self.fixture.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validate_next_claim_and_agent_work_order(self) -> None:
        report = self.controller.validate_all()
        self.assertTrue(report["valid"], report["errors"])
        selected = self.controller.next_experiment()
        self.assertTrue(selected["available"])
        self.assertEqual(selected["id"], self.fixture.experiment_id)

        claim = self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.assertTrue(claim["claimed"])
        work_order = self.controller.run_stage(self.fixture.experiment_id)
        self.assertEqual(work_order["type"], "agent_work_order")
        self.assertEqual(work_order["stage_id"], "implement")
        self.assertEqual(work_order["attempt"], 1)
        self.assertEqual(
            work_order["allowed_write_paths"],
            [
                self.fixture.rel(self.fixture.output_path),
                self.fixture.rel(self.fixture.evidence_path),
            ],
        )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        events = self.fixture.events_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(state["revision"], 2)
        self.assertEqual(state["last_event_seq"], 3)
        self.assertEqual(len(events), 3)

    def test_complete_stage_verifies_output_and_advances(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.controller.run_stage(self.fixture.experiment_id)
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2026-07-31T01:00:00Z",
            "summary": "Fixture implementation passed.",
            "checks": [
                {"id": "fixture_check", "status": "passed", "detail": "verified"}
            ],
            "artifacts": [
                {
                    "id": "implementation_output",
                    "path": self.fixture.rel(self.fixture.output_path),
                    "sha256": hashlib.sha256(
                        self.fixture.output_path.read_bytes()
                    ).hexdigest(),
                    "size_bytes": self.fixture.output_path.stat().st_size,
                }
            ],
            "metrics": {},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        result = self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path
        )
        self.assertEqual(result["next_stage_id"], "closeout")
        self.assertEqual(result["status"], "claimed")
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        stages = {item["id"]: item for item in state["stages"]}
        self.assertEqual(stages["implement"]["status"], "passed")
        self.assertEqual(stages["closeout"]["status"], "ready")
        self.assertEqual(state["revision"], 3)
        self.assertEqual(state["last_event_seq"], 4)
        validation = self.controller.validate_all()
        self.assertTrue(validation["valid"], validation["errors"])

    def test_completion_evidence_may_be_its_own_declared_output(self) -> None:
        self.fixture.plan["stages"][0]["outputs"].append(
            self.fixture.rel(self.fixture.evidence_path)
        )
        self.fixture.refresh_plan_hashes()
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.controller.run_stage(self.fixture.experiment_id)
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2026-07-31T01:00:00Z",
            "summary": "Fixture implementation passed.",
            "checks": [
                {"id": "fixture_check", "status": "passed", "detail": "verified"}
            ],
            "artifacts": [
                {
                    "id": "implementation_output",
                    "path": self.fixture.rel(self.fixture.output_path),
                    "sha256": hashlib.sha256(
                        self.fixture.output_path.read_bytes()
                    ).hexdigest(),
                    "size_bytes": self.fixture.output_path.stat().st_size,
                }
            ],
            "metrics": {},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )

        result = self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path
        )

        self.assertEqual(result["status"], "claimed")
        self.assertEqual(result["next_stage_id"], "closeout")
        validation = self.controller.validate_all()
        self.assertTrue(validation["valid"], validation["errors"])

    def test_self_evidence_exemption_keeps_other_output_checks_strict(self) -> None:
        self.fixture.plan["stages"][0]["outputs"].append(
            self.fixture.rel(self.fixture.evidence_path)
        )
        self.fixture.refresh_plan_hashes()
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.controller.run_stage(self.fixture.experiment_id)
        self.fixture.output_path.write_text("original output\n", encoding="utf-8")
        artifact = {
            "id": "implementation_output",
            "path": self.fixture.rel(self.fixture.output_path),
            "sha256": hashlib.sha256(
                self.fixture.output_path.read_bytes()
            ).hexdigest(),
            "size_bytes": self.fixture.output_path.stat().st_size,
        }
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2026-07-31T01:00:00Z",
            "summary": "Fixture implementation passed.",
            "checks": [
                {"id": "fixture_check", "status": "passed", "detail": "verified"}
            ],
            "artifacts": [artifact],
            "metrics": {},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        self.fixture.output_path.write_text("tampered output\n", encoding="utf-8")

        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "evidence hash mismatch"
        ):
            self.controller.complete_stage(
                self.fixture.experiment_id, self.fixture.evidence_path
            )

        artifact["sha256"] = hashlib.sha256(
            self.fixture.output_path.read_bytes()
        ).hexdigest()
        artifact["size_bytes"] = self.fixture.output_path.stat().st_size + 1
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "evidence size mismatch"
        ):
            self.controller.complete_stage(
                self.fixture.experiment_id, self.fixture.evidence_path
            )

    def test_closeout_commits_before_releasing_lease(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.controller.run_stage(self.fixture.experiment_id)
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        implementation_evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2026-07-31T01:00:00Z",
            "summary": "Fixture implementation passed.",
            "checks": [
                {"id": "fixture_check", "status": "passed", "detail": "verified"}
            ],
            "artifacts": [
                {
                    "id": "implementation_output",
                    "path": self.fixture.rel(self.fixture.output_path),
                    "sha256": hashlib.sha256(
                        self.fixture.output_path.read_bytes()
                    ).hexdigest(),
                    "size_bytes": self.fixture.output_path.stat().st_size,
                }
            ],
            "metrics": {},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(implementation_evidence), encoding="utf-8"
        )
        self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path
        )
        self.controller.run_stage(self.fixture.experiment_id)
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        claim = claims["claims"][0]
        claim.update(
            {
                "verdict": "not_supported",
                "evidence_refs": ["results/closeout.json"],
                "decided_at": "2026-07-31T02:00:00Z",
                "rationale": "The fixture records a valid scientific no-go.",
            }
        )
        self.fixture.claims_path.write_text(
            yaml.safe_dump(claims, sort_keys=False), encoding="utf-8"
        )
        closeout_path = self.fixture.workspace / ".runtime" / "closeout.json"
        closeout_path.parent.mkdir(parents=True, exist_ok=True)
        closeout = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "closeout",
            "status": "passed",
            "completed_at": "2026-07-31T02:00:00Z",
            "summary": "Fixture closed as a valid no-go.",
            "checks": [
                {
                    "id": "closeout_check",
                    "status": "passed",
                    "detail": "verified",
                }
            ],
            "artifacts": [],
            "metrics": {},
            "error": None,
            "outcome": "no_go",
        }
        closeout_path.write_text(json.dumps(closeout), encoding="utf-8")
        result = self.controller.complete_stage(
            self.fixture.experiment_id, closeout_path
        )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["outcome"], "no_go")
        self.assertFalse(
            self.controller.lease_path(self.fixture.experiment_id).exists()
        )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "finished")
        self.assertIsNone(state["executor"])

    def test_command_stage_uses_exact_argv_without_shell(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(
            Path(self.temporary.name), command_stage=True
        )
        controller = experimentctl.ControlPlane(fixture.workspace)
        controller.claim(fixture.experiment_id, "test-agent")
        process = mock.Mock()
        process.pid = 99999999
        process.returncode = 0
        process.communicate.return_value = ("ok\n", "")
        with mock.patch.object(
            experimentctl.subprocess, "Popen", return_value=process
        ) as popen:
            result = controller.run_stage(fixture.experiment_id)
        self.assertEqual(result["next_stage_id"], "closeout")
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["fake-command", "argument with spaces"])
        self.assertIs(kwargs["shell"], False)

    def test_validation_rejects_placeholder_cycle_and_write_escape(self) -> None:
        self.fixture.plan["idea"] = "TODO replace this"
        self.fixture.plan["stages"][0]["needs"] = ["closeout"]
        self.fixture.plan["stages"][0]["allowed_write_paths"].append("/etc")
        self.fixture.refresh_plan_hashes()
        report = self.controller.validate_all()
        codes = {issue["code"] for issue in report["errors"]}
        self.assertIn("placeholder", codes)
        self.assertIn("stage_cycle", codes)
        self.assertIn("write_scope", codes)

    def test_event_tamper_is_detected(self) -> None:
        event = json.loads(self.fixture.events_path.read_text(encoding="utf-8"))
        event["payload"]["status"] = "blocked"
        self.fixture.events_path.write_text(
            canonical_json(event) + "\n", encoding="utf-8"
        )
        report = self.controller.validate_all()
        codes = {issue["code"] for issue in report["errors"]}
        self.assertIn("event_hash", codes)

    def test_stale_lease_reclaim_is_liveness_conservative(self) -> None:
        portfolio = self.controller.load_portfolio()
        lease_path = self.controller.lease_path(self.fixture.experiment_id)
        lease_path.parent.mkdir(parents=True)
        lease = {
            "schema_version": "1.0",
            "lease_id": "old",
            "experiment_id": self.fixture.experiment_id,
            "agent_id": "old-agent",
            "hostname": "unreachable-host",
            "pid": 123,
            "process_start_marker": "1",
            "claimed_at": "2000-01-01T00:00:00Z",
            "heartbeat_at": "2000-01-01T00:00:00Z",
            "heartbeat_seconds": 60,
            "ttl_seconds": 900,
        }
        lease_path.write_text(json.dumps(lease), encoding="utf-8")
        with self.assertRaises(experimentctl.ControlPlaneError):
            self.controller._archive_stale_lease(lease_path, lease, portfolio)
        lease["hostname"] = socket.gethostname()
        lease["pid"] = 99999999
        lease["process_start_marker"] = "missing"
        lease_path.write_text(json.dumps(lease), encoding="utf-8")
        self.controller._archive_stale_lease(lease_path, lease, portfolio)
        self.assertFalse(lease_path.exists())
        archive = (
            self.fixture.workspace / ".runtime" / "stale_leases"
        )
        self.assertEqual(len(list(archive.glob("*.json"))), 1)

    def test_render_is_deterministic(self) -> None:
        first = self.controller.render()
        status_path = self.fixture.workspace / "STATUS.md"
        status_one = status_path.read_text(encoding="utf-8")
        readme_one = (self.fixture.experiment / "README.md").read_text(
            encoding="utf-8"
        )
        second = self.controller.render()
        self.assertEqual(first, second)
        self.assertEqual(status_one, status_path.read_text(encoding="utf-8"))
        self.assertEqual(
            readme_one,
            (self.fixture.experiment / "README.md").read_text(encoding="utf-8"),
        )
        self.assertEqual(readme_one.count("experimentctl:state:start"), 1)

    def test_heartbeat_requires_owner_and_refreshes_lease(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        state_before = json.loads(
            self.fixture.state_path.read_text(encoding="utf-8")
        )
        events_before = self.fixture.events_path.read_text(encoding="utf-8")
        with self.assertRaises(experimentctl.ControlPlaneError):
            self.controller.heartbeat(self.fixture.experiment_id, "other-agent")
        result = self.controller.heartbeat(
            self.fixture.experiment_id, "test-agent"
        )
        self.assertTrue(result["heartbeat"])
        lease = json.loads(
            self.controller.lease_path(self.fixture.experiment_id).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lease["heartbeat_at"], result["heartbeat_at"])
        state_after = json.loads(
            self.fixture.state_path.read_text(encoding="utf-8")
        )
        self.assertEqual(state_after["revision"], state_before["revision"])
        self.assertEqual(
            self.fixture.events_path.read_text(encoding="utf-8"), events_before
        )

    def test_command_cannot_synthesize_declared_scientific_evidence(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(
            Path(self.temporary.name), command_stage=True
        )
        fixture.plan["stages"][0]["completion"] = {
            "evidence_file": fixture.rel(fixture.evidence_path)
        }
        fixture.refresh_plan_hashes()
        controller = experimentctl.ControlPlane(fixture.workspace)
        controller.claim(fixture.experiment_id, "test-agent")
        process = mock.Mock()
        process.pid = 99999999
        process.returncode = 0
        process.communicate.return_value = ("ok\n", "")
        with mock.patch.object(
            experimentctl.subprocess, "Popen", return_value=process
        ):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id)
        state = json.loads(fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["stages"][0]["status"], "failed")
        self.assertFalse(controller.lease_path(fixture.experiment_id).exists())
        self.assertEqual(state["last_event_seq"], len(
            fixture.events_path.read_text(encoding="utf-8").splitlines()
        ))

    def test_reopen_blocked_is_audited_and_resets_only_failed_stage(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(Path(self.temporary.name), command_stage=True)
        fixture.plan["stages"][0]["completion"] = {
            "evidence_file": fixture.rel(fixture.evidence_path)
        }
        fixture.refresh_plan_hashes()
        controller = experimentctl.ControlPlane(fixture.workspace)
        controller.claim(fixture.experiment_id, "test-agent")
        process = mock.Mock()
        process.pid = 99999999
        process.returncode = 0
        process.communicate.return_value = ("ok\n", "")
        with mock.patch.object(experimentctl.subprocess, "Popen", return_value=process):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id)

        blocked_state = json.loads(fixture.state_path.read_text(encoding="utf-8"))
        blocked_state["stages"][0]["evidence_refs"] = [
            "experiments_aiselfdrive/results/prior-attempt.json"
        ]
        fixture.state_path.write_text(
            json.dumps(blocked_state, indent=2) + "\n", encoding="utf-8"
        )
        prior_blockers = copy.deepcopy(blocked_state["blockers"])
        prior_closeout = copy.deepcopy(blocked_state["stages"][1])
        prior_event_count = len(
            fixture.events_path.read_text(encoding="utf-8").splitlines()
        )

        result = controller.reopen_blocked(
            fixture.experiment_id,
            "Completion evidence self-hash contract repaired.",
        )

        self.assertTrue(result["reopened"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["target_stage_id"], "implement")
        state = json.loads(fixture.state_path.read_text(encoding="utf-8"))
        stages = {item["id"]: item for item in state["stages"]}
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["current_stage_id"], "implement")
        self.assertEqual(state["blockers"], [])
        self.assertIsNone(state["outcome"])
        self.assertIsNone(state["executor"])
        self.assertIsNone(state["lease"])
        self.assertEqual(stages["implement"]["status"], "ready")
        self.assertEqual(stages["implement"]["attempts"], 0)
        self.assertIsNone(stages["implement"]["started_at"])
        self.assertIsNone(stages["implement"]["completed_at"])
        self.assertIsNone(stages["implement"]["last_error"])
        self.assertEqual(
            stages["implement"]["evidence_refs"],
            ["experiments_aiselfdrive/results/prior-attempt.json"],
        )
        self.assertEqual(stages["closeout"], prior_closeout)
        events = [
            json.loads(line)
            for line in fixture.events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(events), prior_event_count + 1)
        event = events[-1]
        self.assertEqual(event["event_type"], "blocked_experiment_reopened")
        self.assertEqual(event["payload"]["target_stage_id"], "implement")
        self.assertEqual(event["payload"]["prior_blockers"], prior_blockers)
        self.assertEqual(
            event["payload"]["reason"],
            "Completion evidence self-hash contract repaired.",
        )
        validation = controller.validate_all()
        self.assertTrue(validation["valid"], validation["errors"])

    def test_reopen_blocked_refuses_nonblocked_missing_or_ambiguous_stage(self) -> None:
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "requires blocked state"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["status"] = "blocked"
        state["current_stage_id"] = None
        state["blockers"] = [{"code": "control_plane_repair_required"}]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "exactly one failed/blocked stage"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

        state["stages"][0]["status"] = "failed"
        state["stages"][1]["status"] = "blocked"
        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "implement"},
            {"code": "stage_failed", "stage_id": "closeout"},
        ]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "exactly one failed/blocked stage"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

        state["stages"][1]["status"] = "pending"
        state["blockers"] = [{"code": "stage_failed", "stage_id": "implement"}]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        lease_path = self.controller.lease_path(self.fixture.experiment_id)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "existing lease file"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

    def test_reopen_blocked_refuses_unrelated_blockers_and_unmet_needs(self) -> None:
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["status"] = "blocked"
        state["current_stage_id"] = None
        state["stages"][0]["status"] = "failed"
        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "implement"},
            {"code": "manual_hold"},
        ]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        state_before = self.fixture.state_path.read_bytes()
        events_before = self.fixture.events_path.read_bytes()

        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "unrelated or unscoped blockers"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )
        self.assertEqual(state_before, self.fixture.state_path.read_bytes())
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "implement"},
            "operator hold",
        ]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "unrelated or unscoped blockers"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

        state["stages"][0]["status"] = "ready"
        state["stages"][1]["status"] = "failed"
        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "closeout"}
        ]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "DAG prerequisites pass"
        ):
            self.controller.reopen_blocked(
                self.fixture.experiment_id, "Control-plane repair."
            )

    def test_record_blocker_is_append_only_audited_and_atomic(self) -> None:
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["status"] = "blocked"
        state["current_stage_id"] = None
        state["stages"][0]["status"] = "failed"
        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "implement"}
        ]
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        prior_blockers = copy.deepcopy(state["blockers"])
        prior_stages = copy.deepcopy(state["stages"])
        prior_event_count = len(
            self.fixture.events_path.read_text(encoding="utf-8").splitlines()
        )

        result = self.controller.record_blocker(
            self.fixture.experiment_id,
            "confirmatory_leakage",
            "Confirmatory labels were opened during a prohibited audit.",
        )

        blocker = {
            "code": "confirmatory_leakage",
            "message": "Confirmatory labels were opened during a prohibited audit.",
        }
        self.assertTrue(result["recorded"])
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["current_stage_id"])
        self.assertEqual(result["blocker"], blocker)
        updated = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "blocked")
        self.assertIsNone(updated["current_stage_id"])
        self.assertEqual(updated["stages"], prior_stages)
        self.assertEqual(updated["blockers"], prior_blockers + [blocker])
        events = [
            json.loads(line)
            for line in self.fixture.events_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(len(events), prior_event_count + 1)
        event = events[-1]
        self.assertEqual(
            event["event_type"], "blocked_experiment_blocker_recorded"
        )
        self.assertEqual(event["payload"]["blocker"], blocker)
        self.assertEqual(event["payload"]["prior_blockers"], prior_blockers)
        self.assertIsNone(event["payload"]["current_stage_id"])
        validation = self.controller.validate_all()
        self.assertTrue(validation["valid"], validation["errors"])

        state_before = self.fixture.state_path.read_bytes()
        events_before = self.fixture.events_path.read_bytes()
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "exact duplicate blocker"
        ):
            self.controller.record_blocker(
                self.fixture.experiment_id,
                "confirmatory_leakage",
                "Confirmatory labels were opened during a prohibited audit.",
            )
        self.assertEqual(state_before, self.fixture.state_path.read_bytes())
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

        with mock.patch.object(
            self.controller,
            "_append_prebuilt_event",
            side_effect=OSError("injected append failure"),
        ):
            with self.assertRaises(OSError):
                self.controller.record_blocker(
                    self.fixture.experiment_id,
                    "evidence_contamination",
                    "A second blocker should roll back with the event failure.",
                )
        self.assertEqual(state_before, self.fixture.state_path.read_bytes())
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

    def test_record_blocker_refuses_invalid_nonblocked_and_leased_states(self) -> None:
        state_before = self.fixture.state_path.read_bytes()
        events_before = self.fixture.events_path.read_bytes()
        for code, message, expected in (
            ("Invalid-Code", "message", "blocker code must match"),
            ("valid_code", "   ", "blocker message must be nonempty"),
            ("valid_code", "message", "requires blocked state"),
        ):
            with self.subTest(code=code, message=message):
                with self.assertRaisesRegex(
                    experimentctl.ControlPlaneError, expected
                ):
                    self.controller.record_blocker(
                        self.fixture.experiment_id, code, message
                    )
                self.assertEqual(
                    state_before, self.fixture.state_path.read_bytes()
                )
                self.assertEqual(
                    events_before, self.fixture.events_path.read_bytes()
                )

        state = json.loads(state_before)
        state["status"] = "blocked"
        state["current_stage_id"] = None
        state["stages"][0]["status"] = "failed"
        state["blockers"] = [
            {"code": "stage_failed", "stage_id": "implement"}
        ]
        state["executor"] = {"agent_id": "unexpected-owner"}
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        leased_state_before = self.fixture.state_path.read_bytes()
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "executor lease"
        ):
            self.controller.record_blocker(
                self.fixture.experiment_id, "valid_code", "message"
            )
        self.assertEqual(
            leased_state_before, self.fixture.state_path.read_bytes()
        )
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

        state["executor"] = None
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        lease_path = self.controller.lease_path(self.fixture.experiment_id)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease_path.write_text("{}\n", encoding="utf-8")
        lease_file_state_before = self.fixture.state_path.read_bytes()
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "existing lease file"
        ):
            self.controller.record_blocker(
                self.fixture.experiment_id, "valid_code", "message"
            )
        self.assertEqual(
            lease_file_state_before, self.fixture.state_path.read_bytes()
        )
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

        lease_path.unlink()
        state["blockers"] = []
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        invalid_state_before = self.fixture.state_path.read_bytes()
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "canonical valid portfolio"
        ):
            self.controller.record_blocker(
                self.fixture.experiment_id, "valid_code", "message"
            )
        self.assertEqual(
            invalid_state_before, self.fixture.state_path.read_bytes()
        )
        self.assertEqual(events_before, self.fixture.events_path.read_bytes())

    def test_state_rolls_back_when_event_append_fails(self) -> None:
        before_state = self.fixture.state_path.read_bytes()
        before_events = self.fixture.events_path.read_bytes()
        with mock.patch.object(
            self.controller,
            "_append_prebuilt_event",
            side_effect=OSError("injected append failure"),
        ):
            with self.assertRaises(OSError):
                self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.assertEqual(before_state, self.fixture.state_path.read_bytes())
        self.assertEqual(before_events, self.fixture.events_path.read_bytes())
        self.assertFalse(
            self.controller.lease_path(self.fixture.experiment_id).exists()
        )

    def test_run_stage_rejects_plan_drift_after_claim(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        with self.fixture.plan_path.open("a", encoding="utf-8") as handle:
            handle.write("\n# drift\n")
        with self.assertRaises(experimentctl.ControlPlaneError):
            self.controller.run_stage(self.fixture.experiment_id)


if __name__ == "__main__":
    unittest.main()
