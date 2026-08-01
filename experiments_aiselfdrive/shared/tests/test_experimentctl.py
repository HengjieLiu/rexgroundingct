from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
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
        controller = experimentctl.ControlPlane(self.workspace)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        claims = yaml.safe_load(self.claims_path.read_text(encoding="utf-8"))
        artifacts = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = controller._obligation_projection(
            claims, artifacts, pinned_at="2026-07-31T00:00:00Z"
        )
        self.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        (self.experiment / "results" / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "experiment_id": self.experiment_id,
                    "stage_id": "implement",
                    "plan_sha256": state["plan_sha256"],
                    "plan_revision": state["plan_revision"],
                    "state_revision": state["revision"],
                    "state_status": state["status"],
                    "sentinel": True,
                    "status": "ready",
                    "completed_at": state["updated_at"],
                    "summary": "Fixture initialization sentinel.",
                    "checks": [],
                    "artifacts": [],
                    "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
                    "error": None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        controller._render_views_unchecked()

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
                    "id": f"control_plane_{claim_type}",
                    "type": claim_type,
                    "statement": f"The fixture declares its {claim_type} contract.",
                    "verdict": "pending",
                    "gate_refs": ["evaluation.gates.valid"],
                    "evidence_refs": [],
                    "decided_at": None,
                    "rationale": None,
                }
                for claim_type in (
                    "feasibility",
                    "mechanistic",
                    "causal",
                    "performance",
                    "safety",
                )
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
        claims = yaml.safe_load(self.claims_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = experimentctl.ControlPlane(
            self.workspace
        )._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        self.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        event = json.loads(self.events_path.read_text(encoding="utf-8"))
        event["payload"]["plan_sha256"] = plan_sha
        event["hash"] = event_hash(event)
        self.events_path.write_text(canonical_json(event) + "\n", encoding="utf-8")
        result_path = self.experiment / "results" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["plan_sha256"] = plan_sha
        result_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )


class ExperimentCtlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = WorkspaceFixture(Path(self.temporary.name))
        self.controller = experimentctl.ControlPlane(self.fixture.workspace)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _configure_single_promotion_candidate(
        self, *, gpu_cost: float = 1.0, dice_delta: float = 0.01
    ) -> None:
        t4 = self.fixture.plan["stages"][1]
        t4.update(
            {
                "kind": "internal_replication",
                "required": False,
                "outputs": [
                    "treatment_checkpoint",
                    "treatment_metrics",
                    "control_checkpoint",
                    "control_metrics",
                ],
                "matched_arms": {
                    "treatment": {
                        "id": "treatment",
                        "checkpoint_output_id": "treatment_checkpoint",
                        "metrics_output_id": "treatment_metrics",
                    },
                    "control": {
                        "id": "control",
                        "checkpoint_output_id": "control_checkpoint",
                        "metrics_output_id": "control_metrics",
                    },
                    "matched_hash_fields": [
                        "parent_checkpoint_sha256",
                        "data_sha256",
                        "schedule_sha256",
                        "update_count",
                    ],
                },
            }
        )
        finalize = copy.deepcopy(t4)
        finalize.update(
            {
                "id": "finalize",
                "kind": "closeout",
                "needs": ["closeout"],
                "required": True,
                "outputs": [],
            }
        )
        finalize.pop("matched_arms")
        self.fixture.plan["stages"].append(finalize)
        self.fixture.refresh_plan_hashes()
        portfolio_path = self.fixture.workspace / "portfolio.yaml"
        portfolio = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
        portfolio["promotion_policy"] = {"max_t4_interventions": 1}
        portfolio["experiments"][0]["promotion"] = {
            "candidate": True,
            "t3_stage_id": "implement",
            "t4_stage_id": "closeout",
            "matched_control_stage_id": "closeout",
            "treatment_arm_id": "treatment",
            "matched_control_arm_id": "control",
        }
        portfolio_path.write_text(
            yaml.safe_dump(portfolio, sort_keys=False), encoding="utf-8"
        )
        t3_path = self.fixture.experiment / "results" / "t3.json"
        t3 = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2026-07-31T01:00:00Z",
            "summary": "T3 promotion evidence.",
            "checks": [],
            "artifacts": [],
            "metrics": {
                "resources": {"gpu_hours": gpu_cost, "cpu_hours": 1.0, "storage_gib": 1.0},
                "promotion": {
                    "eligible": True,
                    "all_safety_and_noninferiority_gates_pass": True,
                    "paired_val80_dice_delta": dice_delta,
                    "val80_off_location_fp_reduction": 0.2,
                    "measured_gpu_hours": gpu_cost,
                },
            },
            "error": None,
        }
        t3_path.write_text(json.dumps(t3), encoding="utf-8")
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"].append(
            {
                "id": "promotion_record",
                "path": str(self.fixture.external / "promotions" / f"{self.fixture.experiment_id}.json"),
                "role": "input",
                "ownership": "external_read_only",
                "status": "planned",
                "required": False,
                "sha256": None,
                "size_bytes": None,
                "producer_stage_id": None,
                "controller_producer": "experimentctl promote",
                "retention": "durable",
            }
        )
        self.fixture.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["status"] = "blocked"
        state["current_stage_id"] = None
        state["blockers"] = [{"code": "awaiting_portfolio_promotion", "stage_id": "implement"}]
        state["stages"][0].update(
            {"status": "passed", "attempts": 1, "completed_at": "2026-07-31T01:00:00Z", "evidence_refs": [self.fixture.rel(t3_path)]}
        )
        state["stages"][1]["status"] = "pending"
        state["stages"].append(
            {"id": "finalize", "status": "pending", "attempts": 0, "started_at": None, "completed_at": None, "last_error": None, "evidence_refs": []}
        )
        state["budget_consumed"] = {"gpu_hours": gpu_cost, "cpu_hours": 1.0, "storage_gib": 1.0}
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = self.controller._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        self.controller._render_views_unchecked()

    def _clone_promotion_candidate(self, identifier: str, gpu_cost: float) -> None:
        source_id = self.fixture.experiment_id
        destination = self.fixture.workspace / "experiments" / identifier
        shutil.copytree(self.fixture.experiment, destination)
        for path in destination.rglob("*"):
            if path.is_file():
                path.write_text(
                    path.read_text(encoding="utf-8").replace(source_id, identifier),
                    encoding="utf-8",
                )
        plan_path = destination / "experiment.yaml"
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        plan["id"] = identifier
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        state_path = destination / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiment_id"] = identifier
        state["plan_sha256"] = plan_hash
        state["budget_consumed"]["gpu_hours"] = gpu_cost
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        claims_path = destination / "claims.yaml"
        claims = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
        claims["experiment_id"] = identifier
        claims_path.write_text(yaml.safe_dump(claims, sort_keys=False), encoding="utf-8")
        manifest_path = destination / "artifacts" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_id"] = identifier
        manifest["artifacts"][0]["sha256"] = plan_hash
        manifest["artifacts"][0]["size_bytes"] = plan_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        t3_path = destination / "results" / "t3.json"
        t3 = json.loads(t3_path.read_text(encoding="utf-8"))
        t3["experiment_id"] = identifier
        t3["metrics"]["resources"]["gpu_hours"] = gpu_cost
        t3["metrics"]["promotion"]["measured_gpu_hours"] = gpu_cost
        t3_path.write_text(json.dumps(t3), encoding="utf-8")
        state["obligation_projection"] = self.controller._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        event_path = destination / "events.jsonl"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["experiment_id"] = identifier
        event["event_id"] = f"{identifier}-000001"
        event["payload"]["plan_sha256"] = plan_hash
        event["hash"] = event_hash(event)
        event_path.write_text(canonical_json(event) + "\n", encoding="utf-8")
        portfolio_path = self.fixture.workspace / "portfolio.yaml"
        portfolio = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
        cloned_entry = copy.deepcopy(portfolio["experiments"][0])
        cloned_entry.update(
            {
                "id": identifier,
                "path": str(destination.relative_to(self.fixture.repo_root)),
                "plan_path": str(plan_path.relative_to(self.fixture.repo_root)),
                "state_path": str(state_path.relative_to(self.fixture.repo_root)),
                "priority": 2,
            }
        )
        portfolio["experiments"].append(cloned_entry)
        portfolio_path.write_text(yaml.safe_dump(portfolio, sort_keys=False), encoding="utf-8")
        self.controller._render_views_unchecked()

    def test_validate_next_claim_and_agent_work_order(self) -> None:
        report = self.controller.validate_all()
        self.assertTrue(report["valid"], report["errors"])
        selected = self.controller.next_experiment()
        self.assertTrue(selected["available"])
        self.assertEqual(selected["id"], self.fixture.experiment_id)

        claim = self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.assertTrue(claim["claimed"])
        work_order = self.controller.run_stage(self.fixture.experiment_id, "test-agent")
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
        self.controller.run_stage(self.fixture.experiment_id, "test-agent")
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
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
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        result = self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path, "test-agent"
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
        self.controller.run_stage(self.fixture.experiment_id, "test-agent")
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
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
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )

        result = self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path, "test-agent"
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
        self.controller.run_stage(self.fixture.experiment_id, "test-agent")
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
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Fixture implementation passed.",
            "checks": [
                {"id": "fixture_check", "status": "passed", "detail": "verified"}
            ],
            "artifacts": [artifact],
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
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
                self.fixture.experiment_id, self.fixture.evidence_path, "test-agent"
            )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "blocked")
        self.assertIsNone(state["executor"])

    def test_closeout_commits_before_releasing_lease(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "test-agent")
        self.controller.run_stage(self.fixture.experiment_id, "test-agent")
        self.fixture.output_path.write_text("verified output\n", encoding="utf-8")
        implementation_evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
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
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        self.fixture.evidence_path.write_text(
            json.dumps(implementation_evidence), encoding="utf-8"
        )
        self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path, "test-agent"
        )
        self.controller.run_stage(self.fixture.experiment_id, "test-agent")
        closeout_path = self.fixture.workspace / ".runtime" / "closeout.json"
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        for claim in claims["claims"]:
            claim.update(
                {
                    "verdict": "not_supported",
                    "evidence_refs": [self.fixture.rel(closeout_path)],
                    "decided_at": "2099-07-31T02:00:00Z",
                    "rationale": "The fixture records a valid scientific no-go.",
                }
            )
        self.fixture.claims_path.write_text(
            yaml.safe_dump(claims, sort_keys=False), encoding="utf-8"
        )
        closeout_path.parent.mkdir(parents=True, exist_ok=True)
        closeout = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "closeout",
            "status": "passed",
            "completed_at": "2099-07-31T02:00:00Z",
            "summary": "Fixture closed as a valid no-go.",
            "checks": [
                {
                    "id": "closeout_check",
                    "status": "passed",
                    "detail": "verified",
                }
            ],
            "artifacts": [],
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
            "outcome": "no_go",
        }
        closeout_path.write_text(json.dumps(closeout), encoding="utf-8")
        result = self.controller.complete_stage(
            self.fixture.experiment_id, closeout_path, "test-agent"
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
            result = controller.run_stage(fixture.experiment_id, "test-agent")
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
            "container_label_key": experimentctl.DOCKER_LEASE_LABEL_KEY,
            "container_label_value": "old",
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
        with mock.patch.object(
            self.controller, "_container_liveness", return_value=False
        ):
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
        ), mock.patch.object(controller, "_stop_labeled_containers", return_value=True):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id, "test-agent")
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
        with mock.patch.object(
            experimentctl.subprocess, "Popen", return_value=process
        ), mock.patch.object(controller, "_stop_labeled_containers", return_value=True):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id, "test-agent")

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
            self.controller.run_stage(self.fixture.experiment_id, "test-agent")

    def test_agent_ownership_and_duplicate_run_are_rejected(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "owner")
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "agent-id does not own"
        ):
            self.controller.run_stage(self.fixture.experiment_id, "intruder")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "duplicate execution"
        ):
            self.controller.run_stage(self.fixture.experiment_id, "owner")

    def test_next_reclaims_only_expired_provably_dead_lease(self) -> None:
        self.controller.claim(self.fixture.experiment_id, "owner")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        lease_path = self.controller.lease_path(self.fixture.experiment_id)
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        lease.update(
            {
                "hostname": socket.gethostname(),
                "pid": 99999999,
                "process_start_marker": "dead",
                "process_role": "agent_heartbeat",
                "heartbeat_at": "2000-01-01T00:00:00Z",
            }
        )
        lease_path.write_text(json.dumps(lease), encoding="utf-8")

        with mock.patch.object(
            self.controller, "_container_liveness", return_value=False
        ):
            selected = self.controller.next_experiment()

        self.assertFalse(selected["available"])
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "blocked")
        self.assertIsNone(state["executor"])
        self.assertFalse(lease_path.exists())
        events = [
            json.loads(line)
            for line in self.fixture.events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(events[-1]["event_type"], "stale_lease_reclaimed")

    def test_live_command_process_blocks_manual_completion(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(Path(self.temporary.name), command_stage=True)
        controller = experimentctl.ControlPlane(fixture.workspace)
        controller.claim(fixture.experiment_id, "owner")
        entry = controller.entry(fixture.experiment_id)
        _, plan_path, state_path, _, events_path, _ = controller.experiment_paths(entry)
        with controller.state_lock(fixture.experiment_id):
            plan = experimentctl._read_yaml(plan_path)
            state = experimentctl._read_json(state_path)
            stage_state = state["stages"][0]
            stage_state.update(
                {
                    "status": "running",
                    "attempts": 1,
                    "started_at": experimentctl._utc_now(),
                }
            )
            state["status"] = "running"
            state = controller._commit_state_event(
                state_path,
                events_path,
                state,
                event_type="stage_started",
                actor="owner",
                payload={"stage_id": "implement", "attempt": 1},
            )
            lease_id = state["executor"]["lease_id"]
            controller._refresh_lease(
                fixture.experiment_id,
                lease_id,
                process_pid=os.getpid(),
                process_role="command",
            )
        evidence = {
            "schema_version": "1.0",
            "experiment_id": fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Should not be accepted while command is live.",
            "checks": [{"id": "fixture_check", "status": "passed", "detail": "ok"}],
            "artifacts": [],
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        fixture.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "recorded process is live"
        ):
            controller.complete_stage(
                fixture.experiment_id, fixture.evidence_path, "owner"
            )

    def test_duplicate_yaml_key_fails_closed(self) -> None:
        with self.fixture.plan_path.open("a", encoding="utf-8") as handle:
            handle.write("idea: duplicate key must fail\n")
        report = self.controller.validate_all()
        self.assertTrue(
            any("duplicate mapping key" in issue["message"] for issue in report["errors"]),
            report["errors"],
        )

    def test_large_verified_artifact_hash_is_not_skipped(self) -> None:
        large_path = self.fixture.experiment / "artifacts" / "large.bin"
        with large_path.open("wb") as handle:
            handle.truncate(257 * 1024 * 1024)
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"].append(
            {
                "id": "large_verified",
                "path": self.fixture.rel(large_path),
                "role": "cache",
                "ownership": "workspace",
                "status": "verified",
                "required": False,
                "sha256": experimentctl._sha256_file(large_path),
                "size_bytes": large_path.stat().st_size,
                "producer_stage_id": None,
                "retention": "regenerable",
            }
        )
        self.fixture.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = self.controller._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        state["verified_artifact_ids"].append("large_verified")
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        with large_path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"X")
        report = self.controller.validate_all()
        self.assertIn("artifact_hash", {issue["code"] for issue in report["errors"]})

    def test_obligation_projection_detects_claim_deletion(self) -> None:
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        claims["claims"].pop()
        self.fixture.claims_path.write_text(
            yaml.safe_dump(claims, sort_keys=False), encoding="utf-8"
        )
        report = self.controller.validate_all()
        self.assertIn(
            "obligation_projection_drift",
            {issue["code"] for issue in report["errors"]},
        )

    def test_obligation_projection_pins_static_verified_hash(self) -> None:
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["sha256"] = "0" * 64
        self.fixture.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        report = self.controller.validate_all()
        self.assertIn(
            "obligation_projection_drift",
            {issue["code"] for issue in report["errors"]},
        )

    def test_planned_producerless_input_can_verify_without_repinning(self) -> None:
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        path = self.fixture.experiment / "artifacts" / "later_input.json"
        manifest["artifacts"].append(
            {
                "id": "later_input",
                "path": self.fixture.rel(path),
                "role": "input",
                "ownership": "workspace",
                "status": "planned",
                "required": False,
                "sha256": None,
                "size_bytes": None,
                "producer_stage_id": None,
                "retention": "durable",
            }
        )
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        projection = self.controller._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        path.write_text("{}\n", encoding="utf-8")
        manifest["artifacts"][-1].update(
            {
                "status": "verified",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
        self.assertTrue(self.controller._projection_matches(projection, claims, manifest))

    def test_pre_t4_val120_input_is_rejected(self) -> None:
        self.fixture.plan["stages"][0]["inputs"].append("val120_manifest")
        self.fixture.refresh_plan_hashes()
        report = self.controller.validate_all()
        self.assertIn(
            "confirmatory_leakage", {issue["code"] for issue in report["errors"]}
        )

    def test_status_and_passed_evidence_drift_are_detected(self) -> None:
        status_path = self.fixture.workspace / "STATUS.md"
        status_path.write_text("tampered\n", encoding="utf-8")
        report = self.controller.validate_all()
        self.assertIn("status_drift", {issue["code"] for issue in report["errors"]})
        self.controller._render_views_unchecked()

        self.controller.claim(self.fixture.experiment_id, "owner")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        self.fixture.output_path.write_text("stable\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Stable evidence before drift.",
            "checks": [{"id": "fixture_check", "status": "passed", "detail": "ok"}],
            "artifacts": [
                {
                    "id": "implementation_output",
                    "path": self.fixture.rel(self.fixture.output_path),
                    "sha256": hashlib.sha256(self.fixture.output_path.read_bytes()).hexdigest(),
                    "size_bytes": self.fixture.output_path.stat().st_size,
                }
            ],
            "metrics": {"resources": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        self.fixture.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        self.controller.complete_stage(
            self.fixture.experiment_id, self.fixture.evidence_path, "owner"
        )
        self.fixture.output_path.write_text("drifted\n", encoding="utf-8")
        report = self.controller.validate_all()
        self.assertIn(
            "passed_stage_evidence", {issue["code"] for issue in report["errors"]}
        )

    def test_retry_backoff_is_enforced_for_declared_class(self) -> None:
        retry = self.fixture.plan["stages"][0]["retry"]
        retry.update(
            {
                "max_attempts": 2,
                "retryable_classes": ["transient_io"],
                "backoff_seconds": [3600],
            }
        )
        self.fixture.refresh_plan_hashes()
        self.controller.claim(self.fixture.experiment_id, "owner")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        entry = self.controller.entry(self.fixture.experiment_id)
        _, plan_path, state_path, _, _, _ = self.controller.experiment_paths(entry)
        with self.controller.state_lock(self.fixture.experiment_id):
            plan = experimentctl._read_yaml(plan_path)
            state = experimentctl._read_json(state_path)
            self.controller._command_failure(
                entry,
                plan,
                state,
                plan["stages"][0],
                "transient_io",
                "temporary fixture error",
            )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "retry_wait")
        self.assertIsNotNone(state["stages"][0]["retry_not_before"])
        with self.assertRaisesRegex(experimentctl.ControlPlaneError, "backoff is active"):
            self.controller.run_stage(self.fixture.experiment_id, "owner")

    def test_single_retry_policy_class_is_executable_and_invalid_contract_fails(self) -> None:
        retry = self.fixture.plan["stages"][0]["retry"]
        retry.update(
            {"class": "transient_io", "max_attempts": 2, "backoff_seconds": [1]}
        )
        self.fixture.refresh_plan_hashes()
        self.controller.claim(self.fixture.experiment_id, "owner")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        entry = self.controller.entry(self.fixture.experiment_id)
        _, plan_path, state_path, _, _, _ = self.controller.experiment_paths(entry)
        with self.controller.state_lock(self.fixture.experiment_id):
            state = experimentctl._read_json(state_path)
            updated = self.controller._command_failure(
                entry,
                experimentctl._read_yaml(plan_path),
                state,
                self.fixture.plan["stages"][0],
                "transient_io",
                "retry the declared policy class",
            )
        self.assertEqual(updated["status"], "retry_wait")

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(Path(self.temporary.name))
        controller = experimentctl.ControlPlane(fixture.workspace)
        fixture.plan["stages"][0]["retry"] = {
            "class": "unknown_policy",
            "max_attempts": 2,
            "backoff_seconds": [-1, 2],
        }
        fixture.refresh_plan_hashes()
        report = controller.validate_all()
        self.assertIn("retry_contract", {issue["code"] for issue in report["errors"]})

    def test_retry_policy_allows_only_its_explicit_operational_classes(self) -> None:
        expected = {
            "resource_unavailable": "retry_wait",
            "timeout": "retry_wait",
            "completion_evidence": "blocked",
            "hash_mismatch": "blocked",
            "geometry_mismatch": "blocked",
        }
        for error_class, expected_status in expected.items():
            with self.subTest(error_class=error_class), tempfile.TemporaryDirectory() as root:
                fixture = WorkspaceFixture(Path(root))
                controller = experimentctl.ControlPlane(fixture.workspace)
                fixture.plan["stages"][0]["retry"] = {
                    "class": "resource_then_evaluation",
                    "max_attempts": 2,
                    "backoff_seconds": [0],
                    "resource_wait_timeout_seconds": 3600,
                }
                fixture.refresh_plan_hashes()
                entry = controller.entry(fixture.experiment_id)
                _, plan_path, state_path, _, _, _ = controller.experiment_paths(entry)
                plan = experimentctl._read_yaml(plan_path)
                state = experimentctl._read_json(state_path)
                stage_state = state["stages"][0]
                stage_state.update(
                    {
                        "status": "running",
                        "attempts": 1,
                        "started_at": experimentctl._utc_now(),
                    }
                )
                state["status"] = "running"
                updated = controller._command_failure(
                    entry,
                    plan,
                    state,
                    plan["stages"][0],
                    error_class,
                    "focused retry classification",
                )
                self.assertEqual(updated["status"], expected_status)

    def test_failed_attempt_charges_budget_and_exhaustion_refuses_next_command(self) -> None:
        self.fixture.plan["stages"][0]["resources"] = {"gpu_count": 1}
        self.fixture.plan["stages"][0]["retry"] = {
            "class": "resource_then_evaluation",
            "max_attempts": 2,
            "backoff_seconds": [0],
        }
        self.fixture.plan["budget"] = {"gpu_hours_max": 2.0}
        self.fixture.refresh_plan_hashes()
        entry = self.controller.entry(self.fixture.experiment_id)
        _, plan_path, state_path, _, _, _ = self.controller.experiment_paths(entry)
        plan = experimentctl._read_yaml(plan_path)
        state = experimentctl._read_json(state_path)
        started = experimentctl.dt.datetime.now(
            experimentctl.dt.timezone.utc
        ) - experimentctl.dt.timedelta(minutes=30)
        state["stages"][0].update(
            {
                "status": "running",
                "attempts": 1,
                "started_at": started.isoformat().replace("+00:00", "Z"),
            }
        )
        state["status"] = "running"
        updated = self.controller._command_failure(
            entry,
            plan,
            state,
            plan["stages"][0],
            "timeout",
            "charge supervised elapsed time",
        )
        charged = updated["budget_consumed"]
        self.assertGreaterEqual(charged["gpu_hours"], 0.49)
        self.assertGreaterEqual(charged["cpu_hours"], 0.49)
        events = [
            json.loads(line)
            for line in self.fixture.events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            events[-1]["payload"]["failed_attempt_resources"]["gpu_hours"],
            charged["gpu_hours"],
        )
        exhausted_state = copy.deepcopy(updated)
        exhausted_state["budget_consumed"]["gpu_hours"] = 2.0
        command_stage = copy.deepcopy(plan["stages"][0])
        command_stage["execution_mode"] = "command"
        command_stage["timeout_seconds"] = 3600
        with self.assertRaisesRegex(experimentctl.ControlPlaneError, "budget is exhausted"):
            self.controller._command_budget_timeout_seconds(
                plan, exhausted_state, command_stage
            )

    def test_structured_failed_command_evidence_drives_audited_retry(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(Path(self.temporary.name), command_stage=True)
        controller = experimentctl.ControlPlane(fixture.workspace)
        stage = fixture.plan["stages"][0]
        stage["completion"] = {"evidence_file": fixture.rel(fixture.evidence_path)}
        stage["retry"] = {
            "class": "resource_then_evaluation",
            "max_attempts": 2,
            "backoff_seconds": [0],
            "resource_wait_timeout_seconds": 3600,
        }
        fixture.plan["budget"] = {"gpu_hours_max": 2.0}
        fixture.refresh_plan_hashes()
        controller.claim(fixture.experiment_id, "owner")
        state = experimentctl._read_json(fixture.state_path)
        evidence = {
            "schema_version": "1.0",
            "experiment_id": fixture.experiment_id,
            "stage_id": "implement",
            "attempt": 1,
            "plan_sha256": state["plan_sha256"],
            "status": "failed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "GPU memory is temporarily unavailable.",
            "checks": [],
            "artifacts": [],
            "metrics": {
                "resources": {
                    "gpu_hours": 0.25,
                    "cpu_hours": 0.1,
                    "storage_gib": 1.0,
                }
            },
            "error": {
                "class": "resource_unavailable",
                "message": "GPU memory is temporarily unavailable.",
            },
        }
        fixture.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        completed = experimentctl.subprocess.CompletedProcess(
            ["fake-command"], 7, stdout="", stderr="temporary failure"
        )
        with mock.patch.object(
            controller, "_run_command_with_heartbeat", return_value=completed
        ), mock.patch.object(controller, "_stop_labeled_containers", return_value=True):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id, "owner")
        state = experimentctl._read_json(fixture.state_path)
        self.assertEqual(state["status"], "retry_wait")
        self.assertEqual(
            state["budget_consumed"],
            {"gpu_hours": 0.25, "cpu_hours": 0.1, "storage_gib": 1.0},
        )
        failure_refs = state["stages"][0]["failure_evidence_refs"]
        self.assertEqual(len(failure_refs), 1)
        archived = controller.resolve_declared_path(failure_refs[0])
        self.assertEqual(archived.read_bytes(), fixture.evidence_path.read_bytes())
        events = [
            json.loads(line)
            for line in fixture.events_path.read_text(encoding="utf-8").splitlines()
        ]
        payload = events[-1]["payload"]
        self.assertEqual(payload["error_class"], "resource_unavailable")
        self.assertEqual(payload["failure_evidence"]["path"], failure_refs[0])
        self.assertEqual(
            payload["failure_evidence"]["sha256"],
            hashlib.sha256(archived.read_bytes()).hexdigest(),
        )
        self.assertTrue(controller.validate_all()["valid"])

    def test_failed_evidence_identity_mismatch_cannot_upgrade_command_exit(self) -> None:
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        fixture = WorkspaceFixture(Path(self.temporary.name), command_stage=True)
        controller = experimentctl.ControlPlane(fixture.workspace)
        stage = fixture.plan["stages"][0]
        stage["completion"] = {"evidence_file": fixture.rel(fixture.evidence_path)}
        stage["retry"] = {
            "class": "resource_then_evaluation",
            "max_attempts": 2,
            "backoff_seconds": [0],
        }
        fixture.plan["budget"] = {"gpu_hours_max": 2.0}
        fixture.refresh_plan_hashes()
        controller.claim(fixture.experiment_id, "owner")
        state = experimentctl._read_json(fixture.state_path)
        evidence = {
            "schema_version": "1.0",
            "experiment_id": fixture.experiment_id,
            "stage_id": "implement",
            "attempt": 99,
            "plan_sha256": state["plan_sha256"],
            "status": "failed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Forged retry classification.",
            "checks": [],
            "artifacts": [],
            "metrics": {
                "resources": {
                    "gpu_hours": 0.0,
                    "cpu_hours": 0.0,
                    "storage_gib": 0.0,
                }
            },
            "error": {"class": "resource_unavailable", "message": "forged"},
        }
        fixture.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        completed = experimentctl.subprocess.CompletedProcess(
            ["fake-command"], 7, stdout="", stderr="failed"
        )
        with mock.patch.object(
            controller, "_run_command_with_heartbeat", return_value=completed
        ), mock.patch.object(controller, "_stop_labeled_containers", return_value=True):
            with self.assertRaises(experimentctl.ControlPlaneError):
                controller.run_stage(fixture.experiment_id, "owner")
        state = experimentctl._read_json(fixture.state_path)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["stages"][0]["last_error"]["class"], "command_exit")
        self.assertNotIn("failure_evidence_refs", state["stages"][0])

    def test_directory_tree_digest_changes_with_content(self) -> None:
        tree = self.fixture.experiment / "artifacts" / "tree"
        tree.mkdir()
        (tree / "a.txt").write_text("one\n", encoding="utf-8")
        first_hash, first_size = self.controller._tree_digest_and_size(tree)
        (tree / "a.txt").write_text("two\n", encoding="utf-8")
        second_hash, second_size = self.controller._tree_digest_and_size(tree)
        self.assertNotEqual(first_hash, second_hash)
        self.assertEqual(first_size, second_size)

    def test_control_plane_lock_detects_source_drift(self) -> None:
        tools_dir = self.fixture.workspace / "tools"
        tools_dir.mkdir()
        shutil.copy2(MODULE_PATH, tools_dir / "experimentctl.py")
        shutil.copytree(
            WORKSPACE_ROOT / "templates" / "experiment",
            self.fixture.workspace / "templates" / "experiment",
        )
        portfolio_path = self.fixture.workspace / "portfolio.yaml"
        portfolio = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
        portfolio.update(
            {
                "registry_revision": 2,
                "previous_registry_sha256": "1" * 64,
                "control_plane_lock_ref": self.fixture.rel(
                    self.fixture.workspace
                    / "shared"
                    / "manifests"
                    / "control_plane_lock.json"
                ),
            }
        )
        portfolio_path.write_text(
            yaml.safe_dump(portfolio, sort_keys=False), encoding="utf-8"
        )
        self.controller._render_views_unchecked()
        result = self.controller.lock_control_plane("lock-agent")
        self.assertTrue(result["locked"])
        self.assertTrue(self.controller.validate_all()["valid"])
        original_claims = self.fixture.claims_path.read_bytes()
        original_state = self.fixture.state_path.read_bytes()
        claims = yaml.safe_load(original_claims)
        claims["claims"][0]["statement"] = "A coordinated weakening attempt."
        self.fixture.claims_path.write_text(
            yaml.safe_dump(claims, sort_keys=False), encoding="utf-8"
        )
        state = json.loads(original_state)
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = self.controller._obligation_projection(
            claims,
            manifest,
            pinned_at=state["obligation_projection"]["pinned_at"],
        )
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        weakened = self.controller.validate_all()
        self.assertIn(
            "control_plane_lock_obligation_drift",
            {issue["code"] for issue in weakened["errors"]},
        )
        self.fixture.claims_path.write_bytes(original_claims)
        self.fixture.state_path.write_bytes(original_state)
        template = self.fixture.workspace / "templates" / "experiment" / "claims.yaml"
        with template.open("a", encoding="utf-8") as handle:
            handle.write("# drift\n")
        report = self.controller.validate_all()
        self.assertIn(
            "control_plane_lock_drift", {issue["code"] for issue in report["errors"]}
        )

    def test_optional_gpu_zero_hours_is_allowed_but_ordinary_gpu_is_not(self) -> None:
        entry = self.controller.entry(self.fixture.experiment_id)
        stage = copy.deepcopy(self.fixture.plan["stages"][0])
        stage["resources"] = {
            "gpu_count": 0,
            "gpu_optional": True,
            "gpu_count_max": 1,
        }
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "CPU cache hit used no GPU.",
            "checks": [{"id": "fixture_check", "status": "passed", "detail": "ok"}],
            "artifacts": [],
            "metrics": {
                "resources": {"gpu_hours": 0.0, "cpu_hours": 0.1, "storage_gib": 0.0}
            },
            "error": None,
        }
        stage["outputs"] = []
        self.controller._verify_completion_evidence(
            entry, stage, evidence, self.fixture.evidence_path
        )
        stage["resources"] = {"gpu_count": 1}
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "positive gpu_hours"
        ):
            self.controller._verify_completion_evidence(
                entry, stage, evidence, self.fixture.evidence_path
            )

    def test_auxiliary_inference_export_class_is_bounded_and_not_t0(self) -> None:
        image_id = "sha256:" + "a" * 64
        profile = {
            "gpu_access": {
                "training_image": "fixture:gpu",
                "training_image_id": image_id,
            },
            "gpus": {"count": 4},
            "auxiliary_compute_classes": {
                "bounded_inference_export": {
                    "training_prohibited": True,
                    "gpu_optional": True,
                    "gpu_count_max": 1,
                    "gpu_hours_total_max": 2,
                    "minimum_free_gib": 32,
                }
            },
        }
        stage = copy.deepcopy(self.fixture.plan["stages"][0])
        stage.update({"id": "gpu_preflight", "kind": "gpu_preflight"})
        stage["resources"] = {
            "resource_class": "bounded_inference_export",
            "gpu_count": 0,
            "gpu_optional": True,
            "gpu_count_max": 1,
            "gpu_hours_max": 2,
            "gpu_memory_free_gib_min": 32,
            "docker_image": "fixture:gpu",
            "docker_image_id": image_id,
        }
        plan = {"resources": {}, "stages": [stage]}
        issues = self.controller._validate_hardware_semantics(
            plan, profile, self.fixture.plan_path
        )
        self.assertFalse(
            [issue for issue in issues if issue.code == "hardware_resource_class"],
            issues,
        )
        stage["resources"]["gpu_hours_max"] = 2.1
        issues = self.controller._validate_hardware_semantics(
            plan, profile, self.fixture.plan_path
        )
        self.assertIn("hardware_resource_class", {issue.code for issue in issues})

    def test_container_liveness_blocks_reclaim_and_query_failure_is_unknown(self) -> None:
        lease = {
            "hostname": socket.gethostname(),
            "lease_id": "lease-1",
            "container_label_key": experimentctl.DOCKER_LEASE_LABEL_KEY,
            "container_label_value": "lease-1",
        }
        live = mock.Mock(returncode=0, stdout="container-id\n")
        with mock.patch.object(experimentctl.subprocess, "run", return_value=live) as run:
            self.assertTrue(self.controller._container_liveness(lease))
        self.assertIn(
            "label=rexgroundingct.aiselfdrive.lease=lease-1",
            run.call_args.args[0],
        )
        failed = mock.Mock(returncode=1, stdout="")
        with mock.patch.object(experimentctl.subprocess, "run", return_value=failed):
            self.assertIsNone(self.controller._container_liveness(lease))

    def test_duplicate_resolved_artifact_path_is_rejected(self) -> None:
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(manifest["artifacts"][1])
        duplicate["id"] = "duplicate_output_alias"
        manifest["artifacts"].append(duplicate)
        self.fixture.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        claims = yaml.safe_load(self.fixture.claims_path.read_text(encoding="utf-8"))
        state["obligation_projection"] = self.controller._obligation_projection(
            claims, manifest, pinned_at="2026-07-31T00:00:00Z"
        )
        self.fixture.state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        report = self.controller.validate_all()
        self.assertIn(
            "duplicate_artifact_path", {issue["code"] for issue in report["errors"]}
        )

    def test_declared_canonical_result_is_not_overwritten_by_other_evidence(self) -> None:
        result_path = self.fixture.experiment / "results" / "result.json"
        self.fixture.plan["stages"][0]["outputs"].append(self.fixture.rel(result_path))
        self.fixture.plan["stages"][0]["allowed_write_paths"].append(
            self.fixture.rel(result_path)
        )
        self.fixture.refresh_plan_hashes()
        report = self.controller.validate_all()
        self.assertIn(
            "canonical_result_contract", {issue["code"] for issue in report["errors"]}
        )
        self.fixture.plan["stages"][0]["completion"] = {
            "evidence_file": self.fixture.rel(result_path)
        }
        self.fixture.refresh_plan_hashes()
        self.controller.claim(self.fixture.experiment_id, "owner")
        self.controller.run_stage(self.fixture.experiment_id, "owner")
        scientific = b'{"scientific_payload":"must survive"}\n'
        result_path.write_bytes(scientific)
        self.fixture.output_path.write_text("output\n", encoding="utf-8")
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Differing evidence must not replace result.",
            "checks": [{"id": "fixture_check", "status": "passed", "detail": "ok"}],
            "artifacts": [],
            "metrics": {
                "resources": {"gpu_hours": 0.0, "cpu_hours": 0.1, "storage_gib": 0.0}
            },
            "error": None,
        }
        self.fixture.evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        with self.assertRaisesRegex(
            experimentctl.ControlPlaneError, "cannot be overwritten"
        ):
            self.controller.complete_stage(
                self.fixture.experiment_id, self.fixture.evidence_path, "owner"
            )
        self.assertEqual(result_path.read_bytes(), scientific)

    def test_resource_accounting_uses_storage_high_water_and_stage_cap(self) -> None:
        state = {"budget_consumed": {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}}
        plan = {"budget": {"gpu_hours_max": 10, "storage_gib_max": 10}}
        for gpu, cpu, storage in ((1.0, 2.0, 4.0), (2.0, 3.0, 3.0)):
            self.controller._ingest_resource_metrics(
                plan,
                state,
                {"metrics": {"resources": {"gpu_hours": gpu, "cpu_hours": cpu, "storage_gib": storage}}},
            )
        self.assertEqual(state["budget_consumed"], {"gpu_hours": 3.0, "cpu_hours": 5.0, "storage_gib": 4.0})
        entry = self.controller.entry(self.fixture.experiment_id)
        stage = copy.deepcopy(self.fixture.plan["stages"][0])
        stage["outputs"] = []
        stage["resources"] = {"gpu_count": 2, "gpu_hours_per_arm_max": 1.0}
        evidence = {
            "schema_version": "1.0",
            "experiment_id": self.fixture.experiment_id,
            "stage_id": "implement",
            "status": "passed",
            "completed_at": "2099-07-31T01:00:00Z",
            "summary": "Over cap.",
            "checks": [{"id": "fixture_check", "status": "passed", "detail": "ok"}],
            "artifacts": [],
            "metrics": {"resources": {"gpu_hours": 2.1, "cpu_hours": 0.0, "storage_gib": 0.0}},
            "error": None,
        }
        with self.assertRaisesRegex(experimentctl.ControlPlaneError, "budget exceeded"):
            self.controller._verify_completion_evidence(
                entry, stage, evidence, self.fixture.evidence_path
            )

    def test_git_hygiene_rejects_tracked_runtime_and_heavy_files(self) -> None:
        listed = (
            b"experiments_aiselfdrive/.runtime/locks/x.lock\0"
            b"experiments_aiselfdrive/experiments/asd_000_fixture/runtime\0"
            b"experiments_aiselfdrive/experiments/asd_000_fixture/results/model.ckpt\0"
        )
        completed = mock.Mock(returncode=0, stdout=listed, stderr=b"")
        with mock.patch.object(experimentctl.subprocess, "run", return_value=completed):
            issues = self.controller._validate_git_hygiene()
        self.assertEqual(len(issues), 3)
        self.assertTrue(all(issue.code == "git_hygiene" for issue in issues))

    def test_promotion_ranks_by_cumulative_cost_and_bounds_metrics(self) -> None:
        self._configure_single_promotion_candidate(gpu_cost=5.0, dice_delta=0.01)
        self._clone_promotion_candidate("asd_001_fixture", gpu_cost=1.0)
        result = self.controller.promote("portfolio-agent")
        self.assertEqual(result["selected_ids"], ["asd_001_fixture"])
        for entry in self.controller.load_portfolio()["experiments"]:
            _, _, state_path, claims_path, _, artifacts_path = self.controller.experiment_paths(entry)
            promoted_state = experimentctl._read_json(state_path)
            promoted_claims = experimentctl._read_yaml(claims_path)
            promoted_artifacts = experimentctl._read_json(artifacts_path)
            self.assertTrue(
                self.controller._projection_matches(
                    promoted_state["obligation_projection"],
                    promoted_claims,
                    promoted_artifacts,
                )
            )
        first_state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        first_evidence = experimentctl._read_json(
            self.fixture.experiment / "results" / "t3.json"
        )
        first_evidence["metrics"]["promotion"]["paired_val80_dice_delta"] = 1.1
        (self.fixture.experiment / "results" / "t3.json").write_text(
            json.dumps(first_evidence), encoding="utf-8"
        )
        with self.assertRaisesRegex(experimentctl.ControlPlaneError, "within \[-1, 1\]"):
            self.controller._promotion_candidate_row(
                self.controller.entry(self.fixture.experiment_id), first_state
            )

    def test_promotion_mid_commit_failure_rolls_back_every_candidate(self) -> None:
        self._configure_single_promotion_candidate(gpu_cost=2.0)
        self._clone_promotion_candidate("asd_001_fixture", gpu_cost=1.0)
        portfolio = self.controller.load_portfolio()
        tracked: list[Path] = [self.fixture.workspace / "STATUS.md"]
        for entry in portfolio["experiments"]:
            root, _, state_path, _, events_path, artifacts_path = self.controller.experiment_paths(entry)
            tracked.extend(
                [state_path, events_path, artifacts_path, root / "results" / "result.json", root / "README.md"]
            )
        before = {path: path.read_bytes() for path in tracked}
        original = self.controller._commit_state_event
        calls = 0

        def fail_second(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise experimentctl.ControlPlaneError("injected promotion failure")
            return original(*args, **kwargs)

        with mock.patch.object(self.controller, "_commit_state_event", side_effect=fail_second):
            with self.assertRaisesRegex(experimentctl.ControlPlaneError, "injected"):
                self.controller.promote("portfolio-agent")
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
        promotion_root = self.fixture.external / "promotions"
        self.assertFalse(any(promotion_root.glob("*.json")))
        retried = self.controller.promote("portfolio-agent")
        self.assertTrue(retried["promoted"])

    def test_promotion_integrity_rejects_forged_plan_access_and_control(self) -> None:
        self._configure_single_promotion_candidate(gpu_cost=1.0)
        self.controller.promote("portfolio-agent")
        entry = self.controller.entry(self.fixture.experiment_id)
        state = experimentctl._read_json(self.fixture.state_path)
        _, events = self.controller._validate_events(
            self.fixture.events_path, self.fixture.experiment_id
        )
        record_path = self.fixture.external / "promotions" / f"{self.fixture.experiment_id}.json"
        original = experimentctl._read_json(record_path)
        for field, value, expected_code in (
            ("plan_sha256", "f" * 64, "promotion_record_plan"),
            ("val120_access_authorized", False, "promotion_record_access"),
            ("matched_control_required", False, "promotion_record_control"),
        ):
            forged = copy.deepcopy(original)
            forged[field] = value
            forged.pop("record_sha256")
            forged["record_sha256"] = hashlib.sha256(
                canonical_json(forged).encode("utf-8")
            ).hexdigest()
            record_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
            issues = self.controller._validate_promotion_integrity(
                entry, state, self.controller.load_portfolio(), events
            )
            self.assertIn(expected_code, {issue.code for issue in issues})
        record_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

    def test_adopt_current_replan_preserves_hash_lineage(self) -> None:
        previous = (
            self.fixture.workspace
            / ".runtime"
            / "previous_package"
            / "experiment.yaml"
        )
        previous.parent.mkdir(parents=True, exist_ok=True)
        previous.write_bytes(self.fixture.plan_path.read_bytes())
        previous_plan_bytes = previous.read_bytes()
        shutil.copy2(self.fixture.claims_path, previous.parent / "claims.yaml")
        previous_claims_bytes = (previous.parent / "claims.yaml").read_bytes()
        (previous.parent / "artifacts").mkdir()
        shutil.copy2(
            self.fixture.manifest_path,
            previous.parent / "artifacts" / "manifest.json",
        )
        previous_artifacts_bytes = (
            previous.parent / "artifacts" / "manifest.json"
        ).read_bytes()
        self.fixture.plan["plan_revision"] = 2
        self.fixture.plan["idea"] = "Exercise an audited second plan revision."
        self.fixture.plan_path.write_text(
            yaml.safe_dump(self.fixture.plan, sort_keys=False), encoding="utf-8"
        )
        new_hash = hashlib.sha256(self.fixture.plan_path.read_bytes()).hexdigest()
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["sha256"] = new_hash
        manifest["artifacts"][0]["size_bytes"] = self.fixture.plan_path.stat().st_size
        self.fixture.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        legacy_state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        legacy_state.pop("obligation_projection", None)
        legacy_state["budget_consumed"] = {
            "gpu_hours": 5.0,
            "cpu_hours": 7.0,
            "storage_gib": 3.0,
        }
        self.fixture.state_path.write_text(
            json.dumps(legacy_state, indent=2) + "\n", encoding="utf-8"
        )
        previous_state_bytes = self.fixture.state_path.read_bytes()
        previous_result_bytes = (
            self.fixture.experiment / "results" / "result.json"
        ).read_bytes()

        result = self.controller.replan(
            self.fixture.experiment_id,
            "planner",
            "Exercise audited plan revision semantics.",
            f"file:{previous}",
            adopt_current=True,
        )

        self.assertTrue(result["replanned"])
        self.assertEqual(result["new_plan_revision"], 2)
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["plan_sha256"], new_hash)
        self.assertEqual(
            state["budget_consumed"],
            {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0},
        )
        self.assertEqual(state["verified_artifact_ids"], ["experiment_plan"])
        history = state["plan_history"][-1]
        archive_bytes = {
            "archived_plan_path": previous_plan_bytes,
            "archived_state_path": previous_state_bytes,
            "archived_result_path": previous_result_bytes,
            "archived_claims_path": previous_claims_bytes,
            "archived_artifacts_path": previous_artifacts_bytes,
        }
        for path_key, expected_bytes in archive_bytes.items():
            archived_path = self.controller.resolve_declared_path(history[path_key])
            self.assertEqual(archived_path.read_bytes(), expected_bytes)
        self.assertEqual(
            history["previous_budget_consumed"],
            {"gpu_hours": 5.0, "cpu_hours": 7.0, "storage_gib": 3.0},
        )
        self.assertEqual(
            history["carried_budget_consumed"],
            {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0},
        )
        events = [
            json.loads(line)
            for line in self.fixture.events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(events[-1]["event_type"], "plan_revised")
        for field in (
            "archived_plan_sha256",
            "archived_plan_size_bytes",
            "archived_state_sha256",
            "archived_state_size_bytes",
            "archived_result_sha256",
            "archived_result_size_bytes",
            "previous_claims_sha256",
            "previous_claims_size_bytes",
            "previous_artifacts_sha256",
            "previous_artifacts_size_bytes",
            "previous_budget_consumed",
            "carried_budget_consumed",
        ):
            self.assertEqual(events[-1]["payload"][field], history[field])
        self.assertTrue(self.controller.validate_all()["valid"])

        original_events_bytes = self.fixture.events_path.read_bytes()
        events[-1]["payload"]["archived_state_size_bytes"] += 1
        events[-1]["hash"] = event_hash(events[-1])
        self.fixture.events_path.write_text(
            "\n".join(canonical_json(event) for event in events) + "\n",
            encoding="utf-8",
        )
        report = self.controller.validate_all()
        self.assertIn(
            "plan_history_event", {issue["code"] for issue in report["errors"]}
        )
        self.fixture.events_path.write_bytes(original_events_bytes)

        archived_plan = self.controller.resolve_declared_path(
            history["archived_plan_path"]
        )
        archived_plan.write_bytes(previous_plan_bytes + b"tampered\n")
        report = self.controller.validate_all()
        self.assertIn(
            "plan_history_archive_hash", {issue["code"] for issue in report["errors"]}
        )
        archived_plan.write_bytes(previous_plan_bytes)
        archived_claims = self.controller.resolve_declared_path(
            history["archived_claims_path"]
        )
        archived_claims.unlink()
        report = self.controller.validate_all()
        self.assertIn(
            "plan_history_archive_missing",
            {issue["code"] for issue in report["errors"]},
        )


if __name__ == "__main__":
    unittest.main()
