#!/usr/bin/env python3
"""Validated control plane for the ReXGroundingCT self-drive portfolio.

The scientific plans are immutable inputs.  This module is the only supported
writer for mutable experiment state, event ledgers, executor leases, and
generated status views.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_VERSION = "1.0"
EXPERIMENT_ID_RE = re.compile(r"^asd_[0-9]{3}_[a-z0-9_]+$")
STAGE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|__[A-Z0-9_]+__|<[^>\n]+>)",
    re.IGNORECASE,
)
ACTIVE_STATUSES = {"claimed", "running", "retry_wait", "stopping"}
TERMINAL_OUTCOMES = {
    "go",
    "no_go",
    "inconclusive",
    "failed",
    "cancelled",
    "invalid",
}
TERMINAL_STAGE_STATUSES = {"passed", "failed", "skipped"}
SCHEMA_FILES = {
    "portfolio": "portfolio.schema.json",
    "experiment": "experiment.schema.json",
    "state": "state.schema.json",
    "event": "event.schema.json",
    "claims": "claims.schema.json",
    "artifact_manifest": "artifact_manifest.schema.json",
    "result": "result.schema.json",
}


class ControlPlaneError(RuntimeError):
    """A user-facing control-plane failure."""


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _event_hash(event: Mapping[str, Any]) -> str:
    body = dict(event)
    body.pop("hash", None)
    return _sha256_bytes(_canonical_json(body).encode("utf-8"))


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ControlPlaneError(f"cannot read YAML {path}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlPlaneError(f"cannot read JSON {path}: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _walk_strings(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _walk_strings(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            yield from _walk_strings(child, child_prefix)


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _dag_cycle(nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    trail: list[str] = []

    def visit(node: str) -> list[str]:
        if node in visited:
            return []
        if node in visiting:
            start = trail.index(node)
            return trail[start:] + [node]
        visiting.add(node)
        trail.append(node)
        for dependency in dependencies.get(node, []):
            cycle = visit(dependency)
            if cycle:
                return cycle
        trail.pop()
        visiting.remove(node)
        visited.add(node)
        return []

    for candidate in nodes:
        cycle = visit(candidate)
        if cycle:
            return cycle
    return []


class ControlPlane:
    """Repository-local validated state machine."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        default_root = Path(__file__).resolve().parents[1]
        configured = workspace_root or os.environ.get(
            "REXGROUNDINGCT_AISELFDRIVE_ROOT"
        )
        self.workspace_root = Path(configured or default_root).resolve(strict=False)
        self.repo_root = self.workspace_root.parent.resolve(strict=False)
        self.portfolio_path = self.workspace_root / "portfolio.yaml"
        self.schemas_root = self.workspace_root / "schemas"
        self.runtime_control_root = self.workspace_root / ".runtime"
        self._schemas: dict[str, dict[str, Any]] = {}

    def resolve_declared_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        return (self.repo_root / candidate).resolve(strict=False)

    def relative_display(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def load_portfolio(self) -> dict[str, Any]:
        value = _read_yaml(self.portfolio_path)
        if not isinstance(value, dict):
            raise ControlPlaneError(f"{self.portfolio_path} must contain an object")
        return value

    def load_schemas(self) -> dict[str, dict[str, Any]]:
        if self._schemas:
            return self._schemas
        for name, filename in SCHEMA_FILES.items():
            path = self.schemas_root / filename
            schema = _read_json(path)
            if not isinstance(schema, dict):
                raise ControlPlaneError(f"schema is not an object: {path}")
            Draft202012Validator.check_schema(schema)
            self._schemas[name] = schema
        return self._schemas

    def _schema_issues(self, name: str, value: Any, path: Path) -> list[Issue]:
        schema = self.load_schemas()[name]
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        issues: list[Issue] = []
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        ):
            location = "/".join(str(part) for part in error.absolute_path)
            shown = self.relative_display(path)
            if location:
                shown = f"{shown}#/{location}"
            issues.append(Issue("schema", shown, error.message))
        return issues

    def entry(self, experiment_id: str) -> dict[str, Any]:
        if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
            raise ControlPlaneError(f"invalid experiment id: {experiment_id}")
        matches = [
            item
            for item in self.load_portfolio().get("experiments", [])
            if item.get("id") == experiment_id
        ]
        if len(matches) != 1:
            raise ControlPlaneError(
                f"portfolio must contain exactly one entry for {experiment_id}"
            )
        return matches[0]

    def experiment_paths(
        self, entry: Mapping[str, Any]
    ) -> tuple[Path, Path, Path, Path, Path, Path]:
        root = self.resolve_declared_path(str(entry["path"]))
        return (
            root,
            self.resolve_declared_path(str(entry["plan_path"])),
            self.resolve_declared_path(str(entry["state_path"])),
            root / "claims.yaml",
            root / "events.jsonl",
            root / "artifacts" / "manifest.json",
        )

    def load_package(
        self, entry: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        _, plan_path, state_path, claims_path, _, artifacts_path = (
            self.experiment_paths(entry)
        )
        plan = _read_yaml(plan_path)
        state = _read_json(state_path)
        claims = _read_yaml(claims_path)
        artifacts = _read_json(artifacts_path)
        for label, value in (
            ("plan", plan),
            ("state", state),
            ("claims", claims),
            ("artifact manifest", artifacts),
        ):
            if not isinstance(value, dict):
                raise ControlPlaneError(
                    f"{entry['id']} {label} must contain an object"
                )
        return plan, state, claims, artifacts

    def allowed_roots(self, portfolio: Mapping[str, Any]) -> list[Path]:
        return [
            self.resolve_declared_path(str(path))
            for path in portfolio.get("allowed_write_roots", [])
        ]

    def _dependency_state(
        self, dependency: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]
    ) -> tuple[bool, str]:
        dependency_id = str(dependency["id"])
        state = states.get(dependency_id)
        if state is None:
            return False, "dependency state is missing"
        if state.get("status") != "finished":
            return False, f"dependency status is {state.get('status')}"
        if state.get("outcome") not in dependency.get("satisfying_outcomes", []):
            return False, f"dependency outcome is {state.get('outcome')}"
        return True, "satisfied"

    def dependencies_satisfied(
        self, entry: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for dependency in entry.get("depends_on", []):
            if dependency.get("type") == "resource":
                continue
            satisfied, reason = self._dependency_state(dependency, states)
            if not satisfied and dependency.get("type") == "hard":
                failures.append(f"{dependency['id']}: {reason}")
        return not failures, failures

    def _validate_events(
        self, path: Path, experiment_id: str
    ) -> tuple[list[Issue], list[dict[str, Any]]]:
        issues: list[Issue] = []
        events: list[dict[str, Any]] = []
        if not path.exists():
            return [Issue("missing_file", self.relative_display(path), "missing event ledger")], []
        previous_hash: str | None = None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return [Issue("read_error", self.relative_display(path), str(exc))], []
        while lines and not lines[-1].strip():
            lines.pop()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                issues.append(
                    Issue(
                        "event_blank_line",
                        f"{self.relative_display(path)}:{line_number}",
                        "blank lines are not allowed in JSONL ledgers",
                    )
                )
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    Issue(
                        "event_json",
                        f"{self.relative_display(path)}:{line_number}",
                        str(exc),
                    )
                )
                continue
            issues.extend(self._schema_issues("event", event, path))
            if event.get("sequence") != line_number:
                issues.append(
                    Issue(
                        "event_sequence",
                        f"{self.relative_display(path)}:{line_number}",
                        f"expected sequence {line_number}, got {event.get('sequence')}",
                    )
                )
            if event.get("experiment_id") != experiment_id:
                issues.append(
                    Issue(
                        "identity_mismatch",
                        f"{self.relative_display(path)}:{line_number}",
                        "event experiment_id does not match package",
                    )
                )
            if event.get("previous_hash") != previous_hash:
                issues.append(
                    Issue(
                        "event_chain",
                        f"{self.relative_display(path)}:{line_number}",
                        "previous_hash does not match preceding event",
                    )
                )
            computed = _event_hash(event)
            if event.get("hash") != computed:
                issues.append(
                    Issue(
                        "event_hash",
                        f"{self.relative_display(path)}:{line_number}",
                        f"event hash mismatch; computed {computed}",
                    )
                )
            previous_hash = event.get("hash")
            events.append(event)
        if not events:
            issues.append(
                Issue(
                    "event_genesis",
                    self.relative_display(path),
                    "ledger must contain a genesis event",
                )
            )
        elif events[0].get("previous_hash") is not None:
            issues.append(
                Issue(
                    "event_genesis",
                    f"{self.relative_display(path)}:1",
                    "genesis previous_hash must be null",
                )
            )
        return issues, events

    def _validate_plan_semantics(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        portfolio: Mapping[str, Any],
        plan_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        experiment_id = str(entry["id"])
        shown = self.relative_display(plan_path)
        if plan.get("id") != experiment_id:
            issues.append(
                Issue("identity_mismatch", shown, "plan id does not match portfolio id")
            )
        if plan.get("priority") != entry.get("priority"):
            issues.append(
                Issue(
                    "priority_mismatch",
                    shown,
                    "plan priority does not match portfolio priority",
                )
            )
        if state.get("plan_revision") != plan.get("plan_revision"):
            issues.append(
                Issue(
                    "plan_revision",
                    shown,
                    "state plan_revision does not match immutable plan",
                )
            )
        actual_hash = _sha256_file(plan_path)
        if state.get("plan_sha256") != actual_hash:
            issues.append(
                Issue(
                    "plan_hash",
                    shown,
                    f"state plan_sha256 mismatch; computed {actual_hash}",
                )
            )
        stage_list = plan.get("stages", [])
        stage_ids = [str(stage.get("id")) for stage in stage_list]
        for repeated in sorted(_duplicates(stage_ids)):
            issues.append(
                Issue("duplicate_stage", shown, f"duplicate stage id {repeated}")
            )
        stage_set = set(stage_ids)
        dependencies: dict[str, list[str]] = {}
        for stage in stage_list:
            stage_id = str(stage.get("id"))
            dependencies[stage_id] = [str(item) for item in stage.get("needs", [])]
            for needed in dependencies[stage_id]:
                if needed not in stage_set:
                    issues.append(
                        Issue(
                            "unknown_stage_dependency",
                            shown,
                            f"stage {stage_id} needs unknown stage {needed}",
                        )
                    )
            if stage.get("execution_mode") == "command":
                argv = stage.get("command", {}).get("argv")
                if not isinstance(argv, list) or not argv or not all(
                    isinstance(arg, str) for arg in argv
                ):
                    issues.append(
                        Issue(
                            "command_argv",
                            shown,
                            f"stage {stage_id} must declare an exact argv array",
                        )
                    )
            if stage.get("execution_mode") == "agent":
                instructions = stage.get("instructions")
                valid_instructions = isinstance(instructions, str) and bool(
                    instructions.strip()
                )
                valid_instructions |= (
                    isinstance(instructions, list)
                    and bool(instructions)
                    and all(isinstance(item, str) and item.strip() for item in instructions)
                )
                if not valid_instructions or not stage.get("allowed_write_paths"):
                    issues.append(
                        Issue(
                            "agent_contract",
                            shown,
                            f"stage {stage_id} needs instructions and allowed_write_paths",
                        )
                    )
        cycle = _dag_cycle(stage_ids, dependencies)
        if cycle:
            issues.append(
                Issue(
                    "stage_cycle",
                    shown,
                    f"stage DAG cycle: {' -> '.join(cycle)}",
                )
            )

        global_roots = self.allowed_roots(portfolio)
        experiment_root = self.resolve_declared_path(str(entry["path"]))
        experiments_root = self.workspace_root / "experiments"
        plan_paths = plan.get("paths", {})
        declared_workspace = plan_paths.get(
            "workspace", plan_paths.get("experiment_root")
        )
        if declared_workspace is None or self.resolve_declared_path(
            str(declared_workspace)
        ) != experiment_root:
            issues.append(
                Issue(
                    "experiment_path",
                    shown,
                    "plan workspace/experiment_root must match portfolio path",
                )
            )
        declared_runtime = plan_paths.get(
            "runtime_root", plan_paths.get("runtime")
        )
        runtime_base = self.resolve_declared_path(str(portfolio["runtime_root"]))
        if declared_runtime is None:
            issues.append(
                Issue("runtime_path", shown, "plan must declare a runtime path")
            )
        else:
            runtime_path = self.resolve_declared_path(str(declared_runtime))
            expected_experiment_runtime = runtime_base / experiment_id
            if not _is_within(runtime_path, expected_experiment_runtime):
                issues.append(
                    Issue(
                        "runtime_path",
                        shown,
                        f"runtime must be within {expected_experiment_runtime}",
                    )
                )

        for stage in stage_list:
            stage_id = str(stage.get("id"))
            for raw_path in stage.get("allowed_write_paths", []):
                path = self.resolve_declared_path(str(raw_path))
                if not any(_is_within(path, root) for root in global_roots):
                    issues.append(
                        Issue(
                            "write_scope",
                            shown,
                            f"{stage_id} write path is outside portfolio roots: {raw_path}",
                        )
                    )
                if _is_within(path, experiments_root) and not _is_within(
                    path, experiment_root
                ):
                    issues.append(
                        Issue(
                            "cross_experiment_write",
                            shown,
                            f"{stage_id} may not write another experiment: {raw_path}",
                        )
                    )

        if state.get("status") == "ready":
            for location, text in _walk_strings(plan):
                match = PLACEHOLDER_RE.search(text)
                if match:
                    issues.append(
                        Issue(
                            "placeholder",
                            f"{shown}#/{location}",
                            f"ready plan contains unresolved token {match.group(0)!r}",
                        )
                    )
        return issues

    def _validate_state_semantics(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        claims: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        state_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        experiment_id = str(entry["id"])
        shown = self.relative_display(state_path)
        for label, value in (
            ("state", state),
            ("claims", claims),
            ("artifact manifest", artifacts),
        ):
            if value.get("experiment_id") != experiment_id:
                issues.append(
                    Issue(
                        "identity_mismatch",
                        shown,
                        f"{label} experiment_id does not match portfolio",
                    )
                )
        plan_stages = {str(stage["id"]): stage for stage in plan.get("stages", [])}
        state_stages = {
            str(stage.get("id")): stage for stage in state.get("stages", [])
        }
        if len(state_stages) != len(state.get("stages", [])):
            issues.append(Issue("duplicate_stage_state", shown, "stage state ids repeat"))
        if set(plan_stages) != set(state_stages):
            issues.append(
                Issue(
                    "stage_state_ids",
                    shown,
                    "state stage ids must exactly match plan stage ids",
                )
            )
        current = state.get("current_stage_id")
        if current is not None and current not in plan_stages:
            issues.append(
                Issue("current_stage", shown, f"unknown current stage {current}")
            )
        status = state.get("status")
        outcome = state.get("outcome")
        executor = state.get("executor") or state.get("lease")
        if (
            state.get("executor") is not None
            and state.get("lease") is not None
            and state.get("executor") != state.get("lease")
        ):
            issues.append(
                Issue(
                    "lease_alias",
                    shown,
                    "state executor and lease aliases must be identical",
                )
            )
        if status == "finished":
            if outcome not in TERMINAL_OUTCOMES:
                issues.append(
                    Issue("finished_outcome", shown, "finished state needs an outcome")
                )
            if executor is not None:
                issues.append(
                    Issue("finished_lease", shown, "finished state cannot own a lease")
                )
            if current is not None:
                issues.append(
                    Issue(
                        "finished_stage",
                        shown,
                        "finished state current_stage_id must be null",
                    )
                )
            if not state.get("artifacts_verified") or not state.get(
                "closeout_passed"
            ):
                issues.append(
                    Issue(
                        "finished_closeout",
                        shown,
                        "finished state requires verified artifacts and passed closeout",
                    )
                )
            for stage_id, stage_plan in plan_stages.items():
                stage_status = state_stages.get(stage_id, {}).get("status")
                if stage_plan.get("required") and stage_status not in {
                    "passed",
                    "skipped",
                }:
                    if outcome not in {"failed", "cancelled", "invalid"}:
                        issues.append(
                            Issue(
                                "unfinished_required_stage",
                                shown,
                                f"required stage {stage_id} is {stage_status}",
                            )
                        )
            for claim in claims.get("claims", []):
                if claim.get("verdict") == "pending":
                    issues.append(
                        Issue(
                            "pending_claim",
                            shown,
                            f"finished state has pending claim {claim.get('id')}",
                        )
                    )
                if claim.get("verdict") != "pending" and (
                    not claim.get("decided_at") or not claim.get("rationale")
                ):
                    issues.append(
                        Issue(
                            "claim_evidence",
                            shown,
                            f"decided claim {claim.get('id')} needs date and rationale",
                        )
                    )
        else:
            if outcome is not None:
                issues.append(
                    Issue(
                        "premature_outcome",
                        shown,
                        "non-finished state outcome must be null",
                    )
                )
        if status in ACTIVE_STATUSES and executor is None:
            issues.append(
                Issue("active_lease", shown, f"{status} state requires an executor lease")
            )
        if status not in ACTIVE_STATUSES and executor is not None:
            issues.append(
                Issue(
                    "inactive_lease",
                    shown,
                    f"{status} state cannot retain an executor lease",
                )
            )
        if status == "ready":
            if current is None:
                issues.append(
                    Issue("ready_stage", shown, "ready state needs current_stage_id")
                )
            elif state_stages.get(str(current), {}).get("status") != "ready":
                issues.append(
                    Issue(
                        "ready_stage",
                        shown,
                        "current ready stage must have stage status ready",
                    )
                )
        if status == "blocked" and not state.get("blockers"):
            issues.append(
                Issue("blocked_reason", shown, "blocked state needs a blocker")
            )
        if "last_event_seq" in state and state.get("last_event_seq") != len(events):
            issues.append(
                Issue(
                    "event_sequence",
                    shown,
                    "state last_event_seq does not match event ledger",
                )
            )
        if events:
            genesis_payload = events[0].get("payload", {})
            if genesis_payload.get("plan_sha256") != state.get("plan_sha256"):
                issues.append(
                    Issue(
                        "genesis_plan_hash",
                        shown,
                        "genesis event plan_sha256 does not match state",
                    )
                )
            latest_revision = events[-1].get("payload", {}).get("revision")
            if latest_revision is not None and latest_revision != state.get("revision"):
                issues.append(
                    Issue(
                        "event_revision",
                        shown,
                        "latest event revision does not match state revision",
                    )
                )

        claim_ids = [str(claim.get("id")) for claim in claims.get("claims", [])]
        for repeated in sorted(_duplicates(claim_ids)):
            issues.append(Issue("duplicate_claim", shown, f"duplicate claim {repeated}"))
        artifact_ids = [
            str(artifact.get("id")) for artifact in artifacts.get("artifacts", [])
        ]
        for repeated in sorted(_duplicates(artifact_ids)):
            issues.append(
                Issue("duplicate_artifact", shown, f"duplicate artifact {repeated}")
            )
        for artifact in artifacts.get("artifacts", []):
            producer = artifact.get("producer_stage_id")
            if producer is not None and producer not in plan_stages:
                issues.append(
                    Issue(
                        "artifact_producer",
                        shown,
                        f"artifact {artifact.get('id')} has unknown producer {producer}",
                    )
                )
        if "verified_artifact_ids" in state:
            manifest_verified_ids = {
                str(artifact.get("id"))
                for artifact in artifacts.get("artifacts", [])
                if artifact.get("status") == "verified"
            }
            state_verified_ids = set(state.get("verified_artifact_ids", []))
            if state_verified_ids != manifest_verified_ids:
                issues.append(
                    Issue(
                        "verified_artifact_ids",
                        shown,
                        "state verified_artifact_ids must exactly match the manifest",
                    )
                )
        return issues

    def _validate_artifact_semantics(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        portfolio: Mapping[str, Any],
        manifest_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        shown = self.relative_display(manifest_path)
        global_roots = self.allowed_roots(portfolio)
        experiment_root = self.resolve_declared_path(str(entry["path"]))
        runtime_root = (
            self.resolve_declared_path(str(portfolio["runtime_root"]))
            / str(entry["id"])
        )
        for artifact in artifacts.get("artifacts", []):
            artifact_id = artifact.get("id")
            path = self.resolve_declared_path(str(artifact.get("path", "")))
            ownership = artifact.get("ownership")
            if ownership == "workspace" and not _is_within(
                path, self.workspace_root
            ):
                issues.append(
                    Issue(
                        "artifact_scope",
                        shown,
                        f"workspace artifact {artifact_id} is outside workspace",
                    )
                )
            if ownership == "workspace":
                experiments_root = self.workspace_root / "experiments"
                if _is_within(path, experiments_root) and not _is_within(
                    path, experiment_root
                ):
                    issues.append(
                        Issue(
                            "artifact_scope",
                            shown,
                            f"workspace artifact {artifact_id} points to another experiment",
                        )
                    )
            if ownership == "external_runtime" and not _is_within(
                path, runtime_root
            ):
                issues.append(
                    Issue(
                        "artifact_scope",
                        shown,
                        f"runtime artifact {artifact_id} is outside experiment runtime",
                    )
                )
            if ownership != "external_read_only" and not any(
                _is_within(path, root) for root in global_roots
            ):
                issues.append(
                    Issue(
                        "artifact_scope",
                        shown,
                        f"artifact {artifact_id} is outside allowed roots",
                    )
                )
            status = artifact.get("status")
            if status in {"present", "verified"} and not path.exists():
                issues.append(
                    Issue(
                        "artifact_missing",
                        shown,
                        f"{status} artifact {artifact_id} does not exist: {path}",
                    )
                )
            if status == "verified":
                expected_hash = artifact.get("sha256")
                if not expected_hash:
                    issues.append(
                        Issue(
                            "artifact_hash",
                            shown,
                            f"verified artifact {artifact_id} has no SHA-256",
                        )
                    )
                elif path.is_file() and path.stat().st_size <= 256 * 1024 * 1024:
                    actual_hash = _sha256_file(path)
                    if actual_hash != expected_hash:
                        issues.append(
                            Issue(
                                "artifact_hash",
                                shown,
                                f"artifact {artifact_id} hash mismatch; computed {actual_hash}",
                            )
                        )
            if state.get("status") == "finished" and artifact.get(
                "required"
            ) and status != "verified":
                issues.append(
                    Issue(
                        "artifact_unverified",
                        shown,
                        f"required artifact {artifact_id} is not verified",
                    )
                )
        return issues

    def _validate_lease_consistency(
        self, experiment_id: str, state: Mapping[str, Any]
    ) -> list[Issue]:
        issues: list[Issue] = []
        executor = state.get("executor") or state.get("lease")
        lease_path = self.lease_path(experiment_id)
        if executor is None:
            if lease_path.exists():
                issues.append(
                    Issue(
                        "orphan_lease",
                        self.relative_display(lease_path),
                        "lease exists but state has no executor",
                    )
                )
            return issues
        if not lease_path.exists():
            issues.append(
                Issue(
                    "missing_lease",
                    self.relative_display(lease_path),
                    "active state lease file is missing",
                )
            )
            return issues
        try:
            lease = _read_json(lease_path)
        except ControlPlaneError as exc:
            return [
                Issue("lease_read", self.relative_display(lease_path), str(exc))
            ]
        if lease.get("lease_id") != executor.get("lease_id"):
            issues.append(
                Issue(
                    "lease_identity",
                    self.relative_display(lease_path),
                    "state and lease file lease_id differ",
                )
            )
        if lease.get("agent_id") != executor.get("agent_id"):
            issues.append(
                Issue(
                    "lease_identity",
                    self.relative_display(lease_path),
                    "state and lease file agent_id differ",
                )
            )
        return issues

    def validate_all(self) -> dict[str, Any]:
        issues: list[Issue] = []
        warnings: list[Issue] = []
        try:
            self.load_schemas()
            portfolio = self.load_portfolio()
        except (ControlPlaneError, Exception) as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return {
                "valid": False,
                "errors": [
                    Issue(
                        "control_plane_load", str(self.workspace_root), str(exc)
                    ).as_dict()
                ],
                "warnings": [],
                "experiments_checked": 0,
            }
        issues.extend(self._schema_issues("portfolio", portfolio, self.portfolio_path))

        report = portfolio.get("source_report", {})
        report_path = self.resolve_declared_path(str(report.get("path", "")))
        if not report_path.is_file():
            issues.append(
                Issue(
                    "source_report_missing",
                    self.relative_display(report_path),
                    "portfolio source report is missing",
                )
            )
        elif report.get("sha256") != _sha256_file(report_path):
            issues.append(
                Issue(
                    "source_report_hash",
                    self.relative_display(report_path),
                    "source report SHA-256 does not match portfolio",
                )
            )
        hardware_path = self.resolve_declared_path(
            str(portfolio.get("hardware_profile_ref", ""))
        )
        if not hardware_path.is_file():
            issues.append(
                Issue(
                    "hardware_profile_missing",
                    self.relative_display(hardware_path),
                    "hardware profile is missing",
                )
            )

        entries = portfolio.get("experiments", [])
        experiment_ids = [str(entry.get("id")) for entry in entries]
        for repeated in sorted(_duplicates(experiment_ids)):
            issues.append(
                Issue(
                    "duplicate_experiment",
                    self.relative_display(self.portfolio_path),
                    f"duplicate experiment id {repeated}",
                )
            )
        dependencies = {
            str(entry.get("id")): [
                str(dependency.get("id"))
                for dependency in entry.get("depends_on", [])
            ]
            for entry in entries
        }
        id_set = set(experiment_ids)
        for owner, needed_ids in dependencies.items():
            for needed in needed_ids:
                if needed not in id_set:
                    issues.append(
                        Issue(
                            "unknown_dependency",
                            self.relative_display(self.portfolio_path),
                            f"{owner} depends on unknown experiment {needed}",
                        )
                    )
        cycle = _dag_cycle(experiment_ids, dependencies)
        if cycle:
            issues.append(
                Issue(
                    "portfolio_cycle",
                    self.relative_display(self.portfolio_path),
                    f"experiment dependency cycle: {' -> '.join(cycle)}",
                )
            )

        states: dict[str, dict[str, Any]] = {}
        packages: dict[
            str, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
        ] = {}
        event_sets: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            experiment_id = str(entry.get("id"))
            root, plan_path, state_path, claims_path, events_path, artifacts_path = (
                self.experiment_paths(entry)
            )
            required_files = (
                root / "README.md",
                plan_path,
                state_path,
                claims_path,
                events_path,
                artifacts_path,
            )
            missing = [path for path in required_files if not path.exists()]
            for path in missing:
                issues.append(
                    Issue(
                        "missing_file",
                        self.relative_display(path),
                        f"required package file for {experiment_id} is missing",
                    )
                )
            if missing:
                continue
            try:
                plan, state, claims, artifacts = self.load_package(entry)
            except ControlPlaneError as exc:
                issues.append(
                    Issue(
                        "package_read",
                        self.relative_display(root),
                        str(exc),
                    )
                )
                continue
            states[experiment_id] = state
            packages[experiment_id] = (plan, state, claims, artifacts)
            issues.extend(self._schema_issues("experiment", plan, plan_path))
            issues.extend(self._schema_issues("state", state, state_path))
            issues.extend(self._schema_issues("claims", claims, claims_path))
            issues.extend(
                self._schema_issues("artifact_manifest", artifacts, artifacts_path)
            )
            event_issues, events = self._validate_events(events_path, experiment_id)
            issues.extend(event_issues)
            event_sets[experiment_id] = events
            result_path = root / "results" / "result.json"
            if result_path.exists():
                try:
                    result = _read_json(result_path)
                    issues.extend(self._schema_issues("result", result, result_path))
                    if result.get("experiment_id") != experiment_id:
                        issues.append(
                            Issue(
                                "identity_mismatch",
                                self.relative_display(result_path),
                                "result experiment_id does not match package",
                            )
                        )
                except ControlPlaneError as exc:
                    issues.append(
                        Issue("result_read", self.relative_display(result_path), str(exc))
                    )

        for entry in entries:
            experiment_id = str(entry.get("id"))
            if experiment_id not in packages:
                continue
            plan, state, claims, artifacts = packages[experiment_id]
            _, plan_path, state_path, _, _, artifacts_path = self.experiment_paths(
                entry
            )
            issues.extend(
                self._validate_plan_semantics(
                    entry, plan, state, portfolio, plan_path
                )
            )
            issues.extend(
                self._validate_state_semantics(
                    entry,
                    plan,
                    state,
                    claims,
                    artifacts,
                    event_sets.get(experiment_id, []),
                    state_path,
                )
            )
            issues.extend(
                self._validate_artifact_semantics(
                    entry, state, artifacts, portfolio, artifacts_path
                )
            )
            issues.extend(self._validate_lease_consistency(experiment_id, state))
            dependencies_ok, failures = self.dependencies_satisfied(entry, states)
            if state.get("status") == "ready" and not dependencies_ok:
                issues.append(
                    Issue(
                        "ready_dependency",
                        self.relative_display(state_path),
                        "ready experiment has unsatisfied hard dependencies: "
                        + "; ".join(failures),
                    )
                )
            if state.get("status") == "blocked" and dependencies_ok:
                dependency_blockers = [
                    blocker
                    for blocker in state.get("blockers", [])
                    if isinstance(blocker, Mapping)
                    and blocker.get("code") == "dependency_unsatisfied"
                ]
                if dependency_blockers and len(dependency_blockers) == len(
                    state.get("blockers", [])
                ):
                    warnings.append(
                        Issue(
                            "stale_dependency_block",
                            self.relative_display(state_path),
                            "dependencies are now satisfied; claim will unblock atomically",
                            severity="warning",
                        )
                    )

        issues.sort(key=lambda item: (item.path, item.code, item.message))
        warnings.sort(key=lambda item: (item.path, item.code, item.message))
        return {
            "valid": not issues,
            "errors": [item.as_dict() for item in issues],
            "warnings": [item.as_dict() for item in warnings],
            "experiments_checked": len(packages),
        }

    def states(self, portfolio: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for entry in portfolio.get("experiments", []):
            state_path = self.resolve_declared_path(str(entry["state_path"]))
            if state_path.exists():
                state = _read_json(state_path)
                if isinstance(state, dict):
                    result[str(entry["id"])] = state
        return result

    def next_experiment(self) -> dict[str, Any]:
        report = self.validate_all()
        if not report["valid"]:
            raise ControlPlaneError(
                f"portfolio validation failed with {len(report['errors'])} error(s)"
            )
        portfolio = self.load_portfolio()
        states = self.states(portfolio)
        active = [
            experiment_id
            for experiment_id, state in states.items()
            if state.get("status") in ACTIVE_STATUSES
        ]
        if active:
            return {
                "available": False,
                "reason": "active_experiment",
                "active_experiments": sorted(active),
            }
        candidates: list[tuple[int, str, dict[str, Any], str]] = []
        for entry in portfolio.get("experiments", []):
            experiment_id = str(entry["id"])
            state = states.get(experiment_id, {})
            status = state.get("status")
            dependencies_ok, _ = self.dependencies_satisfied(entry, states)
            if status == "ready" and dependencies_ok:
                candidates.append(
                    (int(entry["priority"]), experiment_id, entry, "ready")
                )
            elif status == "blocked" and dependencies_ok:
                blockers = state.get("blockers", [])
                if blockers and all(
                    isinstance(blocker, Mapping)
                    and blocker.get("code") == "dependency_unsatisfied"
                    for blocker in blockers
                ):
                    candidates.append(
                        (
                            int(entry["priority"]),
                            experiment_id,
                            entry,
                            "dependency_unblockable",
                        )
                    )
        if not candidates:
            terminal_dependency_blocks: list[dict[str, Any]] = []
            for entry in portfolio.get("experiments", []):
                experiment_id = str(entry["id"])
                if states.get(experiment_id, {}).get("status") != "blocked":
                    continue
                failed_dependencies: list[dict[str, Any]] = []
                for dependency in entry.get("depends_on", []):
                    if dependency.get("type") != "hard":
                        continue
                    dependency_id = str(dependency["id"])
                    dependency_state = states.get(dependency_id, {})
                    if (
                        dependency_state.get("status") == "finished"
                        and dependency_state.get("outcome")
                        not in dependency.get("satisfying_outcomes", [])
                    ):
                        failed_dependencies.append(
                            {
                                "id": dependency_id,
                                "outcome": dependency_state.get("outcome"),
                                "satisfying_outcomes": dependency.get(
                                    "satisfying_outcomes", []
                                ),
                            }
                        )
                if failed_dependencies:
                    terminal_dependency_blocks.append(
                        {
                            "id": experiment_id,
                            "priority": entry["priority"],
                            "dependencies": failed_dependencies,
                        }
                    )
            if terminal_dependency_blocks:
                return {
                    "available": False,
                    "reason": "terminal_dependency_unsatisfied",
                    "action": portfolio["selection_policy"].get(
                        "unsatisfied_terminal_dependency_action",
                        "block",
                    ),
                    "experiments": sorted(
                        terminal_dependency_blocks,
                        key=lambda item: (int(item["priority"]), str(item["id"])),
                    ),
                }
            return {"available": False, "reason": "no_eligible_experiment"}
        _, experiment_id, entry, eligibility = sorted(candidates)[0]
        state = states[experiment_id]
        return {
            "available": True,
            "id": experiment_id,
            "priority": entry["priority"],
            "path": entry["path"],
            "status": state["status"],
            "eligibility": eligibility,
            "current_stage_id": state.get("current_stage_id"),
        }

    @contextmanager
    def state_lock(self, experiment_id: str) -> Iterator[None]:
        lock_path = self.runtime_control_root / "locks" / f"{experiment_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def lease_path(self, experiment_id: str) -> Path:
        return self.runtime_control_root / "leases" / f"{experiment_id}.json"

    def _process_start_marker(self, pid: int) -> str | None:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            return fields[21]
        except (OSError, IndexError):
            return None

    def _lease_liveness(self, lease: Mapping[str, Any]) -> bool | None:
        if lease.get("hostname") != socket.gethostname():
            return None
        pid = lease.get("pid")
        if pid is None:
            return None
        try:
            pid_value = int(pid)
            os.kill(pid_value, 0)
        except (ValueError, TypeError, ProcessLookupError):
            return False
        except PermissionError:
            return None
        expected_marker = lease.get("process_start_marker")
        if expected_marker is None:
            return None
        return self._process_start_marker(pid_value) == str(expected_marker)

    def _lease_is_stale(self, lease: Mapping[str, Any]) -> bool:
        heartbeat = _parse_timestamp(str(lease["heartbeat_at"]))
        ttl = int(lease["ttl_seconds"])
        return dt.datetime.now(dt.timezone.utc) > heartbeat + dt.timedelta(seconds=ttl)

    def _archive_stale_lease(
        self, lease_path: Path, lease: Mapping[str, Any], portfolio: Mapping[str, Any]
    ) -> None:
        liveness = self._lease_liveness(lease)
        if liveness is True:
            raise ControlPlaneError(
                f"stale lease still owns a live process: {lease_path}"
            )
        if liveness is None:
            raise ControlPlaneError(
                f"cannot safely establish lease owner liveness: {lease_path}"
            )
        archive_root = self.resolve_declared_path(
            str(portfolio["lease_policy"]["stale_archive_path"])
        )
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / (
            f"{lease.get('experiment_id')}.{lease.get('lease_id')}."
            f"{uuid.uuid4().hex}.json"
        )
        os.replace(lease_path, archive_path)

    def _create_lease(
        self, experiment_id: str, agent_id: str, portfolio: Mapping[str, Any]
    ) -> dict[str, Any]:
        lease_path = self.lease_path(experiment_id)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        if lease_path.exists():
            existing = _read_json(lease_path)
            if not self._lease_is_stale(existing):
                raise ControlPlaneError(
                    f"experiment already has an unexpired lease owned by "
                    f"{existing.get('agent_id')}"
                )
            self._archive_stale_lease(lease_path, existing, portfolio)
        now = _utc_now()
        lease = {
            "schema_version": SCHEMA_VERSION,
            "lease_id": uuid.uuid4().hex,
            "experiment_id": experiment_id,
            "agent_id": agent_id,
            "hostname": socket.gethostname(),
            "pid": None,
            "process_start_marker": None,
            "claimed_at": now,
            "heartbeat_at": now,
            "heartbeat_seconds": int(
                portfolio["lease_policy"]["heartbeat_seconds"]
            ),
            "ttl_seconds": int(portfolio["lease_policy"]["ttl_seconds"]),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(lease_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(lease, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if lease_path.exists():
                lease_path.unlink()
            raise
        return lease

    def _refresh_lease(
        self,
        experiment_id: str,
        lease_id: str,
        *,
        process_pid: int | None = None,
    ) -> dict[str, Any]:
        path = self.lease_path(experiment_id)
        lease = _read_json(path)
        if lease.get("lease_id") != lease_id:
            raise ControlPlaneError("executor lease changed")
        lease["heartbeat_at"] = _utc_now()
        lease["pid"] = process_pid
        lease["process_start_marker"] = (
            self._process_start_marker(process_pid) if process_pid is not None else None
        )
        _atomic_write_json(path, lease)
        return lease

    def _assert_plan_identity(
        self,
        plan_path: Path,
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> None:
        actual_hash = _sha256_file(plan_path)
        if state.get("plan_sha256") != actual_hash:
            raise ControlPlaneError(
                f"immutable plan hash drift: state={state.get('plan_sha256')} "
                f"actual={actual_hash}"
            )
        if state.get("plan_revision") != plan.get("plan_revision"):
            raise ControlPlaneError("immutable plan revision drift")

    def _assert_execution_identity(
        self, experiment_id: str, state: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        executor = self._executor(state)
        lease_path = self.lease_path(experiment_id)
        if not lease_path.is_file():
            raise ControlPlaneError("executor lease file is missing")
        lease = _read_json(lease_path)
        if lease.get("experiment_id") != experiment_id:
            raise ControlPlaneError("lease experiment identity mismatch")
        if lease.get("lease_id") != executor.get("lease_id"):
            raise ControlPlaneError("state and lease lease_id mismatch")
        if lease.get("agent_id") != executor.get("agent_id"):
            raise ControlPlaneError("state and lease agent_id mismatch")
        return executor, lease

    def _write_state_cas(
        self, state_path: Path, new_state: Mapping[str, Any], expected_revision: int
    ) -> None:
        current = _read_json(state_path)
        if current.get("revision") != expected_revision:
            raise ControlPlaneError(
                f"state CAS failed: expected revision {expected_revision}, "
                f"found {current.get('revision')}"
            )
        if new_state.get("revision") != expected_revision + 1:
            raise ControlPlaneError("state revision must increment exactly once")
        _atomic_write_json(state_path, new_state)

    def _event_tail(self, events_path: Path) -> tuple[int, str | None]:
        if not events_path.exists():
            return 0, None
        lines = [
            line
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            return 0, None
        last = json.loads(lines[-1])
        return int(last["sequence"]), str(last["hash"])

    def _build_event(
        self,
        experiment_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
        sequence: int,
        previous_hash: str | None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "event_id": f"{experiment_id}-{sequence:06d}",
            "experiment_id": experiment_id,
            "timestamp": _utc_now(),
            "event_type": event_type,
            "actor": actor,
            "previous_hash": previous_hash,
            "payload": dict(payload),
        }
        event["hash"] = _event_hash(event)
        return event

    def _append_prebuilt_event(
        self,
        events_path: Path,
        event: Mapping[str, Any],
        *,
        expected_sequence: int,
    ) -> None:
        sequence, previous_hash = self._event_tail(events_path)
        if sequence + 1 != expected_sequence:
            raise ControlPlaneError(
                f"event sequence CAS failed: expected {expected_sequence}, "
                f"ledger next is {sequence + 1}"
            )
        if event.get("sequence") != expected_sequence:
            raise ControlPlaneError("prebuilt event sequence mismatch")
        if event.get("previous_hash") != previous_hash:
            raise ControlPlaneError("prebuilt event previous_hash mismatch")
        validation = self._schema_issues("event", event, events_path)
        if validation:
            raise ControlPlaneError(
                f"refusing invalid event: {validation[0].message}"
            )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        original_size = events_path.stat().st_size if events_path.exists() else 0
        try:
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if events_path.exists():
                with events_path.open("r+b") as handle:
                    handle.truncate(original_size)
                    handle.flush()
                    os.fsync(handle.fileno())
            raise

    def _commit_state_event(
        self,
        state_path: Path,
        events_path: Path,
        state: Mapping[str, Any],
        *,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        previous_revision = int(state["revision"])
        original_state_text = state_path.read_text(encoding="utf-8")
        original_state = _read_json(state_path)
        if original_state.get("revision") != previous_revision:
            raise ControlPlaneError(
                f"state CAS failed: expected revision {previous_revision}, "
                f"found {original_state.get('revision')}"
            )
        new_state = copy.deepcopy(dict(state))
        new_state["revision"] = previous_revision + 1
        new_state["updated_at"] = _utc_now()
        sequence, previous_hash = self._event_tail(events_path)
        new_state["last_event_seq"] = sequence + 1
        event_payload = dict(payload)
        event_payload.update(
            {
                "revision": new_state["revision"],
                "status": new_state["status"],
                "outcome": new_state.get("outcome"),
            }
        )
        event = self._build_event(
            str(new_state["experiment_id"]),
            event_type,
            actor,
            event_payload,
            sequence + 1,
            previous_hash,
        )
        validation = self._schema_issues("event", event, events_path)
        if validation:
            raise ControlPlaneError(
                f"refusing invalid event: {validation[0].message}"
            )
        self._write_state_cas(state_path, new_state, previous_revision)
        try:
            self._append_prebuilt_event(
                events_path, event, expected_sequence=sequence + 1
            )
        except Exception:
            _atomic_write_text(state_path, original_state_text)
            raise
        return new_state

    def _root_stage_id(self, plan: Mapping[str, Any]) -> str:
        for stage in plan.get("stages", []):
            if not stage.get("needs"):
                return str(stage["id"])
        raise ControlPlaneError("plan has no root stage")

    def _blocked_stage_for_reopen(
        self, plan: Mapping[str, Any], state: Mapping[str, Any]
    ) -> str:
        plan_map, state_map = self._stage_maps(plan, state)
        failed_stage_ids = [
            str(stage["id"])
            for stage in plan.get("stages", [])
            if state_map[str(stage["id"])].get("status") in {"failed", "blocked"}
        ]
        if len(failed_stage_ids) != 1:
            shown = ", ".join(failed_stage_ids) if failed_stage_ids else "none"
            raise ControlPlaneError(
                "reopen-blocked requires exactly one failed/blocked stage; "
                f"found {shown}"
            )
        target_stage_id = failed_stage_ids[0]

        current_stage_id = state.get("current_stage_id")
        if current_stage_id is not None and current_stage_id != target_stage_id:
            raise ControlPlaneError(
                "blocked state current_stage_id conflicts with failed stage: "
                f"{current_stage_id} != {target_stage_id}"
            )

        unrelated_blockers = [
            blocker
            for blocker in state.get("blockers", [])
            if not isinstance(blocker, Mapping)
            or blocker.get("stage_id") != target_stage_id
        ]
        if unrelated_blockers:
            raise ControlPlaneError(
                "reopen-blocked refuses unrelated or unscoped blockers; "
                f"every blocker must name stage_id {target_stage_id}"
            )

        target_plan = plan_map[target_stage_id]
        unmet_needs = [
            str(needed)
            for needed in target_plan.get("needs", [])
            if state_map[str(needed)].get("status") not in {"passed", "skipped"}
        ]
        if unmet_needs:
            details = ", ".join(
                f"{needed}={state_map[needed].get('status')}"
                for needed in unmet_needs
            )
            raise ControlPlaneError(
                f"cannot reopen stage {target_stage_id} before its DAG "
                f"prerequisites pass: {details}"
            )
        return target_stage_id

    def reopen_blocked(self, experiment_id: str, reason: str) -> dict[str, Any]:
        repair_reason = reason.strip()
        if not repair_reason:
            raise ControlPlaneError("reason must be nonempty")
        entry = self.entry(experiment_id)
        _, plan_path, state_path, _, events_path, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") != "blocked":
                raise ControlPlaneError(
                    "reopen-blocked requires blocked state, got "
                    f"{state.get('status')}"
                )
            if state.get("executor") is not None or state.get("lease") is not None:
                raise ControlPlaneError(
                    "reopen-blocked refuses a state with an executor lease"
                )
            lease_path = self.lease_path(experiment_id)
            if lease_path.exists():
                raise ControlPlaneError(
                    f"reopen-blocked refuses an existing lease file: {lease_path}"
                )

            validation = self.validate_all()
            if not validation["valid"]:
                first = validation["errors"][0]
                raise ControlPlaneError(
                    "reopen-blocked requires a canonical valid portfolio; "
                    f"{first['path']}: {first['message']}"
                )
            portfolio = self.load_portfolio()
            dependencies_ok, failures = self.dependencies_satisfied(
                entry, self.states(portfolio)
            )
            if not dependencies_ok:
                raise ControlPlaneError(
                    "cannot reopen with unsatisfied hard dependencies: "
                    + "; ".join(failures)
                )

            target_stage_id = self._blocked_stage_for_reopen(plan, state)
            _, state_map = self._stage_maps(plan, state)
            target_state = state_map[target_stage_id]
            prior_blockers = copy.deepcopy(state.get("blockers", []))
            prior_current_stage_id = state.get("current_stage_id")
            prior_stage_state = copy.deepcopy(target_state)

            target_state["status"] = "ready"
            target_state["attempts"] = 0
            target_state["started_at"] = None
            target_state["completed_at"] = None
            target_state["last_error"] = None
            state["status"] = "ready"
            state["outcome"] = None
            state["current_stage_id"] = target_stage_id
            state["blockers"] = []
            state["executor"] = None
            if "lease" in state:
                state["lease"] = None
            if "phase" in state:
                state["phase"] = target_stage_id

            state = self._commit_state_event(
                state_path,
                events_path,
                state,
                event_type="blocked_experiment_reopened",
                actor="experimentctl",
                payload={
                    "repair_scope": "control_plane",
                    "reason": repair_reason,
                    "target_stage_id": target_stage_id,
                    "prior_blockers": prior_blockers,
                    "prior_current_stage_id": prior_current_stage_id,
                    "prior_stage_state": prior_stage_state,
                },
            )
        return {
            "reopened": True,
            "id": experiment_id,
            "status": state["status"],
            "current_stage_id": state["current_stage_id"],
            "target_stage_id": target_stage_id,
            "revision": state["revision"],
        }

    def record_blocker(
        self, experiment_id: str, code: str, message: str
    ) -> dict[str, Any]:
        blocker_code = code.strip()
        blocker_message = message.strip()
        if not STAGE_ID_RE.fullmatch(blocker_code):
            raise ControlPlaneError(
                "blocker code must match ^[a-z][a-z0-9_]*$"
            )
        if not blocker_message:
            raise ControlPlaneError("blocker message must be nonempty")

        entry = self.entry(experiment_id)
        _, plan_path, state_path, _, events_path, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") != "blocked":
                raise ControlPlaneError(
                    "record-blocker requires blocked state, got "
                    f"{state.get('status')}"
                )
            if state.get("executor") is not None or state.get("lease") is not None:
                raise ControlPlaneError(
                    "record-blocker refuses a state with an executor lease"
                )
            lease_path = self.lease_path(experiment_id)
            if lease_path.exists():
                raise ControlPlaneError(
                    f"record-blocker refuses an existing lease file: {lease_path}"
                )

            validation = self.validate_all()
            if not validation["valid"]:
                first = validation["errors"][0]
                raise ControlPlaneError(
                    "record-blocker requires a canonical valid portfolio; "
                    f"{first['path']}: {first['message']}"
                )

            blocker = {"code": blocker_code, "message": blocker_message}
            prior_blockers = copy.deepcopy(state.get("blockers", []))
            if any(
                isinstance(existing, Mapping)
                and existing.get("code") == blocker_code
                and existing.get("message") == blocker_message
                for existing in prior_blockers
            ):
                raise ControlPlaneError(
                    "record-blocker refuses an exact duplicate blocker"
                )

            prior_current_stage_id = state.get("current_stage_id")
            state["blockers"].append(blocker)
            state = self._commit_state_event(
                state_path,
                events_path,
                state,
                event_type="blocked_experiment_blocker_recorded",
                actor="experimentctl",
                payload={
                    "blocker": blocker,
                    "prior_blockers": prior_blockers,
                    "current_stage_id": prior_current_stage_id,
                },
            )
        return {
            "recorded": True,
            "id": experiment_id,
            "status": state["status"],
            "current_stage_id": state.get("current_stage_id"),
            "blocker": blocker,
            "revision": state["revision"],
        }

    def claim(self, experiment_id: str, agent_id: str) -> dict[str, Any]:
        if not agent_id.strip():
            raise ControlPlaneError("agent-id must be nonempty")
        validation = self.validate_all()
        if not validation["valid"]:
            raise ControlPlaneError(
                f"portfolio validation failed with {len(validation['errors'])} error(s)"
            )
        portfolio = self.load_portfolio()
        entry = self.entry(experiment_id)
        next_item = self.next_experiment()
        if not next_item.get("available") or next_item.get("id") != experiment_id:
            raise ControlPlaneError(
                f"{experiment_id} is not the portfolio-selected next experiment"
            )
        _, plan_path, state_path, _, events_path, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            states = self.states(portfolio)
            dependencies_ok, failures = self.dependencies_satisfied(entry, states)
            if not dependencies_ok:
                raise ControlPlaneError(
                    "unsatisfied hard dependencies: " + "; ".join(failures)
                )
            if state.get("status") == "blocked":
                blockers = state.get("blockers", [])
                if not blockers or not all(
                    isinstance(blocker, Mapping)
                    and blocker.get("code") == "dependency_unsatisfied"
                    for blocker in blockers
                ):
                    raise ControlPlaneError("experiment has non-dependency blockers")
                root_stage = self._root_stage_id(plan)
                state["status"] = "ready"
                state["blockers"] = []
                state["current_stage_id"] = root_stage
                for stage_state in state["stages"]:
                    if stage_state["id"] == root_stage:
                        stage_state["status"] = "ready"
                state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="dependencies_satisfied",
                    actor="experimentctl",
                    payload={"unblocked": True},
                )
            if state.get("status") != "ready":
                raise ControlPlaneError(
                    f"experiment status must be ready, got {state.get('status')}"
                )
            lease = self._create_lease(experiment_id, agent_id, portfolio)
            now = lease["claimed_at"]
            lease_display = self.relative_display(self.lease_path(experiment_id))
            state["status"] = "claimed"
            if "phase" in state:
                state["phase"] = str(state.get("current_stage_id") or "claimed")
            state["executor"] = {
                "agent_id": agent_id,
                "lease_id": lease["lease_id"],
                "lease_path": lease_display,
                "claimed_at": now,
                "heartbeat_at": now,
            }
            if "lease" in state:
                state["lease"] = copy.deepcopy(state["executor"])
            try:
                state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="experiment_claimed",
                    actor=agent_id,
                    payload={
                        "lease_id": lease["lease_id"],
                        "current_stage_id": state.get("current_stage_id"),
                    },
                )
            except Exception:
                lease_path = self.lease_path(experiment_id)
                if lease_path.exists():
                    lease_path.unlink()
                raise
        return {
            "claimed": True,
            "id": experiment_id,
            "agent_id": agent_id,
            "lease_id": lease["lease_id"],
            "status": state["status"],
            "current_stage_id": state.get("current_stage_id"),
        }

    def _stage_maps(
        self, plan: Mapping[str, Any], state: Mapping[str, Any]
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
        plan_map = {
            str(stage["id"]): stage for stage in plan.get("stages", [])
        }
        state_map = {
            str(stage["id"]): stage for stage in state.get("stages", [])
        }
        return plan_map, state_map

    def _eligible_stage_ids(
        self, plan: Mapping[str, Any], state: Mapping[str, Any]
    ) -> list[str]:
        plan_map, state_map = self._stage_maps(plan, state)
        eligible: list[str] = []
        for stage in plan.get("stages", []):
            stage_id = str(stage["id"])
            stage_status = state_map.get(stage_id, {}).get("status")
            if stage_status not in {"pending", "ready", "retry_wait"}:
                continue
            needs = stage.get("needs", [])
            if all(
                state_map.get(str(needed), {}).get("status") in {"passed", "skipped"}
                for needed in needs
            ):
                eligible.append(stage_id)
        return eligible

    def _select_current_stage(
        self, plan: Mapping[str, Any], state: Mapping[str, Any]
    ) -> str:
        current = state.get("current_stage_id")
        plan_map, state_map = self._stage_maps(plan, state)
        if current in plan_map and state_map[current].get("status") in {
            "ready",
            "running",
            "retry_wait",
        }:
            return str(current)
        eligible = self._eligible_stage_ids(plan, state)
        if not eligible:
            raise ControlPlaneError("no runnable stage")
        return eligible[0]

    def _executor(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        executor = state.get("executor") or state.get("lease")
        if not isinstance(executor, Mapping):
            raise ControlPlaneError("state has no executor")
        return executor

    def _work_order(
        self,
        experiment_id: str,
        stage: Mapping[str, Any],
        stage_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        instructions = stage.get("instructions")
        if isinstance(instructions, str):
            instruction_list = [instructions]
        else:
            instruction_list = list(instructions or [])
        completion = stage.get("completion", {})
        evidence_path = (
            completion.get("evidence_file")
            if isinstance(completion, Mapping)
            else None
        )
        return {
            "type": "agent_work_order",
            "experiment_id": experiment_id,
            "stage_id": stage["id"],
            "attempt": stage_state["attempts"],
            "instructions": instruction_list,
            "allowed_write_paths": stage.get("allowed_write_paths", []),
            "inputs": stage.get("inputs", []),
            "outputs": stage.get("outputs", []),
            "acceptance_checks": stage.get("acceptance_checks", []),
            "stop_conditions": stage.get("stop_conditions", []),
            "resources": stage.get("resources", {}),
            "timeout_seconds": stage.get("timeout_seconds"),
            "evidence_path": evidence_path,
            "completion_command": (
                "python experiments_aiselfdrive/tools/experimentctl.py "
                f"complete-stage --id {experiment_id} --evidence "
                f"{evidence_path or '<evidence-json>'}"
            ),
        }

    def _command_failure(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        state: dict[str, Any],
        stage: Mapping[str, Any],
        error_class: str,
        message: str,
    ) -> dict[str, Any]:
        _, _, state_path, _, events_path, _ = self.experiment_paths(entry)
        _, state_map = self._stage_maps(plan, state)
        stage_state = state_map[str(stage["id"])]
        max_attempts = int(stage.get("retry", {}).get("max_attempts", 1))
        stage_state["last_error"] = {"class": error_class, "message": message}
        stage_state["completed_at"] = _utc_now()
        if stage_state["attempts"] < max_attempts:
            stage_state["status"] = "retry_wait"
            state["status"] = "retry_wait"
            event_type = "stage_retry_wait"
        else:
            stage_state["status"] = "failed"
            state["status"] = "blocked"
            state["blockers"] = [
                {
                    "code": "stage_failed",
                    "stage_id": stage["id"],
                    "error_class": error_class,
                    "message": message,
                }
            ]
            state["current_stage_id"] = None
            state["executor"] = None
            if "lease" in state:
                state["lease"] = None
            event_type = "stage_failed"
        updated_state = self._commit_state_event(
            state_path,
            events_path,
            state,
            event_type=event_type,
            actor="experimentctl",
            payload={
                "stage_id": stage["id"],
                "error_class": error_class,
                "message": message,
            },
        )
        if updated_state["status"] == "blocked":
            lease_path = self.lease_path(str(entry["id"]))
            if lease_path.exists():
                lease_path.unlink()
        return updated_state

    def _run_command_with_heartbeat(
        self,
        experiment_id: str,
        lease_id: str,
        argv: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        lease = _read_json(self.lease_path(experiment_id))
        heartbeat_seconds = max(1, int(lease["heartbeat_seconds"]))
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._refresh_lease(experiment_id, lease_id, process_pid=process.pid)
        started = time.monotonic()
        while True:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    list(argv),
                    timeout_seconds,
                    output=stdout,
                    stderr=stderr,
                )
            try:
                stdout, stderr = process.communicate(
                    timeout=min(float(heartbeat_seconds), remaining)
                )
                return subprocess.CompletedProcess(
                    list(argv), int(process.returncode), stdout, stderr
                )
            except subprocess.TimeoutExpired:
                self._refresh_lease(
                    experiment_id, lease_id, process_pid=process.pid
                )

    def _output_path(self, output: Any) -> str | None:
        if isinstance(output, Mapping):
            raw = output.get("path")
            return str(raw) if raw else None
        if isinstance(output, str) and (
            "/" in output or output.startswith(".") or Path(output).is_absolute()
        ):
            return output
        return None

    def _artifact_evidence(self, identifier: str, path: Path) -> dict[str, Any]:
        if path.is_file():
            return {
                "id": identifier,
                "path": self.relative_display(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        return {
            "id": identifier,
            "path": self.relative_display(path),
            "sha256": None,
            "size_bytes": None,
        }

    def _generated_command_evidence(
        self,
        experiment_id: str,
        stage: Mapping[str, Any],
        completed: subprocess.CompletedProcess[str],
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        for index, check in enumerate(stage.get("acceptance_checks", []), start=1):
            if isinstance(check, Mapping):
                check_id = str(check.get("id", check.get("name", f"check_{index}")))
            else:
                check_id = str(check)
            checks.append(
                {
                    "id": check_id,
                    "status": "passed",
                    "detail": "command exited successfully",
                }
            )
        artifacts: list[dict[str, Any]] = []
        for index, output in enumerate(stage.get("outputs", []), start=1):
            raw_path = self._output_path(output)
            if raw_path is None:
                continue
            path = self.resolve_declared_path(raw_path)
            if path.exists():
                identifier = (
                    str(output.get("id", f"output_{index}"))
                    if isinstance(output, Mapping)
                    else f"output_{index}"
                )
                artifacts.append(self._artifact_evidence(identifier, path))
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "stage_id": stage["id"],
            "status": "passed",
            "completed_at": _utc_now(),
            "summary": "Declared command completed with exit code zero.",
            "checks": checks,
            "artifacts": artifacts,
            "metrics": {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-32768:],
                "stderr_tail": completed.stderr[-32768:],
            },
            "error": None,
        }

    def run_stage(self, experiment_id: str) -> dict[str, Any]:
        entry = self.entry(experiment_id)
        _, plan_path, state_path, _, events_path, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") not in {"claimed", "running", "retry_wait"}:
                raise ControlPlaneError(
                    f"run-stage requires claimed/running/retry_wait, got "
                    f"{state.get('status')}"
                )
            executor, _ = self._assert_execution_identity(experiment_id, state)
            lease = self._refresh_lease(
                experiment_id, str(executor["lease_id"]), process_pid=None
            )
            stage_id = self._select_current_stage(plan, state)
            plan_map, state_map = self._stage_maps(plan, state)
            stage = plan_map[stage_id]
            stage_state = state_map[stage_id]
            if stage_state["status"] != "running":
                stage_state["status"] = "running"
                stage_state["attempts"] += 1
                stage_state["started_at"] = _utc_now()
                stage_state["completed_at"] = None
                stage_state["last_error"] = None
            state["status"] = "running"
            state["current_stage_id"] = stage_id
            if "phase" in state:
                state["phase"] = stage_id
            state["executor"]["heartbeat_at"] = lease["heartbeat_at"]
            if "lease" in state and state["lease"] is not None:
                state["lease"]["heartbeat_at"] = lease["heartbeat_at"]
            state = self._commit_state_event(
                state_path,
                events_path,
                state,
                event_type="stage_started",
                actor=str(executor["agent_id"]),
                payload={"stage_id": stage_id, "attempt": stage_state["attempts"]},
            )
            if stage["execution_mode"] == "agent":
                _, refreshed_state_map = self._stage_maps(plan, state)
                return self._work_order(
                    experiment_id, stage, refreshed_state_map[stage_id]
                )
            command = stage["command"]
            argv = [str(item) for item in command["argv"]]
            cwd = self.resolve_declared_path(str(command["cwd"]))
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in command["env"].items()})
            lease_id = str(executor["lease_id"])

        try:
            completed = self._run_command_with_heartbeat(
                experiment_id,
                lease_id,
                argv,
                cwd,
                environment,
                int(stage["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                self._refresh_lease(experiment_id, lease_id, process_pid=None)
                self._command_failure(
                    entry,
                    plan,
                    state,
                    stage,
                    "timeout",
                    f"command exceeded {stage['timeout_seconds']} seconds",
                )
            raise ControlPlaneError(str(exc)) from exc
        finally:
            if self.lease_path(experiment_id).exists():
                try:
                    self._refresh_lease(experiment_id, lease_id, process_pid=None)
                except ControlPlaneError:
                    pass
        if completed.returncode != 0:
            message = (
                f"command exited {completed.returncode}; stderr tail: "
                f"{completed.stderr[-4096:]}"
            )
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                self._command_failure(
                    entry, plan, state, stage, "command_exit", message
                )
            raise ControlPlaneError(message)

        completion = stage.get("completion", {})
        declared_evidence: str | None = None
        if isinstance(completion, Mapping):
            declared_evidence = completion.get("evidence_file") or completion.get(
                "marker"
            )
        if declared_evidence:
            evidence_path = self.resolve_declared_path(str(declared_evidence))
            evidence_error: str | None = None
            if not evidence_path.is_file():
                evidence_error = (
                    f"declared completion evidence is missing: {evidence_path}"
                )
            else:
                try:
                    evidence = _read_json(evidence_path)
                    schema_issues = self._schema_issues(
                        "result", evidence, evidence_path
                    )
                    if schema_issues:
                        evidence_error = (
                            f"declared completion evidence is invalid: "
                            f"{schema_issues[0].message}"
                        )
                except ControlPlaneError as exc:
                    evidence_error = str(exc)
            if evidence_error is not None:
                with self.state_lock(experiment_id):
                    state = _read_json(state_path)
                    self._command_failure(
                        entry,
                        plan,
                        state,
                        stage,
                        "completion_evidence",
                        evidence_error,
                    )
                raise ControlPlaneError(evidence_error)
            return self.complete_stage(experiment_id, evidence_path)
        if not (
            isinstance(completion, Mapping)
            and completion.get("process_exit_only") is True
        ):
            message = (
                "command stage exited zero but does not declare a valid evidence "
                "file or process_exit_only completion"
            )
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                self._command_failure(
                    entry,
                    plan,
                    state,
                    stage,
                    "completion_evidence",
                    message,
                )
            raise ControlPlaneError(message)
        evidence = self._generated_command_evidence(
            experiment_id, stage, completed
        )
        evidence_path = (
            self.runtime_control_root
            / "evidence"
            / experiment_id
            / f"{stage_id}.json"
        )
        _atomic_write_json(evidence_path, evidence)
        return self.complete_stage(experiment_id, evidence_path)

    def _acceptance_names(self, stage: Mapping[str, Any]) -> list[str]:
        names: list[str] = []
        for index, check in enumerate(stage.get("acceptance_checks", []), start=1):
            if isinstance(check, Mapping):
                names.append(str(check.get("id", check.get("name", f"check_{index}"))))
            else:
                names.append(str(check))
        return names

    def _evidence_check_map(
        self, evidence: Mapping[str, Any]
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for check in evidence.get("checks", []):
            identifier = check.get("id", check.get("name"))
            if identifier is not None:
                result[str(identifier)] = str(check.get("status"))
        return result

    def _verify_completion_evidence(
        self,
        entry: Mapping[str, Any],
        stage: Mapping[str, Any],
        evidence: Mapping[str, Any],
        evidence_path: Path,
    ) -> None:
        issues = self._schema_issues("result", evidence, evidence_path)
        if issues:
            raise ControlPlaneError(
                f"invalid evidence {issues[0].path}: {issues[0].message}"
            )
        experiment_id = str(entry["id"])
        if evidence.get("experiment_id") != experiment_id:
            raise ControlPlaneError("evidence experiment_id mismatch")
        if evidence.get("stage_id") != stage.get("id"):
            raise ControlPlaneError("evidence stage_id mismatch")
        if evidence.get("status") != "passed":
            raise ControlPlaneError(
                f"cannot complete stage from evidence status {evidence.get('status')}"
            )
        if any(
            check.get("status") in {"failed", "blocked"}
            for check in evidence.get("checks", [])
        ):
            raise ControlPlaneError("evidence contains failed or blocked checks")
        expected_checks = self._acceptance_names(stage)
        check_map = self._evidence_check_map(evidence)
        missing_checks = [
            name for name in expected_checks if check_map.get(name) != "passed"
        ]
        if missing_checks:
            raise ControlPlaneError(
                "acceptance checks not evidenced as passed: "
                + ", ".join(missing_checks)
            )
        completion = stage.get("completion", {})
        declared_evidence_path: Path | None = None
        if isinstance(completion, Mapping) and completion.get("evidence_file"):
            expected = self.resolve_declared_path(str(completion["evidence_file"]))
            if evidence_path.resolve(strict=False) != expected:
                raise ControlPlaneError(
                    f"evidence must be written to declared path {expected}"
                )
            declared_evidence_path = expected.resolve(strict=False)
        evidence_artifacts = {
            str(item["path"]): item for item in evidence.get("artifacts", [])
        }
        for output in stage.get("outputs", []):
            raw_path = self._output_path(output)
            if raw_path is None:
                continue
            path = self.resolve_declared_path(raw_path)
            if not path.exists():
                raise ControlPlaneError(f"declared stage output is missing: {path}")
            if (
                declared_evidence_path is not None
                and path.resolve(strict=False) == declared_evidence_path
            ):
                continue
            aliases = {raw_path, self.relative_display(path), str(path)}
            artifact = next(
                (
                    evidence_artifacts[alias]
                    for alias in aliases
                    if alias in evidence_artifacts
                ),
                None,
            )
            if artifact is None:
                raise ControlPlaneError(
                    f"evidence omits declared stage output: {raw_path}"
                )
            if path.is_file():
                actual_hash = _sha256_file(path)
                if artifact.get("sha256") != actual_hash:
                    raise ControlPlaneError(
                        f"evidence hash mismatch for {raw_path}: {actual_hash}"
                    )
                if artifact.get("size_bytes") != path.stat().st_size:
                    raise ControlPlaneError(
                        f"evidence size mismatch for {raw_path}"
                    )

    def _closeout_invariants(
        self,
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        claims: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        outcome: str | None,
    ) -> None:
        if outcome not in TERMINAL_OUTCOMES:
            raise ControlPlaneError(
                "closeout evidence must declare a terminal outcome"
            )
        _, state_map = self._stage_maps(plan, state)
        for stage in plan.get("stages", []):
            stage_state = state_map[str(stage["id"])]
            if stage.get("required") and stage_state.get("status") not in {
                "passed",
                "skipped",
            }:
                if outcome not in {"failed", "cancelled", "invalid"}:
                    raise ControlPlaneError(
                        f"required stage {stage['id']} is not terminal"
                    )
        pending = [
            claim.get("id")
            for claim in claims.get("claims", [])
            if claim.get("verdict") == "pending"
        ]
        if pending:
            raise ControlPlaneError(
                "closeout has pending claims: " + ", ".join(str(item) for item in pending)
            )
        unverified = [
            artifact.get("id")
            for artifact in artifacts.get("artifacts", [])
            if artifact.get("required") and artifact.get("status") != "verified"
        ]
        if unverified:
            raise ControlPlaneError(
                "closeout has unverified required artifacts: "
                + ", ".join(str(item) for item in unverified)
            )

    def complete_stage(
        self, experiment_id: str, evidence_path: Path | str
    ) -> dict[str, Any]:
        entry = self.entry(experiment_id)
        root, plan_path, state_path, claims_path, events_path, artifacts_path = (
            self.experiment_paths(entry)
        )
        evidence_resolved = self.resolve_declared_path(str(evidence_path))
        portfolio = self.load_portfolio()
        if not any(
            _is_within(evidence_resolved, allowed)
            for allowed in self.allowed_roots(portfolio)
        ):
            raise ControlPlaneError("evidence path is outside allowed roots")
        evidence = _read_json(evidence_resolved)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            claims = _read_yaml(claims_path)
            artifacts = _read_json(artifacts_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") != "running":
                raise ControlPlaneError(
                    f"complete-stage requires running state, got {state.get('status')}"
                )
            executor, _ = self._assert_execution_identity(experiment_id, state)
            lease_id = str(executor["lease_id"])
            self._refresh_lease(experiment_id, lease_id, process_pid=None)
            stage_id = state.get("current_stage_id")
            plan_map, state_map = self._stage_maps(plan, state)
            if stage_id not in plan_map:
                raise ControlPlaneError("state has no valid current stage")
            stage = plan_map[str(stage_id)]
            stage_state = state_map[str(stage_id)]
            if stage_state.get("status") != "running":
                raise ControlPlaneError("current stage is not running")
            self._verify_completion_evidence(
                entry, stage, evidence, evidence_resolved
            )
            stage_state["status"] = "passed"
            stage_state["completed_at"] = evidence["completed_at"]
            stage_state["last_error"] = None
            evidence_display = self.relative_display(evidence_resolved)
            if evidence_display not in stage_state["evidence_refs"]:
                stage_state["evidence_refs"].append(evidence_display)

            skipped = set(evidence.get("skipped_stage_ids", []))
            if stage.get("kind") == "closeout":
                skipped.update(
                    candidate["id"]
                    for candidate in plan.get("stages", [])
                    if not candidate.get("required")
                    and state_map[str(candidate["id"])]["status"]
                    not in TERMINAL_STAGE_STATUSES
                )
            for skipped_id in skipped:
                if skipped_id not in plan_map:
                    raise ControlPlaneError(
                        f"evidence requests skip of unknown stage {skipped_id}"
                    )
                if plan_map[skipped_id].get("required"):
                    raise ControlPlaneError(
                        f"cannot skip required stage {skipped_id}"
                    )
                skipped_state = state_map[skipped_id]
                if skipped_state["status"] not in TERMINAL_STAGE_STATUSES:
                    skipped_state["status"] = "skipped"
                    skipped_state["completed_at"] = evidence["completed_at"]
                    skipped_state["evidence_refs"].append(evidence_display)

            if "verified_artifact_ids" in state:
                state["verified_artifact_ids"] = sorted(
                    str(artifact["id"])
                    for artifact in artifacts.get("artifacts", [])
                    if artifact.get("status") == "verified"
                )

            if stage.get("kind") == "closeout":
                outcome = evidence.get("outcome") or evidence.get("metrics", {}).get(
                    "outcome"
                )
                schema_issues = self._schema_issues(
                    "claims", claims, claims_path
                ) + self._schema_issues(
                    "artifact_manifest", artifacts, artifacts_path
                )
                if schema_issues:
                    raise ControlPlaneError(
                        f"closeout metadata is invalid: {schema_issues[0].message}"
                    )
                artifact_issues = self._validate_artifact_semantics(
                    entry,
                    state,
                    artifacts,
                    self.load_portfolio(),
                    artifacts_path,
                )
                if artifact_issues:
                    raise ControlPlaneError(
                        f"closeout artifact verification failed: "
                        f"{artifact_issues[0].message}"
                    )
                self._closeout_invariants(plan, state, claims, artifacts, outcome)
                state["status"] = "finished"
                state["outcome"] = outcome
                state["current_stage_id"] = None
                state["executor"] = None
                if "lease" in state:
                    state["lease"] = None
                state["artifacts_verified"] = True
                state["closeout_passed"] = True
                if "phase" in state:
                    state["phase"] = "finished"
                event_type = "experiment_finished"
            else:
                requested_next = evidence.get("next_stage_id")
                eligible = self._eligible_stage_ids(plan, state)
                if requested_next is not None:
                    if requested_next not in eligible:
                        raise ControlPlaneError(
                            f"requested next stage {requested_next} is not eligible"
                        )
                    next_stage = str(requested_next)
                elif eligible:
                    next_stage = eligible[0]
                else:
                    raise ControlPlaneError(
                        "no eligible stage remains; a closeout stage is required"
                    )
                state_map[next_stage]["status"] = "ready"
                state["current_stage_id"] = next_stage
                state["status"] = "claimed"
                if "phase" in state:
                    state["phase"] = next_stage
                event_type = "stage_completed"
            state = self._commit_state_event(
                state_path,
                events_path,
                state,
                event_type=event_type,
                actor=str(executor["agent_id"]),
                payload={
                    "stage_id": stage_id,
                    "evidence_path": evidence_display,
                    "evidence_sha256": _sha256_file(evidence_resolved),
                    "next_stage_id": state.get("current_stage_id"),
                },
            )
            if state["status"] == "finished":
                lease_path = self.lease_path(experiment_id)
                if lease_path.exists():
                    lease_path.unlink()
        return {
            "completed": True,
            "id": experiment_id,
            "stage_id": stage_id,
            "status": state["status"],
            "outcome": state.get("outcome"),
            "next_stage_id": state.get("current_stage_id"),
            "revision": state["revision"],
        }

    def heartbeat(self, experiment_id: str, agent_id: str) -> dict[str, Any]:
        entry = self.entry(experiment_id)
        _, plan_path, state_path, _, _, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") not in ACTIVE_STATUSES:
                raise ControlPlaneError(
                    f"heartbeat requires active state, got {state.get('status')}"
                )
            executor, _ = self._assert_execution_identity(experiment_id, state)
            if executor.get("agent_id") != agent_id:
                raise ControlPlaneError("agent-id does not own this experiment")
            lease = self._refresh_lease(
                experiment_id, str(executor["lease_id"]), process_pid=None
            )
            # The lease file is the canonical high-frequency liveness record.
            # Heartbeats are intentionally not appended to events.jsonl:
            # that ledger records material state transitions, and a 60-second
            # heartbeat stream would obscure the scientific execution history.
        return {
            "heartbeat": True,
            "id": experiment_id,
            "agent_id": agent_id,
            "lease_id": lease["lease_id"],
            "heartbeat_at": lease["heartbeat_at"],
            "heartbeat_seconds": lease["heartbeat_seconds"],
            "ttl_seconds": lease["ttl_seconds"],
            "revision": state["revision"],
        }

    def _status_markdown(self, portfolio: Mapping[str, Any]) -> str:
        states = self.states(portfolio)
        lines = [
            "<!-- generated by experimentctl; do not edit -->",
            "# ReXGroundingCT AI Self-Drive Status",
            "",
            "| Priority | Experiment | Status | Outcome | Current stage | Blocker | Updated |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
        for entry in sorted(
            portfolio.get("experiments", []),
            key=lambda item: (int(item["priority"]), str(item["id"])),
        ):
            state = states.get(str(entry["id"]), {})
            blockers = state.get("blockers", [])
            blocker_text = ""
            if blockers:
                first = blockers[0]
                blocker_text = (
                    str(first.get("message", first.get("code", "")))
                    if isinstance(first, Mapping)
                    else str(first)
                )
                if len(blockers) > 1:
                    blocker_text += f" (+{len(blockers) - 1})"
            values = [
                str(entry["priority"]),
                str(entry["id"]),
                str(state.get("status", "missing")),
                str(state.get("outcome") or "—"),
                str(state.get("current_stage_id") or "—"),
                blocker_text or "—",
                str(state.get("updated_at") or "—"),
            ]
            escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(escaped) + " |")
        lines.extend(
            [
                "",
                "Run `python experiments_aiselfdrive/tools/experimentctl.py "
                "next --json` to select the next eligible experiment.",
                "",
            ]
        )
        return "\n".join(lines)

    def _readme_state_block(
        self, entry: Mapping[str, Any], state: Mapping[str, Any]
    ) -> str:
        blockers = state.get("blockers", [])
        blocker_lines = []
        for blocker in blockers:
            if isinstance(blocker, Mapping):
                blocker_lines.append(
                    f"- `{blocker.get('code', 'blocked')}`: "
                    f"{blocker.get('message', blocker.get('resolution', ''))}"
                )
            else:
                blocker_lines.append(f"- {blocker}")
        lines = [
            "<!-- experimentctl:state:start -->",
            "## Execution state (generated)",
            "",
            f"- Status: `{state.get('status')}`",
            f"- Outcome: `{state.get('outcome') or 'pending'}`",
            f"- Current stage: `{state.get('current_stage_id') or 'none'}`",
            f"- State revision: `{state.get('revision')}`",
            f"- Updated: `{state.get('updated_at')}`",
        ]
        if blocker_lines:
            lines.extend(["", "Blockers:", "", *blocker_lines])
        lines.append("<!-- experimentctl:state:end -->")
        return "\n".join(lines)

    def render(self) -> dict[str, Any]:
        validation = self.validate_all()
        if not validation["valid"]:
            raise ControlPlaneError(
                f"refusing render with {len(validation['errors'])} validation error(s)"
            )
        portfolio = self.load_portfolio()
        status_path = self.workspace_root / "STATUS.md"
        _atomic_write_text(status_path, self._status_markdown(portfolio))
        rendered_readmes = 0
        start = "<!-- experimentctl:state:start -->"
        end = "<!-- experimentctl:state:end -->"
        states = self.states(portfolio)
        for entry in portfolio.get("experiments", []):
            readme = self.resolve_declared_path(str(entry["path"])) / "README.md"
            if not readme.is_file():
                continue
            text = readme.read_text(encoding="utf-8")
            block = self._readme_state_block(entry, states[str(entry["id"])])
            if start in text and end in text:
                prefix, remainder = text.split(start, 1)
                _, suffix = remainder.split(end, 1)
                updated = prefix.rstrip() + "\n\n" + block + suffix
            else:
                updated = text.rstrip() + "\n\n" + block + "\n"
            _atomic_write_text(readme, updated)
            rendered_readmes += 1
        return {
            "rendered": True,
            "status_path": self.relative_display(status_path),
            "experiment_readmes": rendered_readmes,
        }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validated ReXGroundingCT self-drive experiment controller"
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate portfolio")
    validate.add_argument("--all", action="store_true", required=True)

    next_parser = subparsers.add_parser("next", help="select next experiment")
    next_parser.add_argument("--json", action="store_true", required=True)

    claim = subparsers.add_parser("claim", help="claim selected experiment")
    claim.add_argument("--id", required=True)
    claim.add_argument("--agent-id", required=True)

    run_stage = subparsers.add_parser("run-stage", help="run current stage")
    run_stage.add_argument("--id", required=True)

    heartbeat = subparsers.add_parser(
        "heartbeat", help="refresh an active agent-stage lease"
    )
    heartbeat.add_argument("--id", required=True)
    heartbeat.add_argument("--agent-id", required=True)

    complete = subparsers.add_parser(
        "complete-stage", help="complete current stage from verified evidence"
    )
    complete.add_argument("--id", required=True)
    complete.add_argument("--evidence", type=Path, required=True)

    reopen = subparsers.add_parser(
        "reopen-blocked",
        help="audit and reopen one failed stage after a control-plane repair",
    )
    reopen.add_argument("--id", required=True)
    reopen.add_argument("--reason", required=True)

    record_blocker = subparsers.add_parser(
        "record-blocker",
        help="append an audited blocker to an already blocked experiment",
    )
    record_blocker.add_argument("--id", required=True)
    record_blocker.add_argument("--code", required=True)
    record_blocker.add_argument("--message", required=True)

    subparsers.add_parser("render", help="render generated status views")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    controller = ControlPlane(args.workspace_root)
    try:
        if args.command == "validate":
            result = controller.validate_all()
            _print_json(result)
            return 0 if result["valid"] else 1
        if args.command == "next":
            _print_json(controller.next_experiment())
            return 0
        if args.command == "claim":
            _print_json(controller.claim(args.id, args.agent_id))
            return 0
        if args.command == "run-stage":
            _print_json(controller.run_stage(args.id))
            return 0
        if args.command == "heartbeat":
            _print_json(controller.heartbeat(args.id, args.agent_id))
            return 0
        if args.command == "complete-stage":
            _print_json(controller.complete_stage(args.id, args.evidence))
            return 0
        if args.command == "reopen-blocked":
            _print_json(controller.reopen_blocked(args.id, args.reason))
            return 0
        if args.command == "record-blocker":
            _print_json(
                controller.record_blocker(args.id, args.code, args.message)
            )
            return 0
        if args.command == "render":
            _print_json(controller.render())
            return 0
    except ControlPlaneError as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 2
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
