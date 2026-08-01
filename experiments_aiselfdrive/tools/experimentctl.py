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
import functools
import hashlib
import json
import math
import os
import re
import secrets
import signal
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
    r"(?:\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|\bFILL_ME\b|"
    r"\bTO_BE_LOCKED\b|\bRESOLVED_BEFORE_EXECUTION\b|"
    r"\bRESOLVED_FROM_[A-Z0-9_]+\b|__[A-Z0-9_]+__|<[^>\n]+>)",
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
CLAIM_TYPES = {"feasibility", "mechanistic", "causal", "performance", "safety"}
RETRY_POLICY_CLASSES: dict[str, set[str]] = {
    "non_retryable": set(),
    "non_retryable_agent_implementation": set(),
    "non_retryable_closeout": set(),
    "deterministic_selfcheck": set(),
    "transient": {"command_runtime", "transient_io"},
    "transient_io": {"command_runtime", "transient_io"},
    "network_then_hash_validation": {"command_runtime", "network_error", "transient_io"},
    "resource_then_transient": {"resource_unavailable", "command_runtime", "transient_io"},
    "resource_then_evaluation": {"resource_unavailable", "command_runtime", "timeout", "transient_io"},
    "evaluation": {"command_runtime", "timeout", "transient_io"},
    "training": {"command_runtime", "timeout", "command_exit", "container_start_failure", "transient_io"},
    "training_and_evaluation": {"command_runtime", "timeout", "command_exit", "container_start_failure", "transient_io"},
    "per_case_transient": {"command_runtime", "command_exit", "transient_io"},
}
CONTROLLER_NON_RETRYABLE_CLASSES = {
    "artifact_hash_mismatch",
    "budget_exhausted",
    "completion_evidence",
    "completion_validation",
    "config_hash_drift",
    "geometry_failure",
    "geometry_mismatch",
    "hash_mismatch",
    "orientation_mismatch",
    "schema_validation",
}
CONTROL_PLANE_LOCK_VERSION = "1.0"
DOCKER_LEASE_LABEL_KEY = "rexgroundingct.aiselfdrive.lease"
DOCKER_EXPERIMENT_LABEL_KEY = "rexgroundingct.aiselfdrive.experiment"
COMMAND_EXPERIMENT_ENV = "REXGROUNDINGCT_AISELFDRIVE_EXPERIMENT_ID"
COMMAND_LEASE_ENV = "REXGROUNDINGCT_AISELFDRIVE_LEASE_ID"
COMMAND_DOCKER_LABEL_ENV = "REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL"
T4_CAPABILITY_ENV = "REXGROUNDINGCT_AISELFDRIVE_T4_CAPABILITY"
T4_CAPABILITY_SHA_ENV = "REXGROUNDINGCT_AISELFDRIVE_T4_CAPABILITY_SHA256"
T4_PROMOTION_SHA_ENV = "REXGROUNDINGCT_AISELFDRIVE_PROMOTION_SHA256"
_UNSET = object()
SCHEMA_FILES = {
    "portfolio": "portfolio.schema.json",
    "experiment": "experiment.schema.json",
    "state": "state.schema.json",
    "event": "event.schema.json",
    "claims": "claims.schema.json",
    "artifact_manifest": "artifact_manifest.schema.json",
    "result": "result.schema.json",
    "promotion": "promotion.schema.json",
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


def _hash_operation(method: Any) -> Any:
    """Scope metadata-keyed hash reuse to one logical controller operation.

    A filesystem can expose timestamps too coarsely to distinguish two rapid,
    same-size writes.  Clearing the cache at every outer operation preserves
    intra-operation de-duplication without carrying a digest across mutations.
    """

    @functools.wraps(method)
    def wrapped(self: "ControlPlane", *args: Any, **kwargs: Any) -> Any:
        outermost = self._hash_operation_depth == 0
        if outermost:
            self._hash_cache.clear()
        self._hash_operation_depth += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._hash_operation_depth -= 1
            if outermost:
                self._hash_cache.clear()

    return wrapped


def _promotion_atomic(method: Any) -> Any:
    """Rollback every promotion-controlled file if portfolio commit fails."""

    @functools.wraps(method)
    def wrapped(self: "ControlPlane", *args: Any, **kwargs: Any) -> Any:
        lock_path = self.runtime_control_root / "locks" / "portfolio_promotion.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                portfolio = self.load_portfolio()
                candidates = [
                    entry
                    for entry in portfolio.get("experiments", [])
                    if (entry.get("promotion") or {}).get("candidate")
                ]
                paths: set[Path] = {self.workspace_root / "STATUS.md"}
                promotion_root = (
                    self.resolve_declared_path(str(portfolio["runtime_root"]))
                    / "promotions"
                )
                for entry in portfolio.get("experiments", []):
                    paths.add(
                        self.resolve_declared_path(str(entry["path"])) / "README.md"
                    )
                for entry in candidates:
                    root, _, state_path, _, events_path, artifacts_path = (
                        self.experiment_paths(entry)
                    )
                    paths.update(
                        {
                            state_path,
                            events_path,
                            artifacts_path,
                            root / "results" / "result.json",
                            promotion_root / f"{entry['id']}.json",
                        }
                    )
                snapshots = {
                    path: (path.read_bytes() if path.is_file() else None)
                    for path in paths
                }
                try:
                    return method(self, *args, **kwargs)
                except BaseException:
                    for path, original in snapshots.items():
                        if original is None:
                            if path.exists() and path.is_file():
                                path.unlink()
                        else:
                            _atomic_write_bytes(path, original)
                    raise
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return wrapped


def _completion_failure_transition(method: Any) -> Any:
    """Convert owned completion validation failures into explicit state."""

    @functools.wraps(method)
    def wrapped(
        self: "ControlPlane",
        experiment_id: str,
        evidence_path: Path | str,
        agent_id: str,
    ) -> Any:
        try:
            return method(self, experiment_id, evidence_path, agent_id)
        except ControlPlaneError as original:
            try:
                entry = self.entry(experiment_id)
                _, plan_path, state_path, _, _, _ = self.experiment_paths(entry)
                with self.state_lock(experiment_id):
                    plan = _read_yaml(plan_path)
                    state = _read_json(state_path)
                    executor = state.get("executor") or state.get("lease")
                    if (
                        state.get("status") != "running"
                        or not isinstance(executor, Mapping)
                        or executor.get("agent_id") != agent_id
                    ):
                        raise original
                    _, lease = self._assert_execution_identity(experiment_id, state)
                    if (
                        lease.get("process_role") == "command"
                        and self._lease_liveness(lease) is True
                    ):
                        raise original
                    plan_map, _ = self._stage_maps(plan, state)
                    stage_id = str(state.get("current_stage_id"))
                    stage = plan_map.get(stage_id)
                    if stage is None:
                        raise original
                    updated = self._command_failure(
                        entry,
                        plan,
                        state,
                        stage,
                        "completion_validation",
                        str(original),
                    )
                raise ControlPlaneError(
                    f"completion rejected and stage transitioned to {updated['status']}: {original}"
                ) from original
            except ControlPlaneError as transition_error:
                if transition_error is original:
                    raise
                if str(transition_error).startswith("completion rejected and stage transitioned"):
                    raise
                raise original

    return wrapped


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


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_StrictSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ControlPlaneError(f"cannot read YAML {path}: {exc}") from exc


def _load_yaml_bytes(data: bytes, source: str) -> Any:
    try:
        return yaml.load(data.decode("utf-8"), Loader=_StrictSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ControlPlaneError(f"cannot read YAML {source}: {exc}") from exc


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


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
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
        self._hash_cache: dict[tuple[str, int, int, int, int, int], str] = {}
        self._hash_operation_depth = 0

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

    def _hash_file_cached(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        key = (
            str(resolved),
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )
        digest = self._hash_cache.get(key)
        if digest is None:
            digest = _sha256_file(resolved)
            self._hash_cache[key] = digest
        return digest

    @_hash_operation
    def _tree_digest_and_size(self, root: Path) -> tuple[str, int]:
        """Hash a directory without following links outside the tree."""
        resolved_root = root.resolve(strict=True)
        records: list[dict[str, Any]] = []
        total_size = 0
        for path in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(resolved_root).as_posix()
            if path.is_symlink():
                records.append(
                    {"path": relative, "type": "symlink", "target": os.readlink(path)}
                )
            elif path.is_dir():
                records.append({"path": relative, "type": "directory"})
            elif path.is_file():
                size = path.stat().st_size
                total_size += size
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "size_bytes": size,
                        "sha256": self._hash_file_cached(path),
                    }
                )
            else:
                raise ControlPlaneError(f"unsupported tree entry: {path}")
        digest = _sha256_bytes(_canonical_json(records).encode("utf-8"))
        return digest, total_size

    @_hash_operation
    def _path_digest_and_size(self, path: Path) -> tuple[str, int]:
        if path.is_file():
            return self._hash_file_cached(path), path.stat().st_size
        if path.is_dir():
            return self._tree_digest_and_size(path)
        raise ControlPlaneError(f"artifact is neither a file nor directory: {path}")

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

    def _obligation_projection(
        self,
        claims: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        *,
        pinned_at: str | None = None,
        static_artifact_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if static_artifact_ids is None:
            static_ids = {
                str(artifact.get("id"))
                for artifact in artifacts.get("artifacts", [])
                if artifact.get("producer_stage_id") is None
                and artifact.get("controller_producer") is None
                and artifact.get("status") == "verified"
            }
        else:
            static_ids = {str(identifier) for identifier in static_artifact_ids}
        claim_rows = sorted(
            (
                {
                    "id": str(claim.get("id")),
                    "type": str(claim.get("type")),
                    "statement": claim.get("statement"),
                    "gate_refs": sorted(str(item) for item in claim.get("gate_refs", [])),
                }
                for claim in claims.get("claims", [])
            ),
            key=lambda item: item["id"],
        )
        artifact_rows = sorted(
            (
                {
                    "id": str(artifact.get("id")),
                    "path": artifact.get("path"),
                    "role": artifact.get("role"),
                    "ownership": artifact.get("ownership"),
                    "required": bool(artifact.get("required")),
                    "producer_stage_id": artifact.get("producer_stage_id"),
                    "controller_producer": artifact.get("controller_producer"),
                    "retention": artifact.get("retention"),
                    "media_type": artifact.get("media_type"),
                    "verification": artifact.get("verification"),
                    "locked_status": (
                        artifact.get("status")
                        if str(artifact.get("id")) in static_ids
                        else None
                    ),
                    "locked_sha256": (
                        artifact.get("sha256")
                        if str(artifact.get("id")) in static_ids
                        else None
                    ),
                    "locked_size_bytes": (
                        artifact.get("size_bytes")
                        if str(artifact.get("id")) in static_ids
                        else None
                    ),
                }
                for artifact in artifacts.get("artifacts", [])
            ),
            key=lambda item: item["id"],
        )
        return {
            "claims_sha256": _sha256_bytes(
                _canonical_json(claim_rows).encode("utf-8")
            ),
            "artifacts_sha256": _sha256_bytes(
                _canonical_json(artifact_rows).encode("utf-8")
            ),
            "claim_ids": [item["id"] for item in claim_rows],
            "claim_types": sorted({item["type"] for item in claim_rows}),
            "artifact_ids": [item["id"] for item in artifact_rows],
            "required_artifact_ids": [
                item["id"] for item in artifact_rows if item["required"]
            ],
            "static_artifact_ids": sorted(static_ids),
            "pinned_at": pinned_at or _utc_now(),
        }

    def _projection_matches(
        self,
        pinned: Mapping[str, Any],
        claims: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> bool:
        current = self._obligation_projection(
            claims,
            artifacts,
            pinned_at=str(pinned.get("pinned_at") or _utc_now()),
            static_artifact_ids=pinned.get("static_artifact_ids", []),
        )
        return all(current.get(key) == pinned.get(key) for key in current if key != "pinned_at")

    def _declared_gate_refs(self, plan: Mapping[str, Any]) -> set[str]:
        """Return stable gate identifiers explicitly declared by the plan."""
        refs: set[str] = set()
        evaluation = plan.get("evaluation", {})
        if isinstance(evaluation, Mapping):
            for location, value in _walk_strings(evaluation, "evaluation"):
                # String leaves are descriptive values, not identifiers.
                del value
                parent = location.rsplit("/", 1)[0].replace("/", ".")
                if parent:
                    refs.add(parent)

            def visit_gate_tree(value: Any, prefix: str) -> None:
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        dotted = f"{prefix}.{key}"
                        key_text = str(key).lower()
                        if (
                            "gate" in key_text
                            or key_text.endswith("_go")
                            or prefix != "evaluation"
                        ):
                            refs.add(dotted)
                        visit_gate_tree(child, dotted)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping) and item.get("id"):
                            refs.add(f"{prefix}.{item['id']}")

            visit_gate_tree(evaluation, "evaluation")
        data = plan.get("data", {})
        if isinstance(data, Mapping):
            for key in data:
                if "access" in str(key).lower() or "gate" in str(key).lower():
                    refs.add(f"data.{key}")
        for stage in plan.get("stages", []):
            stage_id = str(stage.get("id"))
            group = f"stages.{stage_id}.acceptance_checks"
            refs.add(group)
            for check in stage.get("acceptance_checks", []):
                if isinstance(check, Mapping):
                    identifier = check.get("id") or check.get("name")
                else:
                    identifier = check
                if identifier:
                    refs.add(str(identifier))
                    refs.add(f"{group}.{identifier}")
        explicit = plan.get("gate_ids", [])
        if isinstance(explicit, list):
            refs.update(str(item) for item in explicit)
        return refs

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

    def _validate_plan_history_archives(
        self,
        experiment_id: str,
        state: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        state_path: Path,
    ) -> list[Issue]:
        """Verify replan history against immutable archived bytes and ledger events."""

        issues: list[Issue] = []
        shown = self.relative_display(state_path)
        raw_history = state.get("plan_history", [])
        if not isinstance(raw_history, list):
            return issues
        history = [row for row in raw_history if isinstance(row, Mapping)]
        if len(history) != len(raw_history):
            return issues  # The state schema reports malformed entries precisely.

        for index, row in enumerate(history):
            revision = row.get("plan_revision")
            plan_hash = row.get("plan_sha256")
            if not isinstance(revision, int) or not isinstance(plan_hash, str):
                continue
            expected_revision = index + 1
            if revision != expected_revision:
                issues.append(
                    Issue(
                        "plan_history_lineage",
                        shown,
                        f"history entry {index} records revision {revision}; expected {expected_revision}",
                    )
                )
            archive_root = (
                self.runtime_control_root
                / "replans"
                / experiment_id
                / f"revision_{revision}_{plan_hash[:12]}"
            ).resolve(strict=False)
            file_contracts = (
                (
                    "plan",
                    "archived_plan_path",
                    "archived_plan_sha256",
                    "archived_plan_size_bytes",
                    archive_root / "experiment.yaml",
                ),
                (
                    "state",
                    "archived_state_path",
                    "archived_state_sha256",
                    "archived_state_size_bytes",
                    archive_root / "state.json",
                ),
                (
                    "result",
                    "archived_result_path",
                    "archived_result_sha256",
                    "archived_result_size_bytes",
                    archive_root / "result.json",
                ),
                (
                    "claims",
                    "archived_claims_path",
                    "previous_claims_sha256",
                    "previous_claims_size_bytes",
                    archive_root / "claims.yaml",
                ),
                (
                    "artifacts",
                    "archived_artifacts_path",
                    "previous_artifacts_sha256",
                    "previous_artifacts_size_bytes",
                    archive_root / "artifacts" / "manifest.json",
                ),
            )
            for label, path_key, hash_key, size_key, expected_path in file_contracts:
                raw_path = row.get(path_key)
                expected_hash = row.get(hash_key)
                expected_size = row.get(size_key)
                if label == "result" and raw_path is None:
                    if expected_hash is not None or expected_size is not None:
                        issues.append(
                            Issue(
                                "plan_history_archive_contract",
                                shown,
                                "archived result path/hash/size must all be null or all be populated",
                            )
                        )
                    continue
                if not isinstance(raw_path, str):
                    continue  # Schema validation owns missing/wrong-type diagnostics.
                resolved = self.resolve_declared_path(raw_path)
                if resolved != expected_path.resolve(strict=False):
                    issues.append(
                        Issue(
                            "plan_history_archive_path",
                            shown,
                            f"archived {label} path is outside its canonical revision directory",
                        )
                    )
                    continue
                if not resolved.is_file():
                    issues.append(
                        Issue(
                            "plan_history_archive_missing",
                            self.relative_display(resolved),
                            f"archived {label} bytes are missing",
                        )
                    )
                    continue
                actual_size = resolved.stat().st_size
                actual_hash = self._hash_file_cached(resolved)
                if expected_size != actual_size:
                    issues.append(
                        Issue(
                            "plan_history_archive_size",
                            self.relative_display(resolved),
                            f"archived {label} size mismatch: {actual_size}",
                        )
                    )
                if expected_hash != actual_hash:
                    issues.append(
                        Issue(
                            "plan_history_archive_hash",
                            self.relative_display(resolved),
                            f"archived {label} SHA-256 mismatch: {actual_hash}",
                        )
                    )

            archived_state_path = row.get("archived_state_path")
            if isinstance(archived_state_path, str):
                resolved_state = self.resolve_declared_path(archived_state_path)
                if resolved_state.is_file():
                    try:
                        archived_state = _read_json(resolved_state)
                    except ControlPlaneError as exc:
                        issues.append(
                            Issue(
                                "plan_history_archive_state",
                                self.relative_display(resolved_state),
                                str(exc),
                            )
                        )
                    else:
                        if (
                            archived_state.get("plan_revision") != revision
                            or archived_state.get("plan_sha256") != plan_hash
                        ):
                            issues.append(
                                Issue(
                                    "plan_history_archive_state",
                                    self.relative_display(resolved_state),
                                    "archived state does not identify the archived plan revision",
                                )
                            )
                        archived_budget = archived_state.get("budget_consumed", {})
                        try:
                            normalized_budget = {
                                key: float(archived_budget.get(key, 0.0))
                                for key in ("gpu_hours", "cpu_hours", "storage_gib")
                            }
                        except (AttributeError, TypeError, ValueError):
                            normalized_budget = None
                        if normalized_budget != row.get("previous_budget_consumed"):
                            issues.append(
                                Issue(
                                    "plan_history_budget",
                                    self.relative_display(resolved_state),
                                    "previous budget totals do not match archived state bytes",
                                )
                            )

            matching_events = [
                event
                for event in events
                if event.get("event_type") == "plan_revised"
                and event.get("payload", {}).get("old_plan_revision") == revision
                and event.get("payload", {}).get("old_plan_sha256") == plan_hash
            ]
            if len(matching_events) != 1:
                issues.append(
                    Issue(
                        "plan_history_event",
                        shown,
                        f"revision {revision} requires exactly one matching plan_revised event",
                    )
                )
                continue
            event_payload = matching_events[0].get("payload", {})
            shared_fields = (
                "previous_plan_source",
                "archived_state_path",
                "archived_plan_path",
                "archived_plan_sha256",
                "archived_plan_size_bytes",
                "archived_state_sha256",
                "archived_state_size_bytes",
                "archived_result_path",
                "archived_result_sha256",
                "archived_result_size_bytes",
                "archived_claims_path",
                "archived_artifacts_path",
                "previous_claims_sha256",
                "previous_claims_size_bytes",
                "previous_artifacts_sha256",
                "previous_artifacts_size_bytes",
                "current_claims_sha256",
                "current_artifacts_sha256",
                "previous_budget_consumed",
                "carried_budget_consumed",
            )
            divergent = [
                field for field in shared_fields if event_payload.get(field) != row.get(field)
            ]
            successor_hash = (
                history[index + 1].get("plan_sha256")
                if index + 1 < len(history)
                else state.get("plan_sha256")
            )
            if event_payload.get("new_plan_revision") != revision + 1:
                divergent.append("new_plan_revision")
            if event_payload.get("new_plan_sha256") != successor_hash:
                divergent.append("new_plan_sha256")
            if divergent:
                issues.append(
                    Issue(
                        "plan_history_event",
                        shown,
                        "history/event mismatch in " + ", ".join(sorted(set(divergent))),
                    )
                )

        if history:
            expected_current_revision = int(history[-1]["plan_revision"]) + 1
            if state.get("plan_revision") != expected_current_revision:
                issues.append(
                    Issue(
                        "plan_history_lineage",
                        shown,
                        f"current revision must be {expected_current_revision} after recorded replans",
                    )
                )
            carried = history[-1].get("carried_budget_consumed", {})
            current = state.get("budget_consumed", {})
            try:
                budget_regressed = any(
                    float(current.get(key, 0.0)) + 1e-12
                    < float(carried.get(key, 0.0))
                    for key in ("gpu_hours", "cpu_hours", "storage_gib")
                )
            except (AttributeError, TypeError, ValueError):
                budget_regressed = False  # State schema reports malformed totals.
            if budget_regressed:
                issues.append(
                    Issue(
                        "plan_history_budget",
                        shown,
                        "current resource totals are below the most recent carried totals",
                    )
                )
        return issues

    def _validate_failure_evidence_archives(
        self,
        experiment_id: str,
        state: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        state_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        expected_root = (
            self.runtime_control_root / "evidence" / experiment_id
        ).resolve(strict=False)
        for stage_state in state.get("stages", []):
            if not isinstance(stage_state, Mapping):
                continue
            stage_id = str(stage_state.get("id"))
            for raw_path in stage_state.get("failure_evidence_refs", []):
                path = self.resolve_declared_path(str(raw_path))
                shown = self.relative_display(path)
                if not _is_within(path, expected_root):
                    issues.append(
                        Issue(
                            "failure_evidence_path",
                            self.relative_display(state_path),
                            f"{stage_id} failure evidence is outside its controller archive",
                        )
                    )
                    continue
                if not path.is_file():
                    issues.append(
                        Issue(
                            "failure_evidence_missing",
                            shown,
                            f"archived failure evidence for {stage_id} is missing",
                        )
                    )
                    continue
                matching_events = [
                    event
                    for event in events
                    if event.get("payload", {}).get("failure_evidence", {}).get("path")
                    == raw_path
                    and event.get("payload", {}).get("stage_id") == stage_id
                ]
                if len(matching_events) != 1:
                    issues.append(
                        Issue(
                            "failure_evidence_event",
                            shown,
                            "failure evidence requires exactly one matching failure event",
                        )
                    )
                    continue
                payload = matching_events[0]["payload"]
                recorded = payload["failure_evidence"]
                actual_hash = self._hash_file_cached(path)
                actual_size = path.stat().st_size
                if (
                    recorded.get("sha256") != actual_hash
                    or recorded.get("size_bytes") != actual_size
                ):
                    issues.append(
                        Issue(
                            "failure_evidence_hash",
                            shown,
                            "failure evidence bytes differ from the failure event",
                        )
                    )
                try:
                    evidence = _read_json(path)
                except ControlPlaneError as exc:
                    issues.append(Issue("failure_evidence_read", shown, str(exc)))
                    continue
                evidence_resources = evidence.get("metrics", {}).get("resources")
                if (
                    evidence.get("status") != "failed"
                    or evidence.get("experiment_id") != experiment_id
                    or evidence.get("stage_id") != stage_id
                    or recorded.get("resources") != evidence_resources
                    or payload.get("failed_attempt_resources") != evidence_resources
                ):
                    issues.append(
                        Issue(
                            "failure_evidence_contract",
                            shown,
                            "archived failure identity/resources differ from the failure event",
                        )
                    )
        return issues

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

            retry = stage.get("retry", {})
            max_attempts = retry.get("max_attempts", 1)
            if (
                isinstance(max_attempts, bool)
                or not isinstance(max_attempts, int)
                or max_attempts < 1
            ):
                issues.append(
                    Issue("retry_contract", shown, f"{stage_id} max_attempts must be >=1")
                )
            policy_class = retry.get("class")
            if policy_class is not None and policy_class not in RETRY_POLICY_CLASSES:
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} names unknown retry class {policy_class}",
                    )
                )
            explicit_classes = retry.get("retryable_classes", [])
            if not isinstance(explicit_classes, list) or any(
                not isinstance(value, str) or not value
                for value in explicit_classes
            ):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} retryable_classes must be a list of names",
                    )
                )
            elif any(
                value in CONTROLLER_NON_RETRYABLE_CLASSES
                or "hash" in value
                or "geometry" in value
                or value.startswith("completion_")
                for value in explicit_classes
            ):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} retryable_classes includes a controller-nonretryable failure",
                    )
                )
            if isinstance(max_attempts, int) and max_attempts > 1 and not (
                policy_class or explicit_classes
            ):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} multiple attempts require class or retryable_classes",
                    )
                )
            backoffs = retry.get("backoff_seconds", [])
            if not isinstance(backoffs, list) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in backoffs
            ):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} backoff_seconds must be nonnegative integers",
                    )
                )
            if isinstance(max_attempts, int) and len(backoffs) > max(max_attempts - 1, 0):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} declares more backoffs than retry transitions",
                    )
                )
            resource_timeout = retry.get("resource_wait_timeout_seconds")
            if resource_timeout is not None and (
                isinstance(resource_timeout, bool)
                or not isinstance(resource_timeout, int)
                or resource_timeout <= 0
            ):
                issues.append(
                    Issue(
                        "retry_contract",
                        shown,
                        f"{stage_id} resource_wait_timeout_seconds must be positive",
                    )
                )

        t4_stage_id = (entry.get("promotion") or {}).get("t4_stage_id")
        for stage in stage_list:
            stage_id = str(stage.get("id"))
            val120_refs = [
                (location, text)
                for source_name, source in (
                    ("inputs", stage.get("inputs", [])),
                    ("command", stage.get("command", {})),
                )
                for location, text in _walk_strings(source, source_name)
                if "val120" in text.lower() or "internal_replication" in text.lower()
            ]
            asd000_provenance_exception = (
                str(entry.get("id")) == "asd_000_evidence_lock_error_atlas"
                and stage_id == "lock_cohorts_and_prompt_ontology"
            )
            if val120_refs and stage_id != t4_stage_id and not asd000_provenance_exception:
                locations = ", ".join(location for location, _ in val120_refs)
                issues.append(
                    Issue(
                        "confirmatory_leakage",
                        shown,
                        f"pre-T4 stage {stage_id} references val120/internal replication at {locations}",
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
                completion = stage.get("completion")
                if not isinstance(completion, Mapping) or not any(
                    completion.get(key)
                    for key in ("evidence_file", "marker", "process_exit_only")
                ):
                    issues.append(
                        Issue(
                            "command_completion",
                            shown,
                            f"stage {stage_id} needs evidence_file, marker, or explicit process_exit_only",
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
            resolved_stage_writes: list[Path] = []
            for raw_path in stage.get("allowed_write_paths", []):
                path = self.resolve_declared_path(str(raw_path))
                resolved_stage_writes.append(path)
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
                if _is_within(path, runtime_base) and not _is_within(
                    path, runtime_base / experiment_id
                ):
                    issues.append(
                        Issue(
                            "cross_experiment_runtime_write",
                            shown,
                            f"{stage_id} may not write another experiment runtime: {raw_path}",
                        )
                    )
            for output in stage.get("outputs", []):
                raw_output = self._output_path(output)
                if raw_output is None:
                    continue
                output_path = self.resolve_declared_path(raw_output)
                if not any(_is_within(output_path, root) for root in resolved_stage_writes):
                    issues.append(
                        Issue(
                            "output_write_scope",
                            shown,
                            f"{stage_id} output is outside its allowed writes: {raw_output}",
                        )
                    )
            completion = stage.get("completion")
            if isinstance(completion, Mapping):
                completion_path = completion.get("evidence_file") or completion.get(
                    "marker"
                )
                if completion_path:
                    resolved_completion = self.resolve_declared_path(
                        str(completion_path)
                    )
                    if not any(
                        _is_within(resolved_completion, root)
                        for root in resolved_stage_writes
                    ):
                        issues.append(
                            Issue(
                                "completion_write_scope",
                                shown,
                                f"{stage_id} completion path is outside its allowed writes: {completion_path}",
                            )
                        )

        if state.get("status") != "draft":
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
                lowered = text.lower()
                if (
                    "val120" in lowered
                    and "inaccessible" in lowered
                    and "supported executor" not in lowered
                ):
                    issues.append(
                        Issue(
                            "protected_data_overclaim",
                            f"{shown}#/{location}",
                            "val120 access control is supported-executor enforcement, not OS-level same-user secrecy",
                        )
                    )
        return issues

    def _validate_hardware_semantics(
        self,
        plan: Mapping[str, Any],
        profile: Mapping[str, Any],
        plan_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        shown = self.relative_display(plan_path)
        max_gpus = int(profile.get("gpus", {}).get("count", 0))
        allowed_images = {
            str(value)
            for key, value in profile.get("gpu_access", {}).items()
            if key.endswith("_image") and value
        }
        image_ids = {
            str(profile.get("gpu_access", {}).get(prefix + "_image")): str(
                profile.get("gpu_access", {}).get(prefix + "_image_id")
            )
            for prefix in ("training", "anatomy")
            if profile.get("gpu_access", {}).get(prefix + "_image")
            and profile.get("gpu_access", {}).get(prefix + "_image_id")
        }
        stages = {str(stage.get("id")): stage for stage in plan.get("stages", [])}
        preflight_ids = {
            stage_id
            for stage_id, stage in stages.items()
            if "gpu_preflight" in stage_id
            or stage.get("kind") == "gpu_preflight"
        }

        def ancestors(stage_id: str) -> set[str]:
            found: set[str] = set()
            pending = list(stages.get(stage_id, {}).get("needs", []))
            while pending:
                candidate = str(pending.pop())
                if candidate in found:
                    continue
                found.add(candidate)
                pending.extend(stages.get(candidate, {}).get("needs", []))
            return found

        top_resources = plan.get("resources", {})
        auxiliary_classes = profile.get("auxiliary_compute_classes", {})
        top_resource_class = top_resources.get("resource_class")
        if top_resource_class and (
            not isinstance(auxiliary_classes, Mapping)
            or top_resource_class not in auxiliary_classes
        ):
            issues.append(
                Issue(
                    "hardware_resource_class",
                    shown,
                    f"plan names unknown resource_class {top_resource_class}",
                )
            )
        if top_resources.get("use_multi_gpu_ddp") is True:
            issues.append(Issue("hardware_policy", shown, "multi-GPU DDP is prohibited"))
        for location, text in _walk_strings(plan):
            lowered = text.lower()
            if ("command" in location or "resources" in location) and any(
                token in lowered for token in ("slurm", "h100", "h200")
            ):
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"prohibited scheduler/hardware assumption at {location}",
                    )
                )
        for stage_id, stage in stages.items():
            resources = stage.get("resources", {})
            resource_class = resources.get("resource_class")
            class_contract = (
                auxiliary_classes.get(resource_class)
                if isinstance(auxiliary_classes, Mapping) and resource_class
                else None
            )
            if resource_class and not isinstance(class_contract, Mapping):
                issues.append(
                    Issue(
                        "hardware_resource_class",
                        shown,
                        f"{stage_id} names unknown resource_class {resource_class}",
                    )
                )
            gpu_count = int(resources.get("gpu_count", 0) or 0)
            gpu_optional = resources.get("gpu_optional", False)
            if not isinstance(gpu_optional, bool):
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"{stage_id} gpu_optional must be boolean",
                    )
                )
                gpu_optional = False
            gpu_count_max = resources.get("gpu_count_max")
            if gpu_optional and (
                isinstance(gpu_count_max, bool)
                or not isinstance(gpu_count_max, int)
                or gpu_count_max != 1
            ):
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"{stage_id} gpu_optional requires gpu_count_max: 1",
                    )
                )
            if isinstance(class_contract, Mapping):
                if bool(class_contract.get("gpu_optional")) != bool(gpu_optional):
                    issues.append(
                        Issue(
                            "hardware_resource_class",
                            shown,
                            f"{stage_id} gpu_optional differs from {resource_class}",
                        )
                    )
                class_gpu_max = int(class_contract.get("gpu_count_max", 0))
                if int(resources.get("gpu_count_max", gpu_count) or 0) > class_gpu_max:
                    issues.append(
                        Issue(
                            "hardware_resource_class",
                            shown,
                            f"{stage_id} exceeds {resource_class} GPU-count cap",
                        )
                    )
                declared_class_hours = next(
                    (
                        float(resources[key])
                        for key in ("gpu_hours_total_max", "gpu_hours_max")
                        if key in resources
                    ),
                    None,
                )
                if (
                    declared_class_hours is None
                    or declared_class_hours
                    > float(class_contract.get("gpu_hours_total_max", 0))
                ):
                    issues.append(
                        Issue(
                            "hardware_resource_class",
                            shown,
                            f"{stage_id} must declare a GPU-hour cap within {resource_class}",
                        )
                    )
                classification = f"{stage_id} {stage.get('kind', '')}".lower()
                if class_contract.get("training_prohibited") and "training" in classification:
                    issues.append(
                        Issue(
                            "hardware_resource_class",
                            shown,
                            f"{stage_id} violates {resource_class} training prohibition",
                        )
                    )
            effective_gpu_count = 1 if gpu_optional else gpu_count
            if gpu_count < 0 or (gpu_optional and gpu_count > 1):
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"{stage_id} has an invalid gpu_count for its optional-GPU contract",
                    )
                )
            if effective_gpu_count > max_gpus:
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"{stage_id} may request {effective_gpu_count} GPUs but profile has {max_gpus}",
                    )
                )
            if resources.get("use_multi_gpu_ddp") is True or resources.get("ddp") is True:
                issues.append(
                    Issue("hardware_policy", shown, f"{stage_id} enables prohibited DDP")
                )
            image = resources.get("docker_image") or next(
                (
                    top_resources[key]
                    for key in ("docker_image", "training_image", "image")
                    if top_resources.get(key)
                ),
                None,
            )
            if image and str(image) not in allowed_images:
                issues.append(
                    Issue(
                        "hardware_policy",
                        shown,
                        f"{stage_id} docker image {image} is not hardware-profile locked",
                    )
                )
            if effective_gpu_count > 0:
                image_id = resources.get("docker_image_id") or next(
                    (
                        top_resources[key]
                        for key in (
                            "docker_image_id",
                            "docker_local_image_id",
                            "training_local_image_id",
                            "local_image_id",
                        )
                        if top_resources.get(key)
                    ),
                    None,
                )
                if not image or image_ids.get(str(image)) != str(image_id):
                    issues.append(
                        Issue(
                            "hardware_image_identity",
                            shown,
                            f"{stage_id} must pin the hardware-profile Docker tag and local image ID",
                        )
                    )
                classification = f"{stage_id} {stage.get('kind', '')}".lower()
                required_memory = 36 if (
                    "training" in classification or "probe_training" in classification
                ) else 32
                if isinstance(class_contract, Mapping):
                    required_memory = max(
                        required_memory,
                        int(class_contract.get("minimum_free_gib", required_memory)),
                    )
                declared_memory = next(
                    (
                        resources[key]
                        for key in (
                            "gpu_memory_free_gib_min_each",
                            "gpu_memory_free_gib_min",
                            "minimum_free_gib_per_gpu",
                        )
                        if key in resources
                    ),
                    None,
                )
                if declared_memory is None:
                    top_memory_keys = (
                        (
                            "gpu_memory_free_gib_min_training",
                            "minimum_free_gib_training",
                        )
                        if required_memory == 36
                        else (
                            "gpu_memory_free_gib_min_inference",
                            "gpu_memory_free_gib_min_feature_export",
                            "gpu_memory_free_gib_min",
                        )
                    )
                    declared_memory = next(
                        (
                            top_resources[key]
                            for key in top_memory_keys
                            if key in top_resources
                        ),
                        None,
                    )
                if (
                    isinstance(declared_memory, bool)
                    or not isinstance(declared_memory, (int, float))
                    or float(declared_memory) < required_memory
                ):
                    issues.append(
                        Issue(
                            "hardware_memory_floor",
                            shown,
                            f"{stage_id} requires at least {required_memory} GiB free per GPU",
                        )
                    )
            if effective_gpu_count > 0 and stage_id not in preflight_ids:
                explicit_ref = resources.get("preflight_ref") or top_resources.get(
                    "gpu_preflight_ref"
                )
                if not explicit_ref and not (ancestors(stage_id) & preflight_ids):
                    issues.append(
                        Issue(
                            "gpu_preflight",
                            shown,
                            f"GPU stage {stage_id} has no preceding preflight or explicit preflight_ref",
                        )
                    )
        return issues

    def _validate_matched_arm_contract(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        plan_path: Path,
    ) -> list[Issue]:
        promotion = entry.get("promotion") or {}
        if not promotion.get("candidate"):
            return []
        shown = self.relative_display(plan_path)
        treatment_id = promotion.get("treatment_arm_id")
        control_id = promotion.get("matched_control_arm_id")
        issues: list[Issue] = []
        if not treatment_id or not control_id or treatment_id == control_id:
            return [
                Issue(
                    "matched_arm_contract",
                    shown,
                    "promotion candidate requires distinct treatment_arm_id and matched_control_arm_id",
                )
            ]
        stage = next(
            (
                item
                for item in plan.get("stages", [])
                if item.get("id") == promotion.get("t4_stage_id")
            ),
            None,
        )
        if stage is None:
            return [Issue("matched_arm_contract", shown, "promotion T4 stage is missing")]
        contract = stage.get("matched_arms")
        if not isinstance(contract, Mapping):
            return [
                Issue(
                    "matched_arm_contract",
                    shown,
                    f"T4 stage {stage.get('id')} requires a structured matched_arms contract",
                )
            ]
        treatment = contract.get("treatment") or {}
        control = contract.get("control") or {}
        if treatment.get("id") != treatment_id or control.get("id") != control_id:
            issues.append(
                Issue(
                    "matched_arm_contract",
                    shown,
                    "T4 matched-arm IDs differ from portfolio promotion IDs",
                )
            )
        produced_ids = [
            treatment.get("checkpoint_output_id"),
            treatment.get("metrics_output_id"),
            control.get("checkpoint_output_id"),
            control.get("metrics_output_id"),
        ]
        if any(not item for item in produced_ids) or len(set(produced_ids)) != 4:
            issues.append(
                Issue(
                    "matched_arm_contract",
                    shown,
                    "treatment/control checkpoint and metric output IDs must all be distinct",
                )
            )
        declared_output_ids = {
            str(output.get("id")) if isinstance(output, Mapping) else str(output)
            for output in stage.get("outputs", [])
        }
        missing = sorted(str(item) for item in produced_ids if item not in declared_output_ids)
        if missing:
            issues.append(
                Issue(
                    "matched_arm_contract",
                    shown,
                    "T4 outputs omit matched-arm artifacts: " + ", ".join(missing),
                )
            )
        required_hash_fields = {
            "parent_checkpoint_sha256",
            "data_sha256",
            "schedule_sha256",
            "update_count",
        }
        if set(contract.get("matched_hash_fields", [])) != required_hash_fields:
            issues.append(
                Issue(
                    "matched_arm_contract",
                    shown,
                    "matched_hash_fields must exactly pin parent, data, schedule, and update count",
                )
            )
        return issues

    def _verify_matched_arm_evidence(
        self,
        entry: Mapping[str, Any],
        stage: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> None:
        promotion = entry.get("promotion") or {}
        if stage.get("id") != promotion.get("t4_stage_id"):
            return
        contract = stage.get("matched_arms") or {}
        attestation = evidence.get("metrics", {}).get("matched_arms")
        if not isinstance(attestation, Mapping):
            raise ControlPlaneError("T4 evidence requires metrics.matched_arms attestation")
        rows: list[Mapping[str, Any]] = []
        for role in ("treatment", "control"):
            expected = contract.get(role) or {}
            row = attestation.get(role)
            if not isinstance(row, Mapping):
                raise ControlPlaneError(f"T4 evidence omits matched arm {role}")
            if row.get("id") != expected.get("id"):
                raise ControlPlaneError(f"T4 {role} arm ID differs")
            for field in ("checkpoint_output_id", "metrics_output_id"):
                if row.get(field) != expected.get(field):
                    raise ControlPlaneError(f"T4 {role} {field} differs")
            for field in (
                "checkpoint_sha256",
                "metrics_sha256",
                "parent_checkpoint_sha256",
                "data_sha256",
                "schedule_sha256",
            ):
                if not SHA256_RE.fullmatch(str(row.get(field, ""))):
                    raise ControlPlaneError(f"T4 {role} {field} is not SHA-256")
            updates = row.get("update_count")
            if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
                raise ControlPlaneError(f"T4 {role} update_count must be positive")
            rows.append(row)
        for field in (
            "parent_checkpoint_sha256",
            "data_sha256",
            "schedule_sha256",
            "update_count",
        ):
            if rows[0].get(field) != rows[1].get(field):
                raise ControlPlaneError(f"T4 arms are not matched on {field}")

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
                if stage_plan.get("required") and stage_status != "passed":
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
                if claim.get("verdict") != "pending" and not claim.get(
                    "evidence_refs"
                ):
                    issues.append(
                        Issue(
                            "claim_evidence",
                            shown,
                            f"decided claim {claim.get('id')} needs evidence references",
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
        if status in ACTIVE_STATUSES:
            if current is None:
                issues.append(
                    Issue("active_stage", shown, f"{status} state requires a current stage")
                )
            elif status == "running" and state_stages.get(str(current), {}).get(
                "status"
            ) != "running":
                issues.append(
                    Issue(
                        "active_stage",
                        shown,
                        "running experiment must point to a running stage",
                    )
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
            lineage_hash = genesis_payload.get("plan_sha256")
            lineage_revision = 1
            for event in events[1:]:
                if event.get("event_type") != "plan_revised":
                    continue
                payload = event.get("payload", {})
                if payload.get("old_plan_sha256") != lineage_hash:
                    issues.append(
                        Issue(
                            "plan_lineage",
                            shown,
                            "plan_revised old hash does not continue event lineage",
                        )
                    )
                if payload.get("old_plan_revision") != lineage_revision:
                    issues.append(
                        Issue(
                            "plan_lineage",
                            shown,
                            "plan_revised old revision does not continue event lineage",
                        )
                    )
                lineage_hash = payload.get("new_plan_sha256")
                lineage_revision = payload.get("new_plan_revision")
            if lineage_hash != state.get("plan_sha256"):
                issues.append(
                    Issue(
                        "plan_lineage",
                        shown,
                        "event plan lineage does not end at current state hash",
                    )
                )
            if lineage_revision != state.get("plan_revision"):
                issues.append(
                    Issue(
                        "plan_lineage",
                        shown,
                        "event plan lineage does not end at current plan revision",
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
            latest_payload = events[-1].get("payload", {})
            for field in ("status", "outcome"):
                if field in latest_payload and latest_payload.get(field) != state.get(field):
                    issues.append(
                        Issue(
                            "event_state_mismatch",
                            shown,
                            f"latest event {field} does not match state",
                        )
                    )

        claim_ids = [str(claim.get("id")) for claim in claims.get("claims", [])]
        for repeated in sorted(_duplicates(claim_ids)):
            issues.append(Issue("duplicate_claim", shown, f"duplicate claim {repeated}"))
        passed_evidence_refs = {
            str(ref)
            for stage in state.get("stages", [])
            if stage.get("status") == "passed"
            for ref in stage.get("evidence_refs", [])
        }
        for claim in claims.get("claims", []):
            if claim.get("verdict") == "pending":
                continue
            if not claim.get("decided_at") or not claim.get("rationale") or not claim.get(
                "evidence_refs"
            ):
                issues.append(
                    Issue(
                        "claim_evidence",
                        shown,
                        f"decided claim {claim.get('id')} needs evidence refs, date, and rationale",
                    )
                )
            for ref in claim.get("evidence_refs", []):
                if str(ref) not in passed_evidence_refs:
                    issues.append(
                        Issue(
                            "claim_evidence",
                            shown,
                            f"claim {claim.get('id')} evidence {ref} is not passed-stage evidence",
                        )
                    )
        claim_types = {str(claim.get("type")) for claim in claims.get("claims", [])}
        missing_claim_types = sorted(CLAIM_TYPES - claim_types)
        if missing_claim_types:
            issues.append(
                Issue(
                    "claim_type_coverage",
                    shown,
                    "claims must predeclare every required type; missing "
                    + ", ".join(missing_claim_types),
                )
            )
        declared_gate_refs = self._declared_gate_refs(plan)
        for claim in claims.get("claims", []):
            for gate_ref in claim.get("gate_refs", []):
                if str(gate_ref) not in declared_gate_refs:
                    issues.append(
                        Issue(
                            "unknown_claim_gate",
                            shown,
                            f"claim {claim.get('id')} references undeclared gate {gate_ref}",
                        )
                    )
        projection = state.get("obligation_projection")
        if projection is not None and not self._projection_matches(
            projection, claims, artifacts
        ):
            issues.append(
                Issue(
                    "obligation_projection_drift",
                    shown,
                    "immutable claim or artifact obligations differ from the pinned projection",
                )
            )
        if status != "draft" and projection is None:
            issues.append(
                Issue(
                    "missing_obligation_projection",
                    shown,
                    "every non-draft state requires pinned claim/artifact obligations",
                )
            )
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
        artifact_paths: dict[str, list[str]] = {}
        for artifact in artifacts.get("artifacts", []):
            normalized = str(
                self.resolve_declared_path(str(artifact.get("path", "")))
            )
            artifact_paths.setdefault(normalized, []).append(str(artifact.get("id")))
        for normalized, identifiers in sorted(artifact_paths.items()):
            if len(identifiers) > 1:
                issues.append(
                    Issue(
                        "duplicate_artifact_path",
                        shown,
                        f"artifacts {', '.join(sorted(identifiers))} share resolved path {normalized}",
                    )
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
                if artifact.get("size_bytes") is None:
                    issues.append(
                        Issue(
                            "artifact_size",
                            shown,
                            f"verified artifact {artifact_id} has no size_bytes",
                        )
                    )
                elif path.exists():
                    try:
                        verification = artifact.get("verification") or {}
                        method = verification.get("method")
                        if method == "manifest_sha256":
                            manifest = self.resolve_declared_path(
                                str(verification["manifest_path"])
                            )
                            if not manifest.is_file():
                                raise ControlPlaneError(
                                    f"verification manifest is missing: {manifest}"
                                )
                            actual_hash = self._hash_file_cached(manifest)
                            actual_size = manifest.stat().st_size
                        elif path.is_dir():
                            if method not in {None, "tree_sha256"}:
                                raise ControlPlaneError(
                                    f"directory artifact requires tree_sha256 or manifest_sha256, got {method}"
                                )
                            actual_hash, actual_size = self._tree_digest_and_size(path)
                        else:
                            if method not in {None, "file_sha256"}:
                                raise ControlPlaneError(
                                    f"file artifact requires file_sha256 or manifest_sha256, got {method}"
                                )
                            actual_hash = self._hash_file_cached(path)
                            actual_size = path.stat().st_size
                    except (ControlPlaneError, OSError, KeyError) as exc:
                        issues.append(
                            Issue(
                                "artifact_hash",
                                shown,
                                f"cannot verify artifact {artifact_id}: {exc}",
                            )
                        )
                    else:
                        if actual_hash != expected_hash:
                            issues.append(
                                Issue(
                                    "artifact_hash",
                                    shown,
                                    f"artifact {artifact_id} hash mismatch; computed {actual_hash}",
                                )
                            )
                        expected_size = artifact.get("size_bytes")
                        if expected_size is not None and expected_size != actual_size:
                            issues.append(
                                Issue(
                                    "artifact_size",
                                    shown,
                                    f"artifact {artifact_id} size mismatch; computed {actual_size}",
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

    def _canonical_result_is_declared(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> bool:
        canonical = (
            self.resolve_declared_path(str(entry["path"]))
            / "results"
            / "result.json"
        ).resolve(strict=False)
        for artifact in artifacts.get("artifacts", []):
            if self.resolve_declared_path(str(artifact.get("path", ""))) == canonical:
                return True
        for stage in plan.get("stages", []):
            for output in stage.get("outputs", []):
                raw_path = self._output_path(output)
                if raw_path and self.resolve_declared_path(raw_path) == canonical:
                    return True
        return False

    def _validate_canonical_result_contract(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        plan_path: Path,
    ) -> list[Issue]:
        if not self._canonical_result_is_declared(entry, plan, artifacts):
            return []
        canonical = (
            self.resolve_declared_path(str(entry["path"]))
            / "results"
            / "result.json"
        ).resolve(strict=False)
        producer_ids = {
            str(artifact.get("producer_stage_id"))
            for artifact in artifacts.get("artifacts", [])
            if self.resolve_declared_path(str(artifact.get("path", ""))) == canonical
            and artifact.get("producer_stage_id") is not None
        }
        for stage in plan.get("stages", []):
            if any(
                (raw := self._output_path(output)) is not None
                and self.resolve_declared_path(raw) == canonical
                for output in stage.get("outputs", [])
            ):
                producer_ids.add(str(stage.get("id")))
        issues: list[Issue] = []
        if not producer_ids:
            issues.append(
                Issue(
                    "canonical_result_contract",
                    self.relative_display(plan_path),
                    "declared results/result.json must name a producer stage",
                )
            )
            return issues
        stage_map = {str(stage.get("id")): stage for stage in plan.get("stages", [])}
        for producer_id in sorted(producer_ids):
            stage = stage_map.get(producer_id)
            completion = stage.get("completion", {}) if stage else {}
            declared = (
                completion.get("evidence_file") or completion.get("marker")
                if isinstance(completion, Mapping)
                else None
            )
            if not declared or self.resolve_declared_path(str(declared)) != canonical:
                issues.append(
                    Issue(
                        "canonical_result_contract",
                        self.relative_display(plan_path),
                        f"stage {producer_id} produces results/result.json but uses different completion evidence",
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
        if (
            lease.get("container_label_key") != DOCKER_LEASE_LABEL_KEY
            or lease.get("container_label_value") != lease.get("lease_id")
            or executor.get("container_label_key") != DOCKER_LEASE_LABEL_KEY
            or executor.get("container_label_value") != lease.get("lease_id")
        ):
            issues.append(
                Issue(
                    "lease_identity",
                    self.relative_display(lease_path),
                    "lease/state Docker label does not exactly bind the lease_id",
                )
            )
        for key in (
            "hostname",
            "pid",
            "process_start_marker",
            "process_role",
            "container_label_key",
            "container_label_value",
            "heartbeat_at",
            "heartbeat_seconds",
            "ttl_seconds",
        ):
            if key not in lease:
                issues.append(
                    Issue(
                        "lease_structure",
                        self.relative_display(lease_path),
                        f"lease is missing required liveness field {key}",
                    )
                )
        try:
            if int(lease.get("heartbeat_seconds", 0)) <= 0 or int(
                lease.get("ttl_seconds", 0)
            ) <= 0:
                raise ValueError
            _parse_timestamp(str(lease.get("heartbeat_at")))
        except (TypeError, ValueError):
            issues.append(
                Issue(
                    "lease_structure",
                    self.relative_display(lease_path),
                    "lease heartbeat/TTL fields are invalid",
                )
            )
        return issues

    def _validate_passed_stage_evidence(
        self,
        entry: Mapping[str, Any],
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> list[Issue]:
        issues: list[Issue] = []
        plan_map, _ = self._stage_maps(plan, state)
        evidence_events: dict[tuple[str, str], Mapping[str, Any]] = {}
        for event in events:
            payload = event.get("payload", {})
            stage_id = payload.get("stage_id")
            evidence_path = payload.get("evidence_path")
            if stage_id and evidence_path:
                evidence_events[(str(stage_id), str(evidence_path))] = event
        for stage_state in state.get("stages", []):
            if stage_state.get("status") != "passed":
                continue
            stage_id = str(stage_state.get("id"))
            shown = self.relative_display(
                self.resolve_declared_path(str(entry["state_path"]))
            )
            refs = [str(item) for item in stage_state.get("evidence_refs", [])]
            if not refs:
                issues.append(
                    Issue(
                        "passed_stage_evidence",
                        shown,
                        f"passed stage {stage_id} has no evidence reference",
                    )
                )
                continue
            for ref in refs:
                event = evidence_events.get((stage_id, ref))
                if event is None:
                    issues.append(
                        Issue(
                            "passed_stage_event",
                            shown,
                            f"passed stage {stage_id} evidence {ref} has no completion event",
                        )
                    )
                    continue
                path = self.resolve_declared_path(ref)
                if not path.is_file():
                    issues.append(
                        Issue(
                            "passed_stage_evidence",
                            self.relative_display(path),
                            "passed-stage evidence file is missing",
                        )
                    )
                    continue
                actual_hash = self._hash_file_cached(path)
                expected_hash = event.get("payload", {}).get("evidence_sha256")
                if expected_hash != actual_hash:
                    issues.append(
                        Issue(
                            "passed_stage_evidence_hash",
                            self.relative_display(path),
                            f"completion event hash differs; computed {actual_hash}",
                        )
                    )
                    continue
                try:
                    evidence = _read_json(path)
                    self._verify_completion_evidence(
                        entry,
                        plan_map[stage_id],
                        evidence,
                        path,
                        state=state,
                    )
                except (ControlPlaneError, KeyError) as exc:
                    issues.append(
                        Issue(
                            "passed_stage_evidence",
                            self.relative_display(path),
                            str(exc),
                        )
                    )
        return issues

    def _validate_result_state_consistency(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
        result: Mapping[str, Any],
        result_path: Path,
    ) -> list[Issue]:
        issues: list[Issue] = []
        state_map = {
            str(item.get("id")): item for item in state.get("stages", [])
        }
        stage_id = str(result.get("stage_id"))
        stage_state = state_map.get(stage_id)
        if stage_state is None:
            return [
                Issue(
                    "result_state_mismatch",
                    self.relative_display(result_path),
                    f"result names unknown state stage {stage_id}",
                )
            ]
        if result.get("sentinel"):
            for field, expected in (
                ("plan_sha256", state.get("plan_sha256")),
                ("plan_revision", state.get("plan_revision")),
                ("state_revision", state.get("revision")),
                ("state_status", state.get("status")),
            ):
                if result.get(field) != expected:
                    issues.append(
                        Issue(
                            "result_state_mismatch",
                            self.relative_display(result_path),
                            f"sentinel {field} conflicts with state",
                        )
                    )
            if result.get("status") not in {"ready", "blocked"}:
                issues.append(
                    Issue(
                        "result_state_mismatch",
                        self.relative_display(result_path),
                        "sentinel result status must be ready or blocked",
                    )
                )
            return issues
        allowed = {
            "ready": {"ready", "blocked"},
            "passed": {"passed"},
            "failed": {"failed"},
            "blocked": {"blocked", "failed"},
        }.get(str(result.get("status")), set())
        if stage_state.get("status") not in allowed:
            issues.append(
                Issue(
                    "result_state_mismatch",
                    self.relative_display(result_path),
                    f"result status {result.get('status')} conflicts with stage status {stage_state.get('status')}",
                )
            )
        if stage_state.get("completed_at") and result.get("completed_at") != stage_state.get(
            "completed_at"
        ):
            issues.append(
                Issue(
                    "result_state_mismatch",
                    self.relative_display(result_path),
                    "result completed_at conflicts with state",
                )
            )
        if result.get("plan_sha256") is not None and result.get(
            "plan_sha256"
        ) != state.get("plan_sha256"):
            issues.append(
                Issue(
                    "result_state_mismatch",
                    self.relative_display(result_path),
                    "result plan_sha256 conflicts with state",
                )
            )
        return issues

    def _write_state_sentinel(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        stage_id: str,
        summary: str,
    ) -> Path:
        root = self.resolve_declared_path(str(entry["path"]))
        result_path = root / "results" / "result.json"
        _atomic_write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": entry["id"],
                "stage_id": stage_id,
                "plan_sha256": state["plan_sha256"],
                "plan_revision": state["plan_revision"],
                "state_revision": state["revision"],
                "state_status": state["status"],
                "sentinel": True,
                "status": "blocked" if state["status"] == "blocked" else "ready",
                "completed_at": state["updated_at"],
                "summary": summary,
                "checks": [],
                "artifacts": [],
                "metrics": {},
                "error": None,
            },
        )
        return result_path

    def _control_plane_source_paths(
        self, portfolio: Mapping[str, Any]
    ) -> list[Path]:
        paths: set[Path] = {
            self.portfolio_path.resolve(strict=False),
            (self.workspace_root / "tools" / "experimentctl.py").resolve(strict=False),
            self.resolve_declared_path(str(portfolio["hardware_profile_ref"])),
        }
        paths.update(path.resolve(strict=False) for path in self.schemas_root.glob("*.json"))
        for relative in (
            "templates/experiment/experiment.yaml",
            "templates/experiment/claims.yaml",
            "templates/experiment/state.json",
            "templates/experiment/artifacts/manifest.json",
            "templates/experiment/results/result.json",
        ):
            paths.add((self.workspace_root / relative).resolve(strict=False))
        contract_ref = portfolio.get("shared_training_contract_ref")
        if contract_ref:
            paths.add(self.resolve_declared_path(str(contract_ref)))
        else:
            fallback = self.workspace_root / "shared" / "manifests" / "planning_sources.yaml"
            if fallback.is_file():
                paths.add(fallback.resolve(strict=False))
        for entry in portfolio.get("experiments", []):
            root = self.resolve_declared_path(str(entry["path"]))
            paths.add(self.resolve_declared_path(str(entry["plan_path"])))
            protocol = root / "configs" / "protocol.yaml"
            if protocol.is_file():
                paths.add(protocol.resolve(strict=False))
        lock_ref = portfolio.get("control_plane_lock_ref")
        if lock_ref:
            paths.discard(self.resolve_declared_path(str(lock_ref)).resolve(strict=False))
        return sorted(paths, key=lambda path: self.relative_display(path))

    def _validate_control_plane_lock(
        self, portfolio: Mapping[str, Any]
    ) -> list[Issue]:
        lock_ref = portfolio.get("control_plane_lock_ref")
        if not lock_ref:
            return []
        path = self.resolve_declared_path(str(lock_ref))
        shown = self.relative_display(path)
        if not path.is_file():
            return [Issue("control_plane_lock", shown, "control-plane lock is missing")]
        try:
            lock = _read_json(path)
        except ControlPlaneError as exc:
            return [Issue("control_plane_lock", shown, str(exc))]
        issues: list[Issue] = []
        if lock.get("schema_version") != CONTROL_PLANE_LOCK_VERSION:
            issues.append(Issue("control_plane_lock", shown, "unsupported lock version"))
        if lock.get("registry_revision") != portfolio.get("registry_revision", 1):
            issues.append(Issue("control_plane_lock", shown, "registry revision differs"))
        expected_paths = self._control_plane_source_paths(portfolio)
        rows = lock.get("sources", [])
        row_map = {str(row.get("path")): row for row in rows if isinstance(row, Mapping)}
        expected_names = {self.relative_display(source) for source in expected_paths}
        if set(row_map) != expected_names:
            issues.append(
                Issue(
                    "control_plane_lock_sources",
                    shown,
                    "locked source set differs from canonical control-plane source set",
                )
            )
        for source in expected_paths:
            name = self.relative_display(source)
            row = row_map.get(name)
            if row is None:
                continue
            if not source.is_file():
                issues.append(Issue("control_plane_lock_source", name, "locked source is missing"))
                continue
            digest = self._hash_file_cached(source)
            size = source.stat().st_size
            if row.get("sha256") != digest or row.get("size_bytes") != size:
                issues.append(
                    Issue(
                        "control_plane_lock_drift",
                        name,
                        f"locked source hash/size differs; sha256={digest} size={size}",
                    )
                )
        obligation_rows = {
            str(row.get("experiment_id")): row
            for row in lock.get("obligations", [])
            if isinstance(row, Mapping)
        }
        expected_ids = {
            str(entry.get("id")) for entry in portfolio.get("experiments", [])
        }
        if set(obligation_rows) != expected_ids:
            issues.append(
                Issue(
                    "control_plane_lock_obligations",
                    shown,
                    "sealed obligation experiment set differs from portfolio",
                )
            )
        for entry in portfolio.get("experiments", []):
            experiment_id = str(entry["id"])
            row = obligation_rows.get(experiment_id)
            if row is None:
                continue
            _, _, state_path, claims_path, _, artifacts_path = self.experiment_paths(entry)
            try:
                state = _read_json(state_path)
                claims = _read_yaml(claims_path)
                artifacts = _read_json(artifacts_path)
                current = self._obligation_projection(
                    claims,
                    artifacts,
                    pinned_at=(state.get("obligation_projection") or {}).get("pinned_at"),
                    static_artifact_ids=(state.get("obligation_projection") or {}).get(
                        "static_artifact_ids", []
                    ),
                )
                projection = state.get("obligation_projection")
                digest = _sha256_bytes(_canonical_json(current).encode("utf-8"))
                if projection != current or row.get("projection_sha256") != digest:
                    issues.append(
                        Issue(
                            "control_plane_lock_obligation_drift",
                            experiment_id,
                            "claims/artifacts/state projection differs from sealed obligations",
                        )
                    )
            except ControlPlaneError as exc:
                issues.append(
                    Issue("control_plane_lock_obligation_drift", experiment_id, str(exc))
                )
        return issues

    def _validate_git_hygiene(self) -> list[Issue]:
        try:
            workspace_relative = self.workspace_root.relative_to(self.repo_root)
        except ValueError:
            return []
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--", workspace_relative.as_posix()],
            cwd=self.repo_root,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            return []
        forbidden_suffixes = (
            ".nii",
            ".nii.gz",
            ".npy",
            ".npz",
            ".pt",
            ".pth",
            ".ckpt",
            ".lock",
        )
        issues: list[Issue] = []
        for raw in completed.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
            if not raw:
                continue
            path = Path(raw)
            lowered_parts = {part.lower() for part in path.parts}
            lowered = raw.lower()
            resolved = self.repo_root / path
            forbidden = bool(
                lowered_parts & {".runtime", "runtime", "locks", "leases", "logits", "checkpoints"}
                or lowered.endswith(forbidden_suffixes)
                or resolved.is_symlink()
            )
            if forbidden:
                issues.append(
                    Issue(
                        "git_hygiene",
                        raw,
                        "runtime/heavy artifact, lock, medical array, checkpoint, or symlink is tracked",
                    )
                )
        return issues

    def _validate_promotion_integrity(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
        portfolio: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> list[Issue]:
        promotion = entry.get("promotion") or {}
        if not promotion.get("candidate") or not promotion.get("t4_stage_id"):
            return []
        state_map = {str(item.get("id")): item for item in state.get("stages", [])}
        t4_id = str(promotion["t4_stage_id"])
        if state_map.get(t4_id, {}).get("status") not in {"ready", "running", "passed"}:
            return []
        path = (
            self.resolve_declared_path(str(portfolio["runtime_root"]))
            / "promotions"
            / f"{entry['id']}.json"
        )
        shown = self.relative_display(path)
        if not path.is_file():
            return [Issue("promotion_record", shown, "T4 requires a promotion record")]
        try:
            record = _read_json(path)
        except ControlPlaneError as exc:
            return [Issue("promotion_record", shown, str(exc))]
        issues = self._schema_issues("promotion", record, path)
        body = dict(record)
        claimed_hash = body.pop("record_sha256", None)
        if claimed_hash != _sha256_bytes(_canonical_json(body).encode("utf-8")):
            issues.append(Issue("promotion_record_hash", shown, "record hash is invalid"))
        full_file_hash = self._hash_file_cached(path)
        if record.get("portfolio_sha256") != self._hash_file_cached(self.portfolio_path):
            issues.append(Issue("promotion_record", shown, "portfolio hash differs"))
        if record.get("registry_revision") != portfolio.get("registry_revision", 1):
            issues.append(Issue("promotion_record", shown, "registry revision differs"))
        if record.get("experiment_id") != entry.get("id"):
            issues.append(Issue("promotion_record_subject", shown, "experiment_id differs"))
        if record.get("source") != "experimentctl promote":
            issues.append(Issue("promotion_record_source", shown, "source differs"))
        if record.get("plan_sha256") != state.get("plan_sha256"):
            issues.append(Issue("promotion_record_plan", shown, "plan_sha256 differs"))
        if record.get("treatment_arm_id") != promotion.get("treatment_arm_id"):
            issues.append(Issue("promotion_record_control", shown, "treatment arm ID differs"))
        if record.get("matched_control_arm_id") != promotion.get("matched_control_arm_id"):
            issues.append(Issue("promotion_record_control", shown, "control arm ID differs"))
        selected_ids = record.get("selected_ids", [])
        selected = entry["id"] in selected_ids
        if not selected or len(selected_ids) > 2:
            issues.append(Issue("promotion_record", shown, "subject is not validly selected"))
        candidates = record.get("candidates", [])
        subject_rows = [row for row in candidates if row.get("id") == entry.get("id")]
        if len(subject_rows) != 1:
            issues.append(
                Issue("promotion_record_subject", shown, "candidate table must contain subject exactly once")
            )
            subject_row: Mapping[str, Any] = {}
        else:
            subject_row = subject_rows[0]
        if record.get("t3_evidence_sha256") != subject_row.get("t3_evidence_sha256"):
            issues.append(
                Issue("promotion_record_evidence", shown, "subject T3 evidence hash differs")
            )
        if record.get("rank") != subject_row.get("rank"):
            issues.append(Issue("promotion_record_ranking", shown, "subject rank differs"))
        if subject_row and subject_row.get("selected") is not selected:
            issues.append(Issue("promotion_record_subject", shown, "subject selected flag differs"))
        if record.get("matched_control_required") is not selected:
            issues.append(
                Issue(
                    "promotion_record_control",
                    shown,
                    "matched_control_required must be true exactly for selected subjects",
                )
            )
        if record.get("val120_access_authorized") is not selected:
            issues.append(
                Issue(
                    "promotion_record_access",
                    shown,
                    "val120_access_authorized must be true exactly for selected subjects",
                )
            )
        rankable = sorted(
            (
                row
                for row in candidates
                if row.get("eligible") is True and row.get("safety_pass") is True
            ),
            key=lambda row: (
                -float(row["dice_delta"]),
                -float(row["off_location_fp_reduction"]),
                float(row["gpu_hours"]),
                str(row["id"]),
            ),
        )
        expected_selected = [
            row["id"]
            for row in rankable[: int(record.get("policy", {}).get("max_t4_interventions", 0))]
        ]
        if selected_ids != expected_selected:
            issues.append(Issue("promotion_record_ranking", shown, "selected_ids do not prove ranking"))
        for row in candidates:
            evidence_path = row.get("t3_evidence_path")
            evidence_hash = row.get("t3_evidence_sha256")
            if evidence_path and evidence_hash:
                resolved = self.resolve_declared_path(str(evidence_path))
                if not resolved.is_file() or self._hash_file_cached(resolved) != evidence_hash:
                    issues.append(
                        Issue(
                            "promotion_evidence_hash",
                            shown,
                            f"candidate {row.get('id')} T3 evidence drifted",
                        )
                    )
        matching_events = [
            event
            for event in events
            if event.get("event_type") == "portfolio_promotion_decided"
            and event.get("payload", {}).get("record_path") == shown
            and event.get("payload", {}).get("record_sha256") == full_file_hash
            and event.get("payload", {}).get("selected") is True
        ]
        if not matching_events:
            issues.append(Issue("promotion_event", shown, "selected T4 lacks matching promotion event"))
        return issues

    @_hash_operation
    def validate_all(
        self, *, check_generated_views: bool = True, check_control_plane_lock: bool = True
    ) -> dict[str, Any]:
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
        issues.extend(self._validate_git_hygiene())

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
            hardware_profile: Mapping[str, Any] = {}
        else:
            try:
                loaded_hardware = _read_yaml(hardware_path)
                hardware_profile = (
                    loaded_hardware if isinstance(loaded_hardware, Mapping) else {}
                )
            except ControlPlaneError as exc:
                hardware_profile = {}
                issues.append(
                    Issue("hardware_profile_read", self.relative_display(hardware_path), str(exc))
                )
        if check_control_plane_lock:
            issues.extend(self._validate_control_plane_lock(portfolio))

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
                root / "results" / "result.json",
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
            issues.extend(
                self._validate_plan_history_archives(
                    experiment_id, state, events, state_path
                )
            )
            issues.extend(
                self._validate_failure_evidence_archives(
                    experiment_id, state, events, state_path
                )
            )
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
                    issues.extend(
                        self._validate_result_state_consistency(
                            entry, state, result, result_path
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
                self._validate_hardware_semantics(
                    plan, hardware_profile, plan_path
                )
            )
            issues.extend(self._validate_matched_arm_contract(entry, plan, plan_path))
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
            issues.extend(
                self._validate_canonical_result_contract(
                    entry, plan, artifacts, plan_path
                )
            )
            issues.extend(self._validate_lease_consistency(experiment_id, state))
            issues.extend(
                self._validate_passed_stage_evidence(
                    entry,
                    plan,
                    state,
                    event_sets.get(experiment_id, []),
                )
            )
            issues.extend(
                self._validate_promotion_integrity(
                    entry,
                    state,
                    portfolio,
                    event_sets.get(experiment_id, []),
                )
            )
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

        if check_generated_views:
            status_path = self.workspace_root / "STATUS.md"
            expected_status = self._status_markdown(portfolio)
            if not status_path.is_file():
                issues.append(
                    Issue(
                        "status_drift",
                        self.relative_display(status_path),
                        "generated STATUS.md is missing",
                    )
                )
            elif status_path.read_text(encoding="utf-8") != expected_status:
                issues.append(
                    Issue(
                        "status_drift",
                        self.relative_display(status_path),
                        "generated STATUS.md differs from canonical state",
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
        recovered = self._recover_expired_active_leases(portfolio)
        if recovered:
            report = self.validate_all()
            if not report["valid"]:
                raise ControlPlaneError(
                    "portfolio became invalid after stale-lease recovery"
                )
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

    def _process_liveness(self, lease: Mapping[str, Any]) -> bool | None:
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

    def _container_liveness(self, lease: Mapping[str, Any]) -> bool | None:
        """Return whether a running container owns the exact lease label.

        A failed Docker query is deliberately unknown rather than dead: stale
        recovery must never race an unobservable containerized worker.
        """
        if lease.get("hostname") != socket.gethostname():
            return None
        label_key = lease.get("container_label_key")
        label_value = lease.get("container_label_value")
        if label_key != DOCKER_LEASE_LABEL_KEY or label_value != lease.get("lease_id"):
            return None
        identifiers = self._labeled_container_ids(lease)
        return None if identifiers is None else bool(identifiers)

    def _labeled_container_ids(
        self, lease: Mapping[str, Any]
    ) -> list[str] | None:
        if lease.get("hostname") != socket.gethostname():
            return None
        label_key = lease.get("container_label_key")
        label_value = lease.get("container_label_value")
        if label_key != DOCKER_LEASE_LABEL_KEY or label_value != lease.get("lease_id"):
            return None
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"label={label_key}={label_value}",
                    "--format",
                    "{{.ID}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def _stop_labeled_containers(self, lease: Mapping[str, Any]) -> bool | None:
        identifiers = self._labeled_container_ids(lease)
        if identifiers is None:
            return None
        if not identifiers:
            return True
        try:
            completed = subprocess.run(
                ["docker", "stop", "--time", "10", *identifiers],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        remaining = self._labeled_container_ids(lease)
        return None if remaining is None else not remaining

    def _lease_liveness(self, lease: Mapping[str, Any]) -> bool | None:
        process_live = self._process_liveness(lease)
        container_live = self._container_liveness(lease)
        if process_live is True or container_live is True:
            return True
        if process_live is False and container_live is False:
            return False
        return None

    def _lease_is_stale(self, lease: Mapping[str, Any]) -> bool:
        heartbeat = _parse_timestamp(str(lease["heartbeat_at"]))
        ttl = int(lease["ttl_seconds"])
        return dt.datetime.now(dt.timezone.utc) > heartbeat + dt.timedelta(seconds=ttl)

    def _archive_stale_lease(
        self, lease_path: Path, lease: Mapping[str, Any], portfolio: Mapping[str, Any]
    ) -> Path:
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
        return archive_path

    def _recover_expired_active_leases(
        self, portfolio: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Recover only expired leases whose process and containers are proven dead."""
        recovered: list[dict[str, Any]] = []
        for entry in portfolio.get("experiments", []):
            experiment_id = str(entry["id"])
            with self.state_lock(experiment_id):
                _, plan_path, state_path, _, events_path, _ = self.experiment_paths(
                    entry
                )
                state = _read_json(state_path)
                if state.get("status") not in ACTIVE_STATUSES:
                    continue
                lease_path = self.lease_path(experiment_id)
                if not lease_path.is_file():
                    continue
                lease = _read_json(lease_path)
                if not self._lease_is_stale(lease):
                    continue
                liveness = self._lease_liveness(lease)
                if liveness is not False:
                    continue
                plan = _read_yaml(plan_path)
                self._assert_plan_identity(plan_path, plan, state)
                executor, _ = self._assert_execution_identity(experiment_id, state)
                stage_id = state.get("current_stage_id")
                plan_map, state_map = self._stage_maps(plan, state)
                if stage_id not in plan_map or stage_id not in state_map:
                    continue
                stage = plan_map[str(stage_id)]
                stage_state = state_map[str(stage_id)]
                archived = self._archive_stale_lease(lease_path, lease, portfolio)
                try:
                    max_attempts = int(stage.get("retry", {}).get("max_attempts", 1))
                    error = {
                        "class": "stale_lease",
                        "message": "expired executor lease was reclaimed after its process and labeled containers were proven dead",
                    }
                    stage_state["last_error"] = error
                    stage_state["completed_at"] = _utc_now()
                    state["executor"] = None
                    if "lease" in state:
                        state["lease"] = None
                    if int(stage_state.get("attempts", 0)) < max_attempts:
                        stage_state["status"] = "ready"
                        stage_state["started_at"] = None
                        stage_state["completed_at"] = None
                        state["status"] = "ready"
                        state["current_stage_id"] = stage_id
                        state["blockers"] = []
                        action = "retry"
                    else:
                        stage_state["status"] = "failed"
                        state["status"] = "blocked"
                        state["current_stage_id"] = None
                        state["blockers"] = [
                            {
                                "code": "stale_lease_attempts_exhausted",
                                "stage_id": stage_id,
                                "message": error["message"],
                            }
                        ]
                        action = "blocked"
                    state = self._commit_state_event(
                        state_path,
                        events_path,
                        state,
                        event_type="stale_lease_reclaimed",
                        actor="experimentctl",
                        payload={
                            "stage_id": stage_id,
                            "lease_id": executor.get("lease_id"),
                            "archived_lease_path": self.relative_display(archived),
                            "action": action,
                        },
                    )
                    self._write_state_sentinel(
                        entry,
                        state,
                        stage_id=str(stage_id),
                        summary="Expired executor lease reclaimed after process and container liveness verification.",
                    )
                except Exception:
                    if archived.exists() and not lease_path.exists():
                        os.replace(archived, lease_path)
                    raise
                recovered.append(
                    {
                        "id": experiment_id,
                        "stage_id": stage_id,
                        "action": action,
                        "revision": state["revision"],
                    }
                )
        if recovered:
            self._render_views_unchecked()
        return recovered

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
            "pid": os.getpid(),
            "process_start_marker": self._process_start_marker(os.getpid()),
            "process_role": "claim_controller",
            "container_label_key": DOCKER_LEASE_LABEL_KEY,
            "container_label_value": None,
            "claimed_at": now,
            "heartbeat_at": now,
            "heartbeat_seconds": int(
                portfolio["lease_policy"]["heartbeat_seconds"]
            ),
            "ttl_seconds": int(portfolio["lease_policy"]["ttl_seconds"]),
        }
        lease["container_label_value"] = lease["lease_id"]
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
        process_pid: int | None | object = _UNSET,
        process_role: str | None | object = _UNSET,
        t4_capability_sha256: str | None | object = _UNSET,
        promotion_record_sha256: str | None | object = _UNSET,
    ) -> dict[str, Any]:
        path = self.lease_path(experiment_id)
        lease = _read_json(path)
        if lease.get("lease_id") != lease_id:
            raise ControlPlaneError("executor lease changed")
        lease["heartbeat_at"] = _utc_now()
        if process_pid is not _UNSET:
            lease["pid"] = process_pid
            lease["process_start_marker"] = (
                self._process_start_marker(int(process_pid))
                if process_pid is not None
                else None
            )
        if process_role is not _UNSET:
            lease["process_role"] = process_role
        if t4_capability_sha256 is not _UNSET:
            lease["t4_capability_sha256"] = t4_capability_sha256
        if promotion_record_sha256 is not _UNSET:
            lease["promotion_record_sha256"] = promotion_record_sha256
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
        _, plan_path, state_path, claims_path, events_path, artifacts_path = (
            self.experiment_paths(entry)
        )
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            claims = _read_yaml(claims_path)
            artifacts = _read_json(artifacts_path)
            self._assert_plan_identity(plan_path, plan, state)
            projection = state.get("obligation_projection")
            if projection is None:
                state["obligation_projection"] = self._obligation_projection(
                    claims, artifacts
                )
            elif not self._projection_matches(projection, claims, artifacts):
                raise ControlPlaneError(
                    "immutable claim/artifact obligations drifted after pinning"
                )
            if state.get("status") != "blocked":
                raise ControlPlaneError(
                    "reopen-blocked requires blocked state, got "
                    f"{state.get('status')}"
                )
            if not state.get("blockers"):
                raise ControlPlaneError(
                    "reopen-blocked requires a canonical valid portfolio with an existing blocker"
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

            target_stage_id = self._blocked_stage_for_reopen(plan, state)
            validation = self.validate_all(check_generated_views=False)
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
            self._write_state_sentinel(
                entry,
                state,
                stage_id=target_stage_id,
                summary="Blocked stage reopened after an audited control-plane repair.",
            )
            self._render_views_unchecked()
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
            if not state.get("blockers"):
                raise ControlPlaneError(
                    "record-blocker requires a canonical valid portfolio with an existing blocker"
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
            result_path = self.resolve_declared_path(str(entry["path"])) / "results" / "result.json"
            if result_path.is_file() and _read_json(result_path).get("sentinel"):
                stage_id = state.get("current_stage_id") or next(
                    str(item["id"])
                    for item in state.get("stages", [])
                    if item.get("status") in {"blocked", "ready", "pending"}
                )
                self._write_state_sentinel(
                    entry,
                    state,
                    stage_id=stage_id,
                    summary="Blocked experiment sentinel updated with an audited blocker.",
                )
            self._render_views_unchecked()
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
        _, plan_path, state_path, claims_path, events_path, artifacts_path = (
            self.experiment_paths(entry)
        )
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            claims = _read_yaml(claims_path)
            artifacts = _read_json(artifacts_path)
            self._assert_plan_identity(plan_path, plan, state)
            projection = state.get("obligation_projection")
            if projection is None:
                state["obligation_projection"] = self._obligation_projection(
                    claims, artifacts
                )
            elif not self._projection_matches(projection, claims, artifacts):
                raise ControlPlaneError(
                    "immutable claim/artifact obligations drifted after pinning"
                )
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
                "container_label_key": lease["container_label_key"],
                "container_label_value": lease["container_label_value"],
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
            self._write_state_sentinel(
                entry,
                state,
                stage_id=str(state.get("current_stage_id")),
                summary="Experiment claimed; current stage has not completed.",
            )
            self._render_views_unchecked()
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
        agent_id: str,
        lease_id: str,
        stage: Mapping[str, Any],
        stage_state: Mapping[str, Any],
        t4_capability: str | None = None,
        t4_capability_sha256: str | None = None,
        promotion_record_sha256: str | None = None,
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
            "agent_id": agent_id,
            "stage_id": stage["id"],
            "attempt": stage_state["attempts"],
            "instructions": instruction_list,
            "allowed_write_paths": stage.get("allowed_write_paths", []),
            "inputs": stage.get("inputs", []),
            "outputs": stage.get("outputs", []),
            "acceptance_checks": stage.get("acceptance_checks", []),
            "stop_conditions": stage.get("stop_conditions", []),
            "resources": stage.get("resources", {}),
            "container_contract": {
                "required_label_key": DOCKER_LEASE_LABEL_KEY,
                "required_label_value": lease_id,
                "docker_run_arguments": [
                    "--label",
                    f"{DOCKER_LEASE_LABEL_KEY}={lease_id}",
                    "--label",
                    f"{DOCKER_EXPERIMENT_LABEL_KEY}={experiment_id}",
                ],
                "environment": {
                    COMMAND_EXPERIMENT_ENV: experiment_id,
                    COMMAND_LEASE_ENV: lease_id,
                    COMMAND_DOCKER_LABEL_ENV: f"{DOCKER_LEASE_LABEL_KEY}={lease_id}",
                },
                "requirement": "Every docker run started by this stage must carry the exact lease label.",
            },
            "protected_data_contract": {
                "scope": "supported_executor_only; not an OS secrecy boundary against same-user processes",
                "authorized": t4_capability is not None,
                "capability_env": T4_CAPABILITY_ENV,
                "capability": t4_capability,
                "capability_sha256": t4_capability_sha256,
                "promotion_record_sha256": promotion_record_sha256,
                "requirement": "Reject and log val120 access unless capability, lease, and promotion hashes all match.",
            },
            "timeout_seconds": stage.get("timeout_seconds"),
            "evidence_path": evidence_path,
            "completion_command": (
                "python experiments_aiselfdrive/tools/experimentctl.py "
                f"complete-stage --id {experiment_id} --agent-id {agent_id} --evidence "
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
        failure_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, _, state_path, _, events_path, _ = self.experiment_paths(entry)
        _, state_map = self._stage_maps(plan, state)
        stage_state = state_map[str(stage["id"])]
        retry = stage.get("retry", {})
        max_attempts = int(retry.get("max_attempts", 1))
        retryable_classes = {
            str(item) for item in retry.get("retryable_classes", [])
        }
        policy_class = retry.get("class")
        if policy_class is not None:
            retryable_classes.update(RETRY_POLICY_CLASSES.get(str(policy_class), set()))
        global_non_retryable = CONTROLLER_NON_RETRYABLE_CLASSES | {
            str(item)
            for item in self.load_portfolio()
            .get("retry_defaults", {})
            .get("non_retryable_classes", [])
        }
        stage_state["last_error"] = {"class": error_class, "message": message}
        failure_time = _utc_now()
        stage_state["completed_at"] = failure_time
        started_at = stage_state.get("started_at")
        elapsed_hours = 0.0
        if started_at:
            elapsed_hours = max(
                0.0,
                (_parse_timestamp(failure_time) - _parse_timestamp(str(started_at))).total_seconds()
                / 3600.0,
            )
        stage_resources = stage.get("resources", {})
        gpu_multiplier = int(stage_resources.get("gpu_count", 0) or 0)
        if stage_resources.get("gpu_optional") is True:
            gpu_multiplier = max(
                gpu_multiplier, int(stage_resources.get("gpu_count_max", 1) or 1)
            )
        cpu_multiplier = int(
            stage_resources.get("cpu_cores", stage_resources.get("cpu_threads", 1)) or 1
        )
        if failure_evidence is not None:
            declared_resources = failure_evidence.get("resources", {})
            failed_resources = {
                key: float(declared_resources[key])
                for key in ("gpu_hours", "cpu_hours", "storage_gib")
            }
            resource_basis = "declared_failure_evidence"
            failure_ref = str(failure_evidence["path"])
            failure_refs = stage_state.setdefault("failure_evidence_refs", [])
            if failure_ref not in failure_refs:
                failure_refs.append(failure_ref)
        else:
            failed_resources = {
                "gpu_hours": elapsed_hours * max(gpu_multiplier, 0),
                "cpu_hours": elapsed_hours * max(cpu_multiplier, 1),
                "storage_gib": float(
                    state.get("budget_consumed", {}).get("storage_gib", 0.0)
                ),
            }
            resource_basis = "conservative_elapsed_requested_resources"
        try:
            self._ingest_resource_metrics(
                plan, state, {"metrics": {"resources": failed_resources}}
            )
        except ControlPlaneError:
            error_class = "budget_exhausted"
            message = "failed attempt exhausted the declared cumulative budget"
            stage_state["last_error"] = {"class": error_class, "message": message}
        failure_payload = (
            {
                "path": failure_evidence["path"],
                "source_path": failure_evidence["source_path"],
                "sha256": failure_evidence["sha256"],
                "size_bytes": failure_evidence["size_bytes"],
                "resources": failed_resources,
            }
            if failure_evidence is not None
            else None
        )
        if stage.get("execution_mode") == "command":
            lease_path = self.lease_path(str(entry["id"]))
            lease = _read_json(lease_path) if lease_path.is_file() else None
            cleanup = self._stop_labeled_containers(lease) if lease else None
            if cleanup is not True:
                cleanup_message = (
                    "Docker liveness/cleanup could not prove all exact-lease containers stopped"
                )
                stage_state["status"] = "blocked"
                stage_state["last_error"] = {
                    "class": "container_cleanup_unknown",
                    "message": cleanup_message,
                }
                state["status"] = "stopping"
                state["blockers"] = [
                    {
                        "code": "container_cleanup_unknown",
                        "stage_id": stage["id"],
                        "message": cleanup_message,
                    }
                ]
                updated_state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="stage_stopping",
                    actor="experimentctl",
                    payload={
                        "stage_id": stage["id"],
                        "error_class": error_class,
                        "message": message,
                        "container_cleanup": "unknown",
                        "failed_attempt_resources": failed_resources,
                        "resource_basis": resource_basis,
                        "failure_evidence": failure_payload,
                    },
                )
                self._write_state_sentinel(
                    entry,
                    updated_state,
                    stage_id=str(stage["id"]),
                    summary=cleanup_message,
                )
                self._render_views_unchecked()
                return updated_state
        controller_forbidden = (
            error_class in global_non_retryable
            or "hash" in error_class
            or "geometry" in error_class
            or error_class.startswith("completion_")
        )
        retryable = (
            not controller_forbidden and error_class in retryable_classes
        )
        now = dt.datetime.now(dt.timezone.utc)
        backoff_seconds = 0
        if retryable:
            if error_class in {"resource_unavailable", "evidence_dependency_not_terminal"}:
                wait_seconds = int(
                    retry.get(
                        "resource_wait_seconds",
                        (retry.get("backoff_seconds") or [60])[0],
                    )
                )
                wait_attempts = int(retry.get("resource_wait_attempts", max_attempts))
                max_attempts = max(max_attempts, wait_attempts)
                started_text = stage_state.get("resource_wait_started_at")
                if started_text is None:
                    stage_state["resource_wait_started_at"] = _utc_now()
                    started = now
                else:
                    started = _parse_timestamp(str(started_text))
                wait_timeout = int(
                    retry.get(
                        "resource_wait_timeout_seconds",
                        stage.get("timeout_seconds", 1),
                    )
                )
                if (now - started).total_seconds() + wait_seconds > wait_timeout:
                    retryable = False
                    error_class = "resource_wait_timeout"
                    stage_state["last_error"] = {
                        "class": error_class,
                        "message": "declared resource wait timeout was exhausted",
                    }
                backoff_seconds = wait_seconds
            else:
                declared_backoffs = [
                    int(value) for value in retry.get("backoff_seconds", [])
                ]
                if declared_backoffs:
                    backoff_seconds = declared_backoffs[
                        min(max(int(stage_state["attempts"]) - 1, 0), len(declared_backoffs) - 1)
                    ]
        if retryable and stage_state["attempts"] < max_attempts:
            stage_state["status"] = "retry_wait"
            stage_state["retry_not_before"] = (
                now + dt.timedelta(seconds=backoff_seconds)
            ).isoformat().replace("+00:00", "Z")
            state["status"] = "retry_wait"
            event_type = "stage_retry_wait"
        else:
            stage_state["status"] = "failed"
            stage_state["retry_not_before"] = None
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
                "retryable": retryable,
                "retry_not_before": stage_state.get("retry_not_before"),
                "failed_attempt_resources": failed_resources,
                "resource_basis": resource_basis,
                "failure_evidence": failure_payload,
            },
        )
        if updated_state["status"] == "blocked":
            lease_path = self.lease_path(str(entry["id"]))
            if lease_path.exists():
                lease_path.unlink()
            result_path = (
                self.resolve_declared_path(str(entry["path"]))
                / "results"
                / "result.json"
            )
            failure_result = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": entry["id"],
                    "stage_id": stage["id"],
                    "attempt": stage_state["attempts"],
                    "plan_sha256": updated_state["plan_sha256"],
                    "status": "failed",
                    "completed_at": stage_state["completed_at"],
                    "summary": message or "stage failed",
                    "checks": [],
                    "artifacts": [],
                    "metrics": {},
                    "error": {"class": error_class, "message": message},
                }
            _, _, _, _, _, artifacts_path = self.experiment_paths(entry)
            artifacts = _read_json(artifacts_path)
            if self._canonical_result_is_declared(entry, plan, artifacts):
                rejected_path = (
                    self.runtime_control_root
                    / "evidence"
                    / str(entry["id"])
                    / f"{stage['id']}.rejected.json"
                )
                _atomic_write_json(rejected_path, failure_result)
            else:
                _atomic_write_json(result_path, failure_result)
        else:
            self._write_state_sentinel(
                entry,
                updated_state,
                stage_id=str(stage["id"]),
                summary="Stage is waiting for its declared retry backoff.",
            )
        self._render_views_unchecked()
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
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._refresh_lease(
                experiment_id,
                lease_id,
                process_pid=process.pid,
                process_role="command",
            )
            started = time.monotonic()
            while True:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
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
                        experiment_id,
                        lease_id,
                        process_pid=process.pid,
                        process_role="command",
                    )
        except BaseException:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    process.wait()
            raise
        finally:
            if self.lease_path(experiment_id).exists():
                try:
                    self._refresh_lease(
                        experiment_id,
                        lease_id,
                        process_pid=os.getpid(),
                        process_role="controller",
                    )
                except ControlPlaneError:
                    pass

    def _output_path(self, output: Any) -> str | None:
        if isinstance(output, Mapping):
            raw = output.get("path")
            return str(raw) if raw else None
        if isinstance(output, str):
            raw = output.strip()
            if not raw or any(character.isspace() for character in raw):
                return None
            if Path(raw).is_absolute() or raw.startswith(("./", "../")):
                return raw
            path_prefixes = (
                "experiments_aiselfdrive/",
                ".runtime/",
                "runtime/",
                "results/",
                "artifacts/",
                "configs/",
                "scripts/",
                "src/",
                "shared/",
            )
            if raw.startswith(path_prefixes):
                return raw
        return None

    def _artifact_evidence(self, identifier: str, path: Path) -> dict[str, Any]:
        digest, size = self._path_digest_and_size(path)
        return {
            "id": identifier,
            "path": self.relative_display(path),
            "sha256": digest,
            "size_bytes": size,
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
                "resources": {
                    "gpu_hours": 0.0,
                    "cpu_hours": 0.0,
                    "storage_gib": 0.0,
                },
            },
            "error": None,
        }

    def _declared_failed_command_evidence(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
        stage: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Load and archive a command's structured failed-result evidence.

        A nonzero process exit alone is deliberately coarse.  Commands may
        declare a precise operational failure class only through their normal
        result path, bound to the current plan and attempt.  The controller
        copies accepted bytes to an immutable per-attempt evidence path before
        making a retry decision.
        """

        completion = stage.get("completion", {})
        raw_path = None
        if isinstance(completion, Mapping):
            raw_path = completion.get("evidence_file") or completion.get("marker")
        if not raw_path:
            return None, None
        source_path = self.resolve_declared_path(str(raw_path))
        if not source_path.is_file():
            return None, f"declared failed evidence is missing: {source_path}"
        try:
            source_bytes = source_path.read_bytes()
            evidence = json.loads(source_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"cannot read declared failed evidence: {exc}"
        if not isinstance(evidence, Mapping):
            return None, "declared failed evidence must be a JSON object"
        schema_issues = self._schema_issues("result", evidence, source_path)
        if schema_issues:
            return None, f"declared failed evidence is invalid: {schema_issues[0].message}"
        _, state_map = self._stage_maps(
            {"stages": [stage]},
            {
                "stages": [
                    candidate
                    for candidate in state.get("stages", [])
                    if candidate.get("id") == stage.get("id")
                ]
            },
        )
        stage_state = state_map.get(str(stage.get("id")), {})
        required_identity = {
            "experiment_id": entry["id"],
            "stage_id": stage["id"],
            "attempt": stage_state.get("attempts"),
            "plan_sha256": state.get("plan_sha256"),
        }
        mismatches = [
            key for key, expected in required_identity.items() if evidence.get(key) != expected
        ]
        if mismatches:
            return None, "declared failed evidence identity mismatch in " + ", ".join(
                mismatches
            )
        if evidence.get("status") != "failed":
            return None, "nonzero command evidence must declare status=failed"
        error = evidence.get("error")
        error_class = error.get("class") if isinstance(error, Mapping) else None
        if not isinstance(error_class, str) or not STAGE_ID_RE.fullmatch(error_class):
            return None, "declared failed evidence requires a normalized error.class"
        resources = evidence.get("metrics", {}).get("resources")
        if not isinstance(resources, Mapping):
            return None, "declared failed evidence requires metrics.resources"
        normalized_resources: dict[str, float] = {}
        for key in ("gpu_hours", "cpu_hours", "storage_gib"):
            value = resources.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"declared failed evidence resources.{key} must be numeric"
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                return None, f"declared failed evidence resources.{key} is invalid"
            normalized_resources[key] = numeric
        completed_at = _parse_timestamp(str(evidence["completed_at"]))
        started_at = stage_state.get("started_at")
        if started_at and completed_at < _parse_timestamp(str(started_at)):
            return None, "declared failed evidence predates the active attempt"

        archive_path = (
            self.runtime_control_root
            / "evidence"
            / str(entry["id"])
            / f"{stage['id']}.attempt_{stage_state['attempts']}.failed.json"
        )
        if archive_path.exists():
            if not archive_path.is_file() or archive_path.read_bytes() != source_bytes:
                return None, "immutable failed-evidence archive path already differs"
        else:
            _atomic_write_bytes(archive_path, source_bytes)
        error_message = (
            str(error.get("message"))
            if isinstance(error, Mapping) and error.get("message")
            else str(evidence.get("summary"))
        )
        return (
            {
                "error_class": error_class,
                "message": error_message,
                "source_path": self.relative_display(source_path),
                "path": self.relative_display(archive_path),
                "sha256": _sha256_bytes(source_bytes),
                "size_bytes": len(source_bytes),
                "resources": normalized_resources,
            },
            None,
        )

    def _command_budget_timeout_seconds(
        self,
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        stage: Mapping[str, Any],
    ) -> int:
        """Cap a command attempt by its remaining cumulative compute budget."""

        declared_timeout = int(stage["timeout_seconds"])
        budget = plan.get("budget", {})
        consumed = state.get("budget_consumed", {})
        gpu_limit = next(
            (
                float(budget[key])
                for key in (
                    "total_gpu_hours_max",
                    "aggregate_gpu_hours_max",
                    "maximum_gpu_hours",
                    "gpu_hours_max",
                )
                if key in budget
            ),
            None,
        )
        resources = stage.get("resources", {})
        gpu_count = int(resources.get("gpu_count", 0) or 0)
        if resources.get("gpu_optional") is True:
            gpu_count = max(gpu_count, int(resources.get("gpu_count_max", 1) or 1))
        if gpu_limit is None or gpu_count <= 0:
            return declared_timeout
        remaining_gpu_hours = gpu_limit - float(consumed.get("gpu_hours", 0.0))
        budget_seconds = math.floor(remaining_gpu_hours * 3600.0 / gpu_count)
        if budget_seconds <= 0:
            raise ControlPlaneError(
                "cumulative GPU budget is exhausted; refusing another command attempt"
            )
        return min(declared_timeout, budget_seconds)

    def run_stage(self, experiment_id: str, agent_id: str) -> dict[str, Any]:
        if not agent_id.strip():
            raise ControlPlaneError("agent-id must be nonempty")
        entry = self.entry(experiment_id)
        _, plan_path, state_path, _, events_path, _ = self.experiment_paths(entry)
        with self.state_lock(experiment_id):
            plan = _read_yaml(plan_path)
            state = _read_json(state_path)
            self._assert_plan_identity(plan_path, plan, state)
            if state.get("status") == "running":
                raise ControlPlaneError(
                    "run-stage refuses duplicate execution of an already running stage"
                )
            if state.get("status") not in {"claimed", "retry_wait"}:
                raise ControlPlaneError(
                    f"run-stage requires claimed/retry_wait, got "
                    f"{state.get('status')}"
                )
            executor, _ = self._assert_execution_identity(experiment_id, state)
            if str(executor.get("agent_id")) != agent_id:
                raise ControlPlaneError("agent-id does not own this experiment")
            lease = self._refresh_lease(
                experiment_id,
                str(executor["lease_id"]),
                process_pid=os.getpid(),
                process_role="stage_controller",
            )
            stage_id = self._select_current_stage(plan, state)
            plan_map, state_map = self._stage_maps(plan, state)
            stage = plan_map[stage_id]
            stage_state = state_map[stage_id]
            command_timeout_seconds = int(stage["timeout_seconds"])
            if stage.get("execution_mode") == "command":
                command_timeout_seconds = self._command_budget_timeout_seconds(
                    plan, state, stage
                )
            protected_refs = [
                text
                for source in (stage.get("inputs", []), stage.get("command", {}))
                for _, text in _walk_strings(source)
                if "val120" in text.lower() or "internal_replication" in text.lower()
            ]
            provenance_exception = (
                experiment_id == "asd_000_evidence_lock_error_atlas"
                and stage_id == "lock_cohorts_and_prompt_ontology"
            )
            if (
                protected_refs
                and stage_id != (entry.get("promotion") or {}).get("t4_stage_id")
                and not provenance_exception
            ):
                raise ControlPlaneError(
                    "supported executor denies protected val120 artifact access before verified T4 promotion"
                )
            t4_capability: str | None = None
            t4_capability_sha256: str | None = None
            promotion_record_sha256: str | None = None
            if (entry.get("promotion") or {}).get("t4_stage_id") == stage_id:
                _, events = self._validate_events(events_path, experiment_id)
                promotion_issues = self._validate_promotion_integrity(
                    entry, state, self.load_portfolio(), events
                )
                if promotion_issues:
                    raise ControlPlaneError(
                        f"T4 promotion integrity failed: {promotion_issues[0].message}"
                    )
                promotion_path = (
                    self.resolve_declared_path(str(self.load_portfolio()["runtime_root"]))
                    / "promotions"
                    / f"{experiment_id}.json"
                )
                promotion_record_sha256 = self._hash_file_cached(promotion_path)
                t4_capability = secrets.token_urlsafe(32)
                t4_capability_sha256 = _sha256_bytes(
                    (
                        f"{executor['lease_id']}\0{promotion_record_sha256}\0{t4_capability}"
                    ).encode("utf-8")
                )
                lease = self._refresh_lease(
                    experiment_id,
                    str(executor["lease_id"]),
                    t4_capability_sha256=t4_capability_sha256,
                    promotion_record_sha256=promotion_record_sha256,
                )
                state["executor"]["t4_capability_sha256"] = t4_capability_sha256
                state["executor"]["promotion_record_sha256"] = promotion_record_sha256
                if "lease" in state and state["lease"] is not None:
                    state["lease"]["t4_capability_sha256"] = t4_capability_sha256
                    state["lease"]["promotion_record_sha256"] = promotion_record_sha256
            retry_not_before = stage_state.get("retry_not_before")
            if retry_not_before and dt.datetime.now(dt.timezone.utc) < _parse_timestamp(
                str(retry_not_before)
            ):
                raise ControlPlaneError(
                    f"stage retry backoff is active until {retry_not_before}"
                )
            if stage_state["status"] != "running":
                stage_state["status"] = "running"
                stage_state["attempts"] += 1
                stage_state["started_at"] = _utc_now()
                stage_state["completed_at"] = None
                stage_state["last_error"] = None
                stage_state["retry_not_before"] = None
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
            self._write_state_sentinel(
                entry,
                state,
                stage_id=stage_id,
                summary="Stage execution is active; no completion evidence accepted yet.",
            )
            self._render_views_unchecked()
            if stage["execution_mode"] == "agent":
                _, refreshed_state_map = self._stage_maps(plan, state)
                return self._work_order(
                    experiment_id,
                    agent_id,
                    str(executor["lease_id"]),
                    stage,
                    refreshed_state_map[stage_id],
                    t4_capability,
                    t4_capability_sha256,
                    promotion_record_sha256,
                )
            command = stage["command"]
            argv = [str(item) for item in command["argv"]]
            cwd = self.resolve_declared_path(str(command["cwd"]))
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in command["env"].items()})
            lease_id = str(executor["lease_id"])
            environment.update(
                {
                    COMMAND_EXPERIMENT_ENV: experiment_id,
                    COMMAND_LEASE_ENV: lease_id,
                    COMMAND_DOCKER_LABEL_ENV: f"{DOCKER_LEASE_LABEL_KEY}={lease_id}",
                }
            )
            if t4_capability is not None:
                environment.update(
                    {
                        T4_CAPABILITY_ENV: t4_capability,
                        T4_CAPABILITY_SHA_ENV: str(t4_capability_sha256),
                        T4_PROMOTION_SHA_ENV: str(promotion_record_sha256),
                    }
                )
            if len(argv) >= 2 and argv[0] == "docker" and argv[1] == "run":
                argv[2:2] = [
                    "--label",
                    f"{DOCKER_LEASE_LABEL_KEY}={lease_id}",
                    "--label",
                    f"{DOCKER_EXPERIMENT_LABEL_KEY}={experiment_id}",
                ]

        try:
            completed = self._run_command_with_heartbeat(
                experiment_id,
                lease_id,
                argv,
                cwd,
                environment,
                command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                self._refresh_lease(
                    experiment_id,
                    lease_id,
                    process_pid=os.getpid(),
                    process_role="controller",
                )
                self._command_failure(
                    entry,
                    plan,
                    state,
                    stage,
                    "timeout",
                    f"command exceeded {command_timeout_seconds} seconds",
                )
            raise ControlPlaneError(str(exc)) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                self._command_failure(
                    entry, plan, state, stage, "command_runtime", str(exc)
                )
            raise ControlPlaneError(str(exc)) from exc
        if completed.returncode != 0:
            message = (
                f"command exited {completed.returncode}; stderr tail: "
                f"{completed.stderr[-4096:]}"
            )
            with self.state_lock(experiment_id):
                state = _read_json(state_path)
                failure_evidence, evidence_rejection = (
                    self._declared_failed_command_evidence(
                        entry, state, stage
                    )
                )
                error_class = "command_exit"
                if failure_evidence is not None:
                    error_class = str(failure_evidence["error_class"])
                    message = str(failure_evidence["message"])
                elif evidence_rejection:
                    message += f"; {evidence_rejection}"
                self._command_failure(
                    entry,
                    plan,
                    state,
                    stage,
                    error_class,
                    message,
                    failure_evidence=failure_evidence,
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
            return self.complete_stage(experiment_id, evidence_path, agent_id)
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
        return self.complete_stage(experiment_id, evidence_path, agent_id)

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
        *,
        state: Mapping[str, Any] | None = None,
        lease: Mapping[str, Any] | None = None,
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
        self._verify_matched_arm_evidence(entry, stage, evidence)
        stage_resources = stage.get("resources", {})
        gpu_count = int(stage_resources.get("gpu_count", 0) or 0)
        gpu_optional = stage_resources.get("gpu_optional") is True
        requires_resources = True
        resource_metrics = evidence.get("metrics", {}).get("resources")
        if requires_resources:
            if not isinstance(resource_metrics, Mapping):
                raise ControlPlaneError(
                    "every executed stage requires metrics.resources"
                )
            missing_resource_keys = [
                key
                for key in ("gpu_hours", "cpu_hours", "storage_gib")
                if key not in resource_metrics
            ]
            if missing_resource_keys:
                raise ControlPlaneError(
                    "metrics.resources is missing " + ", ".join(missing_resource_keys)
                )
            for key in ("gpu_hours", "cpu_hours", "storage_gib"):
                value = resource_metrics[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ControlPlaneError(f"metrics.resources.{key} must be numeric")
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ControlPlaneError(
                        f"metrics.resources.{key} must be finite and nonnegative"
                    )
            if (
                gpu_count > 0
                and not gpu_optional
                and float(resource_metrics.get("gpu_hours", 0)) <= 0
            ):
                raise ControlPlaneError("GPU stage evidence must record positive gpu_hours")
            stage_caps: list[float] = []
            for key in ("gpu_hours_total_max", "gpu_hours_max"):
                if key in stage_resources:
                    stage_caps.append(float(stage_resources[key]))
            per_arm = next(
                (
                    float(stage_resources[key])
                    for key in ("gpu_hours_per_arm_max", "maximum_gpu_hours_per_arm")
                    if key in stage_resources
                ),
                None,
            )
            if per_arm is not None:
                arm_count = int(
                    stage_resources.get(
                        "arm_count",
                        stage_resources.get("gpu_count", 1) or 1,
                    )
                )
                if arm_count <= 0:
                    raise ControlPlaneError("per-arm GPU cap requires a positive arm_count")
                stage_caps.append(per_arm * arm_count)
            measured_gpu_hours = float(resource_metrics.get("gpu_hours", 0))
            if stage_caps and measured_gpu_hours > min(stage_caps) + 1e-9:
                raise ControlPlaneError(
                    f"stage GPU budget exceeded: {measured_gpu_hours}>{min(stage_caps)}"
                )
        if state is not None:
            _, state_map = self._stage_maps(
                {"stages": [stage]},
                {
                    "stages": [
                        candidate
                        for candidate in state.get("stages", [])
                        if candidate.get("id") == stage.get("id")
                    ]
                },
            )
            stage_state = state_map.get(str(stage.get("id")), {})
            started_at = stage_state.get("started_at")
            if started_at and _parse_timestamp(str(evidence["completed_at"])) < _parse_timestamp(
                str(started_at)
            ):
                raise ControlPlaneError("evidence predates the current stage attempt")
            if evidence.get("attempt") is not None and evidence.get("attempt") != stage_state.get(
                "attempts"
            ):
                raise ControlPlaneError("evidence attempt does not match state")
            if evidence.get("plan_sha256") is not None and evidence.get(
                "plan_sha256"
            ) != state.get("plan_sha256"):
                raise ControlPlaneError("evidence plan_sha256 does not match state")
        if lease is not None and evidence.get("lease_id") is not None and evidence.get(
            "lease_id"
        ) != lease.get("lease_id"):
            raise ControlPlaneError("evidence lease_id does not match executor lease")
        if stage.get("id") == (entry.get("promotion") or {}).get("t4_stage_id"):
            protected = evidence.get("metrics", {}).get("protected_data_access")
            if not isinstance(protected, Mapping):
                raise ControlPlaneError(
                    "T4 evidence requires metrics.protected_data_access audit"
                )
            if protected.get("authorized") is not True:
                raise ControlPlaneError("T4 protected-data access was not authorized")
            if lease is not None:
                if protected.get("capability_sha256") != lease.get(
                    "t4_capability_sha256"
                ):
                    raise ControlPlaneError("T4 capability hash does not match lease")
                if protected.get("promotion_record_sha256") != lease.get(
                    "promotion_record_sha256"
                ):
                    raise ControlPlaneError("T4 promotion hash does not match lease")
            else:
                if not SHA256_RE.fullmatch(str(protected.get("capability_sha256", ""))):
                    raise ControlPlaneError("T4 capability audit hash is invalid")
                promotion_path = (
                    self.resolve_declared_path(str(self.load_portfolio()["runtime_root"]))
                    / "promotions"
                    / f"{entry['id']}.json"
                )
                if (
                    not promotion_path.is_file()
                    or protected.get("promotion_record_sha256")
                    != self._hash_file_cached(promotion_path)
                ):
                    raise ControlPlaneError("T4 promotion audit hash drifted")
            if not SHA256_RE.fullmatch(str(protected.get("access_log_sha256", ""))):
                raise ControlPlaneError("T4 evidence requires a hashed access log")
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
        declared = None
        if isinstance(completion, Mapping):
            declared = completion.get("evidence_file") or completion.get("marker")
        if declared:
            expected = self.resolve_declared_path(str(declared))
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
            actual_hash, actual_size = self._path_digest_and_size(path)
            if artifact.get("sha256") != actual_hash:
                raise ControlPlaneError(
                    f"evidence hash mismatch for {raw_path}: {actual_hash}"
                )
            if artifact.get("size_bytes") != actual_size:
                raise ControlPlaneError(f"evidence size mismatch for {raw_path}")

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
            if stage.get("required") and stage_state.get("status") != "passed":
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

    def _ingest_resource_metrics(
        self,
        plan: Mapping[str, Any],
        state: dict[str, Any],
        evidence: Mapping[str, Any],
    ) -> None:
        resources = evidence.get("metrics", {}).get("resources")
        if resources is None:
            return
        if not isinstance(resources, Mapping):
            raise ControlPlaneError("metrics.resources must be an object")
        consumed = state.setdefault("budget_consumed", {})
        for key in ("gpu_hours", "cpu_hours", "storage_gib"):
            value = resources.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ControlPlaneError(f"metrics.resources.{key} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ControlPlaneError(
                    f"metrics.resources.{key} must be finite and nonnegative"
                )
            if key == "storage_gib":
                consumed[key] = max(float(consumed.get(key, 0)), numeric)
            else:
                consumed[key] = float(consumed.get(key, 0)) + numeric
        budget = plan.get("budget", {})
        limits = {
            "gpu_hours": next(
                (
                    budget[key]
                    for key in (
                        "total_gpu_hours_max",
                        "aggregate_gpu_hours_max",
                        "maximum_gpu_hours",
                        "gpu_hours_max",
                    )
                    if key in budget
                ),
                None,
            ),
            "cpu_hours": next(
                (
                    budget[key]
                    for key in ("total_cpu_hours_max", "maximum_cpu_hours", "cpu_hours_max")
                    if key in budget
                ),
                None,
            ),
            "storage_gib": budget.get(
                "durable_storage_gib_max",
                budget.get("maximum_external_storage_gib", budget.get("storage_gib_max")),
            ),
        }
        exceeded = [
            f"{key}={consumed[key]}>{limit}"
            for key, limit in limits.items()
            if limit is not None and consumed[key] > float(limit)
        ]
        if exceeded:
            raise ControlPlaneError(
                "declared experiment budget exceeded: " + ", ".join(exceeded)
            )

    @_completion_failure_transition
    @_hash_operation
    def complete_stage(
        self, experiment_id: str, evidence_path: Path | str, agent_id: str
    ) -> dict[str, Any]:
        if not agent_id.strip():
            raise ControlPlaneError("agent-id must be nonempty")
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
            executor, lease = self._assert_execution_identity(experiment_id, state)
            if str(executor.get("agent_id")) != agent_id:
                raise ControlPlaneError("agent-id does not own this experiment")
            lease_id = str(executor["lease_id"])
            stage_id = state.get("current_stage_id")
            plan_map, state_map = self._stage_maps(plan, state)
            if stage_id not in plan_map:
                raise ControlPlaneError("state has no valid current stage")
            stage = plan_map[str(stage_id)]
            stage_state = state_map[str(stage_id)]
            if stage_state.get("status") != "running":
                raise ControlPlaneError("current stage is not running")
            result_path = root / "results" / "result.json"
            if (
                self._canonical_result_is_declared(entry, plan, artifacts)
                and evidence_resolved.resolve(strict=False)
                != result_path.resolve(strict=False)
            ):
                raise ControlPlaneError(
                    "results/result.json is a declared artifact/output and cannot be overwritten by differing completion evidence"
                )
            if (
                stage.get("execution_mode") == "command"
                and lease.get("process_role") == "command"
                and self._lease_liveness(lease) is True
            ):
                raise ControlPlaneError(
                    "cannot complete a command stage while its recorded process is live"
                )
            self._refresh_lease(
                experiment_id,
                lease_id,
                process_pid=os.getpid(),
                process_role="completion_controller",
            )
            self._verify_completion_evidence(
                entry,
                stage,
                evidence,
                evidence_resolved,
                state=state,
                lease=lease,
            )
            self._ingest_resource_metrics(plan, state, evidence)
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
                promotion = entry.get("promotion") or {}
                if promotion.get("candidate") and promotion.get("t3_stage_id") == stage_id:
                    state["status"] = "blocked"
                    state["current_stage_id"] = None
                    state["executor"] = None
                    if "lease" in state:
                        state["lease"] = None
                    state["blockers"] = [
                        {
                            "code": "awaiting_portfolio_promotion",
                            "stage_id": stage_id,
                            "message": "T3 is terminal; waiting for deterministic portfolio promotion",
                        }
                    ]
                    event_type = "stage_completed"
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
            if state["status"] in {"finished", "blocked"}:
                lease_path = self.lease_path(experiment_id)
                if lease_path.exists():
                    lease_path.unlink()
            if evidence_resolved.resolve(strict=False) != result_path.resolve(
                strict=False
            ):
                _atomic_write_json(result_path, evidence)
            self._render_views_unchecked()
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
                experiment_id,
                str(executor["lease_id"]),
                process_pid=os.getpid(),
                process_role="agent_heartbeat",
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

    def _recover_previous_plan(self, source: str) -> bytes:
        git_match = re.fullmatch(r"git:([0-9a-f]{7,40}):(.+)", source)
        if git_match:
            commit, relative = git_match.groups()
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ControlPlaneError("git previous-plan path must be repository-relative")
            completed = subprocess.run(
                ["git", "show", f"{commit}:{candidate.as_posix()}"],
                cwd=self.workspace_root.parent,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise ControlPlaneError(
                    "cannot recover previous plan from git: "
                    + completed.stderr.decode("utf-8", errors="replace")[-2048:]
                )
            return completed.stdout
        if source.startswith("file:"):
            path = self.resolve_declared_path(source[5:])
            if not path.is_file():
                raise ControlPlaneError(f"previous plan source is missing: {path}")
            return path.read_bytes()
        raise ControlPlaneError(
            "previous-plan-source must be git:<commit>:<repo-relative-path> or file:<path>"
        )

    def _recover_previous_metadata(
        self,
        source: str,
        claims_path: Path,
        artifacts_path: Path,
    ) -> tuple[bytes, bytes, str]:
        git_match = re.fullmatch(r"git:([0-9a-f]{7,40}):(.+)", source)
        if git_match:
            commit, relative = git_match.groups()
            package_dir = Path(relative).parent
            recovered: list[bytes] = []
            for relative_path in (
                package_dir / "claims.yaml",
                package_dir / "artifacts" / "manifest.json",
            ):
                completed = subprocess.run(
                    ["git", "show", f"{commit}:{relative_path.as_posix()}"],
                    cwd=self.repo_root,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if completed.returncode != 0:
                    raise ControlPlaneError(
                        f"cannot recover predecessor metadata {relative_path}: "
                        + completed.stderr.decode("utf-8", errors="replace")[-2048:]
                    )
                recovered.append(completed.stdout)
            return recovered[0], recovered[1], f"git:{commit}:{package_dir.as_posix()}"
        if source.startswith("file:"):
            plan_source = self.resolve_declared_path(source[5:])
            sibling_claims = plan_source.parent / "claims.yaml"
            sibling_artifacts = plan_source.parent / "artifacts" / "manifest.json"
            if sibling_claims.is_file() and sibling_artifacts.is_file():
                return (
                    sibling_claims.read_bytes(),
                    sibling_artifacts.read_bytes(),
                    f"file:{plan_source.parent}",
                )
        return claims_path.read_bytes(), artifacts_path.read_bytes(), "current_metadata_snapshot"

    def _carried_resource_totals(
        self, stage_states: Sequence[Mapping[str, Any]]
    ) -> dict[str, float]:
        totals = {"gpu_hours": 0.0, "cpu_hours": 0.0, "storage_gib": 0.0}
        for stage_state in stage_states:
            if stage_state.get("status") != "passed":
                continue
            for evidence_ref in stage_state.get("evidence_refs", []):
                path = self.resolve_declared_path(str(evidence_ref))
                if not path.is_file():
                    continue
                resources = _read_json(path).get("metrics", {}).get("resources", {})
                if not isinstance(resources, Mapping):
                    continue
                totals["gpu_hours"] += float(resources.get("gpu_hours", 0.0))
                totals["cpu_hours"] += float(resources.get("cpu_hours", 0.0))
                totals["storage_gib"] = max(
                    totals["storage_gib"], float(resources.get("storage_gib", 0.0))
                )
        return totals

    def replan(
        self,
        experiment_id: str,
        agent_id: str,
        reason: str,
        previous_plan_source: str,
        *,
        new_plan_path: Path | None = None,
        adopt_current: bool = False,
    ) -> dict[str, Any]:
        if not agent_id.strip() or not reason.strip():
            raise ControlPlaneError("agent-id and reason must be nonempty")
        if adopt_current == (new_plan_path is not None):
            raise ControlPlaneError("choose exactly one of --plan or --adopt-current")
        entry = self.entry(experiment_id)
        root, plan_path, state_path, claims_path, events_path, artifacts_path = (
            self.experiment_paths(entry)
        )
        with self.state_lock(experiment_id):
            old_state_bytes = state_path.read_bytes()
            state = _read_json(state_path)
            prior_budget = state.get("budget_consumed", {})
            previous_budget_consumed = {
                key: float(prior_budget.get(key, 0.0))
                for key in ("gpu_hours", "cpu_hours", "storage_gib")
            }
            if state.get("status") not in {"draft", "ready", "blocked"}:
                raise ControlPlaneError("replan requires an inactive draft/ready/blocked state")
            if state.get("executor") is not None or self.lease_path(experiment_id).exists():
                raise ControlPlaneError("replan refuses any live executor or lease file")
            old_hash = str(state["plan_sha256"])
            old_revision = int(state["plan_revision"])
            old_bytes = self._recover_previous_plan(previous_plan_source)
            if _sha256_bytes(old_bytes) != old_hash:
                raise ControlPlaneError(
                    "recovered previous plan hash does not match state lineage"
                )
            old_plan = _load_yaml_bytes(old_bytes, previous_plan_source)
            old_claims_bytes, old_artifacts_bytes, metadata_source = (
                self._recover_previous_metadata(
                    previous_plan_source, claims_path, artifacts_path
                )
            )
            old_claims = _load_yaml_bytes(old_claims_bytes, metadata_source + "/claims.yaml")
            try:
                old_artifacts = json.loads(old_artifacts_bytes)
            except json.JSONDecodeError as exc:
                raise ControlPlaneError(
                    f"cannot read predecessor artifact manifest: {exc}"
                ) from exc
            old_projection = self._obligation_projection(
                old_claims,
                old_artifacts,
                pinned_at=(state.get("obligation_projection") or {}).get("pinned_at"),
                static_artifact_ids=(state.get("obligation_projection") or {}).get(
                    "static_artifact_ids"
                ),
            )
            pinned_projection = state.get("obligation_projection") or {}
            if pinned_projection and any(
                old_projection.get(key) != pinned_projection.get(key)
                for key in ("claims_sha256", "artifacts_sha256")
            ):
                raise ControlPlaneError(
                    "recovered predecessor claims/artifacts do not match pinned state obligations"
                )
            if adopt_current:
                new_bytes = plan_path.read_bytes()
            else:
                assert new_plan_path is not None
                source_path = new_plan_path.resolve(strict=True)
                new_bytes = source_path.read_bytes()
            new_plan = _load_yaml_bytes(new_bytes, str(new_plan_path or plan_path))
            if not isinstance(new_plan, Mapping):
                raise ControlPlaneError("new plan must be a YAML mapping")
            schema_issues = self._schema_issues("experiment", new_plan, plan_path)
            if schema_issues:
                raise ControlPlaneError(
                    f"new plan schema is invalid: {schema_issues[0].message}"
                )
            if new_plan.get("id") != experiment_id:
                raise ControlPlaneError("new plan experiment id differs")
            if new_plan.get("plan_revision") != old_revision + 1:
                raise ControlPlaneError("new plan revision must increment exactly once")
            new_hash = _sha256_bytes(new_bytes)
            if new_hash == old_hash:
                raise ControlPlaneError("new plan hash must differ from previous plan")

            result_path = root / "results" / "result.json"
            old_result_bytes = result_path.read_bytes() if result_path.is_file() else None

            old_stage_map = {
                str(item.get("id")): item for item in old_plan.get("stages", [])
            }
            old_state_map = {
                str(item.get("id")): item for item in state.get("stages", [])
            }
            new_stages: list[dict[str, Any]] = []
            for stage in new_plan.get("stages", []):
                stage_id = str(stage["id"])
                unchanged = old_stage_map.get(stage_id) == stage
                prior = old_state_map.get(stage_id, {})
                if unchanged and prior.get("status") in {"passed", "skipped"}:
                    new_stages.append(copy.deepcopy(prior))
                else:
                    new_stages.append(
                        {
                            "id": stage_id,
                            "status": "pending",
                            "attempts": 0,
                            "started_at": None,
                            "completed_at": None,
                            "last_error": None,
                            "retry_not_before": None,
                            "evidence_refs": [],
                        }
                    )
            state["stages"] = new_stages
            state["plan_sha256"] = new_hash
            state["plan_revision"] = old_revision + 1
            state["status"] = "ready"
            state["outcome"] = None
            state["executor"] = None
            if "lease" in state:
                state["lease"] = None
            state["blockers"] = []
            state["artifacts_verified"] = False
            state["closeout_passed"] = False
            claims = _read_yaml(claims_path)
            artifacts = _read_json(artifacts_path)
            metadata_issues = self._schema_issues(
                "claims", claims, claims_path
            ) + self._schema_issues("artifact_manifest", artifacts, artifacts_path)
            if metadata_issues:
                raise ControlPlaneError(
                    f"revised metadata is invalid: {metadata_issues[0].message}"
                )
            state["obligation_projection"] = self._obligation_projection(
                claims, artifacts
            )
            state["verified_artifact_ids"] = sorted(
                str(artifact["id"])
                for artifact in artifacts.get("artifacts", [])
                if artifact.get("status") == "verified"
            )
            carried_budget_consumed = self._carried_resource_totals(new_stages)
            state["budget_consumed"] = carried_budget_consumed
            eligible = self._eligible_stage_ids(new_plan, state)
            if not eligible:
                raise ControlPlaneError("revised plan has no eligible stage")
            next_stage = eligible[0]
            portfolio = self.load_portfolio()
            prospective_states = self.states(portfolio)
            prospective_states[experiment_id] = state
            dependencies_ok, dependency_failures = self.dependencies_satisfied(
                entry, prospective_states
            )
            for stage_state in state["stages"]:
                if stage_state["id"] == next_stage:
                    stage_state["status"] = "ready" if dependencies_ok else "blocked"
            if dependencies_ok:
                state["current_stage_id"] = next_stage
                state["status"] = "ready"
                state["blockers"] = []
            else:
                state["current_stage_id"] = None
                state["status"] = "blocked"
                state["blockers"] = [
                    {
                        "code": "dependency_unsatisfied",
                        "stage_id": next_stage,
                        "message": failure,
                    }
                    for failure in dependency_failures
                ]
            if "phase" in state:
                state["phase"] = next_stage

            semantic_issues = self._validate_plan_semantics(
                entry, new_plan, state, portfolio, plan_path
            )
            hardware_path = self.resolve_declared_path(
                str(portfolio.get("hardware_profile_ref", ""))
            )
            hardware_profile = (
                _read_yaml(hardware_path) if hardware_path.is_file() else {}
            )
            semantic_issues.extend(
                self._validate_hardware_semantics(
                    new_plan,
                    hardware_profile if isinstance(hardware_profile, Mapping) else {},
                    plan_path,
                )
            )
            semantic_issues.extend(
                self._validate_matched_arm_contract(entry, new_plan, plan_path)
            )
            semantic_issues.extend(
                self._validate_artifact_semantics(
                    entry, state, artifacts, portfolio, artifacts_path
                )
            )
            semantic_issues.extend(
                self._validate_canonical_result_contract(
                    entry, new_plan, artifacts, plan_path
                )
            )
            if semantic_issues:
                raise ControlPlaneError(
                    f"new plan semantics are invalid: {semantic_issues[0].message}"
                )
            for claim in claims.get("claims", []):
                unknown = [
                    str(ref)
                    for ref in claim.get("gate_refs", [])
                    if str(ref) not in self._declared_gate_refs(new_plan)
                ]
                if unknown:
                    raise ControlPlaneError(
                        f"claim {claim.get('id')} has undeclared gates: {', '.join(unknown)}"
                    )
            new_stage_ids = {str(item["id"]) for item in new_plan.get("stages", [])}
            for artifact in artifacts.get("artifacts", []):
                producer = artifact.get("producer_stage_id")
                if producer is not None and producer not in new_stage_ids:
                    raise ControlPlaneError(
                        f"artifact {artifact.get('id')} has unknown producer {producer}"
                    )
            promotion = entry.get("promotion") or {}
            for key in ("t3_stage_id", "t4_stage_id", "matched_control_stage_id"):
                value = promotion.get(key)
                if value is not None and value not in new_stage_ids:
                    raise ControlPlaneError(f"promotion {key} names unknown stage {value}")

            archive_root = self.runtime_control_root / "replans" / experiment_id / (
                f"revision_{old_revision}_{old_hash[:12]}"
            )
            archive_root.mkdir(parents=True, exist_ok=False)
            _atomic_write_text(archive_root / "experiment.yaml", old_bytes.decode("utf-8"))
            _atomic_write_bytes(archive_root / "claims.yaml", old_claims_bytes)
            _atomic_write_bytes(
                archive_root / "artifacts" / "manifest.json", old_artifacts_bytes
            )
            _atomic_write_bytes(archive_root / "state.json", old_state_bytes)
            if old_result_bytes is not None:
                _atomic_write_bytes(archive_root / "result.json", old_result_bytes)
            replanned_at = _utc_now()
            archive_display = self.relative_display(archive_root / "state.json")
            archived_plan_display = self.relative_display(archive_root / "experiment.yaml")
            archived_claims_display = self.relative_display(archive_root / "claims.yaml")
            archived_artifacts_display = self.relative_display(
                archive_root / "artifacts" / "manifest.json"
            )
            old_claims_file_hash = _sha256_bytes(old_claims_bytes)
            old_artifacts_file_hash = _sha256_bytes(old_artifacts_bytes)
            current_claims_file_hash = self._hash_file_cached(claims_path)
            current_artifacts_file_hash = self._hash_file_cached(artifacts_path)
            archive_state_hash = _sha256_bytes(old_state_bytes)
            archive_result_hash = (
                _sha256_bytes(old_result_bytes) if old_result_bytes is not None else None
            )
            archive_result_display = (
                self.relative_display(archive_root / "result.json")
                if old_result_bytes is not None
                else None
            )
            state.setdefault("plan_history", []).append(
                {
                    "plan_revision": old_revision,
                    "plan_sha256": old_hash,
                    "previous_plan_source": previous_plan_source,
                    "archived_state_path": archive_display,
                    "archived_plan_path": archived_plan_display,
                    "archived_plan_sha256": old_hash,
                    "archived_plan_size_bytes": len(old_bytes),
                    "archived_state_sha256": archive_state_hash,
                    "archived_state_size_bytes": len(old_state_bytes),
                    "archived_result_path": archive_result_display,
                    "archived_result_sha256": archive_result_hash,
                    "archived_result_size_bytes": (
                        len(old_result_bytes) if old_result_bytes is not None else None
                    ),
                    "archived_claims_path": archived_claims_display,
                    "archived_artifacts_path": archived_artifacts_display,
                    "previous_claims_sha256": old_claims_file_hash,
                    "previous_claims_size_bytes": len(old_claims_bytes),
                    "previous_artifacts_sha256": old_artifacts_file_hash,
                    "previous_artifacts_size_bytes": len(old_artifacts_bytes),
                    "current_claims_sha256": current_claims_file_hash,
                    "current_artifacts_sha256": current_artifacts_file_hash,
                    "previous_budget_consumed": previous_budget_consumed,
                    "carried_budget_consumed": carried_budget_consumed,
                    "replanned_at": replanned_at,
                }
            )
            try:
                if not adopt_current:
                    _atomic_write_text(plan_path, new_bytes.decode("utf-8"))
                state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="plan_revised",
                    actor=agent_id,
                    payload={
                        "reason": reason.strip(),
                        "old_plan_revision": old_revision,
                        "old_plan_sha256": old_hash,
                        "new_plan_revision": old_revision + 1,
                        "new_plan_sha256": new_hash,
                        "previous_plan_source": previous_plan_source,
                        "archived_state_path": archive_display,
                        "archived_plan_path": archived_plan_display,
                        "archived_plan_sha256": old_hash,
                        "archived_plan_size_bytes": len(old_bytes),
                        "archived_state_sha256": archive_state_hash,
                        "archived_state_size_bytes": len(old_state_bytes),
                        "archived_result_path": archive_result_display,
                        "archived_result_sha256": archive_result_hash,
                        "archived_result_size_bytes": (
                            len(old_result_bytes) if old_result_bytes is not None else None
                        ),
                        "archived_claims_path": archived_claims_display,
                        "archived_artifacts_path": archived_artifacts_display,
                        "previous_claims_sha256": old_claims_file_hash,
                        "previous_claims_size_bytes": len(old_claims_bytes),
                        "previous_artifacts_sha256": old_artifacts_file_hash,
                        "previous_artifacts_size_bytes": len(old_artifacts_bytes),
                        "current_claims_sha256": current_claims_file_hash,
                        "current_artifacts_sha256": current_artifacts_file_hash,
                        "previous_budget_consumed": previous_budget_consumed,
                        "carried_budget_consumed": carried_budget_consumed,
                        "next_stage_id": state.get("current_stage_id"),
                    },
                )
            except Exception:
                if not adopt_current:
                    _atomic_write_text(plan_path, old_bytes.decode("utf-8"))
                shutil.rmtree(archive_root)
                raise
            sentinel_status = "ready" if state["status"] == "ready" else "blocked"
            sentinel_stage_id = next_stage
            _atomic_write_json(
                result_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": experiment_id,
                    "stage_id": sentinel_stage_id,
                    "plan_sha256": new_hash,
                    "plan_revision": old_revision + 1,
                    "state_revision": state["revision"],
                    "state_status": state["status"],
                    "sentinel": True,
                    "status": sentinel_status,
                    "completed_at": state["updated_at"],
                    "summary": "Plan revision adopted; no stage has executed in this revision.",
                    "checks": [],
                    "artifacts": [],
                    "metrics": {},
                    "error": None,
                },
            )
            self._render_views_unchecked()
        return {
            "replanned": True,
            "id": experiment_id,
            "old_plan_revision": old_revision,
            "new_plan_revision": old_revision + 1,
            "old_plan_sha256": old_hash,
            "new_plan_sha256": new_hash,
            "current_stage_id": state.get("current_stage_id"),
            "revision": state["revision"],
        }

    def _promotion_candidate_row(
        self,
        entry: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        promotion = entry.get("promotion") or {}
        stage_id = str(promotion.get("t3_stage_id"))
        stage_state = next(
            (item for item in state.get("stages", []) if item.get("id") == stage_id),
            None,
        )
        evidence_path: Path | None = None
        evidence: Mapping[str, Any] = {}
        if stage_state and stage_state.get("evidence_refs"):
            evidence_path = self.resolve_declared_path(
                str(stage_state["evidence_refs"][-1])
            )
            if evidence_path.is_file():
                evidence = _read_json(evidence_path)
        metrics = evidence.get("metrics", {}).get("promotion", {})
        if not isinstance(metrics, Mapping):
            raise ControlPlaneError(
                f"{entry['id']} T3 evidence metrics.promotion must be an object"
            )
        keys = {
            "eligible": "eligible",
            "safety_pass": "all_safety_and_noninferiority_gates_pass",
            "dice_delta": "paired_val80_dice_delta",
            "off_location_fp_reduction": "val80_off_location_fp_reduction",
            "gpu_hours": "measured_gpu_hours",
        }
        keys.update(promotion.get("metric_keys") or {})
        eligible = metrics.get(keys["eligible"], False)
        safety = metrics.get(keys["safety_pass"], False)
        if not isinstance(eligible, bool) or not isinstance(safety, bool):
            raise ControlPlaneError(
                f"{entry['id']} promotion eligibility/safety metrics must be booleans"
            )
        numeric: dict[str, float] = {}
        for field in ("dice_delta", "off_location_fp_reduction", "gpu_hours"):
            value = metrics.get(keys[field], 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ControlPlaneError(
                    f"{entry['id']} promotion metric {keys[field]} must be numeric"
                )
            numeric[field] = float(value)
            if not math.isfinite(numeric[field]):
                raise ControlPlaneError(
                    f"{entry['id']} promotion metric {keys[field]} must be finite"
                )
        if numeric["gpu_hours"] < 0:
            raise ControlPlaneError(f"{entry['id']} measured_gpu_hours must be nonnegative")
        for field in ("dice_delta", "off_location_fp_reduction"):
            if not -1.0 <= numeric[field] <= 1.0:
                raise ControlPlaneError(
                    f"{entry['id']} promotion metric {keys[field]} must be within [-1, 1]"
                )
        accounted_gpu_hours = float(state.get("budget_consumed", {}).get("gpu_hours", 0.0))
        if accounted_gpu_hours is None or not math.isclose(
            float(accounted_gpu_hours), numeric["gpu_hours"], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ControlPlaneError(
                f"{entry['id']} measured_gpu_hours must equal cumulative state budget_consumed.gpu_hours"
            )
        return {
            "id": str(entry["id"]),
            "eligible": eligible,
            "safety_pass": safety,
            **numeric,
            "rank": None,
            "selected": False,
            "barrier_reason": None,
            "t3_evidence_path": (
                self.relative_display(evidence_path) if evidence_path else None
            ),
            "t3_evidence_sha256": (
                self._hash_file_cached(evidence_path)
                if evidence_path and evidence_path.is_file()
                else None
            ),
        }

    @_promotion_atomic
    @_hash_operation
    def promote(self, agent_id: str) -> dict[str, Any]:
        if not agent_id.strip():
            raise ControlPlaneError("agent-id must be nonempty")
        portfolio = self.load_portfolio()
        states = self.states(portfolio)
        active = [
            identifier
            for identifier, state in states.items()
            if state.get("status") in ACTIVE_STATUSES
        ]
        if active:
            raise ControlPlaneError(
                "promotion refuses active experiments: " + ", ".join(sorted(active))
            )
        candidates = [
            entry
            for entry in portfolio.get("experiments", [])
            if (entry.get("promotion") or {}).get("candidate")
        ]
        if not candidates:
            raise ControlPlaneError("portfolio declares no promotion candidates")
        promotion_root = self.resolve_declared_path(str(portfolio["runtime_root"])) / "promotions"
        existing_paths = {
            str(entry["id"]): promotion_root / f"{entry['id']}.json"
            for entry in candidates
        }
        existing_count = sum(path.is_file() for path in existing_paths.values())
        if existing_count:
            if existing_count != len(existing_paths):
                raise ControlPlaneError("partial promotion record set exists; refusing mutation")
            records = {identifier: _read_json(path) for identifier, path in existing_paths.items()}
            selected_sets = {tuple(record.get("selected_ids", [])) for record in records.values()}
            if len(selected_sets) != 1:
                raise ControlPlaneError("existing promotion records conflict")
            for identifier, record in records.items():
                schema_issues = self._schema_issues("promotion", record, existing_paths[identifier])
                body = dict(record)
                claimed = body.pop("record_sha256", None)
                state = states[identifier]
                subject_rows = [
                    row for row in record.get("candidates", []) if row.get("id") == identifier
                ]
                selected = identifier in record.get("selected_ids", [])
                if (
                    schema_issues
                    or claimed != _sha256_bytes(_canonical_json(body).encode("utf-8"))
                    or record.get("portfolio_sha256") != self._hash_file_cached(self.portfolio_path)
                    or record.get("registry_revision") != portfolio.get("registry_revision", 1)
                    or record.get("experiment_id") != identifier
                    or record.get("source") != "experimentctl promote"
                    or record.get("plan_sha256") != state.get("plan_sha256")
                    or record.get("treatment_arm_id")
                    != (self.entry(identifier).get("promotion") or {}).get("treatment_arm_id")
                    or record.get("matched_control_arm_id")
                    != (self.entry(identifier).get("promotion") or {}).get("matched_control_arm_id")
                    or len(subject_rows) != 1
                    or record.get("t3_evidence_sha256")
                    != (subject_rows[0].get("t3_evidence_sha256") if subject_rows else None)
                    or record.get("rank")
                    != (subject_rows[0].get("rank") if subject_rows else None)
                    or record.get("matched_control_required") is not selected
                    or record.get("val120_access_authorized") is not selected
                ):
                    raise ControlPlaneError("existing promotion record does not match current portfolio")
                _, _, _, _, events_path, _ = self.experiment_paths(self.entry(identifier))
                _, events = self._validate_events(events_path, identifier)
                shown = self.relative_display(existing_paths[identifier])
                full_file_hash = self._hash_file_cached(existing_paths[identifier])
                if not any(
                    event.get("event_type") == "portfolio_promotion_decided"
                    and event.get("payload", {}).get("record_path") == shown
                    and event.get("payload", {}).get("record_sha256") == full_file_hash
                    for event in events
                ):
                    raise ControlPlaneError("existing promotion record lacks matching event")
            return {
                "promoted": True,
                "idempotent": True,
                "selected_ids": list(next(iter(selected_sets))),
                "max_t4_interventions": int(
                    next(iter(records.values()))["policy"]["max_t4_interventions"]
                ),
                "candidates": next(iter(records.values()))["candidates"],
                "updated": [],
            }

        for entry in candidates:
            experiment_id = str(entry["id"])
            promotion = entry.get("promotion") or {}
            plan = _read_yaml(self.resolve_declared_path(str(entry["plan_path"])))
            arm_issues = self._validate_matched_arm_contract(
                entry,
                plan,
                self.resolve_declared_path(str(entry["plan_path"])),
            )
            if arm_issues:
                raise ControlPlaneError(
                    f"{experiment_id} matched-arm contract is invalid: {arm_issues[0].message}"
                )
            stage_ids = {str(stage["id"]) for stage in plan.get("stages", [])}
            if promotion.get("t4_stage_id") not in stage_ids:
                raise ControlPlaneError(f"{experiment_id} has invalid promotion T4 stage")
            if not any(stage.get("kind") == "closeout" for stage in plan.get("stages", [])):
                raise ControlPlaneError(f"{experiment_id} has no closeout stage")
            _, _, _, _, _, artifacts_path = self.experiment_paths(entry)
            manifest = _read_json(artifacts_path)
            artifact = next(
                (item for item in manifest.get("artifacts", []) if item.get("id") == "promotion_record"),
                None,
            )
            if (
                artifact is None
                or self.resolve_declared_path(str(artifact.get("path")))
                != existing_paths[experiment_id]
                or artifact.get("controller_producer") != "experimentctl promote"
            ):
                raise ControlPlaneError(
                    f"{experiment_id} must predeclare the exact controller-produced promotion_record artifact"
                )
        waiting_code = "awaiting_portfolio_promotion"
        terminal_dependency_reasons: dict[str, str] = {}
        for entry in candidates:
            state = states[str(entry["id"])]
            waiting = any(
                isinstance(blocker, Mapping) and blocker.get("code") == waiting_code
                for blocker in state.get("blockers", [])
            )
            promotion = entry.get("promotion") or {}
            t3_id = str(promotion.get("t3_stage_id"))
            t3_state = next(
                (item for item in state.get("stages", []) if item.get("id") == t3_id),
                {},
            )
            if state.get("status") == "finished" and t3_state.get("status") != "passed":
                outcome = state.get("outcome")
                if outcome == "go" and not bool(
                    portfolio.get("promotion_policy", {}).get("allow_go_before_t3", False)
                ):
                    raise ControlPlaneError(
                        f"{entry['id']} finished/go before T3 without explicit policy"
                    )
                limiting_stage = next(
                    (
                        str(item["id"])
                        for item in reversed(state.get("stages", []))
                        if item.get("status") in {"failed", "blocked", "skipped", "passed"}
                    ),
                    "none",
                )
                terminal_dependency_reasons[str(entry["id"])] = (
                    f"terminal_before_t3:{outcome}/{limiting_stage}"
                )
            terminal_failures: list[str] = []
            for dependency in entry.get("depends_on", []):
                if dependency.get("type") != "hard":
                    continue
                dependency_state = states.get(str(dependency["id"]), {})
                if dependency_state.get("status") == "finished" and dependency_state.get(
                    "outcome"
                ) not in dependency.get("satisfying_outcomes", []):
                    terminal_failures.append(
                        f"{dependency['id']}/{dependency_state.get('outcome')}"
                    )
            if terminal_failures and str(entry["id"]) not in terminal_dependency_reasons:
                terminal_dependency_reasons[str(entry["id"])] = (
                    "terminal_unsatisfied_dependency:" + ",".join(terminal_failures)
                )
            elif state.get("status") != "finished" and not waiting:
                raise ControlPlaneError(
                    f"T3 barrier is incomplete: {entry['id']} is not terminal/waiting"
                )
        rows: list[dict[str, Any]] = []
        for entry in candidates:
            experiment_id = str(entry["id"])
            if experiment_id in terminal_dependency_reasons:
                rows.append(
                    {
                        "id": experiment_id,
                        "eligible": False,
                        "safety_pass": False,
                        "dice_delta": 0.0,
                        "off_location_fp_reduction": 0.0,
                        "gpu_hours": float(
                            states[experiment_id]
                            .get("budget_consumed", {})
                            .get("gpu_hours", 0.0)
                        ),
                        "rank": None,
                        "selected": False,
                        "barrier_reason": terminal_dependency_reasons[experiment_id],
                        "t3_evidence_path": None,
                        "t3_evidence_sha256": None,
                    }
                )
            else:
                rows.append(self._promotion_candidate_row(entry, states[experiment_id]))
        rankable = sorted(
            (row for row in rows if row["eligible"] and row["safety_pass"]),
            key=lambda row: (
                -row["dice_delta"],
                -row["off_location_fp_reduction"],
                row["gpu_hours"],
                row["id"],
            ),
        )
        for rank, row in enumerate(rankable, start=1):
            row["rank"] = rank
        policy = dict(portfolio.get("promotion_policy") or {})
        max_selected = min(2, int(policy.get("max_t4_interventions", 2)))
        selected_ids = [row["id"] for row in rankable[:max_selected]]
        for row in rows:
            row["selected"] = row["id"] in selected_ids
        created_at = _utc_now()
        promotion_root.mkdir(parents=True, exist_ok=True)
        portfolio_hash = self._hash_file_cached(self.portfolio_path)
        row_map = {str(row["id"]): row for row in rows}
        for entry in candidates:
            subject_id = str(entry["id"])
            subject_row = row_map[subject_id]
            selected = subject_id in selected_ids
            record = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": subject_id,
                "source": "experimentctl promote",
                "registry_revision": int(portfolio.get("registry_revision", 1)),
                "portfolio_sha256": portfolio_hash,
                "plan_sha256": states[subject_id]["plan_sha256"],
                "t3_evidence_sha256": subject_row["t3_evidence_sha256"],
                "rank": subject_row["rank"],
                "promoted_at": created_at,
                "matched_control_required": selected,
                "val120_access_authorized": selected,
                "treatment_arm_id": (entry.get("promotion") or {})["treatment_arm_id"],
                "matched_control_arm_id": (entry.get("promotion") or {})[
                    "matched_control_arm_id"
                ],
                "policy": {**policy, "max_t4_interventions": max_selected},
                "barrier": {
                    "completed_at": created_at,
                    "all_t3_terminal": True,
                    "candidate_ids": sorted(str(item["id"]) for item in candidates),
                },
                "candidates": sorted(copy.deepcopy(rows), key=lambda row: row["id"]),
                "selected_ids": selected_ids,
            }
            record["record_sha256"] = _sha256_bytes(
                _canonical_json(record).encode("utf-8")
            )
            schema_issues = self._schema_issues(
                "promotion", record, promotion_root / f"{entry['id']}.json"
            )
            if schema_issues:
                raise ControlPlaneError(
                    f"promotion record is invalid: {schema_issues[0].message}"
                )
            _atomic_write_json(promotion_root / f"{entry['id']}.json", record)

        updated: list[dict[str, Any]] = []
        for entry in candidates:
            experiment_id = str(entry["id"])
            with self.state_lock(experiment_id):
                _, plan_path, state_path, claims_path, events_path, artifacts_path = self.experiment_paths(entry)
                state = _read_json(state_path)
                plan = _read_yaml(plan_path)
                claims = _read_yaml(claims_path)
                promotion = entry["promotion"]
                record_path = promotion_root / f"{experiment_id}.json"
                manifest = _read_json(artifacts_path)
                promotion_artifact = next(
                    (
                        artifact
                        for artifact in manifest.get("artifacts", [])
                        if artifact.get("id") == "promotion_record"
                    ),
                    None,
                )
                if promotion_artifact is None:
                    raise ControlPlaneError(
                        f"{experiment_id} does not predeclare promotion_record artifact"
                    )
                if promotion_artifact.get("controller_producer") != "experimentctl promote":
                    raise ControlPlaneError(
                        f"{experiment_id} promotion_record lacks controller producer identity"
                    )
                if self.resolve_declared_path(str(promotion_artifact["path"])) != record_path:
                    raise ControlPlaneError(
                        f"{experiment_id} promotion_record artifact path differs"
                    )
                promotion_artifact["status"] = "verified"
                promotion_artifact["sha256"] = self._hash_file_cached(record_path)
                promotion_artifact["size_bytes"] = record_path.stat().st_size
                original_manifest_text = artifacts_path.read_text(encoding="utf-8")
                _atomic_write_json(artifacts_path, manifest)
                state["verified_artifact_ids"] = sorted(
                    str(artifact["id"])
                    for artifact in manifest.get("artifacts", [])
                    if artifact.get("status") == "verified"
                )
                barrier_reason = terminal_dependency_reasons.get(experiment_id)
                if barrier_reason:
                    try:
                        state = self._commit_state_event(
                            state_path,
                            events_path,
                            state,
                            event_type="portfolio_promotion_decided",
                            actor=agent_id,
                            payload={
                                "selected": False,
                                "rank": None,
                                "selected_ids": selected_ids,
                                "barrier_reason": barrier_reason,
                                "record_path": self.relative_display(record_path),
                                "record_sha256": self._hash_file_cached(record_path),
                            },
                        )
                    except Exception:
                        _atomic_write_text(artifacts_path, original_manifest_text)
                        raise
                    updated.append(
                        {
                            "id": experiment_id,
                            "selected": False,
                            "current_stage_id": state.get("current_stage_id"),
                            "barrier_reason": barrier_reason,
                        }
                    )
                    continue
                if state.get("status") == "finished":
                    try:
                        state = self._commit_state_event(
                            state_path,
                            events_path,
                            state,
                            event_type="portfolio_promotion_decided",
                            actor=agent_id,
                            payload={
                                "selected": False,
                                "rank": next(
                                    row["rank"] for row in rows if row["id"] == experiment_id
                                ),
                                "selected_ids": selected_ids,
                                "record_path": self.relative_display(record_path),
                                "record_sha256": self._hash_file_cached(record_path),
                            },
                        )
                    except Exception:
                        _atomic_write_text(artifacts_path, original_manifest_text)
                        raise
                    updated.append(
                        {"id": experiment_id, "selected": False, "current_stage_id": None}
                    )
                    continue
                _, state_map = self._stage_maps(plan, state)
                t4_id = str(promotion.get("t4_stage_id"))
                if t4_id not in state_map:
                    raise ControlPlaneError(f"{experiment_id} promotion T4 stage is missing")
                selected = experiment_id in selected_ids
                if selected:
                    state_map[t4_id]["status"] = "ready"
                    state["current_stage_id"] = t4_id
                else:
                    state_map[t4_id]["status"] = "skipped"
                    state_map[t4_id]["completed_at"] = created_at
                    for candidate_id in self._eligible_stage_ids(plan, state):
                        if candidate_id != t4_id:
                            state_map[candidate_id]["status"] = "ready"
                            state["current_stage_id"] = candidate_id
                            break
                    else:
                        raise ControlPlaneError(
                            f"{experiment_id} has no closeout path after non-promotion"
                        )
                state["status"] = "ready"
                state["blockers"] = []
                state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="portfolio_promotion_decided",
                    actor=agent_id,
                    payload={
                        "selected": selected,
                        "rank": next(row["rank"] for row in rows if row["id"] == experiment_id),
                        "selected_ids": selected_ids,
                        "record_path": self.relative_display(
                            record_path
                        ),
                        "record_sha256": self._hash_file_cached(record_path),
                    },
                )
                self._write_state_sentinel(
                    entry,
                    state,
                    stage_id=str(state["current_stage_id"]),
                    summary=(
                        "Experiment selected for T4 matched-arm internal replication."
                        if selected
                        else "T4 skipped by deterministic portfolio ranking; closeout is ready."
                    ),
                )
                updated.append(
                    {
                        "id": experiment_id,
                        "selected": selected,
                        "current_stage_id": state["current_stage_id"],
                    }
                )
        self._render_views_unchecked()
        return {
            "promoted": True,
            "selected_ids": selected_ids,
            "max_t4_interventions": max_selected,
            "candidates": rows,
            "updated": updated,
        }

    @_hash_operation
    def lock_control_plane(self, agent_id: str) -> dict[str, Any]:
        if not agent_id.strip():
            raise ControlPlaneError("agent-id must be nonempty")
        portfolio = self.load_portfolio()
        lock_ref = portfolio.get("control_plane_lock_ref")
        if not lock_ref:
            raise ControlPlaneError("portfolio does not declare control_plane_lock_ref")
        lock_path = self.resolve_declared_path(str(lock_ref))
        manifests_root = self.workspace_root / "shared" / "manifests"
        if not _is_within(lock_path, manifests_root):
            raise ControlPlaneError("control-plane lock must be under shared/manifests")

        migrated: list[str] = []
        for entry in portfolio.get("experiments", []):
            experiment_id = str(entry["id"])
            with self.state_lock(experiment_id):
                root, _, state_path, claims_path, events_path, artifacts_path = (
                    self.experiment_paths(entry)
                )
                state = _read_json(state_path)
                claims = _read_yaml(claims_path)
                artifacts = _read_json(artifacts_path)
                projection = state.get("obligation_projection")
                if projection is not None and not self._projection_matches(
                    projection, claims, artifacts
                ):
                    raise ControlPlaneError(
                        f"{experiment_id} obligations drifted; refusing to repin"
                    )
                needs_pin = state.get("status") != "draft" and projection is None
                result_path = root / "results" / "result.json"
                result_invalid = not result_path.is_file()
                if result_path.is_file():
                    result = _read_json(result_path)
                    result_invalid = bool(
                        self._schema_issues("result", result, result_path)
                        or self._validate_result_state_consistency(
                            entry, state, result, result_path
                        )
                    )
                if result_invalid and any(
                    item.get("status") in {"passed", "failed"}
                    for item in state.get("stages", [])
                ):
                    raise ControlPlaneError(
                        f"{experiment_id} has executed-stage result drift; replan or repair explicitly"
                    )
                if not needs_pin and not result_invalid:
                    continue
                if needs_pin:
                    state["obligation_projection"] = self._obligation_projection(
                        claims, artifacts
                    )
                archived_result = None
                if result_invalid and result_path.is_file():
                    archive = self.runtime_control_root / "migrations" / experiment_id
                    archive.mkdir(parents=True, exist_ok=True)
                    digest = self._hash_file_cached(result_path)
                    archived_result = archive / f"result_pre_lock_{digest[:12]}.json"
                    shutil.copy2(result_path, archived_result)
                state = self._commit_state_event(
                    state_path,
                    events_path,
                    state,
                    event_type="control_plane_obligations_pinned",
                    actor=agent_id,
                    payload={
                        "projection_pinned": needs_pin,
                        "result_reconciled": result_invalid,
                        "archived_result_path": (
                            self.relative_display(archived_result)
                            if archived_result
                            else None
                        ),
                    },
                )
                if result_invalid:
                    stage_id = state.get("current_stage_id") or next(
                        str(item["id"])
                        for item in state.get("stages", [])
                        if item.get("status") in {"blocked", "ready", "pending"}
                    )
                    _atomic_write_json(
                        result_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "experiment_id": experiment_id,
                            "stage_id": stage_id,
                            "plan_sha256": state["plan_sha256"],
                            "plan_revision": state["plan_revision"],
                            "state_revision": state["revision"],
                            "state_status": state["status"],
                            "sentinel": True,
                            "status": (
                                "ready" if state["status"] == "ready" else "blocked"
                            ),
                            "completed_at": state["updated_at"],
                            "summary": "Control-plane migration sentinel; no stage executed in this revision.",
                            "checks": [],
                            "artifacts": [],
                            "metrics": {},
                            "error": None,
                        },
                    )
                migrated.append(experiment_id)
        self._render_views_unchecked()
        preflight = self.validate_all(
            check_generated_views=True, check_control_plane_lock=False
        )
        if not preflight["valid"]:
            first = preflight["errors"][0]
            raise ControlPlaneError(
                f"refusing control-plane lock: {first['path']}: {first['message']}"
            )
        sources = []
        for source in self._control_plane_source_paths(portfolio):
            if not source.is_file():
                raise ControlPlaneError(f"control-plane source is missing: {source}")
            sources.append(
                {
                    "path": self.relative_display(source),
                    "sha256": self._hash_file_cached(source),
                    "size_bytes": source.stat().st_size,
                }
            )
        obligations: list[dict[str, Any]] = []
        for entry in portfolio.get("experiments", []):
            _, _, state_path, claims_path, _, artifacts_path = self.experiment_paths(entry)
            state = _read_json(state_path)
            claims = _read_yaml(claims_path)
            artifacts = _read_json(artifacts_path)
            projection = self._obligation_projection(
                claims,
                artifacts,
                pinned_at=(state.get("obligation_projection") or {}).get("pinned_at"),
                static_artifact_ids=(state.get("obligation_projection") or {}).get(
                    "static_artifact_ids", []
                ),
            )
            if state.get("obligation_projection") != projection:
                raise ControlPlaneError(
                    f"{entry['id']} state projection differs before sealing"
                )
            obligations.append(
                {
                    "experiment_id": entry["id"],
                    "projection_sha256": _sha256_bytes(
                        _canonical_json(projection).encode("utf-8")
                    ),
                }
            )
        manifest = {
            "schema_version": CONTROL_PLANE_LOCK_VERSION,
            "created_at": _utc_now(),
            "created_by": agent_id,
            "registry_revision": int(portfolio.get("registry_revision", 1)),
            "sources": sources,
            "obligations": sorted(obligations, key=lambda row: row["experiment_id"]),
        }
        if lock_path.is_file():
            existing = _read_json(lock_path)
            comparable = dict(existing)
            comparable.pop("created_at", None)
            comparable.pop("created_by", None)
            target = dict(manifest)
            target.pop("created_at", None)
            target.pop("created_by", None)
            if comparable == target:
                return {
                    "locked": True,
                    "idempotent": True,
                    "path": self.relative_display(lock_path),
                    "sources": len(sources),
                    "migrated": migrated,
                }
            raise ControlPlaneError(
                "existing control-plane lock conflicts; increment registry revision before relocking"
            )
        _atomic_write_json(lock_path, manifest)
        post = self.validate_all()
        if not post["valid"]:
            lock_path.unlink()
            raise ControlPlaneError("written control-plane lock failed validation")
        return {
            "locked": True,
            "idempotent": False,
            "path": self.relative_display(lock_path),
            "sources": len(sources),
            "migrated": migrated,
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

    def _render_views_unchecked(self) -> dict[str, Any]:
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

    def render(self) -> dict[str, Any]:
        validation = self.validate_all(check_generated_views=False)
        if not validation["valid"]:
            raise ControlPlaneError(
                f"refusing render with {len(validation['errors'])} validation error(s)"
            )
        rendered = self._render_views_unchecked()
        post = self.validate_all()
        if not post["valid"]:
            raise ControlPlaneError("generated views did not validate after render")
        return rendered


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
    run_stage.add_argument("--agent-id", required=True)

    heartbeat = subparsers.add_parser(
        "heartbeat", help="refresh an active agent-stage lease"
    )
    heartbeat.add_argument("--id", required=True)
    heartbeat.add_argument("--agent-id", required=True)

    complete = subparsers.add_parser(
        "complete-stage", help="complete current stage from verified evidence"
    )
    complete.add_argument("--id", required=True)
    complete.add_argument("--agent-id", required=True)
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

    replan = subparsers.add_parser(
        "replan", help="adopt one audited immutable plan revision"
    )
    replan.add_argument("--id", required=True)
    replan.add_argument("--agent-id", required=True)
    replan.add_argument("--reason", required=True)
    replan.add_argument("--previous-plan-source", required=True)
    replan_source = replan.add_mutually_exclusive_group(required=True)
    replan_source.add_argument("--plan", type=Path)
    replan_source.add_argument("--adopt-current", action="store_true")

    promote = subparsers.add_parser(
        "promote", help="rank terminal T3 candidates and issue T4 decisions"
    )
    promote.add_argument("--agent-id", required=True)

    lock_control = subparsers.add_parser(
        "lock-control-plane", help="pin obligations and hash executable control-plane sources"
    )
    lock_control.add_argument("--agent-id", required=True)

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
            _print_json(controller.run_stage(args.id, args.agent_id))
            return 0
        if args.command == "heartbeat":
            _print_json(controller.heartbeat(args.id, args.agent_id))
            return 0
        if args.command == "complete-stage":
            _print_json(
                controller.complete_stage(args.id, args.evidence, args.agent_id)
            )
            return 0
        if args.command == "reopen-blocked":
            _print_json(controller.reopen_blocked(args.id, args.reason))
            return 0
        if args.command == "record-blocker":
            _print_json(
                controller.record_blocker(args.id, args.code, args.message)
            )
            return 0
        if args.command == "replan":
            _print_json(
                controller.replan(
                    args.id,
                    args.agent_id,
                    args.reason,
                    args.previous_plan_source,
                    new_plan_path=args.plan,
                    adopt_current=args.adopt_current,
                )
            )
            return 0
        if args.command == "promote":
            _print_json(controller.promote(args.agent_id))
            return 0
        if args.command == "lock-control-plane":
            _print_json(controller.lock_control_plane(args.agent_id))
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
