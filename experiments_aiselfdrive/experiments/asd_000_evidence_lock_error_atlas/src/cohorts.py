"""Deterministic development and internal-replication cohort validation.

The val120 complement is historically exposed and therefore is not an
independent, blinded, or untouched confirmation cohort.  It is still emitted
as an identity-only manifest because ASD-000 has no scientific reason to load
its labels, predictions, or metrics.  The only full case records copied to an
output are the accepted val80 development records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:  # Package import.
    from .lock_evidence import atomic_write_json, sha256_file
except ImportError:  # Direct import after adding ``src`` to sys.path.
    from lock_evidence import atomic_write_json, sha256_file


PRODUCER_STAGE = "lock_cohorts_and_prompt_ontology"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JSON_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)
_FORBIDDEN_HOLDOUT_KEYS = frozenset(
    {
        "categories",
        "category",
        "category_count",
        "findings",
        "finding",
        "finding_count",
        "declared_finding_count",
        "labels",
        "label",
        "ground_truth",
        "annotation",
        "masks",
        "mask",
        "metrics",
        "metric",
        "predictions",
        "prediction",
        "seg_path",
        "segmentation",
    }
)


class CohortError(ValueError):
    """Raised when a cohort or its provenance violates the locked contract."""


@dataclass(frozen=True)
class CohortPartition:
    """Validated ordered partition of a parent cohort."""

    parent_case_names: tuple[str, ...]
    development_case_names: tuple[str, ...]
    internal_replication_case_names: tuple[str, ...]

    @property
    def parent_count(self) -> int:
        return len(self.parent_case_names)

    @property
    def development_count(self) -> int:
        return len(self.development_case_names)

    @property
    def internal_replication_count(self) -> int:
        return len(self.internal_replication_case_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_case_names": list(self.parent_case_names),
            "development_case_names": list(self.development_case_names),
            "internal_replication_case_names": list(
                self.internal_replication_case_names
            ),
            "parent_count": self.parent_count,
            "development_count": self.development_count,
            "internal_replication_count": self.internal_replication_count,
            "disjoint": True,
            "exhaustive": True,
        }


@dataclass(frozen=True)
class _JsonToken:
    kind: str
    raw: str


class _TokenStream:
    def __init__(self, tokens: Iterable[_JsonToken]) -> None:
        self._tokens = iter(tokens)
        self._buffer: _JsonToken | None = None

    def peek(self) -> _JsonToken | None:
        if self._buffer is None:
            self._buffer = next(self._tokens, None)
        return self._buffer

    def pop(self) -> _JsonToken:
        token = self.peek()
        if token is None:
            raise CohortError("unexpected end of JSON")
        self._buffer = None
        return token

    def expect(self, kind: str) -> _JsonToken:
        token = self.pop()
        if token.kind != kind:
            raise CohortError(
                f"expected JSON token {kind!r}, observed {token.kind!r}"
            )
        return token


def _tokenize_json(text: str) -> Iterator[_JsonToken]:
    """Tokenize JSON without decoding values that the caller chooses to skip."""

    index = 0
    length = len(text)
    punctuation = {"{", "}", "[", "]", ":", ","}
    while index < length:
        character = text[index]
        if character.isspace():
            index += 1
            continue
        if character in punctuation:
            yield _JsonToken(character, character)
            index += 1
            continue
        if character == '"':
            start = index
            index += 1
            while index < length:
                character = text[index]
                if character == '"':
                    index += 1
                    yield _JsonToken("string", text[start:index])
                    break
                if character == "\\":
                    index += 2
                    continue
                if ord(character) < 0x20:
                    raise CohortError("unescaped control character in JSON string")
                index += 1
            else:
                raise CohortError("unterminated JSON string")
            continue

        start = index
        while (
            index < length
            and not text[index].isspace()
            and text[index] not in punctuation
        ):
            index += 1
        raw = text[start:index]
        if raw in {"true", "false", "null"} or _JSON_NUMBER_RE.fullmatch(raw):
            yield _JsonToken("scalar", raw)
            continue
        raise CohortError(f"invalid JSON token at character {start}")


def _decode_json_string(token: _JsonToken) -> str:
    if token.kind != "string":
        raise CohortError(f"expected a JSON string, observed {token.kind!r}")
    try:
        value = json.loads(token.raw)
    except json.JSONDecodeError as error:
        raise CohortError(f"invalid JSON string: {error}") from error
    if not isinstance(value, str):  # Defensive; JSON strings always decode to str.
        raise CohortError("decoded JSON key is not text")
    return value


def _skip_json_value(stream: _TokenStream) -> None:
    """Consume one JSON value without decoding or retaining its contents."""

    token = stream.pop()
    if token.kind in {"string", "scalar"}:
        return
    if token.kind == "[":
        if stream.peek() is not None and stream.peek().kind == "]":
            stream.pop()
            return
        while True:
            _skip_json_value(stream)
            separator = stream.pop()
            if separator.kind == "]":
                return
            if separator.kind != ",":
                raise CohortError("malformed JSON array")
    if token.kind == "{":
        if stream.peek() is not None and stream.peek().kind == "}":
            stream.pop()
            return
        while True:
            stream.expect("string")
            stream.expect(":")
            _skip_json_value(stream)
            separator = stream.pop()
            if separator.kind == "}":
                return
            if separator.kind != ",":
                raise CohortError("malformed JSON object")
    raise CohortError(f"unexpected JSON token {token.kind!r} at value boundary")


def _parse_case_identity_object(stream: _TokenStream) -> str:
    stream.expect("{")
    case_name: str | None = None
    seen_keys: set[str] = set()
    if stream.peek() is not None and stream.peek().kind == "}":
        stream.pop()
        raise CohortError("case record has no name")

    while True:
        key = _decode_json_string(stream.expect("string"))
        if key in seen_keys:
            raise CohortError(f"duplicate key {key!r} in case record")
        seen_keys.add(key)
        stream.expect(":")
        if key == "name":
            case_name = _decode_json_string(stream.pop())
            if not case_name:
                raise CohortError("case name must be non-empty")
        else:
            _skip_json_value(stream)

        separator = stream.pop()
        if separator.kind == "}":
            break
        if separator.kind != ",":
            raise CohortError("malformed case record")

    if case_name is None:
        raise CohortError("case record has no name")
    return case_name


def load_case_names_only(
    path: str | Path,
    *,
    split_key: str = "test",
) -> list[str]:
    """Read only direct case ``name`` fields from an evaluator manifest.

    Non-name values are lexically skipped, rather than deserialized.  This is
    the narrow reader used to derive the internal-replication complement from
    val200.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise CohortError(f"cannot read cohort manifest {source}: {error}") from error

    stream = _TokenStream(_tokenize_json(text))
    stream.expect("{")
    selected: list[str] | None = None
    seen_top_level_keys: set[str] = set()

    if stream.peek() is not None and stream.peek().kind == "}":
        stream.pop()
    else:
        while True:
            key = _decode_json_string(stream.expect("string"))
            if key in seen_top_level_keys:
                raise CohortError(f"duplicate top-level key {key!r}")
            seen_top_level_keys.add(key)
            stream.expect(":")
            if key != split_key:
                _skip_json_value(stream)
            else:
                stream.expect("[")
                selected = []
                if stream.peek() is not None and stream.peek().kind == "]":
                    stream.pop()
                else:
                    while True:
                        selected.append(_parse_case_identity_object(stream))
                        separator = stream.pop()
                        if separator.kind == "]":
                            break
                        if separator.kind != ",":
                            raise CohortError("malformed cohort case array")

            separator = stream.pop()
            if separator.kind == "}":
                break
            if separator.kind != ",":
                raise CohortError("malformed top-level cohort object")

    if stream.peek() is not None:
        raise CohortError("trailing content after cohort JSON")
    if selected is None:
        raise CohortError(f"cohort manifest has no {split_key!r} array")
    _require_unique_names(selected, f"{source}:{split_key}")
    return selected


def _require_unique_names(case_names: Sequence[str], cohort_name: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in case_names:
        if not isinstance(name, str) or not name:
            raise CohortError(f"{cohort_name} contains an invalid case name")
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise CohortError(
            f"{cohort_name} contains duplicate case names: "
            + ", ".join(sorted(set(duplicates)))
        )


def validate_partition(
    val200_case_names: Sequence[str],
    val80_case_names: Sequence[str],
    *,
    expected_val200_cases: int | None = None,
    expected_val80_cases: int | None = None,
    expected_val120_cases: int | None = None,
) -> CohortPartition:
    """Validate and return the ordered val80/val120 partition of val200."""

    parent = tuple(val200_case_names)
    development = tuple(val80_case_names)
    _require_unique_names(parent, "val200")
    _require_unique_names(development, "val80")

    if expected_val200_cases is not None and len(parent) != expected_val200_cases:
        raise CohortError(
            f"val200 case count is {len(parent)}, expected {expected_val200_cases}"
        )
    if expected_val80_cases is not None and len(development) != expected_val80_cases:
        raise CohortError(
            f"val80 case count is {len(development)}, expected {expected_val80_cases}"
        )

    parent_set = set(parent)
    development_set = set(development)
    outside_parent = sorted(development_set - parent_set)
    if outside_parent:
        raise CohortError(
            "val80 is not a subset of val200; unexpected cases: "
            + ", ".join(outside_parent)
        )

    internal_replication = tuple(
        name for name in parent if name not in development_set
    )
    if (
        expected_val120_cases is not None
        and len(internal_replication) != expected_val120_cases
    ):
        raise CohortError(
            f"val120 complement count is {len(internal_replication)}, "
            f"expected {expected_val120_cases}"
        )
    internal_replication_set = set(internal_replication)
    if development_set & internal_replication_set:
        raise CohortError("val80 and val120 overlap")
    if development_set | internal_replication_set != parent_set:
        raise CohortError("val80 and val120 do not exhaust val200")
    if (
        tuple(name for name in parent if name in internal_replication_set)
        != internal_replication
    ):
        raise CohortError("val120 does not preserve val200 order")

    return CohortPartition(parent, development, internal_replication)


def derive_case_complement(
    val200_case_names: Sequence[str],
    val80_case_names: Sequence[str],
    *,
    expected_val200_cases: int | None = None,
    expected_val80_cases: int | None = None,
    expected_val120_cases: int | None = None,
) -> list[str]:
    """Return val200-minus-val80 in val200 order after full partition checks."""

    partition = validate_partition(
        val200_case_names,
        val80_case_names,
        expected_val200_cases=expected_val200_cases,
        expected_val80_cases=expected_val80_cases,
        expected_val120_cases=expected_val120_cases,
    )
    return list(partition.internal_replication_case_names)


def _load_development_records(
    path: str | Path,
    *,
    split_key: str = "test",
) -> tuple[list[dict[str, Any]], list[str], int]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CohortError(f"cannot parse development cohort {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CohortError("development cohort root must be a JSON object")
    records = payload.get(split_key)
    if not isinstance(records, list):
        raise CohortError(f"development cohort has no {split_key!r} list")

    normalized_records: list[dict[str, Any]] = []
    names: list[str] = []
    finding_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CohortError(f"val80 case record {index} is not an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise CohortError(f"val80 case record {index} has no valid name")
        findings = record.get("findings")
        if not isinstance(findings, (Mapping, list)):
            raise CohortError(f"val80 case {name!r} has invalid findings")
        categories = record.get("categories")
        if categories is not None and not isinstance(categories, (Mapping, list)):
            raise CohortError(f"val80 case {name!r} has invalid categories")
        if categories is not None and len(categories) != len(findings):
            raise CohortError(
                f"val80 case {name!r} has mismatched finding/category counts"
            )
        names.append(name)
        finding_count += len(findings)
        normalized_records.append(dict(record))

    _require_unique_names(names, "val80")
    return normalized_records, names, finding_count


def _check_expected_sha256(
    observed: str,
    expected: str | None,
    *,
    label: str,
) -> None:
    if not _SHA256_RE.fullmatch(observed):
        raise CohortError(f"{label} observed SHA-256 is malformed")
    if expected is None:
        return
    if not _SHA256_RE.fullmatch(expected):
        raise CohortError(f"{label} expected SHA-256 is malformed")
    if observed != expected:
        raise CohortError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256"] = _canonical_payload_sha256(payload)
    return result


def assert_identity_only_manifest(payload: Mapping[str, Any]) -> None:
    """Reject any holdout payload containing supervision or evaluation fields."""

    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).casefold() in _FORBIDDEN_HOLDOUT_KEYS:
                    raise CohortError(
                        f"identity-only val120 manifest contains forbidden key {key!r}"
                    )
                stack.append(nested)
        elif isinstance(value, list):
            stack.extend(value)


def seal_cohorts(
    val200_path: str | Path,
    val80_path: str | Path,
    val80_output_path: str | Path,
    val120_output_path: str | Path,
    *,
    expected_val200_cases: int = 200,
    expected_val80_cases: int = 80,
    expected_val80_findings: int = 195,
    expected_val120_cases: int = 120,
    expected_val200_sha256: str | None = None,
    expected_val80_sha256: str | None = None,
    declared_val200_findings: int | None = 381,
    split_key: str = "test",
    producer_stage: str = PRODUCER_STAGE,
) -> dict[str, Any]:
    """Write val80 development and honest identity-only val120 manifests."""

    val200_source = Path(val200_path)
    val80_source = Path(val80_path)
    val80_output = Path(val80_output_path)
    val120_output = Path(val120_output_path)
    source_paths = {val200_source.resolve(), val80_source.resolve()}
    output_paths = {val80_output.resolve(), val120_output.resolve()}
    if len(output_paths) != 2:
        raise CohortError("val80 and val120 output paths must differ")
    if source_paths & output_paths:
        raise CohortError("cohort outputs must not overwrite source manifests")

    val200_sha256 = sha256_file(val200_source)
    val80_sha256 = sha256_file(val80_source)
    _check_expected_sha256(
        val200_sha256, expected_val200_sha256, label="val200 source"
    )
    _check_expected_sha256(val80_sha256, expected_val80_sha256, label="val80 source")

    # The parent reader projects only direct case identities.  It never
    # materializes findings or categories belonging to complement cases.
    val200_names = load_case_names_only(val200_source, split_key=split_key)
    val80_records, val80_names, val80_findings = _load_development_records(
        val80_source, split_key=split_key
    )
    if val80_findings != expected_val80_findings:
        raise CohortError(
            f"val80 finding count is {val80_findings}, "
            f"expected {expected_val80_findings}"
        )

    partition = validate_partition(
        val200_names,
        val80_names,
        expected_val200_cases=expected_val200_cases,
        expected_val80_cases=expected_val80_cases,
        expected_val120_cases=expected_val120_cases,
    )

    parent_identity_provenance: dict[str, Any] = {
        "path": str(val200_source),
        "sha256": val200_sha256,
        "case_count": partition.parent_count,
    }
    parent_development_provenance = dict(parent_identity_provenance)
    if declared_val200_findings is not None:
        parent_development_provenance[
            "declared_finding_count"
        ] = declared_val200_findings

    development_manifest = _with_payload_hash(
        {
            "schema_version": "1.0",
            "manifest_type": "development_cohort",
            "cohort_id": "val80",
            "role": "development",
            "sealed": True,
            "producer_stage": producer_stage,
            "producer_stage_id": producer_stage,
            "source": {
                "path": str(val80_source),
                "sha256": val80_sha256,
            },
            "parent": parent_development_provenance,
            "case_count": partition.development_count,
            "finding_count": val80_findings,
            "case_names": list(partition.development_case_names),
            "cases": val80_records,
        }
    )
    internal_replication_manifest = _with_payload_hash(
        {
            "schema_version": "1.0",
            "manifest_type": "internal_replication_cohort",
            "cohort_id": "val120",
            "role": "historically_exposed_internal_held_out_replication",
            "independence_status": "not_independent_and_not_blinded",
            "allowed_claim": "internal_replication_only",
            "forbidden_claims": [
                "untouched_confirmation",
                "independent_confirmation",
                "external_confirmation",
            ],
            "identity_only": True,
            "access_policy": "asd000_reads_identities_only",
            "derivation": "ordered_case_name_complement_of_val80_within_val200",
            "producer_stage": producer_stage,
            "producer_stage_id": producer_stage,
            "parent": parent_identity_provenance,
            "excluded_development_source": {
                "path": str(val80_source),
                "sha256": val80_sha256,
                "case_count": partition.development_count,
            },
            "case_count": partition.internal_replication_count,
            "case_names": list(partition.internal_replication_case_names),
        }
    )
    assert_identity_only_manifest(internal_replication_manifest)

    atomic_write_json(val80_output, development_manifest)
    atomic_write_json(val120_output, internal_replication_manifest)
    val80_output_sha256 = sha256_file(val80_output)
    val120_output_sha256 = sha256_file(val120_output)

    return {
        "producer_stage": producer_stage,
        "partition": partition.to_dict(),
        "val80": {
            "path": str(val80_output),
            "sha256": val80_output_sha256,
            "case_count": partition.development_count,
            "finding_count": val80_findings,
        },
        "val120": {
            "path": str(val120_output),
            "sha256": val120_output_sha256,
            "case_count": partition.internal_replication_count,
            "identity_only": True,
            "role": "historically_exposed_internal_held_out_replication",
            "independence_status": "not_independent_and_not_blinded",
        },
    }


def seal_val120_complement(
    val200_case_names: Sequence[str],
    val80_case_names: Sequence[str],
    output_path: str | Path,
    *,
    parent_path: str,
    parent_sha256: str,
    val80_path: str,
    val80_sha256: str,
    expected_val120_cases: int | None = None,
    producer_stage: str = PRODUCER_STAGE,
) -> dict[str, Any]:
    """Write an honest names-only internal-replication complement."""

    partition = validate_partition(
        val200_case_names,
        val80_case_names,
        expected_val120_cases=expected_val120_cases,
    )
    payload = _with_payload_hash(
        {
            "schema_version": "1.0",
            "manifest_type": "internal_replication_cohort",
            "cohort_id": "val120",
            "role": "historically_exposed_internal_held_out_replication",
            "independence_status": "not_independent_and_not_blinded",
            "allowed_claim": "internal_replication_only",
            "forbidden_claims": [
                "untouched_confirmation",
                "independent_confirmation",
                "external_confirmation",
            ],
            "identity_only": True,
            "access_policy": "asd000_reads_identities_only",
            "derivation": "ordered_case_name_complement_of_val80_within_val200",
            "producer_stage": producer_stage,
            "producer_stage_id": producer_stage,
            "parent": {
                "path": parent_path,
                "sha256": parent_sha256,
                "case_count": partition.parent_count,
            },
            "excluded_development_source": {
                "path": val80_path,
                "sha256": val80_sha256,
                "case_count": partition.development_count,
            },
            "case_count": partition.internal_replication_count,
            "case_names": list(partition.internal_replication_case_names),
        }
    )
    assert_identity_only_manifest(payload)
    atomic_write_json(Path(output_path), payload)
    return payload


__all__ = [
    "CohortError",
    "CohortPartition",
    "PRODUCER_STAGE",
    "assert_identity_only_manifest",
    "derive_case_complement",
    "load_case_names_only",
    "seal_cohorts",
    "seal_val120_complement",
    "validate_partition",
]
