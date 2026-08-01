"""Conservative literal spatial-prompt ontology and RAS geometry helpers.

The parser is intentionally precision-first.  A restrictive spatial target is
emitted only when literal location spans agree and no safety guard fires.
Unknown is the default, including negation, uncertainty, bilateral or
multifocal language, diffuse language, and conflicting locations.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

try:  # Package import.
    from .lock_evidence import atomic_write_json
except ImportError:  # Direct import after adding ``src`` to sys.path.
    from lock_evidence import atomic_write_json


PARSER_VERSION = "asd_prompt_ontology_v1"
PRODUCER_STAGE = "lock_cohorts_and_prompt_ontology"
ONTOLOGY_GROUPS = (
    "exact_lobe",
    "laterality_only",
    "apical_or_basal",
    "central_or_peripheral",
    "subpleural",
    "hilar",
    "peribronchial",
    "bilateral_or_multifocal",
    "diffuse",
    "unknown",
)


class PromptOntologyError(ValueError):
    """Raised when parser input or physical geometry is malformed."""


@dataclass(frozen=True)
class SpatialTarget(Mapping[str, str]):
    """A literal, unambiguous physical target.

    Empty dimensions are omitted from the mapping representation.  The lobe
    values encode laterality (for example ``right_lower``) but the explicit
    ``laterality`` field is retained for convenient downstream checks.
    """

    laterality: str | None = None
    lobe: str | None = None
    vertical: str | None = None
    radial: str | None = None
    relation: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("laterality", self.laterality),
                ("lobe", self.lobe),
                ("vertical", self.vertical),
                ("radial", self.radial),
                ("relation", self.relation),
            )
            if value is not None
        }

    def __getitem__(self, key: str) -> str:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class PromptParse(Mapping[str, Any]):
    """Immutable parser result with both attribute and mapping access."""

    text: str
    normalized_text: str
    group: str
    target: SpatialTarget | None
    matched_literals: tuple[str, ...]
    reason_codes: tuple[str, ...]
    parser_version: str = PARSER_VERSION

    @property
    def is_restrictive(self) -> bool:
        return self.target is not None

    @property
    def restrictive_target(self) -> dict[str, str] | None:
        return None if self.target is None else self.target.to_dict()

    @property
    def confidence(self) -> str:
        return "certified_literal" if self.is_restrictive else "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_version": self.parser_version,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "group": self.group,
            "confidence": self.confidence,
            "is_restrictive": self.is_restrictive,
            "restrictive_target": self.restrictive_target,
            "matched_literals": list(self.matched_literals),
            "reason_codes": list(self.reason_codes),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class LeftRightAxis(Mapping[str, Any]):
    """Dominant voxel axis contributing to physical RAS left/right."""

    voxel_axis: int
    ras_x_mm_per_voxel: float
    positive_voxel_direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel_axis": self.voxel_axis,
            "ras_x_mm_per_voxel": self.ras_x_mm_per_voxel,
            "positive_voxel_direction": self.positive_voxel_direction,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class _Rule:
    identifier: str
    pattern: str
    dimension: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class _Match:
    identifier: str
    dimension: str | None
    value: str | None
    start: int
    end: int


def _rules(
    dimension: str | None,
    values: Mapping[str, Sequence[tuple[str, str]]],
) -> tuple[_Rule, ...]:
    return tuple(
        _Rule(identifier, pattern, dimension, value)
        for value, entries in values.items()
        for identifier, pattern in entries
    )


_NEGATION_RULES = _rules(
    None,
    {
        "unsafe": (
            ("negation:no", r"\bno\b"),
            ("negation:not", r"\bnot\b"),
            ("negation:without", r"\bwithout\b"),
            ("negation:absent", r"\babs(?:ent|ence)\b"),
            ("negation:negative_for", r"\bnegative\s+for\b"),
            ("negation:free_of", r"\bfree\s+of\b"),
            ("negation:lack", r"\black(?:s|ing|ed)?\b"),
            ("negation:neither_nor", r"\b(?:neither|nor)\b"),
            ("negation:resolved", r"\bresolved\b"),
        )
    },
)

_AMBIGUITY_RULES = _rules(
    None,
    {
        "unsafe": (
            ("ambiguity:possible", r"\bpossibl(?:e|y)\b"),
            ("ambiguity:probable", r"\bprobabl(?:e|y)\b"),
            ("ambiguity:likely", r"\b(?:un)?likely\b"),
            ("ambiguity:uncertain", r"\buncertain(?:ty)?\b"),
            ("ambiguity:indeterminate", r"\bindeterminate\b"),
            ("ambiguity:questionable", r"\bquestionable\b"),
            ("ambiguity:equivocal", r"\bequivocal\b"),
            ("ambiguity:suspected", r"\bsuspect(?:ed|ing)?\b"),
            ("ambiguity:suspicious", r"\bsuspicious\b"),
            ("ambiguity:may_might_could", r"\b(?:may|might|could)\b"),
            ("ambiguity:cannot_exclude", r"\bcannot\s+exclude\b"),
            ("ambiguity:exclude", r"\bexclud(?:e|es|ed|ing)\b"),
            ("ambiguity:except", r"\bexcept(?:ing)?\b"),
            ("ambiguity:suggestive", r"\bsuggest(?:s|ed|ive(?:\s+of)?)\b"),
            ("ambiguity:favored", r"\bfavou?red\b"),
            ("ambiguity:presumed", r"\bpresum(?:ed|ably)\b"),
            ("ambiguity:versus", r"\b(?:versus|vs\.?)\b"),
            ("ambiguity:either", r"\beither\b"),
            ("ambiguity:or", r"\bor\b"),
            ("ambiguity:predominant", r"\bpredominant(?:ly)?\b"),
            ("ambiguity:difficult_to_distinguish", r"\bdifficult\s+to\s+distinguish\b"),
            (
                "ambiguity:coordinated_lobes",
                r"\b(?:upper|middle|lower)\s+(?:lobe\s+)?and\s+"
                r"(?:upper|middle|lower)\s+lobes?\b",
            ),
        )
    },
)

_BILATERAL_MULTIFOCAL_RULES = _rules(
    None,
    {
        "unsafe": (
            ("distribution:bilateral", r"\bbilateral(?:ly)?\b"),
            (
                "distribution:both",
                r"\bboth(?:\s+(?:the\s+)?)?"
                r"(?:(?:upper|middle|lower)\s+)?"
                r"(?:lungs?|lobes?|sides?|hemithoraces)\b",
            ),
            ("distribution:both_apices_bases", r"\bboth\s+(?:lung\s+)?(?:apices|bases)\b"),
            ("distribution:multifocal", r"\bmultifocal(?:ity)?\b"),
            ("distribution:multilobar", r"\bmulti[-\s]?lobar\b"),
            ("distribution:multiple", r"\bmultiple\b"),
            ("distribution:scattered", r"\bscattered\b"),
            ("distribution:numerous", r"\bnumerous\b"),
            ("distribution:several", r"\bseveral\b"),
            (
                "distribution:coordinated_sides",
                r"\b(?:right\s+and\s+left|left\s+and\s+right)\b",
            ),
        )
    },
)

_DIFFUSE_RULES = _rules(
    None,
    {
        "unsafe": (
            ("distribution:diffuse", r"\bdiffus(?:e|ely)\b"),
            ("distribution:widespread", r"\bwidespread\b"),
            ("distribution:throughout", r"\bthroughout\b"),
            ("distribution:generalized", r"\bgenerali[sz]ed\b"),
            ("distribution:extensive", r"\bextensive(?:ly)?\b"),
        )
    },
)

_LOBE_RULES = _rules(
    "lobe",
    {
        "right_upper": (
            ("lobe:right_upper", r"\bright\s+upper\s+lobe\b"),
            (
                "lobe:right_upper_alt",
                r"\bupper\s+lobe\s+of\s+(?:the\s+)?right\s+lung\b",
            ),
            ("lobe:rul", r"\brul\b"),
        ),
        "right_middle": (
            ("lobe:right_middle", r"\bright\s+middle\s+lobe\b"),
            (
                "lobe:right_middle_alt",
                r"\bmiddle\s+lobe\s+of\s+(?:the\s+)?right\s+lung\b",
            ),
            ("lobe:middle_unpaired", r"\bmiddle\s+lobe\b"),
            ("lobe:rml", r"\brml\b"),
        ),
        "right_lower": (
            ("lobe:right_lower", r"\bright\s+lower\s+lobe\b"),
            (
                "lobe:right_lower_alt",
                r"\blower\s+lobe\s+of\s+(?:the\s+)?right\s+lung\b",
            ),
            ("lobe:rll", r"\brll\b"),
        ),
        "left_upper": (
            ("lobe:left_upper", r"\bleft\s+upper\s+lobe\b"),
            (
                "lobe:left_upper_alt",
                r"\bupper\s+lobe\s+of\s+(?:the\s+)?left\s+lung\b",
            ),
            ("lobe:lul", r"\blul\b"),
            ("lobe:lingula", r"\blingul(?:a|ar)\b"),
        ),
        "left_lower": (
            ("lobe:left_lower", r"\bleft\s+lower\s+lobe\b"),
            (
                "lobe:left_lower_alt",
                r"\blower\s+lobe\s+of\s+(?:the\s+)?left\s+lung\b",
            ),
            ("lobe:lll", r"\blll\b"),
        ),
    },
)

_LATERALITY_RULES = _rules(
    "laterality",
    {
        "right": (
            ("laterality:right_word", r"\bright\b"),
            (
                "laterality:right",
                r"\bright(?:[-\s]sided|\s+(?:lung|pulmonary|pleur(?:a|al)|"
                r"hemithorax|thorax|chest|upper|middle|lower)|\s+side\b)",
            ),
            ("laterality:on_right", r"\bon\s+the\s+right\b"),
            ("laterality:rul_rml_rll", r"\b(?:rul|rml|rll)\b"),
        ),
        "left": (
            ("laterality:left_word", r"\bleft\b"),
            (
                "laterality:left",
                r"\bleft(?:[-\s]sided|\s+(?:lung|pulmonary|pleur(?:a|al)|"
                r"hemithorax|thorax|chest|upper|lower)|\s+side\b)",
            ),
            ("laterality:on_left", r"\bon\s+the\s+left\b"),
            ("laterality:lul_lll", r"\b(?:lul|lll)\b"),
            ("laterality:lingula", r"\blingul(?:a|ar)\b"),
        ),
    },
)

_VERTICAL_RULES = _rules(
    "vertical",
    {
        "apical": (
            ("vertical:apical", r"\bapical\b"),
            ("vertical:apex", r"\bap(?:ex|ices)\b"),
            ("vertical:upper_zone", r"\bupper\s+(?:lung\s+)?zone\b"),
        ),
        "basal": (
            ("vertical:basal", r"\b(?:postero|antero|latero|medio)?basal\b"),
            ("vertical:base", r"\b(?:lung\s+)?bases?\b"),
            ("vertical:lower_zone", r"\blower\s+(?:lung\s+)?zone\b"),
        ),
    },
)

_RADIAL_RULES = _rules(
    "radial",
    {
        "central": (
            ("radial:central", r"\bcentral(?:ly)?\b"),
            ("radial:inner", r"\binner\s+(?:lung|zone)\b"),
        ),
        "peripheral": (
            ("radial:peripheral", r"\bperipheral(?:ly)?\b"),
            ("radial:outer", r"\bouter\s+(?:lung|zone)\b"),
        ),
    },
)

_RELATION_RULES = _rules(
    "relation",
    {
        "subpleural": (
            ("relation:subpleural", r"\bsub[-\s]?pleural\b"),
            ("relation:juxtapleural", r"\bjuxta[-\s]?pleural\b"),
        ),
        "hilar": (
            ("relation:hilar", r"\bhilar\b"),
            ("relation:perihilar", r"\bperi[-\s]?hilar\b"),
        ),
        "peribronchial": (
            ("relation:peribronchial", r"\bperi[-\s]?bronchial\b"),
            ("relation:peribronchovascular", r"\bperi[-\s]?bronchovascular\b"),
        ),
    },
)

_LOCATION_RULES = (
    _LOBE_RULES
    + _LATERALITY_RULES
    + _VERTICAL_RULES
    + _RADIAL_RULES
    + _RELATION_RULES
)
_COMPILED: dict[str, re.Pattern[str]] = {
    rule.identifier: re.compile(rule.pattern, flags=re.IGNORECASE)
    for rule in (
        _NEGATION_RULES
        + _AMBIGUITY_RULES
        + _BILATERAL_MULTIFOCAL_RULES
        + _DIFFUSE_RULES
        + _LOCATION_RULES
    )
}


def normalize_prompt(text: str) -> str:
    """Normalize spacing and punctuation without semantic rewriting."""

    if not isinstance(text, str):
        raise PromptOntologyError("prompt must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[_/\\]+", " ", normalized)
    normalized = re.sub(r"[;,():\[\]{}]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _matches(text: str, rules: Sequence[_Rule]) -> tuple[_Match, ...]:
    found: list[_Match] = []
    for rule in rules:
        for match in _COMPILED[rule.identifier].finditer(text):
            found.append(
                _Match(
                    identifier=rule.identifier,
                    dimension=rule.dimension,
                    value=rule.value,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(sorted(found, key=lambda item: (item.start, item.end, item.identifier)))


def _unknown(
    text: str,
    normalized: str,
    *,
    group: str = "unknown",
    matches: Sequence[_Match] = (),
    reason_codes: Sequence[str],
) -> PromptParse:
    return PromptParse(
        text=text,
        normalized_text=normalized,
        group=group,
        target=None,
        matched_literals=tuple(dict.fromkeys(match.identifier for match in matches)),
        reason_codes=tuple(reason_codes),
    )


def parse_prompt(text: str) -> PromptParse:
    """Parse one prompt under the precision-first ontology."""

    normalized = normalize_prompt(text)
    if not normalized:
        return _unknown(text, normalized, reason_codes=("empty_prompt",))

    negated = _matches(normalized, _NEGATION_RULES)
    if negated:
        return _unknown(
            text,
            normalized,
            matches=negated,
            reason_codes=("negation_guard",),
        )
    ambiguous = _matches(normalized, _AMBIGUITY_RULES)
    if ambiguous:
        return _unknown(
            text,
            normalized,
            matches=ambiguous,
            reason_codes=("ambiguity_guard",),
        )
    bilateral_or_multifocal = _matches(normalized, _BILATERAL_MULTIFOCAL_RULES)
    if bilateral_or_multifocal:
        return _unknown(
            text,
            normalized,
            group="bilateral_or_multifocal",
            matches=bilateral_or_multifocal,
            reason_codes=("bilateral_or_multifocal_guard",),
        )
    diffuse = _matches(normalized, _DIFFUSE_RULES)
    if diffuse:
        return _unknown(
            text,
            normalized,
            group="diffuse",
            matches=diffuse,
            reason_codes=("diffuse_guard",),
        )

    location_matches = _matches(normalized, _LOCATION_RULES)
    by_dimension: dict[str, set[str]] = {
        "lobe": set(),
        "laterality": set(),
        "vertical": set(),
        "radial": set(),
        "relation": set(),
    }
    for match in location_matches:
        if match.dimension is not None and match.value is not None:
            by_dimension[match.dimension].add(match.value)

    conflicting_dimensions = sorted(
        dimension for dimension, values in by_dimension.items() if len(values) > 1
    )
    if conflicting_dimensions:
        return _unknown(
            text,
            normalized,
            matches=location_matches,
            reason_codes=tuple(
                f"conflicting_{dimension}" for dimension in conflicting_dimensions
            ),
        )

    values = {
        dimension: next(iter(dimension_values), None)
        for dimension, dimension_values in by_dimension.items()
    }
    lobe = values["lobe"]
    laterality = values["laterality"]
    if lobe is not None:
        lobe_laterality = lobe.split("_", 1)[0]
        if laterality is not None and laterality != lobe_laterality:
            return _unknown(
                text,
                normalized,
                matches=location_matches,
                reason_codes=("conflicting_lobe_laterality",),
            )
        laterality = lobe_laterality

    target = SpatialTarget(
        laterality=laterality,
        lobe=lobe,
        vertical=values["vertical"],
        radial=values["radial"],
        relation=values["relation"],
    )
    if len(target) == 0:
        return _unknown(
            text,
            normalized,
            matches=location_matches,
            reason_codes=("no_safe_literal_location",),
        )

    if lobe is not None:
        group = "exact_lobe"
    elif values["relation"] is not None:
        group = values["relation"]
    elif values["vertical"] is not None:
        group = "apical_or_basal"
    elif values["radial"] is not None:
        group = "central_or_peripheral"
    elif laterality is not None:
        group = "laterality_only"
    else:  # Defensive: every target dimension above maps to a group.
        return _unknown(
            text,
            normalized,
            matches=location_matches,
            reason_codes=("unsupported_target_combination",),
        )

    return PromptParse(
        text=text,
        normalized_text=normalized,
        group=group,
        target=target,
        matched_literals=tuple(
            dict.fromkeys(match.identifier for match in location_matches)
        ),
        reason_codes=("unambiguous_literal_match",),
    )


def classify_prompt(text: str) -> dict[str, Any]:
    """Dictionary-returning compatibility wrapper around :func:`parse_prompt`."""

    return parse_prompt(text).to_dict()


def parse_prompts(texts: Sequence[str]) -> list[dict[str, Any]]:
    """Parse prompts in input order using JSON-serializable results."""

    return [classify_prompt(text) for text in texts]


def _manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_ontology_manifest(
    *,
    producer_stage: str = PRODUCER_STAGE,
) -> dict[str, Any]:
    """Build the deterministic machine-readable ontology contract."""

    def identifiers(rules: Sequence[_Rule]) -> list[str]:
        return [rule.identifier for rule in rules]

    base: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest_type": "prompt_ontology",
        "parser_version": PARSER_VERSION,
        "producer_stage": producer_stage,
        "producer_stage_id": producer_stage,
        "default_group": "unknown",
        "groups": list(ONTOLOGY_GROUPS),
        "restrictive_target_policy": (
            "literal_unambiguous_nonnegated_unilateral_nondiffuse_matches_only"
        ),
        "unsafe_precedence": [
            "negation",
            "ambiguity",
            "bilateral_or_multifocal",
            "diffuse",
            "conflicting_locations",
        ],
        "literal_rule_ids": {
            "negation": identifiers(_NEGATION_RULES),
            "ambiguity": identifiers(_AMBIGUITY_RULES),
            "bilateral_or_multifocal": identifiers(_BILATERAL_MULTIFOCAL_RULES),
            "diffuse": identifiers(_DIFFUSE_RULES),
            "location": identifiers(_LOCATION_RULES),
        },
        "physical_coordinate_system": {
            "name": "RAS",
            "left_right_axis": "world_x",
            "positive_world_x": "right",
            "negative_world_x": "left",
            "midline_required": True,
        },
    }
    return {**base, "payload_sha256": _manifest_payload_sha256(base)}


def write_ontology_manifest(
    path: str | Path,
    *,
    producer_stage: str = PRODUCER_STAGE,
) -> dict[str, Any]:
    """Atomically write and return the deterministic ontology manifest."""

    payload = build_ontology_manifest(producer_stage=producer_stage)
    atomic_write_json(Path(path), payload)
    return payload


def validate_ras_affine(
    affine: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
) -> tuple[tuple[float, float, float, float], ...]:
    """Validate a finite invertible 4x4 voxel-to-RAS affine."""

    try:
        if len(affine) != 4 or any(len(row) != 4 for row in affine):
            raise PromptOntologyError("RAS affine must be 4x4")
        matrix = tuple(tuple(float(value) for value in row) for row in affine)
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, PromptOntologyError):
            raise
        raise PromptOntologyError("RAS affine must be a numeric 4x4 matrix") from error
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise PromptOntologyError("RAS affine contains non-finite values")
    if any(
        abs(observed - expected) > tolerance
        for observed, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise PromptOntologyError("RAS affine bottom row must be [0, 0, 0, 1]")

    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinant) <= tolerance:
        raise PromptOntologyError("RAS affine spatial transform is singular")
    for axis in range(3):
        norm = math.sqrt(sum(matrix[row][axis] ** 2 for row in range(3)))
        if norm <= tolerance:
            raise PromptOntologyError(f"RAS affine voxel axis {axis} has zero spacing")
    return matrix


def voxel_to_ras(
    index_ijk: Sequence[float],
    affine: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Transform a voxel-center index into physical RAS millimetres."""

    matrix = validate_ras_affine(affine)
    try:
        if len(index_ijk) != 3:
            raise PromptOntologyError("voxel index must have three coordinates")
        index = tuple(float(value) for value in index_ijk)
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, PromptOntologyError):
            raise
        raise PromptOntologyError("voxel index must contain three numbers") from error
    if any(not math.isfinite(value) for value in index):
        raise PromptOntologyError("voxel index contains non-finite values")
    return tuple(
        sum(matrix[row][axis] * index[axis] for axis in range(3))
        + matrix[row][3]
        for row in range(3)
    )


def ras_laterality(
    ras_point_or_x: float | Sequence[float],
    *,
    midline_ras_x: float = 0.0,
    tolerance_mm: float = 1e-6,
) -> str:
    """Classify physical laterality using RAS world x, never array order."""

    try:
        is_point = not isinstance(ras_point_or_x, (str, bytes)) and hasattr(
            ras_point_or_x, "__len__"
        )
        if is_point:
            if len(ras_point_or_x) != 3:  # type: ignore[arg-type]
                raise PromptOntologyError("RAS point must have three coordinates")
            ras_x = float(ras_point_or_x[0])  # type: ignore[index]
        else:
            ras_x = float(ras_point_or_x)  # type: ignore[arg-type]
        midline = float(midline_ras_x)
        tolerance = float(tolerance_mm)
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, PromptOntologyError):
            raise
        raise PromptOntologyError("laterality inputs must be numeric") from error
    if not all(math.isfinite(value) for value in (ras_x, midline, tolerance)):
        raise PromptOntologyError("laterality inputs must be finite")
    if tolerance < 0:
        raise PromptOntologyError("laterality tolerance must be non-negative")
    delta = ras_x - midline
    if delta > tolerance:
        return "right"
    if delta < -tolerance:
        return "left"
    return "midline"


def voxel_laterality(
    index_ijk: Sequence[float],
    affine: Sequence[Sequence[float]],
    *,
    midline_ras_x: float = 0.0,
    tolerance_mm: float = 1e-6,
) -> str:
    """Transform a voxel to RAS and classify its physical laterality."""

    return ras_laterality(
        voxel_to_ras(index_ijk, affine),
        midline_ras_x=midline_ras_x,
        tolerance_mm=tolerance_mm,
    )


def left_right_axis_from_affine(
    affine: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
) -> LeftRightAxis:
    """Report the dominant voxel axis contributing to RAS world x.

    This is an orientation diagnostic only.  Spatial classification should use
    :func:`voxel_to_ras` so oblique contributions are retained.
    """

    matrix = validate_ras_affine(affine, tolerance=tolerance)
    contributions = tuple(matrix[0][axis] for axis in range(3))
    magnitudes = tuple(abs(value) for value in contributions)
    dominant = max(range(3), key=magnitudes.__getitem__)
    if magnitudes[dominant] <= tolerance:
        raise PromptOntologyError("affine has no measurable RAS left/right axis")
    sorted_magnitudes = sorted(magnitudes, reverse=True)
    if (
        len(sorted_magnitudes) > 1
        and abs(sorted_magnitudes[0] - sorted_magnitudes[1]) <= tolerance
    ):
        raise PromptOntologyError("affine has an ambiguous dominant left/right axis")
    contribution = contributions[dominant]
    return LeftRightAxis(
        voxel_axis=dominant,
        ras_x_mm_per_voxel=contribution,
        positive_voxel_direction="right" if contribution > 0 else "left",
    )


# Explicit aliases make the physical-coordinate requirement discoverable to
# callers that search for "physical" rather than "RAS".
physical_point_from_voxel = voxel_to_ras
physical_laterality = ras_laterality


__all__ = [
    "LeftRightAxis",
    "ONTOLOGY_GROUPS",
    "PARSER_VERSION",
    "PRODUCER_STAGE",
    "PromptOntologyError",
    "PromptParse",
    "SpatialTarget",
    "build_ontology_manifest",
    "classify_prompt",
    "left_right_axis_from_affine",
    "normalize_prompt",
    "parse_prompt",
    "parse_prompts",
    "physical_laterality",
    "physical_point_from_voxel",
    "ras_laterality",
    "validate_ras_affine",
    "voxel_laterality",
    "voxel_to_ras",
    "write_ontology_manifest",
]
