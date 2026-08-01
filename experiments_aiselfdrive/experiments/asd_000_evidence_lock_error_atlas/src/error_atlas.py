"""Deterministic, array-only primitives for the ASD-000 spatial error atlas.

This module deliberately performs no filesystem discovery or cohort loading.
Callers must provide already-authorized val80 arrays and the values read from
``configs/protocol.yaml``.  In particular, no complement cohort is inspected
here and no unlabeled voxel is inferred to be negative.
"""

from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from itertools import combinations, product
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SUPPORTED_CONNECTIVITY = 26
DEFAULT_BOOTSTRAP_DRAWS = 10_000


class ErrorAtlasError(ValueError):
    """Base class for invalid atlas inputs or violated scientific invariants."""


class AtlasInvariantError(ErrorAtlasError):
    """Raised when an input would make the atlas scientifically ambiguous."""


class UnverifiedLogitsError(AtlasInvariantError):
    """Raised when threshold-dependent output is requested from unverified scores."""


class Val120AccessError(AtlasInvariantError):
    """Raised when the val80-only builder is pointed at internal replication."""


class SupervisionLabel(IntEnum):
    """Stable integer representation of the conservative tri-state policy."""

    UNKNOWN = -1
    CERTIFIED_NEGATIVE = 0
    KNOWN_POSITIVE = 1


_PARTITION_KEYS = (
    "known_positive_prediction",
    "ipsilateral_nontarget_fp",
    "contralateral_fp",
    "outside_lung_fp",
    "other_certified_negative_fp",
    "unknown_prediction",
)
_OFF_LOCATION_KEYS = (
    "ipsilateral_nontarget_fp",
    "contralateral_fp",
    "outside_lung_fp",
)
_CERTIFIED_FP_KEYS = _OFF_LOCATION_KEYS + ("other_certified_negative_fp",)


def _as_binary_mask(value: Any, *, name: str) -> np.ndarray:
    """Return a strict 3-D binary mask without thresholding or post-processing."""

    array = np.asarray(value)
    if array.ndim != 3:
        raise ErrorAtlasError(f"{name} must be a 3-D array, got shape {array.shape}")
    if array.dtype == np.bool_:
        return array
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ErrorAtlasError(f"{name} must be boolean or numeric")
    if not np.all(np.isfinite(array)):
        raise ErrorAtlasError(f"{name} contains non-finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ErrorAtlasError(
            f"{name} must already be a raw binary mask; score thresholding is explicit"
        )
    return array.astype(bool, copy=False)


def _optional_mask(value: Any | None, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    mask = _as_binary_mask(value, name=name)
    if mask.shape != shape:
        raise ErrorAtlasError(
            f"{name} shape {mask.shape} does not match expected shape {shape}"
        )
    return mask


def _require_same_shape(reference: np.ndarray, **masks: np.ndarray) -> None:
    for name, mask in masks.items():
        if mask.shape != reference.shape:
            raise ErrorAtlasError(
                f"{name} shape {mask.shape} does not match {reference.shape}"
            )


def validate_affine(affine: Any) -> np.ndarray:
    """Validate and return a physical voxel-to-world affine.

    Oblique and left-right-reversing affines are valid.  The determinant is
    used for physical voxel volume, so singular transforms are rejected.
    """

    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ErrorAtlasError(f"affine must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ErrorAtlasError("affine contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-8):
        raise ErrorAtlasError("affine must have homogeneous last row [0, 0, 0, 1]")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ErrorAtlasError("affine spatial transform must be nonsingular")
    return matrix


def voxel_indices_to_world(indices_ijk: Any, affine: Any) -> np.ndarray:
    """Map one or more voxel-center indices to physical world coordinates."""

    matrix = validate_affine(affine)
    indices = np.asarray(indices_ijk, dtype=np.float64)
    single = indices.ndim == 1
    if single:
        if indices.shape != (3,):
            raise ErrorAtlasError("a single voxel index must have shape (3,)")
        indices = indices[None, :]
    elif indices.ndim != 2 or indices.shape[1] != 3:
        raise ErrorAtlasError("voxel indices must have shape (3,) or (N, 3)")
    if not np.all(np.isfinite(indices)):
        raise ErrorAtlasError("voxel indices contain non-finite values")
    homogeneous = np.concatenate(
        (indices, np.ones((indices.shape[0], 1), dtype=np.float64)), axis=1
    )
    world = homogeneous @ matrix.T
    return world[0, :3] if single else world[:, :3]


def voxel_volume_mm3(affine: Any) -> float:
    """Return the physical volume of one voxel, including oblique transforms."""

    matrix = validate_affine(affine)
    return float(abs(np.linalg.det(matrix[:3, :3])))


def _fallback_label_components_26(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Small dependency-free fallback with deterministic C-order labels."""

    labels = np.zeros(mask.shape, dtype=np.int32)
    offsets = tuple(
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
    )
    component_id = 0
    for seed in np.ndindex(mask.shape):
        if not mask[seed] or labels[seed] != 0:
            continue
        component_id += 1
        labels[seed] = component_id
        stack = [seed]
        while stack:
            current = stack.pop()
            for offset in offsets:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(3))
                if any(
                    neighbor[axis] < 0 or neighbor[axis] >= mask.shape[axis]
                    for axis in range(3)
                ):
                    continue
                if mask[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
    return labels, component_id


def label_components_26(mask: Any) -> tuple[np.ndarray, int]:
    """Label a 3-D mask with deterministic 26-neighbor connectivity."""

    binary = _as_binary_mask(mask, name="mask")
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - exercised only in minimal runtimes
        return _fallback_label_components_26(binary)
    labels, count = ndimage.label(
        binary, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    return labels.astype(np.int32, copy=False), int(count)


def reconstruct_raw_mask(component_labels: Any) -> np.ndarray:
    """Reconstruct a raw binary mask exactly as the union of component labels."""

    labels = np.asarray(component_labels)
    if labels.ndim != 3:
        raise ErrorAtlasError("component_labels must be a 3-D array")
    if not (
        np.issubdtype(labels.dtype, np.integer)
        or np.issubdtype(labels.dtype, np.floating)
    ):
        raise ErrorAtlasError("component_labels must be numeric")
    if not np.all(np.isfinite(labels)):
        raise ErrorAtlasError("component_labels contains non-finite values")
    if np.issubdtype(labels.dtype, np.floating) and not np.all(
        labels == np.floor(labels)
    ):
        raise ErrorAtlasError("component_labels must contain integer-valued labels")
    if np.any(labels < 0):
        raise ErrorAtlasError("component_labels cannot contain negative labels")
    return labels > 0


def verify_raw_mask_reconstruction(
    raw_mask: Any,
    component_labels: Any | None = None,
    *,
    require_exact: bool = True,
) -> dict[str, Any]:
    """Verify exact component-union reconstruction of an already binary mask."""

    raw = _as_binary_mask(raw_mask, name="raw_mask")
    canonical_labels, canonical_component_count = label_components_26(raw)
    if component_labels is None:
        labels = canonical_labels
        component_count = canonical_component_count
        component_partition_exact = True
    else:
        labels = np.asarray(component_labels)
        if labels.shape != raw.shape:
            raise ErrorAtlasError(
                "component_labels shape does not match the raw mask shape"
            )
        # Validate numeric/integer/nonnegative semantics before inspecting IDs.
        reconstruct_raw_mask(labels)
        positive_ids = np.unique(labels[labels > 0])
        component_count = int(positive_ids.size)
        paired_ids = np.unique(
            np.column_stack((canonical_labels[raw], labels[raw])), axis=0
        )
        canonical_pair_counts = np.bincount(
            paired_ids[:, 0].astype(np.int64, copy=False),
            minlength=canonical_component_count + 1,
        )
        component_partition_exact = (
            component_count == canonical_component_count
            and paired_ids.shape[0] == canonical_component_count
            and np.all(canonical_pair_counts[1:] == 1)
        )
    reconstructed = reconstruct_raw_mask(labels)
    mismatch_count = int(np.count_nonzero(raw != reconstructed))
    exact = mismatch_count == 0 and component_partition_exact
    report = {
        "exact": exact,
        "raw_voxel_count": int(np.count_nonzero(raw)),
        "reconstructed_voxel_count": int(np.count_nonzero(reconstructed)),
        "mismatch_voxel_count": mismatch_count,
        "component_count": component_count,
        "canonical_component_count": canonical_component_count,
        "component_partition_exact": bool(component_partition_exact),
        "connectivity": SUPPORTED_CONNECTIVITY,
    }
    if require_exact and not exact:
        raise AtlasInvariantError(
            "raw-mask reconstruction is not an exact canonical 26-component "
            f"partition (voxel mismatches={mismatch_count})"
        )
    return report


def _component_records_from_labels(
    labels: np.ndarray, component_count: int, affine: Any
) -> list[dict[str, Any]]:
    matrix = validate_affine(affine)
    unit_volume = voxel_volume_mm3(matrix)
    coordinates = np.nonzero(labels)
    component_ids = labels[coordinates].astype(np.int64, copy=False)
    counts = np.bincount(component_ids, minlength=component_count + 1)
    coordinate_sums = np.zeros((component_count + 1, 3), dtype=np.float64)
    bbox_mins = np.full((component_count + 1, 3), np.iinfo(np.int64).max)
    bbox_maxs = np.full((component_count + 1, 3), -1, dtype=np.int64)
    for axis, axis_coordinates in enumerate(coordinates):
        coordinate_sums[:, axis] = np.bincount(
            component_ids,
            weights=axis_coordinates.astype(np.float64, copy=False),
            minlength=component_count + 1,
        )
        np.minimum.at(bbox_mins[:, axis], component_ids, axis_coordinates)
        np.maximum.at(bbox_maxs[:, axis], component_ids, axis_coordinates)
    records: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        voxel_count = int(counts[component_id])
        if voxel_count == 0:
            raise AtlasInvariantError(
                f"component label {component_id} is unexpectedly empty"
            )
        bbox_min = bbox_mins[component_id]
        bbox_max_inclusive = bbox_maxs[component_id]
        centroid_voxel = coordinate_sums[component_id] / voxel_count
        centroid_world = voxel_indices_to_world(centroid_voxel, matrix)
        bbox_corners = np.asarray(
            list(
                product(
                    *(
                        (float(bbox_min[axis]), float(bbox_max_inclusive[axis]))
                        for axis in range(3)
                    )
                )
            ),
            dtype=np.float64,
        )
        world_corners = voxel_indices_to_world(bbox_corners, matrix)
        records.append(
            {
                "component_id": component_id,
                "voxel_count": voxel_count,
                "volume_mm3": float(voxel_count * unit_volume),
                "centroid_voxel_ijk": [float(value) for value in centroid_voxel],
                "centroid_world_xyz_mm": [float(value) for value in centroid_world],
                "bbox_min_ijk": [int(value) for value in bbox_min],
                "bbox_max_exclusive_ijk": [
                    int(value + 1) for value in bbox_max_inclusive
                ],
                "world_center_bounds_min_xyz_mm": [
                    float(value) for value in world_corners.min(axis=0)
                ],
                "world_center_bounds_max_xyz_mm": [
                    float(value) for value in world_corners.max(axis=0)
                ],
            }
        )
    return records


def component_records(mask: Any, affine: Any) -> list[dict[str, Any]]:
    """Describe 26-connected components in voxel and physical coordinates."""

    labels, count = label_components_26(mask)
    return _component_records_from_labels(labels, count, affine)


def build_tristate_supervision(
    known_positive_mask: Any,
    certified_negative_mask: Any,
    *,
    unknown_mask: Any | None = None,
) -> np.ndarray:
    """Build disjoint tri-state labels without complement-as-negative inference.

    When ``unknown_mask`` is omitted, every voxel not explicitly positive or
    explicitly certified negative remains unknown.  When it is supplied, the
    three masks must be disjoint and exhaustive.
    """

    positive = _as_binary_mask(known_positive_mask, name="known_positive_mask")
    negative = _as_binary_mask(
        certified_negative_mask, name="certified_negative_mask"
    )
    _require_same_shape(positive, certified_negative_mask=negative)
    if np.any(positive & negative):
        raise AtlasInvariantError(
            "known-positive and certified-negative masks must not overlap"
        )
    if unknown_mask is None:
        unknown = ~(positive | negative)
    else:
        unknown = _as_binary_mask(unknown_mask, name="unknown_mask")
        _require_same_shape(positive, unknown_mask=unknown)
        if np.any(unknown & (positive | negative)):
            raise AtlasInvariantError("unknown supervision must be disjoint")
        if not np.all(positive | negative | unknown):
            raise AtlasInvariantError(
                "explicit tri-state masks must cover the complete voxel domain"
            )
    supervision = np.full(
        positive.shape, int(SupervisionLabel.UNKNOWN), dtype=np.int8
    )
    supervision[negative] = int(SupervisionLabel.CERTIFIED_NEGATIVE)
    supervision[positive] = int(SupervisionLabel.KNOWN_POSITIVE)
    if not np.array_equal(
        supervision == int(SupervisionLabel.UNKNOWN), unknown
    ):
        raise AtlasInvariantError("unknown supervision was not preserved exactly")
    return supervision


def split_tristate_supervision(supervision: Any) -> dict[str, np.ndarray]:
    """Return explicit masks and reject any value outside the tri-state domain."""

    labels = np.asarray(supervision)
    if labels.ndim != 3:
        raise ErrorAtlasError("supervision must be a 3-D array")
    allowed = np.asarray([int(label) for label in SupervisionLabel])
    if not np.all(np.isin(labels, allowed)):
        raise ErrorAtlasError("supervision contains values outside {-1, 0, 1}")
    return {
        "known_positive": labels == int(SupervisionLabel.KNOWN_POSITIVE),
        "certified_negative": labels
        == int(SupervisionLabel.CERTIFIED_NEGATIVE),
        "unknown": labels == int(SupervisionLabel.UNKNOWN),
    }


def partition_prompt_off_location_fp(
    prediction_mask: Any,
    known_positive_mask: Any,
    *,
    prompt_target_mask: Any | None = None,
    ipsilateral_nontarget_mask: Any | None = None,
    contralateral_mask: Any | None = None,
    outside_lung_mask: Any | None = None,
    certified_negative_mask: Any | None = None,
) -> dict[str, np.ndarray]:
    """Partition predictions into disjoint evidence-aware spatial categories.

    The three named false-positive regions must be explicitly supplied from
    verified anatomy.  Predictions in target-compatible or otherwise
    uncertified tissue are returned as ``unknown_prediction`` and are never
    included in a false-positive category.
    """

    prediction = _as_binary_mask(prediction_mask, name="prediction_mask")
    positive = _as_binary_mask(known_positive_mask, name="known_positive_mask")
    _require_same_shape(prediction, known_positive_mask=positive)
    shape = prediction.shape
    target = _optional_mask(prompt_target_mask, shape, name="prompt_target_mask")
    named_regions = {
        "ipsilateral_nontarget_fp": _optional_mask(
            ipsilateral_nontarget_mask,
            shape,
            name="ipsilateral_nontarget_mask",
        ),
        "contralateral_fp": _optional_mask(
            contralateral_mask, shape, name="contralateral_mask"
        ),
        "outside_lung_fp": _optional_mask(
            outside_lung_mask, shape, name="outside_lung_mask"
        ),
    }
    for (left_name, left), (right_name, right) in combinations(
        named_regions.items(), 2
    ):
        if np.any(left & right):
            raise AtlasInvariantError(
                f"{left_name} and {right_name} certification regions overlap"
            )
    named_certified = np.logical_or.reduce(tuple(named_regions.values()))
    if prompt_target_mask is None and np.any(named_certified):
        raise AtlasInvariantError(
            "named prompt-off-location certification requires an explicit target"
        )
    if certified_negative_mask is None:
        certified = named_certified
    else:
        certified = _as_binary_mask(
            certified_negative_mask, name="certified_negative_mask"
        )
        _require_same_shape(prediction, certified_negative_mask=certified)
        if np.any(named_certified & ~certified):
            raise AtlasInvariantError(
                "named off-location regions must be subsets of certified negatives"
            )
    if np.any(positive & certified):
        raise AtlasInvariantError(
            "certified-negative anatomy overlaps released known-positive voxels"
        )
    if np.any(target & certified):
        raise AtlasInvariantError(
            "prompt-compatible target overlaps certified-negative anatomy"
        )

    partitions: dict[str, np.ndarray] = {
        "known_positive_prediction": prediction & positive,
        **{
            name: prediction & region
            for name, region in named_regions.items()
        },
        "other_certified_negative_fp": prediction
        & certified
        & ~named_certified,
        "unknown_prediction": prediction & ~(positive | certified),
    }
    partition_union = np.logical_or.reduce(tuple(partitions.values()))
    if not np.array_equal(partition_union, prediction):
        raise AtlasInvariantError("prediction partitions are not exhaustive")
    for (left_name, left), (right_name, right) in combinations(
        partitions.items(), 2
    ):
        if np.any(left & right):
            raise AtlasInvariantError(
                f"prediction partitions {left_name} and {right_name} overlap"
            )
    return {key: partitions[key] for key in _PARTITION_KEYS}


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def summarize_prompt_off_location_fp(
    partitions: Mapping[str, Any], affine: Any
) -> dict[str, Any]:
    """Summarize disjoint prompt-location partitions in physical units."""

    missing = [key for key in _PARTITION_KEYS if key not in partitions]
    if missing:
        raise ErrorAtlasError(f"missing prediction partitions: {missing}")
    normalized = {
        key: _as_binary_mask(partitions[key], name=key) for key in _PARTITION_KEYS
    }
    reference = normalized[_PARTITION_KEYS[0]]
    _require_same_shape(reference, **normalized)
    for (left_name, left), (right_name, right) in combinations(
        normalized.items(), 2
    ):
        if np.any(left & right):
            raise AtlasInvariantError(
                f"prediction partitions {left_name} and {right_name} overlap"
            )
    unit_volume = voxel_volume_mm3(affine)
    categories: dict[str, dict[str, Any]] = {}
    for key in _PARTITION_KEYS:
        voxel_count = int(np.count_nonzero(normalized[key]))
        _, component_count = label_components_26(normalized[key])
        categories[key] = {
            "voxel_count": voxel_count,
            "volume_mm3": float(voxel_count * unit_volume),
            "component_count": component_count,
        }
    off_location = np.logical_or.reduce(
        tuple(normalized[key] for key in _OFF_LOCATION_KEYS)
    )
    certified_fp = np.logical_or.reduce(
        tuple(normalized[key] for key in _CERTIFIED_FP_KEYS)
    )
    prediction = np.logical_or.reduce(tuple(normalized.values()))
    off_voxels = int(np.count_nonzero(off_location))
    certified_voxels = int(np.count_nonzero(certified_fp))
    predicted_voxels = int(np.count_nonzero(prediction))
    known_voxels = categories["known_positive_prediction"]["voxel_count"]
    _, off_components = label_components_26(off_location)
    _, certified_components = label_components_26(certified_fp)
    return {
        "categories": categories,
        "off_location_fp_voxel_count": off_voxels,
        "off_location_fp_volume_mm3": float(off_voxels * unit_volume),
        "off_location_fp_component_count": off_components,
        "certified_negative_fp_voxel_count": certified_voxels,
        "certified_negative_fp_volume_mm3": float(certified_voxels * unit_volume),
        "certified_negative_fp_component_count": certified_components,
        "prompt_off_location_fp_fraction": _ratio(
            off_voxels, predicted_voxels
        ),
        "certified_precision": _ratio(
            known_voxels, known_voxels + certified_voxels
        ),
        "predicted_voxel_count": predicted_voxels,
    }


def _dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & reference))
    denominator = int(np.count_nonzero(prediction)) + int(
        np.count_nonzero(reference)
    )
    return 1.0 if denominator == 0 else float(2 * intersection / denominator)


def finding_metrics(
    prediction_mask: Any,
    known_positive_mask: Any,
    affine: Any,
    *,
    hit_threshold: float,
    prompt_target_mask: Any | None = None,
    ipsilateral_nontarget_mask: Any | None = None,
    contralateral_mask: Any | None = None,
    outside_lung_mask: Any | None = None,
    certified_negative_mask: Any | None = None,
    anatomy_compatible_mask: Any | None = None,
) -> dict[str, Any]:
    """Compute one finding's continuity and conservative tri-state metrics."""

    if not 0.0 <= float(hit_threshold) <= 1.0:
        raise ErrorAtlasError("hit_threshold must be in [0, 1]")
    prediction = _as_binary_mask(prediction_mask, name="prediction_mask")
    positive = _as_binary_mask(known_positive_mask, name="known_positive_mask")
    _require_same_shape(prediction, known_positive_mask=positive)
    matrix = validate_affine(affine)
    shape = prediction.shape
    ipsilateral = _optional_mask(
        ipsilateral_nontarget_mask, shape, name="ipsilateral_nontarget_mask"
    )
    contralateral = _optional_mask(
        contralateral_mask, shape, name="contralateral_mask"
    )
    outside_lung = _optional_mask(
        outside_lung_mask, shape, name="outside_lung_mask"
    )
    if certified_negative_mask is None:
        certified = ipsilateral | contralateral | outside_lung
    else:
        certified = _as_binary_mask(
            certified_negative_mask, name="certified_negative_mask"
        )
        _require_same_shape(prediction, certified_negative_mask=certified)
    supervision = build_tristate_supervision(positive, certified)
    supervision_masks = split_tristate_supervision(supervision)
    partitions = partition_prompt_off_location_fp(
        prediction,
        positive,
        prompt_target_mask=prompt_target_mask,
        ipsilateral_nontarget_mask=ipsilateral,
        contralateral_mask=contralateral,
        outside_lung_mask=outside_lung,
        certified_negative_mask=certified,
    )
    partition_summary = summarize_prompt_off_location_fp(partitions, matrix)
    reconstruction = verify_raw_mask_reconstruction(prediction)
    prediction_labels, prediction_component_count = label_components_26(prediction)
    positive_labels, positive_component_count = label_components_26(positive)
    overlapping_positive_ids = np.unique(positive_labels[prediction])
    overlapping_positive_ids = overlapping_positive_ids[
        overlapping_positive_ids > 0
    ]
    positive_overlap_counts = np.bincount(
        prediction_labels.ravel(),
        weights=positive.ravel().astype(np.int8, copy=False),
        minlength=prediction_component_count + 1,
    )
    detached_ids = np.flatnonzero(positive_overlap_counts[1:] == 0) + 1
    off_location_mask = np.logical_or.reduce(
        tuple(partitions[key] for key in _OFF_LOCATION_KEYS)
    )
    off_location_component_ids = set(
        int(value)
        for value in np.unique(prediction_labels[off_location_mask])
        if int(value) > 0
    )
    detached_off_location_count = sum(
        int(component_id) in off_location_component_ids for component_id in detached_ids
    )
    predicted_voxels = int(np.count_nonzero(prediction))
    positive_voxels = int(np.count_nonzero(positive))
    true_positive_voxels = int(np.count_nonzero(prediction & positive))
    dice = _dice(prediction, positive)
    anatomy_overlap_fraction: float | None = None
    if anatomy_compatible_mask is not None:
        anatomy = _as_binary_mask(
            anatomy_compatible_mask, name="anatomy_compatible_mask"
        )
        _require_same_shape(prediction, anatomy_compatible_mask=anatomy)
        anatomy_overlap_fraction = _ratio(
            int(np.count_nonzero(prediction & anatomy)), predicted_voxels
        )
    category_metrics: dict[str, Any] = {}
    for key, values in partition_summary["categories"].items():
        category_metrics[f"{key}_voxels"] = values["voxel_count"]
        category_metrics[f"{key}_volume_mm3"] = values["volume_mm3"]
        category_metrics[f"{key}_components"] = values["component_count"]
    return {
        "raw_mask_dice": dice,
        "raw_mask_hit": bool(dice >= float(hit_threshold)),
        "prediction_empty": predicted_voxels == 0,
        "prediction_voxels": predicted_voxels,
        "predicted_volume_mm3": float(predicted_voxels * voxel_volume_mm3(matrix)),
        "known_positive_voxels": positive_voxels,
        "known_positive_volume_mm3": float(
            positive_voxels * voxel_volume_mm3(matrix)
        ),
        "known_positive_recall": _ratio(true_positive_voxels, positive_voxels),
        "certified_precision": partition_summary["certified_precision"],
        "certified_negative_fp_voxels": partition_summary[
            "certified_negative_fp_voxel_count"
        ],
        "unknown_prediction_voxels": category_metrics[
            "unknown_prediction_voxels"
        ],
        "prompt_off_location_fp_voxels": partition_summary[
            "off_location_fp_voxel_count"
        ],
        "prompt_off_location_fp_volume_mm3": partition_summary[
            "off_location_fp_volume_mm3"
        ],
        "prompt_off_location_fp_fraction": partition_summary[
            "prompt_off_location_fp_fraction"
        ],
        "prompt_off_location_fp_components": partition_summary[
            "off_location_fp_component_count"
        ],
        "prediction_component_count": prediction_component_count,
        "detached_component_count": int(detached_ids.size),
        "detached_component_fraction": _ratio(
            int(detached_ids.size), prediction_component_count
        ),
        "detached_off_location_component_count": detached_off_location_count,
        "detached_off_location_component_fraction": _ratio(
            detached_off_location_count, prediction_component_count
        ),
        "known_positive_component_count": positive_component_count,
        "candidate_recalled_component_count": int(overlapping_positive_ids.size),
        "candidate_recall": _ratio(
            int(overlapping_positive_ids.size), positive_component_count
        ),
        "anatomy_overlap_fraction": anatomy_overlap_fraction,
        "raw_mask_reconstruction_exact": reconstruction["exact"],
        "supervision_known_positive_voxels": int(
            np.count_nonzero(supervision_masks["known_positive"])
        ),
        "supervision_certified_negative_voxels": int(
            np.count_nonzero(supervision_masks["certified_negative"])
        ),
        "supervision_unknown_voxels": int(
            np.count_nonzero(supervision_masks["unknown"])
        ),
        **category_metrics,
    }


def atlas_component_records(
    prediction_mask: Any,
    known_positive_mask: Any,
    affine: Any,
    *,
    partitions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return physical component rows with evidence-category voxel counts."""

    prediction = _as_binary_mask(prediction_mask, name="prediction_mask")
    positive = _as_binary_mask(known_positive_mask, name="known_positive_mask")
    _require_same_shape(prediction, known_positive_mask=positive)
    normalized_partitions = {
        key: _as_binary_mask(partitions[key], name=key) for key in _PARTITION_KEYS
    }
    _require_same_shape(prediction, **normalized_partitions)
    labels, count = label_components_26(prediction)
    records = _component_records_from_labels(labels, count, affine)
    flattened_labels = labels.ravel()
    relation_counts_by_component = {
        key: np.bincount(
            flattened_labels[mask.ravel()], minlength=count + 1
        )
        for key, mask in normalized_partitions.items()
    }
    positive_counts = np.bincount(
        flattened_labels[positive.ravel()], minlength=count + 1
    )
    for record in records:
        component_id = int(record["component_id"])
        relation_counts = {
            key: int(relation_counts_by_component[key][component_id])
            for key in _PARTITION_KEYS
        }
        off_location_voxels = sum(
            relation_counts[key] for key in _OFF_LOCATION_KEYS
        )
        certified_voxels = sum(
            relation_counts[key] for key in _CERTIFIED_FP_KEYS
        )
        record.update(
            {
                "overlap_known_positive_voxels": int(positive_counts[component_id]),
                "relation_voxel_counts": relation_counts,
                "off_location_voxels": off_location_voxels,
                "detached_from_known_positive": bool(
                    positive_counts[component_id] == 0
                ),
                "fully_certified_negative": certified_voxels
                == int(record["voxel_count"]),
            }
        )
    return records


def _validate_score_inputs(scores: Any, *, score_kind: str) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 3:
        raise ErrorAtlasError("scores must be a 3-D array")
    if not np.all(np.isfinite(array)):
        raise ErrorAtlasError("scores contain non-finite values")
    if score_kind not in {"logit", "probability"}:
        raise ErrorAtlasError("score_kind must be 'logit' or 'probability'")
    if score_kind == "probability" and (
        np.any(array < 0.0) or np.any(array > 1.0)
    ):
        raise ErrorAtlasError("probability scores must lie in [0, 1]")
    return array


def raw_mask_from_scores(
    scores: Any,
    *,
    threshold: float,
    score_kind: str,
    inclusive: bool = True,
) -> np.ndarray:
    """Threshold verified score arrays without any spatial post-processing."""

    array = _validate_score_inputs(scores, score_kind=score_kind)
    threshold_value = float(threshold)
    if not 0.0 < threshold_value < 1.0:
        raise ErrorAtlasError("probability threshold must lie strictly in (0, 1)")
    if score_kind == "logit":
        threshold_value = math.log(threshold_value / (1.0 - threshold_value))
    return array >= threshold_value if inclusive else array > threshold_value


def verify_threshold_mask_equivalence(
    scores: Any,
    raw_mask: Any,
    *,
    threshold: float,
    score_kind: str,
    logits_verified: bool,
    inclusive: bool = True,
    require_exact: bool = True,
) -> dict[str, Any]:
    """Require verified scores and compare their threshold mask with the raw mask."""

    if not isinstance(logits_verified, (bool, np.bool_)) or not bool(
        logits_verified
    ):
        raise UnverifiedLogitsError(
            "same-pass mask equivalence requires independently verified logits"
        )
    reference = _as_binary_mask(raw_mask, name="raw_mask")
    reconstructed = raw_mask_from_scores(
        scores,
        threshold=threshold,
        score_kind=score_kind,
        inclusive=inclusive,
    )
    if reconstructed.shape != reference.shape:
        raise ErrorAtlasError("score and raw-mask shapes differ")
    mismatch_count = int(np.count_nonzero(reconstructed != reference))
    result = {
        "exact": mismatch_count == 0,
        "mismatch_voxel_count": mismatch_count,
        "threshold": float(threshold),
        "score_kind": score_kind,
        "inclusive": bool(inclusive),
    }
    if require_exact and mismatch_count:
        raise AtlasInvariantError(
            f"verified-score threshold mask differs at {mismatch_count} voxels"
        )
    return result


def threshold_curves(
    scores: Any | None,
    thresholds: Sequence[float],
    known_positive_mask: Any,
    affine: Any,
    *,
    logits_verified: bool,
    score_kind: str,
    hit_threshold: float,
    certified_negative_mask: Any | None = None,
    inclusive: bool = True,
) -> list[dict[str, Any]] | None:
    """Compute curves only from an explicitly verified complete score array.

    ``None`` scores with a false verification flag mean curves are unavailable
    and return ``None``.  Supplying scores without verification is an error,
    preventing an unverified array from being disguised as a threshold sweep.
    """

    if scores is None:
        if bool(logits_verified):
            raise UnverifiedLogitsError(
                "logits_verified is true but no score array was supplied"
            )
        return None
    if not isinstance(logits_verified, (bool, np.bool_)) or not bool(
        logits_verified
    ):
        raise UnverifiedLogitsError(
            "threshold curves are prohibited without verified logits"
        )
    score_array = _validate_score_inputs(scores, score_kind=score_kind)
    positive = _as_binary_mask(known_positive_mask, name="known_positive_mask")
    if score_array.shape != positive.shape:
        raise ErrorAtlasError("score and known-positive mask shapes differ")
    certified = _optional_mask(
        certified_negative_mask,
        positive.shape,
        name="certified_negative_mask",
    )
    build_tristate_supervision(positive, certified)
    if not 0.0 <= float(hit_threshold) <= 1.0:
        raise ErrorAtlasError("hit_threshold must be in [0, 1]")
    ordered_thresholds = sorted({float(value) for value in thresholds})
    if not ordered_thresholds:
        raise ErrorAtlasError("at least one threshold is required")
    unit_volume = voxel_volume_mm3(affine)
    positive_voxels = int(np.count_nonzero(positive))
    rows: list[dict[str, Any]] = []
    for threshold in ordered_thresholds:
        prediction = raw_mask_from_scores(
            score_array,
            threshold=threshold,
            score_kind=score_kind,
            inclusive=inclusive,
        )
        prediction_voxels = int(np.count_nonzero(prediction))
        true_positive_voxels = int(np.count_nonzero(prediction & positive))
        certified_fp_voxels = int(np.count_nonzero(prediction & certified))
        unknown_prediction_voxels = int(
            np.count_nonzero(prediction & ~(positive | certified))
        )
        dice = _dice(prediction, positive)
        rows.append(
            {
                "threshold": threshold,
                "raw_mask_dice": dice,
                "raw_mask_hit": bool(dice >= float(hit_threshold)),
                "known_positive_recall": _ratio(
                    true_positive_voxels, positive_voxels
                ),
                "certified_precision": _ratio(
                    true_positive_voxels,
                    true_positive_voxels + certified_fp_voxels,
                ),
                "prediction_empty": prediction_voxels == 0,
                "prediction_voxels": prediction_voxels,
                "predicted_volume_mm3": float(prediction_voxels * unit_volume),
                "certified_negative_fp_voxels": certified_fp_voxels,
                "unknown_prediction_voxels": unknown_prediction_voxels,
            }
        )
    return rows


def deterministic_case_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric_keys: Iterable[str],
    *,
    seed: int,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    case_key: str = "case_id",
    confidence: float = 0.95,
    cluster_weighting: str = "pooled",
) -> dict[str, Any]:
    """Bootstrap complete case clusters with deterministic sorted grouping.

    ``pooled`` reproduces a finding-level statistic while resampling cases as
    intact clusters.  ``equal_case`` averages each sampled case's finding mean.
    ``None`` is the only accepted missing value; non-finite numbers are rejected.
    """

    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ErrorAtlasError("draws must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ErrorAtlasError("seed must be an integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ErrorAtlasError("confidence must lie strictly in (0, 1)")
    if cluster_weighting not in {"pooled", "equal_case"}:
        raise ErrorAtlasError("cluster_weighting must be 'pooled' or 'equal_case'")
    metric_names = tuple(dict.fromkeys(str(key) for key in metric_keys))
    if not metric_names:
        raise ErrorAtlasError("metric_keys cannot be empty")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if case_key not in row:
            raise ErrorAtlasError(f"bootstrap row is missing {case_key!r}")
        grouped[str(row[case_key])].append(row)
    if not grouped:
        raise ErrorAtlasError("bootstrap requires at least one row")
    case_ids = sorted(grouped)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    sampled_case_indices = rng.integers(
        0, len(case_ids), size=(draws, len(case_ids)), dtype=np.int64
    )
    alpha = (1.0 - float(confidence)) / 2.0
    metric_results: dict[str, Any] = {}
    for metric_name in metric_names:
        case_sums = np.zeros(len(case_ids), dtype=np.float64)
        case_counts = np.zeros(len(case_ids), dtype=np.int64)
        for case_index, case_id in enumerate(case_ids):
            values: list[float] = []
            for row in grouped[case_id]:
                value = row.get(metric_name)
                if value is None:
                    continue
                if not isinstance(value, (bool, int, float, np.number)):
                    raise ErrorAtlasError(
                        f"bootstrap metric {metric_name!r} must be numeric or None"
                    )
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ErrorAtlasError(
                        f"bootstrap metric {metric_name!r} contains non-finite data"
                    )
                values.append(numeric)
            if values:
                case_sums[case_index] = math.fsum(sorted(values))
                case_counts[case_index] = len(values)
        if cluster_weighting == "pooled":
            draw_numerators = case_sums[sampled_case_indices].sum(axis=1)
            draw_denominators = case_counts[sampled_case_indices].sum(axis=1)
            estimate_denominator = int(case_counts.sum())
            estimate = (
                float(math.fsum(case_sums.tolist()) / estimate_denominator)
                if estimate_denominator
                else None
            )
        else:
            case_means = np.divide(
                case_sums,
                case_counts,
                out=np.zeros_like(case_sums),
                where=case_counts > 0,
            )
            sampled_valid = case_counts[sampled_case_indices] > 0
            draw_numerators = (
                case_means[sampled_case_indices] * sampled_valid
            ).sum(axis=1)
            draw_denominators = sampled_valid.sum(axis=1)
            valid_case_means = case_means[case_counts > 0]
            estimate = (
                float(math.fsum(sorted(valid_case_means.tolist()))
                / valid_case_means.size)
                if valid_case_means.size
                else None
            )
        valid_draws = draw_denominators > 0
        bootstrap_values = np.divide(
            draw_numerators[valid_draws],
            draw_denominators[valid_draws],
        )
        if bootstrap_values.size:
            lower, upper = np.quantile(
                bootstrap_values, (alpha, 1.0 - alpha), method="linear"
            )
            bootstrap_mean = float(bootstrap_values.mean())
            ci_low: float | None = float(lower)
            ci_high: float | None = float(upper)
        else:
            bootstrap_mean = None
            ci_low = None
            ci_high = None
        metric_results[metric_name] = {
            "estimate": estimate,
            "bootstrap_mean": bootstrap_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "valid_draws": int(bootstrap_values.size),
            "case_count_with_values": int(np.count_nonzero(case_counts)),
            "observation_count": int(case_counts.sum()),
        }
    return {
        "draws": draws,
        "seed": int(seed),
        "confidence": float(confidence),
        "cluster_unit": "case",
        "cluster_weighting": cluster_weighting,
        "case_count": len(case_ids),
        "case_ids_sorted": case_ids,
        "metrics": metric_results,
    }


def _assert_val80_only(value: Any) -> None:
    normalized = str(value).strip().lower().replace("-", "_")
    allowed = {"val80", "development", "val80_development"}
    if normalized not in allowed:
        if "120" in normalized or "confirm" in normalized:
            raise Val120AccessError(
                "the val120 internal-replication complement cannot be processed by ASD-000"
            )
        raise ErrorAtlasError(
            f"error-atlas cohort must be the locked val80 development set, got {value!r}"
        )


def _flatten_case_records(
    case_records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, dict[str, Any]]]:
    flattened: list[tuple[str, str, dict[str, Any]]] = []
    for case_record in case_records:
        if "case_id" not in case_record:
            raise ErrorAtlasError("each case record requires case_id")
        case_id = str(case_record["case_id"])
        if "cohort" in case_record:
            _assert_val80_only(case_record["cohort"])
        findings = case_record.get("findings")
        if findings is None:
            finding_records = (case_record,)
            inherited: dict[str, Any] = {}
        else:
            if isinstance(findings, (str, bytes, Mapping)) or not isinstance(
                findings, Sequence
            ):
                raise ErrorAtlasError("case findings must be a sequence of mappings")
            finding_records = findings
            inherited = {
                key: value
                for key, value in case_record.items()
                if key not in {"findings", "finding_id"}
            }
        for finding_record in finding_records:
            if not isinstance(finding_record, Mapping):
                raise ErrorAtlasError("each finding record must be a mapping")
            combined = dict(inherited)
            combined.update(finding_record)
            combined["case_id"] = case_id
            if "cohort" in combined:
                _assert_val80_only(combined["cohort"])
            if "finding_id" not in combined:
                raise ErrorAtlasError(
                    f"case {case_id!r} has a finding without finding_id"
                )
            finding_id = str(combined["finding_id"])
            flattened.append((case_id, finding_id, combined))
    flattened.sort(key=lambda item: (item[0], item[1]))
    identities = [(case_id, finding_id) for case_id, finding_id, _ in flattened]
    if len(identities) != len(set(identities)):
        raise AtlasInvariantError("case_id/finding_id pairs must be unique")
    if not flattened:
        raise ErrorAtlasError("at least one val80 finding record is required")
    return flattened


def _score_spec(record: Mapping[str, Any]) -> tuple[Any, str, bool] | None:
    has_logits = record.get("logits") is not None
    has_scores = record.get("scores") is not None
    if has_logits and has_scores:
        raise ErrorAtlasError("provide either logits or scores, not both")
    if not has_logits and not has_scores:
        if bool(record.get("logits_verified", False)) or bool(
            record.get("scores_verified", False)
        ):
            raise UnverifiedLogitsError(
                "a verification flag is true but no score array was supplied"
            )
        return None
    if has_logits:
        score_kind = str(record.get("score_kind", "logit"))
        verified = record.get("logits_verified", False)
        scores = record["logits"]
    else:
        score_kind = str(record.get("score_kind", "probability"))
        verified = record.get(
            "scores_verified", record.get("logits_verified", False)
        )
        scores = record["scores"]
    if not isinstance(verified, (bool, np.bool_)) or not bool(verified):
        raise UnverifiedLogitsError(
            "score arrays cannot enter the atlas until logits are verified"
        )
    return scores, score_kind, True


def _mean_of_rows(
    rows: Sequence[Mapping[str, Any]], metric_keys: Iterable[str]
) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for key in metric_keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]
        means[key] = (
            float(math.fsum(sorted(values)) / len(values)) if values else None
        )
    return means


def build_error_atlas(
    case_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    primary_threshold: float,
    hit_threshold: float,
    connectivity: int = SUPPORTED_CONNECTIVITY,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    thresholds: Sequence[float] | None = None,
    confidence: float = 0.95,
    cohort: str = "val80",
    score_threshold_inclusive: bool = True,
    expected_case_count: int | None = None,
    expected_finding_count: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-safe val80 atlas from explicit case records.

    A case may contain a ``findings`` list, with ``affine`` and masks inherited
    from the case, or a flat record may represent one finding.  The function
    never resolves paths.  Curves are included only when every finding supplies
    a verified score array whose primary-threshold mask exactly matches the
    provided raw prediction.
    """

    _assert_val80_only(cohort)
    if connectivity != SUPPORTED_CONNECTIVITY:
        raise ErrorAtlasError("ASD-000 requires 26-connectivity")
    flattened = _flatten_case_records(case_records)
    observed_case_count = len({case_id for case_id, _, _ in flattened})
    if expected_case_count is not None and observed_case_count != int(
        expected_case_count
    ):
        raise AtlasInvariantError(
            f"val80 case count is {observed_case_count}, expected "
            f"{int(expected_case_count)}"
        )
    if expected_finding_count is not None and len(flattened) != int(
        expected_finding_count
    ):
        raise AtlasInvariantError(
            f"val80 finding count is {len(flattened)}, expected "
            f"{int(expected_finding_count)}"
        )
    score_specs = [_score_spec(record) for _, _, record in flattened]
    score_count = sum(spec is not None for spec in score_specs)
    if score_count not in {0, len(flattened)}:
        raise AtlasInvariantError(
            "threshold curves require verified logits for every val80 finding"
        )
    if score_count and thresholds is None:
        raise ErrorAtlasError(
            "verified logits were supplied but no configured thresholds were given"
        )

    rows: list[dict[str, Any]] = []
    all_component_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for (case_id, finding_id, record), score_spec in zip(
        flattened, score_specs, strict=True
    ):
        required = ("prediction_mask", "known_positive_mask", "affine")
        missing = [key for key in required if key not in record]
        if missing:
            raise ErrorAtlasError(
                f"{case_id}/{finding_id} is missing required fields {missing}"
            )
        metric_kwargs = {
            key: record.get(key)
            for key in (
                "prompt_target_mask",
                "ipsilateral_nontarget_mask",
                "contralateral_mask",
                "outside_lung_mask",
                "certified_negative_mask",
                "anatomy_compatible_mask",
            )
        }
        metrics = finding_metrics(
            record["prediction_mask"],
            record["known_positive_mask"],
            record["affine"],
            hit_threshold=hit_threshold,
            **metric_kwargs,
        )
        row = {"case_id": case_id, "finding_id": finding_id, **metrics}
        rows.append(row)
        prediction = _as_binary_mask(
            record["prediction_mask"], name="prediction_mask"
        )
        positive = _as_binary_mask(
            record["known_positive_mask"], name="known_positive_mask"
        )
        shape = prediction.shape
        ipsilateral = _optional_mask(
            record.get("ipsilateral_nontarget_mask"),
            shape,
            name="ipsilateral_nontarget_mask",
        )
        contralateral = _optional_mask(
            record.get("contralateral_mask"),
            shape,
            name="contralateral_mask",
        )
        outside_lung = _optional_mask(
            record.get("outside_lung_mask"),
            shape,
            name="outside_lung_mask",
        )
        certified = record.get("certified_negative_mask")
        if certified is None:
            certified = ipsilateral | contralateral | outside_lung
        partitions = partition_prompt_off_location_fp(
            prediction,
            positive,
            prompt_target_mask=record.get("prompt_target_mask"),
            ipsilateral_nontarget_mask=ipsilateral,
            contralateral_mask=contralateral,
            outside_lung_mask=outside_lung,
            certified_negative_mask=certified,
        )
        for component in atlas_component_records(
            prediction,
            positive,
            record["affine"],
            partitions=partitions,
        ):
            all_component_rows.append(
                {"case_id": case_id, "finding_id": finding_id, **component}
            )
        if score_spec is not None:
            scores, score_kind, verified = score_spec
            verify_threshold_mask_equivalence(
                scores,
                prediction,
                threshold=primary_threshold,
                score_kind=score_kind,
                logits_verified=verified,
                inclusive=score_threshold_inclusive,
            )
            curve = threshold_curves(
                scores,
                thresholds or (),
                positive,
                record["affine"],
                logits_verified=verified,
                score_kind=score_kind,
                hit_threshold=hit_threshold,
                certified_negative_mask=certified,
                inclusive=score_threshold_inclusive,
            )
            if curve is None:
                raise AtlasInvariantError("verified logits produced no curve")
            threshold_rows.extend(
                {
                    "case_id": case_id,
                    "finding_id": finding_id,
                    **threshold_row,
                }
                for threshold_row in curve
            )

    bootstrap_metric_keys = (
        "raw_mask_dice",
        "raw_mask_hit",
        "prediction_empty",
        "known_positive_recall",
        "certified_precision",
        "predicted_volume_mm3",
        "prompt_off_location_fp_fraction",
        "detached_off_location_component_fraction",
        "candidate_recall",
        "anatomy_overlap_fraction",
    )
    bootstrap = deterministic_case_cluster_bootstrap(
        rows,
        bootstrap_metric_keys,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
        confidence=confidence,
    )
    total_keys = (
        "prediction_voxels",
        "known_positive_voxels",
        "certified_negative_fp_voxels",
        "unknown_prediction_voxels",
        "prompt_off_location_fp_voxels",
        "prediction_component_count",
        "detached_component_count",
        "detached_off_location_component_count",
    )
    metric_totals = {
        key: int(sum(int(row[key]) for row in rows)) for key in total_keys
    }
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "cohort": "val80",
        "case_count": observed_case_count,
        "finding_count": len(rows),
        "component_connectivity": SUPPORTED_CONNECTIVITY,
        "physical_units": "millimeters",
        "raw_mask_reconstruction_exact": all(
            bool(row["raw_mask_reconstruction_exact"]) for row in rows
        ),
        "summary": {
            "metric_means": _mean_of_rows(rows, bootstrap_metric_keys),
            "metric_totals": metric_totals,
            "bootstrap": bootstrap,
        },
        "rows": rows,
        "component_rows": all_component_rows,
        "logit_status": {
            "verified_complete": score_count == len(flattened),
            "verified_finding_count": score_count,
            "threshold_curves_included": bool(score_count),
        },
    }
    if threshold_rows:
        threshold_metric_keys = (
            "raw_mask_dice",
            "raw_mask_hit",
            "prediction_empty",
            "known_positive_recall",
            "certified_precision",
            "predicted_volume_mm3",
        )
        aggregate_curves: list[dict[str, Any]] = []
        for threshold in sorted(
            {float(row["threshold"]) for row in threshold_rows}
        ):
            selected = [
                row for row in threshold_rows if float(row["threshold"]) == threshold
            ]
            aggregate_curves.append(
                {
                    "threshold": threshold,
                    "metric_means": _mean_of_rows(
                        selected, threshold_metric_keys
                    ),
                    "bootstrap": deterministic_case_cluster_bootstrap(
                        selected,
                        threshold_metric_keys,
                        seed=bootstrap_seed,
                        draws=bootstrap_draws,
                        confidence=confidence,
                    ),
                }
            )
        result["threshold_rows"] = threshold_rows
        result["threshold_curves"] = aggregate_curves
    return result


__all__ = [
    "AtlasInvariantError",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "ErrorAtlasError",
    "SUPPORTED_CONNECTIVITY",
    "SupervisionLabel",
    "UnverifiedLogitsError",
    "Val120AccessError",
    "atlas_component_records",
    "build_error_atlas",
    "build_tristate_supervision",
    "component_records",
    "deterministic_case_cluster_bootstrap",
    "finding_metrics",
    "label_components_26",
    "partition_prompt_off_location_fp",
    "raw_mask_from_scores",
    "reconstruct_raw_mask",
    "split_tristate_supervision",
    "summarize_prompt_off_location_fp",
    "threshold_curves",
    "validate_affine",
    "verify_raw_mask_reconstruction",
    "verify_threshold_mask_equivalence",
    "voxel_indices_to_world",
    "voxel_volume_mm3",
]
