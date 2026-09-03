#!/usr/bin/env python3
"""Focused tests for Experiment 019 bbox geometry."""

from __future__ import annotations

import unittest

import numpy as np

from audit_iso07_lung_bbox_coverage import (
    bbox_contains,
    bbox_from_mask,
    directional_deficits,
    expand_bbox,
    isotropic_expansion,
)


class BboxGeometryTests(unittest.TestCase):
    def test_bbox_from_mask_uses_inclusive_bounds(self) -> None:
        mask = np.zeros((6, 7, 8), dtype=np.uint8)
        mask[2:5, 1:4, 3:7] = 1
        self.assertEqual(bbox_from_mask(mask), [[2, 4], [1, 3], [3, 6]])

    def test_complete_containment_requires_zero_expansion(self) -> None:
        lung = [[2, 9], [3, 10], [4, 11]]
        target = [[3, 8], [4, 9], [5, 10]]
        deficits = directional_deficits(lung, target)
        self.assertEqual(deficits, {"L": 0, "R": 0, "A": 0, "P": 0, "S": 0, "I": 0})
        self.assertEqual(isotropic_expansion(deficits), 0)

    def test_one_face_overflow_needs_one_voxel(self) -> None:
        lung = [[2, 9], [3, 10], [4, 11]]
        target = [[3, 8], [4, 9], [5, 12]]
        deficits = directional_deficits(lung, target)
        self.assertEqual(deficits["R"], 1)
        self.assertEqual(isotropic_expansion(deficits), 1)

    def test_asymmetric_deficits_use_the_maximum_for_all_faces(self) -> None:
        lung = [[10, 20], [10, 20], [10, 20]]
        target = [[5, 23], [9, 25], [10, 21]]
        deficits = directional_deficits(lung, target)
        self.assertEqual(deficits, {"L": 0, "R": 1, "A": 5, "P": 1, "S": 3, "I": 5})
        self.assertEqual(isotropic_expansion(deficits), 5)
        expanded = expand_bbox(lung, 5, (40, 40, 40))
        self.assertEqual(expanded, [[5, 25], [5, 25], [5, 25]])
        self.assertTrue(bbox_contains(expanded, target))

    def test_expansion_clips_at_image_boundaries_without_losing_containment(self) -> None:
        lung = [[0, 3], [0, 4], [0, 5]]
        target = [[0, 4], [0, 5], [0, 6]]
        deficits = directional_deficits(lung, target)
        self.assertEqual(isotropic_expansion(deficits), 1)
        expanded = expand_bbox(lung, 1, (5, 6, 7))
        self.assertEqual(expanded, [[0, 4], [0, 5], [0, 6]])
        self.assertTrue(bbox_contains(expanded, target))

    def test_orientation_axis_mapping_matches_fzyx_ras(self) -> None:
        # Array axis 0 is S/I, axis 1 is A/P, and axis 2 is R/L after RAS FZYX.
        lung = [[10, 20], [10, 20], [10, 20]]
        target = [[8, 22], [9, 24], [7, 23]]
        self.assertEqual(
            directional_deficits(lung, target),
            {"L": 3, "R": 3, "A": 4, "P": 1, "S": 2, "I": 2},
        )

    def test_orientation_reversal_is_reflected_in_the_final_bbox(self) -> None:
        # Simulate a native FXYZ mask whose X axis is reversed during RAS
        # reorientation, then transpose it to the audit's FZYX layout.
        native_fxyz = np.zeros((6, 5, 4), dtype=np.uint8)
        native_fxyz[1, 2, 3] = 1
        reoriented_fxyz = np.flip(native_fxyz, axis=0)
        ras_fzyx = np.transpose(reoriented_fxyz, (2, 1, 0))
        self.assertEqual(bbox_from_mask(ras_fzyx), [[3, 3], [2, 2], [4, 4]])

        # Without applying the reversal, the same target would appear three
        # voxels away on the R face instead of being exactly contained.
        target = [[3, 3], [2, 2], [4, 4]]
        self.assertEqual(
            isotropic_expansion(directional_deficits(bbox_from_mask(ras_fzyx), target)),
            0,
        )

    def test_empty_mask_has_no_bbox(self) -> None:
        with self.assertRaises(ValueError):
            bbox_from_mask(np.zeros((2, 2, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
