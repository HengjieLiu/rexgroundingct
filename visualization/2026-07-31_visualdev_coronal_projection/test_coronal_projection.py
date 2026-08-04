#!/usr/bin/env python3
"""Focused synthetic tests for coronal projection visualization."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from coronal_projection import (
    DEPTH_MISMATCH_COLOR,
    FN_COLOR,
    FP_COLOR,
    MODEL_SPECS,
    OFFICIAL_CATEGORY_CODES,
    TP_COLOR,
    VAL200_CATEGORY_MODE,
    category_directory_name,
    category_figure_path,
    coronal_ct_projection_xz,
    coronal_mask_projection_xz,
    finding_ids_by_category,
    projected_error_rgba,
    radiology_coronal_display,
    resolve_projection_methods,
)


class CoronalProjectionTests(unittest.TestCase):
    def test_model_order_matches_requested_comparison(self) -> None:
        self.assertEqual(
            [model.key for model in MODEL_SPECS],
            ["public_v11", "exp009_s3v1_e100", "noddp_best", "ddp_best"],
        )

    def test_findings_are_grouped_by_category_in_numeric_finding_order(self) -> None:
        metadata_entry = {
            "name": "train_1_a_1.nii.gz",
            "findings": {"2": "third", "0": "first", "1": "second"},
            "categories": {"0": "2d", "1": "1a", "2": "2d"},
        }
        self.assertEqual(
            finding_ids_by_category(metadata_entry),
            {"2d": ["0", "2"], "1a": ["1"]},
        )

    def test_category_directories_follow_official_codes_with_empty_2f(self) -> None:
        self.assertEqual(
            [category_directory_name(code) for code in OFFICIAL_CATEGORY_CODES],
            [
                "1a",
                "1b",
                "1c",
                "1d",
                "1e",
                "1f",
                "2a",
                "2b",
                "2c",
                "2d",
                "2e",
                "2f_empty",
                "2g",
                "2h",
            ],
        )

    def test_category_figure_path_uses_val_first_filename(self) -> None:
        entry = {"name": "train_19753_a_2.nii.gz", "reshuffled_val_index": 0}
        self.assertEqual(
            category_figure_path(Path("/output"), "1e", entry),
            Path("/output/1e/val000_train_19753_a_2.png"),
        )

    def test_full_val_category_mode_enforces_mean_only(self) -> None:
        self.assertEqual(resolve_projection_methods(VAL200_CATEGORY_MODE, None), ("mean",))
        self.assertEqual(resolve_projection_methods(VAL200_CATEGORY_MODE, ["mean"]), ("mean",))
        with self.assertRaises(ValueError):
            resolve_projection_methods(VAL200_CATEGORY_MODE, ["p75"])

    def test_mask_mip_collapses_anterior_posterior_axis(self) -> None:
        mask = np.zeros((3, 4, 5), dtype=np.uint8)
        mask[0, 1, 2] = 1
        mask[2, 3, 4] = 1
        projected = coronal_mask_projection_xz(mask)
        self.assertEqual(projected.shape, (3, 5))
        self.assertTrue(projected[0, 2])
        self.assertTrue(projected[2, 4])
        self.assertEqual(int(projected.sum()), 2)

    def test_projected_error_rgba_colors_tp_only_ray_green(self) -> None:
        gt = np.zeros((1, 3, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[0, 1, 0] = 1
        pred[0, 1, 0] = 1
        np.testing.assert_allclose(projected_error_rgba(gt, pred)[0, 0], TP_COLOR)

    def test_projected_error_rgba_colors_fp_only_ray_red(self) -> None:
        gt = np.zeros((1, 3, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        pred[0, 1, 0] = 1
        np.testing.assert_allclose(projected_error_rgba(gt, pred)[0, 0], FP_COLOR)

    def test_projected_error_rgba_colors_fn_only_ray_blue(self) -> None:
        gt = np.zeros((1, 3, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[0, 1, 0] = 1
        np.testing.assert_allclose(projected_error_rgba(gt, pred)[0, 0], FN_COLOR)

    def test_projected_error_rgba_colors_depth_disjoint_fn_fp_ray_purple(self) -> None:
        gt = np.zeros((1, 3, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[0, 0, 0] = 1
        pred[0, 2, 0] = 1
        np.testing.assert_allclose(projected_error_rgba(gt, pred)[0, 0], DEPTH_MISMATCH_COLOR)

    def test_projected_error_rgba_keeps_real_tp_green_with_ap_offset_errors(self) -> None:
        gt = np.zeros((1, 4, 1), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[0, 0, 0] = 1
        gt[0, 1, 0] = 1
        pred[0, 1, 0] = 1
        pred[0, 2, 0] = 1
        np.testing.assert_allclose(projected_error_rgba(gt, pred)[0, 0], TP_COLOR)

    def test_radiology_display_has_superior_up_and_patient_right_left(self) -> None:
        # RAS X increases toward patient right and Z increases toward superior.
        xz = np.fromfunction(lambda x, z: 10 * x + z, (3, 4), dtype=int)
        displayed = radiology_coronal_display(xz)
        self.assertEqual(displayed.shape, (4, 3))
        self.assertEqual(displayed[0, 0], 23)  # top-left: superior + patient right
        self.assertEqual(displayed[0, -1], 3)  # top-right: superior + patient left
        self.assertEqual(displayed[-1, 0], 20)  # bottom-left: inferior + patient right
        self.assertEqual(displayed[-1, -1], 0)  # bottom-right: inferior + patient left

    def test_p75_is_not_dominated_by_one_high_density_voxel(self) -> None:
        ct = np.full((1, 100, 1), -800.0, dtype=np.float32)
        ct[0, 0, 0] = 1000.0
        projected = coronal_ct_projection_xz(ct, method="p75")
        expected = (-800.0 - (-1350.0)) / 1500.0
        self.assertAlmostEqual(float(projected[0, 0]), expected, places=5)
        self.assertLess(float(projected[0, 0]), 0.5)

    def test_mean_projection_uses_lung_window(self) -> None:
        ct = np.array([[[ -1350.0], [150.0]]], dtype=np.float32)
        projected = coronal_ct_projection_xz(ct, method="mean")
        self.assertAlmostEqual(float(projected[0, 0]), 0.5, places=6)

    def test_unknown_projection_method_fails(self) -> None:
        with self.assertRaises(ValueError):
            coronal_ct_projection_xz(np.zeros((2, 2, 2), dtype=np.float32), method="maximum")


if __name__ == "__main__":
    unittest.main()
