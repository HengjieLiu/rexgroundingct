"""Scientific regression tests for the single-case visualization extension."""
import unittest
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

import render_comparison as render


class ProjectionTests(unittest.TestCase):
    def test_gt_positive_instances_are_unioned_but_predictions_must_be_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.nii.gz"
            labels = np.array([0, 1, 2], dtype=np.uint8).reshape(1, 3, 1, 1)
            nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
            np.testing.assert_array_equal(render.load_gt(path), labels > 0)
            with self.assertRaises(ValueError):
                render.load_binary(path)

    def test_lung_average_excludes_nonlung_and_handles_empty_rays(self):
        # Window limits are -1350 and +150 HU. Nonlung high intensities must
        # neither enter the numerator nor inflate the denominator.
        ct = np.array([[[-1350, 150], [150, 150], [150, 150]]], dtype=np.float32)
        lung = np.array([[[True, False], [True, False], [False, False]]])
        np.testing.assert_allclose(render.lung_mean_projection(ct, lung), [[0.5, 0]])
        np.testing.assert_array_equal(ct, [[[-1350, 150], [150, 150], [150, 150]]])

    def test_all_lung_matches_original_mean(self):
        ct = np.arange(24, dtype=np.float32).reshape(2, 4, 3) * 100 - 1500
        np.testing.assert_allclose(render.lung_mean_projection(ct, np.ones_like(ct, dtype=bool)),
                                   render.july.coronal_ct_projection_xz(ct, "mean"))

    def test_tp_priority_and_depth_disjoint_errors(self):
        gt, pred = np.zeros((4, 3, 1), bool), np.zeros((4, 3, 1), bool)
        gt[0, :2, 0] = True
        pred[0, 1:, 0] = True  # TP, FP, FN all on one ray: green.
        pred[1, 0, 0] = True  # FP only.
        gt[2, 0, 0] = True  # FN only.
        gt[3, 0, 0], pred[3, 2, 0] = True, True  # No 3D overlap: purple.
        masks = render.july.coronal_projected_error_masks_xz(gt, pred)
        for i, mask in enumerate(masks):
            np.testing.assert_array_equal(mask[:, 0], np.arange(4) == i)
        rgba = render.july.projected_error_rgba(gt, pred)
        np.testing.assert_allclose(rgba[0], [render.july.DEPTH_MISMATCH_COLOR, render.july.FN_COLOR,
                                           render.july.FP_COLOR, render.july.TP_COLOR])


class SupportTests(unittest.TestCase):
    def test_twenty_mm_uses_physical_distance_on_anisotropic_grid(self):
        anatomy = np.zeros((45, 15, 5), dtype=np.uint8)
        anatomy[22, 7, 2] = 10
        affine = np.diag([1., 4., 12., 1.])
        support = render.anatomy_support(anatomy, affine, render.UPPER_LABELS)
        self.assertTrue(support[42, 7, 2])  # 20 mm along X.
        self.assertFalse(support[43, 7, 2])
        self.assertTrue(support[22, 12, 2])  # 20 mm along Y.
        self.assertFalse(support[22, 13, 2])
        self.assertTrue(support[22, 7, 3])
        self.assertFalse(support[22, 7, 4])  # 24 mm along Z.
        self.assertFalse(support[42, 8, 2])  # Euclidean, not a rectangular box.

    def test_upper_support_excludes_lower_and_middle_lobes(self):
        anatomy = np.array([10, 11, 12, 13, 14], dtype=np.uint8).reshape(5, 1, 1)
        affine = np.diag([50., 50., 50., 1.])
        np.testing.assert_array_equal(render.anatomy_support(anatomy, affine, render.UPPER_LABELS).ravel(),
                                      [True, False, True, False, False])
        self.assertTrue(render.anatomy_support(anatomy, affine, render.WHOLE_LABELS).all())
        self.assertFalse(render.anatomy_support(np.zeros_like(anatomy), affine, render.UPPER_LABELS).any())


class LayoutTests(unittest.TestCase):
    def test_layout_and_background_only_change(self):
        gt, pred = np.zeros((3, 4, 2), bool), np.zeros((3, 4, 2), bool)
        gt[1, 0, 1] = True
        pred[1, 3, 1] = True  # depth mismatch remains even on an empty CT ray.
        overlay = render.july.projected_error_rgba(gt, pred)
        grid = [[overlay] * 3 for _ in range(3)]
        scores = [[render.dice_score(gt, pred)] * 3 for _ in range(3)]
        images = []
        for i, method in enumerate(render.METHODS):
            fig, axes = render.create_figure(np.full((2, 3), i, dtype=float), np.zeros((2, 3, 4)),
                                             grid, scores, list(render.ROW_KEYS), 2.0, "Synthetic case", method)
            try:
                self.assertEqual(axes.shape, (3, 5))
                self.assertEqual(sum(not ax.images for ax in axes.flat), 4)
                for r, c in render.BLANK_CELLS:
                    self.assertFalse(axes[r, c].texts)
                    self.assertEqual(axes[r, c].get_title(), "")
                self.assertEqual(axes[2, 4].get_aspect(), 2.0)
                images.append(np.asarray(axes[2, 4].images[1].get_array()))
                self.assertIn("3D Dice", axes[2, 4].texts[0].get_text())
            finally:
                render.july.plt.close(fig)
        np.testing.assert_array_equal(images[0], images[1])

    def test_case_index_is_parsed_separately_from_filename(self):
        self.assertEqual(render.normalize_case("val189_train_3026_a_2"), ("train_3026_a_2.nii.gz", 189))
        self.assertEqual(render.normalize_case("train_3026_a_2.nii.gz"), ("train_3026_a_2.nii.gz", None))
        with self.assertRaises(ValueError):
            render.normalize_case("../train_3026_a_2")


if __name__ == "__main__":
    unittest.main()
