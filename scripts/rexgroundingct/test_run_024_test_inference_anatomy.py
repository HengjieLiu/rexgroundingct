#!/usr/bin/env python3
import unittest

import numpy as np

from run_024_test_inference_anatomy import apply_support, physical_dilation, route_prompt


class Exp024RoutingTests(unittest.TestCase):
    def test_bilateral_dominant_side_stays_bilateral(self):
        r = route_prompt("Bilateral ground glass opacities, greater on the right", "2c")
        self.assertEqual(r["scope"], "B")
        self.assertEqual(r["laterality"], "bilateral")
        self.assertEqual(set(r["selected_labels"]), {10, 11, 12, 13, 14})

    def test_explicit_multilobe_is_union_without_segment_inference(self):
        r = route_prompt("Nodule in the right upper lobe and right lower lobe", "2d")
        self.assertEqual(r["scope"], "RL+RU")
        self.assertEqual(set(r["selected_labels"]), {12, 14})

    def test_zone_does_not_create_fine_support(self):
        r = route_prompt("Opacity in the left upper zone", "2b")
        self.assertTrue(r["fine_eligible"])
        self.assertEqual(r["scope"], "L")
        self.assertNotEqual(r["scope"], "LU")
        self.assertTrue(r["eligible"])

    def test_pleural_target_is_noop(self):
        r = route_prompt("Small right pleural effusion", "2e")
        self.assertFalse(r["eligible"])

    def test_anisotropic_physical_dilation(self):
        mask = np.zeros((5, 5, 5), dtype=bool); mask[2, 2, 2] = True
        # A one-voxel move along z is 3 mm; x/y are 0.5 mm.
        affine = np.diag([0.5, 0.5, 3.0, 1.0])
        out = physical_dilation(mask, affine, 2.0)
        self.assertTrue(out[2, 2, 2])
        self.assertTrue(out[1, 2, 2])
        self.assertFalse(out[2, 2, 1])

    def test_empty_mask_and_noop_identity(self):
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        self.assertFalse(physical_dilation(mask, np.eye(4), 20).any())
        pred = np.zeros((3, 3, 3), dtype=np.uint8); pred[1, 1, 1] = 1
        route = route_prompt("Pleural thickening", "2e")
        got = apply_support(pred, mask, np.eye(4), route, "fine")
        np.testing.assert_array_equal(got, pred)
        np.testing.assert_array_equal(apply_support(pred, mask, np.eye(4), route, "unchanged"), pred)


if __name__ == "__main__":
    unittest.main()
