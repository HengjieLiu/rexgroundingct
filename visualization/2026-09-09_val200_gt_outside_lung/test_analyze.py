import importlib.util
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np

spec = importlib.util.spec_from_file_location('outside_analysis', Path(__file__).with_name('analyze.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnalysisTests(unittest.TestCase):
    def test_inside_partial_outside_and_empty(self):
        lung = np.array([True, True, False, False])
        for gt, outside, pct in [([1,1,0,0],0,0), ([1,0,1,0],1,50),
                                 ([0,0,1,1],2,100), ([0,0,0,0],0,None)]:
            r = a.measure(np.array(gt, dtype=bool), lung, 6)
            self.assertEqual(r['outside_voxels'], outside)
            self.assertEqual(r['outside_percentage'], pct)
            self.assertEqual(r['outside_volume_mm3'], outside*6)
            self.assertAlmostEqual(r['outside_volume_ml'], outside*.006)
            self.assertEqual(r['status'], 'EMPTY_GT' if pct is None else 'OK')

    def test_instance_union_and_invalid_gt(self):
        np.testing.assert_array_equal(a.foreground(np.array([0,1,2,17])), [False,True,True,True])
        for values in ([0,-1], [0,np.nan], [0,.5]):
            with self.assertRaises(ValueError):
                a.foreground(np.array(values))

    def test_case_union_avoids_double_counting(self):
        masks = np.array([[True,True,False], [False,True,True]])
        lung = np.array([True,False,False])
        rows = [a.measure(m, lung, 2) for m in masks]
        union = a.measure(np.any(masks, axis=0), lung, 2)
        self.assertEqual(sum(r['outside_voxels'] for r in rows), 3)
        self.assertEqual(union['outside_voxels'], 2)
        self.assertEqual(union['total_gt_voxels'], 3)

    def test_volume_anisotropic_and_handedness(self):
        self.assertAlmostEqual(a.voxel_volume(np.diag([-.5,.8,2,1]), 'mm'), .8)
        with self.assertRaises(ValueError):
            a.voxel_volume(np.eye(4), 'unknown')
        with self.assertRaises(ValueError):
            a.voxel_volume(np.diag([0,1,1,1]), 'mm')

    def test_mixed_spacing_aggregation(self):
        rows = [dict(category='1a', **a.measure(np.ones(4,dtype=bool), np.zeros(4,dtype=bool), vv))
                for vv in (1, 3)]
        s = a.summarize(rows)[0]
        self.assertEqual(s['M_over_N'], '2/2')
        self.assertEqual(s['outside_voxels_sum'], 8)
        self.assertEqual(s['outside_volume_mm3_sum'], 16)
        self.assertAlmostEqual(s['outside_volume_ml_sum'], .016)
        self.assertEqual(a.summarize(rows)[11]['M_over_N'], '0/0')

    def test_geometry(self):
        affine = np.diag([.5,.8,2,1])
        ct = nib.Nifti1Image(np.zeros((2,3,4)), affine)
        anatomy = nib.Nifti1Image(np.zeros((2,3,4)), affine)
        for img in (ct, anatomy):
            img.header.set_xyzt_units('mm')
        gt = nib.Nifti1Image(np.zeros((1,2,3,4)), np.eye(4))
        self.assertAlmostEqual(a.geometry(ct, anatomy, gt, 1), .8)
        bad = nib.Nifti1Image(np.zeros((2,3,4)), np.eye(4))
        with self.assertRaises(ValueError):
            a.geometry(ct, bad, gt, 1)
        with self.assertRaises(ValueError):
            a.geometry(ct, anatomy, gt, 2)

    def test_histogram_population_and_singleton(self):
        for values in ([0,0,1,1,10,100], [0,0], [2], []):
            fig, ax = a.plt.subplots()
            bins = a.positive_bins(values)
            result = a.histogram(ax, values, bins, 'voxels')
            if values:
                self.assertEqual(result['N'], len(values))
                self.assertEqual(result['positive'], sum(v>0 for v in values))
            a.plt.close(fig)

    def test_ecdf_includes_zeros_and_full_mass(self):
        fig, ax = a.plt.subplots()
        a.ecdf(ax, [0,0,2,5], 'voxels', 6)
        x, y = ax.lines[0].get_data()
        self.assertEqual(y[-1], 1)
        self.assertEqual(y[np.where(x==0)[0][-1]], .5)
        self.assertTrue(all(t == 0 or t >= 1 for t in ax.get_xticks()))
        a.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
