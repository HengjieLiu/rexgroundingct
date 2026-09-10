import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('expanded_analysis', Path(__file__).with_name('analyze_20mm.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class ExpandedTests(unittest.TestCase):
    def test_integer_coverage_recovery(self):
        c = dict(gt_voxels=10, gt_coverage=.30000000000000004, selected=True, labels=a.base.LUNG_LABELS)
        self.assertEqual(a.outside_count(c,10),7)
        self.assertEqual(a.outside_count(dict(c,gt_coverage=1),10),0)
        self.assertEqual(a.outside_count(dict(c,gt_coverage=0),10),10)
        for changes in (dict(gt_coverage=.333), dict(gt_coverage=float('nan')),
                        dict(gt_voxels=9), dict(labels=[10]), dict(selected=False)):
            with self.assertRaises(ValueError):
                a.outside_count(dict(c,**changes),10)

    def test_monotonicity_and_eligibility(self):
        old = a.base.measure(np.ones(10,dtype=bool),np.zeros(10,dtype=bool),2)
        active = a.expanded_row(old,3,True)
        bypass = a.expanded_row(old,3,False)
        self.assertEqual(active['baseline_0mm_outside_voxels'],10)
        self.assertEqual(active['outside_percentage'],30)
        self.assertEqual(active['method2_gt_at_risk_voxels'],3)
        self.assertEqual(bypass['outside_voxels'],3)
        self.assertEqual(bypass['method2_gt_at_risk_voxels'],0)
        self.assertAlmostEqual(active['outside_volume_ml'],.006)
        with self.assertRaises(ValueError):
            a.expanded_row(old,11,True)

    def test_native_anisotropic_20mm_boundary(self):
        lung = np.zeros((25,25,25),dtype=bool)
        lung[12,12,12] = True
        affine = np.diag([-2,5,10,1])
        expanded = a.pp2.physical_dilation(lung,affine,20)
        for point in ((22,12,12),(12,16,12),(12,12,14)):
            self.assertTrue(expanded[point])
        for point in ((23,12,12),(12,17,12),(12,12,15),(22,13,12)):
            self.assertFalse(expanded[point])
        self.assertTrue(np.all(expanded[lung]))

    def test_direct_selection_covers_all_remaining_scans(self):
        rows = [dict(case='outside',category='1a',validation_index=9,has_outside=True),
                dict(case='first',category='1a',validation_index=0,has_outside=False),
                dict(case='later',category='1a',validation_index=3,has_outside=False),
                dict(case='other',category='2a',validation_index=4,has_outside=False)]
        affected, selected = a.select_direct_cases(rows)
        self.assertEqual(affected,{'outside'})
        self.assertEqual(selected,{'outside','first','other'})

    def test_fully_contained_union(self):
        support = np.array([True,True,True,False])
        masks = np.array([[True,False,True,False],[False,True,True,False]])
        self.assertTrue(all(a.base.measure(m,support,1)['outside_voxels']==0 for m in masks))
        self.assertEqual(a.base.measure(np.any(masks,axis=0),support,1)['outside_voxels'],0)


if __name__ == '__main__':
    unittest.main()
