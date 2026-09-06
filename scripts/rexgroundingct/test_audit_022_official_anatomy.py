#!/usr/bin/env python3
"""Small synthetic gates; no private data or inference required."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from audit_022_official_anatomy import annotate, census, geometry, metrics, scope_labels, side_labels, support_distances
from report_022_official_anatomy import bootstrap, summarize
from visualize_022_official_anatomy import select


class AuditTests(unittest.TestCase):
    def test_noop(self):
        g = np.array([1, 1, 0], bool)
        p = np.array([1, 0, 1], bool)
        m = metrics(g, p, np.ones(3, bool))
        self.assertAlmostEqual(m['dice'], .5, places=6)
        self.assertEqual((m['tp_removed'], m['fp_removed']), (0, 0))

    def test_accounting(self):
        g = np.array([1, 1, 0, 0], bool)
        p = np.array([1, 1, 1, 0], bool)
        m = metrics(g, p, np.array([1, 0, 0, 1], bool))
        self.assertEqual((m['tp_removed'], m['fp_removed'], m['gt_coverage']), (1, 1, .5))

    def test_complete_exclusion(self):
        m = metrics(np.ones(1, bool), np.ones(1, bool), np.zeros(1, bool))
        self.assertTrue(m['gt_completely_excluded'])
        self.assertTrue(m['prediction_emptied'])
        self.assertFalse(m['hit'])

    def test_empty_inputs(self):
        m = metrics(np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool))
        self.assertEqual(m['dice'], 1.)

    def test_empty_anatomy(self):
        ds, db = support_distances(np.zeros((3, 3, 3), bool), np.array([[1, 1, 1]]), np.ones(3))
        self.assertTrue(np.isinf(ds).all() and np.isinf(db).all())

    def test_anisotropic_distance(self):
        mask = np.zeros((7, 7, 7), bool)
        mask[3, 3, 3] = True
        ds, db = support_distances(mask, np.array([[4, 3, 3], [3, 4, 3], [3, 3, 4], [4, 4, 3]]), np.array([1, 2, 3]))
        np.testing.assert_allclose(ds, [1, 2, 3, np.sqrt(5)])
        np.testing.assert_allclose(db, [1, 2, 3, 2])

    def test_kdtree_equals_edt(self):
        rng = np.random.default_rng(4)
        mask = rng.random((12, 9, 7)) > .75
        coords = np.indices(mask.shape).reshape(3, -1).T
        spacing = np.array([.7, 1.2, 2.1])
        ds, _ = support_distances(mask, coords, spacing)
        expected = distance_transform_edt(~mask, sampling=spacing)
        np.testing.assert_allclose(ds, expected[tuple(coords.T)], atol=1e-10)

    def test_multiple_lobes(self):
        self.assertEqual(scope_labels('RM+LU+LL'), [10, 11, 13])
        self.assertEqual(side_labels(scope_labels('RU+RL')), [12, 13, 14])
        self.assertEqual(side_labels(scope_labels('RU+LL')), [10, 11, 12, 13, 14])

    def test_duplicate_keys(self):
        e = dict(name='x', findings={'0': 'x'}, categories={'0': '2d'})
        with self.assertRaises(ValueError):
            census({'test': [e, e]})

    def test_missing_category(self):
        with self.assertRaises(ValueError):
            census({'test': [dict(name='x', findings={'0': 'x'}, categories={})]})

    def test_semantic_review_not_keyword_routing(self):
        prompts = ['Nodules in both lungs, largest in right lower lobe',
                   'Right lung base nodule', 'Left pleural effusion at lower lobe level']
        rows = [dict(id=i, prompt=p, category='2d') for i, p in enumerate(prompts)]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'review.tsv'
            f.write_text('000 B multifocal | Largest is not exclusive.\n001 R focal | Base is not a lobe.\n002 N focal | Pleural target.\n')
            r = annotate(rows, f)
        self.assertEqual(r[0]['labels'], [10, 11, 12, 13, 14])
        self.assertEqual(r[1]['labels'], [12, 13, 14])
        self.assertFalse(r[2]['eligible'])

    def test_incomplete_review(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'x'; f.write_text('')
            with self.assertRaises(ValueError):
                annotate([dict(id=0)], f)

    def test_geometry(self):
        ct = nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.diag([.7, .8, 2, 1]))
        gt = nib.Nifti1Image(np.zeros((2, 3, 4, 5), np.uint8), np.eye(4))
        np.testing.assert_allclose(geometry(ct, ct, gt, gt, [0, 1]), [.7, .8, 2])
        with self.assertRaises(ValueError):
            geometry(ct, ct, gt, gt, [0])

    def test_wrong_axis(self):
        ct = nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.eye(4))
        gt = nib.Nifti1Image(np.zeros((3, 4, 5, 2), np.uint8), np.eye(4))
        with self.assertRaises(ValueError):
            geometry(ct, ct, gt, gt, [0, 1])

    def test_wrong_affine(self):
        ct = nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.diag([.7, .8, 2, 1]))
        anatomy = nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.eye(4))
        gt = nib.Nifti1Image(np.zeros((1, 3, 4, 5), np.uint8), np.eye(4))
        with self.assertRaises(ValueError):
            geometry(ct, anatomy, gt, gt, [0])

    def test_bootstrap_constant(self):
        np.testing.assert_allclose(bootstrap([.1, .1, .1], ['a', 'a', 'b']), [.1, .1])

    def test_bootstrap_cluster_weighting(self):
        lo, hi = bootstrap([0, 0, 1], ['a', 'a', 'b'])
        self.assertEqual((lo, hi), (0, 1))
        self.assertEqual(bootstrap([0, 1], ['a', 'a']), [.5, .5])

    def test_aggregate_accounting(self):
        rows = []
        for i, (g, p, allowed) in enumerate([
                ([1, 0], [1, 1], [1, 0]), ([1, 0], [1, 0], [0, 0])]):
            g, p, allowed = [np.array(x, bool) for x in [g, p, allowed]]
            b = metrics(g, p, np.ones(2, bool)); m = metrics(g, p, allowed)
            m.update(selected=True, delta=m['dice'] - b['dice'])
            rows.append(dict(id=i, category='2d', case=str(i), baseline=b, candidates={'k': m}, lung_flags={}))
        s = summarize(rows, 'k')
        self.assertEqual((s['improved'], s['worsened'], s['hit_losses']), (1, 1, 1))
        self.assertEqual((s['tp_removed'], s['fp_removed']), (1, 1))
        self.assertAlmostEqual(s['after'], np.mean([r['candidates']['k']['dice'] for r in rows]))
        selection = select(rows)
        self.assertEqual({r['id'] for r in selection}, {0, 1})
        self.assertTrue(any('lost_hit' in why for r in selection for why in r['reasons']))

    def test_gt_instances_foreground_equivalence(self):
        instances = np.array([0, 1, 2, 7], np.uint8)
        predicted = np.array([0, 1, 1, 1], bool)
        self.assertEqual(metrics(instances > 0, predicted, np.ones(4, bool))['dice'], 1.)


if __name__ == '__main__':
    unittest.main()
