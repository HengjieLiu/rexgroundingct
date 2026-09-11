import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sweep as s


class ScoringTests(unittest.TestCase):
    def test_grid_and_inclusive_boundaries(self):
        self.assertEqual(s.PERCENTAGES, tuple(range(5, 100, 5)))
        bounds = s.cutoffs('logit')
        values = np.concatenate([bounds, np.nextafter(bounds, -np.inf), np.nextafter(bounds, np.inf), [-30, 30]]).astype(np.float32)
        gt = np.arange(len(values)) % 2 == 0
        p, i = s.count_chunk(values, gt, bounds)
        for j, threshold in enumerate(bounds):
            self.assertEqual(p[j], (values >= threshold).sum())
            self.assertEqual(i[j], ((values >= threshold) & gt).sum())
        self.assertEqual(bounds[9], 0)

    def test_streamed_counts_against_direct(self):
        rng = np.random.default_rng(43)
        for dtype in [np.float16, np.float32]:
            values = rng.normal(size=1051).astype(dtype)
            gt = rng.random(1051) > .8
            p, i = np.zeros(19, np.int64), np.zeros(19, np.int64)
            bounds = s.cutoffs('logit')
            for start in range(0, len(values), 37):
                dp, di = s.count_chunk(values[start:start+37], gt[start:start+37], bounds)
                p += dp
                i += di
            for j, b in enumerate(bounds):
                pred = values.astype(np.float32) >= b
                self.assertEqual(p[j], int(pred.sum()))
                self.assertEqual(i[j], int((pred & gt).sum()))
                self.assertAlmostEqual(s.dice(int(gt.sum()), int(p[j]), int(i[j])),
                                       (2*(pred & gt).sum()+1e-6)/(gt.sum()+pred.sum()+1e-6))

    def test_empty_masks_and_hit_boundary(self):
        self.assertEqual(s.dice(0, 0, 0), 1)
        self.assertLess(s.dice(10, 0, 0), .1)
        self.assertGreaterEqual(s.dice(10, 10, 1), .1)
        self.assertLess(s.dice(10, 11, 1), .1)
        p, i = s.count_chunk([], [], s.cutoffs('logit'))
        self.assertFalse(p.any() or i.any())

    def test_nonfinite_and_mismatched_input(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                s.count_chunk([value], [True], s.cutoffs('logit'))
        with self.assertRaises(ValueError):
            s.count_chunk([0, 1], [True], s.cutoffs('logit'))

    def test_ensemble_probability_sum(self):
        logits = np.array([[-2, 0, 1, 3], [3, 0, -2, 1]], np.float32)
        total = s.sigmoid(logits[0]) + s.sigmoid(logits[1])
        gt = np.array([True, False, True, False])
        p, i = s.count_chunk(total, gt, s.cutoffs('probability_sum', 2))
        for j, t in enumerate(s.THRESHOLDS):
            mask = total >= np.float32(t*2)
            self.assertEqual(p[j], mask.sum())
            self.assertEqual(i[j], (mask & gt).sum())

    def test_tie_break(self):
        rows = [dict(threshold_pct=p, dice=.5) for p in [45, 55, 10, 90]]
        self.assertEqual(s.best_row(rows, 'dice')['threshold_pct'], 45)
        rows.append(dict(threshold_pct=50, dice=.5))
        self.assertEqual(s.best_row(rows, 'dice')['threshold_pct'], 50)

    def test_category_aggregation_weights_findings(self):
        rows = [dict(category=c, threshold_pct=p, dice=d, hit=int(d >= .1))
                for c, d in [('2d', .2), ('2d', .8), ('1a', .05)] for p in s.PERCENTAGES]
        metrics, comparisons = s.aggregate(rows)
        overall = metrics[0]
        self.assertAlmostEqual(overall['dice'], 1.05/3)
        self.assertAlmostEqual(overall['hit_rate'], 2/3)
        empty = next(r for r in metrics if r['category'] == '2f')
        self.assertEqual(empty['findings'], 0)
        self.assertIsNone(empty['dice'])
        self.assertIsNone(next(r for r in comparisons if r['category'] == '2f')['best_dice_threshold'])


class IntegrityTests(unittest.TestCase):
    def test_logical_array_hash_excludes_file_header(self):
        a = np.arange(48, dtype=np.float16).reshape(2, 3, 4, 2)
        expected = hashlib.sha256(a.tobytes(order='C')).hexdigest()
        self.assertEqual(s.array_hash(a), expected)
        self.assertEqual(s.array_hash(np.asfortranarray(a)), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'array.npy'
            np.save(path, a)
            self.assertNotEqual(s.file_hash(path), expected)
            self.assertEqual(s.array_hash(np.load(path, mmap_mode='r')), expected)

    def setUp(self):
        self.case = dict(name='example.nii.gz', findings={'0': 'prompt'}, categories={'0': '2d'})
        rows = [dict(case='example.nii.gz', finding_index=0, category='2d', threshold_pct=p,
                     threshold=p/100, gt_voxels=10, pred_voxels=10, intersection=5,
                     dice=s.dice(10, 10, 5), hit=1) for p in s.PERCENTAGES]
        self.record = dict(identity='frozen', case=self.case['name'], rows=rows, rows_sha256=s.digest(rows))

    def test_valid_case_and_resume(self):
        s.validate_cases([self.case], 1, 1)
        self.assertEqual(len(s.validate_record(self.record, 'frozen', self.case)), 19)

    def test_dataset_keys_duplicates_categories(self):
        with self.assertRaises(ValueError):
            s.validate_cases([self.case, self.case], 2, 2)
        for key, value in [('categories', {'0': 'unknown'}), ('findings', {'1': 'prompt'})]:
            case = {**self.case, key: value}
            with self.assertRaises(ValueError):
                s.validate_cases([case], 1, 1)

    def test_resume_rejects_source_drift_or_tampering(self):
        with self.assertRaises(ValueError):
            s.validate_record(self.record, 'changed', self.case)
        record = copy.deepcopy(self.record)
        record['rows'][0]['dice'] = .9
        with self.assertRaises(ValueError):
            s.validate_record(record, 'frozen', self.case)
        record['rows_sha256'] = s.digest(record['rows'])
        with self.assertRaises(ValueError):
            s.validate_record(record, 'frozen', self.case)

    def test_resume_rejects_missing_threshold(self):
        record = copy.deepcopy(self.record)
        record['rows'].pop()
        record['rows_sha256'] = s.digest(record['rows'])
        with self.assertRaises(ValueError):
            s.validate_record(record, 'frozen', self.case)

    def test_a1_pinned_and_b1_missing_cache(self):
        registry = {'families': {'a': {'models': [dict(candidate_id='a', checkpoint={'sha256': s.A1_SHA})], 'recipe': {'kind': 'single_model'}}}}
        sources, _ = s.choose_sources('a1', registry, {})
        self.assertEqual(sources[0]['cache_key'], s.A1_KEY)
        registry['families']['a']['models'][0]['checkpoint']['sha256'] = 'changed'
        with self.assertRaises(ValueError):
            s.choose_sources('a1', registry, {})
        registry['families']['b'] = registry['families']['a']
        catalog = {'candidates': [dict(checkpoint={'sha256': 'changed'}, inference_artifacts={'val200': {'active_cache_version': None, 'versions': []}})]}
        with self.assertRaisesRegex(ValueError, 'Missing strict val200 cache'):
            s.choose_sources('b1', registry, catalog)

    def test_ensemble_preserves_frozen_order(self):
        for recipe, n in [('d1', 4), ('e1', 8)]:
            models = [dict(candidate_id=str(i), checkpoint={'sha256': str(i)}, paired_val200={'cache_key': 'key'+str(i)}) for i in range(n)]
            registry = {'families': {recipe[0]: dict(models=models, recipe={'weights': [1/n]*n})}}
            sources, _ = s.choose_sources(recipe, registry, {})
            self.assertEqual([v['candidate_id'] for v in sources], list(map(str, range(n))))


if __name__ == '__main__':
    unittest.main()
