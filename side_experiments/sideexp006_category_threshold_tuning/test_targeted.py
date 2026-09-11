import tempfile
import unittest
from pathlib import Path
import numpy as np
import sweep as s
from targeted_export import storage_array, selected_gt, helpers, check_files

class TargetedTests(unittest.TestCase):
    def test_nonconsecutive_alignment_and_routing_errors(self):
        record = dict(shape=[2, 1, 1, 1], finding_ids=[1, 4], channel_to_finding={'0': 1, '1': 4})
        self.assertEqual(s.channel_index(record, 4), 1)
        source = dict(candidate_id='m', cases={'c': record})
        case = dict(name='c', categories={'1': '1a', '4': '1b'})
        spec = dict(categories={'1a': 'm', '1b': 'm'})
        s.validate_routing([source], spec, [case])
        with self.assertRaises(ValueError): s.validate_routing([source, source], spec, [case])
        with self.assertRaises(ValueError): s.validate_routing([], spec, [case])
        with self.assertRaises(ValueError): s.channel_index(record, 0)
        record['channel_to_finding']['0'] = 4
        with self.assertRaises(ValueError): s.channel_index(record, 4)

    def test_storage_fallback_and_endpoints(self):
        a = np.array([-100, -.01, 0, .1, 100], np.float32)
        stored, mismatch = storage_array(a)
        self.assertEqual(stored.dtype, np.float16)
        self.assertEqual(mismatch, 0)
        np.testing.assert_array_equal(stored >= 0, a >= 0)
        a = np.array([-1e-12, 0, 1e-12], np.float32)
        stored, mismatch = storage_array(a)
        self.assertEqual(stored.dtype, np.float32)
        self.assertEqual(mismatch, 1)
        with self.assertRaises(ValueError): storage_array(np.array([np.nan]))

    def test_selected_gt_native_isotropic_restoration(self):
        import nibabel as nib
        e, _ = helpers()
        data = np.arange(5*8, dtype=np.float32).reshape(5, 2, 2, 2)
        image = selected_gt(nib.Nifti1Image(data, np.eye(4)), [1, 4])
        np.testing.assert_array_equal(image.dataobj, data[[1, 4]])
        for cached in [(2, 2, 2), (4, 4, 4)]:
            meta = dict(native_cropped_shape_zyx=[2,2,2], resampled_shape_zyx=cached,
                original_reoriented_shape_zyx=[4,4,4], crop_bbox_zyx=[[1,3],[1,3],[1,3]])
            restored = e.restore_cached_native_crop(np.full((2,*cached), 2, np.float32), meta, -30)
            self.assertEqual(restored.shape, (2,4,4,4))
            np.testing.assert_array_equal(restored[:,1:3,1:3,1:3], 2)
            np.testing.assert_array_equal(restored[:,0], -30)
        with self.assertRaises(ValueError): selected_gt(nib.Nifti1Image(data, np.eye(4)), [1,1])

    def test_direct_four_model_average(self):
        rng = np.random.default_rng(7)
        arrays = [rng.normal(size=1000).astype(np.float16) for _ in range(4)]
        probabilities = np.zeros(1000, np.float32)
        for a in arrays: probabilities += s.sigmoid(a)
        gt = rng.random(1000) > .5
        p, i = s.count_chunk(probabilities, gt, s.cutoffs('probability_sum',4))
        mean = probabilities / np.float32(4)
        for j, threshold in enumerate(s.THRESHOLDS.astype(np.float32)):
            mask = mean >= threshold
            self.assertEqual(p[j], mask.sum())
            self.assertEqual(i[j], (mask & gt).sum())

    def test_source_change_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'input'; p.write_text('before')
            files = {str(p): s.file_hash(p)}
            check_files(files)
            p.write_text('after')
            with self.assertRaisesRegex(ValueError, 'Source changed'): check_files(files)

    def test_b1_reference_composition(self):
        contract = dict(recipe='b1', recipe_spec={'categories': {'1a': 'm'}},
            cases=[{'name':'c', 'categories':{'4':'1a'}}], sources=[dict(candidate_id='m',
            cases={'c':dict(shape=[2,1,1,1],finding_ids=[1,4],same_pass_reference={'finding_dice':[0, .4]})})])
        reference, historical = s.baseline_references(contract)
        self.assertEqual(reference['dice'], .4)
        self.assertEqual(reference['hits'], 1)
        self.assertEqual(historical['hits'], 298)

    def test_routed_evaluation_and_resume_array_change(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = np.zeros((5,2,2,2), np.float32)
            gt[4,0] = 1
            gtpath = root/'c.nii.gz'
            nib.save(nib.Nifti1Image(gt, np.eye(4)), gtpath)
            logits = np.full((2,2,2,2), -1, np.float16)
            logits[1,0] = 1
            arraypath = root/'logits.npy'
            np.save(arraypath, logits)
            record = dict(shape=list(logits.shape), finding_ids=[1,4],
                channel_to_finding={'0':1, '1':4}, array_path=str(arraypath),
                array_sha256=s.array_hash(logits), same_pass_reference={'finding_dice':[1,1]})
            case = dict(name='c.nii.gz', findings={'4':'prompt'}, categories={'4':'1a'},
                gt_path=str(gtpath), gt_sha256=s.file_hash(gtpath), shape=list(gt.shape))
            manifest = dict(identity='test', contract=dict(recipe='b1', recipe_spec={'categories':{'1a':'m'}},
                sources=[dict(candidate_id='m', cases={'c.nii.gz':record})]))
            result = s.evaluate_case(manifest, case, str(root), 3)
            self.assertFalse(result[1])
            rows = s.read(root/'cases/c.nii.gz.json')['rows']
            self.assertEqual(len(rows), 19)
            self.assertEqual(next(r for r in rows if r['threshold_pct']==50)['dice'], 1)
            self.assertTrue(s.evaluate_case(manifest, case, str(root), 3)[1])
            logits[0,0,0,0] = 3
            np.save(arraypath, logits)
            with self.assertRaisesRegex(ValueError, 'Array hash mismatch'):
                s.evaluate_case(manifest, case, str(root), 3)
