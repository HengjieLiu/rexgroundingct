"""Bounded profiler contracts; no production data required."""
import concurrent.futures
import copy
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import profile_test300 as p

class ProfileTests(unittest.TestCase):
    def test_selection_includes_extremes_and_deduplicates(self):
        rows=[{'case':{'name':str(i)},'workload':i,'input_elements':i,'output_elements':i} for i in range(20)]
        chosen=p.select_cases(rows)
        self.assertEqual([r['workload'] for r in chosen],[19,10,5,18])
        rows[2]['output_elements']=100
        self.assertEqual([r['workload'] for r in p.select_cases(rows)],[19,2,10,5])

    def test_timing_exception_and_resource_limits(self):
        t=p.Timers()
        with self.assertRaises(ValueError), t.stage('failure'):
            raise ValueError('expected')
        self.assertGreater(t.seconds['failure'],0)
        self.assertFalse(p.capacity(65*p.GIB,0,2*p.GIB))
        self.assertTrue(p.capacity(70*p.GIB,0,2*p.GIB))
        with self.assertRaisesRegex(RuntimeError,'reserve'):
            p.space_check(p.tc.MIN_SHARED,100,1,1)
        with self.assertRaisesRegex(RuntimeError,'scratch'):
            p.space_check(p.tc.MIN_SHARED+100,0,1,1)

    def test_cross_filesystem_atomic_storage_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source.npy'; b=Path(d)/'other/dest.npy'
            raw=np.array([-1,0,1],dtype=np.float32)
            p.tc.atomic_save_npy(a,raw)
            self.assertEqual(p.choose_dtype([a]),'float16')
            p.tc.atomic_save_npy(b,raw.astype(np.float16))
            self.assertEqual(b.stat().st_nlink,1)
            np.testing.assert_array_equal(np.load(a)>=0,np.load(b)>=0)
            # Negative underflow becomes negative zero, changing the >=0 mask.
            p.tc.atomic_save_npy(a,np.array([-1e-12,1],dtype=np.float32))
            self.assertEqual(p.choose_dtype([a]),'float32')
            with patch.object(p.os,'replace',side_effect=OSError('interruption')):
                with self.assertRaises(OSError):p.tc.atomic_save_npy(b,raw)
            np.testing.assert_array_equal(np.load(b),[-1,0,1])

    def test_fallback_in_final_partial_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'partial.npy'
            p.tc.atomic_save_npy(path,np.array([-1,0,1,2,3,4,-1e-12],np.float32))
            with patch.object(p,'CHUNK',4):
                self.assertEqual(p.choose_dtype([path]),'float32')

    def test_crop_hash_recovery_and_foreign_job(self):
        with tempfile.TemporaryDirectory() as d:
            job={'sha256':'ours','local_root':d}
            row={'case':{'name':'case.nii.gz'},'metadata_sha256':'meta','crop_elements':4}
            rp=p.record_path(job,5,row['case']['name']);ap=rp.with_suffix('.npy')
            raw=np.ones((1,1,2,2),np.float32);p.tc.atomic_save_npy(ap,raw)
            rec={'path':str(ap),'array_sha256':p.tc.array_sha256(raw),'rank':5,'job_sha256':'ours','metadata_sha256':'meta'}
            p.write(rp,rec);p.verify_crop(job,5,row)
            p.tc.atomic_save_npy(ap,raw*2)
            with self.assertRaisesRegex(RuntimeError,'source hash'):p.verify_crop(job,5,row)
            rec['job_sha256']='foreign';p.write(rp,rec)
            with self.assertRaisesRegex(RuntimeError,'foreign'):p.verify_crop(job,5,row)

    def test_lock_source_drift_and_spawn_file_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'lock'
            with p.tc.exclusive(lock):
                with self.assertRaises(Exception):
                    with p.tc.exclusive(lock):pass
            source=Path(d)/'source';source.write_text('original')
            dataset=Path(d)/'dataset';dataset.write_text('{}')
            job={'sources':{str(source):p.fc.sha256_file(source)},'dataset':{'path':str(dataset),'sha256':p.fc.sha256_file(dataset)}}
            job['sha256']=p.fc.json_sha256(job);p.source_check(job)
            source.write_text('modified')
            with self.assertRaisesRegex(RuntimeError,'source drift'):p.source_check(job)
            ap=Path(d)/'array.npy';p.tc.atomic_save_npy(ap,np.array([-1,0,2],np.float32))
            with concurrent.futures.ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                self.assertEqual(list(pool.map(p.choose_dtype,[[ap],[ap]])),['float16','float16'])

    def test_native_geometry_parity(self):
        try:
            p.tc.inference_imports()
            import nibabel as nib
            import run_voxtell_val_inference as inf
            import common
        except ImportError:
            self.skipTest('requires pinned inference image')
        with tempfile.TemporaryDirectory() as d:
            ctpath=Path(d)/'ct.nii.gz';affine=np.diag([-1.,-2.,3.,1.])
            nib.save(nib.Nifti1Image(np.zeros((3,4,5),np.int16),affine),ctpath)
            meta={'native_cropped_shape_zyx':[5,4,3],'resampled_shape_zyx':[3,2,2],
                  'original_reoriented_shape_zyx':[5,4,3],'crop_bbox_zyx':[[0,5],[0,4],[0,3]],
                  'image_sha256':'image','ct_properties':{'nibabel_stuff':{'original_affine':affine.tolist(),
                  'reoriented_affine':[[1,0,0,-2],[0,2,0,-6],[0,0,3,0],[0,0,0,1]]}}}
            case={'name':'ct.nii.gz','shape':[3,4,5],'findings':{str(i):str(i) for i in reversed(range(11))}}
            crop=np.arange(11*3*2*2,dtype=np.float32).reshape(11,3,2,2)/10-4
            ct=nib.load(ctpath)
            expected,orientation=inf.export_prediction_to_ct_layout(inf.restore_cached_native_crop(crop,meta,-30),ct,meta['ct_properties'],case['name'],output_dtype=None)
            with patch.object(common,'ct_rate_abs_path',return_value=ctpath):
                arr,details=p.geometry_from_crop(crop,meta,case,p.Timers())
            np.testing.assert_array_equal(arr,expected)
            self.assertEqual(details['orientation'],orientation)
            self.assertEqual(details['finding_indices'],list(range(11)))

if __name__=='__main__':unittest.main()
