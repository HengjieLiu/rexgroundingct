"""Continuation-specific handoff, recovery and admission checks."""
import concurrent.futures
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import continue_test300 as c

class ContinuationTests(unittest.TestCase):
    def test_four_workers_allocation_env_and_bounded_spec(self):
        job={'job_spec_sha256':'j','container':{'image_id':'image'},'runtime_root':'/tmp/r'}
        candidate={'wave':2,'gpu':0,'rank':5,'cache_root':'/tmp/cache','staging_root':'/tmp/stage'}
        _,cmd=c.docker_command(job,candidate)
        for value in ('NUMPY_MADVISE_HUGEPAGE=0','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'):
            self.assertIn(value,cmd)
        self.assertIn('device=0',cmd)
        self.assertIn('/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m',cmd)

    def test_memory_reservation_exception_releases_and_low_memory_waits(self):
        with tempfile.TemporaryDirectory() as d:
            job={'runtime_root':d}
            with patch.object(c.prof,'mem_available',side_effect=[65*c.GIB,100*c.GIB]), patch.object(c.time,'sleep') as sleep:
                with self.assertRaisesRegex(ValueError,'inside'):
                    with c.memory_slot(job,2*c.GIB):
                        self.assertTrue(c.fc.read_json(Path(d)/'memory_reservations.json'))
                        raise ValueError('inside')
                sleep.assert_called_once()
            self.assertEqual(c.fc.read_json(Path(d)/'memory_reservations.json'),{})

    def test_foreign_and_corrupt_crop_rejected_before_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            job={'runtime_root':d,'job_spec_sha256':'ours'}
            candidate={'staging_root':d,'cache_key':'key'}
            case={'name':'case.nii.gz'}
            ap,rp=c.crop_paths(candidate,case)
            c.tc.atomic_save_npy(ap,np.ones((1,2,2,2),np.float32))
            rec={'job_spec_sha256':'foreign','cache_key':'key','name':case['name'],'array_sha256':'bad'}
            c.fc.atomic_write_json(rp,rec)
            with patch.object(c.tc,'inference_imports'),patch.object(c.tc,'load_stage',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'foreign crop'):c.restore_one(job,candidate,case)
                rec['job_spec_sha256']='ours';c.fc.atomic_write_json(rp,rec)
                with self.assertRaisesRegex(RuntimeError,'crop hash'):c.restore_one(job,candidate,case)

    def test_existing_stage_spawn_recovery_no_crop_or_inference(self):
        with tempfile.TemporaryDirectory() as d:
            job={'runtime_root':d,'job_spec_sha256':'ours'}
            candidate={'staging_root':d,'cache_key':'key'}
            case={'name':'case.nii.gz','shape':[2,2,2],'findings':{'0':'a'}}
            ap,rp=c.tc.stage_paths(candidate,case['name'])
            raw=np.ones((1,2,2,2),np.float32)
            rec={'job_spec_sha256':'ours','cache_key':'key','name':case['name'],'shape':list(raw.shape),'dtype':'float32',
                 'array_sha256':c.tc.array_sha256(raw),**c.tc.prompt_contract(case)}
            c.tc.atomic_save_npy(ap,raw);c.fc.atomic_write_json(rp,rec)
            with concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=c.cpu_init) as pool:
                self.assertEqual(pool.submit(c.restore_one,job,candidate,case).result(),rec)
            self.assertFalse(c.crop_paths(candidate,case)[0].exists())

    def test_handoff_restoration_matches_reference_and_restart(self):
        c.tc.inference_imports()
        import nibabel as nib
        import common
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ctpath=root/'ct.nii.gz';affine=np.diag([-1.,-2.,3.,1.])
            nib.save(nib.Nifti1Image(np.zeros((3,4,5),np.int16),affine),ctpath)
            meta={'native_cropped_shape_zyx':[5,4,3],'resampled_shape_zyx':[3,2,2],
                  'original_reoriented_shape_zyx':[5,4,3],'crop_bbox_zyx':[[0,5],[0,4],[0,3]],'image_sha256':'image',
                  'ct_properties':{'nibabel_stuff':{'original_affine':affine.tolist(),
                  'reoriented_affine':[[1,0,0,-2],[0,2,0,-6],[0,0,3,0],[0,0,0,1]]}}}
            case={'name':'ct.nii.gz','shape':[3,4,5],'findings':{str(i):str(i) for i in reversed(range(11))}}
            mp=root/'pre/cases/ct/metadata.json';c.fc.atomic_write_json(mp,meta)
            candidate={'id':'candidate','cache_key':'key','staging_root':str(root/'stage'),'cache':{'root':str(root/'pre')},
                       'test_preprocessing_cases':{case['name']:{'metadata_sha256':c.fc.sha256_file(mp)}}}
            job={'runtime_root':d,'staging_root':d,'job_spec_sha256':'ours'}
            crop=np.arange(11*3*2*2,dtype=np.float32).reshape(11,3,2,2)/10-4
            ap,rp=c.crop_paths(candidate,case);c.tc.atomic_save_npy(ap,crop)
            c.fc.atomic_write_json(rp,{'job_spec_sha256':'ours','cache_key':'key','name':case['name'],
                'array_sha256':c.tc.array_sha256(crop),'gpu_wall_seconds':1,'timings':{}})
            with patch.object(common,'ct_rate_abs_path',return_value=ctpath),patch.object(c.prof,'mem_available',return_value=400*c.GIB):
                expected,details=c.prof.geometry_from_crop(crop,meta,case,c.prof.Timers())
                np.clip(expected,-30,30,out=expected)
                result=c.restore_one(job,candidate,case)
            self.assertEqual(result['array_sha256'],c.tc.array_sha256(expected))
            self.assertEqual(result['finding_indices'],list(range(11)))
            self.assertEqual(result['affine'],affine.tolist())
            self.assertFalse(ap.exists());self.assertFalse(rp.exists())
            self.assertEqual(c.restore_one(job,candidate,case),result)

if __name__=='__main__':unittest.main()
