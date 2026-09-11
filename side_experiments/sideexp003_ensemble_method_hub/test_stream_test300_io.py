import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stream_test300_io as s

class StreamIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.job={'job_spec_sha256':'a'*64,'runtime_root':str(self.root),'dataset':{'cases':1,'findings':1}}
        self.c={'cache_key':'key','staging_root':str(self.root/'stage'),'cache_root':str(self.root/'cache')}
        self.case={'name':'case.nii.gz','shape':[1,1,7],'findings':{'0':'prompt'}}
        self.patch=patch.object(s.shutil,'disk_usage',return_value=type('Space',(),{'free':100*1024**4})())
        self.patch.start();self.addCleanup(self.patch.stop)
        self.chunk=patch.object(s,'CHUNK',3);self.chunk.start();self.addCleanup(self.chunk.stop)

    def fixture(self,values=None):
        raw=np.array(values if values is not None else [-30,-1,-.1,0,.1,1,30],np.float32).reshape(1,1,1,7)
        ap,rp=s.tc.stage_paths(self.c,self.case['name']);s.tc.atomic_save_npy(ap,raw)
        proof=s.storage_proof(raw)
        record={'job_spec_sha256':self.job['job_spec_sha256'],'cache_key':'key','name':self.case['name'],
                'shape':list(raw.shape),'dtype':'float32','array_sha256':proof['float32_sha256'],
                'storage_proof':proof,'storage_proof_sha256':s.fc.json_sha256(proof),
                **s.tc.prompt_contract(self.case),'affine':np.eye(4).tolist()}
        s.fc.atomic_write_json(rp,record)
        return raw,record

    def publish(self,record,dtype):
        return s.publish_one(self.job,self.c,self.case,record,dtype,geometry=lambda *a:None)

    def test_exact_values_and_single_source_pass(self):
        raw,record=self.fixture();source=s.tc.stage_paths(self.c,self.case['name'])[0]
        original=Path.open;opened=[];read_bytes=[]
        class Reader:
            def __init__(self,f):self.f=f
            def __getattr__(self,key):return getattr(self.f,key)
            def __enter__(self):self.f.__enter__();return self
            def __exit__(self,*args):return self.f.__exit__(*args)
            def read(self,*args):
                b=self.f.read(*args);read_bytes.append(len(b));return b
        def opening(path,*args,**kw):
            f=original(path,*args,**kw)
            if path==source:
                opened.append(str(path));return Reader(f)
            return f
        with patch.object(Path,'open',opening):out=self.publish(record,'float16')
        self.assertEqual(len(opened),1);self.assertEqual(sum(read_bytes),source.stat().st_size)
        self.assertEqual(out['publication_source_bytes'],source.stat().st_size)
        np.testing.assert_array_equal(np.load(out['array_path']),raw.astype(np.float16))
        self.assertEqual(out['array_sha256'],hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest())
        self.assertEqual(s.completed_manifest(self.job,self.c,[self.case],[out])['status'],'passed')

    def test_checkpoint_fallback_and_zero_ties(self):
        raw,record=self.fixture([-30,-1,-1e-12,-0.,0.,1,30])
        self.assertEqual(record['storage_proof']['float16_mask_mismatch_voxels'],1)
        self.assertEqual(s.choose_dtype([record]),'float32')
        with self.assertRaisesRegex(s.fc.FreshCacheError,'unsafe float16'):self.publish(record,'float16')
        out=self.publish(record,'float32')
        np.testing.assert_array_equal(np.load(out['array_path']),raw)
        self.assertEqual(out['array_sha256'],record['array_sha256'])

    def test_corrupt_source_never_promotes(self):
        raw,record=self.fixture();raw.flat[1]=2
        s.tc.atomic_save_npy(s.tc.stage_paths(self.c,self.case['name'])[0],raw)
        with self.assertRaisesRegex(s.fc.FreshCacheError,'source hash'):self.publish(record,'float16')
        self.assertFalse((Path(self.c['cache_root'])/'cases'/(self.case['name']+'.npy')).exists())

    def test_restart_reuses_verified_destination_without_source(self):
        raw,record=self.fixture();out=self.publish(record,'float16')
        s.tc.stage_paths(self.c,self.case['name'])[0].unlink()
        resumed=self.publish(record,'float16')
        self.assertEqual(resumed['publication_source_bytes'],0)
        self.assertTrue(resumed['reused_verified_destination'])
        self.assertEqual(resumed['array_sha256'],out['array_sha256'])

    def test_orphan_rename_and_corrupt_destination(self):
        raw,record=self.fixture();out=self.publish(record,'float16')
        manifest=Path(self.c['cache_root'])/'case_manifests'/(self.case['name']+'.json')
        manifest.unlink()
        self.assertTrue(self.publish(record,'float16')['reused_verified_destination'])
        bad=np.load(out['array_path']);bad.flat[0]=0;s.tc.atomic_save_npy(Path(out['array_path']),bad)
        with self.assertRaisesRegex(s.fc.FreshCacheError,'destination hash'):self.publish(record,'float16')

    def test_failed_destination_validation_keeps_source(self):
        raw,record=self.fixture()
        with patch.object(s,'verify_file',side_effect=s.fc.FreshCacheError('bad disk')):
            with self.assertRaisesRegex(s.fc.FreshCacheError,'bad disk'):self.publish(record,'float16')
        self.assertTrue(s.tc.stage_paths(self.c,self.case['name'])[0].exists())
        self.assertFalse((Path(self.c['cache_root'])/'cases'/(self.case['name']+'.npy')).exists())
        self.assertEqual(self.publish(record,'float16')['source_read_passes'],1)

    def test_proof_and_prompt_tampering_rejected(self):
        _,record=self.fixture();record['storage_proof']['elements']+=1
        with self.assertRaisesRegex(s.fc.FreshCacheError,'proof hash'):self.publish(record,'float16')
        _,record=self.fixture();record['finding_indices']=[1]
        with self.assertRaisesRegex(s.fc.FreshCacheError,'prompt'):self.publish(record,'float16')

    def test_invalid_numeric_input_and_read_budget_pause(self):
        with self.assertRaisesRegex(s.fc.FreshCacheError,'nonfinite'):
            s.storage_proof(np.array([np.nan],np.float32))
        b=s.ReadBudget(self.root)
        with patch.object(b,'controls',side_effect=[{'gpu_active':True,'paused':True},{'gpu_active':False}]),patch.object(s.time,'sleep') as sleep:
            b.acquire(100);sleep.assert_called_once_with(.5)
        self.assertEqual(b.rate,0)

    def test_space_wait_and_abort_keep_sources(self):
        _,record=self.fixture()
        with patch.object(s.shutil,'disk_usage',return_value=type('Space',(),{'free':0})()):
            (self.root/'control').mkdir();(self.root/'control/ABORT').touch()
            with self.assertRaisesRegex(s.fc.FreshCacheError,'aborted'):self.publish(record,'float16')
        self.assertTrue(s.tc.stage_paths(self.c,self.case['name'])[0].exists())

    def test_throttling_applies_during_gpu_work_only(self):
        b=s.ReadBudget(self.root);now=[100.]
        def sleep(n):now[0]+=n
        with patch.object(s.time,'monotonic',side_effect=lambda:now[0]),patch.object(s.time,'sleep',side_effect=sleep),\
             patch.object(b,'controls',return_value={'gpu_active':True,'read_mib_s':32}):
            b.acquire(32*1024**2);b.acquire(32*1024**2)
            self.assertAlmostEqual(now[0],101.)
            with patch.object(b,'controls',return_value={'gpu_active':False}):b.acquire(32*1024**2)
            self.assertAlmostEqual(now[0],101.)

if __name__=='__main__':unittest.main()
