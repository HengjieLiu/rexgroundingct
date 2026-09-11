import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stream_test300_preflight as p
import stream_test300_transition as t


class PreflightTests(unittest.TestCase):
    def test_eviction_rejects_retained_sources(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'LOCAL',Path(tmp)/'private'):
            with self.assertRaises(p.fc.FreshCacheError):p.evict_private([Path(tmp)/'retained.npy'])

    def test_small_full_reference_comparison_and_checkpoint_fallback(self):
        import nibabel as nib
        p.tc.inference_imports()
        import common
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile=root/'profile.json';parent=root/'parent.json'
            sha='62eada06c82d485229fa18dd14be455aff445eb89baac63712ed660e51c615ab'
            job={'sha256':sha,'local_root':str(root/'retained'),'runtime_root':str(root/'retained_shared'),'candidates':[]}
            for rank in range(5,9):
                c={'rank':rank,'cache_key':'old','selected':[]};job['candidates'].append(c)
                for i in range(4):
                    case={'name':f'train_{i+10}_a_1.nii.gz','shape':[2,2,2],'findings':{str(f):'p'+str(f) for f in reversed(range(12))}}
                    c['selected'].append({'case':case,'output_elements':96})
            sha=p.fc.json_sha256({k:v for k,v in job.items() if k!='sha256'});job['sha256']=sha
            p.fc.atomic_write_json(profile,job)
            p.fc.atomic_write_json(parent,{'runtime_root':str(root/'parent'),'container':{'image_id':'image'}})
            p.fc.atomic_write_json(root/'parent/finish_wave2_state.json',{'status':'stopped_after_wave2','coordinator_exited':True})
            ctroot=root/'ct'
            for c in job['candidates']:
                reference=p.prof.candidate_for(job,c['rank'],'reference')
                for i,row in enumerate(c['selected']):
                    case=row['case'];ct=common.ct_rate_abs_path(case['name'],ctroot);ct.parent.mkdir(parents=True,exist_ok=True)
                    nib.save(nib.Nifti1Image(np.zeros(case['shape'],np.uint8),np.diag([1,2,3,1])),str(ct))
                    raw=np.linspace(-3,3,96,dtype=np.float32).reshape(12,2,2,2)
                    if c['rank']==8 and i==3:raw.flat[0]=-1e-12
                    ap,rp=p.tc.stage_paths(reference,case['name']);p.tc.atomic_save_npy(ap,raw)
                    rec={'name':case['name'],'shape':list(raw.shape),'dtype':'float32','job_spec_sha256':sha,
                         'cache_key':reference['cache_key'],'array_sha256':p.tc.array_sha256(raw),
                         'ct_path':str(ct),'affine':np.diag([1,2,3,1]).tolist(),'orientation':'fixture',
                         'metric_status':p.tc.METRIC_STATUS,**p.tc.prompt_contract(case)}
                    p.fc.atomic_write_json(rp,rec)
            counter=[0]
            def io():counter[0]+=10**9;return {'read_bytes':counter[0]}
            with patch.object(p,'PROFILE',profile),patch.object(p,'PROFILE_SHA',sha),patch.object(p.run,'PARENT',parent),patch.object(p,'ROOT',root/'run'),\
                 patch.object(p,'LOCAL',root/'local'),patch.object(common,'CT_ROOT',ctroot),\
                 patch.object(p.run.shutil,'disk_usage',return_value=type('Space',(),{'free':100*1024**4})()),\
                 patch.object(p.prof,'cpu_init'),patch.object(p.prof,'mem_available',return_value=400*1024**3),\
                 patch.object(p,'io',side_effect=io),patch('builtins.print'):
                p.benchmark()
            report=p.fc.read_json(root/'run/report.json')
            self.assertEqual(report['status'],'passed');self.assertEqual(report['cases'],16)
            self.assertEqual([r['dtype'] for r in report['results']['stream']['models']],['float16']*3+['float32'])
            for r in report['results']['stream']['models']:
                self.assertTrue(all(x['finding_indices']==list(range(12)) for x in r['outputs']))

    def test_transition_gate_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent={'job_spec_sha256':'job','runtime_root':tmp,'candidates':[{'id':str(i)} for i in range(16)]}
            p.fc.atomic_write_json(root/'finish_wave2_state.json',{'status':'stopped_after_wave2','coordinator_exited':True,'job_spec_sha256':'job'})
            with patch.object(t.tr,'progress',return_value={}),patch.object(t.tr,'completed',return_value=True):
                self.assertTrue(t.gate(parent))
                with patch.object(t.tr,'completed',return_value=False):
                    with self.assertRaises(p.fc.FreshCacheError):t.gate(parent)
                p.fc.atomic_write_json(root/'finish_wave2_error.json',{'error':'worker failed'})
                with self.assertRaises(p.fc.FreshCacheError):t.gate(parent)


if __name__=='__main__':unittest.main()
