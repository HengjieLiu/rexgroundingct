import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stream_test300 as s

class StreamRunnerTests(unittest.TestCase):
    def test_throttle_requires_sustained_contention_and_recovery(self):
        t=s.Throttle();t.baseline=10
        t.check(25,True);self.assertFalse(t.paused)
        t.check(25,True);self.assertTrue(t.paused)
        t.check(12,True);self.assertTrue(t.paused)
        t.check(12,True);self.assertFalse(t.paused)
        t.check(30,True);t.check(30,True);t.check(30,False);self.assertFalse(t.paused)

    def test_capacity_accounts_for_backlog_and_existing_bytes(self):
        job={'output_elements_per_model':s.GIB,'dataset':{'cases':300},'candidates':[{'id':str(i)} for i in range(8)]}
        stats={'0':{'started':True,'dtype':'float16','staged_bytes':4*s.GIB+38400,'published_bytes':s.GIB}}
        new=job['candidates'][4:]
        v=s.capacity(job,stats,new,s.tc.MIN_SHARED+60*s.GIB,200*s.GIB)
        self.assertTrue(v['ok']);self.assertAlmostEqual(v['shared_outstanding_bytes']/s.GIB,49,places=3)
        self.assertAlmostEqual(v['local_outstanding_bytes']/s.GIB,144,places=3)
        self.assertFalse(s.capacity(job,stats,new,s.tc.MIN_SHARED+48*s.GIB,200*s.GIB)['ok'])
        self.assertFalse(s.capacity(job,stats,new,s.tc.MIN_SHARED+60*s.GIB,140*s.GIB)['ok'])

    def test_stage_commit_is_metadata_only_and_preserves_global_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job={'runtime_root':d,'job_spec_sha256':'j','dataset':{'cases':2,'findings':2}}
            cand={'id':'c','cache_key':'k','staging_root':str(root/'stage')};cases=[]
            for i,value in enumerate([-1.,-1e-12]):
                case={'name':str(i)+'.nii.gz','shape':[1,1,1],'findings':{'0':'prompt'}};cases.append(case)
                raw=np.array([value],np.float32).reshape(1,1,1,1);proof=s.sio.storage_proof(raw)
                ap,rp=s.tc.stage_paths(cand,case['name']);s.tc.atomic_save_npy(ap,raw)
                record={'name':case['name'],'shape':list(raw.shape),'dtype':'float32','job_spec_sha256':'j','cache_key':'k',
                        'array_sha256':proof['float32_sha256'],'storage_proof':proof,'storage_proof_sha256':s.fc.json_sha256(proof),
                        **s.tc.prompt_contract(case),'npy_bytes':ap.stat().st_size}
                s.fc.atomic_write_json(rp,record)
            with patch.object(s.np,'load',side_effect=AssertionError('array reread forbidden')):
                m=s.commit_staging(job,cand,cases)
            self.assertEqual(m['dtype'],'float32');self.assertEqual(s.staging_manifest(job,cand),m)
            changed=copy.deepcopy(m);changed['dtype']='float16';s.fc.atomic_write_json(s.staging_path(job,cand),changed)
            with self.assertRaises(s.fc.FreshCacheError):s.staging_manifest(job,cand)

    def test_scheduler_launches_next_gpu_wave_before_publication_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent.json';original=root/'original.json'
            s.fc.atomic_write_json(parent,{'runtime_root':str(root/'parent')});s.fc.atomic_write_json(original,{'runtime_root':str(root/'original')})
            s.fc.atomic_write_json(root/'parent/finish_wave2_state.json',{'status':'stopped_after_wave2','coordinator_exited':True})
            job={'job_id':s.JOB_ID,'job_spec_sha256':'j','runtime_root':str(root/'run'),'staging_root':str(root/'stages'),
                 'container':{'image_id':'image'},'parent_completed':[],'output_elements_per_model':1,'dataset':{'cases':300},'candidates':[]}
            for rank in range(9,21):
                job['candidates'].append({'id':str(rank),'rank':rank,'wave':3+(rank-9)//4,'gpu':(rank-9)%4,
                    'cache_key':str(rank),'cache_root':str(root/'cache'/str(rank)),'staging_root':str(root/'stages'/str(rank)),
                    'progress_path':str(root/'run/progress'/(str(rank)+'.json')),'paired_val200':{}})
            registry={};stats={};iterations=[0];overlap=[];all_done=[False]
            def command(args,check=True):
                if args[:2]==['docker','run']:
                    name=args[args.index('--name')+1];registry[name]={'State':{'Running':True,'ExitCode':0},'Config':{'Labels':{'rex.test_job':'j'}}}
                    if '_w04_' in name:overlap.append(not all_done[0])
                return type('R',(),{'stdout':'gpu8\n' if args[0]=='git' else '', 'stderr':''})()
            def docker_state(name):
                state=registry.get(name)
                if state and name.endswith('_publisher'):state['State']['Running']=not all_done[0]
                return state
            def collect(obs):
                iterations[0]+=1
                if iterations[0]>20:raise AssertionError('scheduler did not finish')
                for c in job['candidates']:
                    name,_=s.container_command(job,c);active=name in registry
                    gate=(Path(job['runtime_root'])/'control'/f"wave_{c['wave']:02d}.continue").exists()
                    ready=active and gate
                    phase='staged_ready' if ready else 'smoke_waiting' if active else 'queued'
                    if ready:registry[name]['State']['Running']=False
                    if active:s.fc.atomic_write_json(Path(c['progress_path']),{'free_mib_after_smoke':40000})
                    stats[c['id']]={'rank':c['rank'],'wave':c['wave'],'started':active,'complete':False,'gpu_phase':phase,'phase':phase,
                        'staged_bytes':1 if ready else 0,'published_bytes':0,'dtype':'float16' if ready else None,
                        'normalized_io_cost':1 if ready else None,'gpu_error':None}
                all_done[0]=all(v['gpu_phase']=='staged_ready' for v in stats.values())
                if all_done[0]:
                    for v in stats.values():v['complete']=True
                    for c in job['candidates']:
                        s.fc.atomic_write_json(Path(c['cache_root'])/'reproduction_validation.json',{'dtype':'float16','cases':300,'findings':582,'array_bytes':1})
                        s.fc.atomic_write_json(Path(c['cache_root'])/'export_manifest.json',{'job_spec_sha256':'j'})
                return copy.deepcopy(stats)
            def stage(job,c):return {'dtype':'float16'} if stats.get(c['id'],{}).get('gpu_phase')=='staged_ready' else None
            with patch.object(s,'PARENT',parent),patch.object(s.old,'PARENT',original),patch.object(s,'DEFAULT',root/'job/job_spec.json'),patch.object(s,'validate'),\
                 patch.object(s.socket,'gethostname',return_value='shenggpu8'),patch.object(s.tr,'command',side_effect=command),\
                 patch.object(s.tr,'docker_state',side_effect=docker_state),patch.object(s.tr,'gpu_free',return_value={i:48000 for i in range(4)}),\
                 patch.object(s.Observation,'collect',collect),patch.object(s,'staging_manifest',side_effect=stage),\
                 patch.object(s.prof,'mem_available',return_value=400*s.GIB),\
                 patch.object(s.shutil,'disk_usage',return_value=type('Space',(),{'free':100*1024**4})()),\
                 patch.object(s.time,'sleep'),patch('builtins.print'):
                s.run(job)
            self.assertEqual(overlap,[True]*4)
            self.assertEqual(s.fc.read_json(root/'run/state.json')['status'],'complete')
            self.assertEqual(len(registry),13)

if __name__=='__main__':unittest.main()
