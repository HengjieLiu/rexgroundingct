#!/usr/bin/env python3
"""CPU-only causal follow-up: disable NumPy huge-page advice per process."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
import profile_test300 as p

HERE=Path(__file__).resolve().parent

def init():
    p.cpu_init()
    from numpy._core import multiarray
    p.check(not multiarray._get_madvise_hugepage(), 'NumPy huge-page advice must be disabled before import')

def worker(job):
    p.source_check(job)
    init()
    root=Path(job['runtime_root'])
    rows=[(c['rank'],r) for c in job['candidates'] for r in c['selected']]
    for workers in job['cpu_workers']:
        mode=f'nohuge_cpu{workers}'
        with p.tc.exclusive(root/mode/'lock'):
            p.check(not (root/mode/'complete.json').exists(),'follow-up already completed')
            warm=p.Timers()
            with warm.stage('warm_inputs'):
                for rank,row in rows:p.verify_crop(job,rank,row)
            start=time.perf_counter()
            with futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
                tasks=[((job,rank,row,mode),row['memory_bytes']) for rank,row in rows]
                restored,limits=p.bounded_map(pool,p.cpu_restore,tasks,workers,root,mode)
                restore_wall=time.perf_counter()-start
                scan=p.Timers(); dtypes={}
                with scan.stage('dtype_scan_wall'):
                    for c0 in job['candidates']:
                        c=p.candidate_for(job,c0['rank'],mode)
                        dtypes[c['rank']]=p.choose_dtype([p.tc.stage_paths(c,r['case']['name'])[0] for r in c['selected']])
                        p.check(dtypes[c['rank']]==p.read(root/'reference'/f'r{c["rank"]}_dtype.json')['dtype'],'dtype mismatch')
                pubstart=time.perf_counter()
                tasks=[((job,p.candidate_for(job,rank,mode),row['case'],dtypes[rank],mode),row['memory_bytes']) for rank,row in rows]
                published,publimits=p.bounded_map(pool,p.publish_one,tasks,workers,root,mode)
                pubwall=time.perf_counter()-pubstart
            result={'status':'passed','workers':workers,'cases':16,'numpy_madvise_hugepage':False,
                    'warm':warm.seconds,'wall_seconds':time.perf_counter()-start,
                    'restore_wall_seconds':restore_wall,'publication_wall_seconds':pubwall,**scan.seconds,
                    'limits':limits,'publication_limits':publimits,
                    'case_seconds_sum':sum(r['elapsed_seconds'] for r in restored),
                    'peak_worker_rss_bytes':max(r['rss_peak_bytes'] for r in restored+published),
                    'physical_read_bytes':sum(r['io']['read_bytes'] for r in restored+published),
                    'published_bytes':sum(r['bytes'] for r in published),'dtypes':dtypes,'completed_at_utc':p.fc.utc_now()}
            p.write(root/mode/'complete.json',result);print(json.dumps(result),flush=True)

def run(job):
    root=Path(job['runtime_root']); statepath=root/'nohuge_followup_state.json'
    with p.tc.exclusive(root/'nohuge_followup.lock'):
        policy={'parent_job_sha256':job['sha256'],'numpy_madvise_hugepage':False,
                'extra_inference':0,'cpu_workers':[4,8,16],
                'source_sha256':p.fc.sha256_file(Path(__file__)),
                'extra_shared_bytes':job['space']['native_fp32_bytes']*3+8*p.GIB,
                'extra_local_bytes':job['space']['native_fp32_bytes']*3+8*p.GIB}
        p.write(root/'nohuge_policy.json',policy)
        p.write(statepath,{'status':'waiting_for_baseline','pid':os.getpid(),'at_utc':p.fc.utc_now()})
        while True:
            state=p.read(root/'state.json')
            p.check(state['status']!='failed','baseline failed; follow-up will not launch')
            if state['status']=='complete':break
            time.sleep(10)
        p.space_check(shutil.disk_usage(root).free,shutil.disk_usage(job['local_root']).free,
                      policy['extra_shared_bytes'],policy['extra_local_bytes'])
        p.check(p.fc.sha256_file(Path(__file__))==policy['source_sha256'],'follow-up source drift')
        name='sideexp003_'+p.JOB_ID+'_nohuge'
        cmd=p.docker_args(job,name,'worker')
        at=cmd.index(job['container']['image_id']);cmd[at:at]=['-e','NUMPY_MADVISE_HUGEPAGE=0']
        at=cmd.index('/workspace/side_experiments/sideexp003_ensemble_method_hub/profile_test300.py')
        cmd[at]='/workspace/side_experiments/sideexp003_ensemble_method_hub/profile_test300_nohuge.py'
        p.write(root/'commands'/f'{name}.json',{'argv':cmd})
        p.run_cmd(cmd)
        p.write(statepath,{'status':'running','container':name,'at_utc':p.fc.utc_now()})
        while True:
            p.sample(root,[name])
            with (root/'nohuge_memory_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps({'epoch':time.time(),'vmstat':Path('/proc/vmstat').read_text()})+'\n')
            state=json.loads(p.run_cmd(['docker','inspect',name]))[0]['State']
            if not state['Running']:
                p.check(state['ExitCode']==0,'nohuge replay failed');break
            time.sleep(5)
        comparisons=[p.read(root/f'{prefix}cpu{w}'/'complete.json') for prefix in ('','nohuge_') for w in [4,8,16]]
        p.write(root/'nohuge_comparison.json',{'comparisons':comparisons,'production_resume':False})
        p.write(statepath,{'status':'complete','at_utc':p.fc.utc_now()})

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['run','worker']);ap.add_argument('--job',type=Path,default=p.DEFAULT_JOB);a=ap.parse_args();j=p.read(a.job)
    try:
        (run if a.action=='run' else worker)(j)
    except BaseException as exc:
        if a.action=='run':p.write(Path(j['runtime_root'])/'nohuge_followup_state.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':main()
