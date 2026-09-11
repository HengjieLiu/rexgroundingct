#!/usr/bin/env python3
"""Private retained-logit parity and cold sequential publication benchmark."""
from __future__ import annotations
import os
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import copy
import json
from pathlib import Path
import time
import numpy as np
import stream_test300 as run

fc,tc,prof,sio=run.fc,run.tc,run.prof,run.sio
PROFILE=run.HERE/'profiling_jobs/p001_test300_pipeline16_gpu8/job.json'
PROFILE_SHA='62eada06c82d485229fa18dd14be455aff445eb89baac63712ed660e51c615ab'
ROOT=fc.RUNTIME_ROOT/'cache/jobs'/run.JOB_ID/'preflight'
LOCAL=tc.STAGING_ROOT/run.JOB_ID/'preflight'


def evict_private(paths):
    """Drop only our fsynced private copies, never retained/production inputs."""
    for path in paths:
        tc.require(Path(path).resolve().is_relative_to(LOCAL.resolve()),'non-private eviction forbidden')
        with Path(path).open('rb') as f:
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)


def io():
    return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def status(phase,**kw):
    record={'phase':phase,'at_utc':fc.utc_now(),**kw}
    fc.atomic_write_json(ROOT/'state.json',record)
    with (ROOT/'monitor.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def benchmark():
    parent=fc.read_json(run.PARENT)
    hold=fc.read_json(Path(parent['runtime_root'])/'finish_wave2_state.json')
    tc.require(hold['status']=='stopped_after_wave2' and hold['coordinator_exited'],'Wave 2 must finish before disk tests')
    p=fc.read_json(PROFILE)
    tc.require(p['sha256']==PROFILE_SHA and fc.json_sha256({k:v for k,v in p.items() if k!='sha256'})==PROFILE_SHA,'wrong retained benchmark')
    ROOT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    with tc.exclusive(ROOT/'launch.lock'):
        tc.require(not (ROOT/'report.json').exists(),'preflight already completed; preserve evidence')
        prof.cpu_init()
        total=sum(row['output_elements']*4+128 for c in p['candidates'] for row in c['selected'])
        tc.require(run.shutil.disk_usage(LOCAL).free>total+64*run.GIB,'preflight local capacity')
        tc.require(run.shutil.disk_usage(ROOT).free>tc.MIN_SHARED+2*total+32*run.GIB,'preflight shared reserve')
        fixtures=[];inputs=[]
        for original in p['candidates']:
            rank=original['rank'];reference=prof.candidate_for(p,rank,'reference')
            candidate=copy.deepcopy(reference)
            candidate.update(cache_key='stream_preflight_'+p['sha256']+'_r'+str(rank),staging_root=str(LOCAL/f'r{rank}'))
            cases=[r['case'] for r in original['selected']]
            job={'job_spec_sha256':p['sha256'],'runtime_root':str(ROOT),'staging_root':str(LOCAL),
                 'dataset':{'cases':len(cases),'findings':sum(len(c['findings']) for c in cases)}}
            records=[]
            for case in cases:
                status('preparing_private_stage',rank=rank,case=case['name'])
                source,rp=tc.stage_paths(reference,case['name']);old=fc.read_json(rp)
                tc.require(old['job_spec_sha256']==p['sha256'] and old['cache_key']==reference['cache_key'] and old['name']==case['name'],'foreign retained stage')
                # Materialize one array only; the storage proof is computed in memory.
                tc.require(prof.mem_available()>64*run.GIB+source.stat().st_size*3,'preflight RAM headroom')
                raw=np.load(source,allow_pickle=False)
                tc.require(tc.array_sha256(raw)==old['array_sha256'],'retained float32 reference drift')
                timer=prof.Timers();rec=sio.save_stage(job,candidate,case,raw,old,timer)
                for k in ('array_sha256','shape','affine','orientation','finding_indices','prompt_sha256s'):
                    tc.require(rec[k]==old[k],'staging parity failed: '+k)
                del raw
                records.append(rec);inputs.append({'rank':rank,'case':case['name'],'source':str(source),
                    'record_sha256':fc.sha256_file(rp),'array_sha256':old['array_sha256'],'staging_timings':timer.seconds})
            fixtures.append((job,candidate,cases,records))
        fc.atomic_write_json(ROOT/'inputs.json',inputs)
        results={};reference_outputs={}
        for mode in ('reference','stream'):
            mode_start=time.monotonic();rows=[]
            for job,base,cases,records in fixtures:
                c=copy.deepcopy(base);c['cache_root']=str(ROOT/mode/f"r{c['rank']}")
                Path(c['cache_root']).mkdir(parents=True,exist_ok=True)
                paths=[tc.stage_paths(c,case['name'])[0] for case in cases]
                tc.require(not list((Path(c['cache_root'])/'case_manifests').glob('*.json')),'non-fresh benchmark outputs')
                evict_private(paths);before=io();started=time.monotonic()
                status(mode+'_cold_publication',rank=c['rank'])
                if mode=='reference':
                    for case in cases:tc.load_stage(c,case,job['job_spec_sha256'])
                    dtype,outputs=tc.publish(job,c,cases)
                    validation,outputs=tc.validate_published(job,c,cases)
                else:
                    dtype=sio.choose_dtype(records)
                    outputs=[sio.publish_one(job,c,case,rec,dtype,
                        progress=lambda **kw:status(rank=c['rank'],**kw)) for case,rec in zip(cases,records)]
                    validation=sio.completed_manifest(job,c,cases,outputs)
                seconds=time.monotonic()-started;after=io();physical=after['read_bytes']-before['read_bytes']
                source_bytes=sum(path.stat().st_size for path in paths)
                tc.require(physical>=source_bytes*.8,'cold local reads not demonstrated by physical counters')
                for case,output in zip(cases,outputs):
                    key=(c['rank'],case['name'])
                    if mode=='reference':reference_outputs[key]=output
                    else:
                        for k in ('array_sha256','shape','affine','orientation','finding_indices','prompt_sha256s','dtype'):
                            tc.require(output[k]==reference_outputs[key][k],'publication reference mismatch: '+k)
                        tc.require(output['source_read_passes']==1 and output['publication_source_bytes']==tc.stage_paths(c,case['name'])[0].stat().st_size,'not one local source pass')
                row={'rank':c['rank'],'dtype':dtype,'cases':len(cases),'seconds':seconds,'source_bytes':source_bytes,
                    'physical_local_read_bytes':physical,'cold_read_verified':True,'validation':validation,'outputs':outputs}
                rows.append(row);fc.atomic_write_json(ROOT/mode/f"r{c['rank']}_result.json",row)
            results[mode]={'seconds':time.monotonic()-mode_start,'models':rows,'worker_count':1}
        report={'status':'passed','at_utc':fc.utc_now(),'inputs_sha256':fc.sha256_file(ROOT/'inputs.json'),
            'source_hashes':{name:fc.sha256_file(run.HERE/name) for name in ('stream_test300.py','stream_test300_io.py','stream_test300_preflight.py','stream_test300_transition.py','stream_test300_spec.md')},
            'image_id':parent['container']['image_id'],'cases':16,'results':results,
            'stream_source_bytes_per_second':total/results['stream']['seconds'],
            'limitations':['16 retained model-case arrays; one publisher in both comparisons.',
              'Only initial private source reads are cold; later reference passes can benefit from RAM.',
              'Destination readback may hit NFS/client cache. No global cache flush or production eviction.',
              'This does not prove full-wave overlap speedup or checkpoint-wide dtype for other cases.']}
        fc.atomic_write_json(ROOT/'report.json',report)
        lines=['# Streaming publication preflight','', 'All 16 retained model-case outputs match the reference exactly.','',
               '| Method (one worker) | Seconds | Physical local GiB read |','| --- | ---: | ---: |']
        for mode,r in results.items():lines.append(f"| {mode} | {r['seconds']:.1f} | {sum(v['physical_local_read_bytes'] for v in r['models'])/run.GIB:.2f} |")
        lines+=['','One source pass per streamed file, with a separate destination verification pass.','',*report['limitations']]
        fc.atomic_write_text(ROOT/'report.md','\n'.join(lines)+'\n');status('passed',report=str(ROOT/'report.json'))


if __name__=='__main__':
    try:benchmark()
    except BaseException as exc:
        ROOT.mkdir(parents=True,exist_ok=True);status('failed',error=str(exc));raise
