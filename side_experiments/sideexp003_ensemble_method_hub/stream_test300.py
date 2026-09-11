#!/usr/bin/env python3
"""Overlapped test300 continuation with durable proofs and streaming publication."""
from __future__ import annotations
import os
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import concurrent.futures as futures
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import time
import numpy as np
import continue_test300 as old
import stream_test300_io as sio

fc,tc,tr,prof=old.fc,old.tc,old.tr,old.prof
HERE=Path(__file__).resolve().parent
JOB_ID='j005_test300_r09_r20_streaming_gpu8'
DEFAULT=HERE/'cache_jobs'/JOB_ID/'job_spec.json'
PARENT=old.DEFAULT
GIB=1024**3
aborted=old.aborted
memory_slot=old.memory_slot
crop_paths=old.crop_paths
cpu_init=old.cpu_init


def validate(job):
    errors=fc.validate_job(job);tc.require(not errors,str(errors))
    tc.require(job['job_id']==JOB_ID and [c['rank'] for c in job['candidates']]==list(range(9,21)), 'wrong streaming roster')
    tc.require(job['execution']=={'cpu_workers':4,'pending_per_gpu':2,'publisher_workers':1,'publication_read_mib_s_during_gpu':32,'numpy_madvise_hugepage':0,'cpu_threads':1},'execution drift')
    tc.require(Path(job['runtime_root'])==fc.RUNTIME_ROOT/'cache/jobs'/JOB_ID,'unsafe runtime')
    tc.require(Path(job['staging_root'])==tc.STAGING_ROOT/JOB_ID,'unsafe staging root')
    tc.require(job['dataset']['cases']==300 and job['dataset']['findings']==582 and job['dataset']['sha256']==fc.TEST300_SHA256,'dataset contract drift')
    tc.require(fc.sha256_file(tc.resolve(job['dataset']['path']))==job['dataset']['sha256'],'dataset hash drift')
    tc.require(fc.sha256_file(PARENT)==job['parent_file_sha256'],'parent job drift')
    tc.require(fc.sha256_file(Path(job['preflight']['path']))==job['preflight']['sha256'],'preflight evidence drift')
    for entry in job['source_bundle']['files']:
        tc.require(fc.sha256_file(tc.REPO/entry['path'])==entry['sha256'],'source drift: '+entry['path'])
    parent=fc.read_json(PARENT)
    tc.require(fc.sha256_file(Path(parent['runtime_root'])/'finish_wave2_state.json')==job['wave2_hold_sha256'],'parent handoff drift')
    tc.require(fc.json_sha256(job['source_bundle']['files'])==job['source_bundle']['sha256'],'source bundle digest drift')
    for c,p in zip(job['candidates'],parent['candidates'][4:]):
        for key in ('id','rank','wave','gpu','checkpoint','config','plans','model_spec','cache','test_preprocessing_cases','inference_contract','paired_val200'):
            tc.require(c[key]==p[key],'candidate contract drift: '+key)
        tc.require(Path(c['cache_root'])==fc.RUNTIME_ROOT/'cache/logits/by_cache_key'/c['cache_key'],'unsafe publication root')
        tc.require(Path(c['staging_root'])==Path(job['staging_root'])/c['cache_key'],'unsafe candidate stage')
        tc.require(Path(c['progress_path'])==Path(job['runtime_root'])/'progress'/(c['id']+'.json'),'unsafe progress')
    for p in job['parent_completed']:
        cache=Path(p['cache_root'])
        tc.require(fc.sha256_file(cache/'export_manifest.json')==p['export_manifest_sha256'],'parent completed manifest drift')
        tc.require(fc.sha256_file(cache/'reproduction_validation.json')==p['validation_sha256'],'parent validation drift')


def prepare():
    tc.require(not DEFAULT.exists(),'job already frozen')
    parent=fc.read_json(PARENT)
    hold=fc.read_json(Path(parent['runtime_root'])/'finish_wave2_state.json')
    tc.require(hold['status']=='stopped_after_wave2' and hold['coordinator_exited'],'Wave 2 not finished/held')
    tc.require(not any(c['id'] in tr.progress(parent) for c in parent['candidates'][4:]),'later parent progress exists')
    job=copy.deepcopy(parent)
    entries=list(parent['source_bundle']['files'])
    for name in ('stream_test300.py','stream_test300_io.py','stream_test300_spec.md','stream_test300_preflight.py','stream_test300_transition.py'):
        p=HERE/name;entries.append({'path':str(p.relative_to(tc.REPO)),'sha256':fc.sha256_file(p)})
    entries.sort(key=lambda e:e['path']);bundle={'files':entries,'sha256':fc.json_sha256(entries)}
    job.update(job_id=JOB_ID,runtime_root=str(fc.RUNTIME_ROOT/'cache/jobs'/JOB_ID),staging_root=str(tc.STAGING_ROOT/JOB_ID),
               source_bundle=bundle,parent_file_sha256=fc.sha256_file(PARENT),created_at_utc=fc.utc_now(),
               execution={'cpu_workers':4,'pending_per_gpu':2,'publisher_workers':1,'publication_read_mib_s_during_gpu':32,'numpy_madvise_hugepage':0,'cpu_threads':1},
               continuation={'rank_start':9,'wave_start':3,'parent_job_spec_sha256':parent['job_spec_sha256'],
                  'parent_source_bundle_sha256':parent['source_bundle']['sha256'],
                  'source_drift_audit':{'new_source_bundle_sha256':bundle['sha256'],'reason':'in-memory storage proofs, one-pass publication, overlapped waves'}},
               wave2_hold_sha256=fc.sha256_file(Path(parent['runtime_root'])/'finish_wave2_state.json'))
    parents=[]
    first=fc.read_json(old.PARENT)
    for source,candidates in [(first,first['candidates'][:4]),(parent,parent['candidates'][:4])]:
        for c in candidates:
            tc.require(tr.completed(source,c),'incomplete parent cache')
            cache=Path(c['cache_root'])
            parents.append({'rank':c['rank'],'id':c['id'],'cache_key':c['cache_key'],'cache_root':str(cache),
                'source_job_spec_sha256':source['job_spec_sha256'],'export_manifest_sha256':fc.sha256_file(cache/'export_manifest.json'),
                'validation_sha256':fc.sha256_file(cache/'reproduction_validation.json')})
    job['parent_completed']=parents
    report=Path(job['runtime_root'])/'preflight/report.json'
    tc.require(report.exists() and fc.read_json(report)['status']=='passed','publication preflight not passed')
    job['preflight']={'path':str(report),'sha256':fc.sha256_file(report)}
    job['workloads']={}
    cases=tc.load_cases(tc.resolve(job['dataset']['path']))
    metadata={}
    for c in job['candidates'][4:]:
        patch=fc.read_json(tc.resolve(c['plans']['path']))['configurations']['3d_fullres']['patch_size']
        rows={}
        for case in cases:
            path=Path(c['cache']['root'])/'cases'/case['name'].removesuffix('.nii.gz')/'metadata.json'
            if str(path) not in metadata:
                tc.require(fc.sha256_file(path)==c['test_preprocessing_cases'][case['name']]['metadata_sha256'],'workload metadata drift')
                metadata[str(path)]=fc.read_json(path)
            shape=metadata[str(path)]['resampled_shape_zyx']
            rows[case['name']]=math.prod(math.ceil(max(0,n-p)/(p*.5))+1 for n,p in zip(shape,patch))*len(case['findings'])
        job['workloads'][c['id']]=rows
    profile=fc.read_json(HERE/'profiling_jobs/p001_test300_pipeline16_gpu8/job.json')
    priors={}
    for c in parent['candidates'][:4]:
        timing_dir=Path(parent['runtime_root'])/'case_timings'/c['id']
        timings=[fc.read_json(p) for p in timing_dir.glob('*.json')]
        tc.require(len(timings)==300,'missing Wave 2 timing prior')
        work=sum(row['workload'] for row in profile['full_workloads'][str(c['rank'])])
        rate=max(sum(r['gpu_wall_seconds'] for r in timings),sum(r['cpu_wall_seconds'] for r in timings))/work
        key=c['model_spec']['model_type']+'|'+c['cache']['id']
        priors[key]=max(priors.get(key,0),rate)
    job['stage_prior_seconds']={}
    for c in job['candidates'][4:]:
        key=c['model_spec']['model_type']+'|'+c['cache']['id']
        rate=priors.get(key,max(priors.values()))
        job['stage_prior_seconds'][c['id']]=sum(job['workloads'][c['id']].values())*rate
    job['eta_prior_note']='Wave 2 full-cohort GPU/CPU timings normalized by windows and prompts; unmatched architecture uses slowest observed rate. Publication measured separately.'
    job.pop('parent_wave1',None)
    job['candidates']=job['candidates'][4:]
    for c in job['candidates']:
        parent_key=c['cache_key']
        key='v5_'+fc.json_sha256({'parent_key':parent_key,'source':bundle['sha256'],'execution':job['execution'],'storage_proof':sio.PROOF_SCHEMA})
        c.update(parent_cache_key=parent_key,cache_key=key,cache_version_id='sideexp003_'+key,
                 cache_root=str(fc.RUNTIME_ROOT/'cache/logits/by_cache_key'/key),staging_root=str(Path(job['staging_root'])/key),
                 progress_path=str(Path(job['runtime_root'])/'progress'/(c['id']+'.json')))
    job.pop('job_spec_sha256',None);job['job_spec_sha256']=fc.json_sha256(job)
    validate(job);fc.atomic_write_json(DEFAULT,job)
    print(json.dumps({'job':str(DEFAULT),'job_spec_sha256':job['job_spec_sha256']}))


def staging_path(job,c):return Path(job['runtime_root'])/'staging_manifests'/(c['id']+'.json')


def staging_manifest(job,c):
    path=staging_path(job,c)
    if not path.exists():return None
    m=fc.read_json(path);body=copy.deepcopy(m);sha=body.pop('manifest_sha256')
    tc.require(fc.json_sha256(body)==sha and m['job_spec_sha256']==job['job_spec_sha256'] and m['cache_key']==c['cache_key'],'staging manifest drift')
    tc.require(m['cases']==job['dataset']['cases'] and m['findings']==job['dataset']['findings'],'incomplete staging manifest')
    tc.require(m['dtype']==sio.choose_dtype(m['records']),'dtype selection drift')
    return m


def commit_staging(job,c,cases):
    records=[]
    for case in cases:
        ap,rp=tc.stage_paths(c,case['name']);r=fc.read_json(rp)
        sio.validate_proof(r,case,job,c)
        tc.require(ap.is_file() and ap.stat().st_size==r['npy_bytes'],'staged file missing/size drift')
        records.append(r)
    m={'job_spec_sha256':job['job_spec_sha256'],'candidate_id':c['id'],'cache_key':c['cache_key'],
       'cases':len(records),'findings':sum(len(r['finding_indices']) for r in records),'dtype':sio.choose_dtype(records),
       'array_bytes':sum(r['npy_bytes'] for r in records),'records':records,'completed_at_utc':fc.utc_now()}
    tc.require(m['cases']==job['dataset']['cases'] and m['findings']==job['dataset']['findings'],'incomplete staging')
    m['manifest_sha256']=fc.json_sha256(m);fc.atomic_write_json(staging_path(job,c),m)
    return m


# GPU and native restoration paths are copied from the frozen j004 wrapper;
# only the staging proof writer and end-of-worker staging barrier differ.
def restore_one(job, c, case):
    tc.inference_imports()
    old = tc.load_stage(c, case, job['job_spec_sha256'])
    if old is not None:
        return old
    ap, rp = crop_paths(c, case)
    record = fc.read_json(rp)
    tc.require(record['job_spec_sha256'] == job['job_spec_sha256'] and record['cache_key'] == c['cache_key'] and record['name'] == case['name'], 'foreign crop')
    timer = prof.Timers()
    start = time.monotonic()
    crop = timer.call('crop_read', np.load, ap, mmap_mode='r', allow_pickle=False)
    tc.require(timer.call('crop_hash', tc.array_sha256, crop) == record['array_sha256'], 'crop hash mismatch')
    meta_path = Path(c['cache']['root']) / 'cases' / case['name'].removesuffix('.nii.gz') / 'metadata.json'
    tc.require(fc.sha256_file(meta_path) == c['test_preprocessing_cases'][case['name']]['metadata_sha256'], 'metadata drift')
    with memory_slot(job, math.prod(case['shape']) * len(case['findings']) * 16 + crop.nbytes * 2 + 2 * GIB):
        arr, details = prof.geometry_from_crop(crop, fc.read_json(meta_path), case, timer)
        out = sio.save_stage(job, c, case, arr, details, timer)
        out.update(inference_seconds=record['gpu_wall_seconds'] + time.monotonic() - start,
                   gpu_wall_seconds=record['gpu_wall_seconds'], cpu_wall_seconds=time.monotonic() - start,
                   gpu_timings=record['timings'], cpu_timings=timer.seconds)
        fc.atomic_write_json(tc.stage_paths(c, case['name'])[1], out)
        fc.atomic_write_json(Path(job['runtime_root']) / 'case_timings' / c['id'] / (case['name'] + '.json'), out)
        del arr
    del crop
    # Only remove committed handoffs after the native record is durable.
    ap.unlink()
    rp.unlink()
    return out


def worker(job, c):
    validate(job)
    ex = tc.inference_imports()
    import torch
    import run_voxtell_val_inference as inf
    from common import sorted_prompts
    from voxtell_preprocessed_cache import image_padding_value
    from analyze_candidates import require_candidate_config
    require_candidate_config(c)
    for label in ('checkpoint', 'plans', 'config'):
        tc.require(fc.sha256_file(tc.resolve(c[label]['path'])) == c[label]['sha256'], label + ' drift')
    if c['model_spec']['path']:
        tc.require(fc.sha256_file(tc.resolve(c['model_spec']['path'])) == c['model_spec']['sha256'], 'model spec drift')
    tc.require(fc.sha256_file(Path(c['cache']['manifest_path'])) == c['cache']['manifest_sha256'], 'preprocessing drift')
    tc.require(torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] // 1024**2 >= 20000, 'GPU launch gate')
    torch.set_num_threads(4)
    tc.require(not np._core.multiarray._get_madvise_hugepage(), 'allocation flag not active')
    cases = tc.load_cases(tc.resolve(job['dataset']['path']))
    smokes = tc.smoke_cases(cases, c['cache']['root'])
    ordered = sorted(cases, key=lambda case: case['name'] not in smokes)
    root = Path(c['cache_root'])
    state = {'job_id': JOB_ID, 'job_spec_sha256': job['job_spec_sha256'], 'candidate_id': c['id'],
             'rank': c['rank'], 'wave': c['wave'], 'gpu': c['gpu'], 'cases': 0, 'gpu_cases': 0,
             'findings': 0, 'status': 'loading', 'started_at_utc': fc.utc_now()}
    start = time.monotonic()
    def update(**kw):
        state.update(kw, elapsed_seconds=time.monotonic() - start, updated_at_utc=fc.utc_now())
        fc.atomic_write_json(Path(c['progress_path']), state)
    with tc.exclusive(root / '.worker.lock'):
        try:
            if tr.completed(job, c, verify_hashes=True):
                update(status='strict_passed', cases=300, gpu_cases=300, findings=582)
                return
            update()
            predictor = None
            pending = []
            done = []
            def collect_one():
                result = pending.pop(0).result()
                done.append(result)
                update(cases=len(done), findings=sum(len(r['finding_indices']) for r in done), pending_cpu=len(pending))
            with futures.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'), initializer=cpu_init) as pool:
                for i, case in enumerate(ordered):
                    aborted(job)
                    while pending and (pending[0].done() or len(pending) >= 2):
                        collect_one()
                    update(current_case=case['name'], status='smoke_running' if i < len(smokes) else 'exporting')
                    existing = tc.load_stage(c, case, job['job_spec_sha256'])
                    ap, rp = crop_paths(c, case)
                    if existing is None and not rp.exists():
                        while prof.mem_available() < 64 * GIB + 16 * GIB:
                            aborted(job)
                            time.sleep(2)
                        if predictor is None:
                            begin = time.monotonic()
                            predictor = ex.build_predictor(c, torch.device('cuda:0'), embeddings=None)
                            update(model_load_seconds=time.monotonic() - begin)
                        timer = prof.Timers()
                        begin = time.monotonic()
                        image, meta = timer.call('input_read_hash', tc.load_test_image, c, case)
                        fn = inf.predict_preprocessed_crop_branch_logits if isinstance(predictor, ex.DualBranchVoxTellPredictor) else inf.predict_preprocessed_crop_logits
                        with prof.timed_attr(predictor, 'embed_text_prompts', timer, 'text_embeddings'):
                            with prof.CudaTimers(predictor, timer):
                                crop = timer.call('crop_prediction_total', fn, predictor, image, sorted_prompts(case), padding_value=image_padding_value(meta))
                        if isinstance(crop, dict):
                            crop = crop['final']
                        del image
                        tc.require(shutil.disk_usage(job['staging_root']).free >= crop.nbytes + 2 * GIB, 'handoff disk full')
                        timer.call('handoff_write', tc.atomic_save_npy, ap, crop)
                        sha = timer.call('handoff_hash', tc.array_sha256, crop)
                        fc.atomic_write_json(rp, {'job_spec_sha256': job['job_spec_sha256'], 'cache_key': c['cache_key'],
                            'name': case['name'], 'array_sha256': sha, 'gpu_wall_seconds': time.monotonic() - begin, 'timings': timer.seconds})
                        del crop
                    pending.append(pool.submit(restore_one, job, c, case))
                    update(gpu_cases=i + 1, pending_cpu=len(pending))
                    if i + 1 == len(smokes):
                        while pending:
                            collect_one()
                        free = torch.cuda.mem_get_info()[0] // 1024**2
                        tc.require(free >= 4096, 'post-smoke GPU gate')
                        update(status='smoke_waiting', free_mib_after_smoke=free, smoke_cases=smokes)
                        gate = Path(job['runtime_root']) / 'control' / f"wave_{c['wave']:02d}.continue"
                        while not gate.exists():
                            aborted(job)
                            time.sleep(2)
                        tc.require(fc.read_json(gate)['job_spec_sha256'] == job['job_spec_sha256'], 'foreign gate')
                while pending:
                    collect_one()
                del predictor
                torch.cuda.empty_cache()
                update(status='committing_staging')
                result = pool.submit(commit_staging, job, c, cases).result()
                update(status='staged_ready', dtype=result['dtype'], bytes=result['array_bytes'], eta_seconds=0)
        except BaseException as exc:
            update(status='failed', error=str(exc))
            raise



def publisher(job):
    validate(job);cpu_init()
    root=Path(job['runtime_root']);cases=tc.load_cases(tc.resolve(job['dataset']['path']))
    budget=sio.ReadBudget(root)
    with tc.exclusive(root/'publisher.lock'):
        for c in job['candidates']:
            progress_path=root/'publication_progress'/(c['id']+'.json')
            state={'job_spec_sha256':job['job_spec_sha256'],'rank':c['rank'],'candidate_id':c['id'],
                   'phase':'waiting_for_staging','published_cases':0,'verified_cases':0,'source_bytes_total':0,
                   'destination_bytes_total':0,'started_at_utc':fc.utc_now()}
            def update(**kw):
                state.update(kw,updated_at_utc=fc.utc_now())
                state['written_cases']=state['published_cases']+int(state['phase']=='verifying_destination')
                fc.atomic_write_json(progress_path,state)
                fc.atomic_write_json(root/'publisher_state.json',state)
            try:
                while True:
                    aborted(job)
                    control=fc.read_json(root/'publisher_control.json')
                    tc.require(control['job_spec_sha256']==job['job_spec_sha256'],'foreign publisher control')
                    if c['id'] in control.get('ready_ids',[]):break
                    update();time.sleep(5)
                m=staging_manifest(job,c);tc.require(m is not None,'missing staging manifest')
                tc.require([r['name'] for r in m['records']]==[case['name'] for case in cases],'staging case order mismatch')
                cache=Path(c['cache_root'])
                with tc.exclusive(cache/'.publisher.lock'):
                    already_complete=tr.completed(job,c)
                    if not already_complete:
                        export_path=cache/'export_manifest.json'
                        if export_path.exists():tc.require(fc.read_json(export_path)['job_spec_sha256']==job['job_spec_sha256'],'foreign export manifest')
                        fc.atomic_write_json(export_path,{'status':'publishing','job_id':JOB_ID,'job_spec_sha256':job['job_spec_sha256'],
                                              'candidate_id':c['id'],'cache_key':c['cache_key'],'dataset':job['dataset']})
                    records=[];started=time.monotonic()
                    for case,record in zip(cases,m['records']):
                        aborted(job)
                        if already_complete:
                            existing=fc.read_json(cache/'case_manifests'/(case['name']+'.json'))
                            sio.validate_proof(record,case,job,c)
                            sio.verify_geometry(existing,case)
                            tc.require(existing['job_spec_sha256']==job['job_spec_sha256'] and existing['cache_key']==c['cache_key']
                                       and existing['array_sha256']==record['storage_proof'][m['dtype']+'_sha256'],'completed cache provenance drift')
                            sio.verify_file(Path(existing['array_path']),record['shape'],m['dtype'],existing['array_sha256'])
                            result=existing
                        else:
                            def detail(**kw):
                                if 'case' in kw:kw['active_case']=kw.pop('case')
                                if 'source_bytes' in kw:kw['case_source_bytes']=kw.pop('source_bytes')
                                update(**kw)
                            result=sio.publish_one(job,c,case,record,m['dtype'],budget=budget,progress=detail)
                        records.append(result)
                        update(phase='publishing',published_cases=len(records),verified_cases=len(records),
                               dtype=m['dtype'],case_source_bytes=0,
                               source_bytes_total=sum(r.get('publication_source_bytes',0) for r in records),
                               destination_bytes_total=sum(r['npy_bytes'] for r in records),
                               elapsed_seconds=time.monotonic()-started,throttle_wait_seconds=budget.wait_seconds)
                    validation=sio.completed_manifest(job,c,cases,records)
                    if not already_complete:
                        fc.atomic_write_json(cache/'reproduction_validation.json',validation)
                        fc.atomic_write_json(cache/'export_manifest.json',{'job_id':JOB_ID,'job_spec_sha256':job['job_spec_sha256'],
                            'candidate_id':c['id'],'cache_key':c['cache_key'],'candidate_source':c,'dataset':job['dataset'],
                            'paired_val200':c['paired_val200'],'status':'strict_passed','metric_status':tc.METRIC_STATUS,
                            'cases':records,'case_count':300,'finding_count':582,'completed_at_utc':fc.utc_now(),
                            'publication_wall_seconds':time.monotonic()-started,'source_read_policy':'one pass per unpublished case',
                            'staging_manifest_sha256':m['manifest_sha256']})
                    stages=Path(c['staging_root'])
                    tc.require(stages==Path(job['staging_root'])/c['cache_key'],'unsafe stage cleanup')
                    if stages.exists():tc.cleanup_stage(stages)
                    update(phase='complete',completed_at_utc=fc.utc_now(),array_bytes=validation['array_bytes'])
            except BaseException as exc:
                update(phase='failed',error=str(exc));raise
        fc.atomic_write_json(root/'publisher_complete.json',{'job_spec_sha256':job['job_spec_sha256'],'status':'complete','models':12,'at_utc':fc.utc_now()})


def container_command(job,c=None):
    name=f"sideexp003_{JOB_ID}_publisher" if c is None else f"sideexp003_{JOB_ID}_w{c['wave']:02d}_g{c['gpu']}"
    args=['docker','run','-d','--name',name,'--label','rex.test_job='+job['job_spec_sha256'],'--ipc=host','--shm-size=32g',
          '--user',f'{os.getuid()}:{os.getgid()}','-w','/workspace']
    if c is not None:args+=['--gpus',f"device={c['gpu']}"]
    mounts=[(tc.REPO,'/workspace','ro'),('/data/hengjie','/data/hengjie','ro'),('/mnt/shengdata1','/mnt/shengdata1','ro'),
            (job['runtime_root'],job['runtime_root'],'rw')]
    for cand in ([c] if c is not None else job['candidates']):
        mounts.extend([(cand['cache_root'],cand['cache_root'],'rw'),(cand['staging_root'],cand['staging_root'],'rw')])
    for src,dst,mode in mounts:args+=['-v',f'{src}:{dst}:{mode}']
    args+=['--tmpfs','/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m']
    for key,value in {'NUMPY_MADVISE_HUGEPAGE':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                      'HOME':'/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                      'HF_HOME':'/data/hengjie/datasets/rexgroundingct/.hf_home','HF_HUB_CACHE':'/data/hengjie/datasets/rexgroundingct/.hf_home/hub',
                      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():args+=['-e',f'{key}={value}']
    args+=[job['container']['image_id'],'python','/workspace/'+str(Path(__file__).relative_to(tc.REPO)),'publisher' if c is None else 'worker']
    if c is not None:args+=['--rank',str(c['rank'])]
    return name,args


def capacity(job,stats,new_members,shared_free,local_free):
    """Existing bytes already reduce free space; reserve all outstanding work."""
    needed_ids={c['id'] for c in new_members}
    needed_ids.update(k for k,v in stats.items() if v.get('started') and not v.get('complete'))
    source_full=job['output_elements_per_model']*4+job['dataset']['cases']*128
    shared_need=32*GIB;local_need=128*GIB
    for c in job['candidates']:
        if c['id'] not in needed_ids:continue
        s=stats.get(c['id'],{})
        itemsize=2 if s.get('dtype')=='float16' else 4
        final=job['output_elements_per_model']*itemsize+job['dataset']['cases']*128
        shared_need+=max(0,final-s.get('published_bytes',0))
        local_need+=max(0,source_full-s.get('staged_bytes',0))
    return {'ok':shared_free>=tc.MIN_SHARED+shared_need and local_free>=local_need,
            'shared_outstanding_bytes':shared_need,'local_outstanding_bytes':local_need}


class Throttle:
    def __init__(self):self.baseline=None;self.high=0;self.low=0;self.paused=False
    def check(self,cost,gpu_active):
        if not gpu_active:self.high=0;self.low=0;self.paused=False;return
        if self.baseline is None or cost is None:return
        ratio=cost/self.baseline
        if ratio>2:self.high+=1;self.low=0
        elif ratio<1.5:self.low+=1;self.high=0
        else:self.high=0;self.low=0
        if self.high>=2:self.paused=True
        if self.low>=2:self.paused=False


class Observation:
    def __init__(self,job):self.job=job;self.rows={c['id']:{} for c in job['candidates']};self.completed=set()
    def collect(self):
        root=Path(self.job['runtime_root']);states={}
        for c in self.job['candidates']:
            rows=self.rows[c['id']]
            for path in (root/'case_timings'/c['id']).glob('*.json'):
                if path.name not in rows:
                    r=fc.read_json(path);tc.require(r['job_spec_sha256']==self.job['job_spec_sha256'],'foreign timing');rows[path.name]=r
            progress_path=Path(c['progress_path']);p=fc.read_json(progress_path) if progress_path.exists() else {}
            pub_path=root/'publication_progress'/(c['id']+'.json');pub=fc.read_json(pub_path) if pub_path.exists() else {}
            m=staging_manifest(self.job,c)
            if c['id'] not in self.completed and (Path(c['cache_root'])/'reproduction_validation.json').exists():
                if tr.completed(self.job,c):self.completed.add(c['id'])
            recent=sorted(rows.values(),key=lambda r:r['completed_at_utc'])[-20:]
            cost=None
            if len(recent)>=3:
                cost=sum(r['gpu_timings'].get('input_read_hash',0)+r['gpu_timings'].get('handoff_write',0)+r['cpu_timings'].get('local_write',0)
                         for r in recent)/sum(r['output_elements'] for r in recent)
            gpu_mean=statistics.mean(r['gpu_wall_seconds'] for r in recent) if recent else None
            workload=self.job.get('workloads',{}).get(c['id'],{})
            stage_eta=None
            if len(rows)>=10 and workload:
                done_work=sum(workload[r['name']] for r in rows.values())
                stage_eta=max(sum(r['gpu_wall_seconds'] for r in rows.values()),sum(r['cpu_wall_seconds'] for r in rows.values()))/done_work*sum(v for k,v in workload.items() if k+'.json' not in rows)
            states[c['id']]={'rank':c['rank'],'wave':c['wave'],'started':bool(p or m),'complete':c['id'] in self.completed,
                'gpu_cases':p.get('gpu_cases',0),'staged_cases':len(rows),'staged_bytes':sum(r['npy_bytes'] for r in rows.values()),
                'phase':pub.get('phase') if pub else p.get('status','queued'),'gpu_phase':p.get('status','queued'),
                'dtype':m['dtype'] if m else None,'published_cases':pub.get('published_cases',300 if c['id'] in self.completed else 0),
                'verified_cases':pub.get('verified_cases',300 if c['id'] in self.completed else 0),
                'published_bytes':pub.get('destination_bytes_total',0),'publication':pub,'normalized_io_cost':cost,
                'gpu_seconds_per_case':gpu_mean,'gpu_remaining_seconds':gpu_mean*(300-p.get('gpu_cases',0)) if gpu_mean else None,
                'stage_remaining_seconds':stage_eta,
                'gpu_error':p.get('error')}
        return states


def estimate(job,stats,gpu_active):
    """Separate measured publication cost from workload-weighted staging ETA."""
    pubs=[s.get('publication',{}) for s in stats.values()]
    nbytes=sum(p.get('source_bytes_total',0) for p in pubs)
    seconds=sum(p.get('elapsed_seconds',0) for p in pubs)
    baseline=None
    if 'preflight' in job:baseline=fc.read_json(Path(job['preflight']['path']))['stream_source_bytes_per_second']
    rate=nbytes/seconds if nbytes and seconds else baseline
    remaining=sum(max(0,job['output_elements_per_model']*4-s.get('publication',{}).get('source_bytes_total',0)) for s in stats.values() if not s['complete'])
    publication=remaining/rate if rate else None
    stage_by_id={}
    for cid,s in stats.items():
        value=s.get('stage_remaining_seconds')
        if value is None and cid in job.get('stage_prior_seconds',{}):
            value=job['stage_prior_seconds'][cid]*(1-s.get('staged_cases',0)/300)
        if s['complete'] or s.get('staged_cases')==300:value=0
        stage_by_id[cid]=value
    staging=sum(max((stage_by_id[cid] or 0 for cid,s in stats.items() if s['wave']==wave),default=0) for wave in (3,4,5)) if all(v is not None for v in stage_by_id.values()) else None
    unmeasured=any(not s['started'] for s in stats.values())
    return {'publication_remaining_seconds_at_measured_rate':publication,'publication_source_bytes_per_second':rate,
            'publication_rate_basis':'live (includes throttling)' if seconds else 'private cold benchmark',
            'measured_active_staging_seconds':staging,'future_gpu_waves_unmeasured':unmeasured,
            'completion_range_seconds':[max(staging or 0,publication or 0),((staging or 0)+(publication or 0))*1.5] if staging is not None and publication is not None else None,
            'caveat':'Window/prompt weighted live staging after ten cases, Wave 2 prior otherwise; publication rates change with overlap and disk load. Range is approximate, not a confidence interval.'}


def run(job):
    validate(job);tc.require(socket.gethostname()=='shenggpu8','wrong host')
    tc.require(tr.command(['git','branch','--show-current']).stdout.strip()=='gpu8','wrong branch')
    parent=fc.read_json(PARENT);original=fc.read_json(old.PARENT)
    hold=fc.read_json(Path(parent['runtime_root'])/'finish_wave2_state.json')
    tc.require(hold['status']=='stopped_after_wave2' and hold['coordinator_exited'],'parent not held')
    root=Path(job['runtime_root']);root.mkdir(parents=True,exist_ok=True)
    Path(job['staging_root']).mkdir(parents=True,exist_ok=True)
    for c in job['candidates']:
        Path(c['cache_root']).mkdir(parents=True,exist_ok=True);Path(c['staging_root']).mkdir(parents=True,exist_ok=True)
    owned=[];gpu_active={};finished_gpu=set();started_waves=set();wave_starts={};throttle=Throttle();last_throttle=0
    obs=Observation(job);started=time.monotonic()
    def update(status,stats,**kw):
        value={'status':status,'job_id':JOB_ID,'job_spec_sha256':job['job_spec_sha256'],'pid':os.getpid(),
               'updated_at_utc':fc.utc_now(),'elapsed_seconds':time.monotonic()-started,'models':stats,
               'gpu_wave':max((c['wave'] for c in job['candidates'] if c['id'] in gpu_active),default=None),
               'completed_models':sum(s['complete'] for s in stats.values()),'parent_completed_models':8,
               'available_ram_bytes':prof.mem_available(),'shared_free_bytes':shutil.disk_usage(root).free,
               'local_free_bytes':shutil.disk_usage(job['staging_root']).free,'publisher_paused':throttle.paused,**kw}
        value['eta']=estimate(job,stats,bool(gpu_active))
        fc.atomic_write_json(root/'state.json',value)
        with (root/'monitor.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def launch(c=None):
        name,args=container_command(job,c);d=tr.docker_state(name)
        if d:
            tc.require(not d['State']['Running'] and d['Config']['Labels'].get('rex.test_job')==job['job_spec_sha256'],'foreign/live container')
            log=tr.command(['docker','logs',name],check=False)
            fc.atomic_write_text(root/'logs'/(name+'.previous.log'),log.stdout+log.stderr)
            tr.command(['docker','rm',name])
        fc.atomic_write_json(root/'commands'/((c['id'] if c else 'publisher')+'.json'),{'argv':args,'at_utc':fc.utc_now()})
        tr.command(args);owned.append(name)
        return name
    def signal_stop(signum,frame):raise RuntimeError('coordinator interrupted')
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    with tc.exclusive(root/'launch.lock'),tc.exclusive(Path(parent['runtime_root'])/'launch.lock'),tc.exclusive(Path(original['runtime_root'])/'launch.lock'):
        try:
            for c in [None,*job['candidates']]:
                name,_=container_command(job,c);d=tr.docker_state(name)
                tc.require(not d or not d['State']['Running'],'live owner: '+name)
            fc.atomic_write_json(root/'memory_reservations.json',{})
            initial=obs.collect()
            for c in job['candidates']:
                if staging_manifest(job,c) is not None or initial[c['id']]['complete']:finished_gpu.add(c['id'])
            fc.atomic_write_json(root/'publisher_control.json',{'job_spec_sha256':job['job_spec_sha256'],'gpu_active':True,
                'read_mib_s':32,'paused':False,'ready_ids':sorted(finished_gpu)})
            pubname=launch()
            while True:
                aborted(job);stats=obs.collect()
                pubstate=tr.docker_state(pubname)
                tc.require(pubstate and (pubstate['State']['Running'] or pubstate['State']['ExitCode']==0),'publisher exited unsuccessfully')
                if not pubstate['State']['Running']:
                    tc.require(all(s['complete'] for s in stats.values()),'publisher exited before all caches completed')
                for cid,name in list(gpu_active.items()):
                    d=tr.docker_state(name);tc.require(d is not None,'GPU worker disappeared')
                    tc.require(d['State']['Running'] or d['State']['ExitCode']==0,'GPU worker failed: '+name)
                    tc.require(stats[cid]['gpu_phase']!='failed','GPU failure: '+str(stats[cid].get('gpu_error')))
                    if not d['State']['Running']:
                        c=next(c for c in job['candidates'] if c['id']==cid)
                        tc.require(staging_manifest(job,c) is not None,'worker exited without full staging proof')
                        finished_gpu.add(cid);gpu_active.pop(cid)
                        log=tr.command(['docker','logs',name],check=False)
                        fc.atomic_write_text(root/'logs'/(name+'.log'),log.stdout+log.stderr)
                for wave in list(started_waves):
                    members=[c for c in job['candidates'] if c['wave']==wave]
                    gate=root/'control'/f'wave_{wave:02d}.continue'
                    if not gate.exists() and all(stats[c['id']]['gpu_phase'] in ('smoke_waiting','staged_ready') or c['id'] in finished_gpu for c in members):
                        for c in members:
                            if c['id'] not in finished_gpu:
                                p=fc.read_json(Path(c['progress_path']));tc.require(p['free_mib_after_smoke']>=4096,'smoke memory failed')
                        fc.atomic_write_json(gate,{'job_spec_sha256':job['job_spec_sha256'],'at_utc':fc.utc_now()})
                    if all(c['id'] in finished_gpu for c in members) and not (root/f'wave_{wave:02d}_gpu_timing.json').exists():
                        fc.atomic_write_json(root/f'wave_{wave:02d}_gpu_timing.json',{'wave':wave,'gpu_stage_wall_seconds':time.monotonic()-wave_starts[wave],'at_utc':fc.utc_now()})
                admission=None
                if not gpu_active:
                    next_wave=next((w for w in (3,4,5) if any(c['id'] not in finished_gpu for c in job['candidates'] if c['wave']==w)),None)
                    if next_wave is not None:
                        todo=[c for c in job['candidates'] if c['wave']==next_wave and c['id'] not in finished_gpu]
                        admission=capacity(job,stats,todo,shutil.disk_usage(root).free,shutil.disk_usage(job['staging_root']).free)
                        free=tr.gpu_free()
                        if admission['ok'] and prof.mem_available()>=128*GIB and all(free[c['gpu']]>=20000 for c in todo):
                            validate(job)
                            # Freeze the baseline before the first publication overlaps GPU work.
                            if throttle.baseline is None:
                                costs=[s['normalized_io_cost'] for s in stats.values() if s['wave']==3 and s['normalized_io_cost'] is not None]
                                if costs:throttle.baseline=statistics.median(costs)
                            gate=root/'control'/f'wave_{next_wave:02d}.continue';gate.unlink(missing_ok=True)
                            for c in todo:gpu_active[c['id']]=launch(c)
                            started_waves.add(next_wave);wave_starts[next_wave]=time.monotonic()
                ready=[c['id'] for c in job['candidates'] if c['id'] in finished_gpu]
                if throttle.baseline is None and ready:
                    costs=[s['normalized_io_cost'] for s in stats.values() if s['wave']==3 and s['normalized_io_cost'] is not None]
                    if costs:throttle.baseline=statistics.median(costs)
                current_costs=[stats[cid]['normalized_io_cost'] for cid in gpu_active if stats[cid]['normalized_io_cost'] is not None]
                if time.monotonic()-last_throttle>=60:
                    throttle.check(statistics.median(current_costs) if current_costs else None,bool(gpu_active));last_throttle=time.monotonic()
                fc.atomic_write_json(root/'publisher_control.json',{'job_spec_sha256':job['job_spec_sha256'],'gpu_active':bool(gpu_active),
                    'read_mib_s':32 if gpu_active else 0,'paused':throttle.paused if gpu_active else False,'ready_ids':ready,
                    'io_baseline':throttle.baseline,'at_utc':fc.utc_now()})
                active_publish=any(s['phase'] in ('publishing','verifying_destination') for s in stats.values())
                phase='gpu_export_and_publication' if gpu_active and active_publish else 'gpu_export' if gpu_active else 'publication'
                if admission and not admission['ok']:phase='waiting_for_capacity_and_publishing'
                update(phase,stats,capacity=admission)
                if all(s['complete'] for s in stats.values()) and not pubstate['State']['Running']:
                    tc.require(len(finished_gpu)==12,'incomplete inference roster')
                    inventory={'job_spec_sha256':job['job_spec_sha256'],'parents':job['parent_completed'],
                        'caches':[{'rank':c['rank'],'id':c['id'],'cache_key':c['cache_key'],'cache_root':c['cache_root'],'paired_val200':c['paired_val200']} for c in job['candidates']],
                        'status':'complete','at_utc':fc.utc_now(),'warnings':[],
                        'reproduction_command':f'python {HERE}/stream_test300.py run --job {JOB_ID}'}
                    for row in inventory['caches']:
                        cache=Path(row['cache_root'])
                        v=fc.read_json(cache/'reproduction_validation.json')
                        row.update(dtype=v['dtype'],cases=v['cases'],findings=v['findings'],array_bytes=v['array_bytes'],
                            validation_sha256=fc.sha256_file(cache/'reproduction_validation.json'),
                            export_manifest_sha256=fc.sha256_file(cache/'export_manifest.json'))
                        if v['dtype']=='float32':inventory['warnings'].append(f"Rank {row['rank']}: float32 required for exact zero-threshold preservation")
                    inventory['wave_timings']=[fc.read_json(p) for p in sorted(root.glob('wave_*_gpu_timing.json'))]
                    inventory['array_bytes']=sum(r.get('array_bytes',0) for r in inventory['caches'])
                    fc.atomic_write_json(root/'inventory.json',inventory)
                    fc.atomic_write_json(DEFAULT.parent/'inventory.json',inventory)
                    update('complete',stats);break
                time.sleep(30)
        except BaseException as exc:
            for name in owned:tr.command(['docker','stop','--time','30',name],check=False)
            update('failed',obs.collect(),error=str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker','publisher','watch']);p.add_argument('--rank',type=int);p.add_argument('--once',action='store_true');p.add_argument('--job',default=str(DEFAULT))
    args=p.parse_args()
    if args.action=='prepare':prepare();return
    requested=Path(args.job)
    if args.job==JOB_ID:requested=DEFAULT
    tc.require(requested.resolve()==DEFAULT.resolve(),'wrong job path')
    job=fc.read_json(requested)
    if args.action=='run':run(job)
    elif args.action=='worker':worker(job,next(c for c in job['candidates'] if c['rank']==args.rank))
    elif args.action=='publisher':publisher(job)
    else:print(json.dumps(fc.read_json(Path(job['runtime_root'])/'state.json'),indent=2))


if __name__=='__main__':main()
