#!/usr/bin/env python3
"""Isolated, single-inference test300 pipeline benchmark; never resumes production."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import copy
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import fresh_cache as fc
import test300_cache as tc

HERE = Path(__file__).resolve().parent
JOB_ID = 'p001_test300_pipeline16_gpu8'
ROOT = Path('/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/profiling_jobs') / JOB_ID
LOCAL = Path('/data/hengjie/sideexp003_staging') / JOB_ID
DEFAULT_JOB = HERE / 'profiling_jobs' / JOB_ID / 'job.json'
PARENT = HERE / 'cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json'
GIB = 1024**3
CHUNK = 4194304

def check(value, message):
    if not value:
        raise RuntimeError(message)

def read(path):
    return fc.read_json(Path(path))

def write(path, value):
    fc.atomic_write_json(Path(path), value)

def run_cmd(args):
    return subprocess.check_output(args, text=True).strip()

def mem_available():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))

def io_stats():
    return {k: int(v) for k, v in (s.split(':') for s in Path('/proc/self/io').read_text().splitlines())}

class Timers:
    def __init__(self):
        self.seconds = {}
    @contextlib.contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter()-start
    def call(self, name, func, *args, **kwargs):
        with self.stage(name):
            return func(*args, **kwargs)

def select_cases(rows):
    ordered = sorted(rows, key=lambda r: (r['workload'], r['case']['name']))
    requested = [max(rows, key=lambda r: (r['input_elements'], r['case']['name'])),
                 max(rows, key=lambda r: (r['output_elements'], r['case']['name'])),
                 ordered[round((len(rows)-1)*.5)], ordered[round((len(rows)-1)*.25)]]
    result = []
    for row in requested + list(reversed(ordered)):
        if row['case']['name'] not in [r['case']['name'] for r in result]:
            result.append(row)
        if len(result) == 4:
            break
    check(len(result) == 4, 'need four distinct CTs per model')
    return result

def space_check(shared_free, local_free, shared_need, local_need):
    check(shared_free >= tc.MIN_SHARED + shared_need, 'shared 20 TiB reserve')
    check(local_free >= local_need, 'insufficient local scratch')

def capacity(available, reserved, required):
    return available - reserved - required >= 64*GIB

def source_check(job):
    check(fc.json_sha256({k:v for k,v in job.items() if k != 'sha256'}) == job['sha256'], 'job identity mismatch')
    for path, sha in job['sources'].items():
        check(fc.sha256_file(tc.resolve(path)) == sha, 'source drift: '+path)
    check(fc.sha256_file(tc.resolve(job['dataset']['path'])) == job['dataset']['sha256'], 'dataset drift')

def prepare(path):
    check(not path.exists(), 'immutable job already exists')
    check(run_cmd(['git','branch','--show-current']) == 'gpu8', 'requires gpu8 branch')
    parent = read(PARENT)
    cases = tc.load_cases(parent['dataset']['path'])
    candidates = copy.deepcopy(parent['candidates'][4:8])
    full = {}
    metadata = {}
    for c in candidates:
        plans = read(c['plans']['path'])
        patch = plans['configurations']['3d_fullres']['patch_size']
        rows = []
        for case in cases:
            p = Path(c['cache']['root'])/'cases'/case['name'].removesuffix('.nii.gz')/'metadata.json'
            if str(p) not in metadata:
                m = read(p)
                check(fc.sha256_file(p) == c['test_preprocessing_cases'][case['name']]['metadata_sha256'], 'metadata drift')
                metadata[str(p)] = m
            m = metadata[str(p)]
            windows = math.prod(math.ceil(max(0, n-p)/(p*.5))+1 for n,p in zip(m['resampled_shape_zyx'], patch))
            f = len(case['findings'])
            elements = math.prod(case['shape'])*f
            crop = math.prod(m['resampled_shape_zyx'])*f
            rows.append({'case': case, 'windows': windows, 'workload': windows*f,
                         'output_elements': elements, 'crop_elements': crop,
                         'input_elements': math.prod(m['resampled_shape_zyx']),
                         'memory_bytes': 4*(6*elements+3*crop)+2*GIB,
                         'metadata_path': str(p), 'metadata_sha256': c['test_preprocessing_cases'][case['name']]['metadata_sha256']})
        full[str(c['rank'])] = rows
        c['selected'] = select_cases(rows)
    all_rows = [r for c in candidates for r in c['selected']]
    native = sum(r['output_elements']*4 for r in all_rows)
    crops = sum(r['crop_elements']*4 for r in all_rows)
    # Retain reference and all three replays, plus crop spool and temporary slack.
    shared_need = native*4+8*GIB
    local_need = native*4+crops+8*GIB
    sources = {str(tc.resolve(x['path'])): x['sha256'] for x in parent['source_bundle']['files']}
    for p in (Path(__file__), HERE/'profile_test300_spec.md', PARENT):
        sources[str(p)] = fc.sha256_file(p)
    job = {'job_id': JOB_ID, 'created_at_utc': fc.utc_now(), 'runtime_root': str(ROOT),
           'local_root': str(LOCAL), 'parent_job_sha256': parent['job_spec_sha256'],
           'dataset': parent['dataset'], 'container': parent['container'],
           'sources': sources, 'candidates': candidates, 'full_workloads': full,
           'space': {'shared_need': shared_need, 'local_need': local_need,
                     'native_fp32_bytes': native, 'crop_bytes': crops},
           'inference_passes': 1, 'cpu_workers': [4,8,16], 'production_resume': False}
    job['sha256'] = fc.json_sha256(job)
    source_check(job)
    space_check(shutil.disk_usage(ROOT.parent.parent).free, shutil.disk_usage('/data').free, shared_need, local_need)
    write(path, job)
    print(json.dumps({'job':str(path), 'sha256':job['sha256'], 'space':job['space'],
                      'selected':{c['rank']:[r['case']['name'] for r in c['selected']] for c in candidates}}, indent=2))

def candidate_for(job, rank, mode):
    c = copy.deepcopy(next(c for c in job['candidates'] if c['rank'] == rank))
    c['staging_root'] = str(Path(job['local_root'])/mode/f'r{rank}')
    c['cache_root'] = str(Path(job['runtime_root'])/mode/f'r{rank}')
    c['cache_key'] = 'profile_'+job['sha256']+f'_r{rank}'
    return c

def record_path(job, rank, name):
    return Path(job['local_root'])/'crops'/f'r{rank}'/f'{name}.json'

def verify_crop(job, rank, row):
    p = record_path(job, rank, row['case']['name'])
    rec = read(p)
    check(rec['job_sha256'] == job['sha256'] and rec['rank'] == rank, 'foreign crop')
    check(rec['metadata_sha256'] == row['metadata_sha256'], 'crop geometry provenance')
    a = np.load(rec['path'], mmap_mode='r', allow_pickle=False)
    check(tc.array_sha256(a) == rec['array_sha256'], 'crop source hash mismatch')
    check(a.dtype == np.float32 and a.size == row['crop_elements'], 'crop shape/dtype mismatch')
    return a, rec

def save_stage(job, c, case, arr, details, timer):
    with timer.stage('finite_range_clip'):
        check(np.isfinite(arr).all(), 'nonfinite logits')
        bounds = [float(arr.min()), float(arr.max())]
        np.clip(arr, -30, 30, out=arr)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    ap, rp = tc.stage_paths(c, case['name'])
    check(shutil.disk_usage(job['local_root']).free >= arr.nbytes+GIB, 'local staging full')
    if rp.exists():
        old=read(rp); check(old['job_spec_sha256']==job['sha256'] and old['cache_key']==c['cache_key'], 'foreign staging output')
    timer.call('local_write', tc.atomic_save_npy, ap, arr)
    sha = timer.call('local_array_hash', tc.array_sha256, arr)
    rec = {**details, 'name':case['name'], 'shape':list(arr.shape), 'dtype':'float32',
           'job_spec_sha256':job['sha256'], 'cache_key':c['cache_key'], 'array_sha256':sha,
           'unclipped_range':bounds, 'completed_at_utc':fc.utc_now(), 'output_elements':arr.size}
    write(rp, rec)
    return rec

@contextlib.contextmanager
def timed_attr(obj, name, timer, label):
    old = getattr(obj,name)
    def wrapped(*a, **kw):
        return timer.call(label, old, *a, **kw)
    setattr(obj,name,wrapped)
    try:
        yield
    finally:
        setattr(obj,name,old)

@contextlib.contextmanager
def geometry_timers(timer):
    import run_voxtell_val_inference as inf
    with contextlib.ExitStack() as stack:
        for obj,name,label in [(inf.F,'interpolate','restore_resample_child'),
                               (inf,'insert_crop_into_image','restore_insert_child'),
                               (inf,'export_prediction_to_ct_layout','orientation_total'),
                               (inf,'restore_cached_native_crop','restore_total')]:
            stack.enter_context(timed_attr(obj,name,timer,label))
        yield

class CudaTimers:
    """Events are resolved once per case; no per-window synchronization."""
    def __init__(self, predictor, timer):
        import torch
        self.torch, self.predictor, self.timer = torch, predictor, timer
        self.events = []
        self.forward_stack = []
        self.hooks = []
    def pair_start(self):
        a,b = self.torch.cuda.Event(enable_timing=True),self.torch.cuda.Event(enable_timing=True)
        a.record()
        return a,b
    def pre(self, *args):
        self.forward_stack.append(self.pair_start())
    def post(self, *args):
        a,b=self.forward_stack.pop(); b.record(); self.events.append(('cuda_network',a,b))
    def __enter__(self):
        torch = self.torch
        self.hooks = [self.predictor.network.register_forward_pre_hook(self.pre), self.predictor.network.register_forward_hook(self.post)]
        self.old_to, self.old_cpu = torch.Tensor.to, torch.Tensor.cpu
        def to(tensor,*a,**kw):
            target = kw.get('device', a[0] if a else None)
            if isinstance(target, torch.Tensor): target = target.device
            target_type = str(target).split(':')[0]
            direction = ('d2h' if tensor.is_cuda and target_type == 'cpu' else
                         'h2d' if not tensor.is_cuda and target_type == 'cuda' else None)
            if direction is None:
                return self.old_to(tensor,*a,**kw)
            ev=self.pair_start(); t=time.perf_counter()
            result=self.old_to(tensor,*a,**kw)
            self.timer.seconds[direction+'_call_wall_child']=self.timer.seconds.get(direction+'_call_wall_child',0)+time.perf_counter()-t
            ev[1].record(); self.events.append(('cuda_'+direction,*ev))
            return result
        def cpu(tensor,*a,**kw):
            if not tensor.is_cuda: return self.old_cpu(tensor,*a,**kw)
            ev=self.pair_start(); t=time.perf_counter(); result=self.old_cpu(tensor,*a,**kw)
            self.timer.seconds['d2h_call_wall_child']=self.timer.seconds.get('d2h_call_wall_child',0)+time.perf_counter()-t
            ev[1].record(); self.events.append(('cuda_d2h',*ev)); return result
        torch.Tensor.to, torch.Tensor.cpu = to, cpu
        return self
    def __exit__(self,*exc):
        self.torch.Tensor.to,self.torch.Tensor.cpu=self.old_to,self.old_cpu
        for h in self.hooks: h.remove()
        self.torch.cuda.synchronize()
        for label,a,b in self.events:
            self.timer.seconds[label]=self.timer.seconds.get(label,0)+a.elapsed_time(b)/1000
        self.timer.seconds['network_forward_calls']=sum(label=='cuda_network' for label,_,_ in self.events)


def gpu_worker(job, rank):
    source_check(job)
    ex=tc.inference_imports()
    import torch
    import run_voxtell_val_inference as inf
    import voxtell_preprocessed_cache as prep
    from analyze_candidates import require_candidate_config
    c=candidate_for(job,rank,'reference')
    require_candidate_config(c)
    for label in ('checkpoint','plans','config'):
        check(fc.sha256_file(tc.resolve(c[label]['path']))==c[label]['sha256'],label+' drift')
    if c['model_spec']['path']:
        check(fc.sha256_file(tc.resolve(c['model_spec']['path']))==c['model_spec']['sha256'],'model spec drift')
    check(fc.sha256_file(Path(c['cache']['manifest_path']))==c['cache']['manifest_sha256'],'preprocessing manifest drift')
    check(torch.cuda.device_count()==1,'exactly one visible GPU required')
    check(torch.cuda.mem_get_info()[0]//1024**2>=20000,'GPU launch gate')
    root=Path(job['runtime_root'])
    timer=Timers(); started=time.time()
    predictor=timer.call('model_loading',ex.build_predictor,c,torch.device('cuda:0'),None)
    write(root/'gpu'/f'r{rank}_init.json', {'timings':timer.seconds,'torch_threads':torch.get_num_threads(), 'patch_size':list(predictor.patch_size), 'tile_step_size':predictor.tile_step_size})
    rows=c['selected']
    for index,row in enumerate(rows):
        case=row['case']; name=case['name']
        journal=root/'gpu'/f'r{rank}_{name}.json'
        crop_record=record_path(job,rank,name)
        if journal.exists():
            verify_crop(job,rank,row)
            tc.load_stage(c,case,job['sha256'])
            continue
        marker=root/'gpu'/f'r{rank}_{name}.started.json'
        # Never rerun an interrupted model call. A committed crop permits CPU recovery.
        if marker.exists():
            check(crop_record.exists(), 'interrupted inference without committed crop; refuse another model pass')
            result=cpu_restore((job,rank,row,'reference'))
            write(journal,{**result,'recovered_without_inference':True,'timing_incomplete':True})
            continue
        write(marker,{'started_at_utc':fc.utc_now(),'job_sha256':job['sha256']})
        t=Timers(); before=io_stats(); begin=time.perf_counter()
        original_restore=inf.restore_cached_native_crop
        def retain(crop,meta,*a,**kw):
            with t.stage('diagnostic_spool'):
                p=crop_record.with_suffix('.npy')
                check(shutil.disk_usage(job['local_root']).free >= crop.nbytes+GIB, 'crop spool full')
                tc.atomic_save_npy(p,crop)
                write(crop_record,{'path':str(p),'array_sha256':tc.array_sha256(crop),'rank':rank,
                                  'job_sha256':job['sha256'],'metadata_sha256':row['metadata_sha256']})
            return t.call('restore_total',original_restore,crop,meta,*a,**kw)
        with contextlib.ExitStack() as stack:
            stack.enter_context(timed_attr(np,'load',t,'npy_load_child'))
            stack.enter_context(timed_attr(np,'ascontiguousarray',t,'contiguous_copy_child'))
            stack.enter_context(timed_attr(np,'stack',t,'numpy_stack_child'))
            stack.enter_context(timed_attr(tc,'load_test_image',t,'input_total'))
            stack.enter_context(timed_attr(prep,'sha256_array',t,'input_hash_child'))
            stack.enter_context(timed_attr(predictor,'embed_text_prompts',t,'embeddings_child'))
            for method in ('predict_sliding_window_return_logits','predict_sliding_window_return_branch_logits'):
                if hasattr(predictor,method):
                    stack.enter_context(timed_attr(predictor,method,t,method+'_child'))
            stack.enter_context(timed_attr(inf,'pad_nd_image',t,'padding_child'))
            stack.enter_context(timed_attr(inf,'predict_preprocessed_crop_logits',t,'crop_prediction_total'))
            stack.enter_context(timed_attr(inf,'predict_preprocessed_crop_branch_logits',t,'crop_prediction_total'))
            stack.enter_context(timed_attr(inf.F,'interpolate',t,'restore_resample_child'))
            stack.enter_context(timed_attr(inf,'insert_crop_into_image',t,'restore_insert_child'))
            stack.enter_context(timed_attr(inf,'export_prediction_to_ct_layout',t,'orientation_total'))
            inf.restore_cached_native_crop=retain
            stack.callback(setattr,inf,'restore_cached_native_crop',original_restore)
            with CudaTimers(predictor,t):
                arr,details=tc.predict_test(predictor,c,case)
        rec=save_stage(job,c,case,arr,details,t); del arr
        elapsed=time.perf_counter()-begin
        after=io_stats()
        write(journal,{'rank':rank,'name':name,'timings':t.seconds,'elapsed_seconds':elapsed,
                       'normal_path_seconds':elapsed-t.seconds.get('diagnostic_spool',0),
                       'io':{k:after[k]-before[k] for k in after},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       'reference_array_sha256':rec['array_sha256'],'completed_at_utc':fc.utc_now()})
        if index==min(1,len(rows)-1):
            check(torch.cuda.mem_get_info()[0]//1024**2>=4096,'post-smoke GPU memory gate')
            write(root/'gpu'/f'r{rank}_smoke.json',{'passed':True,'free_mib':torch.cuda.mem_get_info()[0]//1024**2})
            while not (root/'GPU_SMOKE_RELEASE').exists():
                check(not (root/'ABORT').exists(),'benchmark aborted')
                time.sleep(2)
        print(json.dumps({'rank':rank,'case':name,'seconds':elapsed,'timings':t.seconds}),flush=True)
    publish_group(job,c,'reference')
    write(root/'gpu'/f'r{rank}_complete.json',{'status':'passed','wall_seconds':time.time()-started,'completed_at_utc':fc.utc_now()})

def geometry_from_crop(crop, meta, case, timer):
    import nibabel as nib
    import run_voxtell_val_inference as inf
    from common import ct_rate_abs_path, CT_ROOT
    with geometry_timers(timer):
        restored=inf.restore_cached_native_crop(crop,meta,fill_value=-30)
        with timer.stage('ct_header'):
            path=ct_rate_abs_path(case['name'],CT_ROOT); ct=nib.load(str(path))
            check(list(ct.shape)==case['shape'],'CT shape drift')
            check(np.array_equal(ct.affine,np.asarray(meta['ct_properties']['nibabel_stuff']['original_affine'])),'CT affine drift')
        arr,orientation=inf.export_prediction_to_ct_layout(restored,ct,meta['ct_properties'],case['name'],output_dtype=None)
    return arr, {'orientation':orientation,'affine':ct.affine.tolist(),'ct_shape':list(ct.shape),
                 'ct_path':str(path),'metric_status':tc.METRIC_STATUS,**tc.prompt_contract(case),
                 'preprocessing_image_sha256':meta['image_sha256']}

def cpu_init():
    tc.inference_imports()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def cpu_restore(payload):
    job,rank,row,mode=payload
    c=candidate_for(job,rank,mode); case=row['case']; timer=Timers()
    start=time.perf_counter(); before=io_stats()
    crop,_=timer.call('crop_read_hash',verify_crop,job,rank,row)
    check(fc.sha256_file(Path(row['metadata_path']))==row['metadata_sha256'],'metadata source hash mismatch')
    meta=read(row['metadata_path'])
    arr,details=geometry_from_crop(crop,meta,case,timer)
    rec=save_stage(job,c,case,arr,details,timer)
    if mode!='reference':
        reference=read(tc.stage_paths(candidate_for(job,rank,'reference'),case['name'])[1])
        check(rec['array_sha256']==reference['array_sha256'],'CPU native float32 mismatch')
        for k in ('shape','affine','orientation','finding_indices','prompt_sha256s'):
            check(rec[k]==reference[k],'CPU geometry/prompt mismatch '+k)
    after=io_stats()
    result={'rank':rank,'name':case['name'],'timings':timer.seconds,'elapsed_seconds':time.perf_counter()-start,
            'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'io':{k:after[k]-before[k] for k in after},'array_sha256':rec['array_sha256']}
    write(Path(job['runtime_root'])/mode/'timings'/f'r{rank}_{case["name"]}.json',result)
    return result

def choose_dtype(paths):
    for path in paths:
        raw=np.load(path,mmap_mode='r',allow_pickle=False).reshape(-1)
        for start in range(0,raw.size,CHUNK):
            v=raw[start:start+CHUNK]
            if np.any((v>=0)!=(v.astype(np.float16)>=0)):
                return 'float32'
    return 'float16'

def publish_one(payload):
    job,c,case,dtype,mode=payload; t=Timers(); start=time.perf_counter(); before=io_stats()
    source,rp=tc.stage_paths(c,case['name']); rec=read(rp)
    root=Path(c['cache_root']); root.mkdir(parents=True,exist_ok=True)
    with t.stage('publication_read_convert'):
        raw=np.load(source,mmap_mode='r',allow_pickle=False)
        compact=np.ascontiguousarray(raw,dtype=dtype)
    with t.stage('storage_mask_proof'):
        for offset in range(0,raw.size,CHUNK):
            check(np.array_equal(raw.reshape(-1)[offset:offset+CHUNK]>=0,compact.reshape(-1)[offset:offset+CHUNK]>=0),'storage mask mismatch')
    check(shutil.disk_usage(root).free>=tc.MIN_SHARED+compact.nbytes+GIB,'shared reserve reached')
    path=root/'cases'/f'{case["name"]}.npy'
    manifest=root/'case_manifests'/f'{case["name"]}.json'
    if manifest.exists():
        old=read(manifest);check(old['job_spec_sha256']==job['sha256'] and old['cache_key']==c['cache_key'], 'foreign publication')
    t.call('shared_write_fsync',tc.atomic_save_npy,path,compact)
    sha=t.call('publication_hash',tc.array_sha256,compact)
    rec.update(dtype=dtype,array_sha256=sha,array_path=str(path),npy_bytes=path.stat().st_size,
               same_pass_mask_mismatch_voxels=0,storage_reproduction_status='passed',metric_status=tc.METRIC_STATUS)
    write(root/'case_manifests'/f'{case["name"]}.json',rec)
    del compact
    with t.stage('strict_validation'):
        import nibabel as nib
        a=tc.check_array(path,rec,case,job['sha256'],c['cache_key'])
        ct=nib.load(rec['ct_path'])
        check(tuple(a.shape[1:])==ct.shape and np.array_equal(np.asarray(rec['affine']),ct.affine),'published geometry')
    after=io_stats()
    result={'rank':c['rank'],'name':case['name'],'dtype':dtype,'timings':t.seconds,
            'elapsed_seconds':time.perf_counter()-start,'bytes':path.stat().st_size,
            'io':{k:after[k]-before[k] for k in after},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write(Path(job['runtime_root'])/mode/'publication_timings'/f'r{c["rank"]}_{case["name"]}.json',result)
    return result

def publish_group(job,c,mode):
    cases=[r['case'] for r in c['selected']]; t=Timers()
    dtype=t.call('dtype_scan',choose_dtype,[tc.stage_paths(c,x['name'])[0] for x in cases])
    write(Path(job['runtime_root'])/mode/f'r{c["rank"]}_dtype.json',{'dtype':dtype,'timings':t.seconds})
    for case in cases:
        publish_one((job,c,case,dtype,mode))


def bounded_map(pool,func,tasks,workers,root,mode):
    pending=list(tasks); active={}; results=[]; peak=0; reductions=0
    while pending or active:
        available=mem_available(); reserved=sum(v for v in active.values())
        while pending and len(active)<workers and capacity(available,reserved,pending[0][1]):
            item,need=pending.pop(0); active[pool.submit(func,item)]=need; reserved+=need
        peak=max(peak,len(active))
        if pending and len(active)<workers: reductions+=1
        check(active or not pending or capacity(mem_available(),0,pending[0][1]),'RAM floor prevents even one case')
        done,_=futures.wait(active,timeout=2,return_when=futures.FIRST_COMPLETED)
        for future in done:
            results.append(future.result()); del active[future]
        write(root/mode/'progress.json',{'completed':len(results),'active':len(active),'pending':len(pending),
              'requested_workers':workers,'peak_active':peak,'memory_limited_checks':reductions,'updated_at_utc':fc.utc_now()})
    return results,{'peak_active':peak,'memory_limited_checks':reductions}

def replay(job):
    source_check(job); root=Path(job['runtime_root']); ctx=mp.get_context('spawn')
    rows=[(c['rank'],r) for c in job['candidates'] for r in c['selected']]
    for workers in job['cpu_workers']:
        mode=f'cpu{workers}'
        if (root/mode/'complete.json').exists():
            continue
        warm=Timers()
        with warm.stage('warm_inputs'):
            for rank,row in rows:
                verify_crop(job,rank,row)  # Reads all bytes identically before every pass.
        start=time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=cpu_init) as pool:
            tasks=[((job,rank,row,mode),row['memory_bytes']) for rank,row in rows]
            restored,limits=bounded_map(pool,cpu_restore,tasks,workers,root,mode)
            restore_wall=time.perf_counter()-start
            scan=Timers(); dtypes={}
            with scan.stage('dtype_scan_wall'):
                for c0 in job['candidates']:
                    c=candidate_for(job,c0['rank'],mode)
                    dtypes[c['rank']]=choose_dtype([tc.stage_paths(c,r['case']['name'])[0] for r in c['selected']])
                    reference=read(root/'reference'/f'r{c["rank"]}_dtype.json')['dtype']
                    check(dtypes[c['rank']]==reference,'dtype replay mismatch')
            pubstart=time.perf_counter()
            tasks=[((job,candidate_for(job,rank,mode),row['case'],dtypes[rank],mode),row['memory_bytes']) for rank,row in rows]
            published,publimits=bounded_map(pool,publish_one,tasks,workers,root,mode)
            pubwall=time.perf_counter()-pubstart
        write(root/mode/'complete.json',{'status':'passed','workers':workers,'cases':16,'warm':warm.seconds,
              'wall_seconds':time.perf_counter()-start,'restore_wall_seconds':restore_wall,
              'publication_wall_seconds':pubwall,**scan.seconds,'limits':limits,'publication_limits':publimits,
              'case_seconds_sum':sum(r['elapsed_seconds'] for r in restored),
              'peak_worker_rss_bytes':max(r['rss_peak_bytes'] for r in restored+published),
              'physical_read_bytes':sum(r['io']['read_bytes'] for r in restored+published),
              'published_bytes':sum(r['bytes'] for r in published),'dtypes':dtypes,'completed_at_utc':fc.utc_now()})
        print(json.dumps(read(root/mode/'complete.json')),flush=True)


def docker_args(job,name,command,gpu=None):
    root=job['runtime_root']; local=job['local_root']
    args=['docker','run','-d','--name',name,'--label','rex.profile_job='+job['sha256'],
          '--network','none','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
          '-w','/workspace','-v',f'{tc.REPO}:/workspace:ro',
          '-v','/data/hengjie:/data/hengjie:ro','-v','/mnt/shengdata1:/mnt/shengdata1:ro',
          '-v',f'{root}:{root}:rw','-v',f'{local}:{local}:rw',
          '--tmpfs','/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m']
    if gpu is not None: args += ['--gpus',f'device={gpu}']
    for env in ['HOME=/tmp','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1',
                'HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home',
                'HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub',
                'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                'OMP_NUM_THREADS='+('4' if gpu is not None else '1'),
                'MKL_NUM_THREADS='+('4' if gpu is not None else '1'),
                'OPENBLAS_NUM_THREADS='+('4' if gpu is not None else '1')]: args+=['-e',env]
    args += [job['container']['image_id'],'python','/workspace/side_experiments/sideexp003_ensemble_method_hub/profile_test300.py',
             command,'--job',str(Path(root)/'job.json')]
    if gpu is not None: args+=['--rank',str(gpu+5)]
    return args

def sample(root,containers):
    data={'at_utc':fc.utc_now(),'epoch':time.time(),'mem_available':mem_available(),
          'cpu_stat':Path('/proc/stat').read_text().splitlines()[0],
          'diskstats':Path('/proc/diskstats').read_text(),
          'pressure':{k:Path('/proc/pressure',k).read_text() for k in ('cpu','io','memory')},
          'shared_free':shutil.disk_usage(root).free,'local_free':shutil.disk_usage('/data').free}
    try:
        data['gpus']=run_cmd(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'])
        processes={}
        for name in containers:
            info=json.loads(run_cmd(['docker','inspect',name]))[0]
            pid=info['State']['Pid']
            if pid:
                processes[name]={}
                # /proc children capture worker RSS and CPU/IO, including replay pool.
                queue=[pid]; seen=set()
                while queue:
                    p=queue.pop()
                    if p in seen: continue
                    seen.add(p)
                    try:
                        base=Path('/proc')/str(p)
                        processes[name][p]={k:(base/k).read_text() for k in ('stat','io','status')}
                        queue.extend(map(int,(base/'task'/str(p)/'children').read_text().split()))
                    except (OSError,ProcessLookupError): pass
        data['processes']=processes
    except Exception as exc: data['sample_warning']=str(exc)
    with (root/'telemetry.jsonl').open('a') as f: f.write(json.dumps(data)+'\n')

def supervise(job,path):
    source_check(job); root=Path(job['runtime_root']); local=Path(job['local_root'])
    root.mkdir(parents=True,exist_ok=True); local.mkdir(parents=True,exist_ok=True)
    with tc.exclusive(root/'launch.lock'):
        check(not (root/'state.json').exists(),'existing benchmark session; inspect before restarting supervisor')
        space_check(shutil.disk_usage(root).free,shutil.disk_usage(local).free,job['space']['shared_need'],job['space']['local_need'])
        write(root/'job.json',job)
        write(root/'state.json',{'status':'starting','pid':os.getpid(),'at_utc':fc.utc_now()})
        names=[]
        def abort(*_): raise KeyboardInterrupt('coordinator interrupted')
        signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGINT,abort)
        try:
            for gpu in range(4):
                name=f'sideexp003_{JOB_ID}_g{gpu}'
                cmd=docker_args(job,name,'gpu',gpu)
                write(root/'commands'/f'{name}.json',{'argv':cmd})
                run_cmd(cmd);names.append(name)
            write(root/'state.json',{'status':'gpu_reference','pid':os.getpid(),'containers':names,'at_utc':fc.utc_now()})
            while True:
                sample(root,names)
                states=[json.loads(run_cmd(['docker','inspect',n]))[0]['State'] for n in names]
                check(not any(not s['Running'] and s['ExitCode']!=0 for s in states),'GPU worker failed')
                if all((root/'gpu'/f'r{r}_smoke.json').exists() for r in range(5,9)):
                    if not (root/'GPU_SMOKE_RELEASE').exists(): write(root/'GPU_SMOKE_RELEASE',{'at_utc':fc.utc_now()})
                if all(not s['Running'] for s in states): break
                time.sleep(5)
            check(all((root/'gpu'/f'r{r}_complete.json').exists() for r in range(5,9)),'incomplete reference')
            name=f'sideexp003_{JOB_ID}_cpu';cmd=docker_args(job,name,'replay')
            write(root/'commands'/f'{name}.json',{'argv':cmd});run_cmd(cmd);names.append(name)
            write(root/'state.json',{'status':'cpu_replay','pid':os.getpid(),'containers':names,'at_utc':fc.utc_now()})
            while True:
                sample(root,[name]); state=json.loads(run_cmd(['docker','inspect',name]))[0]['State']
                if not state['Running']:
                    check(state['ExitCode']==0,'CPU replay failed');break
                time.sleep(5)
            report(job)
            write(root/'state.json',{'status':'complete','at_utc':fc.utc_now(),'production_resume':False})
        except BaseException as exc:
            write(root/'ABORT',{'reason':str(exc)})
            for name in names:
                subprocess.run(['docker','stop','--time','20',name],capture_output=True)
            write(root/'state.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc(),'at_utc':fc.utc_now()})
            raise


def report(job):
    root=Path(job['runtime_root']); comparisons=[read(root/f'cpu{w}'/'complete.json') for w in job['cpu_workers']]
    gpu=[read(p) for p in sorted((root/'gpu').glob('r*_*.nii.gz.json'))]
    check(len(gpu)==16,'incomplete GPU timing records')
    best=min(comparisons,key=lambda r:r['wall_seconds'])
    stages={}
    for r in gpu:
        for k,v in r['timings'].items(): stages[k]=stages.get(k,0)+v
    # Workload-weighted extrapolation per measured model: GPU crop phase by window*prompts,
    # CPU geometry and publication by native output elements. Unsampled architectures remain uncertain.
    per_rank={}
    for c in job['candidates']:
        rank=c['rank']; measured=[r for r in gpu if r['rank']==rank]
        factor=sum(r['workload'] for r in job['full_workloads'][str(rank)])/sum(r['workload'] for r in c['selected'])
        per_rank[rank]=sum(r['timings'].get('crop_prediction_total',0) for r in measured)*factor
    selected_elements=sum(r['output_elements'] for c in job['candidates'] for r in c['selected'])
    full_elements=tc.output_elements(tc.load_cases(job['dataset']['path']))*4
    cpu_wave=best['wall_seconds']*full_elements/selected_elements
    gpu_wave=max(per_rank.values())
    estimated_wave=max(gpu_wave,cpu_wave)
    summary={'status':'passed','gpu_cases':16,'cpu_comparisons':comparisons,'best_workers':best['workers'],
             'gpu_stage_sums_seconds':stages,'full_wave_gpu_model_seconds':per_rank,
             'estimated_cpu_wave_seconds':cpu_wave,'estimated_overlapped_wave_seconds':estimated_wave,
             'remaining_4_waves_hours_range':[estimated_wave*4/3600*.75,estimated_wave*4/3600*1.75],
             'warnings':['Overlap is estimated, not measured end to end.',
                         'Warm CPU replay reads underrepresent cold shared-disk scans.',
                         'Four samples cannot determine the full 300-case float16 fallback.',
                         'Wave 2 models are proxies for architectures in Waves 3–5.',
                         'GPU child timers overlap parent timers; do not sum them.',
                         'Diagnostic crop spooling perturbs storage caching; its time is reported separately.'],
             'production_resume':False,'completed_at_utc':fc.utc_now()}
    write(root/'report.json',summary)
    with (root/'case_timings.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['mode','rank','case','stage','seconds'])
        for mode,records in [('reference',gpu)]+[(f'cpu{w}',[read(p) for folder in ('timings','publication_timings') for p in (root/f'cpu{w}'/folder).glob('*.json')]) for w in job['cpu_workers']]:
            for rec in records:
                for k,v in rec['timings'].items():writer.writerow([mode,rec['rank'],rec['name'],k,v])
    lines=['# Test300 profiling report','', 'Production Waves 2–5 remain held.','',
           '| CPU workers | Wall min | Cases/hour | Peak worker GiB | Physical reads GiB |',
           '|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['workers']} | {r['wall_seconds']/60:.2f} | {16*3600/r['wall_seconds']:.1f} | {r['peak_worker_rss_bytes']/GIB:.2f} | {r['physical_read_bytes']/GIB:.2f} |")
    lines += ['',f"Fastest measured CPU configuration: **{best['workers']} workers**.",
              f"Estimated overlapped wave: {estimated_wave/3600:.2f} h; remaining four-wave range: {summary['remaining_4_waves_hours_range'][0]:.1f}–{summary['remaining_4_waves_hours_range'][1]:.1f} h.",
              '', 'These projections require interpretation against per-stage timings and telemetry.', '', *['- '+x for x in summary['warnings']]]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','gpu','replay','watch','report'])
    parser.add_argument('--job',type=Path,default=DEFAULT_JOB)
    parser.add_argument('--rank',type=int)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.job);return
    job=read(args.job)
    if args.action=='run': supervise(job,args.job)
    elif args.action=='gpu': gpu_worker(job,args.rank)
    elif args.action=='replay': replay(job)
    elif args.action=='report': report(job)
    elif args.action=='watch':
        root=Path(job['runtime_root']);print(json.dumps({'state':read(root/'state.json'),
             'gpu_completed':len(list((root/'gpu').glob('r*_*.nii.gz.json'))),
             'cpu_progress':{f'cpu{w}':read(root/f'cpu{w}'/'progress.json') for w in job['cpu_workers'] if (root/f'cpu{w}'/'progress.json').exists()}},indent=2))

if __name__=='__main__': main()
