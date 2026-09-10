#!/usr/bin/env python3
"""Source-bound test300 continuation with four bounded GPU/CPU pipelines."""
from __future__ import annotations
import os
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import concurrent.futures as futures
import contextlib
import copy
import fcntl
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import numpy as np
import fresh_cache as fc
import test300_cache as tc
import test300_runner as tr
import profile_test300 as prof

HERE = Path(__file__).resolve().parent
JOB_ID = 'j004_test300_r05_r20_cpu4_gpu8'
DEFAULT = HERE / 'cache_jobs' / JOB_ID / 'job_spec.json'
PARENT = HERE / 'cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json'
GIB = 1024**3


def validate(job):
    tc.require(not fc.validate_job(job), str(fc.validate_job(job)))
    tc.require(job['job_id'] == JOB_ID and [c['rank'] for c in job['candidates']] == list(range(5, 21)), 'wrong continuation roster')
    tc.require(job['execution'] == {'cpu_workers': 4, 'pending_per_gpu': 2, 'numpy_madvise_hugepage': 0, 'cpu_threads': 1}, 'execution drift')
    tc.require(job['dataset']['cases'] == 300 and job['dataset']['findings'] == 582, 'cohort drift')
    tc.require(fc.sha256_file(tc.resolve(job['dataset']['path'])) == job['dataset']['sha256'], 'dataset drift')
    tc.require(fc.sha256_file(PARENT) == job['parent_file_sha256'], 'parent drift')
    parent = fc.read_json(PARENT)
    for entry in job['source_bundle']['files']:
        tc.require(fc.sha256_file(tc.REPO / entry['path']) == entry['sha256'], 'source drift: ' + entry['path'])
    tc.require(Path(job['runtime_root']) == fc.RUNTIME_ROOT / 'cache/jobs' / JOB_ID, 'unsafe runtime')
    tc.require(Path(job['staging_root']) == tc.STAGING_ROOT / JOB_ID, 'unsafe staging')
    for c, old in zip(job['candidates'], parent['candidates'][4:]):
        for key in ('id', 'rank', 'wave', 'gpu', 'checkpoint', 'config', 'plans', 'model_spec', 'cache', 'test_preprocessing_cases', 'inference_contract', 'paired_val200'):
            tc.require(c[key] == old[key], 'candidate drift: ' + key)
        tc.require(Path(c['staging_root']) == Path(job['staging_root']) / c['cache_key'], 'unsafe case staging')
        tc.require(Path(c['cache_root']) == fc.RUNTIME_ROOT / 'cache/logits/by_cache_key' / c['cache_key'], 'unsafe publication')
        tc.require(Path(c['progress_path']) == Path(job['runtime_root']) / 'progress' / (c['id'] + '.json'), 'unsafe progress')


def prepare():
    tc.require(not DEFAULT.exists(), 'job already frozen')
    parent = fc.read_json(PARENT)
    tc.require(not fc.validate_job(parent), 'invalid parent content hash')
    tc.require(parent['job_spec_sha256'] == '50ac85fb91a37a748be45e82e27d4ed09862645cdb06258b07f67fdd5bf8e912', 'unexpected parent')
    tc.require(fc.read_json(Path(parent['runtime_root']) / 'finish_wave1_state.json')['status'] == 'stopped_after_wave1', 'parent not held')
    for c in parent['candidates'][:4]:
        tc.require(tr.completed(parent, c), 'parent Wave 1 not complete')
    job = copy.deepcopy(parent)
    entries = tc.source_bundle()['files']
    for name in ('profile_test300.py', 'continue_test300.py', 'continue_test300_spec.md'):
        p = HERE / name
        entries.append({'path': str(p.relative_to(tc.REPO)), 'sha256': fc.sha256_file(p)})
    entries.sort(key=lambda e: e['path'])
    bundle = {'files': entries, 'sha256': fc.json_sha256(entries)}
    job.update(job_id=JOB_ID, runtime_root=str(fc.RUNTIME_ROOT / 'cache/jobs' / JOB_ID),
               staging_root=str(tc.STAGING_ROOT / JOB_ID), source_bundle=bundle,
               parent_file_sha256=fc.sha256_file(PARENT),
               execution={'cpu_workers': 4, 'pending_per_gpu': 2, 'numpy_madvise_hugepage': 0, 'cpu_threads': 1},
               created_at_utc=fc.utc_now(),
               continuation={'rank_start': 5, 'wave_start': 2,
                 'parent_job_spec_sha256': parent['job_spec_sha256'],
                 'parent_source_bundle_sha256': parent['source_bundle']['sha256'],
                 'source_drift_audit': {'new_source_bundle_sha256': bundle['sha256'],
                    'reason': 'Authorized four CPU workers and process-local allocation fix; frozen numerical functions retained'}})
    job['candidates'] = job['candidates'][4:]
    job['parent_wave1'] = [{'rank': c['rank'], 'cache_key': c['cache_key'], 'cache_root': c['cache_root'],
                           'manifest_sha256': fc.sha256_file(Path(c['cache_root']) / 'export_manifest.json')} for c in parent['candidates'][:4]]
    for c in job['candidates']:
        old = c['cache_key']
        key = 'v4_' + fc.json_sha256({'parent_key': old, 'source': bundle['sha256'], 'execution': job['execution'], 'job': JOB_ID})
        c.update(parent_cache_key=old, cache_key=key, cache_version_id='sideexp003_' + key,
                 cache_root=str(fc.RUNTIME_ROOT / 'cache/logits/by_cache_key' / key),
                 staging_root=str(Path(job['staging_root']) / key),
                 progress_path=str(Path(job['runtime_root']) / 'progress' / (c['id'] + '.json')))
    job.pop('job_spec_sha256', None)
    job['job_spec_sha256'] = fc.json_sha256(job)
    validate(job)
    fc.atomic_write_json(DEFAULT, job)
    print(json.dumps({'job': str(DEFAULT), 'sha256': job['job_spec_sha256']}))


def aborted(job):
    tc.require(not (Path(job['runtime_root']) / 'control/ABORT').exists(), 'job aborted')


@contextlib.contextmanager
def memory_slot(job, needed):
    root = Path(job['runtime_root'])
    lock = root / 'memory.lock'
    ledger = root / 'memory_reservations.json'
    while True:
        aborted(job)
        with lock.open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            current = fc.read_json(ledger) if ledger.exists() else {}
            # Reservations include allocated memory, making admission conservative.
            if prof.mem_available() >= 64 * GIB + needed + sum(current.values()):
                token = str(os.getpid()) + '_' + str(time.time_ns())
                current[token] = needed
                fc.atomic_write_json(ledger, current)
                break
        time.sleep(2)
    try:
        yield
    finally:
        with lock.open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            current = fc.read_json(ledger)
            current.pop(token, None)
            fc.atomic_write_json(ledger, current)


def cpu_init():
    prof.cpu_init()
    tc.require(not np._core.multiarray._get_madvise_hugepage(), 'hugepage advice still enabled')


def crop_paths(c, case):
    root = Path(c['staging_root']) / 'handoff'
    return root / (case['name'] + '.npy'), root / (case['name'] + '.json')


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
        out = prof.save_stage({'local_root': job['staging_root'], 'sha256': job['job_spec_sha256']}, c, case, arr, details, timer)
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


def finalize(job, c, cases):
    tc.inference_imports()
    started = time.monotonic()
    with memory_slot(job, max(math.prod(case['shape']) * len(case['findings']) for case in cases) * 12 + 2 * GIB):
        for case in cases:
            tc.require(tc.load_stage(c, case, job['job_spec_sha256']) is not None, 'missing stage before publication')
        begin = time.monotonic()
        dtype, _ = tc.publish(job, c, cases)
        publish_seconds = time.monotonic() - begin
        validation, published = tc.validate_published(job, c, cases)
    root = Path(c['cache_root'])
    fc.atomic_write_json(root / 'reproduction_validation.json', validation)
    fc.atomic_write_json(root / 'export_manifest.json', {
        'job_id': job['job_id'], 'job_spec_sha256': job['job_spec_sha256'],
        'candidate_id': c['id'], 'cache_key': c['cache_key'], 'candidate_source': c,
        'dataset': job['dataset'], 'paired_val200': c['paired_val200'],
        'status': 'strict_passed', 'metric_status': tc.METRIC_STATUS,
        'cases': published, 'case_count': 300, 'finding_count': 582,
        'publication_seconds': publish_seconds, 'finalization_seconds': time.monotonic() - started,
        'completed_at_utc': fc.utc_now()})
    tc.cleanup_stage(Path(c['staging_root']))
    return validation


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
                update(status='finalizing')
                result = pool.submit(finalize, job, c, cases).result()
                update(status='strict_passed', dtype=result['dtype'], bytes=result['array_bytes'], eta_seconds=0)
        except BaseException as exc:
            update(status='failed', error=str(exc))
            raise


def docker_command(job, c):
    name = f"sideexp003_{JOB_ID}_w{c['wave']:02d}_g{c['gpu']}"
    args = ['docker', 'run', '-d', '--name', name, '--label', 'rex.test_job=' + job['job_spec_sha256'],
            '--gpus', f"device={c['gpu']}", '--ipc=host', '--shm-size=32g', '--user', f'{os.getuid()}:{os.getgid()}', '-w', '/workspace']
    for src, dst, mode in [(tc.REPO, '/workspace', 'ro'), ('/data/hengjie', '/data/hengjie', 'ro'),
                           ('/mnt/shengdata1', '/mnt/shengdata1', 'ro'),
                           (c['cache_root'], c['cache_root'], 'rw'), (c['staging_root'], c['staging_root'], 'rw'),
                           (job['runtime_root'], job['runtime_root'], 'rw')]:
        args += ['-v', f'{src}:{dst}:{mode}']
    args += ['--tmpfs', '/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m']
    for key, value in {'NUMPY_MADVISE_HUGEPAGE': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'HOME': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                       'HF_HOME': '/data/hengjie/datasets/rexgroundingct/.hf_home',
                       'HF_HUB_CACHE': '/data/hengjie/datasets/rexgroundingct/.hf_home/hub',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
        args += ['-e', f'{key}={value}']
    args += [job['container']['image_id'], 'python', '/workspace/' + str(Path(__file__).relative_to(tc.REPO)), 'worker', '--rank', str(c['rank'])]
    return name, args


def snapshot(job, **kw):
    root = Path(job['runtime_root'])
    value = {'job_id': JOB_ID, 'job_spec_sha256': job['job_spec_sha256'], 'updated_at_utc': fc.utc_now(),
             'host': socket.gethostname(), 'pid': os.getpid(), 'candidates': tr.progress(job),
             'gpu_free_mib': tr.gpu_free(), 'available_ram_bytes': prof.mem_available(),
             'shared_free_bytes': shutil.disk_usage(root).free, 'local_free_bytes': shutil.disk_usage(job['staging_root']).free, **kw}
    fc.atomic_write_json(root / 'state.json', value)
    with (root / 'monitor.jsonl').open('a') as f:
        f.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def supervise(job):
    validate(job)
    tc.require(socket.gethostname() == 'shenggpu8', 'wrong host')
    tc.require(tr.command(['git', 'branch', '--show-current']).stdout.strip() == 'gpu8', 'wrong branch')
    root = Path(job['runtime_root'])
    root.mkdir(parents=True, exist_ok=True)
    Path(job['staging_root']).mkdir(parents=True, exist_ok=True)
    parent = fc.read_json(PARENT)
    active = []
    def stopped(signum, frame):
        raise RuntimeError('coordinator interrupted')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    with tc.exclusive(root / 'launch.lock'), tc.exclusive(Path(parent['runtime_root']) / 'launch.lock'):
        try:
            aborted(job)
            tc.require(fc.read_json(Path(parent['runtime_root']) / 'finish_wave1_state.json')['status'] == 'stopped_after_wave1', 'parent not held')
            # Do not clear reservation records until every previous owned worker is dead.
            for c in job['candidates']:
                name, _ = docker_command(job, c)
                d = tr.docker_state(name)
                tc.require(not d or not d['State']['Running'], 'existing live owner: ' + name)
            fc.atomic_write_json(root / 'memory_reservations.json', {})
            for wave in range(2, 6):
                members = [c for c in job['candidates'] if c['wave'] == wave]
                todo = [c for c in members if not tr.completed(job, c, verify_hashes=True)]
                if not todo:
                    continue
                while True:
                    validate(job)
                    aborted(job)
                    free = tr.gpu_free()
                    space = tc.wave_space_ok(shutil.disk_usage(root).free, shutil.disk_usage(job['staging_root']).free - 128 * GIB,
                                             len(todo), job['output_elements_per_model'])
                    if space and prof.mem_available() >= 128 * GIB and all(free[c['gpu']] >= 20000 for c in todo):
                        break
                    snapshot(job, status='waiting_for_resources', wave=wave, space_ok=space)
                    time.sleep(30)
                gate = root / 'control' / f'wave_{wave:02d}.continue'
                gate.unlink(missing_ok=True)
                for c in todo:
                    for directory in (c['cache_root'], c['staging_root']):
                        Path(directory).mkdir(parents=True, exist_ok=True)
                    name, args = docker_command(job, c)
                    d = tr.docker_state(name)
                    if d:
                        tc.require(d['Config']['Labels'].get('rex.test_job') == job['job_spec_sha256'] and not d['State']['Running'], 'foreign/live container')
                        fc.atomic_write_text(root / 'logs' / (name + '.previous.log'), tr.command(['docker', 'logs', name], check=False).stdout)
                        tr.command(['docker', 'rm', name])
                    Path(c['progress_path']).unlink(missing_ok=True)
                    fc.atomic_write_json(root / 'commands' / (c['id'] + '.json'), {'argv': args, 'at_utc': fc.utc_now()})
                    tr.command(args)
                    active.append(name)
                begin = time.monotonic()
                released = False
                while True:
                    aborted(job)
                    records = tr.progress(job)
                    current = [records.get(c['id'], {}) for c in todo]
                    tc.require(not any(r.get('status') == 'failed' for r in current), 'worker failed: ' + str(current))
                    for name in active:
                        d = tr.docker_state(name)
                        tc.require(d and (d['State']['Running'] or d['State']['ExitCode'] == 0), 'worker exited: ' + name)
                    if not released and all(r.get('status') in ('smoke_waiting', 'strict_passed') for r in current):
                        tc.require(all(r.get('free_mib_after_smoke', 4096) >= 4096 for r in current), 'smoke memory failed')
                        fc.atomic_write_json(gate, {'job_spec_sha256': job['job_spec_sha256'], 'at_utc': fc.utc_now()})
                        released = True
                    snapshot(job, status='exporting' if released else 'smoke_running', wave=wave, wave_elapsed_seconds=time.monotonic() - begin)
                    if all(r.get('status') == 'strict_passed' for r in current) and all(not tr.docker_state(n)['State']['Running'] for n in active):
                        break
                    time.sleep(30)
                fc.atomic_write_json(root / f'wave_{wave:02d}_timing.json', {'wave': wave, 'wall_seconds': time.monotonic() - begin, 'at_utc': fc.utc_now()})
                for name in active:
                    log = tr.command(['docker', 'logs', name], check=False)
                    fc.atomic_write_text(root / 'logs' / (name + '.log'), log.stdout + log.stderr)
                active.clear()
                inventory = {'parent_wave1': job['parent_wave1'], 'candidates': tr.progress(job), 'updated_at_utc': fc.utc_now()}
                fc.atomic_write_json(root / 'inventory.json', inventory)
            snapshot(job, status='complete', completed_models=16, parent_completed_models=4)
        except BaseException as exc:
            for name in active:
                tr.command(['docker', 'stop', '--time', '30', name], check=False)
            snapshot(job, status='failed', error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker', 'watch'])
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
        return
    job = fc.read_json(DEFAULT)
    if args.action == 'run':
        supervise(job)
    elif args.action == 'worker':
        worker(job, next(c for c in job['candidates'] if c['rank'] == args.rank))
    else:
        print(json.dumps(fc.read_json(Path(job['runtime_root']) / 'state.json'), indent=2))


if __name__ == '__main__':
    main()
