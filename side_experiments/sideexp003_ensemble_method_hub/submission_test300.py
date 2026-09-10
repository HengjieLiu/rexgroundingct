#!/usr/bin/env python3
"""Frozen, label-free Wave 1 probability ensemble and d1/d2/d3 packaging."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import multiprocessing
import signal
import socket
import sys
import time
import zipfile
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import fresh_cache as fc
import test300_cache as tc
from preliminary_ensemble import sigmoid_float32

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts/rexgroundingct'))
import run_024_test_inference_anatomy as anatomy_policy

JOB_ID = 's001_top4_test300_d123'
PARENT = HERE / 'cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json'
OUTPUT = anatomy_policy.EXP_ROOT / 'outputs'
CHUNK = 4_194_304
VARIANTS = ('d1', 'd2', 'd3')
require = tc.require
read = fc.read_json
write = fc.atomic_write_json
sha = fc.sha256_file


def identity(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha(p)}


def verify_identity(record):
    require(sha(Path(record['path'])) == record['sha256'],
            'source hash mismatch: ' + record['path'])


def freeze(path):
    """One-time preparation; does not require completed logits or create outputs."""
    require(not path.exists(), 'job already frozen')
    parent = read(PARENT)
    cases = tc.load_cases(Path(parent['dataset']['path']))
    route_path = anatomy_policy.EXP_ROOT / 'config/prompt_routing_test.json'
    audit_path = anatomy_policy.SOURCE_TEST / 'reports/private/audit_results.json'
    audit = read(audit_path)
    by_name = {r['volume_name']: r for r in audit['cases']}
    require(set(by_name) == {c['name'] for c in cases}, 'anatomy cohort mismatch')
    rows = []
    for case in cases:
        record = by_name[case['name']]
        require(record['integrity']['status'] == 'VALID'
                and record['geometry']['index_grid_status'] == 'PASS', 'unaudited anatomy')
        mask = {'path': record['integrity']['path'],
                'sha256': record['integrity']['observed_sha256']}
        verify_identity(mask)
        ct_path = record['source_provenance']['path']
        ct = nib.load(ct_path)
        mask_img = nib.load(mask['path'])
        require(list(ct.shape) == case['shape'] == list(mask_img.shape)
                and np.allclose(ct.affine, mask_img.affine, atol=1e-5, rtol=0)
                and np.array_equal(ct.affine, record['geometry']['ct_affine']),
                'audited geometry mismatch')
        rows.append({**case, **tc.prompt_contract(case), 'ct_path': ct_path,
                     'ct_header_sha256': hashlib.sha256(ct.header.binaryblock).hexdigest(),
                     'affine': ct.affine.tolist(), 'anatomy': mask})
    routes = read(route_path)
    require(routes['metadata_sha256'] == parent['dataset']['sha256'], 'routing dataset drift')
    route_map = {(r['case'], r['finding_id']): r for r in routes['rows']}
    require(len(route_map) == len(routes['rows']) == 582, 'routing count')
    for c in rows:
        c['routes'] = [route_map[(c['name'], i)] for i in c['finding_indices']]
        require(all(r['prompt'] == c['findings'][str(i)]
                    for i, r in enumerate(c['routes'])), 'route prompt drift')
    source_paths = [Path(__file__), HERE / 'submission_test300_spec.md',
                    HERE / 'preliminary_ensemble.py', HERE / 'test300_cache.py',
                    HERE / 'fresh_cache.py', Path(anatomy_policy.__file__)]
    report = REPO / 'experiments/023_ct_rate_ts_total_rex_test300_audit/report.md'
    require('PASS_PENDING_MANUAL_VISUAL_REVIEW' in report.read_text(), 'review status drift')
    candidates = [{k: c[k] for k in ('rank', 'id', 'cache_key', 'cache_root',
                                   'checkpoint', 'paired_val200')}
                  for c in parent['candidates'][:4]]
    require([c['rank'] for c in candidates] == [1, 2, 3, 4], 'rank order')
    job = {'schema_version': 1, 'job_id': JOB_ID, 'created_at_utc': fc.utc_now(),
           'parent': identity(PARENT), 'parent_job_sha256': parent['job_spec_sha256'],
           'parent_runtime': parent['runtime_root'], 'dataset': parent['dataset'],
           'container': parent['container'], 'candidates': candidates, 'cases': rows,
           'routing': identity(route_path), 'audit': identity(audit_path),
           'anatomy_review': {'status': 'PASS_PENDING_MANUAL_VISUAL_REVIEW',
                              'source': identity(report)},
           'source_files': [identity(p) for p in source_paths],
           'output_root': str(OUTPUT),
           'runtime_root': str(fc.RUNTIME_ROOT / 'submission_jobs' / JOB_ID),
           'workers': 4, 'chunk_elements': CHUNK, 'poll_seconds': 30,
           'minimum_shared_free_bytes': 20 * 1024**4,
           'headroom_bytes': 350 * 1024**3,
           'method': {'domain': 'post_sigmoid_probability', 'weights': [0.25] * 4,
                      'threshold_sum': 2.0, 'd2': 'eligible_whole_lung_20mm',
                      'd3': 'eligible_prompt_selected_20mm', 'base_for_d2_d3': 'd1'},
           'smoke_cases': list(dict.fromkeys([
               max(rows, key=lambda c: math.prod(c['shape']))['name'],
               max(rows, key=lambda c: math.prod(c['shape']) * len(c['findings']))['name']]))}
    job['job_sha256'] = fc.json_sha256(job)
    write(path, job)
    print(json.dumps({'job': str(path), 'job_sha256': job['job_sha256'],
                      'cases': len(rows), 'smoke_cases': job['smoke_cases']}), flush=True)


def validate_job(job):
    require(fc.json_sha256({k: v for k, v in job.items() if k != 'job_sha256'})
            == job['job_sha256'], 'job hash drift')
    require(job['job_id'] == JOB_ID and job['workers'] == 4
            and job['chunk_elements'] == CHUNK, 'execution contract drift')
    for rec in [job['parent'], job['routing'], job['audit'],
                job['anatomy_review']['source'], *job['source_files']]:
        verify_identity(rec)
    verify_identity(job['dataset'])
    parent = read(Path(job['parent']['path']))
    require(parent['job_spec_sha256'] == job['parent_job_sha256'], 'parent identity drift')
    require([c['rank'] for c in job['candidates']] == [1, 2, 3, 4], 'rank drift')
    for source, selected in zip(parent['candidates'], job['candidates']):
        require(all(source[k] == v for k, v in selected.items()), 'checkpoint binding drift')
    cases = tc.load_cases(Path(job['dataset']['path']))
    require(len(job['cases']) == 300, 'case count drift')
    for source, frozen in zip(cases, job['cases']):
        require(all(frozen[k] == source[k] for k in ('name', 'shape', 'findings')),
                'frozen test case drift')


def gate(job):
    root = Path(job['parent_runtime'])
    require(not (root / 'finish_wave1_error.json').exists(), 'Wave 1 guard failed')
    p = root / 'finish_wave1_state.json'
    if not p.exists():
        return False, 'waiting for finish guard'
    state = read(p)
    require(state['job_spec_sha256'] == job['parent_job_sha256'], 'foreign guard')
    require(state['status'] in ('finishing_wave1_only', 'stopped_after_wave1'),
            'unexpected guard status')
    if state['status'] != 'stopped_after_wave1':
        return False, 'Wave 1 exporting/publishing/validating'
    require(state.get('coordinator_exited') is True, 'coordinator exit unconfirmed')
    request = read(root / 'control/finish_wave1_request.json')
    proc = Path('/proc') / str(state['coordinator_pid']) / 'stat'
    if proc.exists():
        fields = proc.read_text().rsplit(')', 1)[1].split()
        require(fields[19] != request['coordinator']['start_ticks'] or fields[0] == 'Z',
                'original coordinator remains live')
    require(state['held_ranks'] == list(range(5, 21)), 'later-wave hold drift')
    return True, 'Wave 1 stopped and coordinator exited'


def bind_caches(job):
    """Bind final manifests exactly once, then reject changed publications."""
    root = Path(job['runtime_root'])
    bound = []
    for c in job['candidates']:
        cache = Path(c['cache_root'])
        m = read(cache / 'export_manifest.json')
        v = read(cache / 'reproduction_validation.json')
        require(m['status'] == 'strict_passed'
                and m['job_spec_sha256'] == job['parent_job_sha256']
                and m['cache_key'] == c['cache_key'] and m['candidate_id'] == c['id']
                and m['dataset'] == job['dataset']
                and m['candidate_source']['checkpoint'] == c['checkpoint'], 'foreign publication')
        require(v['status'] == 'passed' and v['cases'] == 300 and v['findings'] == 582
                and v['job_spec_sha256'] == job['parent_job_sha256']
                and v['array_hashes_verified'] is True
                and v['same_pass_mask_mismatch_voxels'] == 0, 'invalid cache validation')
        records = {r['name']: r for r in m['cases']}
        require(len(m['cases']) == len(records) == 300
                and set(records) == {r['name'] for r in job['cases']}, 'cache case set')
        for case in job['cases']:
            r = records[case['name']]
            expected = cache / 'cases' / (case['name'] + '.npy')
            require(r['array_path'] == str(expected) and expected.is_file()
                    and expected.stat().st_size == r['npy_bytes'], 'array path/size drift')
            require(r['finding_indices'] == case['finding_indices']
                    and r['prompt_sha256s'] == case['prompt_sha256s']
                    and r['shape'] == [len(case['findings']), *case['shape']]
                    and r['affine'] == case['affine']
                    and r['cache_key'] == c['cache_key']
                    and r['job_spec_sha256'] == job['parent_job_sha256']
                    and r['storage_reproduction_status'] == 'passed'
                    and r['same_pass_mask_mismatch_voxels'] == 0, 'case provenance drift')
        bound.append({'rank': c['rank'], 'cache_key': c['cache_key'],
                      'manifest': identity(cache / 'export_manifest.json'),
                      'validation': identity(cache / 'reproduction_validation.json')})
    value = {'job_sha256': job['job_sha256'], 'caches': bound}
    p = root / 'input_binding.json'
    if p.exists():
        require(read(p) == value, 'completed cache manifest changed since binding')
    else:
        write(p, value)
    return [{r['name']: r for r in read(Path(b['manifest']['path']))['cases']} for b in bound]


def average_arrays(arrays, chunk=CHUNK, expected_hashes=None):
    require(len(arrays) == 4, 'requires four models')
    require(all(a.shape == arrays[0].shape and a.dtype in (np.float16, np.float32)
                and a.flags.c_contiguous for a in arrays), 'logit dtype/layout mismatch')
    out = np.empty(arrays[0].shape, dtype=np.uint8)
    hashes = [hashlib.sha256() for _ in arrays]
    timings = {'reading_seconds': 0.0, 'averaging_seconds': 0.0, 'hashing_seconds': 0.0}
    for start in range(0, out.size, chunk):
        end = min(out.size, start + chunk)
        acc = np.zeros(end - start, dtype=np.float32)
        for i, a in enumerate(arrays):
            t = time.monotonic()
            # Materialize the small slice so page faults belong to reading time.
            values = np.array(a.reshape(-1)[start:end], copy=True)
            timings['reading_seconds'] += time.monotonic() - t
            t = time.monotonic()
            hashes[i].update(memoryview(values).cast('B'))
            timings['hashing_seconds'] += time.monotonic() - t
            t = time.monotonic()
            require(np.isfinite(values).all() and (np.abs(values) <= 30).all(), 'invalid logits')
            acc += sigmoid_float32(values)
            timings['averaging_seconds'] += time.monotonic() - t
        out.reshape(-1)[start:end] = acc >= np.float32(2.0)
    observed = [h.hexdigest() for h in hashes]
    if expected_hashes is not None:
        require(observed == expected_hashes, 'source array hash mismatch')
    return out, timings


def postprocess(raw, anatomy, affine, routes):
    outputs = [raw.copy(), raw.copy()]
    supports = {}
    for i, route in enumerate(routes):
        for out, eligible, labels in (
            (outputs[0], route.get('eligible'), tuple(anatomy_policy.LABELS['W'])),
            (outputs[1], route.get('fine_eligible'), tuple(route.get('selected_labels', []))),
        ):
            if eligible and labels:
                if labels not in supports:
                    supports[labels] = anatomy_policy.physical_dilation(
                        np.isin(anatomy, labels), affine, 20.0)
                out[i] &= supports[labels]
    return outputs


def validate_prediction(path, case, expected=None):
    img = nib.load(str(path))
    require(img.shape == (len(case['findings']), *case['shape'])
            and np.allclose(img.affine, case['affine'], atol=1e-5, rtol=0)
            and img.get_data_dtype() == np.dtype('uint8'), 'prediction geometry/dtype')
    data = np.asanyarray(img.dataobj)
    require(np.isin(data, [0, 1]).all(), 'prediction is not binary')
    if expected is not None:
        require(np.array_equal(data, expected), 'NIfTI roundtrip changed mask')
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'shape': list(img.shape), 'affine': img.affine.tolist(), 'dtype': 'uint8'}


def publish_transaction(tx_path, tx):
    """Journal is durable before the first rename; replay safely after interruption."""
    for item in tx['outputs']:
        dst, tmp = Path(item['path']), Path(item['temporary_path'])
        if dst.exists():
            require(sha(dst) == item['sha256'], 'conflicting published output: ' + str(dst))
        else:
            require(tmp.is_file() and sha(tmp) == item['sha256'], 'missing/corrupt transaction temporary')
            os.replace(tmp, dst)
    tx['status'] = 'complete'
    tx['completed_at_utc'] = fc.utc_now()
    write(tx_path, tx)
    return tx


def process_case(job, case, records):
    start = time.monotonic()
    root = Path(job['runtime_root'])
    tx_path = root / 'cases' / (case['name'] + '.json')
    if tx_path.exists():
        tx = read(tx_path)
        require(tx['job_sha256'] == job['job_sha256']
                and tx['name'] == case['name']
                and tx['finding_indices'] == case['finding_indices']
                and tx['prompt_sha256s'] == case['prompt_sha256s']
                and tx['input_array_hashes'] == [r['array_sha256'] for r in records]
                and [r['path'] for r in tx['outputs']] == [
                    str(Path(job['output_root']) / v / case['name']) for v in VARIANTS],
                'foreign case transaction')
        return publish_transaction(tx_path, tx)
    for variant in VARIANTS:
        require(not (Path(job['output_root']) / variant / case['name']).exists(),
                'unowned existing case output')
    arrays = [np.load(r['array_path'], mmap_mode='r', allow_pickle=False) for r in records]
    require(all(list(a.shape) == r['shape'] and str(a.dtype) == r['dtype']
                for a, r in zip(arrays, records)), 'array header drift')
    raw, timings = average_arrays(arrays, job['chunk_elements'], [r['array_sha256'] for r in records])
    del arrays
    t = time.monotonic()
    verify_identity(case['anatomy'])
    ct = nib.load(case['ct_path'])
    require(hashlib.sha256(ct.header.binaryblock).hexdigest() == case['ct_header_sha256']
            and list(ct.shape) == case['shape'] and np.array_equal(ct.affine, case['affine']),
            'CT header drift')
    anat = nib.load(case['anatomy']['path'])
    require(anat.shape == ct.shape and np.allclose(anat.affine, ct.affine, atol=1e-5, rtol=0),
            'anatomy geometry drift')
    outputs = [raw, *postprocess(raw, np.asanyarray(anat.dataobj), ct.affine, case['routes'])]
    timings['anatomy_seconds'] = time.monotonic() - t
    timings['writing_seconds'] = 0.0
    timings['validation_seconds'] = 0.0
    items = []
    for variant, values in zip(VARIANTS, outputs):
        dst = Path(job['output_root']) / variant / case['name']
        tmp = dst.with_name('.' + job['job_id'] + '.' + case['name'])
        t = time.monotonic()
        hdr = ct.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(values, ct.affine, hdr), str(tmp))
        timings['writing_seconds'] += time.monotonic() - t
        t = time.monotonic()
        item = validate_prediction(tmp, case, values)
        item.update(path=str(dst), temporary_path=str(tmp), variant=variant)
        items.append(item)
        timings['validation_seconds'] += time.monotonic() - t
    tx = {'job_sha256': job['job_sha256'], 'status': 'prepared', 'name': case['name'],
          'outputs': items, 'finding_indices': case['finding_indices'],
          'prompt_sha256s': case['prompt_sha256s'], 'timings': timings,
          'elements': raw.size, 'elapsed_seconds': time.monotonic() - start,
          'input_array_hashes': [r['array_sha256'] for r in records]}
    write(tx_path, tx)
    return publish_transaction(tx_path, tx)


def package(root, variant, records, runtime, job_sha):
    dst = root / (variant + '.zip')
    tx_path = runtime / (variant + '.zip.json')
    if tx_path.exists():
        tx = read(tx_path)
        require(tx['job_sha256'] == job_sha, 'foreign ZIP transaction')
        return publish_transaction(tx_path, tx)['outputs'][0]
    require(not dst.exists(), 'unowned existing ZIP')
    tmp = root / ('.' + JOB_ID + '.' + variant + '.zip')
    expected = {Path(r['path']).name: r for r in records}
    require(len(expected) == len(records), 'duplicate ZIP filename')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, r in sorted(expected.items()):
            require(sha(Path(r['path'])) == r['sha256'], 'output hash changed before ZIP')
            z.write(r['path'], arcname=name)
    with zipfile.ZipFile(tmp) as z:
        require(z.namelist() == sorted(expected), 'ZIP root file set mismatch')
        require(z.testzip() is None, 'ZIP CRC failure')
        for name in z.namelist():
            h = hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    h.update(block)
            require(h.hexdigest() == expected[name]['sha256'], 'ZIP member hash mismatch')
            require(z.getinfo(name).compress_type == zipfile.ZIP_STORED, 'ZIP compression drift')
    item = {'path': str(dst), 'temporary_path': str(tmp), 'sha256': sha(tmp),
            'bytes': tmp.stat().st_size, 'files': len(records)}
    tx = {'job_sha256': job_sha, 'status': 'prepared', 'outputs': [item]}
    write(tx_path, tx)
    return publish_transaction(tx_path, tx)['outputs'][0]


def enough_space(job):
    return shutil.disk_usage(job['output_root']).free >= (
        job['minimum_shared_free_bytes'] + job['headroom_bytes'])


def run(job):
    validate_job(job)
    root = Path(job['runtime_root'])
    root.mkdir(parents=True, exist_ok=True)
    with tc.exclusive(root / 'launch.lock'):
        owner_path = root / 'owner.json'
        if owner_path.exists():
            require(read(owner_path)['job_sha256'] == job['job_sha256'], 'foreign job owner')
        else:
            write(owner_path, {'job_sha256': job['job_sha256'], 'job_id': job['job_id']})
        def status(state, **details):
            value = {'job_id': job['job_id'], 'job_sha256': job['job_sha256'],
                     'pid': os.getpid(), 'host': socket.gethostname(), 'status': state,
                     'updated_at_utc': fc.utc_now(), **details}
            write(root / 'state.json', value)
            print(json.dumps(value), flush=True)
        try:
            while True:
                validate_job(job)
                ready, reason = gate(job)
                if ready:
                    break
                status('waiting_for_wave1', reason=reason)
                time.sleep(job['poll_seconds'])
            records = bind_caches(job)
            while not enough_space(job):
                status('waiting_for_space', free_bytes=shutil.disk_usage(job['output_root']).free)
                time.sleep(job['poll_seconds'])
                validate_job(job)
                require(gate(job)[0], 'dependency gate closed')
            out = Path(job['output_root'])
            for variant in VARIANTS:
                folder = out / variant
                folder.mkdir(exist_ok=True)
                for p in folder.iterdir():
                    if p.name.startswith('.' + job['job_id'] + '.'):
                        continue  # This job's unpublished scratch may be overwritten.
                    require(p.name in {c['name'] for c in job['cases']}
                            and (root / 'cases' / (p.name + '.json')).exists(),
                            'unowned output: ' + str(p))
            started = time.monotonic()
            finished = {}
            total_elements = sum(math.prod(c['shape']) * len(c['findings']) for c in job['cases'])
            def progress(phase):
                done = sum(v['elements'] for v in finished.values())
                elapsed = time.monotonic() - started
                status(phase, cases=len(finished), total_cases=300,
                       findings=sum(len(v['finding_indices']) for v in finished.values()),
                       elapsed_seconds=elapsed,
                       eta_seconds=elapsed / done * (total_elements - done)
                       if len(finished) >= 10 else None,
                       shared_free_bytes=shutil.disk_usage(out).free)
            with futures.ProcessPoolExecutor(max_workers=4,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                for phase, cases in (
                    ('smoke', [c for c in job['cases'] if c['name'] in job['smoke_cases']]),
                    ('processing', [c for c in job['cases'] if c['name'] not in job['smoke_cases']]),
                ):
                    queue = iter(cases)
                    pending = {}
                    def enqueue():
                        c = next(queue, None)
                        if c is not None:
                            pending[pool.submit(process_case, job, c,
                                    [r[c['name']] for r in records])] = c['name']
                    for _ in range(4):
                        enqueue()
                    progress(phase)
                    last = time.monotonic()
                    while pending:
                        done, _ = futures.wait(pending, timeout=30, return_when=futures.FIRST_COMPLETED)
                        for f in done:
                            finished[pending.pop(f)] = f.result()
                        for _ in done:
                            enqueue()
                        if time.monotonic() - last >= 60 or not pending:
                            progress(phase)
                            last = time.monotonic()
                    if phase == 'smoke':
                        write(root / 'smoke_validation.json', {'job_sha256': job['job_sha256'],
                              'status': 'passed', 'cases': list(finished)})
            status('packaging', cases=300)
            validate_job(job)
            bind_caches(job)
            inventories = {}
            archives = {}
            for variant in VARIANTS:
                rows = [next(r for r in finished[c['name']]['outputs'] if r['variant'] == variant)
                        for c in job['cases']]
                require({p.name for p in (out / variant).iterdir()}
                        == {c['name'] for c in job['cases']}, 'final directory file set drift')
                require(len(rows) == 300 and sum(r['shape'][0] for r in rows) == 582,
                        'final cohort mismatch')
                inventories[variant] = {'cases': 300, 'findings': 582, 'files': rows,
                                         'bytes': sum(r['bytes'] for r in rows)}
                write(root / (variant + '.manifest.json'), inventories[variant])
                archives[variant] = package(out, variant, rows, root, job['job_sha256'])
            result = {'job_id': job['job_id'], 'job_sha256': job['job_sha256'],
                      'status': 'complete', 'completed_at_utc': fc.utc_now(),
                      'method': job['method'], 'candidates': job['candidates'],
                      'anatomy_review': job['anatomy_review'], 'archives': archives,
                      'output_bytes': sum(r['bytes'] for r in inventories.values()),
                      'archive_bytes': sum(r['bytes'] for r in archives.values()),
                      'elapsed_seconds': time.monotonic() - started,
                      'phase_seconds': {k: sum(r['timings'][k] for r in finished.values())
                                        for k in next(iter(finished.values()))['timings']},
                      'labels': 'withheld; no test metrics',
                      'reproduction': f'python {Path(__file__)} run --job {HERE / "submission_jobs" / JOB_ID / "job_spec.json"}'}
            write(root / 'completion.json', result)
            report = ('# Top-four test300 submission packages\n\n'
                      'Equal float32 sigmoid-probability mean, threshold >= 0.5. '
                      'd2 applies eligible whole-lung 20 mm support; d3 independently '
                      'applies eligible prompt-selected 20 mm support to d1.\n\n'
                      'Each package contains 300 native CT files / 582 prompts. '
                      'No labels, test metrics or upload. Anatomy review status: '
                      + job['anatomy_review']['status'] + '.\n\n'
                      + '\n'.join(f'- {v}: `{r["path"]}` SHA256 `{r["sha256"]}`'
                                  for v, r in archives.items()) + '\n\n'
                      + 'Reproduce: `' + result['reproduction'] + '`\n')
            fc.atomic_write_text(root / 'report.md', report)
            status('complete', cases=300, findings=582, archives=archives)
        except BaseException as exc:
            status('failed', error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'run', 'watch'):
        p = sub.add_parser(name)
        p.add_argument('--job', type=Path, required=True)
        if name == 'watch':
            p.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.job.resolve())
    elif args.command == 'run':
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run(read(args.job))
    else:
        job = read(args.job)
        while True:
            p = Path(job['runtime_root']) / 'state.json'
            print(json.dumps(read(p) if p.exists() else {'status': 'not_started'}, indent=2), flush=True)
            if args.once:
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
