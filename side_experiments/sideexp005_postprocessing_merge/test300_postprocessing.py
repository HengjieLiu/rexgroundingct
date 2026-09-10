#!/usr/bin/env python3
"""Frozen, label-free d11/d12 test generation; distinct from SideExp003's exporter."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import runner as core
from runner import atomic_json, digest, job_digest, job_lock, pin, read, require, sha, utc

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / 'test300_config.json'
VARIANTS = ('d11', 'd12')
CFG = JOB = None
STOP = False


def scientific():
    global np, nib, methods
    core.scientific()
    np, nib, methods = core.np, core.nib, core.methods


def validation_gate(cfg):
    """Verify completion, never compare or threshold validation scores."""
    dep = cfg['validation_dependency']
    root = Path(dep['runtime_root'])
    for name in ('job', 'config', 'launch', 'collector_source'):
        pin(dep[name])
    job = read(dep['job']['path'])
    require(job_digest(job) == job['job_sha256'] == dep['job_sha256'], 'foreign validation job')
    require(digest(read(dep['config']['path'])) == job['config_sha256'], 'validation configuration drift')
    state = read(root/'state.json') if (root/'state.json').exists() else {}
    close = read(root/'closeout_state.json') if (root/'closeout_state.json').exists() else {}
    require(state.get('status') not in ('failed', 'interrupted'), 'validation execution failed or interrupted')
    require(close.get('status') not in ('failed', 'blocked_by_run'), 'validation collection failed')
    if state:
        require(state.get('job_sha256') == dep['job_sha256'], 'foreign validation state')
    if state.get('status') != 'complete' or close.get('status') != 'complete':
        return None
    done = read(root/'completion.json')
    require(done['status'] == 'complete' and done['job_sha256'] == dep['job_sha256']
            and done['files'] == 1000 and done['finding_evaluations'] == 1905, 'invalid validation completion')
    require(close['job_sha256'] == dep['job_sha256'] and close['files_verified'] == 1000
            and close['finding_evaluations_verified'] == 1905, 'invalid validation closeout')
    report, summary = root/'reports/report.md', root/'reports/summary.json'
    require(sha(report) == done['report_sha256'] == close['report_sha256'], 'validation report hash drift')
    require(sha(summary) == close['summary_sha256'], 'validation summary hash drift')
    aggregate = read(summary)
    require(aggregate['job_sha256'] == dep['job_sha256'] and aggregate['metrics'] == done['metrics']
            and aggregate['changes'] == done['changes'], 'validation report/completion disagreement')
    require(set(done['metrics']) == {'d1', 'd2', 'd3', 'd11', 'd12'}, 'missing validation variants')
    for variant in done['metrics']:
        inventory = root/'manifests'/(variant+'.json')
        require(sha(inventory) == done['inventory_hashes'][variant], 'validation inventory drift')
        inv = read(inventory)
        require(inv['cases'] == len(inv['files']) == 200 and inv['findings'] == 381, 'validation coverage drift')
    return {'job_sha256': dep['job_sha256'], 'completion_sha256': sha(root/'completion.json'),
            'report_sha256': sha(report), 'summary_sha256': sha(summary),
            'closeout_sha256': sha(root/'closeout_state.json'), 'score_requirement': 'none'}


def enough_space(cfg, free_bytes=None):
    free = shutil.disk_usage(cfg['runtime_root']).free if free_bytes is None else free_bytes
    return free >= cfg['storage']['reserve_bytes'] + cfg['storage']['headroom_bytes']


def state(cfg, job, status, **fields):
    value = {'run_id': cfg['run_id'], 'job_sha256': job['job_sha256'], 'status': status,
             'updated_at_utc': utc(), 'workers': cfg['workers'], 'labels': 'withheld', **fields}
    atomic_json(Path(cfg['runtime_root'])/'state.json', value)
    print(__import__('json').dumps(value), flush=True)
    return value


def verify_job(cfg, job):
    require(job_digest(job) == job['job_sha256'] and digest(cfg) == job['config_sha256'], 'test job/config drift')
    for ref in cfg['inputs'].values():
        pin(ref)
    for ref in job['code']:
        require(sha(REPO/ref['relative_path']) == ref['sha256'], 'frozen test implementation drift')


def check_geometry(case):
    ct = nib.load(case['ct_path'])
    require(hashlib.sha256(ct.header.binaryblock).hexdigest() == case['ct_header_sha256'], 'CT header drift')
    require(list(ct.shape) == case['shape'][1:] and np.allclose(ct.affine, case['affine'], atol=1e-5, rtol=0), 'CT geometry drift')
    anatomy = nib.load(case['anatomy']['path'])
    require(list(anatomy.shape) == list(ct.shape) and np.allclose(anatomy.affine, ct.affine, atol=1e-5, rtol=0), 'anatomy geometry drift')
    methods.world_axis_info(ct.affine, ct.shape)
    verified = core.validate_nifti(case['input']['path'], case)
    require(verified['sha256'] == case['input']['sha256'], 'd1 source hash drift')
    return ct


def preflight(cfg):
    scientific()
    runtime = Path(cfg['runtime_root'])
    require(cfg['expected'] == {'cases': 300, 'findings': 582} and cfg['workers'] == 4, 'test execution contract drift')
    require(cfg['zip_policy'] == 'skipped_by_request' and cfg['score_requirement'] == 'none', 'authorization drift')
    with job_lock(runtime):
        for ref in cfg['inputs'].values():
            pin(ref)
        source = read(cfg['inputs']['test_job']['path'])
        require(source['job_sha256'] == cfg['d1_job_sha256'], 'foreign d1 job')
        done = read(cfg['inputs']['d1_completion']['path'])
        require(done['status'] == 'complete' and done['job_sha256'] == source['job_sha256'], 'd1 producer incomplete')
        require([c['checkpoint'] for c in done['candidates']] == [c['checkpoint'] for c in source['candidates']], 'd1 checkpoint drift')
        require([c['rank'] for c in source['candidates']] == [1, 2, 3, 4], 'd1 rank order drift')
        inventory = read(cfg['inputs']['d1_manifest']['path'])
        originals = {c['name']: c for c in read(cfg['inputs']['dataset']['path'])['test']}
        inputs = {Path(c['path']).name: c for c in inventory['files']}
        require(inventory['cases'] == len(inputs) == len(inventory['files']) == len(originals) == 300
                and inventory['findings'] == 582 and set(inputs) == set(originals), 'test cohort mismatch')
        audit = read(cfg['inputs']['anatomy_audit']['path'])
        require(audit['lut_status'] == 'PASS', 'official anatomy LUT failed')
        anatomy = {c['volume_name']: c for c in audit['cases']}
        require(set(anatomy) == set(inputs), 'anatomy cohort mismatch')
        frozen, counts = [], collections.Counter()
        for i, parent in enumerate(source['cases'], 1):
            name = parent['name']
            require(Path(name).name == name and name.endswith('.nii.gz'), 'unsafe prediction name')
            finding_list = methods.ordered_findings(originals[name])
            require(parent['findings'] == originals[name]['findings'] and parent['categories'] == originals[name]['categories'], 'test prompts/categories drift')
            require(parent['finding_indices'] == [f['finding_idx'] for f in finding_list]
                    and parent['prompt_sha256s'] == [hashlib.sha256(f['prompt'].encode()).hexdigest() for f in finding_list], 'prompt ordering drift')
            case = {'name': name, 'findings': finding_list, 'shape': [len(finding_list), *parent['shape']],
                    'affine': parent['affine'], 'ct_path': parent['ct_path'], 'ct_header_sha256': parent['ct_header_sha256'],
                    'anatomy': parent['anatomy'], 'input': inputs[name]}
            require(Path(case['input']['path']).parent == Path(cfg['d1_prediction_root']), 'foreign d1 path')
            require(case['input']['shape'] == case['shape'] and case['input']['affine'] == case['affine'], 'd1 inventory geometry mismatch')
            audited = anatomy[name]
            require(audited['geometry']['index_grid_status'] == audited['geometry']['world_header_status'] == 'PASS'
                    and audited['integrity']['status'] == 'VALID' and audited['source_provenance']['status'] == 'FULL_HASH_MATCH', 'anatomy audit failed')
            require(case['anatomy'] == {'path': audited['integrity']['path'], 'sha256': audited['integrity']['observed_sha256']}, 'anatomy identity mismatch')
            require(set(range(10, 15)).issubset(audited['labels']['unique_label_ids']), 'missing official lung lobes')
            pin(case['anatomy'])
            check_geometry(case)
            for f in finding_list:
                route = methods.parse_finding(f['prompt'])
                counts[route['roi_kind']] += 1
                counts['v2_selected'] += int(route['v2_selected'])
            frozen.append(case)
            if i % 25 == 0:
                print(f'test preflight: {i}/300 input headers and anatomy hashes verified', flush=True)
        require(len(frozen) == len({c['name'] for c in frozen}) == 300
                and sum(len(c['findings']) for c in frozen) == 582, 'frozen finding coverage')
        code = read(runtime/'source_manifest.json')['files']
        job = {'run_id': cfg['run_id'], 'config_sha256': digest(cfg), 'code': code,
               'cases': frozen, 'routing_counts': dict(counts), 'candidates': source['candidates'],
               'smoke_cases': list(dict.fromkeys([max(frozen, key=lambda c: math.prod(c['shape'][1:]))['name'],
                                                 max(frozen, key=lambda c: math.prod(c['shape']))['name']])),
               'frozen_at_utc': utc(), 'image_id': cfg['image_id']}
        job['job_sha256'] = job_digest(job)
        target = runtime/'job_manifest.json'
        if target.exists():
            require(read(target)['job_sha256'] == job['job_sha256'], 'incompatible existing test job')
            job = read(target)
        verify_job(cfg, job)
        for variant in VARIANTS:
            directory = Path(cfg['output_root'])/variant
            for p in directory.glob('*.nii.gz'):
                require((runtime/'cases'/(p.name+'.json')).exists(), 'unowned existing test output')
        atomic_json(target, job)
        atomic_json(runtime/'preflight.json', {'status': 'passed', 'job_sha256': job['job_sha256'],
                    'cases': 300, 'findings': 582, 'routing_counts': dict(counts),
                    'shared_space_available': enough_space(cfg), 'labels_accessed': False, 'checked_at_utc': utc()})
        return job


def paths(cfg, case, variant):
    dest = Path(cfg['output_root'])/variant/case['name']
    return dest, dest.with_name('.'+cfg['run_id']+'.'+case['name'])


def recover_case(cfg, job, case):
    journal = Path(cfg['runtime_root'])/'cases'/(case['name']+'.json')
    if not journal.exists():
        for variant in VARIANTS:
            require(not paths(cfg, case, variant)[0].exists(), 'unowned existing prediction')
        return None
    record = read(journal)
    require(record['job_sha256'] == job['job_sha256'] and record['case'] == case['name']
            and record['status'] in ('prepared', 'complete') and record['input_sha256'] == case['input']['sha256'], 'incompatible case journal')
    require(set(record['artifacts']) == set(VARIANTS), 'case variant coverage')
    for variant in VARIANTS:
        dest, temp = paths(cfg, case, variant)
        rec = record['artifacts'][variant]
        require(rec['path'] == str(dest) and rec['temporary_path'] == str(temp), 'foreign output path')
        source = dest if dest.exists() else temp
        require(source.exists() and sha(source) == rec['sha256'], 'output hash mismatch or missing file')
        core.validate_nifti(source, case)
        if source == temp:
            os.replace(temp, dest)
        elif temp.exists():
            require(sha(temp) == rec['sha256'], 'conflicting temporary output')
            temp.unlink()
    if record['status'] != 'complete':
        record.update(status='complete', completed_at_utc=utc())
        atomic_json(journal, record)
    return record


def postprocess(raw, anatomy, affine, findings):
    disabled = [{'eligible': False, 'fine_eligible': False, 'selected_labels': []} for _ in findings]
    all_outputs, details = methods.postprocess(raw, anatomy, affine, findings, disabled)
    outputs = {v: all_outputs[v] for v in VARIANTS}
    return outputs, [{'finding_idx': d['finding_idx'], **d['collaborator']} for d in details]


def worker_init(cfg, job):
    global CFG, JOB
    CFG, JOB = cfg, job
    scientific()


def case_worker(case):
    old = recover_case(CFG, JOB, case)
    if old is not None:
        return old
    started = time.monotonic()
    timing = {k: 0.0 for k in ('read_hash_seconds', 'postprocessing_seconds', 'nifti_write_seconds', 'validation_seconds')}
    t = time.monotonic()
    pin(case['input'])
    pin(case['anatomy'])
    ct = check_geometry(case)
    source = nib.load(case['input']['path'])
    raw = np.asanyarray(source.dataobj)
    require(np.isin(raw, [0, 1]).all(), 'nonbinary d1 input')
    anatomy = np.asanyarray(nib.load(case['anatomy']['path']).dataobj)
    timing['read_hash_seconds'] = time.monotonic()-t
    t = time.monotonic()
    outputs, routing = postprocess(raw, anatomy, ct.affine, case['findings'])
    timing['postprocessing_seconds'] = time.monotonic()-t
    t = time.monotonic()
    require(np.all(outputs['d11'] <= raw) and np.all(outputs['d12'] <= outputs['d11']), 'postprocessing subset invariant failed')
    finding_records = []
    for f, route in zip(case['findings'], routing):
        idx = f['finding_idx']
        if not route['v2_selected']:
            require(np.array_equal(outputs['d11'][idx], outputs['d12'][idx]), 'v2 bypass changed prediction')
        finding_records.append({**f, 'routing': route, 'd1_voxels': int(raw[idx].sum()),
                                **{v+'_voxels': int(outputs[v][idx].sum()) for v in VARIANTS}})
    timing['validation_seconds'] += time.monotonic()-t
    del raw, anatomy
    artifacts = {}
    for variant, arr in outputs.items():
        dest, temp = paths(CFG, case, variant)
        require(not dest.exists(), 'refusing to overwrite prediction')
        dest.parent.mkdir(parents=True, exist_ok=True)
        # These deterministic temporary paths are owned by the locked frozen job.
        t = time.monotonic()
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(arr, source.affine, header)
        for getter, setter in [('get_qform', 'set_qform'), ('get_sform', 'set_sform')]:
            matrix, code = getattr(source, getter)(coded=True)
            getattr(image, setter)(matrix, int(code))
        nib.save(image, str(temp))
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
        timing['nifti_write_seconds'] += time.monotonic()-t
        t = time.monotonic()
        verified = core.validate_nifti(temp, case, expected=arr)
        timing['validation_seconds'] += time.monotonic()-t
        artifacts[variant] = {**verified, 'path': str(dest), 'temporary_path': str(temp), 'variant': variant}
    timing['wall_seconds'] = time.monotonic()-started
    record = {'job_sha256': JOB['job_sha256'], 'status': 'prepared', 'case': case['name'],
              'input_sha256': case['input']['sha256'], 'artifacts': artifacts, 'findings': finding_records,
              'timings': timing, 'elements': math.prod(case['shape']), 'prepared_at_utc': utc()}
    journal = Path(CFG['runtime_root'])/'cases'/(case['name']+'.json')
    atomic_json(journal, record)
    for rec in artifacts.values():
        os.replace(rec['temporary_path'], rec['path'])
    record.update(status='complete', completed_at_utc=utc())
    atomic_json(journal, record)
    return record


def report(cfg, job):
    scientific()
    verify_job(cfg, job)
    records = [recover_case(cfg, job, c) for c in job['cases']]
    require(all(records) and len(records) == 300 and sum(len(r['findings']) for r in records) == 582, 'incomplete test output coverage')
    runtime = Path(cfg['runtime_root'])
    names = {c['name'] for c in job['cases']}
    outputs = {}
    for variant in VARIANTS:
        directory = Path(cfg['output_root'])/variant
        require({p.name for p in directory.iterdir()} == names, 'unexpected output files')
        files = [r['artifacts'][variant] for r in records]
        inv = {'cases': 300, 'findings': 582, 'job_sha256': job['job_sha256'], 'variant': variant,
               'bytes': sum(r['bytes'] for r in files), 'files': files}
        path = runtime/'manifests'/(variant+'.json')
        atomic_json(path, inv)
        outputs[variant] = {'directory': str(directory), 'verification': str(path), 'verification_sha256': sha(path), 'bytes': inv['bytes']}
    atomic_json(runtime/'reports/findings.json', [dict(case=r['case'], **f) for r in records for f in r['findings']])
    timing = {k: math.fsum(r['timings'][k] for r in records) for k in records[0]['timings']}
    lines = ['# SideExp005 test300 d11/d12', '', 'Complete: two folders, 600 verified files, 582 prompts per variant.', '',
             'd11 applies frozen collaborator semantic-v1 to d1. d12 applies frozen strict semantic-v2 to d11.',
             'Validation scores did not gate generation. Test labels and metrics are unavailable.',
             'ZIPs were skipped by user request; no upload was performed.', '',
             'Anatomy review status: '+cfg['source_status'], '', '| Variant | Bytes | Prediction directory |', '| --- | ---: | --- |']
    lines += [f"| {v} | {outputs[v]['bytes']} | `{outputs[v]['directory']}` |" for v in VARIANTS]
    lines += ['', 'Finding-level routing and foreground counts: `'+str(runtime/'reports/findings.json')+'`.',
              '', 'Reproduction: `python '+str(runtime/'source/side_experiments/sideexp005_postprocessing_merge/test300_postprocessing.py')+' run --config '+str(runtime/'config.json')+'`.']
    report_path = runtime/'reports/report.md'
    report_path.write_text('\n'.join(lines)+'\n')
    completion = {'status': 'complete', 'run_id': cfg['run_id'], 'job_sha256': job['job_sha256'],
                  'cases': 300, 'findings': 582, 'files': 600, 'outputs': outputs, 'labels': 'withheld',
                  'metrics': None, 'zip_policy': 'skipped_by_request', 'archives': {}, 'candidates': job['candidates'],
                  'source_status': cfg['source_status'], 'routing_counts': job['routing_counts'],
                  'validation_dependency': read(runtime/'validation_gate.json'), 'timings_worker_seconds': timing,
                  'report_sha256': sha(report_path), 'findings_sha256': sha(runtime/'reports/findings.json')}
    target = runtime/'completion.json'
    if target.exists():
        require(read(target) == completion, 'conflicting completion')
    else:
        atomic_json(target, completion)
    return completion


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def run(cfg):
    scientific()
    runtime = Path(cfg['runtime_root'])
    job = read(runtime/'job_manifest.json')
    with job_lock(runtime):
        try:
            verify_job(cfg, job)
            signal.signal(signal.SIGTERM, stop_handler)
            signal.signal(signal.SIGINT, stop_handler)
            gate = validation_gate(cfg)
            require(gate is not None, 'validation report not complete; use detached launcher to wait')
            target = runtime/'validation_gate.json'
            if target.exists():
                require(read(target) == gate, 'validation gate evidence changed')
            else:
                atomic_json(target, gate)
            records, remaining = [], []
            for case in job['cases']:
                old = recover_case(cfg, job, case)
                (records if old else remaining).append(old if old else case)
            fresh = []
            started, last = time.monotonic(), 0.0
            timings_path = runtime/'timings.json'
            previous = read(timings_path).get('wall_seconds', 0) if timings_path.exists() else 0

            def progress(status='running'):
                nonlocal last
                elapsed = time.monotonic()-started
                elements_done = sum(r['elements'] for r in fresh)
                elements_left = sum(math.prod(c['shape']) for c in job['cases'])-sum(r['elements'] for r in records)
                eta = elapsed*elements_left/elements_done if len(fresh) >= 10 else None
                state(cfg, job, status, cases_complete=len(records), cases_total=300,
                      findings_complete=sum(len(r['findings']) for r in records), eta_seconds=eta,
                      wall_seconds=previous+elapsed, shared_free_bytes=shutil.disk_usage(runtime).free)
                atomic_json(timings_path, {'wall_seconds': previous+elapsed, 'status': status, 'updated_at_utc': utc()})
                last = time.monotonic()

            groups = [[c for c in remaining if c['name'] in job['smoke_cases']],
                      [c for c in remaining if c['name'] not in job['smoke_cases']]]
            with cf.ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=mp.get_context('spawn'),
                                        initializer=worker_init, initargs=(cfg, job)) as pool:
                for group_index, group in enumerate(groups):
                    queue, pending = collections.deque(group), set()
                    while pending or queue:
                        space = enough_space(cfg)
                        while queue and len(pending) < cfg['workers'] and not STOP and space:
                            pending.add(pool.submit(case_worker, queue.popleft()))
                            space = enough_space(cfg)
                        if time.monotonic()-last >= 60:
                            progress('running' if space or not queue else 'waiting_for_space')
                        if pending:
                            done, pending = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
                            for future in done:
                                record = future.result()
                                records.append(record)
                                fresh.append(record)
                        elif STOP:
                            raise InterruptedError('dispatch stopped; completed outputs preserved')
                        elif queue:
                            time.sleep(cfg['poll_seconds'])
                    if STOP:
                        raise InterruptedError('dispatch stopped; completed outputs preserved')
                    if group_index == 0:
                        atomic_json(runtime/'smoke_validation.json', {'status': 'passed', 'job_sha256': job['job_sha256'],
                                    'cases': job['smoke_cases'], 'outputs_retained': True})
            progress('validating')
            report(cfg, job)
            progress('complete')
        except BaseException as exc:
            state(cfg, job, 'interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', error=str(exc))
            atomic_json(runtime/'error.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'updated_at_utc': utc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run', 'watch', 'report'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    runtime = Path(cfg['runtime_root'])
    if args.command == 'preflight':
        preflight(cfg)
    elif args.command == 'run':
        run(cfg)
    elif args.command == 'report':
        scientific()
        with job_lock(runtime):
            report(cfg, read(runtime/'job_manifest.json'))
    else:
        while True:
            for name in ('supervisor_state.json', 'state.json'):
                if (runtime/name).exists():
                    print(name, __import__('json').dumps(read(runtime/name), indent=2))
            if args.once:
                break
            time.sleep(cfg['poll_seconds'])


if __name__ == '__main__':
    main()
