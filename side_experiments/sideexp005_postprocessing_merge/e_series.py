#!/usr/bin/env python3
"""Arm val200 now; poll Wave 2 every ten minutes; release e test folders automatically."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import e_series_compute as compute
from runner import atomic_json, digest, job_digest, job_lock, pin, read, require, sha, utc

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'e_series_config.json'


def reference(path):
    return {'path': str(path), 'sha256': sha(path)}


def freeze(master):
    parent = read(master['inputs']['d_job']['path']); pin(master['inputs']['d_job'])
    sources = [(Path(master['source_runtime'])/'source'/r['relative_path'], r['relative_path'], r['sha256']) for r in parent['code']]
    for name in ('e_series.py', 'e_series_compute.py', 'e_series_config.json', 'e_series_execution_spec.md'):
        source = Path(master['repo_root'])/'side_experiments/sideexp005_postprocessing_merge'/name
        sources.append((source, str(source.relative_to(master['repo_root'])), sha(source)))
    source = Path(master['repo_root'])/'submissions/manage.py'
    sources.append((source, 'submissions/manage.py', sha(source)))
    for stage in master['runs']:
        runtime = Path(master['runs'][stage]['runtime_root']); entries = []
        for src, relative, expected in sources:
            require(sha(src) == expected, 'source drift before e freeze')
            dest = runtime/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): require(sha(dest) == expected, 'incompatible e source snapshot')
            else: shutil.copyfile(src, dest)
            entries.append({'relative_path': relative, 'sha256': expected})
        manifest = {'files': sorted(entries, key=lambda r: r['relative_path']), 'd_job_sha256': parent['job_sha256']}
        for path, value in [(runtime/'source_manifest.json', manifest), (runtime/'config.json', master)]:
            if path.exists(): require(read(path) == value, 'incompatible frozen e configuration')
            else: atomic_json(path, value)


def docker_info(name):
    result = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True)
    if result.returncode:
        if 'No such object' in result.stderr: return None
        raise RuntimeError('Docker inspection failed: '+result.stderr)
    return json.loads(result.stdout)[0]


def validate_cache(cfg, job, candidate):
    """Small completed manifests only; array contents are hashed during averaging."""
    source = candidate['test_source']; root = Path(source['cache_root'])
    export, validation = root/'export_manifest.json', root/'reproduction_validation.json'
    if not export.exists() or not validation.exists(): return None
    e, v = read(export), read(validation)
    require(e['status'] == 'strict_passed' and v['status'] == v['storage_reproduction_status'] == 'passed', 'test cache not strict-passed')
    require(e['job_spec_sha256'] == v['job_spec_sha256'] == candidate['test_job_sha256']
            and e['candidate_id'] == candidate['id'] and e['cache_key'] == source['cache_key']
            and e['candidate_source'] == source and source['checkpoint'] == candidate['checkpoint'], 'foreign test cache identity')
    require(e['dataset']['sha256'] == cfg['inputs']['test_dataset']['sha256']
            and e['paired_val200'] == source['paired_val200'], 'test split/pairing drift')
    require(e['case_count'] == v['cases'] == len(e['cases']) == 300 and e['finding_count'] == v['findings'] == 582
            and v['array_hashes_verified'] and v['same_pass_mask_mismatch_voxels'] == 0, 'test validation coverage/storage failure')
    expected = {c['name']: c for c in job['cases']}; rows = {c['name']: c for c in e['cases']}
    require(len(rows) == 300 and set(rows) == set(expected), 'test manifest cohort drift')
    import hashlib
    for name, case in expected.items():
        r = rows[name]
        require(r['shape'] == case['shape'] and r['affine'] == case['affine']
                and r['finding_indices'] == [f['finding_idx'] for f in case['findings']]
                and r['prompt_sha256s'] == [hashlib.sha256(f['prompt'].encode()).hexdigest() for f in case['findings']], 'test case prompt/geometry drift')
        require(r['dtype'] in ('float16', 'float32') and r['cache_key'] == source['cache_key']
                and r['job_spec_sha256'] == candidate['test_job_sha256']
                and r['storage_reproduction_status'] == 'passed' and r['same_pass_mask_mismatch_voxels'] == 0, 'test case storage/provenance drift')
        require(Path(r['array_path']).parent == root/'cases', 'foreign cache array path')
    return {'rank': candidate['rank'], 'candidate_id': candidate['id'], 'cache_key': source['cache_key'],
            'export_manifest': reference(export), 'validation': reference(validation)}


def wave2_status(master, test_job):
    cfg = compute.stage_config(master, 'test300')
    pin(master['inputs']['wave2_job']); export_job = read(master['inputs']['wave2_job']['path'])
    root = Path(export_job['runtime_root']); candidates, bindings, waiting, failures = [], [], [], []
    for c in test_job['candidates']:
        try:
            binding = validate_cache(cfg, test_job, c)
            if binding is None: waiting.append(c['rank'])
            else: bindings.append(binding)
        except (ValueError, KeyError, OSError) as exc:
            failures.append({'rank': c['rank'], 'error': str(exc)}); binding = None
        if c['rank'] >= 5:
            path = Path(c['test_source']['progress_path'])
            progress = read(path) if path.exists() else {}
            require(not progress or progress.get('job_spec_sha256') == export_job['job_spec_sha256'], 'foreign Wave 2 progress')
            if progress.get('status') == 'failed': failures.append({'rank': c['rank'], 'error': progress.get('error', 'worker failed')})
            name = f"sideexp003_{export_job['job_id']}_w02_g{c['test_source']['gpu']}"
            info = docker_info(name)
            exited = bool(info and not info['State']['Running'] and info['State']['ExitCode'] == 0)
            if info:
                require(info['Config']['Labels'].get('rex.test_job') == export_job['job_spec_sha256'], 'foreign Wave 2 worker')
                if not info['State']['Running'] and info['State']['ExitCode'] != 0:
                    failures.append({'rank': c['rank'], 'error': 'worker exited unsuccessfully'})
            if not exited: waiting.append(c['rank'])
            elapsed, count = progress.get('elapsed_seconds'), progress.get('cases', 0)
            estimate = elapsed*(300-count)/count if elapsed and 0 < count < 300 else None
            candidates.append({'rank': c['rank'], 'cases': count, 'gpu_cases': progress.get('gpu_cases'),
                               'status': progress.get('status', 'unrecorded'), 'manifest_verified': binding is not None,
                               'worker_exited_successfully': exited, 'container_id': info['Id'] if info else None,
                               'remaining_inference_seconds_estimate': estimate, 'progress_updated_at_utc': progress.get('updated_at_utc')})
    snapshot = {'updated_at_utc': utc(), 'job_spec_sha256': export_job['job_spec_sha256'],
                'status': 'blocked' if failures else 'ready' if not waiting else 'waiting',
                'ranks': candidates, 'failures': failures, 'bindings': bindings,
                'wave2_timing_available': (root/'wave_02_timing.json').exists(),
                'note': 'Inference estimates exclude publication/validation; later waves do not gate this job.'}
    return snapshot


def completion(cfg, job, verify_files=False):
    root = Path(cfg['runtime_root']); path = root/'completion.json'
    if not path.exists(): return None
    result = read(path)
    require(result['status'] == 'complete' and result['job_sha256'] == job['job_sha256']
            and result['stage'] == cfg['stage'] and result['files'] == cfg['expected']['cases']*5
            and result['cases'] == cfg['expected']['cases'] and result['findings'] == cfg['expected']['findings'], 'foreign/incomplete e completion')
    require(result['archives'] == {} and result['zip_policy'] == 'skipped_by_request', 'unexpected e archive contract')
    require(sha(root/'reports/report.md') == result['report_sha256'] and sha(root/'reports/per_finding.json') == result['finding_records_sha256'], 'e report evidence drift')
    names = {c['name'] for c in job['cases']}
    for v in compute.VARIANTS:
        out = result['outputs'][v]; path = root/'manifests'/(v+'.json')
        require(out['directory'] == str(Path(cfg['output_root'])/v) and out['verification'] == str(path)
                and sha(path) == out['verification_sha256'], 'e output inventory binding drift')
        inv = read(path)
        require(inv['job_sha256'] == job['job_sha256'] and inv['cases'] == len(inv['files']) == len(names)
                and inv['findings'] == cfg['expected']['findings'] and {Path(r['path']).name for r in inv['files']} == names, 'e inventory coverage')
        require({p.name for p in Path(out['directory']).iterdir()} == names, 'e prediction directory drift')
        if verify_files:
            for rec in inv['files']: pin(rec)
    return result


def bind_test(master, jobs, snapshot):
    require(snapshot['status'] == 'ready' and len(snapshot['bindings']) == 8, 'Wave 2 not fully verified')
    val = compute.stage_config(master, 'val200'); test = compute.stage_config(master, 'test300')
    require(completion(val, jobs['val200']) is not None, 'val200 report unavailable')
    record = {'job_sha256': jobs['test300']['job_sha256'], 'caches': snapshot['bindings'],
              'val_job_sha256': jobs['val200']['job_sha256'], 'val_completion': reference(Path(val['runtime_root'])/'completion.json'),
              'wave2_workers': [{'rank': r['rank'], 'container_id': r['container_id'], 'exited_successfully': r['worker_exited_successfully']} for r in snapshot['ranks']],
              'validation_score_requirement': 'none'}
    path = Path(test['runtime_root'])/'test_cache_binding.json'
    if path.exists(): require(read(path) == record, 'test input binding drift')
    else: atomic_json(path, record)


def validate_test_binding(cfg, job):
    binding = read(Path(cfg['runtime_root'])/'test_cache_binding.json')
    require(binding['job_sha256'] == job['job_sha256'] and binding['validation_score_requirement'] == 'none', 'foreign e test binding')
    pin(binding['val_completion'])
    val_cfg = compute.stage_config(cfg, 'val200'); val_job = read(Path(val_cfg['runtime_root'])/'job_manifest.json')
    require(binding['val_job_sha256'] == val_job['job_sha256'] and completion(val_cfg, val_job) is not None, 'e validation report not complete')
    require([r['rank'] for r in binding['wave2_workers']] == [5, 6, 7, 8]
            and all(r['exited_successfully'] for r in binding['wave2_workers']), 'Wave 2 worker exit not verified')
    require([c['rank'] for c in binding['caches']] == list(range(1, 9)), 'test binding rank order drift')
    output = {c['name']: [] for c in job['cases']}
    for expected, candidate in zip(binding['caches'], job['candidates']):
        for key in ('export_manifest', 'validation'): pin(expected[key])
        require(validate_cache(cfg, job, candidate) == expected, 'completed test cache changed')
        for rec in read(expected['export_manifest']['path'])['cases']:
            output[rec['name']].append({'path': rec['array_path'], 'sha256': rec['array_sha256'], 'dtype': rec['dtype']})
    return output


def container_name(cfg):
    return 'sideexp005_'+cfg['run_id']


def docker_command(master, stage, action, detach=False):
    cfg = compute.stage_config(master, stage); runtime = Path(cfg['runtime_root'])
    args = ['docker', 'run', '--network', 'none', '--cpus', '4', '--memory', '96g', '--shm-size', '1g',
            '--user', f'{os.getuid()}:{os.getgid()}', '--init',
            '--mount', f"type=bind,src={master['repo_root']},dst={master['repo_root']},readonly",
            '--mount', 'type=bind,src=/mnt/shengdata1,dst=/mnt/shengdata1,readonly',
            '--mount', f"type=bind,src={master['inputs']['test_dataset']['path']},dst={master['inputs']['test_dataset']['path']},readonly",
            '--mount', f'type=bind,src={runtime},dst={runtime}', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'PYTHONUNBUFFERED=1', '-e', 'MPLCONFIGDIR=/tmp/matplotlib']
    if stage == 'val200':
        args += ['--mount', 'type=bind,src=/data/hengjie/datasets/rexgroundingct/segmentations,dst=/data/hengjie/datasets/rexgroundingct/segmentations,readonly']
    for k,v in master['environment'].items(): args += ['-e', k+'='+v]
    if detach:
        args += ['--detach', '--name', container_name(cfg)]
        if stage == 'test300':
            for v in compute.VARIANTS:
                p = Path(cfg['output_root'])/v; args += ['--mount', f'type=bind,src={p},dst={p}']
    else: args += ['--rm']
    entry = runtime/'source/side_experiments/sideexp005_postprocessing_merge/e_series_compute.py'
    return args+[cfg['image_id'], 'python', str(entry), action, '--stage', stage, '--config', str(runtime/'config.json')]


def launch_stage(master, stage, job):
    cfg = compute.stage_config(master, stage); runtime = Path(cfg['runtime_root']); target = runtime/'launch.json'
    if target.exists():
        previous = read(target); require(previous['job_sha256'] == job['job_sha256'], 'foreign e container launch')
        info = docker_info(previous['container_id']); require(info is not None, 'owned e container missing')
        require(info['State']['Running'] or info['State']['ExitCode'] == 0, 'owned e container failed; inspect before explicit restart')
        return previous
    if not compute.enough_space(cfg):
        compute.state(cfg, job, 'waiting_for_space', cases_complete=0, findings_complete=0)
        return None
    require(docker_info(container_name(cfg)) is None, 'conflicting e container name')
    if stage == 'test300':
        for v in compute.VARIANTS: (Path(cfg['output_root'])/v).mkdir(parents=True, exist_ok=True)
    args = docker_command(master, stage, 'run', detach=True)
    cid = subprocess.check_output(args, text=True).strip()
    record = {'container_id': cid, 'job_sha256': job['job_sha256'], 'command': args, 'started_at_utc': utc()}
    atomic_json(target, record); return record


def manage_module(master):
    spec = importlib.util.spec_from_file_location('e_submission_manage', compute.REPO/'submissions/manage.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def update_register(master, jobs):
    repo = Path(master['repo_root']); m = manage_module(master)
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() == 'gpu8', 'registry update requires gpu8')
    vr, tr = [Path(master['runs'][s]['runtime_root']) for s in ('val200', 'test300')]
    for _ in range(5):
        path = repo/'submissions/registry.json'; old_sha = sha(path); registry = read(path)
        models = []
        for c in jobs['test300']['candidates']:
            source = c['test_source']
            models.append({'candidate_id': c['id'], 'checkpoint': c['checkpoint'], 'paired_val200': source['paired_val200'],
                           'preprocessing': source['cache']})
        family = {'name': 'Frozen top-eight probability ensemble',
                  'summary': 'Original ranks 1–8, equally weighted; paired val200 and test300 caches with unchanged anatomy postprocessors.',
                  'models': models, 'recipe': {'kind': 'probability_average', 'weights': [.125]*8,
                   'description': 'Frozen original ranks 1–8; float32 sigmoid-probability sum >= 4.0.'},
                  'sources': [master['inputs']['val_roster'], master['inputs']['parent_test_job'], master['inputs']['wave2_job']]}
        if 'e' in registry['families']: require(registry['families']['e'] == family, 'conflicting e family')
        else: registry['families']['e'] = family
        val_state = read(vr/'state.json') if (vr/'state.json').exists() else {'status': 'pending'}
        test_state = read(tr/'state.json') if (tr/'state.json').exists() else {'status': 'waiting_for_wave2_and_val200'}
        val_done = completion(compute.stage_config(master, 'val200'), jobs['val200'])
        policies = dict(zip(compute.VARIANTS, ('raw', 'whole_lung20', 'fine20', 'semantic_v1', 'semantic_v2_strict')))
        for sid in compute.VARIANTS:
            row = next((r for r in registry['submissions'] if r['id'] == sid), None)
            if row is None:
                row = {'id': sid, 'family': 'e', 'display_name': sid+' — Top-eight equal probability ensemble — '+policies[sid],
                       'postprocessing': policies[sid], 'parent_id': None if sid == 'e1' else 'e11' if sid == 'e12' else 'e1',
                       'prediction': {'path': str(Path(master['test_output_root'])/sid), 'directory_name': sid},
                       'expected': {'cases': 300, 'findings': 582}, 'submission_history': [], 'notes': 'Generate all e variants regardless of validation scores. ZIPs skipped by request.'}
                registry['submissions'].append(row)
            require(row['family'] == 'e' and row['prediction']['path'] == str(Path(master['test_output_root'])/sid), 'conflicting e alias')
            row['evidence'] = {'kind': 'frozen_e_ensemble', 'run_id': master['runs']['test300']['run_id'],
                               'authorization': {'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'},
                               'job_sha256': jobs['test300']['job_sha256'], 'job': reference(tr/'job_manifest.json'),
                               'runner_state': str(tr/'state.json'), 'completion': {'path': str(tr/'completion.json'), 'sha256': None},
                               'verification': {'path': str(tr/'manifests'/(sid+'.json')), 'sha256': None}}
            bundle = tr/'source/side_experiments/sideexp005_postprocessing_merge'
            row['postprocessing_sources'] = {'implementation': reference(bundle/'methods.py'), 'routing': reference(tr/'job_manifest.json'),
                                            'anatomy_review_status': master['source_status'], 'anatomy_audit': master['inputs']['test_anatomy_audit']}
            row['zip'] = {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}
            row['validation'] = {'experiment': master['experiment'], 'run_id': master['runs']['val200']['run_id'],
                                 'job_sha256': jobs['val200']['job_sha256'], 'expected': {'cases': 200, 'findings': 381},
                                 'status': val_state['status'], 'report_path': str(repo/'side_experiments/sideexp005_postprocessing_merge/de_val200_report.md'),
                                 'runtime_report_path': str(vr/'reports/report.md')}
            if val_done: row['validation'].update(status='complete', metrics=val_done['metrics'][sid], report_sha256=val_done['report_sha256'])
            row['observed'] = m.observe(row, registry, {c['name'] for c in jobs['test300']['cases']})
        m.validate(registry)
        if sha(path) != old_sha: time.sleep(1); continue
        m.write(path, registry); m.atomic_text(repo/'submissions/REGISTER.md', m.render(registry)); return
    raise ValueError('concurrent registry edits; manual history preserved')


def collect_stage(master, stage, job):
    cfg = compute.stage_config(master, stage); runtime = Path(cfg['runtime_root'])
    result = completion(cfg, job, verify_files=True); require(result is not None, 'e completion not available')
    repo = Path(master['repo_root']); prefix = 'de_val200' if stage == 'val200' else 'e_test300'
    for name, src in [(prefix+'_report.md', runtime/'reports/report.md'), (prefix+'_completion.json', runtime/'completion.json')]:
        dest = repo/'side_experiments/sideexp005_postprocessing_merge'/name
        if dest.exists(): require(dest.read_bytes() == src.read_bytes(), 'conflicting e aggregate report')
        else:
            temp = dest.with_name('.'+dest.name+'.tmp'); shutil.copyfile(src, temp); os.replace(temp, dest)
    record = {'status': 'complete', 'stage': stage, 'job_sha256': job['job_sha256'], 'files_verified': result['files'], 'updated_at_utc': utc()}
    atomic_json(runtime/'closeout.json', record)
    status_path = repo/'docs/current_status.md'; original = status_path.read_text()
    marker = '**SideExp005 e-series automation armed:**'
    addition = f"  {stage} complete: {result['files']} files verified; see `{prefix}_report.md`.\n"
    if marker in original and addition not in original:
        changed = original.replace(marker, marker+'\n'+addition, 1)
        require(status_path.read_text() == original, 'current-status concurrent edit')
        temp = status_path.with_name('.'+status_path.name+'.e_series.tmp'); temp.write_text(changed); os.replace(temp, status_path)
    index_path = repo/'side_experiments/README.md'; original = index_path.read_text()
    marker = '<!-- sideexp005-e-'+stage+' -->'
    if marker in original:
        lines = original.splitlines()
        lines = [f'  {marker} {stage}: {result["files"]} verified files; [{prefix} report](sideexp005_postprocessing_merge/{prefix}_report.md).'
                 if marker in line else line for line in lines]
        require(index_path.read_text() == original, 'side-index concurrent edit')
        temp = index_path.with_name('.README.e_series.tmp'); temp.write_text('\n'.join(lines)+'\n'); os.replace(temp, index_path)
    return record


def supervise(master):
    roots = {s: Path(r['runtime_root']) for s,r in master['runs'].items()}
    jobs = {s: read(r/'job_manifest.json') for s,r in roots.items()}
    control = roots['val200']/'automation'
    with job_lock(control):
        last_poll, last_register = -float('inf'), -float('inf')
        snapshot = None
        def status(value, **kw):
            atomic_json(control/'state.json', {'status': value, 'pid': os.getpid(), 'updated_at_utc': utc(),
                        'val_job_sha256': jobs['val200']['job_sha256'], 'test_job_sha256': jobs['test300']['job_sha256'], **kw})
        try:
            for s,j in jobs.items():
                # Both stage snapshots contain the same relative code and static input identities.
                compute.verify_job(compute.stage_config(master, s), j)
            status('starting_validation')
            compute.state(compute.stage_config(master, 'test300'), jobs['test300'], 'waiting_for_wave2_and_val200', cases_complete=0, findings_complete=0)
            while True:
                now = time.monotonic()
                if now-last_poll >= master['poll_seconds']:
                    snapshot = wave2_status(master, jobs['test300'])
                    atomic_json(control/'wave2_latest.json', snapshot)
                    with (control/'wave2_history.jsonl').open('a') as stream:
                        stream.write(json.dumps(snapshot, sort_keys=True)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    last_poll = now
                val_launch = launch_stage(master, 'val200', jobs['val200'])
                val_ready = (roots['val200']/'closeout.json').exists()
                if val_launch and not val_ready:
                    info = docker_info(val_launch['container_id']); require(info is not None, 'val worker missing')
                    if not info['State']['Running']:
                        require(info['State']['ExitCode'] == 0, 'e val worker failed')
                        collect_stage(master, 'val200', jobs['val200']); val_ready = True
                test_launch = None
                if (roots['test300']/'launch.json').exists() or (val_ready and snapshot['status'] == 'ready'):
                    if not (roots['test300']/'test_cache_binding.json').exists(): bind_test(master, jobs, snapshot)
                    test_launch = launch_stage(master, 'test300', jobs['test300'])
                    if test_launch:
                        info = docker_info(test_launch['container_id']); require(info is not None, 'test worker missing')
                        if not info['State']['Running']:
                            require(info['State']['ExitCode'] == 0, 'e test worker failed')
                            collect_stage(master, 'test300', jobs['test300']); update_register(master, jobs)
                            status('complete'); return
                current = 'test_running' if test_launch else 'validation_running' if not val_ready else 'waiting_for_wave2'
                if snapshot['status'] == 'blocked': current += '_wave2_blocked'
                status(current, wave2_status=snapshot['status'], next_wave2_poll_seconds=max(0, master['poll_seconds']-(time.monotonic()-last_poll)))
                if now-last_register >= master['heartbeat_seconds']:
                    update_register(master, jobs); last_register = now
                time.sleep(30)
        except BaseException as exc:
            status('failed', error=str(exc), traceback=traceback.format_exc()); raise


def arm(master, preflight_only=False):
    repo = Path(master['repo_root']); root = Path(master['runs']['val200']['runtime_root'])
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() == 'gpu8', 'e launch requires gpu8')
    require(subprocess.check_output(['docker', 'image', 'inspect', master['image_id'], '--format', '{{.Id}}'], text=True).strip() == master['image_id'], 'image drift')
    with job_lock(root/'arming'):
        with job_lock(root/'automation'): pass
        freeze(master)
        for stage, r in master['runs'].items():
            if not (Path(r['runtime_root'])/'job_manifest.json').exists(): subprocess.run(docker_command(master, stage, 'preflight'), check=True)
        if preflight_only: return
        entry = root/'source/side_experiments/sideexp005_postprocessing_merge/e_series.py'
        logs = root/'automation/logs'; logs.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, str(entry), 'supervise', '--config', str(root/'config.json')]
        with (logs/'supervisor.log').open('ab', buffering=0) as stream:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            require(proc.poll() is None, 'e supervisor failed to arm; inspect automation log')
            path = root/'automation/state.json'
            if path.exists() and read(path).get('pid') == proc.pid:
                require(read(path)['status'] != 'failed', 'e supervisor failed to arm'); break
            time.sleep(.1)
        else: raise ValueError('supervisor startup not confirmed')
        result = {'status': 'armed', 'pid': proc.pid, 'command': args, 'armed_at_utc': utc(), 'poll_seconds': master['poll_seconds']}
        atomic_json(root/'automation/launch.json', result)
        atomic_json(repo/'side_experiments/sideexp005_postprocessing_merge/e_series_launch.json', result)
        print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight', 'arm', 'watch', 'report', 'supervise'))
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--once', action='store_true')
    args = p.parse_args(); master = read(args.config)
    if args.action == 'supervise': supervise(master)
    elif args.action in ('preflight', 'arm'): arm(master, args.action == 'preflight')
    elif args.action == 'report':
        for stage, r in master['runs'].items():
            if (Path(r['runtime_root'])/'completion.json').exists(): collect_stage(master, stage, read(Path(r['runtime_root'])/'job_manifest.json'))
    else:
        while True:
            for stage,r in master['runs'].items():
                root = Path(r['runtime_root'])
                for name in (['automation/state.json', 'automation/wave2_latest.json', 'state.json'] if stage == 'val200' else ['state.json']):
                    if (root/name).exists(): print(stage, name, json.dumps(read(root/name), indent=2))
            if args.once: break
            time.sleep(master['poll_seconds'])


if __name__ == '__main__': main()
