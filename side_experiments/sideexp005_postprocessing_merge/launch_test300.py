#!/usr/bin/env python3
"""Arm a detached, separately frozen SideExp005 test postprocessing supervisor."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import test300_postprocessing as run
from runner import atomic_json, digest, job_lock, pin, read, require, sha, utc

ROOT = Path(__file__).resolve().parent


def docker_state(container):
    return json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]['State']


def freeze(cfg):
    runtime = Path(cfg['runtime_root'])
    pin(cfg['validation_dependency']['job'])
    old = read(cfg['validation_dependency']['job']['path'])
    original = Path(cfg['validation_dependency']['runtime_root'])/'source'
    sources = [(original/r['relative_path'], r['relative_path'], r['sha256']) for r in old['code']]
    for name in ('test300_postprocessing.py', 'launch_test300.py', 'test300_config.json', 'test300_execution_spec.md'):
        p = Path(cfg['repo_root'])/'side_experiments/sideexp005_postprocessing_merge'/name
        sources.append((p, str(p.relative_to(cfg['repo_root'])), sha(p)))
    entries = []
    for source, relative, expected in sources:
        require(sha(source) == expected, 'frozen validation source drift')
        target = runtime/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == expected, 'incompatible existing test source snapshot')
        else:
            shutil.copyfile(source, target)
        entries.append({'relative_path': relative, 'sha256': expected})
    manifest = {'files': sorted(entries, key=lambda r: r['relative_path']), 'validation_job_sha256': old['job_sha256']}
    path = runtime/'source_manifest.json'
    if path.exists():
        require(read(path) == manifest, 'source manifest drift')
    else:
        atomic_json(path, manifest)
    path = runtime/'config.json'
    if path.exists():
        require(read(path) == cfg, 'incompatible test configuration')
    else:
        atomic_json(path, cfg)


def command(cfg, action, detached=False):
    runtime = Path(cfg['runtime_root'])
    args = ['docker', 'run', '--network', 'none', '--cpus', '4', '--memory', '96g', '--shm-size', '1g',
            '--user', f'{os.getuid()}:{os.getgid()}', '--init',
            '--mount', f"type=bind,src={cfg['repo_root']},dst={cfg['repo_root']},readonly",
            '--mount', 'type=bind,src=/mnt/shengdata1,dst=/mnt/shengdata1,readonly',
            '--mount', f"type=bind,src={cfg['inputs']['dataset']['path']},dst={cfg['inputs']['dataset']['path']},readonly",
            '--mount', f'type=bind,src={runtime},dst={runtime}',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1', '-e', 'MPLCONFIGDIR=/tmp/matplotlib']
    for key, value in cfg['environment'].items():
        args += ['-e', key+'='+value]
    if detached:
        args += ['--detach', '--name', cfg['container_name']]
        for variant in run.VARIANTS:
            output = Path(cfg['output_root'])/variant
            args += ['--mount', f'type=bind,src={output},dst={output}']
    else:
        args += ['--rm']
    runner = runtime/'source/side_experiments/sideexp005_postprocessing_merge/test300_postprocessing.py'
    args += [cfg['image_id'], 'python', str(runner), action, '--config', str(runtime/'config.json')]
    return args


def load_manage(cfg):
    path = Path(cfg['repo_root'])/'submissions/manage.py'
    spec = importlib.util.spec_from_file_location('submission_manage_test_closeout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_register(cfg, job):
    """Call only after the original collector exits its publication phase."""
    repo, runtime = Path(cfg['repo_root']), Path(cfg['runtime_root'])
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() == 'gpu8', 'register publication requires gpu8')
    manage = load_manage(cfg)
    registry_path = repo/'submissions/registry.json'
    for attempt in range(5):
        previous = sha(registry_path)
        registry = read(registry_path)
        for row in registry['submissions']:
            if row['id'] not in run.VARIANTS:
                continue
            variant = row['id']
            row['evidence'] = {'kind': 'frozen_test_postprocessing', 'job_sha256': job['job_sha256'],
                              'run_id': cfg['run_id'], 'runner_state': str(runtime/'state.json'),
                              'job': {'path': str(runtime/'job_manifest.json'), 'sha256': sha(runtime/'job_manifest.json')},
                              'completion': {'path': str(runtime/'completion.json'), 'sha256': None},
                              'verification': {'path': str(runtime/'manifests'/(variant+'.json')), 'sha256': None},
                              'authorization': {'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'}}
            row['zip'] = {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}
            bundle = runtime/'source/side_experiments/sideexp005_postprocessing_merge'
            for key, name in [('implementation', 'methods.py'), ('source_archive', 'RexGrounding_challenge2026-c2-less-semantic-coadapt-v11-freeze.zip'),
                              ('frozen_policy', 'frozen_sources/semantic_v1_policy.json' if variant == 'd11'
                               else 'frozen_sources/apply_semantic_v2_strict_to_rex_predictions.py')]:
                p = bundle/name
                row['postprocessing_sources'][key] = {'path': str(p), 'sha256': sha(p)}
            row['postprocessing_sources']['routing'] = {'path': str(runtime/'job_manifest.json'), 'sha256': sha(runtime/'job_manifest.json')}
            annotation = 'Test generation authorized automatically after verified val200 report, regardless of scores; ZIPs skipped by request.'
            old = 'Test generation is pending review of SideExp005 val200. No test predictions, ZIPs or upload are recorded.'
            row['notes'] = row.get('notes', '').replace(old, annotation)
            if annotation not in row['notes']:
                row['notes'] += '\n'+annotation
            row['observed'] = manage.observe(row, registry, {c['name'] for c in job['cases']})
        manage.validate(registry)
        if sha(registry_path) != previous:
            time.sleep(1)
            continue
        manage.write(registry_path, registry)
        manage.atomic_text(repo/'submissions/REGISTER.md', manage.render(registry))
        return
    raise ValueError('registry changed during publication; no manual entries overwritten')


def collect(cfg, job):
    runtime, repo = Path(cfg['runtime_root']), Path(cfg['repo_root'])
    done = read(runtime/'completion.json')
    require(done['status'] == 'complete' and done['job_sha256'] == job['job_sha256']
            and done['files'] == 600 and done['cases'] == 300 and done['findings'] == 582
            and done['archives'] == {} and done['zip_policy'] == 'skipped_by_request', 'invalid test completion')
    expected_names = {c['name'] for c in job['cases']}
    for variant in run.VARIANTS:
        output = done['outputs'][variant]
        path = runtime/'manifests'/(variant+'.json')
        require(output['verification'] == str(path) and sha(path) == output['verification_sha256'], 'test inventory drift')
        inv = read(path)
        require(inv['job_sha256'] == job['job_sha256'] and inv['cases'] == len(inv['files']) == 300
                and inv['findings'] == 582 and {Path(r['path']).name for r in inv['files']} == expected_names, 'test inventory coverage')
        directory = Path(cfg['output_root'])/variant
        require({p.name for p in directory.iterdir()} == expected_names, 'test folder coverage drift')
        for r in inv['files']:
            require(Path(r['path']).parent == directory and r['dtype'] == 'uint8', 'foreign test prediction')
            pin(r)
    require(sha(runtime/'reports/report.md') == done['report_sha256'], 'test report drift')
    require(sha(runtime/'reports/findings.json') == done['findings_sha256'], 'finding provenance drift')
    for filename, source in [('test300_report.md', runtime/'reports/report.md'), ('test300_completion.json', runtime/'completion.json')]:
        dest = repo/'side_experiments/sideexp005_postprocessing_merge'/filename
        if dest.exists():
            require(dest.read_bytes() == source.read_bytes(), 'conflicting aggregate')
        else:
            temp = dest.with_name('.'+dest.name+'.tmp')
            shutil.copyfile(source, temp)
            os.replace(temp, dest)
    update_register(cfg, job)
    current = repo/'docs/current_status.md'
    text = current.read_text()
    marker = '**SideExp005 test d11/d12 automation armed:**'
    if marker in text:
        changed = text.replace(marker, '**SideExp005 test d11/d12 complete:** 600 files verified; both variants cover 300 CTs / 582 prompts. ZIPs skipped by request.\n  Historical launch policy:', 1)
        temp = current.with_name('.'+current.name+'.test300.tmp')
        temp.write_text(changed)
        require(current.read_text() == text, 'current status edited during closeout')
        os.replace(temp, current)
    return {'status': 'complete', 'job_sha256': job['job_sha256'], 'files_verified': 600, 'zip_policy': 'skipped_by_request', 'updated_at_utc': utc()}


def supervise(cfg):
    runtime = Path(cfg['runtime_root'])
    job = read(runtime/'job_manifest.json')
    with job_lock(runtime/'supervisor'):
        def status(value, **fields):
            atomic_json(runtime/'supervisor_state.json', {'status': value, 'job_sha256': job['job_sha256'],
                        'pid': os.getpid(), 'updated_at_utc': utc(), **fields})
        try:
            run.verify_job(cfg, job)
            status('waiting_for_validation_report')
            while run.validation_gate(cfg) is None:
                status('waiting_for_validation_report')
                time.sleep(cfg['poll_seconds'])
            dependency = read(cfg['validation_dependency']['launch']['path'])
            finished = docker_state(dependency['container_id'])
            require(not finished['Running'] and finished['ExitCode'] == 0, 'validation container not exited successfully')
            run.state(cfg, job, 'waiting_for_space')
            update_register(cfg, job)
            while not run.enough_space(cfg):
                status('waiting_for_space')
                time.sleep(cfg['poll_seconds'])
            launch = runtime/'worker_launch.json'
            if launch.exists():
                worker = read(launch)
                require(worker['job_sha256'] == job['job_sha256'], 'foreign worker launch')
                health = docker_state(worker['container_id'])
                require(health['Running'] or health['ExitCode'] == 0, 'prior worker failed; inspect before restart')
            else:
                for variant in run.VARIANTS:
                    (Path(cfg['output_root'])/variant).mkdir(parents=True, exist_ok=True)
                argv = command(cfg, 'run', detached=True)
                cid = subprocess.check_output(argv, text=True).strip()
                worker = {'container_id': cid, 'command': argv, 'job_sha256': job['job_sha256'], 'started_at_utc': utc()}
                atomic_json(launch, worker)
            status('running', container_id=worker['container_id'])
            last_registry = 0
            while True:
                health = docker_state(worker['container_id'])
                if not health['Running']:
                    require(health['ExitCode'] == 0, 'test worker exited unsuccessfully')
                    break
                status('running', container_id=worker['container_id'])
                if time.monotonic()-last_registry >= 60:
                    update_register(cfg, job)
                    last_registry = time.monotonic()
                time.sleep(cfg['poll_seconds'])
            result = collect(cfg, job)
            atomic_json(runtime/'supervisor_state.json', result)
        except BaseException as exc:
            status('failed', error=str(exc), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'test300_config.json')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    if args.supervise:
        supervise(cfg)
        return
    repo, runtime = Path(cfg['repo_root']), Path(cfg['runtime_root'])
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() == 'gpu8', 'launch requires gpu8')
    image = subprocess.check_output(['docker', 'image', 'inspect', cfg['image_id'], '--format', '{{.Id}}'], text=True).strip()
    require(image == cfg['image_id'], 'execution image drift')
    with job_lock(runtime/'arming'):
        # Check a running supervisor before touching any frozen state.
        with job_lock(runtime/'supervisor'):
            pass
        freeze(cfg)
        preflight = command(cfg, 'preflight')
        if not (runtime/'job_manifest.json').exists():
            subprocess.run(preflight, check=True)
        if args.preflight_only:
            return
        job = read(runtime/'job_manifest.json')
        require(read(runtime/'preflight.json')['job_sha256'] == job['job_sha256'], 'preflight identity drift')
        entry = runtime/'source/side_experiments/sideexp005_postprocessing_merge/launch_test300.py'
        logs = runtime/'logs'; logs.mkdir(exist_ok=True)
        argv = [sys.executable, str(entry), '--supervise', '--config', str(runtime/'config.json')]
        with (logs/'supervisor.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Keep the arming lock until the child owns its independent supervisor lock.
        for _ in range(50):
            require(proc.poll() is None, 'supervisor exited during arming; inspect logs/supervisor.log')
            p = runtime/'supervisor_state.json'
            if p.exists() and read(p).get('pid') == proc.pid:
                require(read(p)['status'] != 'failed', 'supervisor failed during arming; inspect supervisor_state.json')
                break
            time.sleep(.1)
        else:
            raise ValueError('supervisor startup not confirmed; inspect before retrying')
        record = {'status': 'armed', 'job_sha256': job['job_sha256'], 'supervisor_pid': proc.pid,
                  'command': argv, 'preflight_command': preflight, 'armed_at_utc': utc()}
        atomic_json(runtime/'launch.json', record)
        atomic_json(repo/'side_experiments/sideexp005_postprocessing_merge/test300_launch_report.json', record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
