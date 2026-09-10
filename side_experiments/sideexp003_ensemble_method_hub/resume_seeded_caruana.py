#!/usr/bin/env python3
"""Benchmark and supervise a provenance-checked CPU resume of the frozen search."""
from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

import preliminary_ensemble as pe
import seeded_caruana as sc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT_JOB = "a001_top16_after_wave4_fullval"
CHILD_JOB = "a002_gpu8_top16_seed4_resume"
PARENT_RUN = "r001_top16_per_scope_seed4_fullval"
CHILD_RUN = "r002_top16_seed4_gpu8_resume"
METHOD = "caruana_replacement_seeded_scoped"
PARENT_TRACKED = HERE / "methods" / METHOD / "runs" / PARENT_RUN
PARENT_RUNTIME = sc.RUNTIME_ROOT / "analysis_jobs" / PARENT_JOB
IMAGE = "rexgroundingct-voxtell:cu126"
MIN_DISK = 20 * 1024**4
MIN_RAM = 64 * 1024**3


def read(path):
    return pe.read_json(Path(path))


def write(path, value):
    pe.atomic_write_json(Path(path), value)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def host_path(path):
    path = Path(path)
    return REPO / path.relative_to('/workspace') if path.is_relative_to('/workspace') else path


def container_path(path):
    path = Path(path).resolve()
    return str(Path('/workspace') / path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def fingerprint(execution, label, basket_size, counts):
    return pe.json_sha256({
        'execution_spec_sha256': execution['execution_spec_sha256'],
        'pass': label, 'mode': 'initial' if basket_size is None else 'round',
        'seeds': execution['seed_candidate_ids'], 'counts': counts,
        'basket_size': basket_size,
    })


def counts_before(result, basket_size):
    if basket_size is None:
        return None
    return {scope: next(s['counts'] for s in result['scopes'][scope]['curve']
                        if s['K'] == basket_size) for scope in sc.SCOPES}


def execution_identity(execution):
    return sc._identity_sha(execution, 'execution_spec_sha256', volatile=('created_at_utc',))


def semantic_contract(execution):
    # Worker count, destination paths, and execution identity may change. These
    # fields and the frozen worker hash bind every numerical operation.
    return {k: execution[k] for k in (
        'roster_sha256', 'dataset_sha256', 'chunk_elements', 'scopes',
        'seed_candidate_ids', 'code')}


def validate_import(parent, child):
    require(execution_identity(parent) == parent['execution_spec_sha256'], 'parent identity drift')
    require(execution_identity(child) == child['execution_spec_sha256'], 'child identity drift')
    require(semantic_contract(parent) == semantic_contract(child), 'incompatible numerical contract')


def imported_partial(value, parent_fp, child_fp, source, source_sha):
    require(value['fingerprint'] == parent_fp, 'parent partial fingerprint mismatch')
    out = copy.deepcopy(value)
    out['fingerprint'] = child_fp
    out['reused_from'] = {'path': str(source), 'sha256': source_sha, 'fingerprint': parent_fp}
    return out


def choose_workers(trials):
    for trial in trials:
        require(trial.get('correctness') != 'failed', 'benchmark correctness failure')
    completed = {t['label']: t for t in trials if t.get('status') == 'complete'}
    if not {'baseline_first', 'baseline_repeat'} <= completed.keys():
        return 5, 'incomplete repeated-baseline evidence'
    baseline = min(completed[x]['wall_seconds'] for x in ('baseline_first', 'baseline_repeat'))
    eligible = [t for t in completed.values() if t['workers'] > 5
                and t['wall_seconds'] <= .85 * baseline]
    if not eligible:
        return 5, 'no verified improvement of at least 15%'
    fastest = min(t['wall_seconds'] for t in eligible)
    selected = min(t['workers'] for t in eligible if t['wall_seconds'] <= 1.05 * fastest)
    return selected, 'at least 15% faster; lowest concurrency within 5% of fastest'


def validate_partial(value, expected_fp, cases, candidates, initial):
    require(value['fingerprint'] == expected_fp, 'partial fingerprint mismatch')
    case = cases[value['case']]
    wanted = collections.defaultdict(set)
    for key in case['findings']:
        category = str(case['categories'][key])
        scopes = ['all'] + ([category] if category in sc.SCOPES[1:] else [])
        names = (['uniform'] + ['seed:' + s for s in scopes]) if initial else [
            f'trial:{s}:{c}' for s in scopes for c in candidates]
        for name in names:
            wanted[name].add(int(key))
    require(set(value['rows']) == set(wanted), 'partial trial keys mismatch')
    for key, rows in value['rows'].items():
        require(len(rows) == len(wanted[key]) and {r['finding_index'] for r in rows} == wanted[key],
                'partial finding coverage mismatch')
        for row in rows:
            index = str(row['finding_index'])
            require(row['case'] == case['name'] and row['category'] == str(case['categories'][index]),
                    'partial case/category mismatch')
            g, p, intersection = row['gt_voxels'], row['pred_voxels'], row['intersection_voxels']
            require(all(isinstance(x, int) and x >= 0 for x in (g, p, intersection))
                    and intersection <= min(g, p), 'invalid voxel counts')
            require(row['dice'] == pe.dice_from_counts(g, p, intersection), 'invalid Dice')


def validate_saved(execution, result, cases, expected_last=8, final=False):
    require(result['deterministic_result_sha256'] == sc._deterministic_result_sha(result),
            'saved result identity mismatch')
    case_map = {c['name']: c for c in cases}
    candidates = result['candidate_ids']
    inventory = []
    aggregates = {}
    limit = expected_last if final else expected_last + 1
    for k in [None, *range(5, limit + 1)]:
        label = 'initial' if k is None else f'k{k:02d}'
        before = None if k is None else k - 1
        counts = counts_before(result, before)
        fp = fingerprint(execution, label, before, counts)
        files = sorted((Path(execution['runtime_root']) / 'passes' / label / 'cases').glob('*.json'))
        seen, rows = set(), collections.defaultdict(list)
        for path in files:
            value = read(path)
            require(value['case'] not in seen, 'duplicate case partial')
            seen.add(value['case'])
            validate_partial(value, fp, case_map, candidates, k is None)
            for key, records in value['rows'].items():
                rows[key].extend(records)
            inventory.append({'path': str(path), 'relative': str(path.relative_to(execution['runtime_root'])),
                              'sha256': pe.sha256_file(path), 'case': value['case'], 'pass': label,
                              'fingerprint': fp})
        require(seen <= case_map.keys(), 'unexpected partial case')
        if k is None or k <= expected_last:
            require(seen == case_map.keys(), f'{label}: incomplete completed pass')
            aggregates[label] = rows
        else:
            require(len(files) == 27, 'handoff K9 partial count changed')
        if k is not None and k > expected_last:
            continue
        for scope in sc.SCOPES:
            step = next(s for s in result['scopes'][scope]['curve'] if s['K'] == (4 if k is None else k))
            if k is None:
                metrics = sc._summary_for_scope(rows['seed:' + scope], scope)
            else:
                trials = {c: {'metrics': sc._summary_for_scope(rows[f'trial:{scope}:{c}'], scope)}
                          for c in candidates}
                for c in candidates:
                    require(trials[c]['metrics'] == step['trials'][c]['metrics'], f'{label}: trial metric drift')
                    seq = next(s['sequence'] for s in result['scopes'][scope]['curve'] if s['K'] == before) + [c]
                    require(step['trials'][c]['counts'] == sc._counts(seq)
                            and step['trials'][c]['weights'] == sc._weights(seq), 'trial weight drift')
                require(pe.select_trial(trials) == step['added'], f'{label}: winner drift')
                metrics = trials[step['added']]['metrics']
            require(metrics == step['metrics'], f'{label}/{scope}: saved metric drift')
            require(step['counts'] == sc._counts(step['sequence'])
                    and step['weights'] == sc._weights(step['sequence'])
                    and len(step['sequence']) == step['K'], 'saved sequence/weight drift')
    return inventory, aggregates


def benchmark_worker(ticket_path):
    os.nice(10)
    ticket = read(ticket_path)
    start = time.monotonic()
    try:
        execution = copy.deepcopy(ticket['execution'])
        execution['runtime_root'] = ticket['scratch_root']
        execution['workers'] = ticket['workers']
        paths = {c: {n: Path(p) for n, p in values.items()} for c, values in ticket['paths'].items()}
        output = sc._run_pass(execution=execution, roster=ticket['roster'], cases=ticket['cases'],
                              paths=paths, pass_name=ticket['label'], mode='round',
                              seeds=execution['seed_candidate_ids'], counts=ticket['counts'], basket_size=7)
        for case in ticket['cases']:
            produced = read(sc._case_file(Path(ticket['scratch_root']) / 'passes' / ticket['label'], case['name']))
            reference_path = Path(ticket['reference_paths'][case['name']])
            require(pe.sha256_file(reference_path) == ticket['reference_hashes'][case['name']], 'reference drift')
            require(produced['rows'] == read(reference_path)['rows'], 'benchmark correctness failure: ' + case['name'])
        measurement = {'status': 'complete', 'correctness': 'passed', 'workers': ticket['workers'],
                       'label': ticket['label'], 'wall_seconds': time.monotonic() - start,
                       'pass_timing': output['timing'],
                       'logical_source_bytes': output['timing']['logical_source_bytes'],
                       'max_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                       'completed_at_utc': pe.utc_now()}
        measurement['logical_gib_per_second'] = measurement['logical_source_bytes'] / 1024**3 / measurement['wall_seconds']
        write(ticket['measurement_path'], measurement)
    except Exception as exc:
        write(ticket['measurement_path'], {'status': 'failed', 'correctness': 'failed', 'error': str(exc),
                                          'workers': ticket['workers'], 'label': ticket['label']})
        raise


class Supervisor:
    def __init__(self, session):
        self.session = session
        self.container = None
        self.interrupted = False
        self.execution = read(PARENT_TRACKED / 'execution_spec.json')
        self.original = read(PARENT_TRACKED / 'result.json')
        self.job = read(HERE / 'analysis_jobs' / PARENT_JOB / 'job_spec.json')
        self.roster = read(host_path(self.execution['roster_path']))
        self.cases = pe._load_dataset(host_path(self.execution['dataset_path']))
        self.helper_sha = pe.sha256_file(Path(__file__))
        self.telemetry = []
        self.active_runtime = PARENT_RUNTIME

    def event(self, status, **values):
        record = {'session': str(self.session), 'host': socket.gethostname(), 'pid': os.getpid(),
                  'job_id': self.active_runtime.name, 'status': status, 'container': self.container,
                  'runtime_root': str(self.active_runtime), 'updated_at_utc': pe.utc_now(), **values}
        write(self.session / 'state.json', record)
        write(PARENT_RUNTIME / 'orchestrator_state.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)

    def provenance_check(self):
        require(not self.interrupted, 'operator requested stop')
        require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'gpu8',
                'working branch changed away from gpu8')
        require(pe.sha256_file(Path(__file__)) == self.helper_sha, 'resume helper changed during execution')
        errors = sc.validate_launch_spec(self.job) + sc.validate_merged_roster(self.roster)
        require(not errors, '; '.join(errors))
        require(pe.sha256_file(host_path(self.execution['dataset_path'])) == self.execution['dataset_sha256'], 'dataset drift')

    def disk_check(self):
        require(shutil.disk_usage(PARENT_RUNTIME).free >= MIN_DISK, 'free disk below 20 TiB')

    def remaining_estimate(self, analysis):
        completed = analysis.get('completed_through_k', 0)
        if completed < 9 or not hasattr(self, 'selected_execution'):
            return {}
        result_path = host_path(self.selected_execution['caruana']['result_json'])
        if not result_path.exists():
            return {}
        timings = read(result_path).get('pass_timings', {})
        full_passes = [t['wall_seconds'] for key, t in timings.items()
                       if key.startswith('K=') and int(key[2:]) >= 10]
        if full_passes:
            seconds = statistics.median(full_passes[-3:])
            basis = 'median of recent newly completed full passes'
        elif 'K=9' in timings:
            total = sum(self.case_bytes.values())
            inherited = sum(self.case_bytes[Path(i['path']).name] for i in self.inventory if i['pass'] == 'k09')
            seconds = timings['K=9']['wall_seconds'] * total / (total - inherited)
            basis = 'K9 new-work duration scaled by uncached logical bytes'
        else:
            return {}
        current = self.active_runtime / 'passes' / f'k{completed+1:02d}' / 'cases'
        fraction = sum(self.case_bytes.get(p.name, 0) for p in current.glob('*.json')) / sum(self.case_bytes.values())
        return {'estimated_remaining_seconds': max(0, (16-completed-fraction)*seconds),
                'estimated_full_pass_seconds': seconds, 'eta_basis': basis}

    def running(self):
        if not self.container:
            return False
        p = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', self.container],
                           capture_output=True, text=True, timeout=20)
        if p.returncode:
            require('No such' in p.stderr, 'cannot establish container state: ' + p.stderr)
            return False
        return p.stdout.strip() == 'true'

    def stop_container(self):
        if self.container and self.running():
            subprocess.run(['docker', 'stop', '--time', '5', self.container], capture_output=True, timeout=20)
            if self.running():
                subprocess.run(['docker', 'kill', self.container], capture_output=True, timeout=20)
        require(not self.running(), 'container has not stopped')
        self.container = None

    def cleanup(self):
        # Keep lock ownership while daemon connectivity or termination is uncertain.
        while self.container:
            try:
                self.stop_container()
            except Exception as exc:
                print('Waiting to confirm container exit before releasing lock: ' + str(exc), flush=True)
                time.sleep(10)

    def docker_run(self, label, workers, command, writable, deadline=None, production=False):
        self.provenance_check()
        self.disk_check()
        self.container = ('rex-resume-' + self.session.name + '-' + label).lower().replace('_', '-')
        cmd = ['docker', 'run', '--rm', '--name', self.container, '--user', f'{os.getuid()}:{os.getgid()}',
               '--cpus', str(workers + 1), '--workdir', '/workspace',
               '-v', f'{REPO}:/workspace:ro', '-v', '/data/hengjie:/data/hengjie:ro',
               '-v', '/mnt/shengdata1:/mnt/shengdata1:ro',
               '-e', 'HOME=/tmp', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
               '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1']
        for path in writable:
            cmd += ['-v', f'{path}:{container_path(path)}:rw']
        cmd += [IMAGE, *command]
        write(self.session / f'{label}_command.json', {'argv': cmd})
        stats = []
        started, last_check, last_stats = time.monotonic(), 0., 0.
        started_wall = time.time()
        with (self.session / f'{label}.log').open('x') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(self.lock_fd,))
            timed_out = False
            while process.poll() is None:
                now = time.monotonic()
                if self.interrupted:
                    raise InterruptedError('operator requested stop')
                if deadline is not None and now >= deadline - 8:
                    timed_out = True
                    self.stop_container()
                    process.wait(timeout=20)
                    break
                if now - last_check >= 60:
                    self.provenance_check()
                    self.disk_check()
                    if production:
                        analysis = read(self.active_runtime / 'analysis_state.json') if (self.active_runtime / 'analysis_state.json').exists() else {}
                        pass_counts = {p.name: len(list((p / 'cases').glob('*.json')))
                                       for p in (self.active_runtime / 'passes').glob('*') if p.is_dir()}
                        state_path = self.active_runtime / 'analysis_state.json'
                        if not state_path.exists() or state_path.stat().st_mtime < started_wall:
                            status = 'validating_source_arrays'
                        else:
                            status = 'replaying_saved_results' if analysis.get('completed_through_k', 0) < 8 else 'computing_new_cases'
                        self.event(status, analysis=analysis, completed_cases=pass_counts,
                                   elapsed_seconds=now-started, workers=workers,
                                   **self.remaining_estimate(analysis))
                    else:
                        self.event('benchmarking', trial=label, workers=workers, elapsed_seconds=now-started)
                    last_check = now
                if now - last_stats >= 30:
                    p = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}', self.container],
                                       text=True, capture_output=True, timeout=20)
                    if p.returncode == 0 and p.stdout.strip():
                        stats.append({'elapsed_seconds': now-started, 'stats': json.loads(p.stdout)})
                        write(self.session / f'{label}_telemetry.json', stats)
                    last_stats = now
                time.sleep(1)
            code = process.wait()
        self.stop_container()
        return {'returncode': code, 'timed_out': timed_out, 'container_wall_seconds': time.monotonic()-started}

    def archive_and_validate(self):
        self.provenance_check()
        require(execution_identity(self.execution) == self.execution['execution_spec_sha256'], 'execution drift')
        require(self.original['maximum_basket_size'] == 16, 'unexpected search limit')
        self.inventory, _ = validate_saved(self.execution, self.original, self.cases)
        archive = self.session / 'archive'
        archive.mkdir()
        for label, folder in [('caruana', PARENT_TRACKED), ('uniform', host_path(self.execution['uniform']['result_json']).parent)]:
            target = archive / label
            target.mkdir()
            for p in folder.iterdir():
                if p.is_file() and p.suffix in {'.json', '.md'}:
                    shutil.copy2(p, target / p.name)
        for name in ('analysis_state.json', 'orchestrator_state.json'):
            if (PARENT_RUNTIME / name).exists():
                shutil.copy2(PARENT_RUNTIME / name, archive / name)
        shutil.copy2(HERE / 'analysis_jobs' / PARENT_JOB / 'job_spec.json', archive / 'parent_job_spec.json')
        write(archive / 'partial_inventory.json', self.inventory)
        for item in self.inventory:
            target = archive / 'partials' / item['relative']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item['path'], target)
        image = subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
        source_container = read(self.job['source_jobs'][1]['path'])['container']
        expected_image = source_container.get('image_id')
        if expected_image:
            require(image == expected_image, 'Docker image identity differs from frozen source job')
        write(self.session / 'session_manifest.json', {
            'host': socket.gethostname(), 'uid': os.getuid(), 'gid': os.getgid(),
            'branch': 'gpu8', 'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'image': IMAGE, 'image_id': image, 'helper_sha256': self.helper_sha,
            'parent_execution_sha256': self.execution['execution_spec_sha256'],
            'inventory_sha256': pe.sha256_file(archive / 'partial_inventory.json'),
            'started_at_utc': pe.utc_now(), 'historical_pass_timings': self.original['pass_timings'],
        })
        self.paths = pe._case_paths(self.roster, self.cases)
        ranked = sorted([read(p) for p in (PARENT_RUNTIME / 'passes/k08/cases').glob('*.json')],
                        key=lambda r: (r['logical_source_bytes'], r['case']))
        self.case_bytes = {sc._case_file(Path('.'), r['case']).name: r['logical_source_bytes'] for r in ranked}
        sample = [ranked[math.floor((i + .5)*len(ranked)/20)] for i in range(20)]
        self.sample_names = {r['case'] for r in sample}
        write(self.session / 'benchmark_sample.json', {
            'selection': '20 size-quantile midpoints sorted by logical_source_bytes then case',
            'cases': [r['case'] for r in sample], 'logical_source_bytes': sum(r['logical_source_bytes'] for r in sample),
            'basket_size_before_addition': 7})
        self.event('preflight_passed', verified_partials=len(self.inventory))

    def benchmark(self):
        trials = []
        begun = time.monotonic()
        deadline = begun + 1200
        def trial(label, workers):
            scratch = self.session / 'benchmark' / label
            scratch.mkdir(parents=True)
            ticket = {'execution': self.execution, 'roster': self.roster,
                      'cases': [c for c in self.cases if c['name'] in self.sample_names],
                      'paths': {c: {n: str(p) for n, p in values.items() if n in self.sample_names} for c, values in self.paths.items()},
                      'workers': workers, 'label': label, 'scratch_root': str(scratch),
                      'counts': counts_before(self.original, 7),
                      'reference_paths': {n: str(self.session / 'archive/partials/passes/k08/cases' / sc._case_file(Path('.'), n).name) for n in self.sample_names},
                      'reference_hashes': {i['case']: i['sha256'] for i in self.inventory if i['pass'] == 'k08' and i['case'] in self.sample_names},
                      'measurement_path': str(scratch / 'measurement.json')}
            path = scratch / 'ticket.json'
            write(path, ticket)
            outcome = self.docker_run(label, workers,
                ['python', container_path(Path(__file__)), 'benchmark-worker', '--ticket', str(path)], [scratch], deadline=deadline)
            measured = read(scratch / 'measurement.json') if (scratch / 'measurement.json').exists() else {}
            if measured.get('correctness') == 'failed':
                raise RuntimeError('benchmark correctness failure: ' + measured.get('error', 'unknown'))
            require(outcome['timed_out'] or outcome['returncode'] == 0, 'benchmark execution failure: ' + label)
            if outcome['timed_out']:
                measured = {'label': label, 'workers': workers, 'status': 'timeout'}
            else:
                require(measured.get('status') == 'complete', 'missing benchmark completion record')
            measured.update(outcome)
            trials.append(measured)
            write(self.session / 'benchmark_trials.json', trials)
            return measured
        first = trial('baseline_first', 5)
        if first.get('status') == 'complete' and time.monotonic() < deadline - 10:
            ten = trial('workers10', 10)
            if ten.get('status') == 'complete':
                reserve = 1.25 * first['container_wall_seconds']
                if deadline - time.monotonic() >= reserve + 1.25 * ten['container_wall_seconds'] + 10:
                    trial('workers20', 20)
            if time.monotonic() < deadline - 10:
                trial('baseline_repeat', 5)
        workers, reason = choose_workers(trials)
        report = {'selected_workers': workers, 'reason': reason, 'trials': trials,
                  'benchmark_elapsed_seconds': time.monotonic()-begun, 'budget_seconds': 1200,
                  'cache_caveat': 'No caches dropped; repeated reads may benefit from filesystem caching.',
                  'historical_results_are_optimistic_same_val200': True}
        write(self.session / 'benchmark_report.json', report)
        lines = ['# GPU8 concurrency benchmark', '', f'Selected workers: **{workers}**. {reason}.', '',
                 '| Trial | Workers | Status | Compute seconds | Correctness |', '| --- | ---: | --- | ---: | --- |']
        for t in trials:
            lines.append(f"| {t['label']} | {t['workers']} | {t['status']} | {t.get('wall_seconds', 0):.3f} | {t.get('correctness', 'not established')} |")
        lines += ['', report['cache_caveat'], '', f"Total benchmark elapsed: {report['benchmark_elapsed_seconds']:.1f} seconds."]
        pe.atomic_write_text(self.session / 'benchmark_report.md', '\n'.join(lines) + '\n')
        self.event('benchmark_complete', selected_workers=workers, reason=reason)
        return workers

    def prepare_child(self, workers):
        target = HERE / 'methods' / METHOD / 'runs' / CHILD_RUN
        runtime = sc.RUNTIME_ROOT / 'analysis_jobs' / CHILD_JOB
        tracked_job = HERE / 'analysis_jobs' / CHILD_JOB
        require(not target.exists() and not runtime.exists() and not tracked_job.exists(), 'child destination already exists')
        target.mkdir(parents=True)
        runtime.mkdir(parents=True)
        tracked_job.mkdir(parents=True)
        child = copy.deepcopy(self.execution)
        child.update(workers=workers, runtime_root=str(runtime), launch_job_id=CHILD_JOB, created_at_utc=pe.utc_now())
        child['caruana'] = {'run_id': CHILD_RUN, 'result_json': container_path(target/'result.json'),
                            'report_md': container_path(target/'report.md'), 'decision_json': container_path(target/'decision.json')}
        (runtime/'uniform').mkdir()
        child['uniform'] = {**child['uniform'], 'result_json': str(runtime/'uniform/result.json'),
                            'report_md': str(runtime/'uniform/report.md'), 'decision_json': str(runtime/'uniform/decision.json')}
        child['resume_provenance'] = {'parent_execution_sha256': self.execution['execution_spec_sha256'],
                                     'parent_job_id': PARENT_JOB, 'helper_sha256': self.helper_sha,
                                     'benchmark_report_sha256': pe.sha256_file(self.session/'benchmark_report.json'),
                                     'session': str(self.session)}
        child['execution_spec_sha256'] = execution_identity(child)
        validate_import(self.execution, child)
        write(target/'execution_spec.json', child)
        spec = read(PARENT_TRACKED/'run_spec.json')
        spec['run_id'] = CHILD_RUN
        spec['execution']['worker_processes'] = workers
        spec['outputs'].update(tracked_run_dir=str(target), runtime_run_dir=str(runtime),
                               result_json=str(target/'result.json'), report_md=str(target/'report.md'),
                               final_decision_json=str(target/'decision.json'))
        spec['resume_provenance'] = child['resume_provenance']
        write(target/'run_spec.json', spec)
        pe.atomic_write_text(target/'run_spec.md', sc._render_run_spec(spec))
        write(tracked_job/'resume_spec.json', {'job_kind': 'verified_metric_continuation', 'job_id': CHILD_JOB,
              'execution_spec': str(target/'execution_spec.json'), 'execution_spec_sha256': child['execution_spec_sha256'],
              'parent_job_id': PARENT_JOB, 'session': str(self.session), 'workers': workers})
        receipts = []
        for item in self.inventory:
            source = Path(item['path'])
            require(pe.sha256_file(source) == item['sha256'], 'parent partial changed before import')
            label = item['pass']
            k = None if label == 'initial' else int(label[1:])-1
            fp = fingerprint(child, label, k, counts_before(self.original, k))
            destination = runtime/item['relative']
            value = imported_partial(read(source), item['fingerprint'], fp, source, item['sha256'])
            write(destination, value)
            receipts.append({**item, 'destination': str(destination), 'child_fingerprint': fp,
                             'child_sha256': pe.sha256_file(destination),
                             'metric_rows_sha256': pe.json_sha256(value['rows'])})
        receipt_path = runtime/'import_receipt.json'
        write(receipt_path, {'parent_execution_sha256': self.execution['execution_spec_sha256'],
                            'child_execution_sha256': child['execution_spec_sha256'], 'partials': receipts})
        validate_saved(child, self.original, self.cases)
        return child, target/'execution_spec.json'

    def finish(self, execution, workers):
        result = read(host_path(execution['caruana']['result_json']))
        require(result['status'] == 'complete', 'worker exited without complete result')
        inventory, _ = validate_saved(execution, result, self.cases, expected_last=16, final=True)
        for scope in sc.SCOPES:
            curve = result['scopes'][scope]['curve']
            require([s['K'] for s in curve] == list(range(4,17)), 'incomplete curve')
            require(curve[:5] == self.original['scopes'][scope]['curve'], 'inherited curve changed')
            final = result['scopes'][scope]
            require(final['final_counts'] == sc._counts(curve[-1]['sequence'])
                    and final['final_weights'] == sc._weights(curve[-1]['sequence'])
                    and final['final_sequence'] == curve[-1]['sequence']
                    and sum(final['final_counts'].values()) == 16
                    and math.isclose(sum(final['final_weights'].values()), 1.0, abs_tol=1e-12), 'final weight mismatch')
            peak = min(curve, key=lambda s: (-s['metrics']['dice'], -s['metrics']['hits'], s['K']))
            require(final['peak'] == {'K': peak['K'], 'dice': peak['metrics']['dice'],
                                     'hits': peak['metrics']['hits'], 'counts': peak['counts']}, 'peak mismatch')
        for item in self.inventory:
            require(pe.sha256_file(Path(item['path'])) == item['sha256'], 'original partial was modified')
        summary = {'status': 'complete', 'workers': workers, 'execution_spec_sha256': execution['execution_spec_sha256'],
                   'verified_case_partials': len(inventory), 'historical_pass_timings': self.original['pass_timings'],
                   'resumed_worker_pass_timings': result['pass_timings'], 'resumed_worker_timing': result['timing'],
                   'timing_caveat': 'Replayed passes have short replay wall times; original compute times are archived separately.',
                   'result': str(host_path(execution['caruana']['result_json'])), 'completed_at_utc': pe.utc_now()}
        write(self.session/'closeout.json', summary)
        lines = ['# GPU8 seeded-search resume closeout', '', 'Complete. Optimistic same-val200 diagnostic only.', '',
                 f'Workers: {workers}. Verified partials: {len(inventory)}. Original partials unchanged.', '',
                 '| Scope | K16 Dice | K16 hits | Best K | Best Dice |', '| --- | ---: | ---: | ---: | ---: |']
        for scope in sc.SCOPES:
            s = result['scopes'][scope]; m = s['curve'][-1]['metrics']
            lines.append(f"| {scope} | {m['dice']:.6f} | {m['hits']} | {s['peak']['K']} | {s['peak']['dice']:.6f} |")
        lines += ['', summary['timing_caveat'], '', 'No derived logits, model inference, or final recipe promotion.']
        pe.atomic_write_text(self.session/'closeout.md', '\n'.join(lines)+'\n')
        self.event('complete', closeout=str(self.session/'closeout.json'), result=summary['result'])

    def run(self):
        self.disk_check()
        require(sc._available_memory_bytes() >= MIN_RAM, 'available memory below 64 GiB')
        self.archive_and_validate()
        workers = self.benchmark()
        self.disk_check()
        require(sc._available_memory_bytes() >= MIN_RAM, 'available memory below 64 GiB')
        if workers == 5:
            execution, spec_path = self.execution, PARENT_TRACKED/'execution_spec.json'
        else:
            execution, spec_path = self.prepare_child(workers)
        self.active_runtime = Path(execution['runtime_root'])
        self.selected_execution = execution
        write(self.session/'selected_execution.json', {'workers': workers, 'spec_path': str(spec_path),
              'execution_spec_sha256': execution['execution_spec_sha256'], 'runtime_root': str(self.active_runtime)})
        writable = [self.active_runtime, host_path(execution['caruana']['result_json']).parent]
        uniform_dir = host_path(execution['uniform']['result_json']).parent
        if not uniform_dir.is_relative_to(self.active_runtime):
            writable.append(uniform_dir)
        self.event('launching_resume', workers=workers, execution_spec=str(spec_path))
        outcome = self.docker_run('production', workers,
            ['python', container_path(HERE/'seeded_caruana.py'), 'worker', '--spec', container_path(spec_path)],
            writable, production=True)
        require(outcome['returncode'] == 0, 'production worker failed')
        self.finish(execution, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('benchmark-worker'); worker.add_argument('--ticket', type=Path, required=True)
    run = sub.add_parser('run'); run.add_argument('--session-id', required=True)
    args = parser.parse_args()
    if args.command == 'benchmark-worker':
        benchmark_worker(args.ticket)
        return 0
    require(socket.gethostname() == 'shenggpu8', 'resume must run on gpu8')
    require(subprocess.check_output(['git','branch','--show-current'], cwd=REPO, text=True).strip() == 'gpu8', 'branch must be gpu8')
    require(args.session_id.replace('_','').replace('-','').isalnum(), 'unsafe session ID')
    with (PARENT_RUNTIME/'launch.lock').open('r+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another supervisor owns the run; no changes made.', file=sys.stderr)
            return 73
        # A separate open must fail, proving competing supervisors are excluded.
        with (PARENT_RUNTIME/'launch.lock').open('r+') as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError('shared lock exclusion check failed')
        session = PARENT_RUNTIME/'resume_sessions'/args.session_id
        session.mkdir(parents=True, exist_ok=False)
        supervisor = Supervisor(session)
        supervisor.lock_fd = lock.fileno()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(supervisor, 'interrupted', True))
        try:
            supervisor.run()
        except Exception as exc:
            supervisor.event('failed', error=str(exc))
            raise
        finally:
            supervisor.cleanup()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
