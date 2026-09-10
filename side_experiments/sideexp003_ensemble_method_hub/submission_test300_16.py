#!/usr/bin/env python3
"""Sixteen-case continuation using the unchanged s001 numerical/export functions."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time

import submission_test300 as s

WORKERS = 16
RESERVE_RAM = 64 * 1024**3


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('cannot determine available RAM')


def validate_execution(execution, job):
    s.require(s.fc.json_sha256({k: v for k, v in execution.items() if k != 'execution_sha256'})
              == execution['execution_sha256'], 'execution hash drift')
    s.require(execution['job_sha256'] == job['job_sha256']
              and execution['workers'] == WORKERS and execution['cpu_allowance'] == WORKERS,
              'continuation identity drift')
    for record in execution['source_files']:
        s.verify_identity(record)
    s.validate_job(job)


def execute_cases(job, records, state, workers=WORKERS):
    """Reuse completed transactions before dispatch; never share a case between workers."""
    root = Path(job['runtime_root'])
    done = {}
    pending_cases = []
    for case in job['cases']:
        if (root / 'cases' / (case['name'] + '.json')).exists():
            done[case['name']] = s.process_case(job, case, [r[case['name']] for r in records])
        else:
            pending_cases.append(case)
    retained = set(done)
    s.require(set(job['smoke_cases']).issubset(retained), 'both retained smoke cases required')
    s.write(root / 'smoke_validation.json', {'job_sha256': job['job_sha256'],
            'status': 'passed', 'cases': job['smoke_cases'], 'retained_by_continuation': True})
    total = sum(math.prod(c['shape']) * len(c['findings']) for c in pending_cases)
    started = time.monotonic()
    queue = iter(pending_cases)
    queued = next(queue, None)
    active = {}
    with futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        last = 0.0
        while queued is not None or active:
            ram = available_ram()
            while queued is not None and len(active) < workers and ram >= RESERVE_RAM:
                c = queued
                active[pool.submit(s.process_case, job, c,
                        [r[c['name']] for r in records])] = c['name']
                queued = next(queue, None)
            finished, _ = futures.wait(active, timeout=30, return_when=futures.FIRST_COMPLETED)
            for f in finished:
                done[active.pop(f)] = f.result()
            elapsed = time.monotonic() - started
            if elapsed - last >= 60 or not active:
                new = [r for name, r in done.items() if name not in retained]
                elements = sum(r['elements'] for r in new)
                state('processing' if ram >= RESERVE_RAM else 'waiting_for_memory',
                      cases=len(done), findings=sum(len(r['finding_indices']) for r in done.values()),
                      active_cases=len(active), retained_cases=len(retained),
                      new_cases=len(new), elapsed_seconds=elapsed, available_ram_bytes=ram,
                      eta_seconds=elapsed / elements * (total - elements)
                      if len(new) >= 10 and elements else None)
                last = elapsed
            if not active and queued is not None and ram < RESERVE_RAM:
                time.sleep(30)
    return done


def finalize(job, execution, done, root, started):
    s.bind_caches(job)
    inventories = {}
    archives = {}
    out = Path(job['output_root'])
    for variant in s.VARIANTS:
        rows = [next(r for r in done[c['name']]['outputs'] if r['variant'] == variant)
                for c in job['cases']]
        s.require({p.name for p in (out / variant).iterdir()} == {c['name'] for c in job['cases']},
                  'final directory file set drift')
        s.require(len(rows) == 300 and sum(r['shape'][0] for r in rows) == 582,
                  'final cohort mismatch')
        inventories[variant] = {'cases': 300, 'findings': 582, 'files': rows,
                                'bytes': sum(r['bytes'] for r in rows)}
        s.write(root / (variant + '.manifest.json'), inventories[variant])
        archives[variant] = s.package(out, variant, rows, root, job['job_sha256'])
    result = {'job_id': job['job_id'], 'job_sha256': job['job_sha256'],
              'execution_sha256': execution['execution_sha256'], 'workers': WORKERS,
              'status': 'complete', 'completed_at_utc': s.fc.utc_now(),
              'method': job['method'], 'candidates': job['candidates'],
              'anatomy_review': job['anatomy_review'], 'archives': archives,
              'output_bytes': sum(r['bytes'] for r in inventories.values()),
              'archive_bytes': sum(r['bytes'] for r in archives.values()),
              'continuation_elapsed_seconds': time.monotonic() - started,
              'phase_seconds': {k: sum(r['timings'][k] for r in done.values())
                                for k in next(iter(done.values()))['timings']},
              'labels': 'withheld; no test metrics', 'reproduction': execution['reproduction']}
    s.write(root / 'completion.json', result)
    report = ('# Top-four test300 d1/d2/d3 packages\n\n'
              'Equal float32 sigmoid probabilities, fixed rank order, sum >= 2.0. '
              'd2 applies eligible whole-lung 20 mm support; d3 independently '
              'applies eligible prompt-selected 20 mm support to d1.\n\n'
              'Sixteen-case continuation retained the original smoke outputs and '
              'unchanged numerical/export functions. Each package contains '
              '300 native CT files / 582 prompts. No labels, test metrics or upload.\n\n'
              'Anatomy review status: ' + job['anatomy_review']['status'] + '.\n\n'
              + '\n'.join(f'- {v}: `{r["path"]}` SHA256 `{r["sha256"]}`'
                          for v, r in archives.items()) + '\n\n'
              + 'Reproduce: `' + result['reproduction'] + '`\n')
    s.fc.atomic_write_text(root / 'report.md', report)
    return result


def run(job, execution):
    validate_execution(execution, job)
    root = Path(job['runtime_root'])
    with s.tc.exclusive(root / 'launch.lock'):
        s.require(s.read(root / 'owner.json')['job_sha256'] == job['job_sha256'], 'foreign owner')
        s.require(s.gate(job)[0], 'Wave 1 dependency gate closed')
        s.require(s.enough_space(job), 'insufficient shared headroom')
        s.require(available_ram() >= 300 * 1024**3, 'need 300 GiB available RAM to start')
        records = s.bind_caches(job)
        for variant in s.VARIANTS:
            for p in (Path(job['output_root']) / variant).iterdir():
                if p.name.startswith('.' + job['job_id'] + '.'):
                    continue
                s.require(p.name in {c['name'] for c in job['cases']}
                          and (root / 'cases' / (p.name + '.json')).exists(), 'unowned output')
        def state(status, **kw):
            d = {'job_id': job['job_id'], 'job_sha256': job['job_sha256'],
                 'execution_sha256': execution['execution_sha256'], 'workers': WORKERS,
                 'pid': os.getpid(), 'status': status, 'updated_at_utc': s.fc.utc_now(), **kw}
            s.write(root / 'state.json', d)
            s.write(root / 'continuation_16/state.json', d)
            print(json.dumps(d), flush=True)
        started = time.monotonic()
        try:
            state('validating_retained_outputs')
            done = execute_cases(job, records, state)
            state('packaging', cases=300, findings=582)
            validate_execution(execution, job)
            result = finalize(job, execution, done, root, started)
            state('complete', cases=300, findings=582, archives=result['archives'])
        except BaseException as exc:
            state('failed', error=repr(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--execution', type=Path, required=True)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(s.read(args.job), s.read(args.execution))


if __name__ == '__main__':
    main()
