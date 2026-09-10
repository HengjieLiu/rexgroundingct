#!/usr/bin/env python3
"""Wait for retained smoke outputs, retire the pinned old owner, launch 16 CPUs."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import fresh_cache as fc
import test300_cache as tc
import finish_test300_wave1 as pin


def smoke_ready(job):
    root = Path(job['runtime_root'])
    for name in job['smoke_cases']:
        p = root / 'cases' / (name + '.json')
        if not p.exists():
            return False
        d = fc.read_json(p)
        tc.require(d['job_sha256'] == job['job_sha256'] and d['name'] == name,
                   'foreign smoke record')
        if d['status'] != 'complete':
            return False
        expected = [str(Path(job['output_root']) / v / name) for v in ('d1', 'd2', 'd3')]
        tc.require([r['path'] for r in d['outputs']] == expected, 'smoke output identity')
        for row in d['outputs']:
            tc.require(fc.sha256_file(Path(row['path'])) == row['sha256'], 'smoke output hash drift')
    return True


def docker_state(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]


def run(job_path, execution_path):
    job = fc.read_json(job_path)
    execution = fc.read_json(execution_path)
    root = Path(job['runtime_root'])
    session = root / 'continuation_16'
    with tc.exclusive(session / 'handoff.lock'):
        request = fc.read_json(session / 'handoff_request.json')
        expected = request['old_supervisor']
        def check_sources():
            tc.require(execution['job_sha256'] == job['job_sha256'], 'foreign execution')
            for row in execution['source_files'] + job['source_files']:
                tc.require(fc.sha256_file(Path(row['path'])) == row['sha256'], 'source drift')
        check_sources()
        current = pin.process_identity(expected['pid'])
        tc.require(current['start_ticks'] == expected['start_ticks']
                   and current['argv'] == expected['argv'] and current['state'] in ('T', 't'),
                   'old supervisor identity/state changed')
        fd = pin.open_pidfd(expected['pid'])
        try:
            def state(status, **kw):
                d = {'status': status, 'job_sha256': job['job_sha256'], 'pid': os.getpid(),
                     'updated_at_utc': fc.utc_now(), **kw}
                fc.atomic_write_json(session / 'handoff_state.json', d)
                print(json.dumps(d), flush=True)
            while not smoke_ready(job):
                check_sources()
                tc.require(not select.select([fd], [], [], 0)[0], 'old supervisor exited prematurely')
                d = docker_state(request['old_container'])
                tc.require(d['State']['Running'] and not d['State']['OOMKilled'], 'old container failed')
                state('waiting_for_remaining_smoke')
                time.sleep(30)
            state('smoke_passed_retiring_old_supervisor')
            fc.atomic_write_json(session / 'old_state_before_exit.json', fc.read_json(root / 'state.json'))
            pin.send_pidfd(fd, signal.SIGTERM)
            pin.send_pidfd(fd, signal.SIGCONT)
            deadline = time.monotonic() + 300
            while docker_state(request['old_container'])['State']['Running']:
                tc.require(time.monotonic() < deadline, 'old container did not exit; continuation not launched')
                time.sleep(2)
            tc.require(select.select([fd], [], [], 0)[0], 'old supervisor still alive')
            fc.atomic_write_json(session / 'old_container_after_exit.json', docker_state(request['old_container']))
            fc.atomic_write_json(session / 'old_state_after_exit.json', fc.read_json(root / 'state.json'))
            check_sources()
            with tc.exclusive(root / 'launch.lock'):
                tc.require(smoke_ready(job), 'retained smoke no longer valid')
            launch = fc.read_json(session / 'launch_command.json')
            tc.require(launch['execution_sha256'] == execution['execution_sha256'], 'launch identity drift')
            result = subprocess.run(launch['command'], check=True, capture_output=True, text=True)
            fc.atomic_write_json(session / 'launcher.json', {
                **launch, 'container_id': result.stdout.strip(), 'launched_at_utc': fc.utc_now()})
            state('continuation_launched', container_id=result.stdout.strip(), workers=16)
        except BaseException as exc:
            fc.atomic_write_json(session / 'handoff_error.json', {'error': repr(exc), 'updated_at_utc': fc.utc_now()})
            raise
        finally:
            os.close(fd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--execution', type=Path, required=True)
    args = p.parse_args()
    run(args.job, args.execution)
