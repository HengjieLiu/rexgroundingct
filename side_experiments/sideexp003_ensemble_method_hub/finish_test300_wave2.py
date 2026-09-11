#!/usr/bin/env python3
"""Hold only the j004 scheduler, finish Wave 2, and retire its owner safely."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import select
import signal
import time
import continue_test300 as old
import finish_test300_wave1 as base

fc, tc, tr = old.fc, old.tc, old.tr


def verify(identity, job, state):
    expected = str((old.HERE / 'continue_test300.py').resolve())
    args = identity['argv']
    tc.require(expected in args and args[-1] == 'run', 'wrong coordinator command')
    tc.require(state.get('pid') == identity['pid'] and state.get('wave') == 2
               and state.get('job_spec_sha256') == job['job_spec_sha256'], 'wrong coordinator job/wave')


def assert_no_later(job):
    records = tr.progress(job)
    tc.require(not any(c['id'] in records for c in job['candidates'][4:]), 'later wave has started; reconcile before continuing')
    for c in job['candidates'][4:]:
        name, _ = old.docker_command(job, c)
        tc.require(tr.docker_state(name) is None, 'later-wave container exists')


def run(pid):
    job = fc.read_json(old.DEFAULT)
    old.validate(job)
    root = Path(job['runtime_root'])
    parent = fc.read_json(old.PARENT)
    request_path = root / 'control/finish_wave2_request.json'
    with tc.exclusive(root / 'control/finish_wave2.lock'):
        identity = base.process_identity(pid)
        state = fc.read_json(root / 'state.json')
        verify(identity, job, state)
        fd = base.open_pidfd(pid)
        try:
            assert_no_later(job)
            if request_path.exists():
                request = fc.read_json(request_path)
                tc.require(request['coordinator']['start_ticks'] == identity['start_ticks'] and
                           request['coordinator']['pid'] == pid and request['job_spec_sha256'] == job['job_spec_sha256'], 'foreign guard request')
            else:
                request = {'job_spec_sha256': job['job_spec_sha256'], 'coordinator': identity,
                           'requested_at_utc': fc.utc_now(), 'guard_source_sha256': fc.sha256_file(Path(__file__)),
                           'prior_state': state, 'policy': 'finish_wave2_then_stream_r09_r20'}
                fc.atomic_write_json(request_path, request)
            base.send_pidfd(fd, signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while base.process_identity(pid)['state'] not in ('T', 't'):
                tc.require(time.monotonic() < deadline, 'coordinator did not stop')
                time.sleep(.05)
            # Close the launch race after stopping the exact process.
            assert_no_later(job)
            names = [old.docker_command(job, c)[0] for c in job['candidates'][:4]]
            while True:
                tc.require(not select.select([fd], [], [], 0)[0], 'coordinator exited unexpectedly')
                tc.require(base.process_identity(pid)['state'] in ('T', 't'), 'coordinator resumed externally')
                records = tr.progress(job)
                tc.require(not any(c['id'] in records for c in job['candidates'][4:]), 'unexpected later-wave progress')
                states = [tr.docker_state(name) for name in names]
                for d in states:
                    tc.require(d and d['Config']['Labels'].get('rex.test_job') == job['job_spec_sha256'], 'worker identity drift')
                    tc.require(d['State']['Running'] or d['State']['ExitCode'] == 0, 'worker failed; scheduler remains stopped')
                value = {'status': 'finishing_wave2_only', 'job_spec_sha256': job['job_spec_sha256'],
                         'coordinator_pid': pid, 'guard_pid': os.getpid(), 'scheduler_stopped': True,
                         'updated_at_utc': fc.utc_now(), 'held_ranks': list(range(9,21)), 'models': []}
                for c, d in zip(job['candidates'][:4], states):
                    p = records.get(c['id'], {})
                    value['models'].append({'rank': c['rank'], 'status': p.get('status'), 'cases': p.get('cases'),
                        'published': len(list((Path(c['cache_root']) / 'case_manifests').glob('*.json'))),
                        'running': d['State']['Running']})
                fc.atomic_write_json(root / 'finish_wave2_state.json', value)
                with (root / 'finish_wave2_monitor.jsonl').open('a') as f:
                    f.write(__import__('json').dumps(value) + '\n')
                print(__import__('json').dumps(value), flush=True)
                if base.all_finished(job, records, states):
                    break
                time.sleep(30)
            base.terminate_stopped_coordinator(fd)
            with tc.exclusive(root / 'launch.lock'), tc.exclusive(Path(parent['runtime_root']) / 'launch.lock'):
                fc.atomic_write_json(root / 'coordinator_exit_after_wave2.json', fc.read_json(root / 'state.json'))
                entries=[]
                for c, name in zip(job['candidates'][:4], names):
                    cache=Path(c['cache_root'])
                    entries.append({'rank':c['rank'],'id':c['id'],'cache_key':c['cache_key'],'cache_root':str(cache),
                                    'export_manifest_sha256':fc.sha256_file(cache/'export_manifest.json'),
                                    'validation_sha256':fc.sha256_file(cache/'reproduction_validation.json')})
                    log=tr.command(['docker','logs',name],check=False)
                    fc.atomic_write_text(root/'logs'/(name+'.log'),log.stdout+log.stderr)
                value.update(status='stopped_after_wave2',coordinator_exited=True,completed_models=4,
                             completed_caches=entries,updated_at_utc=fc.utc_now())
                fc.atomic_write_json(root/'finish_wave2_state.json',value)
                fc.atomic_write_json(root/'state.json',value)
                start=datetime.fromisoformat(request['prior_state']['updated_at_utc'].replace('Z','+00:00')).timestamp()-request['prior_state']['wave_elapsed_seconds']
                fc.atomic_write_json(root/'wave_02_timing.json',{'wave':2,'wall_seconds':time.time()-start,'completed_at_utc':fc.utc_now(),'basis':'original wave launch through all workers exiting'})
                for filename in ('finish_wave2_state.json','wave_02_timing.json'):
                    fc.atomic_write_text(old.DEFAULT.parent/filename,(root/filename).read_text())
                print('STOPPED_AFTER_WAVE2',flush=True)
        except BaseException as exc:
            fc.atomic_write_json(root/'finish_wave2_error.json',{'error':str(exc),'at_utc':fc.utc_now(),'coordinator_pid':pid,'policy':'leave scheduler stopped; do not manually resume'})
            raise
        finally:
            os.close(fd)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coordinator-pid',type=int,required=True)
    run(p.parse_args().coordinator_pid)
