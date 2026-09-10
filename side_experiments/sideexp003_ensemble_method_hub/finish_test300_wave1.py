#!/usr/bin/env python3
"""Finish existing Wave 1 workers while preventing all later wave launches."""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import select
import signal
import time

import fresh_cache as fc
import test300_cache as tc
import test300_runner as tr


def pidfd_syscall(number, *args):
    # gpu8's Python 3.9 lacks the wrappers; its Linux 5.4 kernel supports pidfds.
    tc.require(platform.machine() == 'x86_64', 'pidfd syscall mapping requires x86_64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(number, *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    return pidfd_syscall(434, pid, 0)


def send_pidfd(fd, sig):
    return pidfd_syscall(424, fd, sig, ctypes.c_void_p(), 0)


def process_identity(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
            'argv': [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]}


def verify_coordinator(identity, path):
    args = identity['argv']
    tc.require(any(Path(a).name == 'resume_test300_no_pause.py' for a in args),
               'not the authorized recovery coordinator')
    tc.require('--job' in args and Path(args[args.index('--job') + 1]).resolve() == path.resolve(),
               'coordinator job mismatch')


def all_finished(job, records, states):
    tc.require(len(states) == 4, 'expected four worker states')
    for c, state in zip(job['candidates'][:4], states):
        p = records.get(c['id'], {})
        if p.get('status') != 'strict_passed' or state['State']['Running']:
            return False
        tc.require(state['State']['ExitCode'] == 0, 'worker exited unsuccessfully')
        tc.require(tr.completed(job, c), 'worker lacks complete strict publication')
    return True


def terminate_stopped_coordinator(fd):
    # Deliver the stop request before permitting the Python loop to run again.
    send_pidfd(fd, signal.SIGTERM)
    send_pidfd(fd, signal.SIGCONT)
    tc.require(bool(select.select([fd], [], [], 60)[0]), 'coordinator did not exit')


def run(path, pid):
    job = fc.read_json(path)
    tc.validate_job(job)
    root = Path(job['runtime_root'])
    names = [f"sideexp003_{job['job_id']}_w01_g{c['gpu']}" for c in job['candidates'][:4]]
    with tc.exclusive(root / 'control/finish_wave1.lock'):
        identity = process_identity(pid)
        verify_coordinator(identity, path)
        fd = open_pidfd(pid)
        request_path = root / 'control/finish_wave1_request.json'
        try:
            if request_path.exists():
                request = fc.read_json(request_path)
                tc.require(request['coordinator']['start_ticks'] == identity['start_ticks']
                           and request['coordinator']['pid'] == pid
                           and request['job_spec_sha256'] == job['job_spec_sha256'],
                           'different finish-wave owner')
            else:
                old = fc.read_json(root / 'state.json')
                tc.require(old['pid'] == pid and old.get('wave') == 1,
                           'coordinator is not running Wave 1')
                records = tr.progress(job)
                tc.require(any(p.get('status') != 'strict_passed' for p in records.values()),
                           'wave may already have advanced')
                tc.require(not any(c['id'] in records for c in job['candidates'][4:]),
                           'later wave already has progress')
                request = {'requested_at_utc': fc.utc_now(), 'job_id': job['job_id'],
                           'job_spec_sha256': job['job_spec_sha256'],
                           'coordinator': identity, 'previous_state': old,
                           'guard_source_sha256': fc.sha256_file(Path(__file__)),
                           'policy': 'finish_wave1_then_hold_ranks_5_to_20'}
                fc.atomic_write_json(request_path, request)
            send_pidfd(fd, signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while process_identity(pid)['state'] not in ('T', 't'):
                tc.require(time.monotonic() < deadline, 'scheduler did not stop')
                time.sleep(0.05)
            phases_path = root / 'wave1_phase_observations.json'
            phases = fc.read_json(phases_path) if phases_path.exists() else {}
            while True:
                tc.require(not select.select([fd], [], [], 0)[0], 'scheduler unexpectedly exited')
                tc.require(process_identity(pid)['state'] in ('T', 't'), 'scheduler was resumed externally')
                records = tr.progress(job)
                tc.require(not any(c['id'] in records for c in job['candidates'][4:]),
                           'unexpected later-wave progress')
                states = [tr.docker_state(n) for n in names]
                for cid, record in records.items():
                    phases.setdefault(cid, {}).setdefault(record.get('status', 'unknown'), fc.utc_now())
                fc.atomic_write_json(phases_path, phases)
                for d in states:
                    tc.require(d and d['Config']['Labels'].get('rex.test_job') == job['job_spec_sha256'],
                               'worker identity mismatch')
                    tc.require(d['State']['Running'] or d['State']['ExitCode'] == 0,
                               'Wave 1 worker failed; scheduler remains held')
                status = {'status': 'finishing_wave1_only', 'wave': 1, 'pid': os.getpid(),
                          'coordinator_pid': pid, 'scheduler_stopped': True,
                          'job_id': job['job_id'], 'job_spec_sha256': job['job_spec_sha256'],
                          'updated_at_utc': fc.utc_now(), 'held_ranks': list(range(5, 21)),
                          'candidates': {c['id']: records.get(c['id'], {}) for c in job['candidates'][:4]}}
                fc.atomic_write_json(root / 'finish_wave1_state.json', status)
                fc.atomic_write_json(root / 'state.json', status)
                print(fc.utc_now(), 'finishing_wave1_only',
                      [(p.get('rank'), p.get('status'), p.get('cases')) for p in records.values()], flush=True)
                if all_finished(job, records, states):
                    break
                time.sleep(30)
            terminate_stopped_coordinator(fd)
            with tc.exclusive(root / 'launch.lock'):
                fc.atomic_write_json(root / 'coordinator_exit_after_wave1.json', fc.read_json(root / 'state.json'))
                started = min(p['started_at_utc'] for p in records.values())
                timing = {'wave': 1, 'completed_models': 4, 'updated_at_utc': fc.utc_now(),
                          'wall_seconds': (datetime.now(timezone.utc) - datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds(),
                          'basis': 'earliest resumed worker start through all four workers exiting',
                          'prior_interrupted_attempt_excluded': True,
                          'phase_observations': str(phases_path), 'held_ranks': list(range(5, 21))}
                fc.atomic_write_json(root / 'wave_01_timing.json', timing)
                inventory = tr.summarize(job, root)
                tc.require(inventory['completed_models'] == 4, 'unexpected complete model count')
                inventory['status'] = 'stopped_after_wave1'
                inventory['held_ranks'] = list(range(5, 21))
                fc.atomic_write_json(root / 'inventory.json', inventory)
                tr.sync(path)
                status.update(status='stopped_after_wave1', completed_models=4,
                              coordinator_exited=True, updated_at_utc=fc.utc_now())
                fc.atomic_write_json(root / 'finish_wave1_state.json', status)
                fc.atomic_write_json(root / 'state.json', status)
                report = ('# Test300 held after Wave 1\n\n'
                          'Ranks 1–4 passed complete 300-case / 582-prompt publication validation.\n'
                          'Ranks 5–20 were not launched. The coordinator has exited.\n'
                          'All published arrays are retained; later waves require a new decision.\n\n'
                          'See inventory.json for cache identities and actual storage.\n')
                fc.atomic_write_text(root / 'wave1_hold_report.md', report)
                for name in ('inventory.json', 'wave1_hold_report.md', 'finish_wave1_state.json', 'wave_01_timing.json', 'wave1_phase_observations.json'):
                    fc.atomic_write_text(path.parent / name, (root / name).read_text())
                print('STOPPED_AFTER_WAVE1', flush=True)
        except BaseException as exc:
            fc.atomic_write_json(root / 'finish_wave1_error.json',
                                 {'error': str(exc), 'updated_at_utc': fc.utc_now(),
                                  'coordinator_pid': pid, 'note': 'Do not resume scheduler; inspect guard.'})
            raise
        finally:
            os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--coordinator-pid', type=int, required=True)
    args = parser.parse_args()
    run(args.job, args.coordinator_pid)
