#!/usr/bin/env python3
"""Detached, fail-closed Wave 2 wait -> publication preflight -> j005 launch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import stream_test300 as s

fc,tc,tr=s.fc,s.tc,s.tr
ROOT=fc.RUNTIME_ROOT/'cache/jobs'/s.JOB_ID
FILES=('stream_test300.py','stream_test300_io.py','stream_test300_preflight.py',
       'stream_test300_transition.py','stream_test300_spec.md','finish_test300_wave2.py',
       'test_stream_test300.py','test_stream_test300_io.py','test_stream_test300_preflight.py','test_finish_test300_wave2.py')


def source_hashes():return {name:fc.sha256_file(s.HERE/name) for name in FILES}


def check_plan(plan):
    tc.require(socket.gethostname()=='shenggpu8','wrong host')
    tc.require(tr.command(['git','branch','--show-current']).stdout.strip()=='gpu8','wrong branch')
    tc.require(source_hashes()==plan['source_hashes'],'transition implementation changed after arming')
    tc.require(fc.sha256_file(s.PARENT)==plan['parent_file_sha256'],'parent job changed')
    parent=fc.read_json(s.PARENT)
    for entry in parent['source_bundle']['files']:
        tc.require(fc.sha256_file(tc.REPO/entry['path'])==entry['sha256'],'frozen parent source changed')
    return parent


def gate(parent):
    root=Path(parent['runtime_root'])
    tc.require(not (root/'finish_wave2_error.json').exists(),'Wave 2 guard failed; later waves remain held')
    state=fc.read_json(root/'finish_wave2_state.json')
    tc.require(state['job_spec_sha256']==parent['job_spec_sha256'],'foreign Wave 2 handoff')
    tc.require(not any(c['id'] in tr.progress(parent) for c in parent['candidates'][4:]),'later parent work exists')
    if state['status']=='stopped_after_wave2':
        tc.require(state['coordinator_exited'],'coordinator not retired')
        tc.require(all(tr.completed(parent,c) for c in parent['candidates'][:4]),'Wave 2 completion no longer valid')
        return True
    tc.require(state['status']=='finishing_wave2_only' and state['scheduler_stopped'],'Wave 2 hold missing')
    from finish_test300_wave1 import process_identity
    identity=process_identity(state['coordinator_pid'])
    request=fc.read_json(root/'control/finish_wave2_request.json')
    tc.require(identity['start_ticks']==request['coordinator']['start_ticks'] and identity['state'] in ('T','t'),'held coordinator identity changed')
    process_identity(state['guard_pid'])
    return False


def disk_snapshot():
    rows={}
    for line in Path('/proc/diskstats').read_text().splitlines():
        a=line.split()
        if a[2] in ('md0','md0p1','sda','sdb'):
            rows[a[2]]={'read_bytes':int(a[5])*512,'write_bytes':int(a[9])*512,'io_ms':int(a[12]),'weighted_io_ms':int(a[13])}
    return {'at_monotonic':time.monotonic(),'devices':rows,'cpu':Path('/proc/stat').read_text().splitlines()[0]}


def docker_base(image):
    return ['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
            '-v',str(tc.REPO)+':/workspace:ro','-w','/workspace',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','NUMPY_MADVISE_HUGEPAGE=0',
            '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']


def logged(argv,name):
    fc.atomic_write_json(ROOT/'transition_commands'/(name+'.json'),{'argv':argv,'at_utc':fc.utc_now()})
    with (ROOT/'logs'/(name+'.log')).open('w') as f:
        subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,cwd=tc.REPO,check=True)


def run():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    with tc.exclusive(ROOT/'transition.lock'):
        path=ROOT/'transition_plan.json'
        if path.exists():
            plan=fc.read_json(path)
        else:
            parent=fc.read_json(s.PARENT)
            plan={'source_hashes':source_hashes(),'parent_file_sha256':fc.sha256_file(s.PARENT),
                  'image_id':parent['container']['image_id'],'armed_at_utc':fc.utc_now(),
                  'policy':'finish Wave 2 unchanged; require cold parity and regressions; freeze and launch ranks 9–20'}
            fc.atomic_write_json(path,plan)
        previous=None;last_report=0
        def update(phase,**kw):
            nonlocal previous,last_report
            snap=disk_snapshot();throughput={}
            if previous:
                dt=snap['at_monotonic']-previous['at_monotonic']
                for name,v in snap['devices'].items():
                    if name in previous['devices']:
                        throughput[name]={k:(v[k]-previous['devices'][name][k])/dt for k in ('read_bytes','write_bytes')}
            previous=snap
            value={'phase':phase,'pid':os.getpid(),'at_utc':fc.utc_now(),'disk_counters':snap,
                   'disk_bytes_per_second':throughput,'available_ram_bytes':s.prof.mem_available(),**kw}
            fc.atomic_write_json(ROOT/'transition_state.json',value)
            with (ROOT/'transition_monitor.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
            if time.monotonic()-last_report>=300 or phase in ('failed','launched','complete'):
                with (ROOT/'five_minute_reports.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                print(json.dumps(value),flush=True);last_report=time.monotonic()
        try:
            parent=check_plan(plan)
            while not gate(parent):
                update('waiting_for_wave2',wave2=fc.read_json(Path(parent['runtime_root'])/'finish_wave2_state.json'))
                time.sleep(30);check_plan(plan)
            check_plan(plan);update('publication_preflight')
            preflight=ROOT/'preflight';local=tc.STAGING_ROOT/s.JOB_ID/'preflight'
            preflight.mkdir(exist_ok=True);local.mkdir(parents=True,exist_ok=True)
            if not (preflight/'report.json').exists():
                argv=docker_base(plan['image_id'])+['--name','sideexp003_j005_publication_preflight',
                    '-v','/data/hengjie:/data/hengjie:ro','-v','/mnt/shengdata1:/mnt/shengdata1:ro',
                    '-v',str(preflight)+':'+str(preflight)+':rw','-v',str(local)+':'+str(local)+':rw',
                    '--tmpfs','/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m',
                    plan['image_id'],'python','/workspace/'+str((s.HERE/'stream_test300_preflight.py').relative_to(tc.REPO))]
                logged(argv,'publication_preflight')
            report=fc.read_json(preflight/'report.json')
            tc.require(report['status']=='passed' and report['cases']==16,'preflight failed')
            for name,digest in report['source_hashes'].items():tc.require(digest==plan['source_hashes'][name],'preflight source drift')
            check_plan(plan);update('regressions')
            argv=docker_base(plan['image_id'])+[plan['image_id'],'python','-m','unittest','discover','-s',
                'side_experiments/sideexp003_ensemble_method_hub','-p','test_*.py']
            logged(argv,'regressions')
            logged([sys.executable,'scripts/rexgroundingct/check_repo_workflow.py'],'repository_checks')
            check_plan(plan);tc.require(gate(parent),'lost Wave 2 hold');update('freezing')
            if not s.DEFAULT.exists():s.prepare()
            job=fc.read_json(s.DEFAULT);s.validate(job)
            tc.require(not (ROOT/'automatic_launch.json').exists(),'automatic launch already recorded; inspect ownership before restart')
            with (ROOT/'logs/coordinator.log').open('a') as out:
                argv=[sys.executable,'-u',str(s.HERE/'stream_test300.py'),'run','--job',str(s.DEFAULT)]
                child=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT,cwd=tc.REPO,start_new_session=True)
            fc.atomic_write_json(ROOT/'automatic_launch.json',{'argv':argv,'pid':child.pid,'job_spec_sha256':job['job_spec_sha256'],
                'preflight_sha256':fc.sha256_file(preflight/'report.json'),'at_utc':fc.utc_now()})
            launched=time.monotonic();update('launched',coordinator_pid=child.pid)
            while child.poll() is None:
                state=fc.read_json(ROOT/'state.json') if (ROOT/'state.json').exists() else {}
                update('monitoring_first_30_minutes' if time.monotonic()-launched<1800 else 'monitoring',run=state)
                time.sleep(60)
            tc.require(child.returncode==0 and fc.read_json(ROOT/'state.json')['status']=='complete','streaming continuation failed')
            update('complete',run=fc.read_json(ROOT/'state.json'))
        except BaseException as exc:
            update('failed',error=str(exc),policy='No parent coordinator resume; inspect before retry')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','watch']);p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.action=='run':run()
    else:print(json.dumps(fc.read_json(ROOT/'transition_state.json'),indent=2))
