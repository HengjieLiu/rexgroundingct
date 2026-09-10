"""Read-only reduction of recorded profiler telemetry (no new measurements)."""
import datetime as dt
import json
from pathlib import Path

def epoch(s):
    return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()

def counters(text):
    return {k:int(v) for k,v in (line.split() for line in text.splitlines())}

def reduce(root, comparisons):
    root=Path(root)
    samples=[json.loads(line) for line in (root/'telemetry.jsonl').read_text().splitlines()]
    samples.sort(key=lambda s:s['epoch'])
    modes={}
    vm=[]
    for name in ['memory_diagnostics.jsonl','nohuge_memory_diagnostics.jsonl']:
        path=root/name
        if path.exists():vm.extend(json.loads(line) for line in path.read_text().splitlines())
    vm.sort(key=lambda s:s['epoch'])
    for mode,result in comparisons.items():
        end=epoch(result['completed_at_utc']);start=end-result['wall_seconds']
        user=system=total_cpu=iowait=0; rss_peak=0
        for a,b in zip(samples,samples[1:]):
            if not (start<=a['epoch']<b['epoch']<=end):continue
            ca=list(map(int,a['cpu_stat'].split()[1:9]));cb=list(map(int,b['cpu_stat'].split()[1:9]))
            total_cpu+=sum(y-x for x,y in zip(ca,cb));iowait+=cb[4]-ca[4]
            rss=0
            for name,procs in b.get('processes',{}).items():
                if not name.endswith('_nohuge' if mode.startswith('nohuge') else '_cpu'):continue
                oldprocs=a.get('processes',{}).get(name,{})
                for pid,rec in procs.items():
                    v=rec['stat'].split();rss+=int(v[23])*4096
                    if pid in oldprocs:
                        old=oldprocs[pid]['stat'].split()
                        user+=max(0,int(v[13])-int(old[13]));system+=max(0,int(v[14])-int(old[14]))
            rss_peak=max(rss_peak,rss)
        observed=[s for s in vm if start<=s['epoch']<=end]
        compactions={}
        if len(observed)>1:
            a=counters(observed[0]['vmstat']);b=counters(observed[-1]['vmstat']);seconds=observed[-1]['epoch']-observed[0]['epoch']
            compactions={k:b[k]-a[k] for k in ['compact_stall','compact_fail','compact_success','thp_fault_fallback']}
            compactions['observed_seconds']=seconds
        modes[mode]={'process_kernel_cpu_percent':100*system/max(1,user+system),
                     'host_iowait_percent':100*iowait/max(1,total_cpu),
                     'observed_process_rss_sum_peak_bytes':rss_peak,'host_vm_counters':compactions}
    gpu={}
    for rank in range(5,9):
        intervals=[]
        for path in (root/'gpu').glob(f'r{rank}_*.nii.gz.json'):
            rec=json.loads(path.read_text());marker=json.loads(path.with_name(path.name.replace('.nii.gz.json','.nii.gz.started.json')).read_text())
            intervals.append((epoch(marker['started_at_utc']),epoch(rec['completed_at_utc'])))
        active=idle=weighted=0.
        for a,b in zip(samples,samples[1:]):
            span=sum(max(0,min(b['epoch'],end)-max(a['epoch'],start)) for start,end in intervals)
            if not span or 'gpus' not in b:continue
            values={int(line.split(',')[0]):float(line.split(',')[1]) for line in b['gpus'].splitlines()}
            util=values[rank-5];active+=span;weighted+=span*util
            if util==0:idle+=span
        gpu[str(rank)]={'observed_case_seconds':active,'sampled_zero_utilization_seconds':idle,
                        'sampled_zero_utilization_percent':100*idle/max(1,active),
                        'mean_sampled_gpu_utilization_percent':weighted/max(1,active)}
    return {'cpu_modes':modes,'gpu_case_activity':gpu,
            'limitations':['Host VM and iowait counters include unrelated processes.','GPU utilization is sampled during case intervals and includes diagnostic spooling.','RSS sums can double-count shared mappings; worker peak RSS is also reported separately.']}
