#!/usr/bin/env python3
"""Consolidate the allocation-policy and concurrency experiment after validation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import profile_test300 as p

def positive_fit(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    candidates=[np.zeros(2)]
    v=np.linalg.lstsq(a,b,rcond=None)[0]
    if (v>=0).all():candidates.append(v)
    for i in range(2):
        v=np.zeros(2);v[i]=max(0,float(a[:,i]@b)/float(a[:,i]@a[:,i]));candidates.append(v)
    return min(candidates,key=lambda v:float(np.sum((a@v-b)**2)))

def summarize(job):
    root=Path(job['runtime_root']); target=p.HERE/'profiling_jobs'/p.JOB_ID
    p.check(p.read(root/'nohuge_followup_state.json')['status']=='complete','follow-up incomplete')
    modes=[f'{prefix}cpu{w}' for prefix in ('','nohuge_') for w in [4,8,16]]
    comparisons={mode:p.read(root/mode/'complete.json') for mode in modes}
    p.check(all(v['status']=='passed' and v['cases']==16 for v in comparisons.values()),'incomplete comparison')
    gpu=[p.read(x) for x in (root/'gpu').glob('r*_*.nii.gz.json')]
    p.check(len(gpu)==16,'incomplete one-pass inference')
    best_mode=min((x for x in modes if x.startswith('nohuge')),key=lambda x:comparisons[x]['wall_seconds'])
    best=comparisons[best_mode]
    from profile_test300_telemetry import reduce
    telemetry=reduce(root,comparisons)
    p.write(root/'telemetry_summary.json',telemetry)
    stage_sums={k:sum(r['timings'].get(k,0) for r in gpu) for k in ['input_total','embeddings_child','crop_prediction_total','cuda_network','restore_total','orientation_total','finite_range_clip','local_write','local_array_hash','diagnostic_spool']}
    projections={}
    for c in job['candidates']:
        rows=c['selected']; full=job['full_workloads'][str(c['rank'])]
        records={r['name']:r for r in gpu if r['rank']==c['rank']}
        ordered=[records[r['case']['name']]['timings'] for r in rows]
        coef=positive_fit([[r['windows'],r['workload']] for r in rows],[r['cuda_network'] for r in ordered])
        network=float(np.asarray([[r['windows'],r['workload']] for r in full])@coef @ np.ones(len(full)))
        text=float(np.median([x['embeddings_child'] for x in ordered[1:]]))*len(full)
        sliding=[sum(v for k,v in x.items() if k.startswith('predict_sliding_window_') and k.endswith('_child')) for x in ordered]
        overhead=sum(max(0,s-x['cuda_network']) for s,x in zip(sliding,ordered))*sum(r['windows'] for r in full)/sum(r['windows'] for r in rows)
        input_time=sum(x['input_total'] for x in ordered)*sum(r['input_elements'] for r in full)/sum(r['input_elements'] for r in rows)
        # CPU conversion inside the GPU helper was not replayed independently. Keep
        # its observed cost as a conservative sensitivity bound, not a claimed fix.
        helper_cpu=sum(max(0,x['crop_prediction_total']-x['embeddings_child']-s) for s,x in zip(sliding,ordered))
        helper_cpu*=sum(r['crop_elements'] for r in full)/sum(r['crop_elements'] for r in rows)
        projections[str(c['rank'])]={'network_seconds':network,'recurring_text_seconds':text,'sliding_overhead_seconds':overhead,'input_seconds':input_time,'unverified_helper_cpu_sensitivity_seconds':helper_cpu,'gpu_path_lower_seconds':network+text+overhead+input_time,'gpu_path_conservative_seconds':network+text+overhead+input_time+helper_cpu}
    scale=sum(r['output_elements'] for full in job['full_workloads'].values() for r in full)/sum(r['output_elements'] for c in job['candidates'] for r in c['selected'])
    gpu_lower=max(v['gpu_path_lower_seconds'] for v in projections.values())
    gpu_upper=max(v['gpu_path_conservative_seconds'] for v in projections.values())
    restore=best['restore_wall_seconds']*scale
    tail=(best['publication_wall_seconds']+best['dtype_scan_wall'])*scale
    eta=[4*(max(gpu_lower,restore)+tail)/3600*.8,4*(gpu_upper+restore+tail)/3600*1.3]
    result={'status':'passed','parent_job_sha256':job['sha256'],'inference_jobs':16,'inference_passes':1,'comparisons':comparisons,'best_cpu_mode':best_mode,'stage_sums_seconds':stage_sums,'wave2_model_projections':projections,'remaining_waves_hours_range':eta,'eta_assumptions':['Four-case architectural samples proxy Waves 3–5.','Retains measured recurring text-embedding cost; no embedding optimization assumed.','One inference pass: GPU/CPU overlap and helper-allocation improvement are unverified.','Native output bytes scale CPU work; publication is added as a final barrier.','Warm small-sample I/O can understate full-cache storage cost.'],'telemetry':telemetry,'production_resume':False,'completed_at_utc':p.fc.utc_now()}
    lines=['# Test300 pipeline diagnosis and CPU benchmark','', 'All 16 inference jobs and six 16-case CPU comparisons passed. Production Waves 2–5 remain held.','',
           'The demonstrated bottleneck is large-array allocation with NumPy huge-page advice on this host. Kernel compaction, rather than GPU segmentation computation alone, caused long CPU stalls. A 256 MiB allocation improved from 15–16 s to 0.073–0.074 s when advice was disabled locally. The real largest iso07 output preserved its exact native float32 hash while restoration plus orientation fell from 436 s to 25 s.','',
           '| NumPy huge-page advice | CPU workers | Total min | Restore/stage min | Precision scan min | Publish/validate min | Max active | Peak worker GiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,r in comparisons.items():
        lines.append(f"| {'off' if mode.startswith('nohuge') else 'on'} | {r['workers']} | {r['wall_seconds']/60:.2f} | {r['restore_wall_seconds']/60:.2f} | {r['dtype_scan_wall']/60:.2f} | {r['publication_wall_seconds']/60:.2f} | {r['limits']['peak_active']} | {r['peak_worker_rss_bytes']/p.GIB:.2f} |")
    lines+=['',f"Fastest measured CPU replay: **{best['workers']} workers with `NUMPY_MADVISE_HUGEPAGE=0`**. The maximum-memory scheduler retains 64 GiB available RAM. Each worker uses one numerical thread. Warm-input preparation is excluded from these wall times and recorded separately.",'',
            'For a production continuation, carry the process-local allocation setting into both GPU workers and CPU workers, then use a bounded continuous CPU queue rather than waiting for 16 GPU outputs. Keep all inference/geometry/storage checks. The replay measures the CPU pool; it does not prove a full-pipeline overlap speedup. Prompt embedding generation and recurring CPU/GPU backbone transfers are another substantial cost and should be evaluated separately before changing their precision or cache contract.','',
            '| Reference stage (summed across 16 jobs) | Minutes |','|---|---:|']
    for k in ['input_total','embeddings_child','cuda_network','restore_total','orientation_total','finite_range_clip','local_write','local_array_hash','diagnostic_spool']:
        lines.append(f'| {k} | {stage_sums[k]/60:.2f} |')
    lines+=['','Network and embedding measurements are child intervals inside crop prediction; do not add them to their parent. Diagnostic spool time is extra benchmark work. Shared publication and precision scans are in the per-case CSV and comparison records.','',
            f'Conditional planning envelope for Waves 2–5: **{eta[0]:.1f}–{eta[1]:.1f} hours** with the observed text-embedding cost. This retains the measured unfixed GPU/text phases. It is neither a measured runtime for the modified GPU path nor a statistical confidence interval. Replace it using retained GPU smoke timings before treating it as a production ETA.','',*['- '+x for x in result['eta_assumptions']],'',
            'Validated contracts: native float32 hash and CT geometry parity, numeric finding ordering, per-checkpoint sample dtype agreement, finite/clipped values, exact threshold-zero masks after publication, atomic writes and hashes. No labels or Dice, no extra model inference, no production cache publication, no global THP changes. All sampled checkpoints permitted float16; full-cohort fallback remains undecided.','',
            'Reproduction: `python side_experiments/sideexp003_ensemble_method_hub/profile_test300.py prepare`, then `run`; arm `profile_test300_nohuge.py run` for the CPU-only follow-up and use `profile_test300_report.py --wait` for this report. Existing job IDs are immutable; inspect preserved records before any recovery.']
    lines+=['','## Recorded system evidence','','| Mode | Kernel share of worker CPU | Host I/O wait | Peak summed RSS GiB |','|---|---:|---:|---:|']
    for mode,v in telemetry['cpu_modes'].items():
        lines.append(f"| {mode} | {v['process_kernel_cpu_percent']:.1f}% | {v['host_iowait_percent']:.1f}% | {v['observed_process_rss_sum_peak_bytes']/p.GIB:.1f} |")
    lines+=['','GPU zero-utilization fractions during measured case intervals (including diagnostic spooling): '+', '.join(f"rank {rank}: {v['sampled_zero_utilization_percent']:.1f}%" for rank,v in telemetry['gpu_case_activity'].items())+'.','',
            'The replay driver serializes the four per-checkpoint precision scans; this is a measured limit of this harness, not a claim that the original four GPU workers scanned serially. Keep per-checkpoint dtype agreement while distributing these scans in a future coordinator. Dual-branch inference also converts the unused proposal output to CPU before selecting the final branch; removing that transfer is a separate, untested optimization.',
            '','Each concurrency setting was run once, in fixed order. Allocation history, NUMA placement and shared I/O can affect the ordering. The repeated synthetic on/off control and exact-hash real-case replay provide the causal allocation evidence.', '',*['- '+x for x in telemetry['limitations']]]
    p.write(root/'pipeline_analysis.json',result)
    (root/'pipeline_report.md').write_text('\n'.join(lines)+'\n')
    with (root/'pipeline_case_timings.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['mode','rank','case','stage','value','unit'])
        for mode,records in [('gpu_reference',gpu)]+[(mode,[p.read(path) for folder in ['timings','publication_timings'] for path in (root/mode/folder).glob('*.json')]) for mode in ['reference']+modes]:
            for r in records:
                for key,value in r['timings'].items():w.writerow([mode,r['rank'],r['name'],key,value,'count' if key=='network_forward_calls' else 'seconds'])
    for name in ['pipeline_analysis.json','pipeline_report.md','pipeline_case_timings.csv','telemetry_summary.json']:
        (target/name).write_bytes((root/name).read_bytes())
    print(json.dumps({'best':best_mode,'remaining_waves_hours_range':eta,'report':str(target/'pipeline_report.md')}))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait',action='store_true');ap.add_argument('--job',type=Path,default=p.DEFAULT_JOB);a=ap.parse_args();j=p.read(a.job);root=Path(j['runtime_root'])
    while a.wait:
        state=p.read(root/'nohuge_followup_state.json')
        p.check(state['status']!='failed','CPU follow-up failed; inspect logs')
        if state['status']=='complete':break
        time.sleep(10)
    summarize(j)
if __name__=='__main__':main()
