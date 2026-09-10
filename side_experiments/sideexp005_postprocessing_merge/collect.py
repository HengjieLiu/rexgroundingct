#!/usr/bin/env python3
"""Publish verified SideExp005 aggregates after its detached supervisor exits."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

from runner import atomic_json, digest, job_digest, job_lock, pin, read, require, sha, utc

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
VARIANTS = ('d1', 'd2', 'd3', 'd11', 'd12')


def atomic_bytes(path, content):
    temp = path.with_name('.' + path.name + '.collect.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def verify_completion(cfg, runtime):
    job = read(runtime / 'job_manifest.json')
    require(job_digest(job) == job['job_sha256'] and digest(cfg) == job['config_sha256'], 'frozen job/config mismatch')
    done = read(runtime / 'completion.json')
    require(done['status'] == 'complete' and done['job_sha256'] == job['job_sha256']
            and done['files'] == 1000 and done['finding_evaluations'] == 1905, 'incomplete or foreign completion')
    require(set(done['metrics']) == set(VARIANTS), 'missing variant metrics')
    names = {c['name'] for c in job['cases']}
    for variant in VARIANTS:
        path = runtime / 'manifests' / (variant + '.json')
        require(sha(path) == done['inventory_hashes'][variant], 'inventory hash drift')
        inv = read(path)
        require(inv['cases'] == len(inv['files']) == 200 and inv['findings'] == 381, 'inventory coverage')
        require({Path(r['path']).name for r in inv['files']} == names, 'inventory cohort drift')
        require(sum(r['shape'][0] for r in inv['files']) == 381, 'inventory finding count')
        for record in inv['files']:
            require(Path(record['path']).parent == runtime / 'predictions' / variant, 'foreign artifact path')
            pin(record)
    report = runtime / 'reports/report.md'
    require(sha(report) == done['report_sha256'], 'aggregate report hash drift')
    summary = read(runtime / 'reports/summary.json')
    require(summary['job_sha256'] == job['job_sha256'] and summary['metrics'] == done['metrics']
            and summary['changes'] == done['changes'], 'summary/completion disagreement')
    rows = read(runtime / 'reports/per_finding.json')
    require(len(rows) == len({(r['variant'], r['case'], r['finding_index']) for r in rows}) == 1905, 'finding-row coverage')
    for variant in VARIANTS:
        selected = [r for r in rows if r['variant'] == variant]
        require(len(selected) == 381 and {r['case'] for r in selected} == names, 'metric cohort coverage')
        case_scores = collections.defaultdict(list)
        for row in selected:
            g, p, i = row['gt_voxels'], row['pred_voxels'], row['intersection_voxels']
            expected = 1.0 if g + p == 0 else (2 * i + 1e-6) / (g + p + 1e-6)
            require(abs(expected - row['dice']) <= 1e-12 and row['hit'] == (expected >= .1), 'finding metric/count disagreement')
            case_scores[row['case']].append(expected)
        metrics = done['metrics'][variant]
        require(metrics['cases'] == 200 and metrics['findings'] == 381
                and abs(metrics['dice'] - math.fsum(r['dice'] for r in selected) / 381) <= 1e-12
                and metrics['hits'] == sum(r['hit'] for r in selected), 'overall finding aggregation mismatch')
        case_mean = math.fsum(math.fsum(v) / len(v) for v in case_scores.values()) / 200
        require(abs(metrics['case_wise_dice'] - case_mean) <= 1e-12, 'case-wise aggregation mismatch')
    return done, summary


def publish(cfg, runtime):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    require(branch == 'gpu8', 'repository must remain on gpu8 for aggregate publication')
    done, summary = verify_completion(cfg, runtime)
    for dest_name, source_name in [('report.md', 'report.md'), ('metrics_summary.json', 'summary.json')]:
        dest, source = ROOT / dest_name, runtime / 'reports' / source_name
        if dest.exists():
            require(dest.read_bytes() == source.read_bytes(), 'conflicting existing aggregate: ' + dest_name)
        else:
            atomic_bytes(dest, source.read_bytes())
    spec = importlib.util.spec_from_file_location('submission_manage', REPO / 'submissions/manage.py')
    manage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manage)
    registry_path = REPO / 'submissions/registry.json'
    old_hash = sha(registry_path)
    registry = read(registry_path)
    for row in registry['submissions']:
        if row['id'] in VARIANTS:
            row['validation'].update(status='complete', job_sha256=done['job_sha256'],
                                     report_path=str(ROOT / 'report.md'), runtime_report_path=str(runtime / 'reports/report.md'),
                                     summary_sha256=sha(runtime / 'reports/summary.json'), metrics=done['metrics'][row['id']])
    manage.validate(registry)
    require(sha(registry_path) == old_hash, 'registry edited during collection; retry to preserve manual history')
    manage.write(registry_path, registry)
    manage.atomic_text(REPO / 'submissions/REGISTER.md', manage.render(registry))
    current = REPO / 'docs/current_status.md'
    old_hash = sha(current)
    text = current.read_text()
    marker = '**SideExp005 postprocessing audit running:**'
    replacement = '**SideExp005 postprocessing audit complete:**'
    if marker in text:
        metrics = '; '.join(f"{v} Dice {done['metrics'][v]['dice']:.6f}, {done['metrics'][v]['hits']}/381 hits" for v in VARIANTS)
        text = text.replace(marker, replacement + '\n  Verified all 1000 outputs / 1905 finding evaluations. ' + metrics + '.\n  Aggregate report: `side_experiments/sideexp005_postprocessing_merge/report.md`.\n ', 1)
        require(sha(current) == old_hash, 'current status edited during collection; retry')
        atomic_bytes(current, text.encode())
    index = REPO / 'side_experiments/README.md'
    old_hash = sha(index)
    text = index.read_text()
    marker = '  strict v2 is applied to d11 to produce d12. Test d11/d12 remain pending review.'
    if marker in text and 'sideexp005_postprocessing_merge/report.md' not in text:
        text = text.replace(marker, marker + '\n  [Completed val200 report](sideexp005_postprocessing_merge/report.md): 200 CTs / 381 findings, five verified variants.')
        require(sha(index) == old_hash, 'side-experiment index edited during collection; retry')
        atomic_bytes(index, text.encode())
    record = {'status': 'complete', 'job_sha256': done['job_sha256'], 'collected_at_utc': utc(),
              'report_sha256': sha(ROOT / 'report.md'), 'summary_sha256': sha(ROOT / 'metrics_summary.json'),
              'files_verified': 1000, 'finding_evaluations_verified': 1905,
              'test_generation': 'pending_validation_review', 'submission_history_changed': False}
    atomic_json(ROOT / 'closeout.json', record)
    atomic_json(runtime / 'closeout_state.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cfg = read(ROOT / 'config.json')
    runtime = Path(cfg['runtime_root'])
    source_hash = sha(Path(__file__))
    launch = read(runtime / 'launch.json')
    with job_lock(runtime / 'closeout'):
        atomic_json(runtime / 'closeout_source.json', {'path': str(Path(__file__)), 'sha256': source_hash,
                    'job_sha256': read(runtime / 'job_manifest.json')['job_sha256']})
        while True:
            require(sha(Path(__file__)) == source_hash, 'collector source changed while waiting')
            state = read(runtime / 'state.json') if (runtime / 'state.json').exists() else {}
            if state.get('status') in ('failed', 'interrupted'):
                atomic_json(runtime / 'closeout_state.json', {'status': 'blocked_by_run', 'run_status': state['status'], 'updated_at_utc': utc()})
                raise SystemExit('Audit did not complete; preserved all outputs and reports.')
            if state.get('status') == 'complete':
                container = json.loads(subprocess.check_output(['docker', 'inspect', launch['container_id']], text=True))[0]
                if not container['State']['Running']:
                    require(container['State']['ExitCode'] == 0, 'supervisor exited unsuccessfully')
                    print(json.dumps(publish(cfg, runtime)), flush=True)
                    return
            if not args.watch:
                raise SystemExit('Audit is still running; use --watch for automatic collection.')
            atomic_json(runtime / 'closeout_state.json', {'status': 'waiting_for_completion', 'updated_at_utc': utc(), 'run_phase': state.get('phase')})
            time.sleep(30)


if __name__ == '__main__':
    main()
