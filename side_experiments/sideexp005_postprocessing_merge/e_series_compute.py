#!/usr/bin/env python3
"""Independent eight-model computation; frozen d numerical sources stay unchanged."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import runner as core
from runner import atomic_json, digest, job_digest, job_lock, pin, read, require, sha, utc

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
VARIANTS = ('e1', 'e2', 'e3', 'e11', 'e12')
CFG = JOB = None
STOP = False


def scientific():
    global np, nib, methods
    core.scientific()
    np, nib, methods = core.np, core.nib, core.methods


def stage_config(master, stage):
    return {**master, **master['runs'][stage], 'stage': stage,
            'expected': {'cases': 200, 'findings': 381} if stage == 'val200' else {'cases': 300, 'findings': 582},
            'output_root': str(Path(master['runs'][stage]['runtime_root'])/'predictions') if stage == 'val200' else master['test_output_root']}


def enough_space(cfg, free=None):
    if free is None:
        free = shutil.disk_usage(cfg['runtime_root']).free
    return free >= cfg['reserve_bytes'] + cfg['headroom_bytes']


def state(cfg, job, status, **kw):
    value = {'status': status, 'stage': cfg['stage'], 'run_id': cfg['run_id'], 'job_sha256': job['job_sha256'],
             'workers': cfg['workers'], 'cases_total': cfg['expected']['cases'], 'updated_at_utc': utc(), **kw}
    atomic_json(Path(cfg['runtime_root'])/'state.json', value)
    print(json.dumps(value), flush=True)
    return value


def verify_job(cfg, job):
    require(job_digest(job) == job['job_sha256'] and digest(cfg) == job['config_sha256'], 'e job/config drift')
    for ref in cfg['inputs'].values():
        # The test container deliberately cannot access validation segmentation files.
        pin(ref)
    for ref in job['code']:
        require(sha(REPO/ref['relative_path']) == ref['sha256'], 'e frozen implementation drift')
    for c in job['candidates']:
        for key in ('export_manifest', 'validation'):
            pin(c['val_refs'][key])


def candidate_bindings(cfg):
    roster = read(cfg['inputs']['val_roster']['path'])
    parent = read(cfg['inputs']['parent_test_job']['path'])
    wave2 = read(cfg['inputs']['wave2_job']['path'])
    selected = parent['candidates'][:4] + wave2['candidates'][:4]
    require([c['rank'] for c in selected] == list(range(1, 9)), 'rank drift')
    require([c['id'] for c in selected] == [c['id'] for c in roster['candidates']]
            == read(cfg['inputs']['baseline_result']['path'])['candidate_ids'], 'eight-model identity/order mismatch')
    out = []
    for v, t in zip(roster['candidates'], selected):
        require(v['checkpoint'] == t['checkpoint'] and v['cache_key'] == t['paired_val200']['cache_key'], 'val/test checkpoint pairing drift')
        refs = {'export_manifest': {'path': v['export_manifest_path'], 'sha256': v['export_manifest_sha256']},
                'validation': {'path': v['validation_path'], 'sha256': v['validation_sha256']}}
        for ref in refs.values(): pin(ref)
        validation = read(refs['validation']['path'])
        require(validation['status'] == validation['storage_reproduction_status'] == 'passed'
                and validation['array_hashes_verified'] and validation['same_pass_mask_mismatch_voxels'] == 0
                and validation['cases'] == 200 and validation['findings'] == 381, 'incomplete val cache')
        out.append({'id': t['id'], 'rank': t['rank'], 'checkpoint': t['checkpoint'], 'val_refs': refs,
                    'test_source': t, 'test_job_sha256': parent['job_spec_sha256'] if t['rank'] <= 4 else wave2['job_spec_sha256']})
    return out


def geometry(case):
    ct, anatomy = nib.load(case['ct_path']), nib.load(case['anatomy']['path'])
    require(list(ct.shape) == case['shape'][1:] == list(anatomy.shape), 'CT/anatomy shape drift')
    require(np.allclose(ct.affine, case['affine'], atol=1e-5, rtol=0)
            and np.allclose(anatomy.affine, ct.affine, atol=1e-5, rtol=0), 'CT/anatomy affine drift')
    if 'ct_header_sha256' in case:
        require(hashlib.sha256(ct.header.binaryblock).hexdigest() == case['ct_header_sha256'], 'CT header drift')
    methods.world_axis_info(ct.affine, ct.shape)
    return ct


def preflight(cfg):
    scientific()
    runtime = Path(cfg['runtime_root'])
    with job_lock(runtime):
        for ref in cfg['inputs'].values(): pin(ref)
        candidates = candidate_bindings(cfg)
        parent_job = read(cfg['inputs']['d_job' if cfg['stage'] == 'val200' else 'd_test_job']['path'])
        require(job_digest(parent_job) == parent_job['job_sha256'], 'parent geometry job drift')
        dataset = read(cfg['inputs']['val_dataset' if cfg['stage'] == 'val200' else 'test_dataset']['path'])['test']
        originals = {c['name']: c for c in dataset}
        require(len(originals) == len(dataset) == cfg['expected']['cases'], 'split case coverage')
        routing = read(cfg['inputs']['val_routes' if cfg['stage'] == 'val200' else 'test_routes']['path'])
        require(routing['split'] == ('val' if cfg['stage'] == 'val200' else 'test'), 'wrong routing split')
        routes = {(r['case'], r['finding_id']): r for r in routing['rows']}
        require(len(routes) == len(routing['rows']) == cfg['expected']['findings'], 'routing coverage')
        audit = read(cfg['inputs']['val_anatomy_audit' if cfg['stage'] == 'val200' else 'test_anatomy_audit']['path'])
        audited = {r['volume_name']: r for r in audit['cases']}
        require(audit['lut_status'] == 'PASS' and set(audited) == set(originals), 'anatomy audit coverage')
        exports = []
        if cfg['stage'] == 'val200':
            for c in candidates:
                export = read(c['val_refs']['export_manifest']['path'])
                require(export['candidate_id'] == c['id'] and export['candidate_source']['checkpoint'] == c['checkpoint'], 'val manifest checkpoint drift')
                rows = {r['name']: r for r in export['cases']}
                require(len(rows) == len(export['cases']) == 200 and set(rows) == set(originals), 'val cache cohort drift')
                exports.append(rows)
        cases, census = [], collections.Counter()
        for number, prior in enumerate(parent_job['cases'], 1):
            case = copy.deepcopy(prior)
            case.pop('input', None); case.pop('sources', None)
            name = case['name']
            require(Path(name).name == name and name.endswith('.nii.gz'), 'unsafe case name')
            require(case['findings'] == methods.ordered_findings(originals[name]), 'finding/prompt ordering mismatch')
            audit_case = audited[name]
            require(audit_case['geometry']['index_grid_status'] == audit_case['geometry']['world_header_status'] == 'PASS'
                    and audit_case['integrity']['status'] == 'VALID' and audit_case['source_provenance']['status'] == 'FULL_HASH_MATCH', 'anatomy audit failed')
            require(case['anatomy'] == {'path': audit_case['integrity']['path'], 'sha256': audit_case['integrity']['observed_sha256']}, 'anatomy binding drift')
            require(set(range(10, 15)).issubset(audit_case['labels']['unique_label_ids']), 'missing official lung lobes')
            pin(case['anatomy']); ct = geometry(case)
            case['ct_header_sha256'] = hashlib.sha256(ct.header.binaryblock).hexdigest()
            case['routes'] = []
            for f in case['findings']:
                route = routes[(name, f['finding_idx'])]
                expected = methods.route_prompt(f['prompt'], f['category'])
                require(all(route[k] == v for k, v in expected.items()), 'frozen prompt route drift')
                case['routes'].append(route)
                parsed = methods.parse_finding(f['prompt'])
                census[parsed['roi_kind']] += 1; census['v2_selected'] += int(parsed['v2_selected'])
            if cfg['stage'] == 'val200':
                pin(case['gt'])
                require(list(nib.load(case['gt']['path']).shape) == case['shape'], 'GT native layout mismatch')
                case['sources'] = []
                for rows in exports:
                    rec = rows[name]
                    require(rec['shape'] == case['shape'] and rec['status'] == 'complete'
                            and rec['orientation']['applied_transform'] == 'nnunet_FZYX_to_reoriented_FXYZ_then_original_ct_orientation', 'val cache geometry/contract drift')
                    array = np.load(rec['array_path'], mmap_mode='r', allow_pickle=False)
                    require(list(array.shape) == case['shape'] and array.dtype.name == rec['dtype']
                            and array.dtype.name in ('float16', 'float32') and array.flags.c_contiguous, 'val array metadata drift')
                    case['sources'].append({'path': rec['array_path'], 'sha256': rec['array_sha256'], 'dtype': rec['dtype']})
                    del array
            else:
                case.pop('gt', None)
            cases.append(case)
            if number % 25 == 0: print(f"{cfg['stage']} preflight: {number}/{cfg['expected']['cases']}", flush=True)
        require(len(cases) == len({c['name'] for c in cases}) == cfg['expected']['cases']
                and sum(len(c['findings']) for c in cases) == cfg['expected']['findings'], 'frozen case coverage')
        if cfg['stage'] == 'val200':
            require(dict(census) == {'lobe': 110, 'side': 67, 'whole': 204, 'v2_selected': 49}, 'val routing census drift')
        job = {'run_id': cfg['run_id'], 'stage': cfg['stage'], 'config_sha256': digest(cfg),
               'candidates': candidates, 'cases': cases, 'routing_counts': dict(census),
               'code': read(runtime/'source_manifest.json')['files'], 'image_id': cfg['image_id'],
               'smoke_cases': list(dict.fromkeys([max(cases, key=lambda c: math.prod(c['shape'][1:]))['name'],
                                                 max(cases, key=lambda c: math.prod(c['shape']))['name']]))}
        job['job_sha256'] = job_digest(job)
        path = runtime/'job_manifest.json'
        if path.exists(): require(read(path) == job, 'incompatible existing e run')
        verify_job(cfg, job)
        for v in VARIANTS:
            for p in (Path(cfg['output_root'])/v).glob('*.nii.gz'):
                require((runtime/'cases'/('baseline' if v == 'e1' and cfg['stage'] == 'val200' else 'postprocessing')/(p.name+'.json')).exists(), 'unowned existing e prediction')
        atomic_json(path, job)
        atomic_json(runtime/'preflight.json', {'status': 'passed', 'job_sha256': job['job_sha256'],
                    **cfg['expected'], 'routing_counts': dict(census), 'space_available': enough_space(cfg), 'updated_at_utc': utc()})
        return job


def average_arrays(arrays, hashes=None, chunk=4194304):
    require(len(arrays) == 8, 'eight source arrays required')
    shape = arrays[0].shape
    require(all(a.shape == shape and a.flags.c_contiguous and a.dtype.name in ('float16', 'float32') for a in arrays), 'array shape/dtype mismatch')
    out = np.empty(shape, dtype=np.uint8)
    seen = [hashlib.sha256() for _ in arrays]
    timing = {'reading_hash_seconds': 0., 'averaging_seconds': 0.}
    for lo in range(0, out.size, chunk):
        hi = min(lo+chunk, out.size); acc = np.zeros(hi-lo, dtype=np.float32)
        for array, h in zip(arrays, seen):
            begin = time.monotonic()
            values = np.array(array.reshape(-1)[lo:hi], copy=True)
            h.update(memoryview(values).cast('B'))
            require(np.isfinite(values).all(), 'nonfinite source logits')
            timing['reading_hash_seconds'] += time.monotonic()-begin
            begin = time.monotonic(); acc += methods.sigmoid_float32(values)
            timing['averaging_seconds'] += time.monotonic()-begin
        out.reshape(-1)[lo:hi] = acc >= np.float32(4.)
    if hashes is not None: require([h.hexdigest() for h in seen] == hashes, 'source array hash mismatch')
    return out, timing


def artifact_paths(cfg, case, variant):
    dest = Path(cfg['output_root'])/variant/case['name']
    return dest, dest.with_name('.'+cfg['run_id']+'.'+case['name'])


def recover(cfg, job, case, phase):
    path = Path(cfg['runtime_root'])/'cases'/phase/(case['name']+'.json')
    variants = ('e1',) if phase == 'baseline' else VARIANTS[1:] if cfg['stage'] == 'val200' else VARIANTS
    if not path.exists():
        require(all(not artifact_paths(cfg, case, v)[0].exists() for v in variants), 'unowned existing prediction')
        return None
    rec = read(path)
    require(rec['job_sha256'] == job['job_sha256'] and rec['case'] == case['name'] and rec['phase'] == phase
            and rec['status'] in ('prepared', 'complete') and set(rec['artifacts']) == set(variants), 'foreign case journal')
    for v in variants:
        dest, temp = artifact_paths(cfg, case, v); a = rec['artifacts'][v]
        require(a['path'] == str(dest) and a['temporary_path'] == str(temp), 'foreign artifact path')
        source = dest if dest.exists() else temp
        require(source.exists() and sha(source) == a['sha256'], 'output missing/hash drift')
        core.validate_nifti(source, case)
        if source == temp: os.replace(temp, dest)
        elif temp.exists():
            require(sha(temp) == a['sha256'], 'conflicting temporary file'); temp.unlink()
    if rec['status'] != 'complete':
        rec.update(status='complete', completed_at_utc=utc()); atomic_json(path, rec)
    return rec


def worker_init(cfg, job):
    global CFG, JOB
    CFG, JOB = cfg, job
    scientific()


def case_worker(task):
    phase, case = task
    old = recover(CFG, JOB, case, phase)
    if old: return old
    start = time.monotonic(); t = start
    timing = {k: 0. for k in ('reading_hash_seconds', 'averaging_seconds', 'anatomy_seconds', 'writing_seconds', 'scoring_seconds', 'verification_seconds')}
    ct = geometry(case)
    if phase == 'baseline' or CFG['stage'] == 'test300':
        arrays = [np.load(r['path'], mmap_mode='r', allow_pickle=False) for r in case['sources']]
        require(all(list(a.shape) == case['shape'] and a.dtype.name == r['dtype'] for a, r in zip(arrays, case['sources'])), 'logit metadata drift')
        timing['reading_hash_seconds'] += time.monotonic()-t
        raw, times = average_arrays(arrays, [r['sha256'] for r in case['sources']], CFG['chunk_elements'])
        for k, value in times.items(): timing[k] += value
        del arrays
    else:
        baseline = recover(CFG, JOB, case, 'baseline'); require(baseline is not None, 'baseline missing')
        raw = np.asanyarray(nib.load(baseline['artifacts']['e1']['path']).dataobj)
        timing['reading_hash_seconds'] += time.monotonic()-t
    outputs, routing = {'e1': raw}, []
    if phase != 'baseline':
        t = time.monotonic(); pin(case['anatomy'])
        anatomy = np.asanyarray(nib.load(case['anatomy']['path']).dataobj)
        timing['reading_hash_seconds'] += time.monotonic()-t
        t = time.monotonic()
        processed, routing = methods.postprocess(raw, anatomy, ct.affine, case['findings'], case['routes'])
        outputs = {'e'+k[1:]: arr for k, arr in processed.items()}
        if CFG['stage'] == 'test300': outputs = {'e1': raw, **outputs}
        timing['anatomy_seconds'] += time.monotonic()-t
        t = time.monotonic()
        require(all(np.all(arr <= raw) for arr in outputs.values()) and np.all(outputs['e12'] <= outputs['e11']), 'postprocessing subset failure')
        for route in routing:
            if not route['collaborator']['v2_selected']:
                idx = route['finding_idx']; require(np.array_equal(outputs['e11'][idx], outputs['e12'][idx]), 'v2 bypass changed')
        timing['verification_seconds'] += time.monotonic()-t
        del anatomy
    rows = []
    if CFG['stage'] == 'val200':
        t = time.monotonic(); pin(case['gt']); gt = np.asanyarray(nib.load(case['gt']['path']).dataobj)
        timing['reading_hash_seconds'] += time.monotonic()-t
        t = time.monotonic(); rows = [r for v, arr in outputs.items() for r in core.score_case(arr, gt, case, v)]
        timing['scoring_seconds'] += time.monotonic()-t; del gt
    else:
        rows = [{'case': case['name'], 'finding_index': f['finding_idx'], 'prompt': f['prompt'], 'category': f['category'],
                 'foreground_voxels': {v: int(arr[f['finding_idx']].sum()) for v, arr in outputs.items()}} for f in case['findings']]
    artifacts = {}
    for v, arr in outputs.items():
        dest, temp = artifact_paths(CFG, case, v); require(not dest.exists(), 'refusing prediction overwrite')
        dest.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic(); header = ct.header.copy(); header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(arr, ct.affine, header)
        for getter, setter in [('get_qform', 'set_qform'), ('get_sform', 'set_sform')]:
            matrix, code = getattr(ct, getter)(coded=True); getattr(image, setter)(matrix, int(code))
        nib.save(image, str(temp))
        with temp.open('rb') as stream: os.fsync(stream.fileno())
        timing['writing_seconds'] += time.monotonic()-t
        t = time.monotonic(); checked = core.validate_nifti(temp, case, expected=arr)
        timing['verification_seconds'] += time.monotonic()-t
        artifacts[v] = {**checked, 'path': str(dest), 'temporary_path': str(temp), 'variant': v}
    timing['wall_seconds'] = time.monotonic()-start
    record = {'status': 'prepared', 'job_sha256': JOB['job_sha256'], 'case': case['name'], 'phase': phase,
              'artifacts': artifacts, 'rows': rows, 'routing': routing, 'timings': timing,
              'elements': math.prod(case['shape']), 'findings': len(case['findings'])}
    path = Path(CFG['runtime_root'])/'cases'/phase/(case['name']+'.json'); atomic_json(path, record)
    for a in artifacts.values(): os.replace(a['temporary_path'], a['path'])
    record.update(status='complete', completed_at_utc=utc()); atomic_json(path, record)
    return record


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def run_phase(cfg, job, phase):
    runtime = Path(cfg['runtime_root']); records, queue = [], []
    for c in job['cases']:
        existing = recover(cfg, job, c, phase)
        (records if existing else queue).append(existing if existing else c)
    started, last = time.monotonic(), 0.
    timing_path = runtime/(phase+'_timings.json')
    prior = read(timing_path).get('wall_seconds', 0) if timing_path.exists() else 0
    fresh = []
    def progress(status='running'):
        nonlocal last
        elapsed = time.monotonic()-started
        done = sum(r['elements'] for r in fresh)
        left = sum(math.prod(c['shape']) for c in job['cases'])-sum(r['elements'] for r in records)
        eta = elapsed*left/done if len(fresh) >= 10 else None
        state(cfg, job, status, phase=phase, cases_complete=len(records), findings_complete=sum(r['findings'] for r in records),
              eta_seconds=eta, wall_seconds=prior+elapsed, shared_free_bytes=shutil.disk_usage(runtime).free)
        atomic_json(timing_path, {'status': status, 'wall_seconds': prior+elapsed, 'updated_at_utc': utc()}); last = time.monotonic()
    groups = [[c for c in queue if c['name'] in job['smoke_cases']], [c for c in queue if c['name'] not in job['smoke_cases']]]
    with cf.ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=mp.get_context('spawn'), initializer=worker_init, initargs=(cfg, job)) as pool:
        for idx, group in enumerate(groups):
            queue, pending = collections.deque(group), set()
            while queue or pending:
                space = enough_space(cfg)
                while queue and len(pending) < cfg['workers'] and not STOP and space:
                    pending.add(pool.submit(case_worker, (phase, queue.popleft()))); space = enough_space(cfg)
                if time.monotonic()-last >= cfg['heartbeat_seconds']: progress('running' if space or not queue else 'waiting_for_space')
                if pending:
                    done, pending = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        record = future.result(); records.append(record); fresh.append(record)
                elif STOP: raise InterruptedError('owned dispatch stopped; outputs preserved')
                elif queue: time.sleep(30)
            if STOP: raise InterruptedError('owned dispatch stopped; outputs preserved')
            if idx == 0:
                atomic_json(runtime/(phase+'_smoke.json'), {'status': 'passed', 'job_sha256': job['job_sha256'], 'cases': job['smoke_cases']})
    if fresh or not timing_path.exists(): progress('complete')
    return records


def paired_changes(rows, variant, baseline, subset=False):
    if subset: return core.compare_rows(rows, variant, baseline)
    indexed = {(r['variant'], r['case'], r['finding_index']): r for r in rows}
    selected = []
    for r in rows:
        if r['variant'] != variant: continue
        b = indexed[(baseline, r['case'], r['finding_index'])]
        require(r['prompt'] == b['prompt'] and r['category'] == b['category'] and r['gt_voxels'] == b['gt_voxels'], 'paired finding identity/GT mismatch')
        delta = r['dice']-b['dice']
        selected.append({'variant': variant, 'baseline': baseline, 'case': r['case'], 'finding_index': r['finding_index'],
                         'category': r['category'], 'prompt': r['prompt'], 'delta_dice': delta,
                         'change': 'improved' if delta > 1e-12 else 'decreased' if delta < -1e-12 else 'unchanged',
                         'hit_gain': int(r['hit'] and not b['hit']), 'hit_loss': int(b['hit'] and not r['hit'])})
    def aggregate(values):
        return {'findings': len(values), 'mean_delta_dice': math.fsum(r['delta_dice'] for r in values)/len(values) if values else None,
                **{k: sum(r['change'] == k for r in values) for k in ('improved', 'decreased', 'unchanged')},
                **{k: sum(r[k] for r in values) for k in ('hit_gain', 'hit_loss')}}
    return {'overall': aggregate(selected), 'categories': {c: aggregate([r for r in selected if r['category'] == c]) for c in methods.CATEGORIES}}, selected


def report(cfg, job):
    scientific(); verify_job(cfg, job)
    runtime = Path(cfg['runtime_root'])
    phases = ('baseline', 'postprocessing') if cfg['stage'] == 'val200' else ('postprocessing',)
    records = [recover(cfg, job, c, p) for p in phases for c in job['cases']]
    require(all(records), 'incomplete e prediction sets')
    inventories = {v: [] for v in VARIANTS}
    for rec in records:
        for v, artifact in rec['artifacts'].items(): inventories[v].append(artifact)
    outputs = {}
    for v, files in inventories.items():
        directory = Path(cfg['output_root'])/v; names = {c['name'] for c in job['cases']}
        require(len(files) == len({Path(f['path']).name for f in files}) == cfg['expected']['cases']
                and {p.name for p in directory.iterdir()} == names, 'e output coverage')
        path = runtime/'manifests'/(v+'.json')
        atomic_json(path, {**cfg['expected'], 'files': files, 'job_sha256': job['job_sha256'], 'variant': v})
        outputs[v] = {'directory': str(directory), 'verification': str(path), 'verification_sha256': sha(path), 'bytes': sum(r['bytes'] for r in files)}
    rows = [dict(r) for record in records for r in record['rows']]
    routes = {(rec['case'], r['finding_idx']): r for rec in records for r in rec['routing']}
    for r in rows:
        route = routes.get((r['case'], r['finding_index']))
        r['routing'] = {'method': 'raw_probability_average'} if r.get('variant') == 'e1' else route
    reports = runtime/'reports'; atomic_json(reports/'per_finding.json', rows)
    timing = {k: math.fsum(r['timings'][k] for r in records) for k in records[0]['timings']}
    metrics, comparisons, changes = None, {}, []
    lines = ['# SideExp005: '+('d/e val200 comparison' if cfg['stage'] == 'val200' else 'e-series test300 outputs'), '',
             'Eight frozen checkpoints, equal sigmoid-probability averaging. e2/e3/e11 derive from e1; e12 derives from e11.',
             'Anatomy review status: '+cfg['source_status']+'.', 'No ZIPs or uploads are created.', '']
    if cfg['stage'] == 'val200':
        require(len(rows) == len({(r['variant'], r['case'], r['finding_index']) for r in rows}) == 1905, 'e finding evaluation coverage')
        baseline_cfg = {**cfg, 'inputs': {'baseline_result': cfg['inputs']['baseline_result']}}
        core.baseline_gate(baseline_cfg, [r for r in records if r['phase'] == 'baseline'])
        drows = read(cfg['inputs']['d_rows']['path']); dmetrics = read(cfg['inputs']['d_completion']['path'])['metrics']
        require(len(drows) == len({(r['variant'], r['case'], r['finding_index']) for r in drows}) == 1905, 'd evidence coverage')
        for v in methods.VARIANTS: require(core.metric_summary([r for r in drows if r['variant'] == v]) == dmetrics[v], 'd metric evidence drift')
        metrics = {**dmetrics, **{v: core.metric_summary([r for r in rows if r['variant'] == v]) for v in VARIANTS}}
        variants = (*methods.VARIANTS, *VARIANTS)
        fmt = lambda x: 'NA' if x is None else f'{x:.9f}'
        lines += ['200 CTs / 381 findings. Full-val diagnostic; local metrics are separate from leaderboard results.', '',
                  '| Variant | Finding-wise Dice | Case-wise Dice | HITs / 381 | HIT rate |', '| --- | ---: | ---: | ---: | ---: |']
        for v in variants:
            m = metrics[v]; lines.append(f"| {v} | {fmt(m['dice'])} | {fmt(m['case_wise_dice'])} | {m['hits']} | {fmt(m['hit_rate'])} |")
        lines += ['', '## All official categories', '', '| Category | Variant | n | Dice | HITs | HIT rate |', '| --- | --- | ---: | ---: | ---: | ---: |']
        for c in methods.CATEGORIES:
            for v in variants:
                m = metrics[v]['categories'][c]
                lines.append(f"| {c} — {methods.CATEGORY_NAMES[c]} | {v} | {m['findings']} | {fmt(m['dice'])} | {m['hits']} | {fmt(m['hit_rate'])} |")
        pairs = [(v, 'e1', True) for v in VARIANTS[1:]] + [('e12', 'e11', True)] + [(v, 'd'+v[1:], False) for v in VARIANTS]
        for v, baseline, subset in pairs:
            table, details = paired_changes(rows+drows, v, baseline, subset)
            key = v+'_vs_'+baseline; comparisons[key] = table; changes.extend(details)
            lines += ['', '## '+key.replace('_', ' '), '',
                      '| Category | n | Improved | Decreased | Unchanged | Mean ΔDice | HIT gains | HIT losses |'+(' TP removed | FP removed | Newly empty |' if subset else ''),
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'+(' ---: | ---: | ---: |' if subset else '')]
            for c, a in [('overall', table['overall']), *table['categories'].items()]:
                require(a['improved']+a['decreased']+a['unchanged'] == a['findings'], 'change count partition failure')
                line = f"| {c} | {a['findings']} | {a['improved']} | {a['decreased']} | {a['unchanged']} | {fmt(a['mean_delta_dice'])} | {a['hit_gain']} | {a['hit_loss']} |"
                if subset: line += f" {a['tp_removed']} | {a['fp_removed']} | {a['new_empty']} |"
                lines.append(line)
        atomic_json(reports/'changes.json', changes)
        ranked = ['# Largest paired finding changes', '']
        for key in comparisons:
            selected = [r for r in changes if r['variant']+'_vs_'+r['baseline'] == key]
            for title, direction in [('Improvements', 1), ('Regressions', -1)]:
                ranked += ['## '+key+' — '+title, '', '| Case | Finding | Category | ΔDice | Prompt |', '| --- | ---: | --- | ---: | --- |']
                for r in sorted([r for r in selected if direction*r['delta_dice'] > 1e-12], key=lambda r: -direction*r['delta_dice'])[:20]:
                    ranked.append(f"| {r['case']} | {r['finding_index']} | {r['category']} | {r['delta_dice']:+.9f} | {r['prompt'].replace('|', '/')} |")
                ranked.append('')
        (reports/'ranked_changes.md').write_text('\n'.join(ranked)+'\n')
    else:
        require(len(rows) == cfg['expected']['findings'], 'test prompt coverage')
        lines += ['Complete: five folders × 300 files, covering 582 prompts per variant. Test labels and scores are unavailable.', '']
    phase_timings = {p: read(runtime/(p+'_timings.json')) for p in phases}
    lines += ['', '## Provenance and timing', '', 'Job SHA: `'+job['job_sha256']+'`.',
              '| Phase | Wall minutes |', '| --- | ---: |']
    lines += [f"| {p} | {t['wall_seconds']/60:.2f} |" for p,t in phase_timings.items()]
    lines += ['', 'Worker timing totals:', '', '```json', json.dumps(timing, indent=2), '```', '',
              'Detailed findings, routing and largest changes: `'+str(reports)+'`.',
              'Prediction storage: '+f"{sum(x['bytes'] for x in outputs.values())/2**30:.3f} GiB.",
              'Reproduction: frozen `e_series_compute.py run --stage '+cfg['stage']+' --config '+str(runtime/'config.json')+'`.']
    (reports/'report.md').write_text('\n'.join(lines)+'\n')
    result = {'status': 'complete', 'run_id': cfg['run_id'], 'stage': cfg['stage'], 'job_sha256': job['job_sha256'],
              **cfg['expected'], 'files': cfg['expected']['cases']*5, 'finding_evaluations': 1905 if cfg['stage'] == 'val200' else None,
              'outputs': outputs, 'metrics': metrics, 'comparisons': comparisons, 'candidates': job['candidates'],
              'source_status': cfg['source_status'], 'timings_worker_seconds': timing, 'phase_timings': phase_timings,
              'zip_policy': 'skipped_by_request', 'archives': {}, 'report_sha256': sha(reports/'report.md'),
              'finding_records_sha256': sha(reports/'per_finding.json')}
    target = runtime/'completion.json'
    if target.exists(): require(read(target) == result, 'conflicting completed e report')
    else: atomic_json(target, result)
    return result


def run(cfg):
    scientific(); runtime = Path(cfg['runtime_root']); job = read(runtime/'job_manifest.json')
    with job_lock(runtime):
        try:
            verify_job(cfg, job)
            require(read(runtime/'preflight.json')['status'] == 'passed', 'preflight not passed')
            signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
            if cfg['stage'] == 'test300':
                import e_series
                bound = e_series.validate_test_binding(cfg, job)
                job = copy.deepcopy(job)
                for case in job['cases']: case['sources'] = bound[case['name']]
                # Sources are bound separately to the same frozen job, never rehashed into its identity.
            if (runtime/'completion.json').exists():
                report(cfg, read(runtime/'job_manifest.json')); return
            if cfg['stage'] == 'val200':
                records = run_phase(cfg, job, 'baseline')
                metrics = core.baseline_gate(cfg, records)
                atomic_json(runtime/'baseline_gate.json', {'status': 'passed', 'job_sha256': job['job_sha256'], 'metrics': metrics})
            run_phase(cfg, job, 'postprocessing')
            result = report(cfg, read(runtime/'job_manifest.json'))
            state(cfg, job, 'complete', cases_complete=cfg['expected']['cases'], findings_complete=cfg['expected']['findings'], files=result['files'])
        except BaseException as exc:
            state(cfg, job, 'interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed', error=str(exc), traceback=traceback.format_exc())
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight', 'run', 'report'))
    p.add_argument('--stage', required=True, choices=('val200', 'test300'))
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); master = read(args.config); cfg = stage_config(master, args.stage)
    if args.action == 'preflight': preflight(cfg)
    elif args.action == 'run': run(cfg)
    else:
        with job_lock(cfg['runtime_root']): report(cfg, read(Path(cfg['runtime_root'])/'job_manifest.json'))


if __name__ == '__main__': main()
