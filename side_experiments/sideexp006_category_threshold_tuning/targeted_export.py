#!/usr/bin/env python3
"""B1-only targeted val200 exporter; never publishes full-val200 cache status."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import argparse
import fcntl
import gc
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from sweep import (REPO, HUB, CACHE_ROOT, DATASET, DATASET_SHA, SEG_ROOT, read,
                   file_hash, array_hash, digest, atomic_json, validate_cases,
                   validate_routing, baseline_references)

PREPROCESS = Path('/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell')
MISSING = {'exp007_ddp_e025': (1, 1), 'exp011_e4d4_clip_linear_e100': (6, 6),
           'exp012_r02_category_2d_replay50_e100': (22, 19),
           'exp013_category_2all_focal_target100_e100': (49, 40),
           'exp017_cont_e025_abs_e125': (3, 3)}


def helpers():
    sys.path.insert(0, str(REPO/'side_experiments/sideexp002_multimodel_ensemble_selection'))
    import export_logits as export
    from voxtell_preprocessed_cache import cache_case_paths
    return export, cache_case_paths


def storage_array(logits):
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite inference logits')
    half = np.clip(logits, -30, 30).astype(np.float16)
    mismatch = int(np.count_nonzero((half >= 0) != (logits >= 0)))
    return (half if not mismatch else np.asarray(logits, dtype=np.float32)), mismatch


def selected_gt(image, ids):
    import nibabel as nib
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= image.shape[0]:
        raise ValueError('Invalid original finding IDs')
    data = np.asarray(image.dataobj)[ids]
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite GT')
    return nib.Nifti1Image(data, image.affine, image.header.copy())


def code_identity():
    files = [Path(__file__), Path(__file__).with_name('sweep.py'),
             REPO/'side_experiments/sideexp002_multimodel_ensemble_selection/export_logits.py',
             REPO/'scripts/rexgroundingct/run_voxtell_val_inference.py',
             REPO/'scripts/rexgroundingct/voxtell_preprocessed_cache.py']
    return {str(p): file_hash(p) for p in files}


def make_plan(root):
    _, paths_for = helpers()
    if file_hash(DATASET) != DATASET_SHA:
        raise ValueError('Dataset identity changed')
    cases = read(DATASET)['test']
    validate_cases(cases)
    family = read(REPO/'submissions/registry.json')['families']['b']
    catalog = read(HUB/'checkpoint_catalog.json')['candidates']
    sources, jobs = [], []
    for model in family['models']:
        cid, sha = model['candidate_id'], model['checkpoint']['sha256']
        source = dict(candidate_id=cid, checkpoint_sha256=sha)
        if cid not in MISSING:
            candidates = [c for c in catalog if c['checkpoint']['sha256'] == sha]
            if len(candidates) != 1:
                raise ValueError('Ambiguous full cache')
            art = candidates[0]['inference_artifacts']['val200']
            versions = [v for v in art['versions'] if v['cache_version_id'] == art['active_cache_version'] and v['status'] == 'strict_passed' and v.get('cache_key')]
            if len(versions) != 1:
                raise ValueError('Missing strict full cache')
            source.update(cache_key=versions[0]['cache_key'], scope='full_val200')
        else:
            checkpoint = Path(model['checkpoint']['path'])
            if file_hash(checkpoint) != sha:
                raise ValueError('Checkpoint SHA mismatch')
            cache = PREPROCESS/model['preprocessing']['id']
            manifest = cache/'manifest.json'
            meta = read(manifest)
            # The canonical train/validation caches must include all validation cases.
            if 'val' not in meta.get('splits', []) and meta.get('case_stats', {}).get('split_val_cases') != 200:
                raise ValueError('Preprocessing is not a train/val cache')
            selected = []
            for c in cases:
                ids = [int(k) for k in sorted(c['findings'], key=int) if family['recipe']['categories'][c['categories'][k]] == cid]
                if not ids:
                    continue
                paths = paths_for(cache, c['name'])
                if not paths['complete'].is_file():
                    raise ValueError('Incomplete preprocessing case')
                image = np.load(paths['image'], mmap_mode='r', allow_pickle=False)
                gt = SEG_ROOT/c['name']
                selected.append(dict(name=c['name'], finding_ids=ids,
                    prompts=[c['findings'][str(k)] for k in ids],
                    input_files={str(p): file_hash(p) for p in (paths['image'], paths['metadata'], gt)},
                    smoke_size=int(np.prod(image.shape)), gt_path=str(gt)))
            if (sum(len(c['finding_ids']) for c in selected), len(selected)) != MISSING[cid]:
                raise ValueError(f'Unexpected routed subset size: {cid}')
            selected.sort(key=lambda c: (-c['smoke_size'], c['name']))
            plans = checkpoint.parent.parent/'plans.json'
            job = dict(candidate_id=cid, checkpoint_sha256=sha, checkpoint_path=str(checkpoint),
                       model_dir=str(checkpoint.parent.parent), plans_sha256=file_hash(plans),
                       preprocessing_root=str(cache), preprocessing_manifest_sha256=file_hash(manifest),
                       cases=selected)
            jobs.append(job)
            source.update(cache_key='sideexp006_subset_'+cid, scope='routed_subset',
                          export_manifest_path=str(root/'targeted'/cid/'export_manifest.json'))
        sources.append(source)
    contract = dict(dataset_sha256=DATASET_SHA, recipe_spec=family['recipe'], sources=sources,
                    jobs=jobs, code=code_identity(), scope='routed_subset',
                    registry_sha256=file_hash(REPO/'submissions/registry.json'))
    return dict(identity=digest(contract), contract=contract)


def check_files(files):
    for path, expected in files.items():
        if file_hash(path) != expected:
            raise ValueError(f'Source changed: {path}')


def run(root, gpu):
    import nibabel as nib
    import torch
    e, _ = helpers()
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.targeted.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = make_plan(root)
        path = root/'targeted_plan.json'
        if path.exists() and read(path)['identity'] != plan['identity']:
            raise ValueError('Targeted export resume source identity changed')
        atomic_json(path, plan)
        start_all = time.monotonic()
        for job in plan['contract']['jobs']:
            start_model = time.monotonic()
            cid = job['candidate_id']
            out = root/'targeted'/cid
            out.mkdir(parents=True, exist_ok=True)
            records = []
            predictor = None
            for position, case in enumerate(job['cases']):
                started = time.monotonic()
                record_path = out/'records'/f"{case['name']}.json"
                if record_path.exists():
                    record = read(record_path)
                    if record['identity'] != plan['identity'] or record['finding_ids'] != case['finding_ids']:
                        raise ValueError('Subset record resume identity mismatch')
                    a = np.load(record['array_path'], mmap_mode='r', allow_pickle=False)
                    if array_hash(a) != record['array_sha256']:
                        raise ValueError('Subset resume array changed')
                    records.append(record)
                    continue
                check_files(case['input_files'])
                if predictor is None:
                    while torch.cuda.mem_get_info(gpu)[0] < 20*1024**3:
                        print(f'Waiting for 20 GiB free on GPU {gpu}', flush=True)
                        time.sleep(30)
                    if shutil.disk_usage(root).free < 20*1024**4:
                        raise ValueError('20 TiB storage reserve not available')
                    predictor = e.build_predictor(job, torch.device(f'cuda:{gpu}'), None)
                    print(f'Loaded {cid}; largest-case smoke {case["name"]}', flush=True)
                image, targets, metadata = e.load_cached_case(Path(job['preprocessing_root']), case['name'], require_targets=False)
                del targets
                if not np.isfinite(image).all():
                    raise ValueError('Nonfinite preprocessed image')
                crop = e.predict_preprocessed_crop_logits(predictor, image, case['prompts'], padding_value=e.image_padding_value(metadata))
                restored = e.restore_cached_native_crop(crop, metadata, fill_value=-30)
                gt_image = selected_gt(nib.load(case['gt_path']), case['finding_ids'])
                logits, orientation = e.export_prediction_to_gt_layout(restored, gt_image, metadata['ct_properties'], case['name'], output_dtype=None)
                gt = np.asarray(gt_image.dataobj)
                reference = e.mask_metrics(gt, logits)
                stored, mismatch = storage_array(logits)
                storage_reference = e.mask_metrics(gt, stored)
                if storage_reference != reference or np.any((stored >= 0) != (logits >= 0)):
                    raise ValueError('Storage failed same-pass mask reproduction')
                array_path = out/'cases'/f"{case['name']}.npy"
                e.atomic_save_npy(array_path, stored)
                record = dict(identity=plan['identity'], name=case['name'], finding_ids=case['finding_ids'],
                    channel_to_finding={str(i): k for i, k in enumerate(case['finding_ids'])},
                    shape=list(stored.shape), dtype=str(stored.dtype), status='complete',
                    array_path=str(array_path), npy_bytes=array_path.stat().st_size, array_sha256=array_hash(stored),
                    same_pass_reference=reference, storage_reference=storage_reference,
                    float16_mismatch_voxels=mismatch, storage_mask_mismatch_voxels=0,
                    input_files=case['input_files'], prompts_sha256=digest(case['prompts']),
                    geometry=orientation, geometry_sha256=digest(dict(metadata=metadata, orientation=orientation)),
                    elapsed_seconds=time.monotonic()-started, largest_case_smoke=position == 0)
                atomic_json(record_path, record)
                records.append(record)
                torch.cuda.synchronize(gpu)
                free, total = torch.cuda.mem_get_info(gpu)
                if free < 4*1024**3:
                    raise ValueError('GPU free memory below 4 GiB safety floor')
                progress = dict(candidate_id=cid, cases_complete=len(records), cases_total=len(job['cases']),
                    findings_complete=sum(len(r['finding_ids']) for r in records), last_case_seconds=record['elapsed_seconds'],
                    elapsed_seconds=time.monotonic()-start_all, gpu_free_bytes=free,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu))
                atomic_json(root/'targeted_progress.json', progress)
                if position == 0:
                    atomic_json(out/'smoke.json', dict(status='passed', **progress, case=case['name']))
                print(progress, flush=True)
                del image, crop, restored, gt_image, logits, gt, stored
                gc.collect()
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(out/'export_manifest.json', dict(status='complete', scope='routed_subset',
                identity=plan['identity'], dataset_sha256=DATASET_SHA, checkpoint_sha256=job['checkpoint_sha256'],
                candidate_id=cid, job=job, cases=records, case_count=len(records),
                finding_count=sum(len(r['finding_ids']) for r in records), elapsed_seconds=time.monotonic()-start_model))
        sources = plan['contract']['sources']
        for source in sources:
            if source['scope'] == 'routed_subset':
                source['export_manifest_sha256'] = file_hash(source['export_manifest_path'])
        atomic_json(root/'source_manifest.json', dict(status='complete', dataset_sha256=DATASET_SHA,
            recipe_spec=plan['contract']['recipe_spec'], sources=sources, targeted_identity=plan['identity']))
        atomic_json(root/'targeted_completion.json', dict(status='complete', identity=plan['identity'],
            elapsed_seconds=time.monotonic()-start_all, findings=81, model_case_evaluations=69))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.output_root, args.gpu)
