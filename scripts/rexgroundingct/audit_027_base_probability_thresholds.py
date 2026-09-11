#!/usr/bin/env python3
"""CPU-only probability-threshold sweep of verified frozen Exp007 2a logits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
import time

import numpy as np

from exp027_common import REPO, atomic_csv, atomic_json, atomic_text, digest, now, read_json, require_hash, sha256
from exp027_data import case_key, compatible_contracts, validate_sources
from deletion027_data import verify_receipts

RUNTIME = Path('/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement')
PROBABILITIES = np.arange(25, 96, 5, dtype=np.float64) / 100
LOGIT_THRESHOLDS = np.log(PROBABILITIES / (1 - PROBABILITIES))


def count_thresholds(logits, targets, chunk_size=2**21):
    """One full-volume pass; bins retain logits below the usual 0.5 boundary."""
    assert logits.shape == targets.shape
    z, y = logits.reshape(-1), targets.reshape(-1)
    pred_bins = np.zeros(len(PROBABILITIES)+1, dtype=np.int64)
    tp_bins = np.zeros_like(pred_bins)
    total_gt = 0
    for start in range(0, z.size, chunk_size):
        values = np.asarray(z[start:start+chunk_size], dtype=np.float32)
        truth = np.asarray(y[start:start+chunk_size], dtype=bool)
        if not np.isfinite(values).all() or np.any(np.abs(values) > 30):
            raise ValueError('Non-finite or out-of-contract cached logits')
        total_gt += int(truth.sum())
        eligible = values >= LOGIT_THRESHOLDS[0]
        bins = np.searchsorted(LOGIT_THRESHOLDS, values[eligible], side='right')
        pred_bins += np.bincount(bins, minlength=len(pred_bins))
        tp_bins += np.bincount(bins[truth[eligible]], minlength=len(tp_bins))
    pred = np.cumsum(pred_bins[::-1], dtype=np.int64)[::-1][1:]
    tp = np.cumsum(tp_bins[::-1], dtype=np.int64)[::-1][1:]
    return pred, tp, total_gt


def self_test():
    rng = np.random.default_rng(41)
    near = LOGIT_THRESHOLDS.astype(np.float32)
    z = np.concatenate([rng.normal(size=2000).astype(np.float32), near,
                        np.nextafter(near, np.float32(-np.inf)), np.nextafter(near, np.float32(np.inf)),
                        np.array([-30, 0, 30], np.float32)])
    for dtype in (np.float16, np.float32):
        values = z.astype(dtype)
        for gt in (rng.integers(0,2,size=z.size,dtype=np.uint8), np.zeros(z.size,dtype=np.uint8)):
            counts, tp, total = count_thresholds(values, gt, 37)
            probs = 1 / (1 + np.exp(-values.astype(np.float64)))
            for i, threshold in enumerate(PROBABILITIES):
                # Direct logit comparison preserves tiny negative values at p=.5.
                # A floating-point sigmoid may round them to exactly .5.
                mask = values.astype(np.float64) >= LOGIT_THRESHOLDS[i]
                assert counts[i] == int(mask.sum())
                assert tp[i] == int((mask & (gt > 0)).sum())
                unambiguous = probs != threshold
                assert np.array_equal((probs >= threshold)[unambiguous], mask[unambiguous])
            assert total == int(gt.sum())


def process_case(task):
    base, contracts, records, source_sha = task
    name = records[0]['name']
    folder = Path(base['cache']['shared_root']) / case_key(name)
    meta = read_json(folder/'metadata.json')
    if meta['records'] != records or meta['contract'] not in contracts:
        raise ValueError(f'Finding membership/cache contract changed: {name}')
    if meta['origin']['kind'] != 'strict_validation_cache' or meta['origin']['array_sha256'] != source_sha:
        raise ValueError(f'Wrong frozen validation source: {name}')
    geometry = meta['ct_metadata']
    assert geometry['native_cropped_shape_zyx'] == geometry['resampled_shape_zyx'] == meta['shape']
    assert geometry['native_cropped_shape_zyx'] == geometry['original_reoriented_shape_zyx']
    assert geometry['image_resampling'] is None and geometry['mask_resampling'] is None
    assert geometry['preprocess_id'] == 'crop_zscore_native_v1'
    for filename in ['logits.npy', 'targets.npy']:
        require_hash(folder/filename, meta['hashes'][filename])
    z = np.load(folder/'logits.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(folder/'targets.npy', mmap_mode='r', allow_pickle=False)
    assert z.shape == y.shape == (len(records), *meta['shape'])
    assert str(z.dtype) == meta['origin']['dtype'] and z.dtype in (np.float16,np.float32)
    assert y.dtype == np.uint8
    results = []
    for i, record in enumerate(records):
        assert geometry['prompts'][record['channel']] == record['prompt']
        pred, tp, total = count_thresholds(z[i], y[i])
        assert total == record['voxels'] == geometry['source_target_voxels'][record['channel']]
        assert np.all(np.diff(pred) <= 0) and np.all(np.diff(tp) <= 0)
        for j, probability in enumerate(PROBABILITIES):
            n, true = int(pred[j]), int(tp[j])
            dice = (2*true+1e-6)/(n+total+1e-6)
            results.append({**record,'probability_threshold':float(probability),
                            'logit_threshold':float(LOGIT_THRESHOLDS[j]),'tp':true,'fp':n-true,'fn':total-true,
                            'predicted_voxels':n,'dice':dice,'hit':int(dice>=.1),
                            'precision':true/n if n else 0.,'recall':true/total if total else None})
    return {'name':name,'dtype':str(z.dtype),'metadata_sha256':sha256(folder/'metadata.json'),
            'array_hashes':meta['hashes'],'source_array_sha256':source_sha,'findings':results}


def aggregate(rows):
    summaries = []
    baseline = {r['key']:r for r in rows if r['probability_threshold']==.5}
    for p in PROBABILITIES:
        for half in ['A','B','full']:
            selected = [r for r in rows if r['probability_threshold']==p and (half=='full' or r['half']==half)]
            n = len(selected)
            record = {'probability_threshold':float(p),'logit_threshold':float(math.log(p/(1-p))),
                      'subset':half,'findings':n,'cases':len({r['name'] for r in selected})}
            for key in ['dice','precision','recall']:
                values = [r[key] for r in selected if r[key] is not None]
                record[key] = math.fsum(values)/len(values) if values else None
            for key in ['tp','fp','fn','predicted_voxels','hit']:
                record[key] = sum(r[key] for r in selected)
            record['hit_rate'] = record['hit']/n
            record['pooled_dice'] = 2*record['tp']/(2*record['tp']+record['fp']+record['fn'])
            changes = [r['dice']-baseline[r['key']]['dice'] for r in selected]
            record['delta_dice'] = math.fsum(changes)/n
            record['improved'] = sum(v>1e-12 for v in changes)
            record['unchanged'] = sum(abs(v)<=1e-12 for v in changes)
            record['worsened'] = sum(v< -1e-12 for v in changes)
            record['hits_gained'] = sum(r['hit'] > baseline[r['key']]['hit'] for r in selected)
            record['hits_lost'] = sum(r['hit'] < baseline[r['key']]['hit'] for r in selected)
            summaries.append(record)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('Chunked histogram matches direct logit masks and unambiguous sigmoid comparisons for FP16/FP32, boundaries and empty GT.')
        return
    started = time.perf_counter()
    root = RUNTIME/'analysis/base_probability_threshold_sweep'
    root.mkdir(parents=True,exist_ok=True)
    context_path = RUNTIME/'deletion_four_arm_20ep/context.json'
    context = read_json(context_path)
    base = context['base_config']
    prepared_path = RUNTIME/'deletion_four_arm_20ep/prepared.json'
    prepared = read_json(prepared_path)
    sources = validate_sources(base)
    assert sha256(Path(base['base']['validation_cache'])/'export_manifest.json') == prepared['base_manifest_sha256']
    proof = verify_receipts(base,prepared)
    assert proof == context['cache_proof']
    contracts = compatible_contracts(base,prepared)
    source_rows = {r['name']:r for r in sources['cases']}
    groups = {}
    for record in prepared['val']:
        groups.setdefault(record['name'],[]).append(record)
    assert len(groups)==63 and sum(map(len,groups.values()))==69
    atomic_json(root/'status.json',{'status':'running_cpu','updated_at':now(),'cases_done':0,'cases_total':63})
    output = []
    try:
        tasks = [(base,contracts,records,source_rows[name]['array_sha256']) for name,records in groups.items()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_case,t) for t in tasks]
            for future in as_completed(futures):
                case = future.result();output.append(case)
                atomic_json(root/'status.json',{'status':'running_cpu','updated_at':now(),'cases_done':len(output),
                                              'cases_total':63,'latest_case':case['name'],'elapsed_seconds':time.perf_counter()-started})
                print(f"Verified and measured {len(output)}/63 CTs: {case['name']}",flush=True)
        order = {r['key']:i for i,r in enumerate(prepared['val'])}
        rows = sorted([r for c in output for r in c['findings']],key=lambda r:(r['probability_threshold'],order[r['key']]))
        assert len(rows)==69*len(PROBABILITIES)
        old_baseline = {r['key']:r['dice'] for r in prepared['baseline_findings']}
        for r in rows:
            r['baseline_dice'] = old_baseline[r['key']]
            r['delta_dice'] = r['dice']-r['baseline_dice']
            if r['probability_threshold']==.5:
                assert abs(r['delta_dice'])<1e-10,(r['key'],r['delta_dice'])
        summaries = aggregate(rows)
        lookup = {(r['probability_threshold'],r['subset']):r for r in summaries}
        for p in PROBABILITIES:
            a,b,f=[lookup[(float(p),h)] for h in ['A','B','full']]
            assert abs((a['dice']*35+b['dice']*34)/69-f['dice'])<1e-12
            for k in ['tp','fp','fn','hit','predicted_voxels']: assert a[k]+b[k]==f[k]
        atomic_csv(root/'summary.csv',summaries)
        atomic_csv(root/'per_finding.csv',rows)
        result = {'status':'pending_user_review','created_at':now(),'elapsed_seconds':time.perf_counter()-started,
                  'source_checkpoint':base['base'],'source_context_sha256':sha256(context_path),
                  'source_prepared_sha256':sha256(prepared_path),'script_sha256':sha256(__file__),
                  'cache_proof':proof,'probability_thresholds':PROBABILITIES.tolist(),
                  'logit_thresholds':LOGIT_THRESHOLDS.tolist(),'binary_rule':'sigmoid(cached_logit) >= probability_threshold',
                  'geometry':'native cached full crop; axis transforms and uncropping preserve counts; no resampling',
                  'dice_aggregation':'unweighted mean of original finding Dice, smoothing 1e-6',
                  'cache_cases':[{k:v for k,v in c.items() if k!='findings'} for c in sorted(output,key=lambda c:c['name'])],
                  'verification':{'cases':63,'findings':69,'thresholds':len(PROBABILITIES),
                                  'summary_rows':len(summaries),'per_finding_rows':len(rows),
                                  'full_original_extents_verified':True,
                                  'cache_array_hashes_verified':True,'baseline_0_5_reproduced_per_finding':True,
                                  'monotonic_counts':True,'a_plus_b_recomposition':True,'histogram_self_test':True},
                  'summaries':summaries}
        atomic_json(root/'results.json',result)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for h,color in [('A','tab:blue'),('B','tab:orange'),('full','tab:green')]:
            selected=[lookup[(float(p),h)] for p in PROBABILITIES]
            axes[0].plot(PROBABILITIES,[r['dice'] for r in selected],marker='o',label=h,color=color)
        axes[0].set(xlabel='Base foreground probability threshold',ylabel='Mean finding Dice',title='Frozen Exp007 — 2a Dice')
        for k,label,color in [('precision','Precision','tab:purple'),('recall','Recall','tab:red')]:
            axes[1].plot(PROBABILITIES,[lookup[(float(p),'full')][k] for p in PROBABILITIES],marker='o',label=label,color=color)
        axes[1].set(xlabel='Base foreground probability threshold',ylabel='Mean per-finding metric',title='Full 2a cohort — precision / recall')
        for ax in axes:
            ax.axvline(.5,linestyle='--',color='gray',label='Original threshold 0.50');ax.grid(alpha=.25);ax.legend();ax.set_xticks(PROBABILITIES)
        fig.suptitle('Exp007 continuation epoch 50 / absolute epoch 150; cached frozen predictions\n69 2a findings / 63 CTs. A/B partition preserved; no threshold selected.')
        fig.savefig(root/'threshold_sweep.png',dpi=160);fig.savefig(root/'threshold_sweep.pdf');plt.close(fig)
        lines=['# Frozen Exp007 2a probability-threshold sweep','',
               'User-requested probability thresholds 0.25–0.95 in steps of 0.05. This is a binary foreground threshold on the frozen base logits; the deletion module is not used. Status: **pending_user_review**.','',
               'Checkpoint: Exp007 continuation epoch 50 / absolute epoch 150, SHA `'+base['base']['checkpoint_sha256']+'`. All 69 validation 2a findings across 63 CTs are included; A has 35 findings and B has 34.','',
               'Predicted foreground = sigmoid(cached logit) >= threshold, evaluated equivalently in logit space. The entire cached native extent is included, so thresholds below 0.50 can recover additional foreground. The established clipped [-30,30] cache and its recorded storage dtype are preserved; there is no new model inference or quantization. Clipping cannot change decisions within the requested threshold range.','',
               'Dice is the unweighted mean over findings, using smoothing 1e-6. The native geometry has no resampling; original finding IDs, prompt/channel order and GT voxel counts are checked. All 63 CT cache logit/target hashes passed, and all cached extents equal the full original reoriented CT extents. Threshold 0.50 reproduces every cached baseline finding; A+B recomposition and monotonic TP/prediction counts passed.','',
               '![Threshold sweep](runtime/analysis/base_probability_threshold_sweep/threshold_sweep.png)','',
               '| Probability threshold | Logit threshold | A Dice | B Dice | Full Dice | Full change vs 0.50 | Full hits | Full precision | Full recall |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for p in PROBABILITIES:
            a,b,f=[lookup[(float(p),h)] for h in ['A','B','full']]
            lines.append(f"| {p:.2f} | {f['logit_threshold']:.6f} | {a['dice']:.6f} | {b['dice']:.6f} | {f['dice']:.6f} | {f['delta_dice']:+.6f} | {f['hit']}/69 | {100*f['precision']:.3f}% | {100*f['recall']:.3f}% |")
        lines += ['', 'Precision and recall above are also means over findings. Hits use Dice >= 0.10. The exports include raw TP/FP/FN counts, pooled Dice, per-finding changes, and hits gained/lost. Threshold ranking and selection remain with the user.','',
                  f'- [All {len(summaries)} A/B/full summaries: CSV](runtime/analysis/base_probability_threshold_sweep/summary.csv)',
                  f'- [All {len(rows)} per-finding rows: CSV](runtime/analysis/base_probability_threshold_sweep/per_finding.csv)',
                  '- [Results, hashes and provenance: JSON](runtime/analysis/base_probability_threshold_sweep/results.json)',
                  '- [Figure PDF](runtime/analysis/base_probability_threshold_sweep/threshold_sweep.pdf)','',
                  f"CPU analysis completed in {result['elapsed_seconds']:.1f} seconds before plotting. Reproduce with `python scripts/rexgroundingct/audit_027_base_probability_thresholds.py --workers 4` in the existing VoxTell image with GPUs disabled.",'']
        atomic_text(REPO/'experiments/027_voxtell_2a_residual_refinement/base_probability_threshold_sweep.md','\n'.join(lines))
        atomic_json(root/'status.json',{'status':'pending_user_review','cases_done':63,'findings':69,'updated_at':now(),'elapsed_seconds':time.perf_counter()-started})
        for s in summaries:
            if s['subset']=='full':print(f"p={s['probability_threshold']:.2f} Dice={s['dice']:.9f} hits={s['hit']}/69",flush=True)
    except BaseException as error:
        atomic_json(root/'status.json',{'status':'failed','error':repr(error),'updated_at':now()})
        raise


if __name__=='__main__':
    main()
