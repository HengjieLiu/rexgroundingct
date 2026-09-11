"""Read-only Exp027 learning audit; writes only to the requested analysis folder.

No GPU use, optimizer steps, cache generation, or production source changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from exp027_common import ARMS, read_json
from exp027_data import FindingStore


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def base_metrics(z, y, valid):
    """Same valid-voxel objective as production, without a learned residual."""
    z, y = z.astype(np.float32), y.astype(np.float32)
    p = 1 / (1 + np.exp(-z))
    count = float(valid.sum())
    bce = float(((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))) * valid).sum() / count)
    soft = float((2 * (p * y * valid).sum() + 1e-6) / ((p * valid).sum() + (y * valid).sum() + 1e-6))
    pred = z >= 0
    hard = float((2 * (pred * y * valid).sum() + 1e-6) / ((pred * valid).sum() + (y * valid).sum() + 1e-6))
    gt = (y > 0) & (valid > 0)
    fp = pred & ~gt & (valid > 0)
    fn = ~pred & gt
    return dict(base_bce=bce, base_dice_loss=1-soft, base_loss=bce+1-soft,
                base_patch_dice=hard, foreground_voxels=int(gt.sum()),
                valid_voxels=int(count), foreground_fraction=float(gt.sum()/count),
                base_fp=int(fp.sum()), base_fn=int(fn.sum()),
                fp_above_10=int((fp & (z > 10)).sum()), fn_below_minus10=int((fn & (z < -10)).sum()),
                target_probability_mass=float((p*gt).sum()), probability_mass=float((p*valid).sum()))


def evaluation_audit(root, out, prepared, update):
    result = []
    baseline = {r['key']: r for r in prepared['baseline_findings']}
    for arm in ARMS:
        for step in range(1000, update+1, 1000):
            path = root/'full'/arm/'evaluations'/f'update_{step:07d}'/'summary.json'
            s = read_json(path)
            rows = s['findings']
            assert len(rows) == len({r['key'] for r in rows}) == 69
            assert {r['key'] for r in rows} == set(baseline)
            for r in rows:
                assert abs(r['base_dice']-baseline[r['key']]['base_dice']) < 1e-12
                assert r['finding_id'] == baseline[r['key']]['finding_id']
                assert r['prompt'] == baseline[r['key']]['prompt']
                assert abs(r['dice']-(2*r['tp']+1e-6)/(2*r['tp']+r['fp']+r['fn']+1e-6)) < 1e-12
                base_tp = r['tp'] - r['fn_recovered'] + r['tp_removed']
                base_fp = r['fp'] + r['fp_removed'] - r['tn_added']
                base_fn = r['fn'] + r['fn_recovered'] - r['tp_removed']
                assert abs(r['base_dice']-(2*base_tp+1e-6)/(2*base_tp+base_fp+base_fn+1e-6)) < 1e-10
                r.update(base_precision=base_tp/(base_tp+base_fp) if base_tp+base_fp else 0,
                         base_recall=base_tp/(base_tp+base_fn) if base_tp+base_fn else 0)
            subsets = {}
            for half in ('A','B','full'):
                sub = [r for r in rows if half == 'full' or r['half'] == half]
                assert abs(mean(sub,'dice')-s['metrics'][half]['dice']) < 1e-12
                subset = dict(count=len(sub), dice=mean(sub,'dice'), base_dice=mean(sub,'base_dice'),
                              delta_dice=mean(sub,'delta_dice'), improved=sum(r['delta_dice']>1e-12 for r in sub),
                              worsened=sum(r['delta_dice'] < -1e-12 for r in sub),
                              unchanged=sum(abs(r['delta_dice']) <= 1e-12 for r in sub),
                              hits=sum(r['hit'] for r in sub))
                for key in ('improved','worsened','unchanged'):
                    assert subset[key] == s['metrics'][half][key]
                for key in ('fp_removed','fn_recovered','tp_removed','tn_added'):
                    subset[key] = sum(r[key] for r in sub)
                for key in ('precision','recall','base_precision','base_recall','residual_mean_abs',
                            'residual_p95_abs','edited_fraction_union'):
                    subset[key] = mean(sub,key)
                subset['residual_max_abs'] = max(r['residual_max_abs'] for r in sub)
                subsets[half] = subset
            result.append(dict(arm=arm, update=step, exposure=s['exposure'], subsets=subsets,
                               checkpoint_sha256=s['checkpoint_sha256'], findings=rows))
    write(out/'evaluations.json', result)
    return result


def patch_audit(root, out, config, prepared, update):
    store = FindingStore(root.parent, prepared, config)
    all_rows, histories, coverage = [], {}, {}
    for ai, arm in enumerate(ARMS):
        folder = root/'full'/arm
        events = read_json(folder/'schedule.json')['events'][:update]
        history = [json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
        hist = {r['update']: r for r in history if r['update'] <= update}
        assert sorted(hist) == list(range(1, update+1))
        histories[arm] = []
        for lo in range(0,update,1000):
            sub = [hist[i] for i in range(lo+1,lo+1001)]
            histories[arm].append(dict(update=lo+1000, **{k:mean(sub,k) for k in
                ('loss','bce','dice_loss','patch_dice','residual_mean_abs','residual_penalty','grad_norm')},
                clipped_fraction=sum(r['grad_norm']>1 for r in sub)/len(sub),
                overflow_retries=sum(r['amp_overflow_retries'] for r in sub)))
        coverage[arm] = {}
        for source in sorted({e['source'] for e in events}):
            pool = prepared['train'] if source == 'train' else [r for r in prepared['val'] if source=='val' or r['half']=='A']
            counts = Counter(e['key'] for e in events if e['source']==source)
            gtcounts = Counter(e['key'] for e in events if e['source']==source and e['mode']=='gt')
            coverage[arm][source] = dict(pool=len(pool), sampled_findings=len(counts),
                gt_sampled_findings=len(gtcounts), updates=sum(counts.values()), gt_updates=sum(gtcounts.values()))
        selected = []
        rng = np.random.default_rng(270040+ai)
        for lo in (0, update-1000):
            for mode,n in (('gt',8),('prediction',4),('random',4)):
                choices = [i for i in range(lo,lo+1000) if events[i]['mode']==mode]
                indices = rng.choice(choices,n,replace=False).tolist()
                if lo == 0 and events[0]['mode']==mode and 0 not in indices:
                    indices[0]=0
                selected.extend(indices)
        for index in sorted(selected):
            event = events[index]
            logged = hist[event['update']]
            assert (logged['key'],logged['source']) == (event['key'],event['source'])
            x, y, valid, info = store.patch(event, (192,192,192))
            stats = base_metrics(x[1],y[0],valid[0])
            if event['update']==1:
                assert abs(stats['base_loss']-logged['loss']) < 3e-6
                assert abs(stats['base_patch_dice']-logged['patch_dice']) < 3e-6
            row = dict(arm=arm, window='early' if index<1000 else 'late', **event, **info, **stats,
                       refined_loss=logged['loss'], refined_bce=logged['bce'], refined_dice_loss=logged['dice_loss'],
                       refined_patch_dice=logged['patch_dice'], residual_mean_abs=logged['residual_mean_abs'],
                       delta_loss=logged['loss']-stats['base_loss'],
                       delta_patch_dice=logged['patch_dice']-stats['base_patch_dice'])
            all_rows.append(row)
        print(f'{arm}: replayed 32 exact 192-cubed patches', flush=True)
        write(out/'patches.json',all_rows)
    write(out/'training_blocks.json',histories)
    write(out/'schedule_coverage.json',coverage)
    groups = []
    for arm in ARMS:
        for window in ('early','late'):
            for mode in ('all','gt','prediction','random'):
                rows = [r for r in all_rows if r['arm']==arm and r['window']==window and (mode=='all' or r['mode']==mode)]
                groups.append(dict(arm=arm,window=window,mode=mode,count=len(rows),
                    empty_targets=sum(r['foreground_voxels']==0 for r in rows),
                    **{k:mean(rows,k) for k in ('base_loss','refined_loss','delta_loss','base_patch_dice',
                       'refined_patch_dice','delta_patch_dice','foreground_fraction','base_bce','base_dice_loss')},
                    **{k:sum(r[k] for r in rows) for k in ('base_fp','base_fn','fp_above_10','fn_below_minus10')}))
    write(out/'patch_groups.json',groups)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ev = read_json(out/'evaluations.json')
    blocks = read_json(out/'training_blocks.json')
    fig, axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    colors = ('#0072B2','#E69F00','#009E73','#CC79A7')
    for ai,arm in enumerate(ARMS):
        rows = [r for r in ev if r['arm']==arm]
        for j,half in enumerate(('A','B','full')):
            axes[0,j].plot([r['update'] for r in rows],[100*r['subsets'][half]['delta_dice'] for r in rows],
                          '.-',color=colors[ai],label=f'Run {ai+1}')
        for j,key in enumerate(('loss','patch_dice','residual_mean_abs')):
            axes[1,j].plot([r['update'] for r in blocks[arm]], [r[key] for r in blocks[arm]],'.-',color=colors[ai])
    for j,half in enumerate(('A','B','full')):
        axes[0,j].axhline(0,color='gray',ls='--',lw=1)
        axes[0,j].set_title(f'{half}: full-volume Dice change (percentage points)')
    for j,title in enumerate(('Training loss (unpaired patch means)','Training hard patch Dice (unpaired)','Mean absolute patch residual')):
        axes[1,j].set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update (training means over preceding 1,000)')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Exp027 interim diagnosis through epoch 40 — no checkpoint selection\nA trained by runs 2–4; B trained by run 4; run 2/3 full scores mix exposure',fontsize=13)
    fig.savefig(out/'learning_diagnosis.png',dpi=150)
    plt.close(fig)


def bootstrap(out, update):
    """Descriptive uncertainty, preserving patient clusters and paired changes."""
    results = []
    for evaluation in read_json(out/'evaluations.json'):
        if evaluation['update'] != update:
            continue
        for half in ('A','B','full'):
            rows = [r for r in evaluation['findings'] if half=='full' or r['half']==half]
            patients = sorted({r['patient'] for r in rows})
            sums = np.array([sum(r['delta_dice'] for r in rows if r['patient']==p) for p in patients])
            sizes = np.array([sum(r['patient']==p for r in rows) for p in patients])
            indices = np.random.default_rng(270040).integers(len(patients),size=(10000,len(patients)))
            means = sums[indices].sum(1)/sizes[indices].sum(1)
            results.append(dict(arm=evaluation['arm'],half=half,patients=len(patients),
                                mean_delta=mean(rows,'delta_dice'),ci95=np.quantile(means,[.025,.975]).tolist()))
    write(out/'paired_cluster_bootstrap.json',dict(seed=270040,replicates=10000,
        note=f'Descriptive paired patient-cluster bootstrap at fixed interim update {update}; no multiplicity adjustment or confirmatory claim.',
        results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--update',type=int,default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert args.update >= 2000 and args.update%1000==0
    prepared = read_json(args.root/'prepared.json')
    evaluation_audit(args.root,args.output,prepared,args.update)
    patch_audit(args.root,args.output,read_json(args.config),prepared,args.update)
    plot(args.output)
    bootstrap(args.output,args.update)
    write(args.output/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        root=str(args.root),update=args.update,patches_per_arm=32,selection_seed_base=270040,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        note='Exact replay of historical patch inputs; recorded predictions use varying historical weights. Descriptive sample, not an unbiased held-out evaluation. No GPU work.'))
    print(args.output,flush=True)


if __name__=='__main__':
    main()
