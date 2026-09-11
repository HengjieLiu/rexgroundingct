"""CPU-only geometry audit of a proposed gate; no model inference or training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from exp027_common import read_json, atomic_json, atomic_csv, code_fingerprint
from exp027_data import FindingStore, case_key
from exp027_model import tile_starts


def coordinates(array, positive):
    parts=[]
    for start in range(0,array.shape[0],24):
        slab=array[start:start+24]
        points=np.argwhere(slab>=0 if positive else slab>0).astype(np.int32)
        points[:,0]+=start
        parts.append(points)
    return np.concatenate(parts)


def bounds(shape, size):
    starts=tile_starts(shape,size,.5)
    return [(np.maximum(s,0),np.minimum(np.array(s)+size,shape)) for s in starts]


def inside(points, lo, hi):
    return np.all((points>=lo)&(points<hi),axis=1)


def union_measure(shape, boxes, active, points):
    """Exact box-union volume on a small coordinate-compressed grid."""
    edges=[np.unique([0,shape[d],*[int(x[d]) for box in boxes for x in box]]) for d in range(3)]
    cover=np.zeros(tuple(len(e)-1 for e in edges),dtype=bool)
    for (lo,hi),enabled in zip(boxes,active):
        if enabled:
            slices=tuple(slice(int(np.searchsorted(e,l)),int(np.searchsorted(e,h))) for e,l,h in zip(edges,lo,hi))
            cover[slices]=True
    widths=[np.diff(e).astype(np.int64) for e in edges]
    volumes=widths[0][:,None,None]*widths[1][None,:,None]*widths[2][None,None,:]
    index=tuple(np.searchsorted(e,points[:,d],side='right')-1 for d,e in enumerate(edges))
    return int(volumes[cover].sum()),cover[index]


def smoke():
    rng=np.random.default_rng(27)
    for shape in ((5,7,9),(17,18,19),(8,10,14)):
        size=(8,8,8)
        boxes=bounds(shape,size)
        active=rng.random(len(boxes))>.5
        direct=np.zeros(shape,dtype=bool)
        for (lo,hi),enabled in zip(boxes,active):
            if enabled:direct[tuple(slice(a,b) for a,b in zip(lo,hi))]=True
        points=np.array(list(itertools.product(*(range(n) for n in shape))))
        volume,covered=union_measure(shape,boxes,active,points)
        assert volume==int(direct.sum())
        assert np.array_equal(covered,direct[tuple(points.T)])
    for threshold in (True,False):
        a=rng.integers(-2,3,size=(31,6,7))
        assert np.array_equal(coordinates(a,threshold),np.argwhere(a>=0 if threshold else a>0))


def summary(rows):
    totals={k:sum(r[k] for r in rows) for k in ('tiles','active_tiles','active_tiles_without_gt',
        'inactive_tiles_with_gt','native_voxels','editable_voxels','original_gt','native_gt',
        'reachable_gt','unreachable_native_gt','gt_outside_crop','base_positive_voxels',
        'active_tiles_le10_base_voxels')}
    totals.update(findings=len(rows),scans=len({r['name'] for r in rows}),
                  no_base_prediction=sum(r['base_positive_voxels']==0 for r in rows),
                  findings_losing_gt=sum(r['unreachable_native_gt']>0 for r in rows),
                  active_tile_fraction=totals['active_tiles']/totals['tiles'],
                  editable_volume_fraction=totals['editable_voxels']/totals['native_voxels'],
                  reachable_gt_fraction=totals['reachable_gt']/totals['original_gt'],
                  mean_editable_volume_fraction=float(np.mean([r['editable_voxels']/r['native_voxels'] for r in rows])),
                  mean_oracle_dice=float(np.mean([r['oracle_dice_inside_gate'] for r in rows])))
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    assert not torch.cuda.is_available(), 'This audit requires GPUs disabled'
    torch.set_num_threads(1)
    smoke()
    args.output.mkdir(parents=True,exist_ok=True)
    prepared=read_json(args.root/'prepared.json')
    config=read_json(args.config)
    store=FindingStore(args.root.parent,prepared,config,capacity=1)
    rows=[]; started=time.monotonic()
    for index,record in enumerate(prepared['val']):
        _,base,target,points,meta=store.get(record['key'])
        assert meta['ct_metadata']['image_resampling'] in (None,False)
        assert meta['ct_metadata']['mask_resampling'] in (None,False)
        pred=coordinates(base,True); gt=coordinates(target,False)
        assert bool(len(pred))==bool(points['prediction'])
        boxes=bounds(base.shape,np.array([192,192,192]))
        pc=np.array([int(inside(pred,lo,hi).sum()) for lo,hi in boxes])
        gc=np.array([int(inside(gt,lo,hi).sum()) for lo,hi in boxes])
        active=pc>0
        editable,covered=union_measure(base.shape,boxes,active,gt)
        _,covered_pred=union_measure(base.shape,boxes,active,pred)
        assert covered_pred.all(), 'The gate must retain every base-positive voxel'
        assert int(record['voxels'])>=len(gt)
        reachable=int(covered.sum()); original=int(record['voxels'])
        rows.append(dict(key=record['key'],name=record['name'],half=record['half'],
            finding_id=record['finding_id'],prompt=record['prompt'],shape=list(base.shape),
            tiles=len(boxes),active_tiles=int(active.sum()),
            active_tiles_without_gt=int((active&(gc==0)).sum()),
            inactive_tiles_with_gt=int((~active&(gc>0)).sum()),
            native_voxels=int(np.prod(base.shape)),editable_voxels=editable,
            original_gt=original,native_gt=len(gt),reachable_gt=reachable,
            unreachable_native_gt=len(gt)-reachable,gt_outside_crop=original-len(gt),
            base_positive_voxels=len(pred),active_tiles_le10_base_voxels=int(((pc>0)&(pc<=10)).sum()),
            oracle_dice_inside_gate=(2*reachable+1e-6)/(original+reachable+1e-6)))
        if (index+1)%10==0 or index==68:
            print(f'Validation {index+1}/69; {time.monotonic()-started:.1f}s',flush=True)
            atomic_json(args.output/'validation_findings.json',rows)
    atomic_csv(args.output/'validation_findings.csv',rows)
    metadata={}; availability=[]
    for record in prepared['train']:
        name=record['name']
        if name not in metadata:
            meta=read_json(args.root.parent/'cache'/case_key(name)/'metadata.json')
            assert meta['contract'] in store.contracts
            metadata[name]={k:bool(v['prediction']) for k,v in meta['points'].items()}
        availability.append(dict(key=record['key'],name=name,has_base_prediction=metadata[name][record['key']]))
    atomic_json(args.output/'training_prediction_availability.json',availability)
    historical=read_json(args.root/'analysis/learning_diagnosis_e040/patches.json')
    replay=[]
    for arm in sorted({r['arm'] for r in historical}):
        sub=[r for r in historical if r['arm']==arm]
        active=[r for r in sub if r['base_fp']+r['foreground_voxels']-r['base_fn']>0]
        replay.append(dict(arm=arm,patches=len(sub),base_positive_patches=len(active),
            base_positive_gt_empty=sum(r['foreground_voxels']==0 for r in active),
            rejected_by_mode=dict(Counter(r['mode'] for r in sub if r not in active))))
    result=dict(status='audit_only_no_method_change',created_at=datetime.now(timezone.utc).isoformat(),
                gate='At least one valid voxel with the same finding base logit >= 0',
                tile_size=[192,192,192],overlap=.5,
                validation={h:summary([r for r in rows if h=='full' or r['half']==h]) for h in ('A','B','full')},
                training=dict(findings=len(availability),scans=len(metadata),
                              findings_without_prediction=sum(not r['has_base_prediction'] for r in availability)),
                historical_patch_replay=replay,seconds=time.monotonic()-started,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                production_source=code_fingerprint(),
                note='Geometry only: no refined logits were inferred. Editable means union of active tiles; inactive overlapping tiles do not guarantee voxel immutability. Oracle assumes perfect prediction inside the active union and unchanged base outside, and is not an achievable performance estimate.')
    atomic_json(args.output/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
