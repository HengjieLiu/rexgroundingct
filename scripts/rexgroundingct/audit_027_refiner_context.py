"""CPU-only, read-only checkpoint and two-window probe on one diagnostic case."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from audit_027_refiner_learning import base_metrics, write
from exp027_common import ARMS, read_json
from exp027_data import FindingStore, extract_patch
from exp027_model import make_model, refiner_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available(), 'Run this audit with GPUs disabled'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    config = read_json(args.config)
    result = dict(torch_version=torch.__version__, cuda_available=False, cpu_threads=2,
                  update=4000, checkpoints=[], probes=[])
    # Independently validate the NumPy paired-loss calculation against production.
    rng = np.random.default_rng(27)
    for empty in (False,True):
        z = rng.uniform(-30,30,size=(7,8,9)).astype('float32')
        y = (rng.random(z.shape)<.05).astype('float32') if not empty else np.zeros_like(z)
        valid = np.ones_like(z)
        valid[0]=0
        n = base_metrics(z,y,valid)
        tensors = [torch.from_numpy(a)[None,None] for a in (z,np.zeros_like(z),y,valid)]
        loss, details = refiner_loss(*tensors)
        assert abs(n['base_loss']-float(loss)) < 3e-6
        assert abs(n['base_patch_dice']-float(details['patch_dice'])) < 3e-6
    result['numpy_loss_vs_production_tests'] = 2
    pristine = None
    for arm in ARMS:
        folder = args.root/'full'/arm/'checkpoints'
        initial = torch.load(folder/'update_0000000.pth',map_location='cpu',weights_only=False)
        state = torch.load(folder/'update_0004000.pth',map_location='cpu',weights_only=False)
        if pristine is None:
            pristine = initial['model']
        assert all(torch.equal(value,pristine[name]) for name,value in initial['model'].items())
        assert state['update']==state['sampling_cursor']==4000
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        changes = {}
        for prefix in ('stem','blocks','head'):
            keys = [k for k in pristine if k.startswith(prefix)]
            changes[prefix] = dict(change_l2=float(torch.sqrt(sum(((state['model'][k]-pristine[k])**2).sum() for k in keys))),
                                   final_l2=float(torch.sqrt(sum((state['model'][k]**2).sum() for k in keys))))
        result['checkpoints'].append(dict(arm=arm,changes=changes,scaler=state['scaler'],
            dtypes=sorted({str(v.dtype) for v in state['model'].values()}),
            optimizer_steps=sorted({int(v['step']) for v in state['optimizer']['state'].values()}),
            lr=[g['lr'] for g in state['optimizer']['param_groups']]))
    result['common_initial_weights_verified'] = True
    write(args.output/'context_probe.json',result)
    print('Checkpoint and loss checks passed',flush=True)
    store = FindingStore(args.root.parent,read_json(args.root/'prepared.json'),config)
    key = 'train_3006_a_2.nii.gz::1'
    image, base, target, points, _ = store.get(key)
    center = np.median(np.asarray(points['gt']),axis=0).astype(int)
    start1 = [int(np.clip(c-96,0,n-192)) for c,n in zip(center,base.shape)]
    start2 = start1.copy()
    start2[0] += 32 if start1[0]+224 <= base.shape[0] else -32
    starts = [start1,start2]
    lo = np.maximum(start1,start2)+17
    hi = np.minimum(np.array(start1)+192,np.array(start2)+192)-17
    world = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    core_base = np.asarray(base[world],dtype=np.float32)
    core_gt = np.asarray(target[world],dtype=bool)
    for arm in ARMS[2:]:
        state = torch.load(args.root/'full'/arm/'checkpoints/update_0004000.pth',map_location='cpu',weights_only=False)
        model = make_model(config).eval()
        model.load_state_dict(state['model'])
        predictions, times = [], []
        for start in starts:
            ct,_ = extract_patch(image,start,(192,192,192),0)
            z,_ = extract_patch(base,start,(192,192,192),-30)
            x = torch.from_numpy(np.stack([ct,z]).astype('float32'))[None]
            begun = time.monotonic()
            with torch.inference_mode():
                delta = model(x)[0,0].numpy()
            times.append(time.monotonic()-begun)
            local = tuple(slice(int(a-s),int(b-s)) for a,b,s in zip(lo,hi,start))
            predictions.append(delta[local].copy())
            print(f'{arm}: window {start}, {times[-1]:.1f}s CPU',flush=True)
        d1,d2 = predictions
        pred1,pred2 = core_base+d1>=0,core_base+d2>=0
        interesting = core_gt | (core_base>=0) | pred1 | pred2
        disagreement = pred1!=pred2
        rows = []
        for name,p in (('base',core_base>=0),('window1',pred1),('window2',pred2)):
            tp = int((p&core_gt).sum()); fp = int((p&~core_gt).sum()); fn = int((~p&core_gt).sum())
            rows.append(dict(name=name,tp=tp,fp=fp,fn=fn,dice=(2*tp+1e-6)/(2*tp+fp+fn+1e-6)))
        result['probes'].append(dict(arm=arm,key=key,starts=starts,
            note='One case selected for a large run3 regression. Shared interior excludes the 17-voxel convolutional radius at both window edges. These are window predictions, not full-volume Gaussian-blended evaluation.',
            core_shape=list(core_gt.shape),gt_voxels=int(core_gt.sum()),interest_voxels=int(interesting.sum()),
            mean_abs_delta_difference=float(np.abs(d1-d2).mean()),
            interest_mean_abs_delta_difference=float(np.abs(d1-d2)[interesting].mean()),
            max_abs_delta_difference=float(np.abs(d1-d2).max()),
            mask_disagreement_voxels=int(disagreement.sum()),
            mask_disagreement_on_gt=int((disagreement&core_gt).sum()),
            local_metrics=rows,cpu_forward_seconds=times))
        write(args.output/'context_probe.json',result)
    print(args.output/'context_probe.json',flush=True)


if __name__=='__main__':
    main()
