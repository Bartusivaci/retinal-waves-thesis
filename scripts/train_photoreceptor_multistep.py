"""Photoreceptor-input FlyVis with multi-step, pair-state retinal-wave loss.

Run from repository root:
 python -m scripts.train_photoreceptor_multistep --steps 5 --val-windows 4 --validate-every 5 --save-every 5 --run-name multistep_smoke
 python -m scripts.train_photoreceptor_multistep --steps 1000 --val-windows 30 --validate-every 100 --save-every 200 --run-name multistep_gap1_1000

Frame timing remains UNVERIFIED: one original frame per 0.02-second numerical step.
Existing baseline checkpoint is read-only. No pretrained network weights are used.
"""
from __future__ import annotations
import argparse, csv, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from flyvis.network import Network
from scripts.train_smoke import (ROOT, CELL_TYPES, aligned_column_indices,
                                 nearest_hex_interpolation, grid_flow)
from scripts.train_gap5_10000 import PairFlowHead, photo_loss, flow_smoothness
from scripts.train_photoreceptor_finalstate import PhotoreceptorWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset


def all_pair_flows(responses, head, columns, near, weights, side, start_index):
    """For target t, use A[t-1], A[t] and difference; t is 0-based."""
    flows=[]
    for t in range(start_index, responses.shape[1]):
        a=responses[:,t-1,columns].permute(0,2,1)
        b=responses[:,t,columns].permute(0,2,1)
        local=head(a,b,1)
        flows.append(grid_flow(local,near,weights,side))
    return torch.cat(flows,dim=0)  # [number_of_pairs,2,H,W]


def compute_losses(frames, flows, args, device):
    """Each predicted field reconstructs previous image into current image."""
    images=F.interpolate(frames.to(device),size=(args.side,args.side),mode='area')
    source=images[args.warmup_frames-1:-1]
    target=images[args.warmup_frames:]
    if len(flows)!=len(source):
        raise RuntimeError(f'Flow count {len(flows)} != pair count {len(source)}')
    # Shared loss and image normalization with the prior experiments.
    photo=photo_loss(source,target,flows,args.active_weight)
    smooth=flow_smoothness(flows)
    zero=photo_loss(source,target,torch.zeros_like(flows),args.active_weight)
    return photo,smooth,zero


def evaluate(net, head, mapper, ds, indices, columns, near, w, args, device):
    net.eval(); head.eval()
    photo_sum=zero_sum=smooth_sum=flow_sum=0.
    with torch.no_grad():
        for idx in indices:
            frames=ds[idx]['frames']
            stimulus=mapper(frames).to(device)
            responses=net(stimulus,dt=args.dt)
            flows=all_pair_flows(responses,head,columns,near,w,args.side,args.warmup_frames)
            photo,smooth,zero=compute_losses(frames,flows,args,device)
            photo_sum+=float(photo);zero_sum+=float(zero)
            smooth_sum+=float(smooth);flow_sum+=float(flows.abs().mean())
    net.train(); head.train()
    n=len(indices)
    p,z=photo_sum/n,zero_sum/n
    return {'windows':n,'pairs_per_window':args.window_size-args.warmup_frames,
            'model_weighted':p,'zero_weighted':z,
            'improvement_percent':100*(z-p)/max(z,1e-12),
            'mean_abs_flow_pixels':flow_sum/n,'mean_smoothness':smooth_sum/n}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--window-size',type=int,default=20)
    p.add_argument('--warmup-frames',type=int,default=7,help='No loss for first N frames; pairs target f(N+1)...fT')
    p.add_argument('--val-windows',type=int,default=30)
    p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--side',type=int,default=64)
    p.add_argument('--dt',type=float,default=.02)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--active-weight',type=float,default=5.)
    p.add_argument('--smooth-weight',type=float,default=.01)
    p.add_argument('--network-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-name',default='multistep_gap1_1000')
    args=p.parse_args()
    if args.steps<1 or args.val_windows<1 or args.validate_every<1 or args.save_every<1:
        p.error('steps, val-windows, validate-every, save-every must be >=1')
    if args.window_size<=args.warmup_frames or args.warmup_frames<2:
        p.error('Require 2 <= warmup-frames < window-size')
    if args.side<16 or args.dt<=0 or args.gain<=0:
        p.error('Invalid side/dt/gain')
    if '/' in args.run_name or '\\' in args.run_name or args.run_name in ('','.','..'):
        p.error('run-name must be a directory name')
    out=ROOT/'results'/'multi_window'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'history.csv').exists() or (out/'final.pt').exists():
        raise FileExistsError(f'Run already exists: {out}. Choose a new --run-name.')
    device=torch.device(args.device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    train=RetinalWaveDataset(ROOT/'dataset/images.zip',ROOT/'wave_manifest.csv',
                             window_size=args.window_size,split='train')
    val=RetinalWaveDataset(ROOT/'dataset/images.zip',ROOT/'wave_manifest.csv',
                           window_size=args.window_size,split='val')
    rng=random.Random(args.seed)
    if len(val)<args.val_windows:raise ValueError('Too many validation samples requested')
    val_indices=rng.sample(range(len(val)),args.val_windows)
    net=Network()
    ckpt=torch.load(ROOT/'checkpoints/flyvis_initial.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(ckpt['network'],strict=True)
    mapper=PhotoreceptorWaveMapper(net,gain=args.gain)
    cols=aligned_column_indices(net).to(device)
    near,w=nearest_hex_interpolation(mapper,args.side,device)
    net=net.to(device)
    head=PairFlowHead(len(CELL_TYPES)).to(device)
    nparams=[param for param in net.parameters() if param.requires_grad]
    init=[t.detach().clone() for t in nparams]
    optim=torch.optim.Adam([{'params':nparams,'lr':args.network_lr},
                            {'params':head.parameters(),'lr':args.head_lr}])
    def save(path,step):
        torch.save({'network':net.state_dict(),'head':head.state_dict(),
                    'optimizer':optim.state_dict(),'step':step,
                    'head_type':'PairFlowHead','input_pathway':'BoxEye photoreceptors',
                    'training_method':'multi-step pair-state weighted L1',
                    'timing_assumption':'one original frame per 0.02s solver step; physical interval unverified',
                    'cell_types':CELL_TYPES,'settings':vars(args)},out/path)
    print('Photoreceptors via BoxEye; A[t-1],A[t] paired decoder; all pairs after warm-up.')
    print(f'Using {args.window_size} frames; warm-up {args.warmup_frames}; '
          f'loss on {args.window_size-args.warmup_frames} pairs/window.')
    print('WARNING: original dataset physical sampling interval unknown; dt is a numerical assumption.')
    best=-float('inf');best_step=None
    fields=['step','folder','segment_id','photo','smooth','total','zero_photo',
            'improvement_pct','flow_mean_abs_px','network_grad_norm','network_change_l2']
    try:
        with (out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for step in range(1,args.steps+1):
                sample=train[rng.randrange(len(train))]
                frames=sample['frames']
                stim_cpu=mapper(frames)
                if step==1:print('Stimulus check:',mapper.verify(stim_cpu))
                stim=stim_cpu.to(device)
                optim.zero_grad(set_to_none=True)
                responses=net(stim,dt=args.dt)
                flows=all_pair_flows(responses,head,cols,near,w,args.side,args.warmup_frames)
                photo,smooth,zero=compute_losses(frames,flows,args,device)
                total=photo+args.smooth_weight*smooth
                if not bool(torch.isfinite(total)):
                    raise RuntimeError(f'Non-finite total loss at step {step}')
                total.backward()
                grad=math.sqrt(sum(float(v.grad.square().sum()) for v in nparams if v.grad is not None))
                if not math.isfinite(grad) or grad<=0:
                    raise RuntimeError(f'FlyVis gradient missing at step {step}')
                torch.nn.utils.clip_grad_norm_(nparams,1.)
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                optim.step()
                change=math.sqrt(sum(float((v.detach()-i).square().sum()) for v,i in zip(nparams,init)))
                ph=float(photo.detach());ze=float(zero.detach())
                row={'step':step,'folder':sample['folder'],'segment_id':sample['segment_id'],
                     'photo':ph,'smooth':float(smooth.detach()),'total':float(total.detach()),
                     'zero_photo':ze,'improvement_pct':100*(ze-ph)/max(ze,1e-12),
                     'flow_mean_abs_px':float(flows.detach().abs().mean()),
                     'network_grad_norm':grad,'network_change_l2':change}
                writer.writerow(row);f.flush()
                if step==1 or step%100==0:print('Train',row,flush=True)
                if step%args.validate_every==0 or step==args.steps:
                    metrics=evaluate(net,head,mapper,val,val_indices,cols,near,w,args,device)
                    (out/f'validation_step_{step:06d}.json').write_text(json.dumps({'step':step,**metrics},indent=2))
                    print('Validation:',json.dumps(metrics),flush=True)
                    score=metrics['improvement_percent']
                    if not math.isfinite(score):raise RuntimeError('Nonfinite validation score')
                    if score>best:
                        best=score;best_step=step;save('best.pt',step)
                        (out/'best_metrics.json').write_text(json.dumps({'step':step,**metrics},indent=2))
                if step%args.save_every==0:save(f'checkpoint_step_{step:06d}.pt',step)
        save('final.pt',args.steps)
        (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
        (out/'validation_indices.json').write_text(json.dumps(val_indices,indent=2))
        print(f'Complete, best step {best_step}: {best:.4f}% validation improvement. Results: {out}')
    finally:
        train.close();val.close()

if __name__=='__main__':main()
