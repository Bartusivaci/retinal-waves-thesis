"""Train original FlyVis + two-state flow head on synthetic moving edges.

Architecture identical to corrected retinal-wave multi-step pilot:
    BoxEye photoreceptors -> FlyVis from scratch -> A[t-1],A[t] PairFlowHead
    -> corrected 721-to-64x64 projection -> backward warp -> weighted L1+smoothness.

Optional --objective flow is a supervised POSITIVE CONTROL with known synthetic
optical flow. It probes architecture/optimization separately from photometric loss.

Run:
  python -m scripts.train_moving_edges_multistep --steps 5 --val-samples 4 --run-name edge_smoke
  python -m scripts.train_moving_edges_multistep --steps 1000 --run-name edge_photo_1000
  python -m scripts.train_moving_edges_multistep --steps 1000 --objective flow --run-name edge_supervised_1000
"""
from __future__ import annotations
import argparse,csv,json,math,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flyvis.network import Network
from scripts.train_smoke import ROOT,CELL_TYPES,aligned_column_indices
from scripts.train_gap5_10000 import PairFlowHead,photo_loss,flow_smoothness
from scripts.train_photoreceptor_finalstate import PhotoreceptorWaveMapper,warp_border
from scripts.train_photoreceptor_multistep_corrected import corrected_boxeye_interpolation,all_pair_flows


def load_records(root):
    with (root/'manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    return {s:[r for r in rows if r['split']==s] for s in ('train','val','test')}


def get_sample(root,row,device,dt_side=256):
    with np.load(root/row['filename']) as data:
        v=np.array(data['frames'],copy=True)
    x=torch.from_numpy(v).float().unsqueeze(1)/255. # [T,1,64,64]
    high=F.interpolate(x,size=(dt_side,dt_side),mode='bilinear',align_corners=False)
    truth=torch.tensor([-float(row['dx_px_per_frame']),
                        -float(row['dy_px_per_frame'])],device=device,dtype=torch.float32)
    return x.to(device),high,truth


def forward_losses(net,head,mapper,record,dataset_root,columns,near,w,args,device,require_grad):
    images,high,truth=get_sample(dataset_root,record,device)
    # mapper internally renders BoxEye and constructs externally stimulated input
    stimulus=mapper(high).to(device)
    with torch.set_grad_enabled(require_grad):
        response=net(stimulus,dt=args.dt)
        flows=all_pair_flows(response,head,columns,near,w,args.side,args.warmup_frames)
        source=images[args.warmup_frames-1:-1]
        target=images[args.warmup_frames:]
        assert flows.shape==(args.frames-args.warmup_frames,2,args.side,args.side),(flows.shape,source.shape)
        photo=photo_loss(source,target,flows,args.active_weight)
        zero=photo_loss(source,target,torch.zeros_like(flows),args.active_weight)
        smooth=flow_smoothness(flows)
        true_flow=truth.view(1,2,1,1).expand_as(flows)
        # Motion direction and magnitude are known; measure EPE in the image interior.
        epe=(flows[:,:,6:-6,6:-6]-true_flow[:,:,6:-6,6:-6]).square().sum(1).sqrt().mean()
        oracle=photo_loss(source,target,true_flow,args.active_weight)
        if args.objective=='flow':
            fit=(flows-true_flow).square().mean()
        else:
            fit=photo
        total=fit+args.smooth_weight*smooth
    return total,{'photo':float(photo.detach()),'zero_photo':float(zero.detach()),
                  'oracle_photo':float(oracle.detach()),'epe_px':float(epe.detach()),
                  'flow_mean_abs_px':float(flows.detach().abs().mean()),
                  'smooth':float(smooth.detach()),'total':float(total.detach())}


def evaluate(net,head,mapper,records,dataset_root,columns,near,w,args,device):
    net.eval();head.eval();metrics=[]
    for rec in records:
        with torch.no_grad():
            _,m=forward_losses(net,head,mapper,rec,dataset_root,columns,near,w,args,device,False)
            metrics.append(m)
    net.train();head.train()
    summary={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    z=summary['zero_photo'];p=summary['photo']
    summary['improvement_percent']=100*(z-p)/max(z,1e-12)
    summary['samples']=len(records)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=ROOT/'dataset'/'synthetic_moving_edges')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--frames',type=int,default=20)
    p.add_argument('--warmup-frames',type=int,default=7)
    p.add_argument('--side',type=int,default=64)
    p.add_argument('--dt',type=float,default=.02)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--objective',choices=['photo','flow'],default='photo')
    p.add_argument('--active-weight',type=float,default=5.)
    p.add_argument('--smooth-weight',type=float,default=.01)
    p.add_argument('--network-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--val-samples',type=int,default=20)
    p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-name',default='moving_edges_photo_1000')
    args=p.parse_args()
    if args.steps<1 or args.validate_every<1 or args.save_every<1 or args.val_samples<1:p.error('steps/validation/save options must be positive')
    if args.side!=64 or not 2<=args.warmup_frames<args.frames:p.error('Requires side=64 and 2 <= warmup-frames < frames')
    if '/' in args.run_name or '\\' in args.run_name or args.run_name in ('','.','..'):p.error('run-name must be a simple directory name')
    config=json.loads((args.dataset/'config.json').read_text())
    if config['frames']!=args.frames or config['side']!=args.side:
        p.error(f'Dataset shape {config["frames"]}x{config["side"]} incompatible with --frames/--side')
    records=load_records(args.dataset)
    if not records['train'] or len(records['val'])<args.val_samples:p.error('Insufficient training/validation examples')
    device=torch.device(args.device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    rng=random.Random(args.seed)
    val=rng.sample(records['val'],args.val_samples)
    out=ROOT/'results'/'synthetic_moving_edges'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'history.csv').exists() or (out/'final.pt').exists():
        raise FileExistsError(f'Run exists: {out}. Specify another --run-name.')
    net=Network()
    checkpoint=torch.load(ROOT/'checkpoints/flyvis_initial.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['network'],strict=True)
    mapper=PhotoreceptorWaveMapper(net,gain=args.gain)
    cols=aligned_column_indices(net).to(device)
    near,w=corrected_boxeye_interpolation(mapper,cols,args.side,device)
    net=net.to(device)
    head=PairFlowHead(len(CELL_TYPES)).to(device)
    trainable=[param for param in net.parameters() if param.requires_grad]
    original=[param.detach().clone() for param in trainable]
    optim=torch.optim.Adam([{'params':trainable,'lr':args.network_lr},
                            {'params':head.parameters(),'lr':args.head_lr}])
    def save(name,step):
        torch.save({'network':net.state_dict(),'head':head.state_dict(),'optimizer':optim.state_dict(),
                    'step':step,'input_pathway':'BoxEye photoreceptors',
                    'method':'multistep paired-state corrected coordinates','objective':args.objective,
                    'dataset':'synthetic_moving_edges','settings':vars(args)},out/name)
    print(f'Synthetic edges: train={len(records["train"])}, val={len(records["val"])}, test={len(records["test"])}')
    print(f'Objective={args.objective}; photoreceptors via BoxEye; FlyVis+head trainable; {args.frames-args.warmup_frames} pairs/window.')
    print('Ground truth backward flow is used for evaluation only when objective=photo.')
    history_fields=['step','id','photo','zero_photo','oracle_photo','improvement_percent',
                    'epe_px','flow_mean_abs_px','smooth','total','network_grad_norm','network_change_l2']
    best=float('inf')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history_fields);writer.writeheader()
        for step in range(1,args.steps+1):
            row=records['train'][rng.randrange(len(records['train']))]
            net.train();head.train();optim.zero_grad(set_to_none=True)
            loss,m=forward_losses(net,head,mapper,row,args.dataset,cols,near,w,args,device,True)
            if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite loss step {step}')
            loss.backward()
            g=math.sqrt(sum(float(x.grad.square().sum()) for x in trainable if x.grad is not None))
            if not math.isfinite(g) or g<=0:raise RuntimeError(f'FlyVis gradient missing at step {step}')
            torch.nn.utils.clip_grad_norm_(trainable,1.)
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optim.step()
            delta=math.sqrt(sum(float((x.detach()-o).square().sum()) for x,o in zip(trainable,original)))
            m.update({'step':step,'id':row['id'],'improvement_percent':100*(m['zero_photo']-m['photo'])/max(m['zero_photo'],1e-12),
                      'network_grad_norm':g,'network_change_l2':delta})
            writer.writerow(m);f.flush()
            if step==1 or step%100==0:print('Train:',m,flush=True)
            if step%args.validate_every==0 or step==args.steps:
                val_metrics=evaluate(net,head,mapper,val,args.dataset,cols,near,w,args,device)
                val_metrics.update({'step':step,'objective':args.objective})
                (out/f'validation_step_{step:06d}.json').write_text(json.dumps(val_metrics,indent=2))
                print('Validation:',json.dumps(val_metrics),flush=True)
                # Flow EPE is reliable objective-independent measure for controlled synthetic data.
                if val_metrics['epe_px']<best:
                    best=val_metrics['epe_px'];save('best.pt',step)
                    (out/'best_metrics.json').write_text(json.dumps(val_metrics,indent=2))
            if step%args.save_every==0:save(f'checkpoint_step_{step:06d}.pt',step)
    save('final.pt',args.steps)
    # Independent test sequences were never used for model selection.
    test_records=records['test'][:min(20,len(records['test']))]
    if test_records:
        test_metrics=evaluate(net,head,mapper,test_records,args.dataset,cols,near,w,args,device)
        test_metrics.update({'step':args.steps,'objective':args.objective,'split':'test'})
        (out/'test_metrics.json').write_text(json.dumps(test_metrics,indent=2))
        print('Final held-out TEST:',json.dumps(test_metrics),flush=True)
    (out/'settings.json').write_text(json.dumps({k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},indent=2))
    (out/'validation_ids.json').write_text(json.dumps([r['id'] for r in val],indent=2))
    print(f'Finished. Best held-out EPE={best:.3f}px. Results at {out}',flush=True)
if __name__=='__main__':main()
