"""Verify BoxEye column positions against the flow decoder's hex spatial coordinates.

From repository root:
    python -m scripts.verify_boxeye_coordinates --run-name boxeye_alignment

Read-only: Does NOT modify weights, data, existing outputs, or training scripts.
Requires existing scripts/train_smoke.py and scripts/train_photoreceptor_finalstate.py.

Tests 8 possible signed axis permutations against asymmetric Gaussian probes.
The best fit is an empirical coordinate convention, not a proof of anatomy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flyvis.network import Network
from scripts.train_smoke import ROOT, nearest_hex_interpolation, grid_flow
from scripts.train_photoreceptor_finalstate import PhotoreceptorWaveMapper


def transforms(xy):
    x, y = xy[:,0], xy[:,1]
    return {
        'identity_xy': np.column_stack((x,y)),
        'flip_x': np.column_stack((-x,y)),
        'flip_y': np.column_stack((x,-y)),
        'flip_xy': np.column_stack((-x,-y)),
        'swap_xy': np.column_stack((y,x)),
        'swap_flip_x': np.column_stack((-y,x)),
        'swap_flip_y': np.column_stack((y,-x)),
        'swap_flip_xy': np.column_stack((-y,-x)),
    }


def gaussian_probe(cx, cy, side=256, sigma=0.09, device='cpu'):
    pos = torch.linspace(-1.,1.,side, device=device)
    yy,xx = torch.meshgrid(pos,pos,indexing='ij')
    return torch.exp(-0.5*((xx-cx)**2+(yy-cy)**2)/(sigma*sigma))


def normalized_corr(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    a=a-a.mean();b=b-b.mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>1e-12 else 0.


def spatial_center(values,xy):
    base = np.quantile(values,0.10)
    weights = np.maximum(values-base,0)
    return ((weights[:,None]*xy).sum(axis=0)/weights.sum()).tolist() if weights.sum()>1e-8 else [None,None]


def make_figure(out, recordings, best, raw_grid):
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ax, rec in zip(axs.ravel(),recordings):
        xy=rec['grid_by_transform'][best]
        im=ax.scatter(xy[:,0],xy[:,1],c=rec['rendered'],cmap='viridis',s=9)
        cx,cy=rec['input_xy']
        ax.plot(cx,cy,'rx',markersize=10,mew=2,label='input Gaussian center')
        ax.set(xlim=(-1.1,1.1),ylim=(1.1,-1.1),aspect='equal',title=f'Input ({cx:+.2f},{cy:+.2f})')
    fig.suptitle(f'BoxEye receptor response plotted using {best} (image y down)')
    fig.tight_layout()
    fig.savefig(out/'best_transform_probes.png',dpi=170)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-name',default='boxeye_alignment')
    ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--side',type=int,default=256)
    args=ap.parse_args()
    if args.side<64: ap.error('side must be >=64')
    out=ROOT/'results'/'diagnostics'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    # No pretrained network weights are loaded; Network() is used only for
    # connectome node ordering and the usual 721-column coordinate grid.
    net=Network()
    mapper=PhotoreceptorWaveMapper(net)
    eye=mapper.eye
    eye_device=eye.conv.weight.device
    raw_grid=mapper.grid.reshape(-1,2).detach().cpu().numpy()
    if raw_grid.shape!=(721,2):raise RuntimeError(f'Unexpected grid shape: {raw_grid.shape}')
    candidates=transforms(raw_grid)

    # Asymmetric probe positions eliminate ambiguity from symmetric patterns.
    probes=[(-.55,-.32),(.41,-.20),(-.29,.48),(.57,.31),(.03,-.61),(-.62,.12)]
    recordings=[]
    for cx,cy in probes:
        image=gaussian_probe(cx,cy,side=args.side,device=eye_device)
        with torch.no_grad():
            # BoxEye expects [B,T,H,W] in 0..1
            obs=eye(image[None,None]).detach().cpu().numpy().reshape(-1)
        if obs.shape!=(721,):raise RuntimeError(f'Expected 721 BoxEye columns: {obs.shape}')
        rec={'input_xy':[cx,cy], 'rendered':obs, 'grid_by_transform':candidates}
        recordings.append(rec)

    results={}
    sigma=0.10
    for name,xy in candidates.items():
        cors=[]; dist=[]; centroids=[]
        for rec in recordings:
            cx,cy=rec['input_xy']
            signal=np.exp(-0.5*((xy[:,0]-cx)**2+(xy[:,1]-cy)**2)/(sigma*sigma))
            cors.append(normalized_corr(rec['rendered'],signal))
            centroid=spatial_center(rec['rendered'],xy)
            centroids.append(centroid)
            dist.append(float(np.linalg.norm(np.array(centroid)-np.array([cx,cy]))))
        results[name]={
            'mean_response_template_correlation':float(np.mean(cors)),
            'minimum_response_template_correlation':float(np.min(cors)),
            'mean_centroid_error_normalized_coordinates':float(np.mean(dist)),
            'individual_correlations':[float(v) for v in cors],
            'individual_centroids':centroids
        }
    ranking=sorted(results,key=lambda name:(-results[name]['mean_response_template_correlation'],results[name]['mean_centroid_error_normalized_coordinates']))
    best=ranking[0]
    for rank,name in enumerate(ranking,1):
        record=results[name]
        print(f'{rank:>2}. {name:<14}  corr={record["mean_response_template_correlation"]:.5f}  centroid_error={record["mean_centroid_error_normalized_coordinates"]:.5f}',flush=True)

    # Show concrete effect of using different coordinate transforms for
    # nonconstant flows. This is a positional mapping test, not a training run.
    side=64
    device=torch.device(args.device)
    from scipy.spatial import cKDTree
    s=np.linspace(-1.,1.,side)
    xpix,ypix=np.meshgrid(s,s,indexing='xy')
    pixels=np.stack([xpix.ravel(),ypix.ravel()],axis=-1)
    # Synthetic field f(x,y)=(2*x, -1.5*y) evaluated at each receptor's
    # *actual image position*, then interpolated using either guessed grid.
    actual_xy=candidates[best]
    hex_flow=np.stack([2*actual_xy[:,0], -1.5*actual_xy[:,1]],axis=-1)
    hex_flow_t=torch.from_numpy(hex_flow.astype('float32')).to(device)[None]
    oracle=torch.from_numpy(np.stack([2*xpix,-1.5*ypix],axis=0).astype('float32')).to(device)[None]
    positional={}
    for name,xy in candidates.items():
        ds,ids=cKDTree(xy).query(pixels,k=3)
        ws=1/np.maximum(ds,1e-6)
        ws/=ws.sum(axis=1,keepdims=True)
        ids_t=torch.tensor(ids,device=device,dtype=torch.long)
        ws_t=torch.tensor(ws,device=device,dtype=torch.float32)
        projected=grid_flow(hex_flow_t,ids_t,ws_t,side)
        error=(projected-oracle).abs()
        positional[name]={
            'mean_abs_error_flow_pixels':float(error.mean()),
            'max_abs_error_flow_pixels':float(error.max()),
        }
    make_figure(out,recordings,best,raw_grid)
    payload={
        'description':'BoxEye receptor signals compared against L1 connectome-derived hex grid under signed axis permutations',
        'recommended_transform_for_image_coordinates':best,
        'candidate_order':ranking,
        'candidate_scores':results,
        'spatially_varying_flow_projection':positional,
        'probes':probes,
        'warning':'A candidate transform succeeding establishes coordinate compatibility for these probes; it does not prove full biological coordinate calibration or correct decoder neural column order. Current 1000-step model must be evaluated with its original mapping before altering interpretation.'
    }
    (out/'coordinate_validation.json').write_text(json.dumps(payload,indent=2))
    print('\nSelected transform:',best)
    print('Spatially varying flow errors for identity vs selected:')
    for key in ('identity_xy',best):print(key,positional[key])
    print('Saved',out/'coordinate_validation.json')
    print('Saved',out/'best_transform_probes.png')

if __name__=='__main__':main()
