"""Audit retinal-wave sampling and FlyVis hex-to-Cartesian flow geometry.

Run: python -m scripts.audit_wave_timing_geometry --windows 12
Does NOT train, modify checkpoints, or infer milliseconds from timestep IDs.
Requires existing scripts/train_smoke.py and src modules.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flyvis.network import Network
from flyvis.datasets.rendering import BoxEye
from scripts.train_smoke import ROOT, nearest_hex_interpolation, grid_flow
from scripts.train_photoreceptor_finalstate import PhotoreceptorWaveMapper, warp_border
from src.retinal_wave_dataset import RetinalWaveDataset


def movement_stats(frames, gaps):
    frames = F.interpolate(frames, size=(64, 64), mode='area').cpu()
    out = {}
    for gap in gaps:
        if gap >= len(frames):
            continue
        delta = (frames[gap:] - frames[:-gap]).abs()
        # Sparse activity can make whole-image means hide moving wavefronts.
        union = torch.maximum(frames[gap:], frames[:-gap])
        active = union > 0.1
        out[str(gap)] = {
            'mean_abs_difference': float(delta.mean()),
            'fraction_difference_gt_0.05': float((delta > .05).float().mean()),
            'active_fraction': float(active.float().mean()),
            'active_region_mean_abs_difference': float(delta[active].mean()) if active.any() else None,
        }
    return out


def make_texture(side):
    y, x = torch.meshgrid(torch.arange(side, dtype=torch.float32),
                          torch.arange(side, dtype=torch.float32), indexing='ij')
    z = torch.zeros_like(x) + .1
    for cx, cy, sigma in ((18,18,5),(36,20,3),(25,40,6),(48,46,4)):
        z += .6 * torch.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma))
    return z.clamp(0,1)[None,None]


def projection_test(mapper, side, device):
    near, w = nearest_hex_interpolation(mapper, side, device)
    # Constant Cartesian flow is exactly representable on hex samples. If even
    # this check fails, interpolation, component ordering, or warp is incorrect.
    source = make_texture(side).to(device)
    results = {}
    for name, ux, vy in [('right', -3., 0.), ('left', 3., 0.),
                         ('down', 0., -3.), ('up', 0., 3.)]:
        hexflow = torch.empty((1,721,2), device=device).fill_(0)
        hexflow[...,0] = ux; hexflow[...,1] = vy
        projected = grid_flow(hexflow, near, w, side)
        gt = torch.zeros_like(projected); gt[:,0] = ux; gt[:,1] = vy
        with torch.no_grad():
            recon = warp_border(source, projected)
            target = warp_border(source, gt)
        results[name] = {
            'max_flow_error_px': float((projected-gt).abs().max()),
            'mean_reconstruction_error': float((recon-target).abs().mean()),
        }
    # Spatially varying flow: compare interpolated field to an analytic vector
    # function at each pixel. This is approximate, not an exact oracle test.
    coords = mapper.grid.reshape(-1,2).to(device)
    hexflow = torch.stack((-2.0*coords[:,0], 1.5*coords[:,1]),dim=-1)[None]
    projected = grid_flow(hexflow, near, w, side)
    ys, xs = torch.meshgrid(torch.linspace(-1,1,side,device=device),
                            torch.linspace(-1,1,side,device=device),indexing='ij')
    analytic = torch.stack((-2.0*xs,1.5*ys),dim=0)[None]
    results['spatially_varying'] = {
        'mean_abs_flow_error_px': float((projected-analytic).abs().mean()),
        'max_abs_flow_error_px': float((projected-analytic).abs().max()),
        'comment': 'Approximation error is expected from hex-to-pixel interpolation.'
    }
    return results


def boxeye_spatial_probe(mapper, device, side=256):
    """Check whether 721 rendered columns align with the L1 geometry used by our decoder.
    This only tests position consistency and ordering, not full physical calibration.
    """
    eye = mapper.eye
    eye_device = eye.conv.weight.device
    y, x = torch.meshgrid(torch.linspace(-1,1,side),torch.linspace(-1,1,side),indexing='ij')
    xy = mapper.grid.reshape(-1,2).cpu().numpy()
    probes = [(-.45,0),(.45,0),(0,-.45),(0,.45),(-.35,-.35),(.35,.35)]
    results=[]
    for cx,cy in probes:
        blob = torch.exp(-((x-cx)**2+(y-cy)**2)/(2*.07**2))
        movie = blob[None,None].to(eye_device)
        with torch.no_grad():
            rendered=eye(movie).detach().cpu().reshape(-1).numpy()
        # Use top 10% signal above minimum; robust against all-positive offsets.
        strength=np.maximum(rendered-np.quantile(rendered,.1),0)
        if strength.sum() <= 1e-8:
            centroid=[None,None]; peak=[None,None]
        else:
            centroid=np.average(xy,axis=0,weights=strength).tolist()
            peak=xy[int(np.argmax(rendered))].tolist()
        results.append({'input_xy': [cx,cy], 'rendered_weighted_center_xy':centroid,
                        'peak_hex_xy':peak, 'peak_value':float(rendered.max()),
                        'minimum_value':float(rendered.min())})
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--windows',type=int,default=12)
    ap.add_argument('--window-size',type=int,default=20)
    ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--run-name',default='timing_geometry_audit')
    args=ap.parse_args()
    if args.windows<1 or args.window_size<=10: ap.error('Need windows>=1 and window-size>10')
    out=ROOT/'results'/'diagnostics'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'wave_manifest.csv'
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    count=Counter(); lengths=[]; n_rows=0
    with manifest.open(newline='') as f:
        for row in csv.DictReader(f):
            n_rows+=1
            n=int(row['frame_count']); lengths.append(n)
            # Saved frames are contiguous within a segment by construction;
            # this is an index/timestep audit, NOT biological timing in ms.
            count[int(row['end_timestep'])-int(row['start_timestep'])-(n-1)]+=1
    report={'manifest':{'segments':n_rows,'segment_length_median':float(np.median(lengths)),
                        'timestep_continuity_offset_counts':dict(count),
                        'physical_frame_interval_ms':None,
                        'warning':'No frame-to-millisecond mapping is present in wave_manifest.csv. Locate generator metadata or source code before interpreting dt physically.'}}
    dataset=RetinalWaveDataset(ROOT/'dataset/images.zip',manifest,window_size=args.window_size,split='train')
    try:
        rng=random.Random(123)
        inds=rng.sample(range(len(dataset)), min(args.windows,len(dataset)))
        samples=[]
        for idx in inds:
            item=dataset[idx]
            samples.append({'index':idx,'folder':item['folder'],'segment_id':item['segment_id'],
                            'start_timestep':item['start_timestep'],
                            'gaps':movement_stats(item['frames'],[1,3,5,10])})
        report['sampled_windows']=samples
    finally: dataset.close()
    network=Network()
    mapper=PhotoreceptorWaveMapper(network)
    report['hex_projection']=projection_test(mapper,64,torch.device(args.device))
    report['boxeye_spatial_probes']=boxeye_spatial_probe(mapper,torch.device(args.device))
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print('Manifest contains simulation timestep indices, not a known physical interval in milliseconds.')
    print('Hex projection:',json.dumps(report['hex_projection'],indent=2))
    print('BoxEye probes:',json.dumps(report['boxeye_spatial_probes'],indent=2))
    print('Saved',out/'audit.json')

if __name__=='__main__': main()
