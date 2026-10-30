"""Fixed-pair frame-gap experiments for retinal-wave FlyVis.

Run from repository root:
    python -m scripts.compare_frame_gaps --steps 30 --gaps 1 3 5

Requires scripts.train_smoke.py and existing ZIP loader / neural mapper.
Creates separate output per gap; never overwrites the initial baseline.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flyvis.network import Network
from src.neuron_stimulation import NeuronWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset
from scripts.train_smoke import (
    CELL_TYPES, LocalFlowHead, aligned_column_indices,
    nearest_hex_interpolation, grid_flow, warp_source_to_target,
)

ROOT = Path(__file__).resolve().parent.parent


def model_flow(network, head, stim, columns, neighbors, weights, side, dt):
    responses = network(stim, dt=dt)
    features = responses[:, -1, :][:, columns].permute(0, 2, 1)
    return grid_flow(head(features), neighbors, weights, side)


def warp_border(source, flow):
    """Differentiable warping with border values; no incentive to mask pixels."""
    b, _, h, w = source.shape
    y, x = torch.meshgrid(torch.arange(h, device=source.device, dtype=source.dtype),
                          torch.arange(w, device=source.device, dtype=source.dtype), indexing='ij')
    sx = x[None] + flow[:, 0]
    sy = y[None] + flow[:, 1]
    grid = torch.stack((2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1), dim=-1)
    return F.grid_sample(source, grid, mode='bilinear', padding_mode='border', align_corners=True)


def score(target, source, flow, active_weight, shared_mask=None):
    warped = warp_border(source, flow)
    err = (warped - target).abs()
    weights = 1 + active_weight * torch.maximum(target, source)
    roi = (torch.maximum(target, source) > 0.1)
    if shared_mask is None:
        shared_mask = torch.ones_like(target, dtype=torch.bool)
    m = shared_mask.to(target.dtype)
    denom = m.sum().clamp_min(1)
    weighted = (err * weights * m).sum() / (weights * m).sum().clamp_min(1)
    roi_m = roi.to(target.dtype) * m
    return {
        'l1': float((err * m).sum().item() / denom.item()),
        'weighted': float(weighted.item()),
        'active_l1': float((err * roi_m).sum().item() / roi_m.sum().clamp_min(1).item()),
        'n_evaluated_pixels': int(m.sum().item()),
        'reconstruction': warped.detach(),
        'error': err.detach(),
    }


def shared_valid_mask(source, flows):
    """Intersection of in-bounds pixels of ALL three methods, used only for fair reporting."""
    valid = torch.ones_like(source, dtype=torch.bool)
    for flow in flows:
        _, this_valid = warp_source_to_target(source, flow)
        valid &= this_valid
    return valid


def compact(s):
    return {k: v for k, v in s.items() if not isinstance(v, torch.Tensor)}


def plot(target, source, scores, output, title):
    keys = list(scores)
    fig, axes = plt.subplots(2, 2 + len(keys), figsize=(3.5 * (2 + len(keys)), 7))
    for col, (name, tensor) in enumerate([('Target (new)', target), ('Source (old)', source)]):
        axes[0, col].imshow(tensor[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(name)
        axes[1, col].axis('off')
    for col, name in enumerate(keys, start=2):
        s = scores[name]
        axes[0, col].imshow(s['reconstruction'][0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(name)
        axes[1, col].imshow(s['error'][0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=.3)
        axes[1, col].set_title(f"L1={s['l1']:.5f}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def load_pair(ds, index, gap, side, device):
    sample = ds[index]
    frames = sample['frames']
    if gap >= len(frames):
        raise ValueError('Gap must be smaller than window_size')
    # Current window ends at start_frame + (window_size - 1).
    source_offset = len(frames) - 1 - gap
    target_offset = len(frames) - 1
    pair = torch.stack([frames[source_offset], frames[target_offset]])
    pair = F.interpolate(pair.to(device), size=(side, side), mode='area')
    metadata = {k: v for k, v in sample.items() if k != 'frames'}
    metadata.update(source_frame=int(sample['start_frame']) + source_offset,
                    target_frame=int(sample['start_frame']) + target_offset,
                    source_timestep=int(sample['start_timestep']) + source_offset,
                    target_timestep=int(sample['start_timestep']) + target_offset)
    return frames, pair[0:1], pair[1:2], metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gaps', type=int, nargs='+', default=[1, 3, 5])
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--train-index', type=int, default=0)
    p.add_argument('--val-index', type=int, default=0)
    p.add_argument('--side', type=int, default=64)
    p.add_argument('--window-size', type=int, default=20)
    p.add_argument('--active-weight', type=float, default=5.0)
    p.add_argument('--smooth-weight', type=float, default=0.01)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.steps < 0 or args.side < 16 or args.window_size <= max(args.gaps) or min(args.gaps) < 1:
        p.error('Require steps >= 0 and 1 <= gap < window-size, side >= 16')
    device = torch.device(args.device)
    rootout = ROOT / 'results' / 'frame_gap_diagnostics'
    rootout.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: RetinalWaveDataset(zip_path=ROOT / 'dataset/images.zip',
                                  manifest_path=ROOT / 'wave_manifest.csv',
                                  window_size=args.window_size, split=split)
        for split in ('train', 'val')
    }
    try:
        samples = {}
        for split, idx in [('train', args.train_index), ('val', args.val_index)]:
            sample = datasets[split][idx]
            samples[split] = sample
    finally:
        for ds in datasets.values():
            ds.close()
    if samples['train']['folder'] == samples['val']['folder']:
        raise RuntimeError('Train/val samples must come from different folders')

    summary = []
    for gap in args.gaps:
        print(f'\n========== GAP {gap} ==========', flush=True)
        torch.manual_seed(123); np.random.seed(123)
        network = Network()
        checkpoint = torch.load(ROOT / 'checkpoints/flyvis_initial.pt', map_location='cpu', weights_only=True)
        network.load_state_dict(checkpoint['network'], strict=True)
        mapper = NeuronWaveMapper(network, target_types=('L1',), gain=args.gain)
        columns = aligned_column_indices(network).to(device)
        neighbors, weights = nearest_hex_interpolation(mapper, args.side, device)
        packed = {}
        for split in ('train', 'val'):
            frame_tensor = samples[split]['frames']
            source_offset = len(frame_tensor)-1-gap
            pair = torch.stack([frame_tensor[source_offset], frame_tensor[-1]])
            pair = F.interpolate(pair.to(device), (args.side, args.side), mode='area')
            packed[split] = dict(source=pair[0:1], target=pair[1:2],
                                 stimulus=mapper(frame_tensor).to(device),
                                 frame_indices=dict(source_frame=int(samples[split]['start_frame']) + source_offset,
                                                    target_frame=int(samples[split]['start_frame']) + len(frame_tensor)-1,
                                                    source_timestep=int(samples[split]['start_timestep']) + source_offset,
                                                    target_timestep=int(samples[split]['start_timestep']) + len(frame_tensor)-1))
        print('Train pair:', packed['train']['frame_indices'], 'Validation pair:', packed['val']['frame_indices'])
        print('Direct input:', mapper.verify(mapper(samples['train']['frames'])))
        network = network.to(device)
        head = LocalFlowHead(n_features=len(CELL_TYPES)).to(device)
        net_params = [t for t in network.parameters() if t.requires_grad]
        start_params = [t.detach().clone() for t in net_params]
        optimizer = torch.optim.Adam([{'params': net_params, 'lr': 1e-5},
                                      {'params': head.parameters(), 'lr': 1e-4}])

        with torch.no_grad():
            initial_flows = {s: model_flow(network, head, d['stimulus'], columns, neighbors, weights,
                                          args.side, args.dt).detach().clone()
                             for s, d in packed.items()}
        history = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            d = packed['train']
            flow = model_flow(network, head, d['stimulus'], columns, neighbors, weights,
                              args.side, args.dt)
            warped = warp_border(d['source'], flow)
            err = (warped - d['target']).abs()
            weights_photo = (1 + args.active_weight * torch.maximum(d['source'], d['target'])).detach()
            photo = (err * weights_photo).sum() / weights_photo.sum()
            smooth = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() + \
                     (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
            loss = photo + args.smooth_weight * smooth
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training objective')
            loss.backward()
            grad = math.sqrt(sum(float(t.grad.square().sum().item()) for t in net_params if t.grad is not None))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError('No finite FlyVis gradient')
            torch.nn.utils.clip_grad_norm_(net_params, 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            history.append(dict(step=step+1, total=float(loss.item()), photo=float(photo.item()),
                                smooth=float(smooth.item()), flyvis_grad_norm=grad,
                                mean_abs_flow=float(flow.abs().mean().item())))
            if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
                print(history[-1], flush=True)

        result = {'settings': vars(args), 'gap': gap, 'pairs': {}, 'results': {}, 'history': history}
        with torch.no_grad():
            for split, d in packed.items():
                zero = torch.zeros_like(initial_flows[split])
                final_flow = model_flow(network, head, d['stimulus'], columns, neighbors, weights,
                                        args.side, args.dt)
                flows = {'zero': zero, 'untrained': initial_flows[split], 'trained': final_flow}
                mask = shared_valid_mask(d['source'], list(flows.values()))
                result['pairs'][split] = dict(folder=samples[split]['folder'],
                                             segment_id=samples[split]['segment_id'],
                                             **d['frame_indices'])
                result['results'][split] = dict(common_mask_fraction=float(mask.float().mean().item()), methods={})
                full_scores = {}
                for name, pred in flows.items():
                    # Full-image with border padding is the primary fair score.
                    full = score(d['target'], d['source'], pred, args.active_weight)
                    shared = score(d['target'], d['source'], pred, args.active_weight, mask)
                    full_scores[name] = full
                    result['results'][split]['methods'][name] = dict(
                        full=compact(full), common_mask=compact(shared),
                        mean_abs_flow_pixels=float(pred.abs().mean().item()))
                outdir = rootout / f'gap_{gap}'
                outdir.mkdir(parents=True, exist_ok=True)
                plot(d['target'], d['source'], full_scores, outdir / f'{split}_reconstructions.png',
                     f'{split}: gap {gap} frames | full-image errors, border padding')
                val = result['results'][split]['methods']
                print(f'{split}: zero weighted={val["zero"]["full"]["weighted"]:.6f}, '
                      f'untrained={val["untrained"]["full"]["weighted"]:.6f}, '
                      f'trained={val["trained"]["full"]["weighted"]:.6f}; '
                      f'common mask fraction={mask.float().mean().item():.3f}', flush=True)
        change = math.sqrt(sum(float((p.detach()-i).square().sum().item()) for p, i in zip(net_params, start_params)))
        result['flyvis_parameter_change_l2'] = change
        (rootout / f'gap_{gap}' / 'results.json').write_text(json.dumps(result, indent=2))
        summary.append({'gap': gap, 'parameter_change_l2': change, 'validation': result['results']['val']})
        del network, head, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (rootout / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nSaved:', rootout)
    print('Note: this fixed-pair experiment checks reconstruction, not recovered true optical flow.')

if __name__ == '__main__':
    main()
