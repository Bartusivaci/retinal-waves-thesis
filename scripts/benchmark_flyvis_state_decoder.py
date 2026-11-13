"""Controlled synthetic motion benchmark: FlyVis final-state vs paired-state heads.

Run from the retinal-waves-thesis repository root:
  python -m scripts.benchmark_flyvis_state_decoder --steps 5 --run-name state_smoke
  python -m scripts.benchmark_flyvis_state_decoder --steps 300 --run-name state_300

- Photoreceptors receive synthetic *moving* textured scenes through BoxEye.
- FlyVis is loaded from checkpoints/flyvis_initial.pt and FROZEN.
- Exactly 15 inputs; reconstruct f15 by backward-warping f14.
- A15-only and [A14,A15,A15-A14] heads use the SAME data/loss/iterations.
- Ground truth is 1 pixel/frame translation at the 64x64 flow resolution.
- Reports reconstruction vs zero flow and EPE against known displacement.
This is a decoder-probe experiment, NOT evidence of FlyVis learning.
"""
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis.network import Network
from scripts.train_photoreceptor_finalstate import (
    ROOT, CELL_TYPES, PhotoreceptorWaveMapper, FinalStateFlowHead,
    warp_border, photo_loss, flow_smoothness,
    aligned_column_indices, nearest_hex_interpolation, grid_flow,
)

DIRECTIONS = {'right': (1, 0), 'left': (-1, 0), 'down': (0, 1), 'up': (0, -1)}


class PairedStateFlowHead(nn.Module):
    """Same hidden layers/width as the final-state head; extra temporal features."""
    def __init__(self, features, max_flow=4.0):
        super().__init__()
        self.max_flow = float(max_flow)
        self.net = nn.Sequential(
            nn.Linear(features * 3, 32), nn.Softplus(),
            nn.Linear(32, 16), nn.Softplus(), nn.Linear(16, 2)
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, a14, a15):
        features = torch.cat((a14, a15, a15 - a14), dim=-1)
        return self.max_flow * torch.tanh(self.net(features))


def scene(seed, device):
    """Produce textured static scene at 256x256 before rigid translation."""
    rng = np.random.default_rng(seed)
    yy, xx = torch.meshgrid(
        torch.arange(256, dtype=torch.float32, device=device),
        torch.arange(256, dtype=torch.float32, device=device), indexing='ij')
    pic = torch.full((256, 256), 0.07, device=device)
    for _ in range(17):
        cx, cy = rng.uniform(40, 216, size=2)
        sigma = float(rng.uniform(5, 20))
        strength = float(rng.uniform(0.25, 0.85))
        pic = pic + strength * torch.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma*sigma))
    return pic.clamp(0, 1)[None, None]


@torch.no_grad()
def synthetic_movie(seed, direction, device):
    """15 frames at 256x256. Motion = 4px per frame at 256 => 1px at 64."""
    dx, dy = DIRECTIONS[direction]
    base = scene(seed, device)
    seq = []
    for t in range(15):
        # position goes from -28 to +28 256px, constant speed 4px/frame.
        position = (t - 7) * 4
        backward = torch.zeros((1, 2, 256, 256), device=device)
        backward[:, 0] = -position * dx
        backward[:, 1] = -position * dy
        seq.append(warp_border(base, backward)[0])
    return torch.stack(seq).detach()  # [15,1,256,256]


def extract(network, mapper, cols, frames, device, dt):
    with torch.no_grad():
        stim = mapper(frames).to(device)
        network.eval()
        response = network(stim, dt=dt)
        # Each row is one hex column; neural features are CELL_TYPES.
        a14 = response[:, -2, cols].permute(0, 2, 1).detach()
        a15 = response[:, -1, cols].permute(0, 2, 1).detach()
        assert a14.shape[-1] == len(CELL_TYPES), (a14.shape, CELL_TYPES)
        source = F.interpolate(frames[13:14], size=(64, 64), mode='area').detach()
        target = F.interpolate(frames[14:15], size=(64, 64), mode='area').detach()
    return a14, a15, source, target


def flow_from_head(head, mode, item, near, weight):
    a14, a15, source, target, truth = item
    if mode == 'single':
        local_flow = head(a15)
    else:
        local_flow = head(a14, a15)
    return grid_flow(local_flow, near, weight, 64)


def assess(head, mode, items, near, weight, active_weight):
    head.eval()
    values = []
    with torch.no_grad():
        for item in items:
            a14, a15, source, target, truth = item
            flow = flow_from_head(head, mode, item, near, weight)
            loss = photo_loss(source, target, flow, active_weight)
            zero = photo_loss(source, target, torch.zeros_like(flow), active_weight)
            epe = (flow[:, :, 8:-8, 8:-8] - truth[:, :, 8:-8, 8:-8]).square().sum(dim=1).sqrt().mean()
            values.append((loss.item(), zero.item(), epe.item(), flow.abs().mean().item()))
    arr = np.asarray(values)
    l, z, e, mag = arr.mean(axis=0)
    return {'photo': float(l), 'zero_photo': float(z),
            'improvement_pct': float(100 * (z-l)/max(z, 1e-12)),
            'center_epe_px': float(e), 'mean_abs_flow_px': float(mag)}


def train_head(mode, items, val_items, near, weight, steps, lr, active_weight,
               smooth_weight, seed):
    torch.manual_seed(seed)
    device = items[0][0].device
    head = (FinalStateFlowHead(len(CELL_TYPES)) if mode == 'single'
            else PairedStateFlowHead(len(CELL_TYPES))).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for item in items:
            a14, a15, source, target, truth = item
            flow = flow_from_head(head, mode, item, near, weight)
            losses.append(photo_loss(source, target, flow, active_weight)
                          + smooth_weight * flow_smoothness(flow))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}: {mode}')
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            train = assess(head, mode, items, near, weight, active_weight)
            val = assess(head, mode, val_items, near, weight, active_weight)
            history.append({'step': step, 'train': train, 'heldout': val})
            print(f'{mode:6} step {step:4}: train EPE={train["center_epe_px"]:.3f}, '
                  f'heldout EPE={val["center_epe_px"]:.3f}, '
                  f'heldout improvement={val["improvement_pct"]:+.2f}%', flush=True)
    return head, history


def plot_case(head, mode, item, near, weight, path):
    with torch.no_grad():
        a14, a15, source, target, truth = item
        pred = flow_from_head(head, mode, item, near, weight)
        recon = warp_border(source, pred)
    panels = [(source[0, 0], 'Frame 14'), (target[0, 0], 'Frame 15'),
              (recon[0, 0], 'Reconstructed frame 15'),
              (truth[0, 0], 'True backward u'), (pred[0, 0], 'Predicted backward u'),
              ((recon-target)[0, 0].abs(), 'Absolute image error')]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (data, title) in zip(axes.flat, panels):
        ax.imshow(data.detach().cpu().numpy(), cmap='coolwarm' if 'backward' in title else 'gray')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--active-weight', type=float, default=5.0)
    parser.add_argument('--smooth-weight', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run-name', default='finalstate_vs_pairstate')
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error('steps must be positive, lr > 0')
    if '/' in args.run_name or '\\' in args.run_name:
        parser.error('run name must be a simple name')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = ROOT / 'results' / 'synthetic_flow' / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ROOT / 'checkpoints/flyvis_initial.pt', map_location='cpu', weights_only=True)
    net = Network()
    net.load_state_dict(ckpt['network'], strict=True)
    mapper = PhotoreceptorWaveMapper(net)
    cols = aligned_column_indices(net).to(device)
    near, weights = nearest_hex_interpolation(mapper, 64, device)
    net = net.to(device).eval()
    for param in net.parameters():
        param.requires_grad_(False)
    train_items, val_items = [], []
    for name in DIRECTIONS:
        for seed, dest in [(100 + list(DIRECTIONS).index(name), train_items),
                           (900 + list(DIRECTIONS).index(name), val_items)]:
            print(f'Extracting frozen FlyVis states: {name}, scene {seed}', flush=True)
            frames = synthetic_movie(seed, name, device)
            a14, a15, source, target = extract(net, mapper, cols, frames, device, args.dt)
            dx, dy = DIRECTIONS[name]
            true_flow = torch.zeros((1, 2, 64, 64), device=device)
            true_flow[:, 0] = -dx
            true_flow[:, 1] = -dy
            dest.append((a14, a15, source, target, true_flow))
    # Verify the synthetic labels and backward-warp direction before optimizing.
    for label, collection in [('train', train_items), ('heldout', val_items)]:
        for j, (a14, a15, source, target, truth) in enumerate(collection):
            with torch.no_grad():
                oracle = photo_loss(source, target, truth, args.active_weight).item()
                zero = photo_loss(source, target, torch.zeros_like(truth), args.active_weight).item()
            print(f'Oracle check {label} #{j}: correct-flow loss={oracle:.6f}; '
                  f'zero-flow loss={zero:.6f}', flush=True)
            if not (math.isfinite(oracle) and math.isfinite(zero) and oracle < zero):
                raise RuntimeError('Synthetic oracle does not beat zero-flow. '
                                   'Check stimulus generation / flow convention.')
    del net
    results = {'protocol': {'frames': 15, 'target': 'f15', 'source': 'f14',
                           'training': 'decoder only, FlyVis frozen from untrained baseline',
                           'motion': '4 cardinal directions, 1 px/frame at 64x64',
                           'dataset': 'four training textures and four held-out textures',
                           'dt_assumed_s': args.dt,
                           'objective': 'same weighted L1 + smoothness as retinal-wave pilot'},
               'models': {}}
    for mode in ['single', 'paired']:
        head, history = train_head(mode, train_items, val_items, near, weights,
                                   args.steps, args.lr, args.active_weight,
                                   args.smooth_weight, args.seed)
        results['models'][mode] = {
            'train': assess(head, mode, train_items, near, weights, args.active_weight),
            'heldout': assess(head, mode, val_items, near, weights, args.active_weight),
            'history': history,
        }
        plot_case(head, mode, val_items[0], near, weights, out / f'{mode}_right_heldout.png')
        torch.save({'head': head.state_dict(), 'kind': mode, 'settings': vars(args)},
                   out / f'{mode}_decoder_only.pt')
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
    print(f'\nCompleted. Results in {out}', flush=True)
    print('Compare heldout.center_epe_px first; reconstruction improvement second.')


if __name__ == '__main__':
    main()
