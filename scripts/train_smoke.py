"""Minimal *unsupervised* retinal-wave warping training smoke test.

From repository root:
  python -m scripts.train_smoke --steps 5

Never edits checkpoints/flyvis_initial.pt. Not a production training recipe.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flyvis.network import Network
from scipy.spatial import cKDTree

from src.neuron_stimulation import NeuronWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset

ROOT = Path(__file__).resolve().parent.parent
CELL_TYPES = ('L1', 'Mi1', 'Tm3', 'T4a', 'T5a')


class LocalFlowHead(nn.Module):
    """Shared local readout. Returns flow in pixels on the reduced image grid."""

    def __init__(self, n_features=5, max_flow=3.0):
        super().__init__()
        self.max_flow = float(max_flow)
        self.net = nn.Sequential(nn.Linear(n_features, 16), nn.Softplus(), nn.Linear(16, 2))
        # Small flow at initialization, but nonzero weights allow upstream gradients.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, responses):  # [B, 721, features]
        return self.max_flow * torch.tanh(self.net(responses))  # [B, 721, 2]


def aligned_column_indices(network):
    """Same 721 hex (u,v) locations, in L1's order, for all readout populations."""
    df = network.connectome.nodes.to_df().sort_values('index')
    labels = df['type'].astype(str).to_numpy()
    uv = list(zip(df['u'].to_numpy(), df['v'].to_numpy()))
    l1 = np.flatnonzero(labels == 'L1')
    keys = [uv[i] for i in l1]
    if len(keys) != 721 or len(set(keys)) != len(keys):
        raise ValueError('Expected 721 unique L1 hex locations')
    rows = []
    for name in CELL_TYPES:
        ids = np.flatnonzero(labels == name)
        lookup = {uv[i]: int(i) for i in ids}
        if len(lookup) != len(keys) or set(lookup) != set(keys):
            raise ValueError(f'{name} does not match the 721 L1 spatial columns')
        rows.append([lookup[key] for key in keys])
    return torch.as_tensor(np.array(rows), dtype=torch.long)  # [5,721]


def nearest_hex_interpolation(mapper, side, device):
    """Fixed three-nearest-neighbor interpolation from hex values to square pixels.

    Both hex coordinates and image grid use the same [-1, 1] convention as
    NeuronWaveMapper. This is an approximation, not a biological decoder.
    """
    hex_xy = mapper.grid.reshape(-1, 2).cpu().numpy()
    s = np.linspace(-1.0, 1.0, side)
    xx, yy = np.meshgrid(s, s, indexing='xy')
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    distances, inds = cKDTree(hex_xy).query(pixels, k=3)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (torch.tensor(inds, dtype=torch.long, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device))


def grid_flow(hex_flow, near_idx, near_weight, side):
    """[B,721,2] -> [B,2,side,side]; differentiable in hex_flow."""
    flow_pixels = (hex_flow[:, near_idx, :] * near_weight[None, :, :, None]).sum(dim=2)
    return flow_pixels.transpose(1, 2).reshape(hex_flow.shape[0], 2, side, side)


def warp_source_to_target(source, backward_flow):
    """source [B,1,H,W]; flow [B,2,H,W] is target->source, in pixels."""
    b, _, h, w = source.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype), indexing='ij')
    sx = xs[None] + backward_flow[:, 0]
    sy = ys[None] + backward_flow[:, 1]
    valid = ((sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)).unsqueeze(1)
    grid = torch.stack((2*sx/(w-1)-1, 2*sy/(h-1)-1), dim=-1)
    return F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True), valid


def losses(target, source, flow):
    reconstructed, valid = warp_source_to_target(source, flow)
    # Weight active wave regions to avoid a trivial mostly-black objective.
    with torch.no_grad():
        activity_weight = 0.1 + 0.9 * torch.maximum(target, source)
        mask = valid.float() * activity_weight
    photo = ((reconstructed - target).abs() * mask).sum() / mask.sum().clamp_min(1)
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return photo, dx + dy, reconstructed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--side', type=int, default=64)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--smooth-weight', type=float, default=0.01)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.steps < 1 or args.side < 16 or not (0 < args.dt <= 0.02) or args.gain <= 0:
        p.error('Require steps>=1, side>=16, 0<dt<=0.02 and gain>0')
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device(args.device)

    ds = RetinalWaveDataset(zip_path=ROOT / 'dataset/images.zip',
                            manifest_path=ROOT / 'wave_manifest.csv',
                            window_size=20, split='train')
    try:
        sample = ds[args.sample_index]
    finally:
        ds.close()
    frames = sample['frames']  # [T,1,256,256]
    network = Network()
    ckpt = torch.load(ROOT / 'checkpoints/flyvis_initial.pt',
                      map_location='cpu', weights_only=True)
    network.load_state_dict(ckpt['network'], strict=True)
    mapper = NeuronWaveMapper(network, target_types=('L1',), gain=args.gain)
    stimulus = mapper(frames)
    print('Stimulus:', mapper.verify(stimulus))
    network = network.to(device)
    stimulus = stimulus.to(device)
    columns = aligned_column_indices(network).to(device)
    neighbor, weights = nearest_hex_interpolation(mapper, args.side, device)
    head = LocalFlowHead(n_features=len(CELL_TYPES)).to(device)
    net_params = [p for p in network.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in net_params]
    optimizer = torch.optim.Adam([
        {'params': net_params, 'lr': 1e-5},
        {'params': head.parameters(), 'lr': 1e-4},
    ])

    images = F.interpolate(frames[-2:].to(device), size=(args.side, args.side),
                           mode='area')
    source = images[0:1]  # previous frame, warped to reconstruct last frame
    target = images[1:2]
    print('Training target last frame; warping previous frame. target=', tuple(target.shape))
    print('Model trains from saved untrained baseline; photoreceptors receive zero external input.')

    # Only a few steps on one fixed window: check end-to-end differentiability.
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        responses = network(stimulus, dt=args.dt)  # [1,T,N]
        last = responses[:, -1, :]  # last timepoint after viewing temporal window
        local_features = last[:, columns].permute(0, 2, 1)  # [B,721,5]
        local_flow = head(local_features)
        flow = grid_flow(local_flow, neighbor, weights, args.side)
        photo, smooth, reconstructed = losses(target, source, flow)
        total = photo + args.smooth_weight * smooth
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        total.backward()
        norm = math.sqrt(sum(float(p.grad.detach().square().sum().item())
                             for p in net_params if p.grad is not None))
        if step == 0 and (not math.isfinite(norm) or norm <= 0):
            raise RuntimeError('No FlyVis gradients: verify computational graph and stimulation.')
        torch.nn.utils.clip_grad_norm_(net_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        record = {'step': step+1, 'loss': float(total.item()),
                  'photometric': float(photo.item()), 'smoothness': float(smooth.item()),
                  'flyvis_grad_norm': norm, 'flow_mean_abs_px': float(flow.abs().mean().item())}
        history.append(record)
        print(record, flush=True)

    change = math.sqrt(sum(float((p.detach()-q).square().sum().item())
                           for p, q in zip(net_params, initial)))
    print(f'FlyVis parameter change L2 norm: {change:.9g}')
    if change <= 0:
        raise RuntimeError('FlyVis parameters did not change despite optimizer steps')
    out = ROOT / 'results' / 'training_smoke'
    out.mkdir(parents=True, exist_ok=True)
    # Save derived experiment ONLY. Never overwrite initial checkpoint.
    torch.save({'network': network.state_dict(), 'head': head.state_dict(),
                'sample_index': args.sample_index, 'settings': vars(args)}, out / 'smoke_checkpoint.pt')
    (out / 'metrics.json').write_text(json.dumps({'history': history,
        'flyvis_parameter_change_l2': change}, indent=2))
    print('PASS: reconstruction loss backpropagated to FlyVis, and weights changed.')
    print('Experiment artifacts:', out)


if __name__ == '__main__':
    main()
