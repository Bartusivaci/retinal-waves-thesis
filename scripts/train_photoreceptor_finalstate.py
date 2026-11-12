"""Photoreceptor-input final-state FlyVis retinal-wave training pilot.

Run in repository root:
    python -m scripts.train_photoreceptor_finalstate --steps 10 --val-windows 4 --validate-every 5 --save-every 5 --run-name finalstate_smoke
    python -m scripts.train_photoreceptor_finalstate --steps 1000 --run-name finalstate_gap1_1000

Requires existing scripts/train_smoke.py, src/*, dataset/images.zip,
wave_manifest.csv, checkpoints/flyvis_initial.pt.

Retinal waves enter via BoxEye PHOTORECEPTORS (not direct L1).
The fixed base checkpoint is read only; no pretrained checkpoint is loaded.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flyvis.network import Network
from flyvis.datasets.rendering import BoxEye

from scripts.train_smoke import (ROOT, CELL_TYPES, aligned_column_indices,
    nearest_hex_interpolation, grid_flow)
from src.neuron_stimulation import NeuronWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset


class PhotoreceptorWaveMapper:
    """Same wave frames; BoxEye sends external input only to photoreceptors.

    L1 coordinates are used solely as the *output flow grid*.
    """
    def __init__(self, network, target_types=('L1',), gain=1.0):
        self.network = network
        self.eye = BoxEye(extent=15, kernel_size=13)
        # Identical flow-grid coordinates to the direct-L1 experiment.
        self.grid = NeuronWaveMapper(network, target_types=('L1',)).grid
        nodes = network.connectome.nodes.to_df().sort_values('index')
        roles = nodes['role'].astype(str).to_numpy()
        self.photo_indices = torch.as_tensor(np.flatnonzero(roles == 'input'), dtype=torch.long)
        if not len(self.photo_indices):
            raise RuntimeError('No input-role photoreceptors found')
        self.n_nodes = int(network.n_nodes)
        self.gain = float(gain)

    @torch.no_grad()
    def __call__(self, frames):
        if frames.ndim == 4:  # [T,1,H,W]
            frames = frames.unsqueeze(0)
        if frames.ndim != 5 or frames.shape[2] != 1 or frames.shape[0] != 1:
            raise ValueError(f'Expected [1,T,1,H,W], got {tuple(frames.shape)}')
        # BoxEye expects [B,T,H,W] in 0..1. Match our first-vision evaluation.
        eye_device = self.eye.conv.weight.device
        rendered = self.eye(frames[:, :, 0].float().to(eye_device)).detach().cpu()
        if rendered.shape[-1] != 721:
            raise RuntimeError(f'Unexpected BoxEye output: {tuple(rendered.shape)}')
        self.network.stimulus.zero(n_samples=1, n_frames=frames.shape[1])
        self.network.stimulus.add_input(rendered * self.gain)
        return self.network.stimulus().detach()

    def verify(self, stimulus):
        photo = self.photo_indices.to(stimulus.device)
        nonphoto = torch.ones(self.n_nodes, dtype=torch.bool, device=stimulus.device)
        nonphoto[photo] = False
        if stimulus.shape[-1] != self.n_nodes:
            raise AssertionError('Incorrect number of network nodes')
        if not torch.all(stimulus[..., nonphoto] == 0):
            raise AssertionError('External input unexpectedly reaches non-photoreceptors')
        if stimulus[..., photo].abs().max() <= 0:
            raise AssertionError('No photoreceptor input')
        return dict(photoreceptors=int(photo.numel()),
                    nonzero_photoreceptor_values=int((stimulus[..., photo] != 0).sum()),
                    max_external_drive=float(stimulus.abs().max()))


class FinalStateFlowHead(nn.Module):
    """Predict target-to-source backward flow from the final recurrent state only.

    This probes whether FlyVis state carries recent-motion information without
    explicitly handing the decoder both source and target neural states.
    """
    def __init__(self, n_features, max_flow=4.0):
        super().__init__()
        self.max_flow = max_flow
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.Softplus(),
            nn.Linear(32, 16), nn.Softplus(), nn.Linear(16, 2)
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, final_features):
        return self.max_flow * torch.tanh(self.net(final_features))


def warp_border(source, backward_flow):
    """Differentiable full-image backward warp, pixel units, border padding."""
    b, _, h, w = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype), indexing='ij')
    x = xx[None] + backward_flow[:, 0]
    y = yy[None] + backward_flow[:, 1]
    grid = torch.stack((2*x/(w-1)-1, 2*y/(h-1)-1), dim=-1)
    return F.grid_sample(source, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


def photo_loss(source, target, flow, active_weight=5.0):
    reconstructed = warp_border(source, flow)
    weight = (1 + active_weight * torch.maximum(source, target)).detach()
    return ((reconstructed - target).abs() * weight).sum() / weight.sum()


def flow_smoothness(flow):
    return ((flow[:, :, 1:] - flow[:, :, :-1]).abs().mean()
            + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean())


def get_pair(frames, gap, side, device):
    # frames [T,1,H,W], source is older frame, target is the final frame.
    images = F.interpolate(frames[[-gap-1, -1]].to(device),
                           size=(side, side), mode='area')
    return images[0:1], images[1:2]


def predict(network, head, stimulus, columns, neighbor, weights, side, dt, gap):
    # Input consists of 15 *consecutive* original wave frames, no automatic
    # resampling. With gap=1, the final response contains f1..f15 history.
    responses = network(stimulus, dt=dt)
    final_activity = responses[:, -1, :]
    final_features = final_activity[:, columns].permute(0, 2, 1)
    local_flow = head(final_features)
    return grid_flow(local_flow, neighbor, weights, side)


def get_random_samples(dataset, number, generator):
    if number > len(dataset):
        raise ValueError(f'Requested {number} samples from dataset of {len(dataset)}')
    return generator.sample(range(len(dataset)), number)


def evaluate(network, head, mapper, dataset, validation_specs,
             columns, neighbor, weights, args, device):
    # Baseline zero flow is evaluated on EXACTLY the same pixels and pairs.
    network.eval(); head.eval()
    totals = {gap: {'model': [], 'zero': [], 'flow': []} for gap in args.eval_gaps}
    with torch.no_grad():
        for sample_idx, gap in validation_specs:
            sample = dataset[sample_idx]
            frames = sample['frames']
            stimulus = mapper(frames).to(device)
            source, target = get_pair(frames, gap, args.side, device)
            predicted = predict(network, head, stimulus, columns, neighbor,
                                weights, args.side, args.dt, gap)
            zero = torch.zeros_like(predicted)
            totals[gap]['model'].append(float(photo_loss(source, target, predicted,
                                              args.active_weight).item()))
            totals[gap]['zero'].append(float(photo_loss(source, target, zero,
                                              args.active_weight).item()))
            totals[gap]['flow'].append(float(predicted.abs().mean().item()))
    network.train(); head.train()
    result = {}
    for gap, values in totals.items():
        if not values['model']:
            continue
        model_loss = float(np.mean(values['model']))
        zero_loss = float(np.mean(values['zero']))
        result[str(gap)] = {
            'samples': len(values['model']),
            'model_weighted': model_loss,
            'zero_weighted': zero_loss,
            'improvement_percent': 100 * (zero_loss - model_loss) / max(zero_loss, 1e-12),
            'mean_abs_flow_pixels': float(np.mean(values['flow']))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--val-windows', type=int, default=30,
                        help='Total fixed validation pairs, distributed evenly across evaluation gaps')
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--train-gap', type=int, default=1, help='Final consecutive-frame reconstruction (must be 1)')
    parser.add_argument('--eval-gaps', type=int, nargs='+', default=[1], help='Fixed to gap=1 in final-state pilot')
    parser.add_argument('--window-size', type=int, default=15)
    parser.add_argument('--side', type=int, default=64)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--active-weight', type=float, default=5.0)
    parser.add_argument('--smooth-weight', type=float, default=0.01)
    parser.add_argument('--network-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--run-name', default='photoreceptor_finalstate_gap1_1000')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if (args.steps < 1 or args.val_windows < len(args.eval_gaps) or
        args.validate_every < 1 or args.save_every < 1 or
        args.window_size <= max([args.train_gap, *args.eval_gaps]) or min([args.train_gap, *args.eval_gaps]) < 1 or
        len(set(args.eval_gaps)) != len(args.eval_gaps) or args.side < 16):
        parser.error('Check steps, validation count, gaps, window-size, and side')
    if '/' in args.run_name or '\\' in args.run_name or args.run_name in ('', '.', '..'):
        parser.error('run-name must be a simple directory name')

    if args.train_gap != 1 or args.eval_gaps != [1]:
        parser.error('This pilot uses consecutive frames: --train-gap 1 --eval-gaps 1')
    out = ROOT / 'results' / 'multi_window' / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.pt').exists() or (out / 'history.csv').exists():
        raise FileExistsError(f'Run already exists: {out}. Choose another --run-name.')
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = RetinalWaveDataset(ROOT / 'dataset/images.zip', ROOT / 'wave_manifest.csv',
                                  window_size=args.window_size, split='train')
    val_ds = RetinalWaveDataset(ROOT / 'dataset/images.zip', ROOT / 'wave_manifest.csv',
                                window_size=args.window_size, split='val')
    rng = random.Random(args.seed)
    # FIXED validation pairs, split across evaluation gaps, no train-folder overlap.
    val_specs = []
    counts = [args.val_windows // len(args.eval_gaps) + (i < args.val_windows % len(args.eval_gaps))
              for i in range(len(args.eval_gaps))]
    val_indices = get_random_samples(val_ds, args.val_windows, rng)
    cursor = 0
    for gap, n in zip(args.eval_gaps, counts):
        val_specs.extend((idx, gap) for idx in val_indices[cursor:cursor+n]); cursor += n

    network = Network()
    ckpt = torch.load(ROOT / 'checkpoints/flyvis_initial.pt',
                      map_location='cpu', weights_only=True)
    network.load_state_dict(ckpt['network'], strict=True)
    mapper = PhotoreceptorWaveMapper(network, gain=args.gain)
    cols = aligned_column_indices(network).to(device)
    near, w = nearest_hex_interpolation(mapper, args.side, device)
    network = network.to(device)
    head = FinalStateFlowHead(len(CELL_TYPES)).to(device)
    net_params = [p for p in network.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in net_params]
    optimizer = torch.optim.Adam([
        {'params': net_params, 'lr': args.network_lr},
        {'params': head.parameters(), 'lr': args.head_lr},
    ])
    print('Fresh baseline; BoxEye photoreceptors; final-state-only decoder; f14 -> f15.', flush=True)
    print('TIMING CAVEAT: 1 dataset frame per dt=%.3f s, frame interval not verified.' % args.dt, flush=True)
    print(f'Datasets: train={len(train_ds):,}, val={len(val_ds):,}; device={device}', flush=True)
    print(f'Training exclusively with gap={args.train_gap}; validation on {len(val_specs)} fixed pairs, gaps={args.eval_gaps}', flush=True)

    def save_checkpoint(filename, step):
        torch.save({'network': network.state_dict(), 'head': head.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step,
                    'settings': vars(args), 'head_type': 'FinalStateFlowHead',
                    'cell_types': CELL_TYPES, 'input_pathway': 'BoxEye photoreceptors',
                    'timing_assumption': 'one dataset frame per solver step; dataset frame interval unverified',
                    'flow_target': 'warp f14 to f15 from final recurrent state'}, out / filename)

    # Choose best checkpoint by held-out gap-1 improvement over zero-flow.
    best_improvement = -float('inf')
    best_step = None
    history_path = out / 'history.csv'
    fieldnames = ['step', 'gap', 'folder', 'segment_id', 'source_frame', 'target_frame',
                  'source_timestep', 'target_timestep', 'total', 'photo', 'smooth',
                  'zero_photo', 'flow_mean_abs_px', 'network_grad_norm', 'network_change_l2']
    try:
        with history_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(1, args.steps + 1):
                sample_index = rng.randrange(len(train_ds))
                gap = args.train_gap
                sample = train_ds[sample_index]
                frames = sample['frames']
                stim_cpu = mapper(frames)
                if step == 1:
                    print('Stimulus check:', mapper.verify(stim_cpu), flush=True)
                stimulus = stim_cpu.to(device)
                source, target = get_pair(frames, gap, args.side, device)
                optimizer.zero_grad(set_to_none=True)
                predicted = predict(network, head, stimulus, cols, near, w,
                                    args.side, args.dt, gap)
                photo = photo_loss(source, target, predicted, args.active_weight)
                smooth = flow_smoothness(predicted)
                total = photo + args.smooth_weight * smooth
                if not torch.isfinite(total):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                total.backward()
                grad = math.sqrt(sum(float(p.grad.square().sum().item())
                                     for p in net_params if p.grad is not None))
                if not math.isfinite(grad) or grad == 0:
                    raise RuntimeError(f'FlyVis gradient missing/nonfinite at step {step}')
                torch.nn.utils.clip_grad_norm_(net_params, 1.0)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                change = math.sqrt(sum(float((p.detach()-q).square().sum().item())
                                       for p, q in zip(net_params, initial_params)))
                with torch.no_grad():
                    zero_photo = float(photo_loss(source, target, torch.zeros_like(predicted),
                                                  args.active_weight).item())
                row = {'step': step, 'gap': gap, 'folder': sample['folder'],
                       'segment_id': sample['segment_id'],
                       'source_frame': sample['start_frame']+len(frames)-1-gap,
                       'target_frame': sample['start_frame']+len(frames)-1,
                       'source_timestep': sample['start_timestep']+len(frames)-1-gap,
                       'target_timestep': sample['start_timestep']+len(frames)-1,
                       'total': float(total.item()), 'photo': float(photo.item()),
                       'smooth': float(smooth.item()), 'zero_photo': zero_photo,
                       'flow_mean_abs_px': float(predicted.detach().abs().mean().item()),
                       'network_grad_norm': grad, 'network_change_l2': change}
                writer.writerow(row); file.flush()
                if step == 1 or step % 100 == 0:
                    print('Train:', row, flush=True)
                if step % args.validate_every == 0 or step == args.steps:
                    result = evaluate(network, head, mapper, val_ds, val_specs,
                                      cols, near, w, args, device)
                    report_path = out / f'validation_step_{step:06d}.json'
                    report_path.write_text(json.dumps({'step': step, 'gaps': result}, indent=2))
                    print('Validation:', json.dumps(result), flush=True)
                    primary = result[str(args.train_gap)]
                    score = primary['improvement_percent']
                    if not math.isfinite(score):
                        raise RuntimeError(f'Nonfinite validation improvement at step {step}')
                    if score > best_improvement:
                        best_improvement, best_step = score, step
                        save_checkpoint('best.pt', step)
                        (out / 'best_metrics.json').write_text(json.dumps(
                            {'step': best_step, 'gap': args.train_gap,
                             'improvement_percent': best_improvement, 'validation': result}, indent=2))
                        print(f'New best gap-{args.train_gap} validation: {score:.3f}% at step {step}', flush=True)
                if step % args.save_every == 0:
                    save_checkpoint(f'checkpoint_step_{step:06d}.pt', step)
        save_checkpoint('final.pt', args.steps)
        (out / 'settings.json').write_text(json.dumps(vars(args), indent=2))
        (out / 'validation_pairs.json').write_text(json.dumps(val_specs, indent=2))
        print(f'Completed. Best checkpoint step={best_step}, gap-{args.train_gap} improvement={best_improvement:.3f}%. All outputs in {out}', flush=True)
    finally:
        train_ds.close(); val_ds.close()

if __name__ == '__main__':
    main()
