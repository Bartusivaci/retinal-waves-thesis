"""Compare zero-flow, untrained, and trained FlyVis warping on retinal-wave pairs.

Run from repository root:
    python -m scripts.diagnose_flow --steps 30

Reuses scripts.train_smoke utilities; does not overwrite the baseline or smoke checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from flyvis.network import Network

from src.neuron_stimulation import NeuronWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset
from scripts.train_smoke import (
    CELL_TYPES, LocalFlowHead, aligned_column_indices,
    nearest_hex_interpolation, grid_flow, warp_source_to_target,
)

ROOT = Path(__file__).resolve().parent.parent


def metrics(target, source, flow, active_scale=5.0):
    """Measure both standard and activity-weighted error on VALID pixels only.

    Activity weights depend on original frames, never on prediction; they cannot
    be gamed by learning to change the mask. All errors are masked consistently.
    """
    warped, valid = warp_source_to_target(source, flow)
    valid_f = valid.to(target.dtype)
    err = (warped - target).abs()
    active = torch.maximum(target, source)
    weights = 1.0 + active_scale * active
    l1 = (err * valid_f).sum() / valid_f.sum().clamp_min(1)
    weighted = (err * valid_f * weights).sum() / (valid_f * weights).sum().clamp_min(1)
    # Focused region metric: union of reasonably active pixels in either frame.
    roi = valid_f * (active > 0.1).to(valid_f.dtype)
    roi_l1 = (err * roi).sum() / roi.sum().clamp_min(1)
    valid_fraction = valid_f.mean()
    smooth = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() + \
             (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return dict(l1=l1, weighted=weighted, active_l1=roi_l1,
                smooth=smooth, valid_fraction=valid_fraction,
                warped=warped, error=err)


def flow_from_network(network, head, stimulus, columns, neighbors, weights, side, dt):
    responses = network(stimulus, dt=dt)
    local_features = responses[:, -1, :][:, columns].permute(0, 2, 1)
    return grid_flow(head(local_features), neighbors, weights, side)


def load_window(dataset, index, side, device):
    sample = dataset[index]
    frames = sample['frames']
    imgs = F.interpolate(frames[-2:].to(device), size=(side, side), mode='area')
    # Direction: target is newest frame, source is previous frame. Flow target->source.
    return sample, frames, imgs[0:1], imgs[1:2]


def snapshot(m):
    return {k: float(v.detach().item()) for k, v in m.items()
            if k in ('l1', 'weighted', 'active_l1', 'smooth', 'valid_fraction')}


def plot_comparison(target, source, outputs, dest):
    methods = list(outputs)
    fig, axes = plt.subplots(3, len(methods) + 2, figsize=(4 * (len(methods) + 2), 10), squeeze=False)
    ref_imgs = [target[0, 0].detach().cpu().numpy(), source[0, 0].detach().cpu().numpy()]
    for col, (name, img) in enumerate(zip(('Target (new frame)', 'Source (old frame)'), ref_imgs)):
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(name)
    for row in range(1, 3):
        for col in range(2):
            axes[row, col].axis('off')
    for col, name in enumerate(methods, start=2):
        m = outputs[name]
        img = m['warped'][0, 0].detach().cpu().numpy()
        error = m['error'][0, 0].detach().cpu().numpy()
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'{name}: reconstruction')
        axes[1, col].imshow(error, cmap='magma', vmin=0, vmax=0.3)
        axes[1, col].set_title(f'Error (L1={m["l1"].item():.5f})')
        # Fixed shared scaling ensures actual visual comparison between methods.
        axes[2, col].imshow((error > 0.05).astype(np.float32), cmap='gray', vmin=0, vmax=1)
        axes[2, col].set_title('Pixels with error > 0.05')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--train-index', type=int, default=0)
    p.add_argument('--eval-index', type=int, default=0, help='Index within separate validation split')
    p.add_argument('--side', type=int, default=64)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--active-weight', type=float, default=5.0)
    p.add_argument('--smooth-weight', type=float, default=0.01)
    p.add_argument('--objective', choices=['weighted', 'l1'], default='weighted')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.steps < 0 or args.active_weight < 0 or args.side < 16:
        p.error('Require steps>=0, active-weight>=0, side>=16')
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device(args.device)
    outdir = ROOT / 'results' / 'flow_diagnostics'
    outdir.mkdir(parents=True, exist_ok=True)

    ds = RetinalWaveDataset(zip_path=ROOT / 'dataset/images.zip',
                            manifest_path=ROOT / 'wave_manifest.csv',
                            window_size=20, split='train')
    try:
        train_sample, train_frames, train_source, train_target = load_window(ds, args.train_index, args.side, device)
    finally:
        ds.close()
    val_ds = RetinalWaveDataset(zip_path=ROOT / 'dataset/images.zip',
                               manifest_path=ROOT / 'wave_manifest.csv',
                               window_size=20, split='val')
    try:
        eval_sample, eval_frames, eval_source, eval_target = load_window(val_ds, args.eval_index, args.side, device)
    finally:
        val_ds.close()
    print('Train sample:', {k:v for k,v in train_sample.items() if k != 'frames'})
    print('Eval sample:', {k:v for k,v in eval_sample.items() if k != 'frames'})
    assert train_sample['folder'] != eval_sample['folder'], 'Expected independent train/validation folders'

    network = Network()
    ckpt = torch.load(ROOT / 'checkpoints/flyvis_initial.pt', map_location='cpu', weights_only=True)
    network.load_state_dict(ckpt['network'], strict=True)
    mapper = NeuronWaveMapper(network, target_types=('L1',), gain=args.gain)
    train_stimulus = mapper(train_frames)
    eval_stimulus = mapper(eval_frames)
    print('External stimulus verification:', mapper.verify(train_stimulus))
    network = network.to(device)
    train_stimulus, eval_stimulus = train_stimulus.to(device), eval_stimulus.to(device)
    columns = aligned_column_indices(network).to(device)
    neighbors, weights = nearest_hex_interpolation(mapper, args.side, device)
    head = LocalFlowHead(n_features=len(CELL_TYPES)).to(device)
    net_params = [param for param in network.parameters() if param.requires_grad]
    initial_params = [param.detach().clone() for param in net_params]
    optimizer = torch.optim.Adam([{'params': net_params, 'lr': 1e-5},
                                  {'params': head.parameters(), 'lr': 1e-4}])

    def evaluate(source, target, stimulus):
        zeros = torch.zeros(target.shape[0], 2, args.side, args.side, device=device)
        z = metrics(target, source, zeros, args.active_weight)
        pred = flow_from_network(network, head, stimulus, columns, neighbors, weights, args.side, args.dt)
        m = metrics(target, source, pred, args.active_weight)
        return z, m, pred

    with torch.no_grad():
        before_train_zero, before_train, _ = evaluate(train_source, train_target, train_stimulus)
        before_eval_zero, before_eval, _ = evaluate(eval_source, eval_target, eval_stimulus)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        flow = flow_from_network(network, head, train_stimulus, columns, neighbors, weights, args.side, args.dt)
        m = metrics(train_target, train_source, flow, args.active_weight)
        primary = m[args.objective]
        # Guard against a method that reduces loss by pushing many samples out of bounds.
        penalty = (1.0 - m['valid_fraction']) * 0.01
        loss = primary + args.smooth_weight * m['smooth'] + penalty
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step+1}')
        loss.backward()
        grad_norm = math.sqrt(sum(float(x.grad.square().sum().item()) for x in net_params if x.grad is not None))
        if not math.isfinite(grad_norm) or grad_norm == 0:
            raise RuntimeError(f'Invalid FlyVis gradient norm: {grad_norm}')
        torch.nn.utils.clip_grad_norm_(net_params, 1.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        row = dict(step=step+1, total_loss=float(loss.item()), weighted=float(m['weighted'].item()),
                   l1=float(m['l1'].item()), active_l1=float(m['active_l1'].item()),
                   flow_abs_pixels=float(flow.abs().mean().item()),
                   valid_fraction=float(m['valid_fraction'].item()), flyvis_grad_norm=grad_norm)
        history.append(row)
        if (step+1) % max(1, args.steps//5) == 0 or step == 0:
            print(row, flush=True)

    with torch.no_grad():
        after_train_zero, after_train, train_flow = evaluate(train_source, train_target, train_stimulus)
        after_eval_zero, after_eval, eval_flow = evaluate(eval_source, eval_target, eval_stimulus)
    change = math.sqrt(sum(float((param.detach()-initial).square().sum().item())
                           for param, initial in zip(net_params, initial_params)))
    results = dict(settings=vars(args), train_sample={k:v for k,v in train_sample.items() if k != 'frames'},
                   eval_sample={k:v for k,v in eval_sample.items() if k != 'frames'},
                   train=dict(zero=snapshot(after_train_zero), untrained=snapshot(before_train), trained=snapshot(after_train)),
                   eval=dict(zero=snapshot(after_eval_zero), untrained=snapshot(before_eval), trained=snapshot(after_eval)),
                   flyvis_parameter_change_l2=change, history=history,
                   train_mean_abs_flow_px=float(train_flow.abs().mean().item()),
                   eval_mean_abs_flow_px=float(eval_flow.abs().mean().item()))
    (outdir / 'comparison.json').write_text(json.dumps(results, indent=2))
    plot_comparison(train_target, train_source,
                    {'Zero flow': after_train_zero, 'Untrained': before_train, 'Trained': after_train},
                    outdir / 'train_comparison.png')
    plot_comparison(eval_target, eval_source,
                    {'Zero flow': after_eval_zero, 'Untrained': before_eval, 'Trained': after_eval},
                    outdir / 'eval_comparison.png')
    print('\nCOMPARISON (lower is better; same source/target pairs):')
    for split in ('train', 'eval'):
        print(split)
        for kind, values in results[split].items():
            print(f'  {kind:>9}: L1={values["l1"]:.6f}; weighted={values["weighted"]:.6f}; '
                  f'active-only L1={values["active_l1"]:.6f}; valid={values["valid_fraction"]:.3f}')
    print('Mean absolute predicted flow (pixels): train', results['train_mean_abs_flow_px'],
          'eval', results['eval_mean_abs_flow_px'])
    print('FlyVis parameter change L2:', change)
    print('Results saved:', outdir)
    if results['eval']['trained']['weighted'] >= results['eval']['zero']['weighted']:
        print('WARNING: Trained flow does not beat zero flow on the diagnostic evaluation pair.')
    # No model checkpoint here: this is a diagnostic; full training has not been validated.

if __name__ == '__main__':
    main()
