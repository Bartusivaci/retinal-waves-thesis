"""Compare wave-driven vs zero-input FlyVis activity from the identical baseline.

Run from repository root:
    python -m scripts.analyze_neural_activity
    python -m scripts.analyze_neural_activity --gain 0.25 --sample-index 10

Outputs under results/neural_activity/; no model weights are changed.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from flyvis.network import Network

from src.neuron_stimulation import NeuronWaveMapper
from src.retinal_wave_dataset import RetinalWaveDataset

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default=str(ROOT / 'results/neural_activity'))
    args = p.parse_args()
    if not (math.isfinite(args.dt) and 0 < args.dt <= 0.02):
        p.error('--dt must be in (0, 0.02]')

    torch.manual_seed(42)
    device = torch.device(args.device)
    dataset = RetinalWaveDataset(
        zip_path=ROOT / 'dataset/images.zip',
        manifest_path=ROOT / 'wave_manifest.csv',
        window_size=20, split='train',
    )
    try:
        sample = dataset[args.sample_index]
    finally:
        dataset.close()
    frames = sample['frames']
    network = Network()
    checkpoint = torch.load(ROOT / 'checkpoints/flyvis_initial.pt',
                            map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['network'], strict=True)
    network.eval().to(device)

    mapper = NeuronWaveMapper(network, target_types=('L1',), gain=args.gain)
    wave_drive_cpu = mapper(frames)
    print('External drive verification:', mapper.verify(wave_drive_cpu))
    wave_drive = wave_drive_cpu.to(device)
    zero_drive = torch.zeros_like(wave_drive)

    # forward(state=None) uses the same network-defined initial state for both.
    # No visual image enters the photoreceptor channel during either simulation.
    with torch.no_grad():
        baseline = network(zero_drive, dt=args.dt)
        wave = network(wave_drive, dt=args.dt)
    assert wave.shape == baseline.shape == wave_drive.shape
    if not torch.isfinite(wave).all() or not torch.isfinite(baseline).all():
        raise RuntimeError('Nonfinite neural responses; check stimulation gain/dt')
    difference = (wave - baseline).squeeze(0).cpu().numpy()  # [T,N]
    baseline_np = baseline.squeeze(0).cpu().numpy()
    wave_np = wave.squeeze(0).cpu().numpy()

    nodes = network.connectome.nodes.to_df().sort_values('index')
    type_labels = nodes['type'].astype(str).to_numpy()
    u = nodes['u'].to_numpy(dtype=np.float64)
    v = nodes['v'].to_numpy(dtype=np.float64)
    xx = u + 0.5 * v
    yy = (math.sqrt(3) / 2) * v
    roles = nodes['role'].astype(str).to_numpy()
    populations = ['L1', 'Mi1', 'Tm3', 'T4a', 'T5a']
    # All cells designated inputs by connectome (including R1-R8).
    groups = [(name, np.flatnonzero(type_labels == name)) for name in populations]
    groups.append(('Photoreceptors (input role)', np.flatnonzero(roles == 'input')))
    groups = [(name, ids) for name, ids in groups if ids.size]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(wave.shape[1]) * args.dt
    summary = {'sample': {k: (v if not torch.is_tensor(v) else list(v.shape))
                          for k, v in sample.items()},
               'device': str(device), 'dt': args.dt, 'gain': args.gain,
               'groups': {}}
    curves = {}
    for name, ids in groups:
        d = difference[:, ids]
        # Signed means can cancel; mean absolute deviation measures response change.
        mean_abs = np.abs(d).mean(axis=1)
        signed = d.mean(axis=1)
        curves[name] = (mean_abs, signed)
        summary['groups'][name] = {
            'neurons': int(ids.size),
            'mean_abs_change': float(np.abs(d).mean()),
            'peak_abs_change': float(np.abs(d).max()),
            'mean_signed_change': float(d.mean()),
            'baseline_mean_activity': float(baseline_np[:, ids].mean()),
            'wave_mean_activity': float(wave_np[:, ids].mean()),
        }
        print(f'{name:27s} n={ids.size:5d} | mean|Δ|={np.abs(d).mean():.6g} '
              f'peak|Δ|={np.abs(d).max():.6g}')

    with (output / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    with (output / 'activity_curves.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s'] + [f'{name} mean_abs_delta' for name in curves])
        for i, time in enumerate(times):
            writer.writerow([float(time)] + [float(x[0][i]) for x in curves.values()])

    fig, axs = plt.subplots(len(curves), 1, figsize=(10, 2.1*len(curves)), sharex=True)
    if len(curves) == 1:
        axs = [axs]
    for ax, (name, (mean_abs, _)) in zip(axs, curves.items()):
        ax.plot(times, mean_abs, marker='.', linewidth=1.2)
        ax.set_ylabel(name + '\nmean |Δ|', fontsize=9)
        ax.grid(alpha=0.25)
    axs[-1].set_xlabel('Simulation time (s); dt is provisional')
    fig.suptitle('FlyVis: neural response difference, wave vs zero external drive')
    fig.tight_layout()
    fig.savefig(output / 'population_response_curves.png', dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (name, ids) in zip(axs.flat, groups):
        delta_last = difference[-1, ids]
        # Signed effects can reveal excitation as well as suppression.
        limit = max(float(np.max(np.abs(delta_last))), 1e-9)
        scat = ax.scatter(xx[ids], yy[ids], c=delta_last, cmap='coolwarm',
                          vmin=-limit, vmax=limit, s=6, linewidths=0)
        ax.set_title(f'{name} (last frame)')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(scat, ax=ax, shrink=0.7, label='Wave - zero activity')
    for ax in list(axs.flat)[len(groups):]:
        ax.axis('off')
    fig.suptitle('Spatial activity changes (last frame; independent color scales)')
    fig.savefig(output / 'spatial_response_maps.png', dpi=160)
    plt.close(fig)

    print('\nSaved analysis to:', output)
    print('No training, no parameter updates, no checkpoint modification.')


if __name__ == '__main__':
    main()
