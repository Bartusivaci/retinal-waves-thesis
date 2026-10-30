"""Untrained FlyVis retinal-wave propagation: GIF, peak maps and multiple windows.

From repo root:
    python -m scripts.inspect_wave_dynamics --num-windows 12
    python -m scripts.inspect_wave_dynamics --num-windows 20 --gain 0.5

Loads the original checkpoint, never changes model parameters.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from flyvis.network import Network

from src.retinal_wave_dataset import RetinalWaveDataset
from src.neuron_stimulation import NeuronWaveMapper

ROOT = Path(__file__).resolve().parent.parent
POPULATIONS = ('L1', 'Mi1', 'Tm3', 'T4a', 'T5a')


def select_indices(dataset, n, seed):
    """Favor distinct folders and include long segments when available.

    dataset.segments holds (folder, segment_id, zip_prefix, first_frame,
    first_time, window_count); dataset.cumulative gives global offsets.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for i, (folder, seg_id, _, _, _, count) in enumerate(dataset.segments):
        if count <= 0:
            continue
        base = dataset.cumulative[i - 1] if i else 0
        candidates.append((folder, base + int(rng.integers(count)), count, seg_id))
    rng.shuffle(candidates)
    # One from each folder first; prefer roughly half from long recordings.
    long = [x for x in candidates if x[2] >= 50]
    all_c = long + candidates
    result, used_folders, used_indices = [], set(), set()
    for folder, index, count, seg_id in all_c:
        if folder in used_folders or index in used_indices:
            continue
        result.append((index, count, seg_id))
        used_folders.add(folder)
        used_indices.add(index)
        if len(result) == n:
            break
    if len(result) < n:
        raise ValueError(f'Only {len(result)} distinct usable folders, requested {n}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-windows', type=int, default=12)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=str(ROOT / 'results' / 'neural_activity' / 'multi_window'))
    args = parser.parse_args()
    if args.num_windows < 1 or not math.isfinite(args.dt) or args.dt <= 0:
        parser.error('num-windows must be positive and dt must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = RetinalWaveDataset(ROOT / 'dataset/images.zip', ROOT / 'wave_manifest.csv',
                                window_size=20, split='train')
    selected = select_indices(dataset, args.num_windows, args.seed)
    print(f'Selected {len(selected)} windows from distinct folders; long segments prioritized.')

    network = Network()
    checkpoint = torch.load(ROOT / 'checkpoints/flyvis_initial.pt',
                            map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['network'], strict=True)
    network.eval().to(device)
    mapper = NeuronWaveMapper(network, target_types=('L1',), gain=args.gain)
    nodes = network.connectome.nodes.to_df().sort_values('index')
    labels = nodes['type'].astype(str).to_numpy()
    roles = nodes['role'].astype(str).to_numpy()
    u = nodes['u'].to_numpy(dtype=float)
    v = nodes['v'].to_numpy(dtype=float)
    xx, yy = u + 0.5 * v, (np.sqrt(3) / 2.0) * v
    groups = {name: np.flatnonzero(labels == name) for name in POPULATIONS}
    groups['Photoreceptors'] = np.flatnonzero(roles == 'input')
    for name, indices in groups.items():
        if not len(indices):
            raise RuntimeError(f'No neurons found for {name}')

    # Same network initial state and time window for every comparison.
    with torch.no_grad():
        blank = torch.zeros((1, 20, network.n_nodes), device=device)
        baseline = network(blank, dt=args.dt).squeeze(0).cpu().numpy()
    if not np.isfinite(baseline).all():
        raise RuntimeError('Baseline activity is non-finite')

    rows = []
    for k, (index, segment_window_count, seg_id) in enumerate(selected, 1):
        sample = dataset[index]
        frames = sample['frames']
        external = mapper(frames)
        verified = mapper.verify(external)
        with torch.no_grad():
            driven = network(external.to(device), dt=args.dt).squeeze(0).cpu().numpy()
        if not np.isfinite(driven).all():
            raise RuntimeError(f'Non-finite activity in sample index {index}')
        delta = driven - baseline
        population_curve = {}
        for name, ids in groups.items():
            d = delta[:, ids]
            curve = np.abs(d).mean(axis=1)
            population_curve[name] = curve
            rows.append({
                'sample_index': index, 'folder': sample['folder'],
                'segment_id': sample['segment_id'],
                'start_frame': sample['start_frame'],
                'start_timestep': sample['start_timestep'],
                'source_segment_windows': segment_window_count,
                'population': name, 'neurons': len(ids),
                'mean_abs_change': float(np.abs(d).mean()),
                'peak_abs_change': float(np.abs(d).max()),
                'mean_signed_change': float(d.mean()),
                'peak_frame_mean_abs': int(np.argmax(curve)),
            })
        print(f'[{k:02d}/{len(selected)}] folder={sample["folder"]} '
              f'segment={sample["segment_id"]} idx={index} '
              f'L1={population_curve["L1"].mean():.5g} '
              f'T4a={population_curve["T4a"].mean():.5g} '
              f'PR={population_curve["Photoreceptors"].mean():.5g}')

        if k != 1:
            continue
        # First selected sequence: full temporal visualization and peak maps.
        frame_array = frames[:, 0].numpy()
        maps = [(name, ids, delta[:, ids], population_curve[name]) for name, ids in groups.items()]
        limits = [max(float(np.max(np.abs(d))), 1e-7) for _, _, d, _ in maps]
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        ax_img = axes[0, 0]
        im = ax_img.imshow(frame_array[0], cmap='gray', vmin=0, vmax=1, origin='upper')
        ax_img.set_title('Original retinal-wave frame')
        ax_img.axis('off')
        scatters = []
        for ax, ((name, ids, d, _), lim) in zip(list(axes.flat)[1:7], zip(maps, limits)):
            scatter = ax.scatter(xx[ids], yy[ids], c=d[0], cmap='coolwarm',
                                 vmin=-lim, vmax=lim, s=7, linewidth=0)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name)
            fig.colorbar(scatter, ax=ax, shrink=.68)
            scatters.append(scatter)
        ax_info = axes[1, 3]
        ax_info.axis('off')
        info = ax_info.text(.03, .8, '', fontsize=11, va='top')
        fig.suptitle('Untrained FlyVis: wave-driven minus zero-drive activity\n'
                     'Per-population color limits fixed across time (not shared across populations)')

        def update(t):
            im.set_data(frame_array[t])
            for scatter, (_, _, d, _) in zip(scatters, maps):
                scatter.set_array(d[t])
            info.set_text(f'Frame {t + 1}/20\nTime {t * args.dt:.3f} s\n'
                          f'Folder {sample["folder"]}\nSegment {sample["segment_id"]}\n'
                          f'L1 gain {args.gain:g}')
            return [im, *scatters, info]

        anim = FuncAnimation(fig, update, frames=20, interval=250, blit=False)
        gif_path = out / 'neural_response_animation.gif'
        anim.save(gif_path, writer=PillowWriter(fps=4), dpi=105)
        plt.close(fig)
        print('Saved', gif_path)

        fig, axarr = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        axarr[0, 0].imshow(frame_array[0], cmap='gray', vmin=0, vmax=1)
        axarr[0, 0].set_title('First input wave frame')
        axarr[0, 0].axis('off')
        for ax, (name, ids, d, curve) in zip(list(axarr.flat)[1:7], maps):
            peak_t = int(np.argmax(curve))
            lim = limits[list(groups).index(name)]
            sc = ax.scatter(xx[ids], yy[ids], c=d[peak_t], cmap='coolwarm',
                            vmin=-lim, vmax=lim, s=7, linewidth=0)
            ax.set_title(f'{name} | peak mean|Δ| at frame {peak_t + 1}')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, shrink=.7)
        axarr[1, 3].axis('off')
        fig.suptitle('Population peak-response maps (different peak times)')
        fig.savefig(out / 'peak_response_maps.png', dpi=140)
        plt.close(fig)

    dataset.close()
    with (out / 'multi_window_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stats = {}
    for name in groups:
        vals = np.array([r['mean_abs_change'] for r in rows if r['population'] == name])
        stats[name] = {
            'mean': float(vals.mean()), 'median': float(np.median(vals)),
            'min': float(vals.min()), 'max': float(vals.max()),
            'nonzero_windows_above_1e-6': int((vals > 1e-6).sum()),
            'n_windows': len(vals),
        }
    report = {'dt_provisional': args.dt, 'gain': args.gain, 'windows': args.num_windows,
              'split': 'train', 'different_folders': True, 'stats': stats}
    with (out / 'multi_window_summary.json').open('w') as f:
        json.dump(report, f, indent=2)
    print('\nAcross-window mean absolute differences:')
    for name, st in stats.items():
        print(f'  {name:16s} mean={st["mean"]:.6g} '
              f'[{st["min"]:.6g}, {st["max"]:.6g}]')
    print('\nDone. Model checkpoint was read-only. Outputs:', out)


if __name__ == '__main__':
    main()
