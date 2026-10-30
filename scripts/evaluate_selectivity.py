"""First-vision edge and direction selectivity evaluation.
Run from repo root: python -m scripts.evaluate_selectivity
Requires the existing scripts/evaluate_first_vision.py from our previous step.
No parameter updates or retinal-wave stimulation occur here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.evaluate_first_vision import ROOT, load_network, run_movie, coords
from flyvis.datasets.rendering import BoxEye

POPS = ('L1', 'Mi1', 'Tm3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d')
DIRECTIONS = ('right', 'left', 'down', 'up')


def tree_frame(size=256):
    im = Image.new('L', (size, size), 127)
    d = ImageDraw.Draw(im)
    d.rectangle((122, 139, 134, 230), fill=30)
    d.polygon([(128, 24), (72, 145), (184, 145)], fill=220)
    d.polygon([(128, 55), (85, 168), (171, 168)], fill=220)
    return np.asarray(im, dtype=np.uint8)


def static_movie(frame, lead=10, steps=20):
    gray = np.full_like(frame, 127)
    return np.stack([gray] * lead + [frame] * steps)


def moving_movie(direction, size=256, lead=10, steps=20):
    neutral = np.full((size, size), 127, dtype=np.uint8)
    frames = [neutral] * lead
    for j in range(steps):
        f = np.full((size, size), 35, dtype=np.uint8)
        a = 0.2 + 0.6 * j / max(steps - 1, 1)
        if direction == 'right':
            x = int(size * a); f[:, x:] = 220
        elif direction == 'left':
            x = int(size * (1-a)); f[:, :x] = 220
        elif direction == 'down':
            y = int(size * a); f[y:, :] = 220
        elif direction == 'up':
            y = int(size * (1-a)); f[:y, :] = 220
        frames.append(f)
    return np.stack(frames)


def geometry_and_edge_map(rendered, x, y):
    """Approximate local contrast on the *same* hex-column positions as responses.
    Use 6 nearest hex neighbours; mask edge/interior based on input contrast.
    """
    xy = np.column_stack((x, y))
    dist2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    near = np.argsort(dist2, axis=1)[:, 1:7]
    local_contrast = np.abs(rendered[:, None] - rendered[near]).mean(axis=1)
    # A nearly uniform bright or dark area inside the figure should be excluded from edges.
    edge = local_contrast > 0.075
    interior = local_contrast < 0.015
    return local_contrast, edge, interior


def response_amplitude(output, idx, lead=10):
    # Use average absolute change from pre-onset state over all stimulus frames.
    a = output[:, idx].numpy()
    return np.mean(np.abs(a[lead:] - a[lead-1]), axis=0)


def moving_amplitude(output, idx, lead=10):
    a = output[:, idx].numpy()
    return np.mean(np.abs(a[lead:] - a[lead-1]), axis=0)


def safe_float(a):
    return float(a) if np.isfinite(a) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'checkpoints/flyvis_initial.pt')
    parser.add_argument('--trained', type=Path, default=ROOT/'results/multi_window/retinal_waves_1000/final.pt')
    parser.add_argument('--run-name', default='selectivity_1000')
    parser.add_argument('--dt', type=float, default=.02)
    args = parser.parse_args()
    torch.manual_seed(42)
    outdir = ROOT/'results'/'first_vision'/args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nets = {'baseline': load_network(args.baseline, device), 'trained': load_network(args.trained, device)}
    eye = BoxEye(extent=15, kernel_size=13)
    if hasattr(eye, 'to'):
        eye = eye.to(device)
    bnet = nets['baseline']
    pops = [p for p in POPS if p in bnet.stimulus.layer_index]
    if not pops:
        raise RuntimeError('Expected cell types absent from installed FlyVis.')
    result = {'baseline': str(args.baseline), 'trained': str(args.trained), 'dt_provisional': args.dt,
              'edge': {}, 'direction': {},
              'notes': ['Input retinotopy alone can preserve a tree silhouette; outline extraction requires boundary selectivity.',
                        'Direction metric uses nonnegative evoked response amplitude, not signed polarity.',
                        'One training seed and synthetic stimuli; descriptive results, not evidence of improved vision.']}
    # Run static images, including a neutral control and a simple filled tree.
    tree = tree_frame()
    movies = {'tree': static_movie(tree), 'uniform': static_movie(np.full((256, 256), 220, dtype=np.uint8))}
    outputs = {}
    for condition, movie in movies.items():
        outputs[condition] = {}
        for label, net in nets.items():
            print('Static:', condition, label, flush=True)
            outputs[condition][label] = run_movie(net, movie, eye, device, args.dt)
    # A photoreceptor-input projection from FlyVis' actual BoxEye renderer.
    rendered = outputs['tree']['baseline'][1][-1]
    # Each cell population uses the same 721 columns but may differ in ordering.
    # Match cell coordinate to input hex coordinate using the photoreceptor mapping indices.
    receptor_indices = np.asarray(bnet.stimulus.layer_index['R1'], dtype=np.int64)
    rx, ry = coords(bnet, receptor_indices)
    # Determine the BoxEye ordering by the known R1 input mapping: FlyVis aligns each 721 input with column order.
    # If the ordering differs in your installed release, this diagnostic requires explicit index alignment.
    for pop in pops:
        idx = np.asarray(bnet.stimulus.layer_index[pop], dtype=np.int64)
        x, y = coords(bnet, idx)
        # Map each population column to R1 spatial coordinates, independent of node order.
        d2 = (x[:, None] - rx[None, :])**2 + (y[:, None] - ry[None, :])**2
        order = np.argmin(d2, axis=1)
        if float(np.max(np.min(d2, axis=1))) > 1e-8:
            print('WARNING: unmatched spatial coordinates for', pop)
        local, edge, interior = geometry_and_edge_map(rendered[order], x, y)
        record = {'edge_locations': int(edge.sum()), 'interior_locations': int(interior.sum())}
        for label in nets:
            a = response_amplitude(outputs['tree'][label][0], idx)
            edge_response = float(a[edge].mean()) if edge.any() else float('nan')
            interior_response = float(a[interior].mean()) if interior.any() else float('nan')
            ratio = edge_response / (interior_response + 1e-8)
            record[label] = {'mean_edge_amplitude': safe_float(edge_response),
                             'mean_interior_amplitude': safe_float(interior_response),
                             'edge_to_interior_ratio': safe_float(ratio)}
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            scatter = ax.scatter(x, y, c=a, cmap='viridis', s=13)
            ax.scatter(x[edge], y[edge], s=12, facecolors='none', edgecolors='magenta', linewidths=.45)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f'{pop}: {label} | response amplitude\nMagenta = input-defined boundary locations')
            fig.colorbar(scatter, ax=ax, shrink=.7)
            fig.savefig(outdir/f'edge_{pop}_{label}.png', dpi=140, bbox_inches='tight'); plt.close(fig)
        result['edge'][pop] = record
    # Motion: 4 directions, identical input and simulation protocol for both models.
    motion_outputs = {label: {} for label in nets}
    for direction in DIRECTIONS:
        movie = moving_movie(direction)
        for label, net in nets.items():
            print('Motion:', direction, label, flush=True)
            motion_outputs[label][direction], _ = run_movie(net, movie, eye, device, args.dt)
    for pop in pops:
        idx = np.asarray(bnet.stimulus.layer_index[pop], dtype=np.int64)
        record = {}
        for label in nets:
            responses = np.stack([moving_amplitude(motion_outputs[label][d], idx) for d in DIRECTIONS], axis=0)
            # Across 4 cardinal directions: (max-min)/(max+min) per neuron.
            top = responses.max(axis=0)
            bottom = responses.min(axis=0)
            selectivity = (top-bottom)/(top+bottom+1e-8)
            preferred = responses.argmax(axis=0)
            record[label] = {
                'mean_direction_selectivity': float(selectivity.mean()),
                'median_direction_selectivity': float(np.median(selectivity)),
                'mean_response_by_direction': {d: float(responses[k].mean()) for k, d in enumerate(DIRECTIONS)},
                'preferred_direction_counts': {d: int((preferred==k).sum()) for k, d in enumerate(DIRECTIONS)}
            }
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(selectivity, bins=24, range=(0,1))
            ax.set(xlabel='Four-direction response selectivity', ylabel='Neuron count',
                   title=f'{pop}: {label}'); fig.tight_layout()
            fig.savefig(outdir/f'direction_selectivity_{pop}_{label}.png', dpi=140); plt.close(fig)
        result['direction'][pop] = record
    (outdir/'summary.json').write_text(json.dumps(result, indent=2))
    print('Saved', outdir/'summary.json')
    print('Evaluation complete: no parameters modified.')

if __name__ == '__main__':
    main()
