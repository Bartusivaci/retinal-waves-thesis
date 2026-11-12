"""Evaluate FlyVis thesis checkpoints using the validated official MovingEdge/NetworkView pipeline.

From repository root:
    python -m scripts.evaluate_training_progress
    python -m scripts.evaluate_training_progress --intermediate 1000 2500 5000 7500
    python -m scripts.evaluate_training_progress --all-intermediate

The pretrained network directory is ONLY a configuration template. All state dicts
come from the explicitly listed thesis .pt files, and are verified tensor-by-tensor.
No training is performed. Each checkpoint gets a separate response cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Memory

from flyvis import results_dir
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import (
    direction_selectivity_index,
    preferred_direction,
    correlation_to_known_tuning_curves,
)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(results_dir) / 'flow/0000/000'
CELL_TYPES = ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d')


def load_thesis_weights(network, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'network' not in checkpoint:
        raise KeyError(f'Missing network state dict in {checkpoint_path}')
    network.load_state_dict(checkpoint['network'], strict=True)
    network.eval()
    return network


def checkpoint_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()[:12]


def make_view(label, path, output_dir):
    if not path.is_file():
        raise FileNotFoundError(f'{label}: missing checkpoint: {path}')
    if not TEMPLATE.is_dir():
        raise FileNotFoundError(f'Missing official FlyVis template {TEMPLATE}; run flyvis download-pretrained')

    digest = checkpoint_hash(path)
    cache_dir = output_dir / 'cache' / f'{label}_{digest}' / '__cache__'
    cache_dir.mkdir(parents=True, exist_ok=True)
    view = NetworkView(
        TEMPLATE,
        recover_fn=load_thesis_weights,
        best_checkpoint_fn=lambda _directory, **_kwargs: path,
    )
    view.memory = Memory(location=cache_dir, backend='xarray_dataset_h5', verbose=0)
    view.name = f'thesis/{label}/{digest}'
    view._clear_cache()

    # Verify actual tensors, rather than trusting checkpoint selection or cache.
    loaded = view.init_network().state_dict()
    expected = torch.load(path, map_location='cpu', weights_only=True)['network']
    if set(loaded) != set(expected):
        raise AssertionError(f'{label}: loaded state-dict keys differ')
    for key, tensor in expected.items():
        if not torch.equal(loaded[key].detach().cpu(), tensor.detach().cpu()):
            raise AssertionError(f'{label}: weight mismatch: {key}')
    print(f'  Weight verification passed (SHA256 {digest})', flush=True)
    return view, digest


def model_paths(args):
    gap_dir = ROOT / 'results/multi_window/gap5_10000'
    entries = [
        ('baseline', ROOT / 'checkpoints/flyvis_initial.pt', 0, 'baseline'),
        ('trained_1000_mixed', ROOT / 'results/multi_window/retinal_waves_1000/final.pt', 1000, 'mixed_1_3_5'),
    ]
    if args.all_intermediate:
        checkpoints = sorted(gap_dir.glob('checkpoint_step_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f'No intermediate checkpoints in {gap_dir}')
        for path in checkpoints:
            step = int(path.stem.split('_')[-1])
            if step == 10000:
                continue  # the final.pt is evaluated separately
            entries.append((f'gap5_step_{step:06d}', path, step, 'gap5'))
    else:
        for step in sorted(set(args.intermediate)):
            if step <= 0 or step >= 10000:
                raise ValueError('Intermediate steps must be between 1 and 9999')
            entries.append((
                f'gap5_step_{step:06d}',
                gap_dir / f'checkpoint_step_{step:06d}.pt',
                step,
                'gap5',
            ))
    entries += [
        ('gap5_best', gap_dir / 'best.pt', None, 'gap5'),
        ('gap5_final', gap_dir / 'final.pt', 10000, 'gap5'),
    ]
    if args.skip_best:
        entries = [x for x in entries if x[0] != 'gap5_best']
    return entries


def read_best_step():
    path = ROOT / 'results/multi_window/gap5_10000/best_metrics.json'
    if path.exists():
        return int(json.loads(path.read_text())['step'])
    return None


def finite_float(x):
    val = float(x)
    return val if math.isfinite(val) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediate', type=int, nargs='*', default=[],
                        help='Gap-5 intermediate training steps, e.g. 1000 2500 5000 7500')
    parser.add_argument('--all-intermediate', action='store_true',
                        help='Evaluate all saved gap-5 checkpoint_step_*.pt files')
    parser.add_argument('--skip-best', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/motion_debug/training_progress')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = model_paths(args)
    for label, path, step, family in entries:
        if not path.is_file():
            raise FileNotFoundError(f'{label}: {path}\n'
                                    'Check that training finished and checkpoints were saved.')
    print('Models to evaluate:', len(entries), flush=True)
    dataset = MovingEdge(
        offsets=[-10, 11], intensities=[0, 1], speeds=[19], height=80,
        post_pad_mode='continue', t_pre=1.0, t_post=1.0,
        dt=1 / 200, angles=list(np.arange(0, 360, 30)),
    )
    print('Official MovingEdge stimuli:', len(dataset), flush=True)

    best_step = read_best_step()
    results, metadata, rows = {}, {}, []
    for ix, (label, path, step, family) in enumerate(entries, 1):
        if label == 'gap5_best':
            step = best_step
        print(f'\n[{ix}/{len(entries)}] {label}: {path}', flush=True)
        view, digest = make_view(label, path, output_dir)
        responses = view.moving_edge_responses(dataset, batch_size=args.batch_size)
        dsis = direction_selectivity_index(responses)
        preferred = preferred_direction(responses)
        correlations = correlation_to_known_tuning_curves(responses)
        metrics = {}
        for cell_type in CELL_TYPES:
            polarity = 'ON' if cell_type.startswith('T4') else 'OFF'
            selection = {'cell_type': cell_type, 'intensity': 1 if polarity == 'ON' else 0}
            # Match the previously successful pretrained and thesis evaluation scripts.
            dsi = finite_float(np.nanmean(dsis.custom.where(**selection).values))
            pref = finite_float(np.rad2deg(np.nanmean(preferred.custom.where(**selection).values)))
            corr = finite_float(np.nanmean(correlations.custom.where(**selection).values))
            metrics[cell_type] = {
                'polarity': polarity, 'dsi': dsi, 'preferred_deg': pref,
                'tuning_correlation': corr,
            }
            rows.append({
                'model': label, 'training_family': family, 'step': step,
                'cell_type': cell_type, 'polarity': polarity, 'dsi': dsi,
                'preferred_deg': pref, 'tuning_correlation': corr,
            })
            print(f'  {cell_type}: DSI={dsi:.8f} preferred={pref:.1f}° corr={corr:.4f}', flush=True)
        results[label] = metrics
        metadata[label] = {
            'checkpoint': str(path), 'sha256_prefix': digest,
            'training_step': step, 'training_family': family,
        }
        # Save incrementally so work survives interruption.
        (output_dir / 'comparison.json').write_text(json.dumps({
            'protocol': 'FlyVis official MovingEdge: 12 angles x ON/OFF, dt=0.005 s',
            'note': 'Pretrained directory supplies configuration only; trained weights are never loaded.',
            'models': metadata, 'results': results,
        }, indent=2))

    baseline = results['baseline']
    one_k = results['trained_1000_mixed']
    final = results['gap5_final']
    best = results.get('gap5_best')
    print('\n=== DSI comparison ===', flush=True)
    print(f'{"Cell":<6} {"baseline":>12} {"1K mixed":>12} {"10K final":>12} {"10K best":>12}')
    for cell_type in CELL_TYPES:
        def d(model):
            return model[cell_type]['dsi'] if model is not None else None
        fmt = lambda val: f'{val:.7f}' if val is not None else '--'
        print(f'{cell_type:<6} {fmt(d(baseline)):>12} {fmt(d(one_k)):>12} '
              f'{fmt(d(final)):>12} {fmt(d(best)):>12}')

    comparison = {
        'protocol': 'FlyVis official MovingEdge: 12 angles x ON/OFF, dt=0.005 s',
        'note': 'Same protocol as validated pretrained positive control; no pretrained weights loaded.',
        'models': metadata,
        'results': results,
        'dsi_differences': {
            label: {
                ct: metrics[ct]['dsi'] - baseline[ct]['dsi']
                for ct in CELL_TYPES
            }
            for label, metrics in results.items() if label != 'baseline'
        },
    }
    (output_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    with (output_dir / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CELL_TYPES))
    plotting = [('baseline', 'Baseline'), ('trained_1000_mixed', '1K mixed'),
                ('gap5_final', '10K gap5 final')]
    if best is not None:
        plotting.append(('gap5_best', f'10K gap5 best (step {best_step})'))
    width = 0.8 / len(plotting)
    for i, (label, display) in enumerate(plotting):
        y = [results[label][ct]['dsi'] for ct in CELL_TYPES]
        ax.bar(x + (i - (len(plotting)-1)/2)*width, y, width=width, label=display)
    ax.set_xticks(x, CELL_TYPES)
    ax.set_ylabel('Official FlyVis direction-selectivity index (DSI)')
    ax.set_title('Direction selectivity: baseline vs retinal-wave training')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'dsi_comparison.png', dpi=160)
    plt.close(fig)

    gap5_series = [(metadata[label]['training_step'], metrics)
                   for label, metrics in results.items()
                   if metadata[label]['training_family'] == 'gap5'
                   and metadata[label]['training_step'] is not None
                   and label != 'gap5_best']
    gap5_series = sorted(gap5_series, key=lambda pair: pair[0])
    if len(gap5_series) >= 2:
        fig, ax = plt.subplots(figsize=(11, 6))
        for ct in CELL_TYPES:
            steps = [0] + [s for s, _ in gap5_series]
            dsis = [baseline[ct]['dsi']] + [result[ct]['dsi'] for _, result in gap5_series]
            ax.plot(steps, dsis, marker='o', label=ct)
        ax.axvline(1000, linestyle=':', color='gray', alpha=0.5)
        ax.set(xlabel='Gap-5 training step', ylabel='Official FlyVis DSI',
               title='Direction selectivity along gap-5 training trajectory')
        ax.grid(alpha=0.25)
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(output_dir / 'dsi_vs_steps.png', dpi=160)
        plt.close(fig)

    print('\nSaved:', output_dir / 'comparison.json')
    print('Saved:', output_dir / 'metrics.csv')
    print('Saved:', output_dir / 'dsi_comparison.png')
    if (output_dir / 'dsi_vs_steps.png').exists():
        print('Saved:', output_dir / 'dsi_vs_steps.png')
    print('\nIMPORTANT: The 1K model used mixed training gaps and is not part of the gap-5 trajectory.')
    print('Very low DSIs require cautious interpretation, even when relative changes are large.')


if __name__ == '__main__':
    main()
