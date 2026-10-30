"""Compare baseline vs retinal-wave-trained FlyVis on first photoreceptor input.
Run: python -m scripts.evaluate_first_vision --run-name first_vision_1000
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

from flyvis.network import Network
from flyvis.datasets.rendering import BoxEye
from flyvis.utils.hex_utils import hex_to_pixel

ROOT = Path(__file__).resolve().parent.parent
POPULATIONS = ('L1', 'Mi1', 'Tm3', 'T4a', 'T5a')


def stimuli(size=256, lead=10, steps=20):
    """Toy stimuli, NOT natural-scene edge-detection benchmarks."""
    neutral = np.full((size, size), 127, dtype=np.uint8)
    tree = Image.new('L', (size, size), 127)
    d = ImageDraw.Draw(tree)
    d.rectangle((122, 139, 134, 230), fill=30)
    d.polygon([(128, 24), (72, 145), (184, 145)], fill=220)
    d.polygon([(128, 55), (85, 168), (171, 168)], fill=220)
    tree_frame = np.asarray(tree, dtype=np.uint8)
    stationary = np.stack([neutral] * lead + [tree_frame] * steps)

    edges = [neutral] * lead
    for j in range(steps):
        a = np.full((size, size), 35, dtype=np.uint8)
        x = int(size * (0.2 + 0.6 * j / max(1, steps - 1)))
        a[:, x:] = 220
        edges.append(a)
    moving = np.stack(edges)

    reverse = [neutral] * lead
    for j in range(steps):
        a = np.full((size, size), 35, dtype=np.uint8)
        x = int(size * (0.8 - 0.6 * j / max(1, steps - 1)))
        a[:, x:] = 220
        reverse.append(a)
    return {'tree_onset': stationary, 'edge_right': moving, 'edge_left': np.stack(reverse)}


def load_network(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or 'network' not in ckpt:
        raise ValueError(f'Checkpoint {checkpoint_path} lacks a network state_dict')
    network = Network()
    network.load_state_dict(ckpt['network'], strict=True)
    network = network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)
    return network


def run_movie(network, movie, box_eye, device, dt):
    # BoxEye expects (batch,time,H,W), outputs (batch,time,1,721).
    image = torch.from_numpy(movie.astype(np.float32) / 255)[None].to(device)
    rendered = box_eye(image).detach().cpu()
    if tuple(rendered.shape[:3]) != (1, movie.shape[0], 1) or rendered.shape[-1] != 721:
        raise ValueError(f'Unexpected BoxEye result {tuple(rendered.shape)}')

    # Use FlyVis' own photoreceptor input mapper, to guarantee correct ordering
    # across the R-cell types. No internal L1 injection in evaluation.
    network.stimulus.zero(n_samples=1, n_frames=movie.shape[0])
    network.stimulus.add_input(rendered)
    whole_network_input = network.stimulus().to(device)
    index = np.asarray(network.stimulus.input_index, dtype=np.int64)
    other = torch.ones(whole_network_input.shape[-1], dtype=torch.bool, device=device)
    other[torch.as_tensor(index, dtype=torch.long, device=device)] = False
    assert not torch.any(whole_network_input[..., other] != 0), 'Non-photoreceptors received external stimulus!'
    with torch.no_grad():
        response = network(whole_network_input, dt=dt).cpu()[0]
    if not torch.isfinite(response).all():
        raise RuntimeError('Non-finite neural response')
    return response, rendered[0, :, 0].numpy()


def coords(network, indices):
    nodes = network.connectome.nodes
    u = np.asarray(nodes.u)[indices].astype(float)
    v = np.asarray(nodes.v)[indices].astype(float)
    return hex_to_pixel(u, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, default=ROOT/'checkpoints/flyvis_initial.pt')
    ap.add_argument('--trained', type=Path, default=ROOT/'results/multi_window/retinal_waves_1000/final.pt')
    ap.add_argument('--run-name', default='first_vision_1000')
    ap.add_argument('--dt', type=float, default=0.02)
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = ROOT/'results/first_vision'/args.run_name
    out.mkdir(parents=True, exist_ok=True)
    models = {'baseline': load_network(args.baseline, device), 'trained': load_network(args.trained, device)}
    eye = BoxEye(extent=15, kernel_size=13)
    if hasattr(eye, 'to'):
        eye = eye.to(device)
    records = {}
    movies = stimuli()
    for name, movie in movies.items():
        print('Evaluating', name, 'on', device, flush=True)
        np.save(out/f'{name}_stimulus.npy', movie)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, fi in zip(axs, [9, 15, len(movie)-1]):
            ax.imshow(movie[fi], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'Frame {fi}'); ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'{name}_input.png', dpi=140); plt.close(fig)
        outputs = {}
        for label, net in models.items():
            outputs[label], rendering = run_movie(net, movie, eye, device, args.dt)
        records[name] = {}
        for pop in POPULATIONS:
            idx = np.asarray(models['baseline'].stimulus.layer_index[pop], dtype=np.int64)
            x, y = coords(models['baseline'], idx)
            b = outputs['baseline'][:, idx].numpy()
            t = outputs['trained'][:, idx].numpy()
            # Change after stimulus onset relative to each model's own pre-onset state
            b_change = b[10:] - b[9]
            t_change = t[10:] - t[9]
            difference = t_change - b_change
            records[name][pop] = {
                'mean_abs_difference_in_evoked_response': float(np.abs(difference).mean()),
                'rms_evoked_baseline': float(np.sqrt(np.mean(b_change**2))),
                'rms_evoked_trained': float(np.sqrt(np.mean(t_change**2))),
                'evoked_response_correlation': float(np.corrcoef(b_change.ravel(), t_change.ravel())[0,1]) if np.std(b_change) > 1e-10 and np.std(t_change)>1e-10 else None,
            }
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(np.arange(20)*args.dt, np.mean(np.abs(b_change), axis=1), label='Baseline')
            ax.plot(np.arange(20)*args.dt, np.mean(np.abs(t_change), axis=1), label='Trained')
            ax.set_xlabel('Time after stimulus onset (s)')
            ax.set_ylabel('Mean |activity - pre-onset activity|')
            ax.set_title(f'{name}: {pop} evoked responses')
            ax.legend(); fig.tight_layout()
            fig.savefig(out/f'{name}_{pop}_timecourse.png', dpi=145); plt.close(fig)
            # Consistent color range across baseline/trained maps for THIS population.
            vals = [b_change[-1], t_change[-1], difference[-1]]
            vmax = max(float(np.max(np.abs(z))) for z in vals) or 1e-8
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
            for ax, z, title in zip(axs, vals, ('Baseline evoked', 'Trained evoked', 'Trained − baseline')):
                im = ax.scatter(x, y, c=z, cmap='coolwarm', vmin=-vmax, vmax=vmax, s=8)
                ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
            fig.colorbar(im, ax=axs, shrink=.7)
            fig.suptitle(f'{name} / {pop} (last frame)')
            fig.savefig(out/f'{name}_{pop}_maps.png', dpi=150, bbox_inches='tight'); plt.close(fig)
        print('  L1 evoked difference:', records[name]['L1']['mean_abs_difference_in_evoked_response'])
    result = {'baseline': str(args.baseline), 'trained': str(args.trained), 'device': str(device), 'dt': args.dt,
              'pre_onset_frames': 10, 'stimulus_frames': 20, 'results': records,
              'interpretation': 'These are response differences, not proof of improved edge or motion selectivity.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    print('Done. Results:', out)

if __name__ == '__main__':
    main()
